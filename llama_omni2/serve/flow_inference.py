import os
import sys
import uuid
import threading
import torchaudio
import torch
import logging
import torch.nn.functional as F
import numpy as np
from hyperpyyaml import load_hyperpyyaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR, "third_party/Matcha-TTS"))

from cosyvoice.cli.frontend import CosyVoiceFrontEnd


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                         fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


class SpeechDecoder:
    def __init__(self,
        model_dir,
        device="cuda",
        hop_len=None,
        load_onnx=False,
    ):
        self.device = device

        # Config
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        # Frontend
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            '{}/campplus.onnx'.format(model_dir),
            '{}/speech_tokenizer_v2.onnx'.format(model_dir),
            '{}/spk2info.pt'.format(model_dir),
            False,
            configs['allowed_special']
        )
        self.sample_rate = configs['sample_rate']

        # Load models
        self.flow = configs['flow']
        self.flow.load_state_dict(torch.load('{}/flow.pt'.format(model_dir), map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        self.flow.decoder.fp16 = False
        self.hift = configs['hift']
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load('{}/hift.pt'.format(model_dir), map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        if load_onnx:
            self.load_onnx('{}/flow.decoder.estimator.fp32.onnx'.format(model_dir))

        self.token_hop_len = hop_len if hop_len is not None else 2 * self.flow.input_frame_rate
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # dict used to store session related variable
        self.lock = threading.Lock()
        self.hift_cache_dict = {}
    
    def load_onnx(self, flow_decoder_estimator_model):
        print("Loading ONNX model")
        import onnxruntime
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        del self.flow.decoder.estimator
        self.flow.decoder.estimator = onnxruntime.InferenceSession(flow_decoder_estimator_model, sess_options=option, providers=providers)

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                         token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_token=prompt_token.to(self.device),
                                         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_feat=prompt_feat.to(self.device),
                                         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                         embedding=embedding.to(self.device),
                                         finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech
    
    def init_prompt(self, prompt_speech_16k):
        prompt_speech_feat = torch.zeros(1, 0, 80)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        embedding = self.frontend._extract_spk_embedding(prompt_speech_16k)
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.hift_cache_dict[this_uuid] = None

        session = {
            'uuid': this_uuid,
            'prompt_feat': prompt_speech_feat,
            'prompt_token': prompt_speech_token,
            'embedding': embedding,
            'token_offset': 0,
            'generated_tokens': None
        }
        return session

    def process_unit_chunk(self, new_chunk, session, finalize=False):
        if session["generated_tokens"] is None:
            session["generated_tokens"] = new_chunk
        else:
            session["generated_tokens"] = torch.cat([session["generated_tokens"], new_chunk], dim=-1)
        
        token_offset = session['token_offset']
        this_uuid = session['uuid']
        prompt_feat = session['prompt_feat']
        prompt_token = session['prompt_token']
        embedding = session['embedding']
        generated_tokens = session["generated_tokens"]

        tts_speech = self.token2wav(
            token=generated_tokens.unsqueeze(0),
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            uuid=this_uuid,
            token_offset=token_offset,
            finalize=finalize,
        )
        if not finalize:
            session["token_offset"] = len(generated_tokens) - self.flow.pre_lookahead_len
        else:
            session["token_offset"] = len(generated_tokens)
        return tts_speech.cpu(), session