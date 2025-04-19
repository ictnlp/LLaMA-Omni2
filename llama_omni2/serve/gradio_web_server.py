import argparse
import datetime
import json
import os
import sys
import copy
import torch

import gradio as gr
import requests
import tempfile
import torchaudio

from llama_omni2.constants import LOGDIR, DEFAULT_SPEECH_TOKEN
from llama_omni2.utils import build_logger, server_error_msg
from llama_omni2.serve.flow_inference import SpeechDecoder

from os.path import dirname
ROOT_DIR = dirname(dirname(dirname(__file__)))
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR, "third_party/Matcha-TTS"))
from cosyvoice.utils.file_utils import load_wav

os.environ['GRADIO_TEMP_DIR'] = './tmp'

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaMA-Omni2 Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

prompt_speech = "llama_omni2/inference/prompt_en.wav"
prompt_speech_16k = load_wav(prompt_speech, 16000)


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")

    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    logger.info(f"Models: {models}")
    return models


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return [], dropdown_update


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    return [], None, None, []


def inference_fn(history_state, audio_input, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    model_name = model_selector

    # 查询 worker 地址
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # 更新对话历史
    history_state.append({"role": "user", "content": {"path": audio_input}})
    pload = {
        "model": model_name,
        "history": history_state,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
    }
    
    yield (history_state, None, copy.deepcopy(history_state))

    try:
        response = requests.post(worker_addr + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=20)
        device = "cuda"
        
        session = vocoder.init_prompt(prompt_speech_16k)
        tts_speechs = []
        num_generated_units = 0
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output_unit = list(map(int, data["unit"].strip().split()))
                    if len(output_unit) > num_generated_units:
                        new_units = output_unit[num_generated_units:]
                        num_generated_units = len(output_unit)
                        new_units_tensor = torch.LongTensor(new_units).to(device)
                        tts_speech, session = vocoder.process_unit_chunk(new_units_tensor, session, finalize=data["finalize"])
                        if tts_speech is not None:
                            tts_speechs.append(tts_speech)
                    if history_state[-1]["role"] == "assistant":
                        history_state[-1]["content"] = data["text"]
                    else:
                        history_state.append({"role": "assistant", "content": data["text"]})
                    if tts_speechs:
                        wav_full = torch.cat(tts_speechs, dim=-1).cpu().numpy()
                        yield (history_state, (24000, wav_full.T), copy.deepcopy(history_state))
                    else:
                        yield (history_state, None, copy.deepcopy(history_state))
                else:
                    error_output = data["text"] + f" (error_code: {data['error_code']})"
                    history_state.append({"role": "assistant", "content": error_output})
                    yield (history_state, None, copy.deepcopy(history_state))
                    return
    except requests.exceptions.RequestException as e:
        history_state.append({"role": "assistant", "content": "Server error."})
        yield (history_state, None, copy.deepcopy(history_state))
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f, torch.tensor(wav_full), 24000, format="wav")
        history_state.append({"role": "assistant", "content": {"path": f.name, "type": "audio/wav"}})
        yield(history_state, None, copy.deepcopy(history_state))


title_markdown = ("""
# 🦙🎧 LLaMA-Omni 2
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

def build_demo(cur_dir=None, concurrency_count=10):
    models = get_model_list()

    with gr.Blocks(title="LLaMA-Omni2 Demo", theme=gr.themes.Default(), css=block_css) as demo:
        gr.Markdown(title_markdown)

        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False
            )

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            type="messages",
            scale=1,
        )

        with gr.Row():
            audio_input_box = gr.Audio(label="Input audio", type='filepath', show_download_button=True, visible=True)
            audio_output_box = gr.Audio(label="Output audio", show_download_button=False)

        with gr.Accordion("Parameters", open=False) as parameter_row:
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature")
            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P")
            max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens")

        with gr.Row():
            submit_btn = gr.Button(value="Send", variant="primary")
            clear_btn = gr.Button(value="Clear")

        history_state = gr.State([])

        submit_btn.click(
            inference_fn,
            inputs=[
                history_state,
                audio_input_box,
                model_selector,
                temperature,
                top_p,
                max_output_tokens,    
            ],
            outputs=[history_state, audio_output_box, chatbot]
        )

        clear_btn.click(
            clear_history,
            None,
            [history_state, audio_input_box, audio_output_box, chatbot],
            queue=False
        )

        demo.load(
            load_demo_refresh_model_list,
            None,
            [history_state, model_selector],
            queue=False
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--vocoder-dir", type=str, default=None)
    parser.add_argument("--hop-len", type=int, default=10)
    parser.add_argument("--load-onnx", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    vocoder = SpeechDecoder(
        model_dir=args.vocoder_dir,
        hop_len=args.hop_len,
        load_onnx=args.load_onnx,
    )

    logger.info(args)
    demo = build_demo(concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )