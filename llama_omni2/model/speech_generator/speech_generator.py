import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config
from llama_omni2.constants import IGNORE_INDEX


def lengths_to_attention_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return ~mask


class LLMSpeechGenerator(nn.Module):
    def __init__(self, config):
        super(LLMSpeechGenerator, self).__init__()
        self.model = Qwen2ForCausalLM(Qwen2Config(**config.speech_generator))
        self.tokenizer = AutoTokenizer.from_pretrained(config.tts_tokenizer)
        self.input_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, self.model.config.hidden_size)
        )
        self.stream_params = config.stream_params
        self.gate = nn.Sequential(
            nn.Linear(2 * self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Sigmoid()
        )

    def fusion(self, rep, emb):
        gate = self.gate(torch.cat([rep, emb], dim=-1))
        return rep * gate + emb * (1 - gate)

    def generate_units(self, tts_inputs, new_hidden_states, new_tokens, is_finished=False):
        # only for batch size = 1
        new_hidden_states = self.input_proj(new_hidden_states)
        new_token_embeddings = self.model.get_input_embeddings()(new_tokens)
        new_hidden_states = self.fusion(new_hidden_states, new_token_embeddings)
        if tts_inputs is not None:
            tts_inputs = torch.cat([tts_inputs, new_hidden_states], dim=0)
        else:
            tts_inputs = new_hidden_states
        if is_finished:
            device = tts_inputs.device
            sep_id = torch.LongTensor([self.tokenizer.convert_tokens_to_ids("<sep>")]).to(device)
            sep_emb = self.model.get_input_embeddings()(sep_id)
            tts_inputs = torch.cat([tts_inputs, sep_emb], dim=0)

        _, M = eval(self.stream_params)
        max_new_tokens = M if not is_finished else 1024
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=tts_inputs.unsqueeze(0),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0]
        generated_units = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_tokens_embeds = self.model.get_input_embeddings()(generated_tokens)
        tts_inputs = torch.cat([tts_inputs, generated_tokens_embeds], dim=0)
        return tts_inputs, generated_units