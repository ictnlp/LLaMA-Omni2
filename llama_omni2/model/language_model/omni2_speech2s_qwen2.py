#    Copyright 2023 Haotian Liu
#    Copyright 2024 Qingkai Fang
#
#    This project is modified based on LLaVA by Haotian Liu, Qingkai Fang adds further supports for speech-to-text/speech tasks.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llama_omni2.model.language_model.omni2_speech_qwen2 import Omni2SpeechQwen2ForCausalLM
from llama_omni2.model.speech_generator.builder import build_speech_generator
from llama_omni2.model.speech_generator.generation import GenerationWithTTS


class Omni2Speech2SConfig(Qwen2Config):
    model_type = "omni2_speech2s_qwen2"


class Omni2Speech2SQwen2ForCausalLM(Omni2SpeechQwen2ForCausalLM, GenerationWithTTS):
    config_class = Omni2Speech2SConfig

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "speech_generator", None):
            self.speech_generator = build_speech_generator(config)

        self.post_init()
    
    def get_speech_generator(self):
        speech_generator = getattr(self, "speech_generator", None)
        return speech_generator
    
    def initialize_speech_generator(self, model_args):
        self.config.speech_generator = getattr(model_args, 'speech_generator', None)
        self.config.unit_vocab_size = getattr(model_args, 'unit_vocab_size', 4096)
        self.config.stream_params = getattr(model_args, 'stream_params', "(1,1)")

        if getattr(self, "speech_generator", None) is None:
            self.speech_generator = build_speech_generator(self.config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        tgt_units: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )

        lm_output = super(Omni2SpeechQwen2ForCausalLM, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )
        loss = lm_output.loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_output.logits,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        outputs = GenerationWithTTS.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs
        )

        return outputs.sequences, outputs.unit_sequences

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs

AutoConfig.register("omni2_speech2s_qwen2", Omni2Speech2SConfig)
AutoModelForCausalLM.register(Omni2Speech2SConfig, Omni2Speech2SQwen2ForCausalLM)
