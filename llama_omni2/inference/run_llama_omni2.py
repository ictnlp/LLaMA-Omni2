import argparse
import torch
import os
import json
import whisper

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from llama_omni2.model import *
from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from torch.utils.data import Dataset, DataLoader


class MultiturnSpeechDataset(Dataset):
    def __init__(self, questions, tokenizer, model_config):
        self.questions = questions
        self.tokenizer = tokenizer
        self.model_config = model_config
    
    def load_speech(self, path):
        speech = whisper.load_audio(path)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        return speech

    def process_messages(self, messages):
        assert len(messages) % 2 == 1, "Number of history messages must be odd"
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
        input_ids[input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        return input_ids

    def __getitem__(self, index):
        item = self.questions[index]
        messages = []
        speech_list = []
        for i, turn in enumerate(item["conversation"]):
            if i % 2 == 0:
                messages.append({
                    "role": "user",
                    "content": DEFAULT_SPEECH_TOKEN,
                })
                speech_list.append(self.load_speech(turn["speech"]))
            else:
                messages.append({
                    "role": "assistant",
                    "content": turn["text"],
                })
        input_ids = self.process_messages(messages)
        return {
            "input_ids": input_ids,
            "speech": speech_list
        }

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids = [instance["input_ids"] for instance in batch]
    all_speech = [speech for instance in batch for speech in instance["speech"]]
    input_ids = torch.stack(input_ids, dim=0)
    speech_tensors = torch.nn.utils.rnn.pad_sequence(
        all_speech,
        batch_first=True,
        padding_value=0
    )
    speech_lengths = torch.LongTensor([len(speech) for speech in all_speech])
    return input_ids, speech_tensors, speech_lengths


def create_data_loader(questions, tokenizer, model_config, batch_size=1, num_workers=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = MultiturnSpeechDataset(questions, tokenizer, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def load_pretrained_model(model_path, s2s=False):
    model_cls = Omni2Speech2SQwen2ForCausalLM if s2s else Omni2SpeechQwen2ForCausalLM
    config = AutoConfig.from_pretrained(model_path)
    config.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = model_cls.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    model.cuda()
    return tokenizer, model


def eval_model(args):
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model = load_pretrained_model(model_path, s2s=args.s2s)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    answers_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    data_loader = create_data_loader(questions, tokenizer, model.config)

    for (input_ids, speech_tensor, speech_lengths), item in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = item["id"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
        speech_lengths = speech_lengths.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                speech=speech_tensor,
                speech_lengths=speech_lengths,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            if args.s2s:
                output_ids, output_units = outputs
            else:
                output_ids = outputs

        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"T-{idx}\t{output_text}")
        if args.s2s:
            print(f"S-{idx}\t{output_units}")

        if args.s2s:
            ans_file.write(json.dumps({"question_id": idx, "prediction": output_text, "prediction_units": output_units}, ensure_ascii=False) + "\n")
        else:
            ans_file.write(json.dumps({"question_id": idx, "prediction": output_text}, ensure_ascii=False) + "\n")
    
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--question_file", type=str)
    parser.add_argument("--answer_file", type=str)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--s2s", action="store_true", default=False)
    args = parser.parse_args()

    eval_model(args)
