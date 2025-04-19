"""
A model worker executes the model.
"""
import os
import argparse
import asyncio
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
import whisper
import numpy as np
from functools import partial

from llama_omni2.constants import WORKER_HEART_BEAT_INTERVAL
from llama_omni2.utils import build_logger, server_error_msg, pretty_print_semaphore
from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from llama_omni2.model import *
from transformers import TextIteratorStreamer, AutoTokenizer, AutoConfig
from threading import Thread


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def build_unit_tokenizer(vocab_size):
    import os
    from transformers import BertTokenizer
    with open("unit_vocab.txt", "w") as f:
        for i in range(vocab_size + 1):
            f.write(str(i) + "\n")
    tokenizer = BertTokenizer(vocab_file="unit_vocab.txt")
    os.remove("unit_vocab.txt")
    return tokenizer


def load_pretrained_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    config.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Omni2Speech2SQwen2ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    model.cuda()
    return tokenizer, model


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register, model_path, model_name):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_name
        self.tokenizer, self.model = load_pretrained_model(model_path)
        self.unit_tokenizer = build_unit_tokenizer(self.model.config.unit_vocab_size)

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,), daemon=True)
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
    
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

    def get_input_params(self, history):
        messages = []
        speech_list = []
        for i, turn in enumerate(history):
            if i % 3 == 0:
                messages.append({
                    "role": "user",
                    "content": DEFAULT_SPEECH_TOKEN,
                })
                speech_list.append(self.load_speech(turn["content"]["path"]))
            elif i % 3 == 1:
                messages.append({
                    "role": "assistant",
                    "content": turn["content"]
                })
            else:
                continue
        input_ids = self.process_messages(messages).unsqueeze(0)
        speech_tensors = torch.nn.utils.rnn.pad_sequence(
            speech_list,
            batch_first=True,
            padding_value=0
        )
        speech_lengths = torch.LongTensor([len(speech) for speech in speech_list])
        return input_ids, speech_tensors, speech_lengths

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model

        history = params["history"]
        input_ids, speech_tensors, speech_lengths = self.get_input_params(history)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensors = speech_tensors.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
        speech_lengths = speech_lengths.to(device='cuda', non_blocking=True)

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        do_sample = True if temperature > 0.001 else False

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False, timeout=15)
        streamer_unit = TextIteratorStreamer(self.unit_tokenizer, skip_prompt=False, skip_special_tokens=False, timeout=15)

        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            speech=speech_tensors,
            speech_lengths=speech_lengths,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            streamer_unit=streamer_unit,
            use_cache=True,
        ))
        thread.start()

        generated_text = ""
        stop_str = "<|im_end|>"
        for new_text in streamer:
            generated_text += new_text
            generated_unit = " ".join(map(str, streamer_unit.token_cache))
            finalize = generated_text.endswith(stop_str)
            if finalize:
                streamer_unit.end()
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "unit": generated_unit, "finalize": finalize, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")