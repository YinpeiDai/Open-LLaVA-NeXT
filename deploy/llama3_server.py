import argparse
import base64
from io import BytesIO
import json

from PIL import Image

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import torch
import uvicorn

from llava.utils import server_error_msg
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from transformers import TextIteratorStreamer
from threading import Thread

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
from functools import partial

from llava.constants import WORKER_HEART_BEAT_INTERVAL
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread

from llava.train.my_train_commongrid import CONV_COMMONGRID_LLAMA3_TEMPLATE, preprocess_llama3
from llava import conversation as conversation_lib

conversation_lib.default_conversation = CONV_COMMONGRID_LLAMA3_TEMPLATE



global_counter = 0

model_semaphore = None


class ModelWorker:
    def __init__(
            self,
            model_name,
            model_path, 
            model_base,
            device,
        ):

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, device=device)
        
        self.device = device
        self.model.eval()
        self.model.tie_weights()

        
    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        prompt = params["messages"]
        ori_prompt = prompt

        input_ids = tokenizer.apply_chat_template(prompt, return_tensors='pt')
        print(tokenizer.decode(input_ids[0])) # print the prompt
        
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        do_sample = False


        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=False,
            max_new_tokens=10,
            streamer=streamer,
            use_cache=True,
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

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


def release_model_semaphore():
    model_semaphore.release()

@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks)

@app.get("/test")
async def test():
    return {"message": "Hello World"}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--model-path", type=str, default="/data/daiyp/foundation_models/llama3-llava-next-8b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="llava_llama3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    args = parser.parse_args()

    worker = ModelWorker(
        model_name=args.model_name,
        model_path=args.model_path,
        model_base=args.model_base,
        device=args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    # python deploy/llava_server.py --model-path <lora_model_save_path> --model-base <llama3-llava-next-8b-path> --model-name llava_llama3_lora
    # then run llava_api in rvt folder


    # CUDA_VISIBLE_DEVICES=1 python deploy/llava_server.py --model-path /home/daiyp/Open-LLaVA-NeXT/checkpoints/commongrid_llama3_lora_ep2_bs64 --model-base /nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF --model-name llama3_lora
