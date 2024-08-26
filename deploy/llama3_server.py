import argparse
import json

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import torch
import uvicorn

from llava.utils import server_error_msg
from llava.model.builder import load_pretrained_model
from transformers import TextIteratorStreamer
from threading import Thread

import argparse
import asyncio
import json

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import torch
import uvicorn

from llava.utils import server_error_msg
from transformers import TextIteratorStreamer
from threading import Thread

import time


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
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
        self.device = device
        self.model.eval()
        self.model.tie_weights()

        
    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        prompt = params["messages"]
        
        # trucate the left context
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors='pt', padding="longest", max_length=5120)
        
        while input_ids.shape[1] > 5100:
            prompt = prompt[:1] + prompt[3:]
            input_ids = tokenizer.apply_chat_template(prompt, return_tensors='pt', padding="longest", max_length=5120)
            
        
        # print(input_ids)
        # print(tokenizer.decode(input_ids[0])) # print the prompt
        # print(input_ids.shape)
        
        input_ids = input_ids.to(self.device)
        
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        do_sample = False
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # # no streaming
        # tstart = time.time()
        # with torch.no_grad():
        #     outputs = model.generate(
        #         inputs=input_ids,
        #         do_sample=do_sample,
        #         max_new_tokens=256,
        #         temperature=temperature,
        #         top_p=top_p,
        #         use_cache=True,
        #         eos_token_id=terminators,
        #     )
        # t = time.time() - tstart
        # toal_toks = outputs[0].shape.numel()
        # input_toks = input_ids[0].shape.numel()
        # print(f"Time to generate: {t:.4f}s", "Tokens:", toal_toks, "Generated Tokens:", (toal_toks-input_toks))
        # only return the generated text, ignore the input prompt
        # new_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        # print("Generated text:", new_text)
        # yield json.dumps({"text": new_text, "error_code": 0}).encode() + b"\0"

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=False,
            max_new_tokens=256,
            streamer=streamer,
            use_cache=True,
            eos_token_id=terminators
            
        ))
        thread.start()

        for new_text in streamer:
            yield json.dumps({"text": new_text, "error_code": 0}).encode() + b"\0"

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
    model_semaphore.release() # type: ignore

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
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21004)
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

    # python deploy/llama3_server.py --model-path <lora_model_save_path> --model-base <llama3-llava-next-8b-path> --model-name llava_llama3_lora
    # then run llava_api in rvt folder


    # CUDA_VISIBLE_DEVICES=0 python deploy/llama3_server.py --model-path /home/daiyp/Open-LLaVA-NeXT/checkpoints/commongrid_llama3_lora_ep2_bs64 --model-base /nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF --model-name llama3_lora
