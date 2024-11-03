
# transformers              4.37.2
import argparse
import json
import os
import random
import re
import time
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

import logging

class ModelWorker:
    def __init__(
        self,
        model_base, 
        device,
        logname="eval.log"
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(model_base)
        self.model = AutoModelForCausalLM.from_pretrained(model_base, device_map="balanced", torch_dtype=torch.float16, low_cpu_mem_usage=True)

        # # put on device
        # self.model.to(device)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
        self.device = device
        self.model.eval()
        self.model.tie_weights()
        logging.basicConfig(filename=f"/home/daiyp/Open-LLaVA-NeXT/playground/llmchoice/{logname}", level=logging.INFO)

        
    @torch.inference_mode()
    def generate(self, params):
        tokenizer, model = self.tokenizer, self.model
        prompt = params["messages"]

        # print("prompt", prompt)
        
        # trucate the left context
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors='pt', padding=True, max_length=1024)            
        
        # print(input_ids)
        # print(tokenizer.decode(input_ids[0])) # print the prompt
        # print(input_ids.shape)
        
        input_ids = input_ids.to(self.device)
        
        temperature = float(params.get("temperature", 0.7))
        top_p = float(params.get("top_p", 1.0))
        do_sample = False
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                do_sample=do_sample,
                max_new_tokens=64,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id
            )
        new_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return self.postprocess(new_text)
    

    def postprocess(self, text):
        text = text.replace("assistant", "")
        text = text.strip()
        try:
            dic = json.loads(text)
            logging.info(f"Postprocessing: {dic}")
            return dic
        except Exception as e:
            logging.error(f"Error in postprocessing: {e}")
            logging.error(f"{text}")
            return {}
    



       
if __name__ == "__main__":
    # model_name = "Llama-3.1-8B-Instruct"
    model_name = "Llama-3.1-70B-Instruct"
    model_id = "Llama3.1-70B"
    model_worker = ModelWorker(
        model_base = f"/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/{model_name}",
        device="cuda",
        logname=f"eval_{model_name}.log"
    )

        # model_base="/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF",

    filepath = "/home/daiyp/Open-LLaVA-NeXT/playground/llmchoice/survey-5-transportation-choice.jsonl"
    with open(filepath, 'r') as infile:
        lines = infile.readlines()

    output=[]

    for line in tqdm(lines):
        data = json.loads(line.strip())
        messages = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": data["input"]
            }
        ]
        label = model_worker.generate({"messages": messages})
        
        data[model_id] = label

        output.append(json.dumps(data))

    with open(f"/home/daiyp/Open-LLaVA-NeXT/playground/llmchoice/survey-5-transportation-choice-notrain-{model_name}.jsonl", 'w') as outfile:
        for line in output:
            outfile.write(line + "\n")
