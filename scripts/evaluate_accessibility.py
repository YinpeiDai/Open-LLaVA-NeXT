
import argparse
import json
import os
import random
import re
import time
import torch
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.train.my_train_accessibility import system_prompt


LABELS =  ["positive", "negative", "neutral", "unrelated"]

import logging

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
        # if os.path.exists(f"{model_path}/eval.log"):
        #     os.remove(f"{model_path}/eval.log")
        # logging.basicConfig(filename=f"{model_path}/eval.log", level=logging.INFO)

        
    @torch.inference_mode()
    def generate(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        prompt = params["messages"]

        # print("prompt", prompt)
        
        # trucate the left context
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors='pt', padding="longest", max_length=1280)            
        
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

        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                do_sample=do_sample,
                max_new_tokens=16,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                eos_token_id=terminators,
            )
        new_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"raw text: {new_text}")
        # logging.info(f"raw text: {new_text}")
        return self.post_process(new_text)
    
    def post_process(self, response):
        response = response.lower()
        m = re.search(r"^label: (\w+)\.?", response)
        if m:
            s = m.group(1)
            if s in ["positive", "negative", "neutral", "unrelated"]:
                return s
        # label this review as "positive"
        m = re.search(r"label this review as \"(\w+)\"", response)
        if m:
            s = m.group(1)
            if s in ["positive", "negative", "neutral", "unrelated"]:
                return s
        logging.info(f"Failed to extract label from response: {response}")
        print(f"Failed to extract label from response: {response}")
        return random.choice(LABELS[:2])


       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF")
    parser.add_argument("--model-name", type=str, default="llama3_lora")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dirname", type=str, default="playground/accessibility_data")
    parser.add_argument("--test-files", type=str, nargs="+", default=[])
    parser.add_argument("--key", type=str, default="text")
    args = parser.parse_args()


    model_worker = ModelWorker(
        model_name=args.model_name,
        model_path=args.model_path,
        model_base=args.model_base,
        device=args.device,
    )
    print(args.test_files)
    for test_file in args.test_files:
        print(f"Processing {test_file}")
        outputs = []
        with open(f"{args.dirname}/{test_file}.jsonl", 'r') as infile:
            lines = infile.readlines()

        llava_format = []

        for line in tqdm(lines):
            data = json.loads(line.strip())
            text = data[args.key]
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            label = model_worker.generate({"messages": messages})
            result = {"text": text, "label": label}
            print(f"result: {result}")
            outputs.append(result)
        
        model_str = args.model_path.split("/")[-1]
        output_file = f"{args.dirname}/{test_file}-prediction.jsonl" 
        with open(output_file, 'w') as outfile:
            for output in outputs:
                json.dump(output, outfile)
                outfile.write("\n")
        print(f"Saved to {output_file}")

