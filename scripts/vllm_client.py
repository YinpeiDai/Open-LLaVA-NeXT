import json
import logging
from openai import OpenAI
from tqdm import tqdm
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


model_id = "Llama-3.1-70B-Instruct"

logging.basicConfig(filename=f"/home/daiyp/Open-LLaVA-NeXT/playground/llmchoice/{model_id}.log", level=logging.INFO)

filepath = "/home/daiyp/Open-LLaVA-NeXT/playground/llmchoice/survey-5-transportation-choice.jsonl"
with open(filepath, 'r') as infile:
    lines = infile.readlines()

output=[]

for line in tqdm(lines):
    data = json.loads(line.strip())

    chat_response = client.chat.completions.create(
        model="/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": data["input"]},
        ],
        max_tokens=100,
        temperature=1.0,
        top_p=1.0
    )
    label = chat_response.choices[0].message.content
    label = label.strip()

    try:
        dic = json.loads(label)
        logging.info(f"Postprocessing: {dic}")
        data[model_id] = dic
    except Exception as e:
        logging.error(f"Error in postprocessing: {e}")
        logging.error(f"{label}")
        data[model_id] = {}
    
    output.append(json.dumps(data))

with open(f"/home/daiyp/Open-LLaVA-NeXT/playground/llmchoice/survey-5-transportation-choice-notrain-{model_id}.jsonl", 'w') as outfile:
    for line in output:
        outfile.write(line + "\n")







