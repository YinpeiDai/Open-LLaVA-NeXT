import json
import uuid

# Read the input file
with open('playground/accessibility_data/sample_train.jsonl', 'r') as infile:
    lines = infile.readlines()

# Prepare the LLaVA format output
llava_format = []

for line in lines:
    data = json.loads(line.strip())
    text = data["text"]
    label = data["label"]
    
    # Create the "human" part
    human_part = {"from": "human", "value": text}    
    # Create the "gpt" part with the label
    gpt_part = {"from": "gpt", "value": f"Label: {label}"}
    
    llava_format.append(
        {"id": uuid.uuid4().hex,
        "conversations": [human_part, gpt_part]}
    )

# Save the LLaVA formatted data into a JSON file
with open('playground/accessibility_data/sample_train_llava_format.json', 'w') as outfile:
    json.dump(llava_format, outfile, indent=1)