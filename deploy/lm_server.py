import os
import time
from fastapi import FastAPI
import torch
import clip
from transformers import AutoTokenizer, T5EncoderModel,  BartModel, RobertaModel, LlamaModel
from pydantic import BaseModel


MAIN_DIR = "/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/"
MODEL_DICT = {
    "t5-11b": os.path.join(MAIN_DIR, "t5-11b"),
}

device_map = {
    "t5-11b": "cuda",
    "clip": "cuda:0",
}

# Initialize models
models = {}
tokenizers = {}

tstart = time.time()
clip_model, _ = clip.load("RN50", device=device_map["clip"])
clip_model = clip_model.to(device_map["clip"])
clip_model.eval()
print(f"Loaded language model CLIP in {time.time() - tstart:.2f}s")

# Load models and tokenizers
for model_name, model_path in MODEL_DICT.items():
    tstart = time.time()
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
    if "t5-3b" in model_name:
        models[model_name] = T5EncoderModel.from_pretrained(model_path, device_map=device_map[model_name])
    elif "t5-11b" in model_name:
        models[model_name] = T5EncoderModel.from_pretrained(model_path, device_map=device_map[model_name])
    elif "bart" in model_name:
        models[model_name] = BartModel.from_pretrained(model_path, device_map=device_map[model_name])
    elif "roberta" in model_name:
        models[model_name] = RobertaModel.from_pretrained(model_path, device_map=device_map[model_name])
    elif "llama3" in model_name:
        tokenizers[model_name].pad_token_id = 128010
        models[model_name] = LlamaModel.from_pretrained(model_path, device_map=device_map[model_name], 
                                                        torch_dtype=torch.float16)
    print(f"Loaded language model { model_name} in {time.time() - tstart:.2f}s")

for model in models.values():
    model.eval()


class TextInput(BaseModel):
    text: str
    model: str # ["t5-3b", "t5-11b", "bart-large", "roberta-large", "llama3", "clip"]


app = FastAPI()

# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    return x

@app.post("/encode/")
async def encode_text(input: TextInput):
    model_name = input.model
    input_text = input.text

    if "clip" in model_name:
        token_tensor = clip.tokenize(input_text, context_length=300)
        token_len = token_tensor.numpy()[0].tolist().index(49407) + 1
        token_tensor = token_tensor.to(device_map[model_name])
        token_tensor = token_tensor[:, :77]
        if token_len > 77: token_len = 77
        with torch.no_grad():
            embeddings = _clip_encode_text(clip_model, token_tensor)
    else:
        if model_name not in models:
            return {"error": f"Model {model_name} not loaded."}
        # Get the tokenizer and model
        tokenizer = tokenizers[model_name]
        model = models[model_name]
        # Tokenize the input text
        input_ids = tokenizer(input_text, return_tensors='pt')
        input_ids = input_ids.input_ids.to(device_map[model_name])
        token_len = input_ids.shape[1]
        with torch.no_grad():
            embeddings = model(input_ids=input_ids).last_hidden_state
    
    return_embeddings = embeddings.float().detach().cpu().numpy()
    print(model_name, return_embeddings.dtype, return_embeddings.shape, token_len)
    return {"embeddings": return_embeddings.tolist(), "token_len": token_len}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# CUDA_VISIBLE_DEVICES=0 python service.py