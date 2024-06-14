from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from transformers import AutoTokenizer, T5EncoderModel, BartModel, RobertaModel


from PIL import Image
import requests
import copy
import torch

# pretrained = "/data/daiyp/foundation_models/open-llava-next-llama3-8b"
pretrained = "/data/daiyp/foundation_models/llama3-llava-next-8b"
# model_name = "llava_llama3"
device = "cuda"
device_map = "auto"

###### LORA ##### 
lora_path = "/home/daiyp/Open-LLaVA-NeXT/checkpoints/llava-v1.6-8b_llama3-original-debug"
model_base_path = pretrained

tokenizer, model, image_processor, max_length = load_pretrained_model(
    model_path=lora_path,
    model_base=pretrained,
    model_name="llava_llama3_lora",
    device_map=device_map)


# for k, v in model.get_model().named_parameters():
#     print(k, v.requires_grad, v.numel())
# input()



# print(tokenizer)
model.eval()
model.tie_weights()

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]



conv_template = "llava_llama_3_rvt" # or "llava_llama_3_rvt_<task_name>"
question = DEFAULT_IMAGE_TOKEN + "\nWhat should the robot do in the next?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
print(prompt_question)

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# print(input_ids)
image_sizes = [image.size]

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    use_cache=True,
    max_new_tokens=128,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)