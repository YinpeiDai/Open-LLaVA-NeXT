import base64
from io import BytesIO
import json
import time
import numpy as np
import requests
from PIL import Image
import requests
from llava.conversation import Conversation, SeparatorStyle


DEFAULT_IMAGE_TOKEN = "<image>"

HEADERS = {"User-Agent": "LLaVA Client"}



class LlavaAPI:
    def __init__(self, addr):
        self.addr = addr

    def wrap_image(self, image: Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8') 
        return image_base64

    def wrap_prompt(self, messages):
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        user_msg = messages[1]["content"]
        system_prompt = messages[0]["content"]

        conv = Conversation(
            system="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"+system_prompt,
            roles=("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|start_header_id|>assistant<|end_header_id|>\n\n"), # type: ignore
            version="llava_llama_3_commongrid",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|eot_id|>",
        ).copy()
        
        if DEFAULT_IMAGE_TOKEN not in user_msg:
            question = DEFAULT_IMAGE_TOKEN + f"\n{user_msg}"
        else:
            question = user_msg
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    

    def get_response(self, messages, image):
        if isinstance(image, np.ndarray):
            if image.shape[0] == 3: # (3, 512, 512) -> (512, 512, 3)
                image = image.transpose(1, 2, 0)
            image = Image.fromarray(image)
        pload = {
            "prompt": self.wrap_prompt(messages),
            "image": self.wrap_image(image)
        }       
        print(json.dumps(pload["prompt"]))
        input("Press Enter to continue...") 
        generated_text = ""
        while True:
            try:
                # Stream output
                response = requests.post(self.addr + "/worker_generate_stream",
                    headers=HEADERS, json=pload, stream=True, timeout=10)
                last_len = len(pload["prompt"])
                for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode())
                        if data["error_code"] == 0:
                            output = data["text"][last_len:]
                            generated_text += output
                            last_len = len(data["text"])
                        else:
                            output = data["text"] + f" (error_code: {data['error_code']})"
                break
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
            time.sleep(1)
        return generated_text
    

    def get_response_stream(self, user_msg, image):
        if isinstance(image, np.ndarray):
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            image = Image.fromarray(image)
        pload = {
            "prompt": self.wrap_prompt(user_msg),
            "image": self.wrap_image(image)
        }

        while True:
            try:
                response = requests.post(self.addr + "/worker_generate_stream",
                    headers=HEADERS, json=pload, stream=True, timeout=30)
                last_len = len(pload["prompt"])
                for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode())
                        if data["error_code"] == 0:
                            output = data["text"][last_len:]
                            yield output
                            last_len = len(data["text"])
                        else:
                            print(f" (error_code: {data['error_code']})")
                break
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
            time.sleep(0.1)



if __name__ == "__main__":
    from minigrid.utils.data_preprocess.finetuned_prompt import (
    DESCRIPTIVE_SYSTEM_PROMPT_NO_BELIEF_VISION, 
    DESCRIPTIVE_SYSTEM_PROMPT_ZEROTH_AND_FIRST_BELIEF_VISION, 
    DESCRIPTIVE_SYSTEM_PROMPT_ZEROTH_BELIEF_VISION
)

    # api = LlavaAPI("http://141.212.106.177:21002")
    api = LlavaAPI("http://141.212.110.118:21002")
    width, height = 6, 6
    windown_size = 12 # Fixed
    info = {
        "width": width,
        "height": height,
        "window_size": windown_size
    }
    setting = "zeroth"
    image = Image.open("/nfs/turbo/coe-chaijy/roihn/commongrid/dataset/SFT/img/pick_two_balls_dis_com/episode_0000/agent0_memory_000.png")

    if setting == "none":
        system_prompt = DESCRIPTIVE_SYSTEM_PROMPT_NO_BELIEF_VISION.replace("WIDTH", str(info["width"])).replace("HEIGHT", str(info["height"])).replace("WINSIZE", str(info["window_size"]))
    elif setting == "zeroth":
        system_prompt = DESCRIPTIVE_SYSTEM_PROMPT_ZEROTH_BELIEF_VISION.replace("WIDTH", str(info["width"])).replace("HEIGHT", str(info["height"])).replace("WINSIZE", str(info["window_size"]))
    elif setting == "first":
        system_prompt = DESCRIPTIVE_SYSTEM_PROMPT_ZEROTH_AND_FIRST_BELIEF_VISION.replace("WIDTH", str(info["width"])).replace("HEIGHT", str(info["height"])).replace("WINSIZE", str(info["window_size"]))
    else:
        raise ValueError(f"Unknown setting: {setting}")


    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "Step 0\nYou are agent0 at [3, 1]\nTask: Each agent picks up one ball. You pick up the red ball, and your partner picks up the blue ball.\nPossible actions: left, right, forward, exchange, share [x, y], request action <action>, request [x, y]\nObjects to track:agent0,agent1,ball0,ball1\nYour partner's action in the last step: unknown\n\nSummary of what you have observed in the history:\nIn front there is your partner agent1 facing towards up direction on your left, 2 steps away and 1 steps to the left.\nYou are facing down.\n\nCurrent observation:\nan agent1 facing towards up way is in front of you on your left, 2 steps away and 1 steps to the left.\nYou are facing towards down with empty hand."
        },
    ] # "{\"Your belief of the world\": {\"agent0\": \"Agent0 at (3, 1), facing down\", \"agent1\": \"Agent1 at (4, 3), facing up\", \"ball0\": \"Unseen\", \"ball1\": \"Unseen\"}, \"action\": \"forward\"}"

    response = api.get_response(messages, image)
    print(response)
    