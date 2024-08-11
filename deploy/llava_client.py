import base64
from io import BytesIO
import json
import time
import numpy as np
import requests
from llava.conversation import conv_llava_llama_3_rvt        
from PIL import Image
import requests
from llava.utils import TEMP_0_label, TEMP_0_nolabel, TEMP_label, TEMP_nolabel



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

    def wrap_prompt(self, user_msg="What is shown in the legend?"):
        if DEFAULT_IMAGE_TOKEN not in user_msg:
            question = DEFAULT_IMAGE_TOKEN + f"\n{user_msg}"
        else:
            question = user_msg
        conv = conv_llava_llama_3_rvt.copy()
        # conv = conv_llava_llama_3.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    

    def get_response(self, user_msg, image):
        if isinstance(image, np.ndarray):
            if image.shape[0] == 3: # (3, 512, 512) -> (512, 512, 3)
                image = image.transpose(1, 2, 0)
            image = Image.fromarray(image)
        pload = {
            "prompt": self.wrap_prompt(user_msg),
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


    def prepare_user_msg_for_vlm(self, task_goal, previous_instruction, robot_delta_state, predict_failure_label=True):
        if previous_instruction is None: # first turn
            if predict_failure_label:
                return TEMP_0_label.format(task_goal=task_goal)
            else:
                return TEMP_0_nolabel.format(task_goal=task_goal)
        else:
            if predict_failure_label:
                return TEMP_label.format(task_goal=task_goal, previous_instruction=previous_instruction, robot_delta_state=robot_delta_state)
            else:
                return TEMP_nolabel.format(task_goal=task_goal, previous_instruction=previous_instruction, robot_delta_state=robot_delta_state)


if __name__ == "__main__":
    # api = LlavaAPI("http://127.0.0.1:21002")
    api = LlavaAPI("http://141.212.110.118:21002")
    image = Image.open("/home/daiyp/Open-LLaVA-NeXT/augmented_data_heuristic/train/close_jar/2/front_rgb/120_expert.png")
    usrmsg = api.prepare_user_msg_for_vlm(
        task_goal="close the purple jar",
        previous_instruction="Transport the lid over the purple jar by moving backward and to the left.",
        robot_delta_state="Then the robot moved backward, moved left, didn't rotate the gripper, and kept the gripper closed by planning a motion path that avoids any collision."
    )

    response = api.get_response(usrmsg, image)
    print(response)
    
    