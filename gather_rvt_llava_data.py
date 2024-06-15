import json
import os
from llava.utils  import RLBENCH_TASKS

data_path  = "/home/daiyp/Open-LLaVA-NeXT/augmented_data_heuristic"
save_dir = "playground/rvt_llava_data"
os.makedirs(save_dir, exist_ok=True)

# TODO: combine catastrophy data and more failure data

all_data = []
for task in RLBENCH_TASKS:
    task_data = []
    for d in ["train"]:
        for i in range(100):
            file_path = os.path.join(data_path, d, task, str(i), "llava_v0.json")
            if not os.path.exists(file_path):
                continue
            task_data.extend(json.load(open(file_path)))
    all_data.extend(task_data)

    with open(os.path.join(save_dir, task + ".json"), "w") as f:
        json.dump(task_data, f, indent=2)

with open(os.path.join(save_dir, "all_tasks.json"), "w") as f:
    json.dump(all_data, f)


print("total data size: ", len(all_data))
            
# ## find the longest one to test the GPU memory limit
# max_length = 0
# max_id = 0
# for idx, item in enumerate(all_data):
#     length = len(item["conversations"][0]["task_goal"].split()) + \
#     len(item["conversations"][0]["previous_instruction"].split()) +\
#     len(item["conversations"][0]["robot_delta_state"].split()) if "robot_delta_state" in item["conversations"][0] else 0 + \
#     max(len(item["conversations"][1]["gpt_instruction"].split()), len(item["conversations"][1]["heuristic_instruction"].split())) 
#     if length > max_length:
#         max_length = length
#         max_idx = idx

# print(max_length, max_idx)
# with open(os.path.join(save_dir, "max_length_fake_data.json"), "w") as f:
#     json.dump([all_data[max_idx] for _  in range(10)], f, indent=2)
    