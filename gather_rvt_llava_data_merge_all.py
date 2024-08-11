from collections import defaultdict
import json
import os
from llava.utils  import RLBENCH_TASKS

save_dir = "playground/rvt_llava_data"
os.makedirs(save_dir, exist_ok=True)

# TODO: combine catastrophy data and more failure data

all_data = defaultdict(list)
for task in RLBENCH_TASKS:
    if task == "slide_block_to_color_target":
        continue
    if task in ["put_item_in_drawer", "reach_and_drag", "turn_tap", "open_drawer"]:
        llava_file = "llava_v1.json"
    else:
        llava_file = "llava_v0.json"
    task_data = []
    for d in ["train", "val"]:
        for i in range(100):
            file_path = os.path.join("augmented_data_heuristic", d, task, str(i), llava_file)
            if not os.path.exists(file_path):
                continue
            json_data =  json.load(open(file_path))
            add_end_task = False
            for sample in json_data:
                if sample["conversations"][1]["label"] == "task success": # only get end-of-task success
                    if add_end_task: continue
                    task_data.append(sample)
                    add_end_task = True
                # elif sample["conversations"][1]["label"] == "subgoal success": # double the expert step data
                #     task_data.append(sample)
                #     task_data.append(sample)
                else:
                    task_data.append(sample)
                
    print(task, len(task_data))
    all_data[task].extend(task_data)


print("---")
for task in "insert_onto_square_peg,place_shape_in_shape_sorter,put_item_in_drawer,stack_cups".split(","):
    task_data = []
    for d in ["train", "val"]:
        for i in range(100):
            file_path = os.path.join("augmented_data_heuristic/augmented_data_v2", d, task, str(i), "llava_v0.json")
            if not os.path.exists(file_path):
                continue
            json_data = json.load(open(file_path))
            for sample in json_data:
                if sample["conversations"][1]["label"] == "subgoal failure":
                    task_data.append(sample)
    print(task, len(task_data))
    if task == "put_item_in_drawer":
        task_data = task_data[:len(task_data)//2]
    elif task == "stack_cups":
        task_data = task_data[:len(task_data)//3]
    all_data[task].extend(task_data)

print("---")
task = "slide_block_to_color_target"
for dirname in ["0622_v1", "0622_v2"]:
    task_data = []
    for d in ["train", "val"]:
        for i in range(100):
            file_path = os.path.join("augmented_data_heuristic", dirname, d, task, str(i), "llava_v0.json")
            if not os.path.exists(file_path):
                continue
            json_data = json.load(open(file_path))
            add_end_task = False
            for sample in json_data:
                if sample["conversations"][1]["label"] == "task success": # only get end-of-task success
                    if add_end_task: continue
                    task_data.append(sample)
                    add_end_task = True
                else:
                    task_data.append(sample)
    print(task, len(task_data))
    all_data["slide_block_to_color_target"].extend(task_data)


print("---")
all_data_list = []
for k in all_data:
    print(k, len(all_data[k]))
    all_data_list.extend(all_data[k])
    with open(os.path.join(save_dir, k + ".json"), "w") as f:
        json.dump(all_data[k], f, indent=2)

with open(os.path.join(save_dir, "all_tasks.json"), "w") as f:
    json.dump(all_data_list, f)

for sample in all_data_list:
    assert os.path.exists(sample["image"]), sample["image"]

print("total data size: ", len(all_data_list))

# 0622v1,v2 use all data, but no success
# augment_data v2, dont use slide_block_to_color_target. only include perturbation
# augment data v1, use all, but for slide_block (only success)



# put_item_in_drawer 12978
# reach_and_drag 6447
# turn_tap 2700
# slide_block_to_color_target 4291
# open_drawer 4444
# put_groceries_in_cupboard 6684
# place_shape_in_shape_sorter 5303
# put_money_in_safe 6500
# push_buttons 4740
# close_jar 7784
# stack_blocks 17021
# place_cups 11555
# place_wine_at_rack_location 6500
# light_bulb_in 7270
# sweep_to_dustpan_of_size 6395
# insert_onto_square_peg 6480
# meat_off_grill 6480
# stack_cups 12603
# total data size:  136175
            
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
    