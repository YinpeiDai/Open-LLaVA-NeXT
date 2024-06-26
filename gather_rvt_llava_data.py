import json
import os
from llava.utils  import RLBENCH_TASKS

data_path  = "augmented_data_heuristic"
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
    print(task, len(task_data))

with open(os.path.join(save_dir, "all_tasks.json"), "w") as f:
    json.dump(all_data, f)


# 0622v1,v2 use all data, but no success
# augment_data v2, dont use slide_block_to_color_target. only include perturbation
# augment data v1, use all, but for slide_block (only success)




print("total data size: ", len(all_data))
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
    