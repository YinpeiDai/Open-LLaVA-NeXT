from collections import defaultdict
import json
import os
import random
from llava.utils  import RLBENCH_TASKS

save_dir = "playground/rvt_llava_data_real_robot"
os.makedirs(save_dir, exist_ok=True)

# TODO: combine catastrophy data and more failure data

all_data = []
task_data = []
for task in ["pick_and_place_fruit","push_buttons","put_item_in_shelf","open_drawer"]:
    for i in range(15):
        file_path = os.path.join("augmented_data_heuristic", "real_robot/train", task, str(i), "llava.json")
        json_data =  json.load(open(file_path))
        add_end_task = False
        for sample in json_data:
            img_path = sample["image"]
            img_path = img_path.replace("augmented_data_heuristic/real_robot/", "augmented_data_heuristic/real_robot/train/")
            sample["image"] = img_path
            if sample["conversations"][1]["label"] == "task success": # only get end-of-task success
                if add_end_task: continue
                task_data.append(sample)
                add_end_task = True
            elif sample["conversations"][1]["label"] == "subgoal success": # double the expert step data
                task_data.append(sample)
                # if random.random()>0.5:
                #     task_data.append(sample)
            else:
                task_data.append(sample)
                
    print(task, len(task_data))
    all_data.extend(task_data)


print("total data size: ", len(all_data))

# with open(os.path.join(save_dir, "all_tasks_dup.json"), "w") as f:
#     json.dump(all_data, f, indent=2)


# dup
# pick_and_place_fruit 550
# push_buttons 893
# put_item_in_shelf 1364
# open_drawer 1697
# total data size:  4504

# no dup
# pick_and_place_fruit 420
# push_buttons 687
# put_item_in_shelf 1043
# open_drawer 1301
# total data size:  3451