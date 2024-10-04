import json
import os
from llava.utils  import RLBENCH_TASKS

data_path  = "augmented_rlbench"
save_dir = "playground/racer_llava_data"
os.makedirs(save_dir, exist_ok=True)

all_data = []
for task in RLBENCH_TASKS:
    task_data = []
    for d in ["train", "val"]:
        for ep in sorted(os.listdir(os.path.join(data_path,  d, task)), key=lambda x: int(x.split("_")[0])):
            file_path = os.path.join(data_path, d, task, ep, "llava.json")
            if not os.path.exists(file_path):
                continue
            task_data.extend(json.load(open(file_path)))
    all_data.extend(task_data)

    with open(os.path.join(save_dir, task + ".json"), "w") as f:
        json.dump(task_data, f, indent=1)
    print(task, len(task_data))

with open(os.path.join(save_dir, "all_tasks.json"), "w") as f:
    json.dump(all_data, f)
print("total data size: ", len(all_data))