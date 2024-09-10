path = "/home/daiyp/Open-LLaVA-NeXT/playground/accessibility_data/all_eval"

import os

files = os.listdir(path)
origin_fiels = [file for file in files if "processed.jsonl" in file]
origin_fiels = sorted(origin_fiels)
predict_files = [file for file in files if "prediction.jsonl" in file]
predict_files = sorted(predict_files)

# for origin_fiels, predict_files in zip(origin_fiels, predict_files):
#     with open(os.path.join(path, origin_fiels), "r") as f:
#         origin_data = f.readlines()
    
#     with open(os.path.join(path, predict_files), "r") as f:
#         predict_data = f.readlines()

#     assert len(origin_data) == len(predict_data)

target_path  = "/home/daiyp/Open-LLaVA-NeXT/playground/accessibility_data/all_eval/results"

for predict_files in predict_files:
    os.system(f"mv {os.path.join(path, predict_files)} {os.path.join(target_path, predict_files)}")