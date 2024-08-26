from collections import Counter
from copy import deepcopy
import json
import os
import uuid

from minigrid.utils.data_preprocess.finetuned_prompt import (
    SYSTEM_PROMPT_NO_BELIEF, 
    SYSTEM_PROMPT_ZEROTH_BELIEF, 
    SYSTEM_PROMPT_ZEROTH_AND_FIRST_BELIEF,
    dict2str_action, dict2str_obs
    )


def generate_data(
    file_path,
    setting = "no_belief", # "no_belief", "zeroth_belief", "zeroth_and_firstbelief"
    success_agent_only = False,
    window_size = 30
    ):
    all_data = []
    with open(file_path, "r") as fb:
        raw_data = json.load(fb)
    for idx, data in enumerate(raw_data):
        if not data["terminated"]: continue
        if success_agent_only:
            success_agent = data["success_agent"]
            if success_agent in ["agent0", 0]:
                agents = [0]
            elif success_agent in ["agent1", 1]:
                agents = [1]
            else:
                raise ValueError("Invalid success_agent value")
        else:
            agents = [0, 1]
        
        width = data["width"]
        height = data["height"]
        if idx == 0:
            # input("---System Prompt (Enter):")
            if setting == "none":
                system_prompt = SYSTEM_PROMPT_NO_BELIEF.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(window_size))
            elif setting == "zeroth":
                system_prompt = SYSTEM_PROMPT_ZEROTH_BELIEF.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(window_size))
            elif setting == "first":
                system_prompt = SYSTEM_PROMPT_ZEROTH_AND_FIRST_BELIEF.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(window_size))
            system_prompt = system_prompt.strip()
            # print(system_prompt)

        for agent_id in agents:
            dataset = []
            for step, dic in enumerate(data["agent" + str(agent_id)]):
                obs_dic = dict2str_obs(dic["obs"], agent_id, step)
                act_dic = dict2str_action(dic["action"], dic["obs"], setting)
                obs_dic["act_dic"] = act_dic # type: ignore
                dataset.append(obs_dic)
        
            
            dialog = []
            for d in dataset[:window_size]:
                dialog.append(
                    {
                        "from": "human",
                        "value": d["observation_no_mem"],
                        "info": {
                            "width": width,
                            "height": height,
                            "window_size": window_size
                        }
                    }
                )
                dialog.append(
                    {
                        "from": "gpt",
                        "value": json.dumps(d["act_dic"])
                    }
                )
            all_data.append(
                {
                    "id": uuid.uuid4().hex,
                    "conversations": deepcopy(dialog)
                }                
            )
            if len(dataset) > window_size:
                # sample every 5 steps
                for end in range(len(dataset), window_size, -5):
                    dialog = []
                    for idx, d in enumerate(dataset[end-window_size:end]):
                        if idx == 0: 
                            obs_key = "observation_mem"
                        else:
                            obs_key = "observation_no_mem"
                        dialog.append(
                            {
                                "from": "human",
                                "value": d[obs_key],
                                "info": {
                                    "width": width,
                                    "height": height,
                                    "window_size": window_size
                                }
                            }
                        )
                        dialog.append(
                            {
                                "from": "gpt",
                                "value": json.dumps(d["act_dic"])
                            }
                        )
                    all_data.append(
                        {
                            "id": uuid.uuid4().hex,
                            "conversations": deepcopy(dialog)
                        }
                    )
    return all_data
                
        
if __name__ == "__main__":
    window_size = 12
    dirname = "/home/daiyp/Open-LLaVA-NeXT/playground/commongrid/dataset/SFT/meta/samples"
    for setting in ["none", "zeroth", "first"]:
        all_data = []
        for file in sorted(os.listdir(dirname)):
            if file.endswith(".json") and "llava" not in file:
                file_path = os.path.join(dirname, file)
                data = generate_data(file_path, setting=setting, success_agent_only=False, window_size=window_size)
                all_data.extend(data)
        with open(f"{dirname}/llava_format_{setting}_belief.json", "w") as fb:
            json.dump(all_data, fb)
