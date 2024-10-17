from copy import deepcopy
import json
import os
import uuid

from minigrid.utils.data_preprocess.finetuned_prompt import (
    DESCRIPTIVE_SYSTEM_PROMPT_NO_BELIEF_VISION, 
    DESCRIPTIVE_SYSTEM_PROMPT_ZEROTH_BELIEF_VISION, 
    DESCRIPTIVE_SYSTEM_PROMPT_ZEROTH_AND_FIRST_BELIEF_VISION,
    dict2str_action, dict2str_obs_descriptive
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
                system_prompt = DESCRIPTIVE_SYSTEM_PROMPT_NO_BELIEF_VISION.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(window_size))
            elif setting == "zeroth":
                system_prompt = DESCRIPTIVE_SYSTEM_PROMPT_ZEROTH_BELIEF_VISION.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(window_size))
            elif setting == "first":
                system_prompt = DESCRIPTIVE_SYSTEM_PROMPT_ZEROTH_AND_FIRST_BELIEF_VISION.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(window_size))
            system_prompt = system_prompt.strip()
            # print(system_prompt)

        for agent_id in agents:
            dataset = []
            for step, dic in enumerate(data["agent" + str(agent_id)]):
                obs_dic = dict2str_obs_descriptive(dic["obs"], agent_id, step, belief_mode=setting)
                if agent_id == 0:
                    opponent_next_action = "unknown"
                    if step != len(data["agent1"]) - 1:
                        opponent_next_action = data["agent1"][step]["action"] if data["agent0"][step]["obs"]["opponent_next_action_predictable"] else "unknown"
                    act_dic = dict2str_action(dic["action"], dic["obs"], setting, opponent_next_action)
                else:
                    opponent_next_action = "unknown"
                    if step != len(data["agent0"]) - 1:
                        opponent_next_action = data["agent0"][step+1]["action"] if data["agent1"][step]["obs"]["opponent_next_action_predictable"] else "unknown"
                    act_dic = dict2str_action(dic["action"], dic["obs"], setting, opponent_next_action)
                
                obs_dic["act_dic"] = act_dic # type: ignore
                obs_dic["image"] = dic["obs"]["img_path"]
                dataset.append(obs_dic)
            
            
            for d in dataset:
                dialog = []
                dialog.append(
                    {
                        "from": "human",
                        "value": d["observation_mem"],
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
                        "image": deepcopy(d["image"]),
                        "conversations": deepcopy(dialog)
                    }                
                )
    return all_data
                
        
if __name__ == "__main__":
    window_size = 12
    dirname = "/nfs/turbo/coe-chaijy/roihn/commongrid/dataset/SFT/samples"
    for setting in ["none", "zeroth", "first"]:
        all_data = []
        for file in sorted(os.listdir(dirname)):
            if file.endswith("3k_v2.json") and "llava" not in file:
                file_path = os.path.join(dirname, file)
                print(file_path)
                data = generate_data(file_path, setting=setting, success_agent_only=False, window_size=window_size)
                all_data.extend(data)
        # print(f"Number of data: {len(all_data)}")
        with open(f"playground/llava_format_sampledata_{setting}_belief_v2_vision.json", "w") as fb:
            json.dump(all_data, fb)
