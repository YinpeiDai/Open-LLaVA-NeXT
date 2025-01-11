from collections import Counter
from copy import deepcopy
import json
import os
import uuid

from minigrid.utils.data_preprocess.rm_prompt import (
    SYSTEM_PROMPT_NO_BELIEF, 
    SYSTEM_PROMPT_ZEROTH_BELIEF, 
    SYSTEM_PROMPT_FIRST_BELIEF,
    USER_TEMPLATE,
    dict2str_action, dict2str_obs_action
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
                if agent_id == 0:
                    opponent_next_action = "unknown"
                    if step != len(data["agent1"]) - 1:
                        opponent_next_action = data["agent1"][step]["action"] if data["agent0"][step]["obs"]["opponent_next_action_predictable"] else "unknown"
                    act_dic = dict2str_action(dic["action"], dic["obs"], setting, opponent_next_action)
                else:
                    opponent_next_action = "unknown"
                    if step != len(data["agent0"]) - 1:
                        opponent_next_action = data["agent0"][step]["action"] if data["agent1"][step]["obs"]["opponent_next_action_predictable"] else "unknown"
                    act_dic = dict2str_action(dic["action"], dic["obs"], setting, opponent_next_action)
                
                obs_dic = dict2str_obs_action(dic["obs"], agent_id, act_dic, step)
                # act_dic should be good/bad
                act_dic = "<judgement>" + "GOOD" if data["agent0"]["reward"] == "1" else "BAD" + "</judgement>"
                obs_dic["act_dic"] = act_dic # type: ignore
                obs_dic["task_desc"] = dic["task_desc"]
                obs_dic['possible_actions'] = dic['possible_actions']
                obs_dic['possible_objects'] = ""
                if setting != "none":
                    obs_dic["possible_objects"] = dic["zeroth_belief"].keys()
                    obs_dic["possible_objects"] = "Objects to track:" + ",".join(obs_dic["possible_objects"]) + "\n"

                dataset.append(obs_dic)

            window = []
            for d in dataset:
                dialog = []
                window.append(d)
                if len(window) > window_size:
                    window.pop(0)
                history_context = ""
                if len(window) == window_size:
                    # Trigger memory
                    history_context = window[0]["memory_prompt"]
                    for w in window[1:]:
                        history_context += w["stepwise_prompt"]
                else:
                    for w in window:
                        history_context += w["stepwise_prompt"]
                
                user_prompt = USER_TEMPLATE.format(
                    agent_id=agent_id,
                    task_desc=d["task_desc"],
                    possible_actions=d["possible_actions"],
                    possible_objects=d["possible_objects"],
                    stacked_history=history_context
                )

                dialog.append(
                    {
                        "from": "system",
                        "value": system_prompt
                    }
                )
                dialog.append(
                    {
                        "from": "human",
                        "value": user_prompt,
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
                        "value": d["act_dic"]
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
    dirname = "/nfs/turbo/coe-chaijy/roihn/commongrid/dataset/RM/"
    for setting in ["none", "zeroth", "first"]:
        all_data = []
        for file in sorted(os.listdir(dirname)):
            if file.endswith(".json") and "llava" not in file:
                file_path = os.path.join(dirname, file)
                data = generate_data(file_path, setting=setting, success_agent_only=False, window_size=window_size)
                all_data.extend(data)
        with open(f"{dirname}/llava_format_{setting}_belief.json", "w") as fb:
            json.dump(all_data, fb)
