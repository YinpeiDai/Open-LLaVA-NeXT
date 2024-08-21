from collections import Counter
from copy import deepcopy
import json
import os
import uuid

from llava.train.my_train_commongrid import system_prompt_no_belief, system_prompt_zeroth_belief, system_prompt_zeroth_and_first_belief

observation_template_mem = """Step {step_num}
You are agent{agent_id} at {agent_pos}
Task: {task_desc}
{possible_actions}{communicate_info}{environment_feedback}

Memory map before this step:
{memory_text}
Description for object and agent states in memory map:
{memory_obj_desc_list}

Current observed 3x3 grid:
{grid_text}
Description for object and agent states in local 3x3 grid:
{obj_desc_list}"""

# observation_template_mem_zeroth_belief = observation_template_mem_no_belief + """Your belief of the world for all agents and objects:
# {zeroth_belief}"""

# observation_template_mem_zeroth_and_first_belief = observation_template_mem_zeroth_belief + """Your belief of your partner's belief of the world for all agents and objects:
# {first_belief}"""


observation_template_no_mem = """Step {step_num}
You are agent{agent_id} at {agent_pos}
Task: {task_desc}
{possible_actions}{communicate_info}{environment_feedback}

Current observed 3x3 grid:
{grid_text}
Description for object and agent states in local 3x3 grid:
{obj_desc_list}"""

# observation_template_no_mem_zeroth_belief = observation_template_no_mem_no_belief + """Your belief of the world for all agents and objects:
# {zeroth_belief}"""

# observation_template_no_mem_zeroth_and_first_belief = observation_template_no_mem_zeroth_belief + """Your belief of your partner's belief of the world for all agents and objects:
# {first_belief}"""

# action_no_belief_template = {
#     "action": None
#     }

# action_template_zeroth_belief = {
#     "Your belief of the world": None

# }
# action_template_zeroth_and_first_belief = """{
#     "Your belief of the world": "ZERO_BELIEF",
#     "Your belief of your partner's belief of the world": "FIRST_BELIEF",
#     "action": "ACTION"
# }"""



def dict2str(dict_data, agent_id, step_num):
    obsdict_data = dict_data["obs"]
    communicate = obsdict_data["communicate"] if "communicate" in obsdict_data else None
    act_response = obsdict_data["act_response"] if "act_response" in obsdict_data else None
    communicate_info = f"\nYour partner's message: \"{communicate.strip()}\"" if communicate else ""
    environment_feedback = f"Environment feedback for last action: {act_response}" if act_response else ""
    if environment_feedback:
        communicate_info += "\n"

    observation_no_mem = observation_template_no_mem.format(
        step_num=step_num,
        agent_id=agent_id,
        agent_pos=str(obsdict_data["agent_pos"]),
        task_desc=obsdict_data["task_desc"],
        possible_actions=obsdict_data["possible_actions"],
        communicate_info=communicate_info,
        environment_feedback=environment_feedback,
        grid_text=obsdict_data["grid_text"],
        obj_desc_list="\n".join(obsdict_data["obj_desc_list"])
    )

    observation_mem = observation_template_mem.format(
        step_num=step_num,
        agent_id=agent_id,
        agent_pos=str(obsdict_data["agent_pos"]),
        task_desc=obsdict_data["task_desc"],
        possible_actions=obsdict_data["possible_actions"],
        communicate_info=communicate_info,
        environment_feedback=environment_feedback,
        memory_text=obsdict_data["memory_text"],
        memory_obj_desc_list="\n".join(obsdict_data["memory_obj_desc_list"]),
        grid_text=obsdict_data["grid_text"],
        obj_desc_list="\n".join(obsdict_data["obj_desc_list"])
    )

    action = dict_data["action"]
    zeroth_belief_str = obsdict_data["zeroth_belief"]
    first_belief_str =  obsdict_data["first_belief"]


    action_no_belief = {
        "action": action
    }
    action_zeroth_belief = {
        "Your belief of the world": zeroth_belief_str,
        "action": action
    }
    action_zeroth_and_first_belief = {
        "Your belief of the world": zeroth_belief_str,
        "Your belief of your partner's belief of the world": first_belief_str,
        "action": action
    }

    return {
        "observation_no_mem": observation_no_mem.replace('><', '> <'),
        "observation_mem": observation_mem.replace('><', '> <'),
        "action_no_belief": action_no_belief,
        "action_zerothbelief": action_zeroth_belief,
        "action_zeroth_and_first_belief": action_zeroth_and_first_belief
    }

  
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
            if setting == "no_belief":
                system_prompt = system_prompt_no_belief.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(window_size))
            elif setting == "zeroth_belief":
                system_prompt = system_prompt_zeroth_belief.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(window_size))
            elif setting == "zeroth_and_firstbelief":
                system_prompt = system_prompt_zeroth_and_first_belief.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(window_size))
            system_prompt = system_prompt.strip()
            # print(system_prompt)

        for agent_id in agents:
            dataset = [dict2str(dic, agent_id, step) for step, dic in enumerate(data["agent" + str(agent_id)])]
            if setting == "no_belief":
                act_key = "action_no_belief"
            elif setting == "zeroth_belief":
                act_key = "action_zerothbelief"
            elif setting == "zeroth_and_firstbelief":
                act_key = "action_zeroth_and_first_belief"
            
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
                        "value": json.dumps(d[act_key])
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
                                "value": json.dumps(d[act_key])
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
    dirname = "/home/daiyp/Open-LLaVA-NeXT/playground/commongrid/dataset/SFT/meta"
    for setting in ["no_belief", "zeroth_belief", "zeroth_and_first_belief"]:
        all_data = []
        for file in sorted(os.listdir(dirname)):
            if file.endswith(".json") and "llava" not in file:
                file_path = os.path.join(dirname, file)
                data = generate_data(file_path, setting=setting, success_agent_only=False, window_size=window_size)
                all_data.extend(data)
        with open(f"{dirname}/llava_format_{setting}.json", "w") as fb:
            json.dump(all_data, fb)
