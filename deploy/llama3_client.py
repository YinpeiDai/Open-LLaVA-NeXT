import json
import time
import requests   

HEADERS = {"User-Agent": "CommonGrid-LLAMA3 Client"}

system_prompt = """
This is a 12x12 2D grid world where two agents collobratively accomplish tasks. 
We use the following symbols to represent the cell:
{
    '<_>': 'empty cell',
    '<agent0>': 'agent0',
    '<agent1>': 'agent1',
    '<W>': 'wall cell',
    '<X>': 'unseen cell'
}
Other objects will be given in the inputs in the form of '<object+id>'

You can take actions:
1. forward/backward/left/right: move the agent in the corresponding direction.
2. pick: pick up the object in the current cell, only is valid when agent is facing to the object.
3. exchange: exchange the object with the other agent, only is valid when the agent is facing to the other agent.

You will be specified with either agent0 or agent1 and given:
1. The task description.
2. A 3x3 local grid observed by the specified agent.
3. Possible actions that you can take.
4. Description for objects and agent states when they are observed.

Then you need to predict the correction action the specified agent should take to accomplish the task with another agent together. The complete map is not given to you, you should infer from the input information.

Be careful about the movement and the orientation of the agent in the observed grid. For example, if the specific agent is facing down, the action 'move forward' will move the agent to the cell below it, not the cell above it; if the agent is facing left, the action 'move forward' will move the agent to the cell on its left, not the cell on its right.
"""

class LlavaAPI:
    def __init__(self, addr):
        self.addr = addr
    

    def get_response(self, messages):
        pload = {
            "messages": messages
        }
        while True:
            try:
                # Stream output
                response = requests.post(self.addr + "/worker_generate_stream",
                    headers=HEADERS, json=pload, stream=True, timeout=10)
                generated_text = ""
                for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode())
                        if data["error_code"] == 0:
                            generated_text += data["text"]
                break
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
            time.sleep(1)
        return generated_text
    


if __name__ == "__main__":
    # use this template
    # step_id = 0
    # agent_id = "agent0"
    # task_desc = "pickup a key and open the door"
    # possible_actions = "left, right, forward, open, pick, exchange"
    # grid_text = "<W><agent0><_>\n<W><key0><_>\n<door0><_><agent1>\n"
    # obj_desc = "<agent0>: agent0 facing down\n<key0>: a red key\n<door0>: a red locked door\n<agent1>: agent1 facing up\n"
    # content_template = "Step {step_id}\nYou are {agent_id}\nTask: {task_desc}\nPossible actions: {possible_actions}\n\nCurrent observed 3x3 grid:\n{grid_text}\nDescription for object and agent states:\n{obj_desc}".format(step_id=step_id, agent_id=agent_id, task_desc=task_desc, possible_actions=possible_actions, grid_text=grid_text, obj_desc=obj_desc)
    
    # api = LlavaAPI("http://127.0.0.1:21004")
    api = LlavaAPI("http://141.212.106.177:21004")
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "Step 0\nYou are agent0\nTask: pickup a key and open the door\nPossible actions: left, right, forward, open, pick, exchange\n\nCurrent observed 3x3 grid:\n<W><agent0><_>\n<W><key0><_>\n<door0><_><agent1>\n\nDescription for object and agent states:\n<agent0>: agent0 facing down\n<key0>: a red key\n<door0>: a red locked door\n<agent1>: agent1 facing up\n"
        },
        {
            "role": "assistant",
            "content": "Action: pick."
        },
        {
            "role": "user",
            "content": "Step 1\nYou are agent0\nTask: pickup a key and open the door\nPossible actions: left, right, forward, open, pick, exchange\n\nCurrent observed 3x3 grid:\n<W><agent0><_>\n<W><_><agent1>\n<door0><_><_>\n\nDescription for object and agent states:\n<agent0>: agent0 carrying key facing down\n<agent1>: agent1 facing up\n<door0>: a red locked door\n"
        },
        {
            "role": "assistant",
            "content": "Action: forward."
        },
        {
            "role": "user",
            "content": "Step 2\nYou are agent1\nTask: pickup a key and open the door\nPossible actions: left, right, forward, open, pick, exchange\n\nCurrent observed 3x3 grid:\n<W><agent0><_>\n<W><_><agent1>\n<door0><_><_>\n\nDescription for object and agent states:\n<agent0>: agent0 carrying key facing down\n<agent1>: agent1 facing up\n<door0>: a red locked door\n"
        },
        {
            "role": "assistant",
            "content": "Action: forward."
        },
        {
            "role": "user",
            "content": "Step 3\nYou are agent1\nTask: pickup a key and open the door\nPossible actions: left, right, forward, open, pick, exchange\n\nCurrent observed 3x3 grid:\n<W><agent0><_>\n<W><_><agent1>\n<door0><_><_>\n\nDescription for object and agent states:\n<agent0>: agent0 carrying key facing down\n<agent1>: agent1 facing up\n<door0>: a red locked door\n"
        },
        {
            "role": "assistant",
            "content": "Action: left."
        },
        {
            "role": "user",
            "content": "Step 4\nYou are agent1\nTask: pickup a key and open the door\nPossible actions: left, right, forward, open, pick, exchange\n\nCurrent observed 3x3 grid:\n<W><agent0><_>\n<W><_><agent1>\n<door0><_><_>\n\nDescription for object and agent states:\n<agent0>: agent0 carrying key facing down\n<agent1>: agent1 facing up\n<door0>: a red locked door\n"
        }
    ]


    response = api.get_response(messages=messages)
    print(response)
    
    