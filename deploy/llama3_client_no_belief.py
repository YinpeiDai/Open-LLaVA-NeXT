import json
import time
import requests   

HEADERS = {"User-Agent": "CommonGrid-LLAMA3 Client"}



system_prompt_no_belief = """This is a WIDTHxHEIGHT 2D grid world where two agents collaboratively accomplish tasks. You are one of the agent. 
We use the following symbols to represent the cell:
{
    '<_>': 'empty cell',
    '<W>': 'wall cell',
    '<X>': 'unseen cell'
}
All possible actions you may take:
1. forward/left/right: move the agent in the corresponding direction.
2. pick: pick up the object in the front cell. Agent can carry at most one object at a time.
3. open: open the object. Some objects need to be opened with a key.
4. exchange: exchange the object with the other agent, only is valid when the agent is facing to the other agent and synchronously choose to exchange.
5. share [x, y]: share the information of grid at column x, row y to your partner. x,y are 0-based global coordinates, increasing from left to right, from top to bottom.
6. request action <action>: request your partner to perform action <action>.

You will be specified with either agent0 or agent1 and given:
1. The task description.
2. A 3x3 local grid observed by the specified agent.
3. Possible actions that you can take. You can only perform actions included in the possible actions list.
4. Description for objects and agent states when they are observed.
5. When the step number > WINSIZE, a memory map that accumulates all your 3x3 local observations in the history and the description for object and agent states in memory map are given. The map is represented by the cell symbols.
6. Your partner's message in last round (if any).
7. Environment feedback on your last action (if any).

Then you need to take actions to accomplish the task with another agent together. The complete map is not given to you, you should infer from the input information. But you may be given a memory map that records your previous exploration history. You should generate an action for the current step in this following dictionary format:
{
	"action": <action>
}

Be careful about the movement and the orientation of the agent in the observed grid. For example, if the specific agent is facing down, the action 'forward' will move the agent to the cell below it, not the cell above it; if the agent is facing left, the action 'forward' will move the agent to the cell on its left, not the cell on its right."""



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
    api = LlavaAPI("http://141.212.110.118:21006")
    
    
    # example without memory map
    width, height = 6, 6
    windown_size = 12 # Fixed
    system_prompt = system_prompt_no_belief.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(windown_size))
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "Step 0\nYou are agent0 at [1, 3]\nTask: pass the key to your partner\nPossible actions: left, right, forward, exchange, request <action>\n\nCurrent observed 3x3 grid:\n<W> <_> <_>\n<W> <_> <_>\n<W> <agent0> <_>\n\nDescription for object and agent states in local 3x3 grid:\n<agent0>: agent0 carrying key facing up"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"right\"}"
        },
        {
            "role": "user",
            "content": "Step 1\nYou are agent0 at [1, 3]\nTask: pass the key to your partner\nPossible actions: left, right, forward, exchange, request <action>\n\nCurrent observed 3x3 grid:\n<_> <_> <_>\n<agent0> <_> <_>\n<_> <_> <_>\n\nDescription for object and agent states in local 3x3 grid:\n<agent0>: agent0 carrying key facing right",
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"right\"}"
        },
        {
            "role": "user",
            "content": "Step 2\nYou are agent0 at [1, 3]\nTask: pass the key to your partner\nPossible actions: left, right, forward, exchange, request <action>\n\nCurrent observed 3x3 grid:\n<W> <agent0> <_>\n<W> <_> <_>\n<W> <W> <W>\n\nDescription for object and agent states in local 3x3 grid:\n<agent0>: agent0 carrying key facing down",
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"right\"}"
        },
        {
            "role": "user",
            "content": "Step 3\nYou are agent0 at [1, 3]\nTask: pass the key to your partner\nPossible actions: left, right, forward, exchange, request <action>\n\nCurrent observed 3x3 grid:\n<X> <W> <_>\n<X> <W> <agent0>\n<X> <W> <_>\n\nDescription for object and agent states in local 3x3 grid:\n<agent0>: agent0 carrying key facing left",
        },
    ] # "{\"action\": \"forward\"}"
    
    
    # example with memory map
    width, height = 10, 10
    windown_size = 12 # Fixed
    system_prompt = system_prompt_no_belief.replace("WIDTH", str(width)).replace("HEIGHT", str(height)).replace("WINSIZE", str(windown_size))
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "Step 10\nYou are agent1 at [7, 2]\nTask: explore the room\nPossible actions: left, right, forward\n\nMemory map before this step:\n<X> <X> <X> <X> <X> <X> <W> <W> <W> <X>\n<X> <X> <X> <X> <X> <X> <_> <_> <_> <X>\n<X> <X> <X> <X> <X> <X> <_> <agent1> <_> <X>\n<X> <X> <X> <X> <X> <X> <_> <_> <_> <X>\n<X> <X> <X> <X> <_> <_> <_> <_> <_> <X>\n<X> <X> <X> <_> <_> <_> <_> <_> <_> <W>\n<X> <X> <X> <_> <_> <_> <_> <_> <_> <W>\n<X> <X> <X> <_> <_> <_> <_> <_> <_> <W>\n<X> <X> <X> <X> <_> <_> <_> <X> <X> <X>\n<X> <X> <X> <X> <X> <X> <X> <X> <X> <X>\n\nDescription for object and agent states in memory map:\n<agent1>: agent1 facing up\n\nCurrent observed 3x3 grid:\n<W> <W> <W>\n<_> <_> <_>\n<_> <agent1> <_>\n\nDescription for object and agent states in local 3x3 grid:\n<agent1>: agent1 facing up",
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"left\"}"
        },
        {
            "role": "user",
            "content": "Step 11\nYou are agent1 at [7, 2]\nTask: explore the room\nPossible actions: left, right, forward\n\nCurrent observed 3x3 grid:\n<_> <_> <_>\n<_> <_> <agent1>\n<_> <_> <_>\n\nDescription for object and agent states in local 3x3 grid:\n<agent1>: agent1 facing left",
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"forward\"}"
        },
        {
            "role": "user",
            "content": "Step 12\nYou are agent1 at [6, 2]\nTask: explore the room\nPossible actions: left, right, forward\n\nCurrent observed 3x3 grid:\n<_> <_> <_>\n<_> <_> <agent1>\n<_> <_> <_>\n\nDescription for object and agent states in local 3x3 grid:\n<agent1>: agent1 facing left",
        }
    ]  # "{\"action\": \"forward\"}"
    
    response = api.get_response(messages=messages)
    print(response)
    
    