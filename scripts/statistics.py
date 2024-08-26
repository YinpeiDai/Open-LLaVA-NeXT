from collections import defaultdict
import json
import matplotlib.pyplot as plt



def plot_statitsics(path, task_name):
    with open(path, 'r') as f:
        data = json.load(f)

    total_episode = len(data)
    agent_number = [0, 0]
    lengths = []
    actions = defaultdict(int)
    consecutive_actions = defaultdict(lambda: defaultdict(int))

    for item in data:
        success_agent = item['success_agent']
        if success_agent in [0, "agent0"]:
            agent_number[0] += 1
        elif success_agent in [1, "agent1"]:
            agent_number[1] += 1
        else:
            agent_number[0] += 1
            agent_number[1] += 1
        ep_len = len(item['agent0'])
        assert len(item["agent0"]) == len(item["agent1"])
        lengths.append(ep_len)
        agent0_action_prev = None
        agent1_action_prev = None

        for agent0_turn, agent1_turn in zip(item['agent0'], item['agent1']):
            agent0_action = agent0_turn['action']
            agent1_action = agent1_turn['action']

            actions[agent0_action] += 1
            actions[agent1_action] += 1

            if agent0_action_prev is not None:
                consecutive_actions[agent0_action_prev][agent0_action] += 1
            if agent1_action_prev is not None:
                consecutive_actions[agent1_action_prev][agent1_action] += 1
            
            agent0_action_prev = agent0_action
            agent1_action_prev = agent1_action

    print("total episode: ", total_episode)
    print("agent0 vs. agent1: ", agent_number[0], " vs. ", agent_number[1])
    avg_len = sum(lengths) / len(lengths)
    print(f"average length: {avg_len:.2f}")

    # plot distribution of lengths
    plt.figure()
    plt.hist(lengths, bins=range(0, max(lengths)+1, 1))
    plt.savefig(f"lengths_{task_name}.png")
    plt.close()

    # plot distribution of actions
    plt.figure()
    plt.bar(actions.keys(), actions.values())
    if len(actions) > 6:
        plt.xticks(rotation=90)
    plt.tight_layout() 
    plt.savefig(f"actions_{task_name}.png")
    plt.close()

    # plot gray heatmap of consecutive actions using plt
    plt.figure()
    plt.imshow([[consecutive_actions[a][b] for b in actions.keys()] for a in actions.keys()], cmap='gray')
    plt.xticks(range(len(actions)), actions.keys())
    plt.yticks(range(len(actions)), actions.keys())
    if len(actions) > 6:
        plt.xticks(rotation=90)
    plt.savefig(f"consecutive_actions_{task_name}.png")
    plt.close()



    
if __name__ == "__main__":
    dirname = "/home/daiyp/Open-LLaVA-NeXT/playground/"
    dic = {
        "exchange" : f"{dirname}/commongrid/dataset/SFT/meta/exchange_and_req_5k_v1.json",
        "find": f"{dirname}/commongrid/dataset/SFT/meta/find_5k_v1.json",
        "explore": f"{dirname}/commongrid/dataset/SFT/meta/explore_5k_v1.json",
        "find_and_share": f"{dirname}/commongrid/dataset/SFT/meta/find_and_share_5k_v1.json",
        "goto": f"{dirname}/commongrid/dataset/SFT/meta/goto_5k_v1.json",
        "open": f"{dirname}/commongrid/dataset/SFT/meta/open_5k_v1.json",
        "pick": f"{dirname}/commongrid/dataset/SFT/meta/pick_5k_v1.json",
    }

    for task_name, path in dic.items():
        plot_statitsics(path, task_name)
        print("done: ", task_name)
        print()
