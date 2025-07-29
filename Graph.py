import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from os import makedirs, path

class Graph:

    iter_count = 1

    def __init__(self, agents_data, hunt_type):

        # Initialise Utility Variables

        self.agents_data = agents_data
        self.type = str(hunt_type.__name__)
        self.agent_count = len(agents_data)
        self.epoch_count = len(self.agents_data[0].running_objective_reward) - 1

        # Initialise Pyplot

        plt.rcParams["figure.figsize"] = (8,8)
        plt.rcParams.update({'font.size': 15})

        # Ensure Filepath Presence

        makedirs(path.dirname(f"./{self.type}_partner_history/"), exist_ok=True)
        makedirs(path.dirname(f"./{self.type}_reward/"), exist_ok=True)
        makedirs(path.dirname(f"./{self.type}_hunt_choice/"), exist_ok=True)
        makedirs(path.dirname(f"./{self.type}_hunt_vote_conformity/"), exist_ok=True)
        makedirs(path.dirname(f"./{self.type}_hunt_reputations_over_time/"), exist_ok=True)



    def cumulative_partner_graph(self, agent=-1):

        if agent == -1:
            agents = range(self.agent_count)
            file_name_prefix = "all_agent"
        else:
            agents = [agent]

        for a in agents:
            if agent != -1:
                file_name_prefix = f"{self.agents_data[a].agent_type}_agent_{a}"
            plt.figure()
            plt.title(f"{self.type} Hunt: {self.agents_data[a].agent_type} Agent {a}'s Partner Selections Over Time")
            ordered_agents = list(self.agents_data[a].partner_choice_history.keys())
            ordered_agents.sort()
            for y in ordered_agents:
                y_coords = self.agents_data[a].partner_choice_history[y]
                x_coords = range(self.epoch_count+1)
                plt.plot(x_coords, y_coords, label=f"Agent {y}", color=self.colour(y))
            plt.ylabel("Number of Selections")
            plt.xlabel("Epochs")
            plt.legend()
            plt.draw()
            plt.savefig(f"./{self.type}_partner_history/{self.type}_{file_name_prefix}_partner_history{Graph.iter_count}.png")
            plt.close()


    def cumulative_reward_graph(self, agent=-1):

        plt.figure()
        plt.ylabel("Reward")
        plt.xlabel("Epochs")

        if agent == -1:
            file_name_prefix = "all_agent"
            graph_title_prefix = "All Agent"
            for x in range(self.agent_count):
                y_coords = [self.agents_data[x].running_objective_reward[i] / (i+1) for i in range(1, len(self.agents_data[x].running_objective_reward))]
                x_coords = range(len(y_coords))
                plt.plot(x_coords, y_coords, label=f"Agent {x}",color=self.colour(x))

        else:
            file_name_prefix = f"{self.agents_data[agent].agent_type}_agent_{agent}"
            graph_title_prefix = f"{self.agents_data[agent].agent_type} Agent {agent}"
            y_coords = [self.agents_data[agent].running_objective_reward[i] / (i+1) for i in range(1, len(self.agents_data[agent].running_objective_reward))]
            x_coords = range(self.epoch_count)
            plt.plot(x_coords, y_coords, label=f"Agent {agent}",color=self.colour(agent))

        plt.title(f"{self.type} Hunt: {graph_title_prefix} Learner Reward Over Time")
        plt.legend()
        plt.draw()
        plt.savefig(f"./{self.type}_reward/{self.type}_{file_name_prefix}_reward_graph_{Graph.iter_count}.png")
        plt.close()

    def average_reward_graph(self):
        plt.figure()  
        plt.title(f"{self.type} Hunt: Average Hunt Rewards Over Time")
        average_risky_reward = np.zeros(self.epoch_count)
        average_average_reward = np.zeros(self.epoch_count)
        average_hareless_reward = np.zeros(self.epoch_count)
        average_hare_brained_reward = np.zeros(self.epoch_count)
        for i in range(self.agent_count):
            match self.agents_data[i].agent_type:
                case "Average":
                    for j in range(1,self.epoch_count):
                        average_average_reward[j] += self.agents_data[i].running_objective_reward[j] / (j+1)
                case "Risky":
                    for j in range(1,self.epoch_count):
                        average_risky_reward[j] += self.agents_data[i].running_objective_reward[j] / (j+1)
                case "Hareless":
                    for j in range(1,self.epoch_count):
                        average_hareless_reward[j] += self.agents_data[i].running_objective_reward[j] / (j+1)
                case "Hare-Brained":
                    for j in range(1,self.epoch_count):
                        average_hare_brained_reward[j] += self.agents_data[i].running_objective_reward[j] / (j+1)

        x_coords = list(range(self.epoch_count))        
        plt.plot(x_coords, [ y / (self.agent_count/2) for y in average_average_reward ], label="Average Agents", color="blue")
        plt.plot(x_coords, [ y / (self.agent_count/4) for y in average_risky_reward ], label="Risky (Explorative) Agents", color="darkorange")
        plt.plot(x_coords, [ y for y in average_hareless_reward ], label="Hareless Agent", color="forestgreen")
        plt.plot(x_coords, [ y for y in average_hare_brained_reward ], label="Hare-Brained Agent", color="yellow")
        plt.ylabel("Group Average Reward")
        plt.xlabel("Epochs")
        plt.legend()
        plt.draw()
        plt.savefig(f"./{self.type}_reward/{self.type}_average_reward_graph_{Graph.iter_count}.png")
        plt.close()
               

    def hunt_choice_history_graph(self, agent=-1):

        plt.figure()

        if agent == -1:
            agents = range(self.agent_count)
        else:
            agents = [agent]

        for a in agents:
            file_name_prefix = f"{self.agents_data[a].agent_type}_agent_{a}"                   
            stag_like = [0]
            hare = [0]
            stag_like_total = 0
            hare_total = 0
            plt.title(f"{self.type} Hunt: {self.agents_data[a].agent_type} Agent Hunt Choice History")
            i = 0
            for y in self.agents_data[a].hunt_choice_history:
                if y == 0:
                    stag_like_total += 1
                elif y == 1:
                    hare_total += 1
                stag_like.append(stag_like_total / (i + 1) )
                hare.append(hare_total / (i + 1) )
                i += 1
                x_coords = range(len(hare))
            plt.stackplot(x_coords, (hare,stag_like), labels=("Hare",self.type))
            plt.ylabel("Choice Proportion")
            plt.xlabel("Epochs")
            plt.legend()
            plt.draw()
            plt.savefig(f"./{self.type}_hunt_choice/{self.type}_{file_name_prefix}_hunt_choice_history_{Graph.iter_count}.png")
            plt.close()


    def rot_graph(self, rot):
        rot = np.asarray(rot)
        rot = np.swapaxes(rot, 0, 1)
        new_rot= np.zeros((self.agent_count, self.epoch_count//500), dtype=np.float32)
        for i in range(self.agent_count):
            for j in range(0, self.epoch_count, 500):
                new_rot[i, j//500] = sum(rot[i, j:j+(self.epoch_count//500)])/(self.epoch_count//500)

        plt.figure()
        plt.ylabel("Reputation Proportion")
        plt.xlabel("Epochs")
        labels=[f"Agent {i}" for i in range(self.agent_count)]
        colours=[self.colour(i) for i in range(self.agent_count)]
        x_coords = list(range(0, self.epoch_count, 500))
        for x in range(self.agent_count):
            plt.plot(x_coords, new_rot[x], label=labels[x], color=colours[x]) 
        plt.title(f"{self.type} Hunt: Agent Reputations Over Time")
        plt.legend()
        plt.draw()
        plt.savefig(f"./{self.type}_hunt_reputations_over_time/{self.type}_rot_graph_{Graph.iter_count}.png")
        plt.close()


    def agreement_graph(self, agent=-1):

        plt.figure()
        plt.ylabel("Agreement")
        plt.xlabel("Epochs")

        if agent == -1:
            file_name_prefix = "averaged"
            graph_title_prefix = "Averaged"
            agreed = 0
            disagreed = 0
            agreements = [0]
            disagreements = [0]
            for x in range(self.epoch_count):
                agreed = 0
                disagreed = 0
                for y in range(self.agent_count):
                    if self.agents_data[y].agreements[x]:
                        agreed += 1
                    else:
                        disagreed += 1
                if agreed % 2 == 1 or disagreed % 2 == 1:
                    raise BaseException
                else:
                    agreements.append(agreements[x] + agreed/2)
                    disagreements.append(disagreements[x] + disagreed/2)  
            y_coords = (agreements, disagreements)
            x_coords = range(len(agreements))
            plt.stackplot(x_coords, y_coords, labels=("Agreed", "Disagreed"))

        else:
            agreements = [0]
            disagreements = [0]
            file_name_prefix = f"{self.agents_data[agent].agent_type}_agent_{agent}"
            graph_title_prefix = f"{self.agents_data[agent].agent_type} Agent {agent}"
            for x in range(self.epoch_count):
                    
                if self.agents_data[agent].agreements[x]:
                    agreements.append(agreements[x] + 1)
                    disagreements.append(disagreements[x])
                else:
                    agreements.append(agreements[x])
                    disagreements.append(disagreements[x] + 1)

            for x in range(2, self.epoch_count+2):
                agreements[x-1] /= x
                disagreements[x-1] /= x

            y_coords = (agreements, disagreements)
            x_coords = list(range(len(agreements)))
            plt.stackplot(x_coords, y_coords, labels=("Agreed", "Disagreed"))

        plt.title(f"{self.type} Hunt: {graph_title_prefix} Vote Conformity Rate")
        plt.legend()
        plt.draw()
        plt.savefig(f"./{self.type}_hunt_vote_conformity/{self.type}_{file_name_prefix}_hunt_vote_conformity_graph_{Graph.iter_count}.png")
        plt.close()



    # Matches Agents to Associated Colours

    def colour(self, agent_number):
        match agent_number:
            case 0:
                return "darkorange"
            case 1:
                return "orangered"
            case 2:
                return "darkblue"                
            case 3:
                return "mediumblue"
            case 4:
                return "blue"        
            case 5:
                return "cornflowerblue"
            case 6:
                return "forestgreen"    
            case 7:
                return "yellow"