from QLearner import *
from AgentStats import AgentStats
from Prey import Stag, Hare, Bison, Random
import numpy as np
from utils import *

class Hunt:

    def __init__(self, hunt_type=Stag, pair_num=1, choice_num=2, gossip_value=0):
        ######################
        # FUNCTIONAL VARIABLES
        ######################

        # Hunt Choices

        self.ANIMALS = (hunt_type, Hare)

        # Number of Hunts

        self.epochs = 20000

        # Variables for Pairing

        self.agents = []
        self.pairs = []
        self.pair_num = pair_num

        # Variables for Reputation Management

        self.gossip_value = gossip_value
        self.reputations = np.ones(pair_num * 2, dtype=np.float32)

        # Data-gathering Variables

        self.rep_over_time = []
        self.total_reward = 0
        self.agreement = np.empty(self.epochs, dtype=bool)

        # Environmental Discount Factor for Reputation Updates

        self.discount_factor = 0.9

        # Initialise Agents of Different Types

        for x in range(pair_num * 2):
            if x < pair_num / 2:
                self.agents.append(QLearner(pair_num *2, choice_num, self.epochs, "Risky"))
            elif x < pair_num / 2 * 3:
                self.agents.append(QLearner(pair_num *2, choice_num, self.epochs))
            elif x < pair_num / 4 * 7:
                self.agents.append(Hareless(pair_num *2, choice_num, self.epochs))
            else:
                self.agents.append(HareBrained(pair_num *2, choice_num, self.epochs))
        Bison.win_val = 15

    ####################
    # VOTING AND RESULTS
    ####################

    def poll_agents(self):

        # Store a Temporary Copy of Reputations to Maintain Atomicity during the Sequential Updating of Agents
        temp_reps = self.reputations.copy()

        for y in range(self.pair_num):

            # Vote Collection

            agreement = 0
            pair = self.pairs[y]
            agent_1_vote = self.agents[pair[0]].vote()
            agent_2_vote = self.agents[pair[1]].vote()

            # Vote Results Analysis 

            if agent_1_vote == agent_2_vote:
                self.agents[pair[0]].agreed = True
                self.agents[pair[1]].agreed = True
                agent_1_reward = self.ANIMALS[agent_1_vote].win()
                agent_2_reward = self.ANIMALS[agent_2_vote].win()
                agreement += 1
            else:
                self.agents[pair[0]].agreed = False
                self.agents[pair[1]].agreed = False
                agent_1_reward = self.ANIMALS[agent_1_vote].lose()
                agent_2_reward = self.ANIMALS[agent_2_vote].lose()
            
            # Reward Distribution

            self.total_reward += agent_1_reward + agent_2_reward
            self.agents[pair[0]].reward(agent_1_reward)
            self.agents[pair[1]].reward(agent_2_reward)

            # Reputation Update

            discounts = normalise(temp_reps, 0, 1)

            self.reputations[pair[0]] = q_update(
                discounts[pair[0]],
                self.discount_factor,   
                agent_1_reward, 
                temp_reps,
                temp_reps[pair[1]] 
                )
            
            self.reputations[pair[1]] = q_update(
                discounts[pair[1]],
                self.discount_factor,
                agent_2_reward, 
                temp_reps,
                temp_reps[pair[0]] 
                )
            
            self.agreement[y] = agreement
        if self.gossip_value == 0:
            self.reputations = [0] * (2 * self.pair_num)

    ################
    # PAIR SELECTION
    ################

    def pair_selection(self):

        # Initialise Storage Structures 

        self.pairs = []
        partner_picks = {x: -1 for x in range(self.pair_num * 2)}

        while len(self.pairs) != self.pair_num:
            discord = True

            # Each agent elects its preferred candidate from the pool of available candidates

            for agent in partner_picks.keys():
                partner_picks[agent] = (self.agents[agent].choose_partner(partner_picks.keys(), self.reputations.copy(), self.gossip_value))

            # Singleton (unpaired) Agents are Listed

            singletons = list(partner_picks.keys())

            # Any 2 Partners who Voted for one another are Paired.

            for agent in singletons:
                check = partner_picks.get(agent)
                if partner_picks.get(check) == agent:
                    self.pairs.append((agent, check))
                    self.agents[agent].partner = check
                    self.agents[check].partner = agent
                    partner_picks.pop(agent)
                    partner_picks.pop(check)
                    discord = False

            # If only two Singleton Agents remain, they are paired.
            # If this is how Agent Pairing concludes, Epsilon Decay occurs


            if len(partner_picks) == 2:
                singletons = list(partner_picks.keys())
                self.pairs.append((singletons[0], singletons[1]))
                self.agents[singletons[0]].partner = singletons[1]
                self.agents[singletons[1]].partner = singletons[0]
                for i in range(len(self.agents)):
                    self.agents[i].exploration_rate /= 1.001
                break

            # If more than two Singleton Agents all voted for Candidates who did not return their vote, they are paired Randomly

            elif discord:
                self.random_pairing(partner_picks.keys())

            # If, in the prior round of Selections, some Agents were paired, the cycle renews

            else:
                for x in partner_picks.keys():
                    partner_picks[x] = -1

    # Random Singleton Agent Pairing in the case of Discord

    def random_pairing(self, singletons):
        singletons = list(singletons)
        while singletons:
            agent1, agent2 = -1, -1
            while agent1 == agent2:
                agent1, agent2 = \
                    singletons[np.random.randint(len(singletons))], \
                    singletons[np.random.randint(len(singletons))]
            if agent1 == agent2:
                agent1 += 1
            self.pairs.append((agent1, agent2))
            self.agents[agent1].partner = agent2
            self.agents[agent2].partner = agent1
            singletons.remove(agent1)
            singletons.remove(agent2)

    #######################
    # SIMULATION PARAMETERS
    #######################

    def simulate(self):
        print(f"Gossip value: {self.gossip_value}")

        print(f"{self.ANIMALS[0]} Hunt")
        print("Running...")

        print()

        # Each agent, once paired, goes through 20 Hunts.

        for _ in range(self.epochs//20):
            self.pair_selection()
            for _ in range(self.epochs// (self.epochs // 20)):
                self.poll_agents()

                # Data Collation

                AgentStats.iterator +=1
                if self.gossip_value != 0:
                    self.rep_over_time.append([x / sum(self.reputations) for x in self.reputations])

        print(self.total_reward)

        # Housekeeping

        self.pairs = []
        QLearner.identifier = 0
        AgentStats.iterator = 0
        print()
        return self.total_reward