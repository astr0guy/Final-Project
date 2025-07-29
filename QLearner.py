import numpy as np
from AgentStats import AgentStats
from utils import *

class QLearner:
    identifier = 0
    iterator = 0

    def __init__(self, agent_num=1, choice_num=2, epochs=20000, agent_type="Average"):

        # Initialise Functional Variables

        QLearner.iterator = 0
        self.partner = -1
        self.id = QLearner.identifier
        self.action = False
        self.agreed = False
        self.partner_choices = set(range(agent_num))
        self.partner_choices.remove(self.id)


        # Set Exploration Rate (Epsilon Value) to the Relevant Value based on Agent Type  

        match agent_type:
            case "Risky":
                self.exploration_rate = 0.3
            case "Average":
                self.exploration_rate = 0.1  

        # Learning Rates and Discount Factors which differ based on purpose 

        self.partner_learning_rate = 0.1
        self.partner_discount_factor = 0.3
        self.vote_learning_rate = 0.1
        self.vote_discount_factor = 0.9

        # Hunt Outcome History

        self.q_table = np.full((agent_num, choice_num), 10.0, dtype=np.float32)

        # Partner Trust

        self.partner_values = np.zeros(agent_num, dtype=np.float32)
        self.partner_values[self.id] = 0

        # Data structure

        self.data = AgentStats(self.id, epochs, agent_type)



        QLearner.identifier += 1

    ########
    # VOTING
    ########

    def vote(self):
        if np.random.rand() < self.exploration_rate:
            self.action = bool(np.argmin(self.q_table[self.partner]))
        else:
            self.action = bool(np.argmax(self.q_table[self.partner]))
        self.data.hunt_choice_history[QLearner.iterator] = (self.action)
        return self.action

    ################
    # CHOOSE PARTNER
    ################

    def choose_partner(self, partner_pool, reputations, gossip_value):

        self.partner = -1

        # Gather a List of Available Singletons

        available_partners = list(self.partner_choices.intersection(partner_pool))

        # If Exploring, Choose a Random Candidate 

        if np.random.rand() < self.exploration_rate:
            self.partner = available_partners[np.random.randint(len(available_partners))]
            return self.partner

        normalised_partner_values = [0] * len(available_partners)

        # Ensures that "Infinite Gossip" Allows for Circumvention of the Q-Learner's Trust Table

        if gossip_value != "Inf.":

            # Create a list of the Available Partner's Associate Q-Values and Min-Max Feature Scale them to be between 1 and 2

            partner_values = []
            for w in available_partners:
                partner_values.append(self.partner_values[w])
            normalised_partner_values = normalise(partner_values, 1, 2)
        else:
            gossip_value = 1.0

        # Min-Max Feature Scale Agent Reputations to be between 1 and 2

        normalised_reputations = normalise(reputations, 1, 2)

        # Add Reputations to the list of Available Partner's Values Scaled by the Gossip Value

        for v in range(len(normalised_partner_values)):
            normalised_partner_values[v] += gossip_value * normalised_reputations[v]

        # Have the Agent choose a Weightedly Random partner with Probabilities Scaling with said Candidate's Value

        cumulative_probabilities = cum_probs(normalised_partner_values)      
        partner_choice = np.random.rand()

        for i in range(len(cumulative_probabilities)):
            if partner_choice <= cumulative_probabilities[i]:
                self.partner = available_partners[i]
                return self.partner
        raise BaseException

    #################
    # REWARD FUNCTION
    #################

    def reward(self, reward=0):

        self.q_table[self.partner, int(self.action)] = q_update(self.vote_learning_rate, 
                                                                self.vote_discount_factor, 
                                                                reward, 
                                                                [0],
                                                                self.q_table[self.partner, int(self.action)])
        self.partner_values[self.partner] = q_update(self.partner_learning_rate, 
                                                                self.partner_discount_factor, 
                                                                reward, self.partner_values, 
                                                                self.partner_values[self.partner])
        self.partner_values[self.id] = 0.0        

        self.data.update(reward, self.partner, self.action, self.agreed)


# Subclasses of Q-Learner for slight efficiency gains

class HareBrained(QLearner):
    def __init__(self, agent_num=1, choice_num=2, epochs=20000, risk_taker_val="Hare-Brained"):
        super().__init__(agent_num, choice_num, epochs, risk_taker_val)
        self.exploration_rate = 0.1
        self.action = True
    
    def vote(self):
        return True

class Hareless(QLearner):
    def __init__(self, agent_num=1, choice_num=2, epochs=20000, risk_taker_val="Hareless"):
        super().__init__(agent_num, choice_num, epochs, risk_taker_val)
        self.exploration_rate = 0.1
        self.action = False
    
    def vote(self):
        return False