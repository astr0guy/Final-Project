import numpy as np

# A Class which acts as a structure containing agent information

class AgentStats:

    iterator = 0

    def __init__(self, agent_id, epochs, agent_type):

        # Agent Type

        self.agent_type = agent_type

        # Data Types Stored

        self.running_objective_reward = np.empty(epochs+1, dtype=np.float32)
        self.running_objective_reward[0] = 0.0
        self.partner_choice_history = dict()
        self.partner_choice_tally = dict()
        self.hunt_choice_history = np.empty(epochs, dtype=bool)
        self.id = agent_id
        self.agreements = np.empty(epochs, dtype=bool)


    def update(self, new_objective_reward, newest_partner, new_choice, new_agreed):

        # Cumulative List of Rewards
        
        self.running_objective_reward[AgentStats.iterator + 1] = (self.running_objective_reward[AgentStats.iterator] + new_objective_reward)        

        # List of agreements to turn into agreement rate

        self.agreements[AgentStats.iterator] = new_agreed

        # Record of Hunt Choice History

        self.hunt_choice_history[AgentStats.iterator] = new_choice

        # Tally of each time a partner is chosen irrespective of time

        if self.partner_choice_tally.get(newest_partner):
            self.partner_choice_tally[newest_partner] += 1
        else:
            self.partner_choice_tally[newest_partner] = 1

        # Record of Partner Choice History

        if not self.partner_choice_history.get(newest_partner):
            self.partner_choice_history[newest_partner] = [0] * (AgentStats.iterator + 1)


        for x in self.partner_choice_history:
            if x != newest_partner:
                self.partner_choice_history[x] += [self.partner_choice_history[x][-1]]
            else:
                self.partner_choice_history[x] += [self.partner_choice_history[x][-1] + 1]
