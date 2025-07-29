from Hunt import Hunt
from Graph import Graph
import matplotlib.pyplot as plt
from os import path, makedirs
from Prey import *
from numpy import zeros


# Available High-Risk Prey types

TYPES = (Stag, Bison, Random)

# Examined Reputation Value Multipliers

gossips = [0, 1, 2, 5, 10, 100, "Inf."]


total_rewards = {}
avg_maker = {}
graphs = {}
hunts = {}
pair_num = 4
choice_num = 2
sim_count = 10
partner_table = {}
averages_tables = {}

# Initialise Files Containing Average Partner Choice Tables 

for t in TYPES:
    with open(f"{t.__name__}partner_tables.txt", "w") as f:
        f.write("")

# Initialise Hunt Total Reward Data 

for t in TYPES:
    total_rewards[t] = []

############################
# EXPERIMENT AND OUTPUT LOOP
############################

for g in gossips:

    # Initialise Data Storage for Appending

    for t in TYPES:
        avg_maker[t] = []
        graphs[t] = []
        hunts[t] = []
        partner_table[t] = []


##################
# HUNT SIMULATIONS
##################

    for _ in range(sim_count):
        for t in TYPES:
            hunts[t] = Hunt(t, pair_num, choice_num, g)
            avg_maker[t].append(hunts[t].simulate())
            partner_table[t].append([x.partner_choice_tally for x in [y.data for y in hunts[t].agents]])


####################################
# DATA COLLATION FOR FULL EXPERIMENT
####################################

    # Collate Data from Current Round of Hunts

    for t in TYPES: 

        total_rewards[t].append(sum(avg_maker[t])/sim_count)
        graphs[t].append(Graph([a.data for a in hunts[t].agents], t))


    # Create Tables Averaging Each Agent's Matchup Tally

    for t in TYPES:
        averages_tables[t] = zeros((pair_num * 2, pair_num * 2), int)
        for a in range (pair_num * 2):
            for b in range (pair_num * 2):
                if a !=b :
                    averages_tables[t][a,b] = sum([partner_table[t][s][a][b] for s in range(sim_count)])/sim_count


    # Output Tables Averaging Each Agent's Matchup Tally

    for t in TYPES:
        buffer = "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        for x in range(pair_num * 2):              
            buffer += str(f"{t.__name__} Hunt with Gossip Level {g}: Agent {x}'s Average Partner Frequency:\n")
            for y in range(len(averages_tables[t][x])):
                if y != x:
                    buffer += str(f"{y}\t|\t{averages_tables[t][x][y]}\n")
                else:
                    pass
            buffer += "\n"
        with open(f"{t.__name__}_partner_tables.txt", "a") as f:
            f.write(buffer)

##############
# AGENT GRAPHS
##############

    ("Exporting Graphs...")

    for t in TYPES:

        for h in graphs[t]:
            h.average_reward_graph()
            h.cumulative_reward_graph(-1)
            h.hunt_choice_history_graph(-1)
            h.agreement_graph(-1)
            if g != 0:
                h.rot_graph(hunts[t].rep_over_time)
            for i in range(pair_num * 2):
                h.cumulative_partner_graph(i)
                h.cumulative_reward_graph(i)
                h.hunt_choice_history_graph(i)
                h.agreement_graph(i)
    Graph.iter_count += 1

    print("Export Complete!")
    print()

###################################
# GOSSIP CHART COLLATION AND OUTPUT
###################################


plt.rcParams.update({'font.size': 15})

for hunt_data in total_rewards.items():

    plt.figure()
    plt.bar([str(x) for x in gossips], [int(round(x)) for x in hunt_data[1]])
    match hunt_data[0].__name__:
        case "Bison":
            plt.bar("Maximum", 830565)
        case _:
            plt.bar("Maximum", 1400000)
    plt.title(f"{hunt_data[0].__name__} Hunt: Gossip Value to Total Reward correlation")
    plt.xlabel('Gossip Value')
    plt.ylabel('Total Reward')

    for z in range(len(gossips)):
        plt.text(z, int(round(hunt_data[1][z])), int(round(hunt_data[1][z])), ha = "center", fontsize=11)
    match hunt_data[0].__name__:
        case "Bison":
            plt.text(7, 830565, 830565, ha = "center", fontsize=11)
        case _:
            plt.text(7, 1400000, 1400000, ha = "center", fontsize=11)
    makedirs(path.dirname(f"./{hunt_data[0].__name__}_gossip_chart/"), exist_ok=True)
    plt.savefig(f"./{hunt_data[0].__name__}_gossip_chart/{hunt_data[0].__name__}_total_reward_chart.png")