## This script will perform a markov chain on the  subset faces of the north carolina graph, picking a face UAR and sierpinskifying or de-sierpinskifying it, then running gerrychain on the graph, recording central seat tendencies.
from facefinder import *
import random
import statistics
import math
import gerrychain
import networkx
from functools import partial
from gerrychain.tree import bipartition_tree as bpt
from gerrychain import Graph, MarkovChain
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population, UpperBound)
from gerrychain.proposals import propose_random_flip, propose_chunk_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain import GeographicPartition
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap
from gerrychain.tree import recursive_tree_part, bipartition_tree_random, PopulatedGraph, contract_leaves_until_balanced_or_none, find_balanced_edge_cuts
#gerrychain parameters
#num districts
k = 12
epsilon = .05
updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        }
graph, dual = preprocessing("jsons/NC.json")
ideal_population= sum( graph.nodes[x]["population"] for x in graph.nodes())/k
faces = graph.graph["faces"]
faces = list(faces)
#random.choice(faces) will return a random face
#TODO: run gerrychain on graph 
totpop = 0
for node in graph.nodes():
    totpop += int(graph.nodes[node]['population'])
# length of chain 
steps = 30000
#beta thereshold, how many steps to hold beta at 0
beta_threshold = 10000
#length of each gerrychain step 
gerrychain_steps = 250
#faces that are currently sierp
special_faces = []
chain_output = { 'dem_seat_data': [], 'rep_seat_data':[], 'score':[] }
#start with small score to move in right direction
chain_output['score'].append(1/ 1100000)
z = 0
for i in range(steps):
    if z % 100 == 0:
        z += 1
        print("step ", z)
    face = random.choice(faces)

    if random.random() > .5:
        if not face in special_faces:
            special_faces.append(face)
        else:
            special_faces.remove(face)
    step_graph = face_sierpinski_mesh(graph, special_faces)

    #need to create initial partition
    cddict = recursive_tree_part(step_graph,range(k),totpop/k,"population", .01,1)
    
    for node in graph.nodes():
        graph.nodes[node]['part'] = cddict[node]
    
    initial_partition = Partition(step_graph, assignment='part', updaters=updaters)
    popbound = within_percent_of_ideal_population(initial_partition, epsilon)
    tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=epsilon,
                                node_repeats=1, method=my_mst_bipartition_tree_random)
    #make new function
    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=always_true, initial_state=initial_partition, total_steps=gerrychain_steps)
    seats_won_for_republicans = []
    seats_won_for_democrats = []
    for part in exp_chain:
        rep_seats_won = 0
        dem_seats_won = 0
        for i in range(k):
            rep_votes = 0
            dem_votes = 0
            for n in graph.nodes():
                if part.assignment[n] == i:
                    rep_votes += graph.nodes[n]["EL16G_PR_R"]
                    dem_votes += graph.nodes[n]["EL16G_PR_D"]
            total_seats_dem = int(dem_votes > rep_votes)
            total_seats_rep = int(rep_votes > dem_votes)
            rep_seats_won += total_seats_rep
            dem_seats_won += total_seats_dem
        seats_won_for_republicans.append(rep_seats_won)
        seats_won_for_democrats.append(dem_seats_won)
    #check if sign is wrong, unsure 
    #rand < min(1, P(x')/P(x))
    #
    if random.random() < min(1, math.exp(statistics.mean(seats_won_for_republicans)) / chain_output['score'][z - 1]):
        chain_output['dem_seat_data'].append(seats_won_for_democrats)
        chain_output['rep_seat_data'].append(seats_won_for_republicans)
        chain_output['score'].append(math.exp(statistics.mean(seats_won_for_republicans)))
    else:
        chain_output['dem_seat_data'].append(chain_output['dem_seat_data'][z-1])
        chain_output['rep_seat_data'].append(chain_output['rep_seat_data'][z-1])
        chain_output['score'].append(chain_output['score'][z-1])

        


