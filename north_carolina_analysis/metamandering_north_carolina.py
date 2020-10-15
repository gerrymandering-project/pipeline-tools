import os
import random
import json
import geopandas as gpd
import functools
import datetime
import matplotlib
from facefinder import *
import time
import requests
import zipfile
import io

import matplotlib.pyplot as plt
import numpy as np
import csv
from networkx.readwrite import json_graph
import math
import seaborn as sns
from functools import partial
import networkx as nx
import numpy as np
import copy

from gerrychain.tree import bipartition_tree as bpt
from gerrychain import Graph
from gerrychain import MarkovChain
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



def step_num(partition):
                parent = partition.parent

                if not parent:
                    return 0

                return parent["step_num"] + 1
def always_true(proposal):
    return True

def produce_gerrymanders(graph, k, tag, sample_size, chaintype):
    #Samples k partitions of the graph
    #stores vote histograms, and returns most extreme partitions.
    for n in graph.nodes():
        graph.nodes[n]["last_flipped"] = 0
        graph.nodes[n]["num_flips"] = 0
      
    ideal_population= sum( graph.nodes[x]["population"] for x in graph.nodes())/k
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    initial_partition = Partition(graph, assignment='part', updaters=updaters)
    pop1 = .05
    popbound = within_percent_of_ideal_population(initial_partition, pop1)
    
    if chaintype == "tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=my_mst_bipartition_tree_random)
    
    elif chaintype == "uniform_tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=my_uu_bipartition_tree_random)
    else: 
        print("Chaintype used: ", chaintype)
        raise RuntimeError("Chaintype not recongized. Use 'tree' or 'uniform_tree' instead")
    
    
    
    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=always_true, initial_state=initial_partition, total_steps=sample_size)
        
    seats_won_table = []
    best_left = np.inf
    best_right = -np.inf
    ctr = 0
    for part in exp_chain:
        ctr += 1
        seats_won = 0

        if ctr % 100 == 0:
            print("step ", ctr)
        for i in range(k):
            rep_votes = 0
            dem_votes = 0
            for n in graph.nodes():
                if part.assignment[n] == i:
                    rep_votes += graph.nodes[n]["EL16G_PR_R"]
                    dem_votes += graph.nodes[n]["EL16G_PR_D"]
            total_seats = int(rep_votes > dem_votes)
            seats_won += total_seats
        #total seats won by rep
        seats_won_table.append(seats_won)
        # save gerrymandered partitionss
        if seats_won < best_left:
            best_left = seats_won
            left_mander = copy.deepcopy(part.parts)
        if seats_won > best_right:
            best_right = seats_won
            right_mander = copy.deepcopy(part.parts)
        #print("finished round"
    
    print("max", best_right, "min:", best_left)
    
    #plt.figure()
    #plt.hist(seats_won_table, bins = 10)
    
    name = "./plots/seats_histogram" + tag +".png"
    #plt.savefig(name)
    #plt.close() 
    sns_plot = sns.distplot(seats_won_table, label="North Carolina Republican Vote Distribution").get_figure()
    plt.legend()
    sns_plot.savefig(name)
    return left_mander, right_mander

def assign_special_faces(graph, k):
    special_faces = []
    for node in graph.nodes():
        if graph.nodes[node]['distance'] >= k:
            special_faces.append(node)
    return special_faces

def metamander_around_partition(graph, dual, target_partition, tag,num_dist):

    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    
    assignment = {}
    for x in graph.nodes():
        color = 0
        for block in target_partition.keys():
            if x in target_partition[block]:
                assignment[x] = color
            color += 1
    
    target_partition = Partition(graph, assignment, updaters = updaters)
    plt.figure()
    
    viz(graph, set([]), target_partition.parts)
    plt.savefig("./plots/target_map" + tag + ".png", format = 'png')
    plt.close()
    
    print("made partition")
    crosses = compute_cross_edge(graph, target_partition)
    
    k = len(target_partition.parts)
    
    dual_crosses = []
    for edge in dual.edges:
        if dual.edges[edge]["original_name"] in crosses:
            dual_crosses.append(edge)
            
    print("making dual distances")
    dual = distance_from_partition(dual, dual_crosses)
    print('finished making dual distances')
    special_faces = assign_special_faces(dual,2)
    print('finished assigning special faces')
    g_sierpinsky = face_sierpinski_mesh(graph, special_faces)
    print("made metamander")
    # change from RVAP and UVAP to approprate election data columns 
    for node in g_sierpinsky:
        g_sierpinsky.nodes[node]['C_X'] = g_sierpinsky.nodes[node]['pos'][0]
        g_sierpinsky.nodes[node]['C_Y'] = g_sierpinsky.nodes[node]['pos'][1]
        if 'population' not in g_sierpinsky.nodes[node]:
            g_sierpinsky.nodes[node]['population'] = 0
        if 'EL16G_PR_D' not in g_sierpinsky.nodes[node]:
            g_sierpinsky.nodes[node]['EL16G_PR_D'] = 0
        if 'EL16G_PR_R' not in g_sierpinsky.nodes[node]:
            g_sierpinsky.nodes[node]['EL16G_PR_R'] = 0
        ##Need to add the voting data
    print("assigning districts to metamander")
    total_pop = sum( [ g_sierpinsky.nodes[node]['population'] for node in g_sierpinsky])
    cddict = recursive_tree_part(graph,range(num_dist),total_pop/num_dist,"population", .01,1)
    for node in graph.nodes():
        graph.nodes[node]['part'] = cddict[node]
    #sierp_partition = build_trivial_partition(g_sierpinsky)
    print("assigned districts")
    plt.figure()
    nx.draw(g_sierpinsky, pos=nx.get_node_attributes(g_sierpinsky, 'pos'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'))
    plt.title("North Carolina Metamander")
    plt.savefig("./plots/sierpinsky_mesh.png", format='png')
    plt.close()
    return g_sierpinsky, k
    

def produce_sample(graph, k, tag, sample_size = 500, chaintype='tree'):
    #Samples k partitions of the graph, stores the cut edges and records them graphically
    #Also stores vote histograms, and returns most extreme partitions.
    print("producing sample")
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['cut_times'] = 0
    
    for n in graph.nodes():
        #graph.nodes[n]["population"] = 1 #graph.nodes[n]["POP10"] #This is something gerrychain will refer to for checking population balance
        graph.nodes[n]["last_flipped"] = 0
        graph.nodes[n]["num_flips"] = 0
    print("set up chain")
    ideal_population= sum( graph.nodes[x]["population"] for x in graph.nodes())/k
    initial_partition = Partition(graph, assignment='part', updaters=updaters)
    pop1 = .05
    print("popbound")
    popbound = within_percent_of_ideal_population(initial_partition, pop1)
    
    if chaintype == "tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=my_mst_bipartition_tree_random)
    
    elif chaintype == "uniform_tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=my_uu_bipartition_tree_random)
    else: 
        print("Chaintype used: ", chaintype)
        raise RuntimeError("Chaintype not recongized. Use 'tree' or 'uniform_tree' instead")
    
    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=always_true, initial_state=initial_partition, total_steps=sample_size)
    
    
    z = 0
    num_cuts_list = []
    seats_won_table = []
    best_left = np.inf
    best_right = -np.inf
    print("begin chain")
    for part in exp_chain:

        
        z += 1
        if z % 100 == 0:
            print("step ", z)
        seats_won = 0
        for edge in part["cut_edges"]:
            graph[edge[0]][edge[1]]["cut_times"] += 1
        for i in range(k):
            rep_votes = 0
            dem_votes = 0
            for n in graph.nodes():
                if part.assignment[n] == i:
                    rep_votes += graph.nodes[n]["EL16G_PR_R"]
                    dem_votes += graph.nodes[n]["EL16G_PR_D"]
            total_seats = int(rep_votes > dem_votes)
            seats_won += total_seats)
        #total seats won by rep
        seats_won_table.append(seats_won)
        # save gerrymandered partitionss
        if seats_won < best_left:
            best_left = seats_won
            #left_mander = copy.deepcopy(part.parts)
        if seats_won > best_right:
            best_right = seats_won
            #right_mander = copy.deepcopy(part.parts)
        #print("finished round"
    
    print("max", best_right, "min:", best_left)
    
    plt.figure()
    plt.hist(seats_won_table, bins = 10)
    
    name = "./plots/seats_histogram_metamander" + tag +".png"
    plt.savefig(name)
    plt.close()    
        
    edge_colors = [graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()]
    
    pos=nx.get_node_attributes(graph, 'pos')
    
    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=1,
                        edge_color=edge_colors, node_shape='s',
                        cmap='magma', width=3)
    plt.savefig("./plots/edges" + tag + ".png")
    plt.close()

    return 


def main():
    graph, dual = preprocessing("jsons/NC.json")
    left_mander, right_mander = produce_gerrymanders(graph,12,'_nc',100,'tree')
    hold_graph = copy.deepcopy(graph)
    hold_dual = copy.deepcopy(dual)
    num_dist = 13
    metamander , k = metamander_around_partition(graph, dual, left_mander, '_nc' + "LEFTMANDER",num_dist)

    produce_sample(metamander, k , '_nc')




main()
