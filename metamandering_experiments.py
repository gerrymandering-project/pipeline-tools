import os
import random
import json
import geopandas as gpd
import functools
import datetime
import matplotlib
from facefinder import *
from graph_tools import *

import graph_tools

import matplotlib.pyplot as plt
import numpy as np
import csv
from networkx.readwrite import json_graph
import math
import seaborn as sns
from functools import partial
import networkx as nx
import numpy as np

import seannas_code

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

# functions below are tools needed for metamandering experiment
def graph_from_url_processing(link):
    r = requests.get(url=link)
    data = json.loads(r.content)
    g = json_graph.adjacency_graph(data)
    graph = Graph(g)
    graph.issue_warnings()
    for node in graph.nodes():
        graph.nodes[node]["pos"] = [graph.nodes[node]['C_X'], graph.nodes[node]['C_Y'] ]
    deg_one_nodes = []
    for v in graph:
        if graph.degree(v) == 1:
            deg_one_nodes.append(v)
    for node in deg_one_nodes:
        graph.remove_node(node)
    return graph

def build_trivial_partition(graph):
    assignment = {}
    for y in graph.nodes():
        assignment[y] = 1
    first_node = list(graph.nodes())[0]
    assignment[first_node] = -1
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    partition = Partition(graph, assignment=assignment, updaters=updaters)
    return partition

def build_balanced_partition(graph, pop_col, pop_target, epsilon):
    
    block = my_mst_bipartition_tree_random(graph, pop_col, pop_target, epsilon)
    assignment = {}
    for y in graph.nodes():
        if y in block:
            assignment[y] = 1
        else:
            assignment[y] = -1
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    partition = Partition(graph, assignment=assignment, updaters=updaters)
    return partition


def build_balanced_k_partition(graph, k, pop_col, pop_target, epsilon):
    
    assignment = recursive_tree_part(graph, k, pop_target, pop_col, epsilon)
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    partition = Partition(graph, assignment=assignment, updaters=updaters)
    return partition
    
    

def build_partition_meta(graph, mean):
    assignment = {}
    for y in graph.nodes():
        if graph.nodes[y]['C_Y'] < mean:
            assignment[y] = -1
        else:
            assignment[y] = 1
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    partition = Partition(graph, assignment=assignment, updaters=updaters)
    print("cut edges are", partition["cut_edges"])
    return partition

def special_faces(graph, k):
    special_faces = []
    for node in graph.nodes():
        if graph.nodes[node]['distance'] >= k:
            special_faces.append(node)
    return special_faces


def remove_undirected_edge(graph, v, u):
    if (v,u) in graph.edges():
        graph.remove_edge(v,u)
        return 
    
    if (u,v) in graph.edges():
        graph.remove_edge(u,v)
        return 
    
    #print("nodes ", v, ",", u, " not connected in graph")


def face_sierpinski_mesh(graph, special_faces):
    #parameters: 
    #graph: graph object that edges will be added to
    #special_faces: list of faces that we want to add node/edges to
    #k: integer depth parameter for depth of face refinement
    max_label = max(list(graph.nodes()))
    for face in special_faces:
        graph.add_node(face)
        neighbor_list = []
        locations = []
        connections = []
        location = np.array([0,0]).astype("float64")
        for v in face:
            neighbor_list.append(v)
            location += np.array(graph.nodes[v]["pos"]).astype("float64")
        graph.nodes[face]["pos"] = location / len(face)
        for w in face:
            locations.append(graph.nodes[w]["pos"] - graph.nodes[face]["pos"])
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)
        for v in range(0,len(neighbor_list)):
            next_index = (v+1) % len(neighbor_list)
            distance = np.array(graph.nodes[neighbor_list[v]]["pos"]) + np.array(graph.nodes[neighbor_list[next_index]]["pos"])
            distance = distance * .5
            label = max_label + 1
            max_label += 1
            graph.add_node(label)
            graph.nodes[label]['pos'] = distance
            remove_undirected_edge(graph, neighbor_list[v], neighbor_list[next_index])
            graph.add_edge(neighbor_list[v],label)
            graph.add_edge(label,neighbor_list[next_index])
            connections.append(label)
        for v in range(0,len(connections)):
            if v+1 < len(connections):
                graph.add_edge(connections[v],connections[v+1])
            else:
                graph.add_edge(connections[v],connections[0])
        graph.remove_node(face)
    return graph

def cut_accept(partition):
    bound = 1
    if partition.parent is not None:
        bound = (partition["base"] ** (-len(partition["cut_edges"]) + len(
            partition.parent["cut_edges"])))  # *(len(boundaries1)/len(boundaries2))
    return random.random() < bound

def step_num(partition):
                parent = partition.parent

                if not parent:
                    return 0

                return parent["step_num"] + 1
            
            
def get_spanning_tree_mst(graph):
    for edge in graph.edges:
        graph.edges[edge]["weight"] = random.random()

    spanning_tree = nx.tree.maximum_spanning_tree(
        graph, algorithm="kruskal", weight="weight"
    )
    return spanning_tree


def my_mst_bipartition_tree_random(
    graph,
    pop_col,
    pop_target,
    epsilon,
    node_repeats=1,
    spanning_tree=None,
    choice=random.choice):
    populations = {node: graph.nodes[node][pop_col] for node in graph}

    possible_cuts = []
    if spanning_tree is None:
        spanning_tree = get_spanning_tree_mst(graph)

    while len(possible_cuts) == 0:
        spanning_tree = get_spanning_tree_mst(graph)
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = find_balanced_edge_cuts(h, choice=choice)

    return choice(possible_cuts).subset


def always_true(proposal):
    return True
           
# Experiement setup

def smooth_node(graph, v):
    #print(v)
    neighbors = list(graph.neighbors(v))
    graph.remove_node(v)
    try:
        graph.add_edge(neighbors[0], neighbors[1])
    except:
        print(neighbors)
    
    return graph

def preprocessing():
    link = "https://people.csail.mit.edu/ddeford/COUSUB/COUSUB_13.json"
    
    link = "https://people.csail.mit.edu/ddeford/COUSUB/COUSUB_55.json"
    g = graph_from_url_processing(link)
    
    
    #Have to remove bad nodes in order for the duality thing to work properly
    
    bad_nodes = []
    for v in g.nodes():
        if g.degree(v) == 1:
            bad_nodes.append(v)
    print(bad_nodes)
    g.remove_nodes_from(bad_nodes)
    
    deg_2_nodes = []
    for v in g.nodes():
        if g.degree(v) == 2:
            deg_2_nodes.append(v)
            
    print(deg_2_nodes)
    for v in deg_2_nodes:
        g = smooth_node(g, v)    
        #Some weird bug here I don't understand
        
    
    print("making dual")
    dual = restricted_planar_dual(g)
    print("made dual")
    plt.figure()
    nx.draw(g, pos=nx.get_node_attributes(g, 'pos'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'))
    plt.savefig("./plots/UnderlyingGraph.png", format='png')
    plt.close()
    
    
    for node in g.nodes():
        g.nodes[node]["pos"] = [g.nodes[node]["C_X"], g.nodes[node]["C_Y"]]
        g.nodes[node]["population"] = g.nodes[node]["POP10"]


    return g, dual

g, dual = preprocessing()
# Quick vertical partition, use for initial partition


k = 4
##Number of Partitions Goes Here

print("making initial partition")
ideal_population = sum( g.nodes[x]["population"] for x in g.nodes())/k
partition_y = build_balanced_k_partition(g, list(range(k)), "population", ideal_population, .05)

plt.figure()
viz(g_sierpinsky, set([]), sierp_partition.parts)
plt.savefig("./plots/target_map.png", format = 'png')
plt.close()

print("made partition")
crosses = compute_cross_edge(g, partition_y)

dual_crosses = []
for edge in dual.edges:
    if dual.edges[edge]["original_name"] in crosses:
        dual_crosses.append(edge)
        
print("making dual distances")
dual = distance_from_partition(dual, dual_crosses)
print('finished making dual distances')
special_faces = special_faces(dual,2)
print('finished assigning special faces')
g_sierpinsky = face_sierpinski_mesh(g, special_faces)
print("made metamander")

for node in g_sierpinsky:
    g_sierpinsky.nodes[node]['C_X'] = g_sierpinsky.nodes[node]['pos'][0]
    g_sierpinsky.nodes[node]['C_Y'] = g_sierpinsky.nodes[node]['pos'][1]
    if 'population' not in g_sierpinsky.nodes[node]:
        g_sierpinsky.nodes[node]['population'] = 0

total_pop = sum( [ g_sierpinsky.nodes[node]['population'] for node in g_sierpinsky])

#sierp_partition = build_trivial_partition(g_sierpinsky)

plt.figure()
nx.draw(g_sierpinsky, pos=nx.get_node_attributes(g_sierpinsky, 'pos'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'))
plt.savefig("./plots/sierpinsky_mesh.eps", format='eps')
plt.close()

for edge in g_sierpinsky.edges():
    g_sierpinsky[edge[0]][edge[1]]['cut_times'] = 0

    for n in g_sierpinsky.nodes():
        g_sierpinsky.nodes[n]["population"] = 1 #This is something gerrychain will refer to for checking population balance
        g_sierpinsky.nodes[n]["last_flipped"] = 0
        g_sierpinsky.nodes[n]["num_flips"] = 0

#sierp_partition = build_balanced_partition(g_sierpinsky, "population", ideal_population, .01)



ideal_population= sum( g_sierpinsky.nodes[x]["population"] for x in g_sierpinsky.nodes())/k
sierp_partition = build_balanced_k_partition(g_sierpinsky, list(range(k)), "population", ideal_population, .05)
#viz(g_sierpinsky, set([]), sierp_partition.parts)
pop1 = .1


popbound = within_percent_of_ideal_population(sierp_partition, pop1)
#ideal_population = sum(sierp_partition["population"].values()) / len(sierp_partition)
print(ideal_population)

tree_proposal = partial(recom,pop_col="population",pop_target=ideal_population,epsilon= 1 ,node_repeats=1)
steps = 200


chaintype = "tree"

if chaintype == "tree":
    tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                            node_repeats=1, method=my_mst_bipartition_tree_random)

if chaintype == "uniform_tree":
    tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                            node_repeats=1, method=my_uu_bipartition_tree_random)



exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=always_true, initial_state=sierp_partition, total_steps=steps)


z = 0
num_cuts_list = []


for part in exp_chain:

#for i in range(steps):
#    part = build_balanced_partition(g_sierpinsky, "population", ideal_population, .05)


    z += 1
    print("step ", z)

    for edge in part["cut_edges"]:
        g_sierpinsky[edge[0]][edge[1]]["cut_times"] += 1
    #print("finished round")


edge_colors = [g_sierpinsky[edge[0]][edge[1]]["cut_times"] for edge in g_sierpinsky.edges()]

pos=nx.get_node_attributes(g_sierpinsky, 'pos')

plt.figure()
nx.draw(g_sierpinsky, pos=nx.get_node_attributes(g_sierpinsky, 'pos'), node_size=1,
                    edge_color=edge_colors, node_shape='s',
                    cmap='magma', width=3)
plt.savefig("./plots/edges.png")
plt.close()
