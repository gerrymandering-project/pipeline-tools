import os
import random
import json
import geopandas as gpd
import functools
import datetime
import matplotlib
from facefinder import *
from graph_tools import *

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


<<<<<<< HEAD:metandering_experiments.py
=======
   

def face_serpinsky_mesh(graph, special_faces):
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
            if v+1 < len(neighbor_list):
                distance = np.array(graph.nodes[neighbor_list[v]]["pos"]) + np.array(graph.nodes[neighbor_list[v+1]]["pos"])
                distance = distance * .5
                label = max_label + 1
                max_label += 1
            else:
                distance = np.array(graph.nodes[neighbor_list[v]]["pos"]) + np.array(graph.nodes[neighbor_list[0]]["pos"])
                distance = distance * .5
                label = max_label + 10
                max_label += 10
            graph.add_node(label)
            graph.nodes[label]['pos'] = distance
            connections.append(label)
        for v in range(0,len(connections)):
            if v+1 < len(connections):
                graph.add_edge(connections[v],connections[v+1])
            else:
                graph.add_edge(connections[v],connections[0])
        graph.remove_node(face)
    return graph
>>>>>>> 59cec8e1f483220b0600a6075e4b8fee9acc9ac5:metamandering_experiments.py

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
# Experiement setup
link = "https://people.csail.mit.edu/ddeford//COUNTY/COUNTY_13.json"
g = graph_from_url_processing(link)

dual = restricted_planar_dual(g)

plt.figure()
nx.draw(g, pos=nx.get_node_attributes(g, 'pos'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'))
plt.savefig("./plots/UnderlyingGraph.eps", format='eps')
plt.close()

# Quick vertical partition, use for initial partition

vertical = []
for node in g.nodes():
    g.nodes[node]["pos"] = [g.nodes[node]["C_X"], g.nodes[node]["C_Y"]]
    vertical.append(g.nodes[node]["C_Y"])
mean_y_coord = sum(vertical) / len(vertical)

partition_y = build_partition_meta(g,mean_y_coord)



crosses = compute_cross_edge(g, partition_y)

dual_crosses = []
for edge in dual.edges:
    if dual.edges[edge]["original_name"] in crosses:
        dual_crosses.append(edge)

dual = distance_from_partition(dual, dual_crosses)

special_faces = special_faces(dual,2)
g_sierpinski = face_sierpinski_mesh(g, special_faces)


for node in g_serpinsky:
    g_serpinsky.nodes[node]['C_X'] = g_serpinsky.nodes[node]['pos'][0]
    g_serpinsky.nodes[node]['C_Y'] = g_serpinsky.nodes[node]['pos'][1]
    if 'population' not in g_serpinsky.nodes[node]:
        g_serpinsky.nodes[node]['population'] = 0

print("creating partition")

total_pop = sum( [ g_serpinsky.nodes[node]['population'] for node in g_serpinsky])

serp_partition = build_trivial_partition(g_serpinsky)

print("created partition")

plt.figure()
nx.draw(g_sierpinski, pos=nx.get_node_attributes(g_sierpinski, 'pos'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'))
plt.savefig("./plots/Serpinsky_mesh.eps", format='eps')
plt.close()

for edge in g_sierpinski.edges():
    g_sierpinski[edge[0]][edge[1]]['cut_times'] = 0

<<<<<<< HEAD:metandering_experiments.py
    for n in g_sierpinski.nodes():
        g_sierpinski.node[n]["population"] = 1 #This is something gerrychain will refer to for checking population balance
        g_sierpinski.node[n]["last_flipped"] = 0
        g_sierpinski.node[n]["num_flips"] = 0
=======
    for n in g_serpinsky.nodes():
        g_serpinsky.nodes[n]["population"] = 1 #This is something gerrychain will refer to for checking population balance
        g_serpinsky.nodes[n]["last_flipped"] = 0
        g_serpinsky.nodes[n]["num_flips"] = 0
>>>>>>> 59cec8e1f483220b0600a6075e4b8fee9acc9ac5:metamandering_experiments.py
pop1 = .05

base = 1          



popbound = within_percent_of_ideal_population(partition_y, pop1)
ideal_population = sum(serp_partition["population"].values()) / len(partition_y)

tree_proposal = partial(recom,pop_col="population",pop_target=ideal_population,epsilon=0.05,node_repeats=1)
steps = 1000

exp_chain = MarkovChain(tree_proposal, Validator([single_flip_contiguous]), accept=True, initial_state=serp_partition,
                                        total_steps=steps)
z = 0
num_cuts_list = []
for part in exp_chain:
    z += 1
    print("step ", z)

    for edge in part["cut_edges"]:
<<<<<<< HEAD:metandering_experiments.py
        g_sierpinski[edge[0]][edge[1]]["cut_times"] += 1
=======
        g_serpinsky[edge[0]][edge[1]]["cut_times"] += 1
    print("finished round")
>>>>>>> 59cec8e1f483220b0600a6075e4b8fee9acc9ac5:metamandering_experiments.py

plt.figure()
nx.draw(g_sierpinski, pos={x: x for x in g_sierpinski.nodes()}, node_color=[0 for x in g_sierpinski.nodes()], node_size=1,
                    edge_color=[g_sierpinski[edge[0]][edge[1]]["cut_times"] for edge in g_sierpinski.edges()], node_shape='s',
                    cmap='magma', width=3)
plt.savefig("./plots/edges.eps", format='eps')
plt.close()
