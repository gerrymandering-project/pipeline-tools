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
        graph.nodes[node]["pos"] = [graph.node[node]['C_X'], graph.node[node]['C_Y'] ]
    deg_one_nodes = []
    for v in graph:
        if graph.degree(v) == 1:
            deg_one_nodes.append(v)
    for node in deg_one_nodes:
        graph.remove_node(node)
    return graph

def build_partition_meta(graph, mean):
    assignment = {}
    for y in graph.node():
        if graph.node[y]['C_Y'] < mean:
            assignment[y] = -1
        else:
            assignment[y] = 1
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    partition = Partition(graph, assignment=assignment, updaters=updaters)
    return partition

def special_faces(graph, k):
    special_faces = []
    for node in graph.nodes():
        if graph.node[node]['distance'] >= k:
            special_faces.append(node)
    return special_faces



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
    g.nodes[node]["pos"] = [g.node[node]["C_X"], g.node[node]["C_Y"]]
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

plt.figure()
nx.draw(g_sierpinski, pos=nx.get_node_attributes(g_sierpinski, 'pos'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'))
plt.savefig("./plots/Serpinsky_mesh.eps", format='eps')
plt.close()

for edge in g_sierpinski.edges():
    g_sierpinski[edge[0]][edge[1]]['cut_times'] = 0

    for n in g_sierpinski.nodes():
        g_sierpinski.node[n]["population"] = 1 #This is something gerrychain will refer to for checking population balance
        g_sierpinski.node[n]["last_flipped"] = 0
        g_sierpinski.node[n]["num_flips"] = 0
pop1 = .05

base = 1          
popbound = within_percent_of_ideal_population(partition_y, pop1)
ideal_population = sum(partition_y["population"].values()) / len(partition_y)

tree_proposal = partial(recom,pop_col="population",pop_target=ideal_population,epsilon=0.05,node_repeats=1)
steps = 1000
exp_chain = MarkovChain(tree_proposal, Validator([single_flip_contiguous, popbound]), accept=True, initial_state=partition_y,
                                        total_steps=steps)
z = 0
num_cuts_list = []
for part in exp_chain:
    z += 1
    print("step ", z)

    for edge in part["cut_edges"]:
        g_sierpinski[edge[0]][edge[1]]["cut_times"] += 1

plt.figure()
nx.draw(g_sierpinski, pos={x: x for x in g_sierpinski.nodes()}, node_color=[0 for x in g_sierpinski.nodes()], node_size=1,
                    edge_color=[g_sierpinski[edge[0]][edge[1]]["cut_times"] for edge in g_sierpinski.edges()], node_shape='s',
                    cmap='magma', width=3)
plt.savefig("./plots/edges.eps", format='eps')
plt.close()