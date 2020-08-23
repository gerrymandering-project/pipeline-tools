import facefinder
#from secret_matamandering import *

import matplotlib.pyplot as plt

from functools import partial
import networkx as nx
import numpy as np
import copy
import random
import math
import json
import sys

from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain import accept
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population, UpperBound)

from gerrychain.updaters import Election, Tally, cut_edges

from gerrychain.partition import Partition
from gerrychain.proposals import recom

from gerrychain.tree import recursive_tree_part


def face_sierpinski_mesh(graph, special_faces):
    """'Sierpinskifies' certain faces of the graph by adding nodes and edges to
    certain faces.

    Args:
        graph (Gerrychain Graph): graph object that edges will be added to
        special_faces (list): list of faces that we want to add node/edges to

    Raises:
        RuntimeError if SIERPINSKI_POP_STYLE of config file is neither 'uniform'
        nor 'zero'

    TODO:
        Add parameter for depth of sierpinskification
    """

    # Get maximum node label.
    label = max(list(graph.nodes()))

    for face in special_faces:
        neighbors = [] #  Neighbors of face
        locationCount = np.array([0,0]).astype("float64")
        # For each face, add to neighbor_list and add to location count
        for vertex in face:
            neighbors.append(vertex)
            locationCount += np.array(graph.nodes[vertex]["pos"]).astype("float64")
        # Save the average of each of the face's positions
        facePosition = locationCount / len(face)

        # In order, store relative position of each vertex to the position of the face
        locations = [graph.nodes[vertex]['pos'] - facePosition for vertex in face]
        # Sort neighbors according to each node's angle with the center of the face
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbors.sort(key=dict(zip(neighbors, angles)).get)

        connections = [] # List of all new nodes added
        # For each consecutive pair of nodes, remove their edge, create a new
        # node at their average position, and connect edge node to the new node:
        for vertex, next_vertex in zip(neighbors, neighbors[1:] + [neighbors[0]]):
            label += 1
            # Add new node to graph with corresponding label at the average position
            # of vertex and next_vertex
            graph.add_node(label)
            avgPos = (np.array(graph.nodes[vertex]['pos']) +
                      np.array(graph.nodes[next_vertex]['pos'])) / 2
            graph.nodes[label]['pos'] = avgPos
            graph.nodes[label][config['X_POSITION']] = avgPos[0]
            graph.nodes[label][config['Y_POSITION']] = avgPos[1]

            # For each new node, 'move' a third of the population, Party A votes,
            # and Party B votes from its two adjacent nodes which previously exists
            # to itself (so that each previously existing node equally shares
            # its statistics with the two new nodes adjacent to it)
            if config['SIERPINSKI_POP_STYLE'] == 'uniform':
                for vert in [vertex, next_vertex]:
                    # Save original values if not done already
                    if 'orig_pop' not in graph.nodes[vert]:
                        graph.nodes[vert]['orig_pop'] = graph.nodes[vert][config['POP_COL']]
                        graph.nodes[vert]['orig_A'] = graph.nodes[vert][config['PARTY_A_COL']]
                        graph.nodes[vert]['orig_B'] = graph.nodes[vert][config['PARTY_B_COL']]

                    # Set values of new node to 0 by default
                    for keyword in ['POP_COL', 'PARTY_A_COL', 'PARTY_B_COL']:
                        if config[keyword] not in graph.nodes[label]:
                            graph.nodes[label][config[keyword]] = 0

                    # Increment values of new node and decrement values of old nodes
                    # by the appropriate amount
                    graph.nodes[label][config['POP_COL']] += graph.nodes[vert]['orig_pop'] // 3
                    graph.nodes[label][config['PARTY_A_COL']] += graph.nodes[vert]['orig_A'] // 3
                    graph.nodes[label][config['PARTY_B_COL']] += graph.nodes[vert]['orig_B'] // 3
                    for keyword in ['POP_COL', 'PARTY_A_COL', 'PARTY_B_COL']:
                        graph.nodes[vert][config[keyword]] -= graph.nodes[label][config[keyword]]

            # Set the population and votes of the new nodes to zero. Do not change
            # previously existing nodes.
            elif config['SIERPINSKI_POP_STYLE'] == 'zero':
                graph.nodes[label][config['POP_COL']] = 0
                graph.nodes[label][config['PARTY_A_COL']] = 0
                graph.nodes[label][config['PARTY_A_COL']] = 0
            else:
                raise RuntimeError('SIERPINSKI_POP_STYLE must be "uniform" or "zero"')

            # Remove edge between consecutive nodes if it exists
            try:
                graph.remove_edge(vertex, next_index)
            except:
                pass

            # Add edge between both of the original nodes and the new node
            graph.add_edge(vertex, label)
            graph.add_edge(label, next_vertex)
            # Add node to connections
            connections.append(label)

        # Add an edge between each consecutive new node
        for vertex in range(len(connections)):
            graph.add_edge(connections[vertex], connections[(vertex+1) % len(connections)])


def preprocessing(path_to_json):
    """Takes file path to JSON graph, and returns the appropriate

    Args:
        path_to_json ([String]): path to graph in JSON format

    Returns:
        graph (Gerrychain Graph): graph in JSON file following cleaning
        dual (Gerrychain Graph): planar dual of graph
    """
    graph = Graph.from_json(path_to_json)
    # For each node in graph, set 'pos' keyword to position
    for node in graph.nodes():
        graph.nodes[node]['pos'] = (graph.nodes[node][config['X_POSITION']],
                                    graph.nodes[node][config['Y_POSITION']])

    save_fig(graph, config['UNDERLYING_GRAPH_FILE'], config['WIDTH'])

    # Cleans graph to be able to find planar dual
    # TODO: why is this necessary?
    #duality_cleaning(graph)

    print('Making Dual')
    dual = facefinder.restricted_planar_dual(graph)
    print('Made Dual')

    return graph, dual

# TODO: What's going on here?
def duality_cleaning(graph):
    # Have to remove bad nodes in order for the duality thing to work properly
    while True:
        print("Clean Up Phase")
        print(len(graph))
        deg_one_nodes = []
        for v in graph.nodes():
            if graph.degree(v) == 1:
                deg_one_nodes.append(v)
        graph.remove_nodes_from(deg_one_nodes)

        deg_2_nodes = []
        for v in graph.nodes():
            if graph.degree(v) == 2:
                deg_2_nodes.append(v)

        for v in deg_2_nodes:
            smooth_node(graph, v)

        # Exit loop if there are no more degree 1 or 2 nodes
        if (not any([(graph.degree(node) == 1 or graph.degree(node) == 2) for node in graph.nodes()])):
            break


def smooth_node(graph, v):
    neighbors = list(graph.neighbors(v))
    graph.remove_node(v)
    try:
        graph.add_edge(neighbors[0], neighbors[1])
    except:
        print(neighbors)

def step_num(partition):
    """Determines the step in a chain a given partion is. Used as an updater.

    Args:
        partition (Gerrychain Partition): partition in question

    Returns:
        int: step in chain
    """
    parent = partition.parent

    if not parent:
        return 0

    return parent["step_num"] + 1


def save_fig(graph, path, size):
    """Saves graph to file in desired formed

    Args:
        graph (Gerrychain Graph): graph to be saved
        path (String): path to file location
        size (int): width of image
    """
    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=1, width=size, cmap=plt.get_cmap('jet'))
    # Gets format from end of filename
    plt.savefig(path, format=path.split('.')[-1])
    plt.close()

def produce_gerrymanders(graph, tag, sample_size, chaintype):
    """Runs a Recom chain, and saves the seats won histogram to a file and
       returns the most Gerrymandered plans for both PartyA and PartyB

    Args:
        graph (Gerrychain Graph): graph to run chain on on
        tag (String): tag added to filename of histogram
        sample_size (int): total steps of chain
        chaintype (String): indicates which proposal to be used to generate
        spanning tree during Recom. Must be either "tree" or "uniform_tree"

    Raises:
        RuntimeError: If chaintype is not "tree" nor 'uniform_tree"

    Returns:
        left_mander [Gerrymander Partition]: the most gerrymandered plan for
        PartyB generated by the chain
        right_mander [Gerrymander Partition]: the most gerrymandered plan for
        PartyA generated by the chain
    """
    for n in graph.nodes():
        graph.nodes[n]["last_flipped"] = 0
        graph.nodes[n]["num_flips"] = 0

    election = Election(
                        config['ELECTION_NAME'],
                        {'PartyA': config['PARTY_A_COL'],
                        'PartyB': config['PARTY_B_COL']}
                        )

    updaters = {'population': Tally(config['POP_COL']),
                'cut_edges': cut_edges,
                'step_num': step_num,
                config['ELECTION_NAME'] : election
                }
    initial_partition = Partition(graph, assignment=config['ASSIGN_COL'], updaters=updaters)
    ideal_population = sum(graph.nodes[x][config["POP_COL"]] for x in graph.nodes()) / len(initial_partition)
    popbound = within_percent_of_ideal_population(initial_partition, config['POPULATION_EPSILON'])

    # Determine proposal for generating spanning tree based upon parameter
    if chaintype == "tree":
        tree_proposal = partial(recom, pop_col=config["POP_COL"], pop_target=ideal_population,
                           epsilon=config['POPULATION_EPSILON'], node_repeats=config['NODE_REPEATS'],
                           method=facefinder.my_mst_bipartition_tree_random)

    elif chaintype == "uniform_tree":
        tree_proposal = partial(recom, pop_col=config["POP_COL"], pop_target=ideal_population,
                           epsilon=config['POPULATION_EPSILON'], node_repeats=config['NODE_REPEATS'],
                           method=facefinder.my_uu_bipartition_tree_random)
    else:
        print("Chaintype used: ", chaintype)
        raise RuntimeError("Chaintype not recognized. Use 'tree' or 'uniform_tree' instead")

    # Chain to be run
    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept, initial_state=initial_partition,
                            total_steps=sample_size)

    # Run chain while saving seats won for PartyA and the plans most gerrymandered
    # for both PartyA and PartyB
    seats_won_table = []
    best_left = np.inf
    best_right = -np.inf
    for i, partition in enumerate(exp_chain):

        if i % 100 == 0:
            print('Step', i)

        # Total seats won by PartyA
        seats_won = partition[config['ELECTION_NAME']].seats('PartyA')
        seats_won_table.append(seats_won)

        # Saves copies of gerrymandered partitions
        if seats_won < best_left:
            best_left = seats_won
            left_mander = Partition(graph=partition.graph, assignment=partition.assignment,
                                    updaters=partition.updaters)
        if seats_won > best_right:
            best_right = seats_won
            right_mander = Partition(graph=partition.graph, assignment=partition.assignment,
                                    updaters=partition.updaters)

    print('PartyA-mander:', best_right, 'PartyB-mander:', best_left)

    # Save histogram to appropriate file
    plt.figure()
    plt.hist(seats_won_table, bins=len(initial_partition)+1)
    name = './plots/large_sample/seats_hist/seats_histogram_orig' + tag + '.png'
    plt.savefig(name)
    plt.close()

    return left_mander, right_mander


def determine_special_faces(graph, dist):
    """Determines the special faces, which are those nodes whose distance is 
    at least k

    Args:
        graph (Gerrychain Graph): graph to determine special faces of
        dist (numeric): distance such that nodes are considered special if
        they have a 'distance' of at least this value

    Returns:
        list: list of nodes which are special
    """
    return [node for node in graph.nodes() if graph.nodes[node]['distance'] >= dist]

def determine_special_faces_random(graph, exp=1):
    """Determines the special faces, which are determined randomly with the probability
    of a given node being considered special being proportional to its distance
    raised to the exp power

    Args:
        graph (Gerrychain Graph): graph to determine special faces of
        exp (float, optional): exponent appearing in probability of a given node
        being considered special. Defaults to 1.

    Returns:
        list: list of nodes which are special
    """
    max_dist = max(graph.nodes[node]['distance'] for node in graph.nodes())
    return [node for node in graph.nodes() if random.uniform < (graph.nodes[node]['distance'] / max_dist) ** exp]

def metamander_around_partition(partition, dual, secret=False, special_param=2):
    """Metamanders around a partition by determining the set of special faces,
    and then sierpinskifying them.

    Args:
        partition (Gerrychain Partition): Partition to metamander around
        dual (Networkx Graph): planar dual of partition's graph
        secret (Boolean): whether to metamander 'in secret'. If True, determines
        special faces randomly, else not.
        special_param (numeric): additional parameter passed to special faces function
    """

    graph = partition.graph
#    facefinder.viz(partition, set([]))
#    plt.savefig("./plots/large_sample/target_maps/target_map" + tag + ".png", format='png')
#    plt.close()

    # Set of edges which cross from one district to another one
    cross_edges = facefinder.compute_cross_edges(partition)
    # Edges of dual graph corresponding to cross_edges
    dual_crosses = [edge for edge in dual.edges if dual.edges[edge]['original_name'] in cross_edges]

    # Assigns the graph distance from the dual_crosses to each node of the dual graph
    facefinder.distance_from_partition(dual, dual_crosses)
    # Assign special faces based upon set distances
    if secret:
        special_faces = determine_special_faces_random(dual, special_param)
    else:
        special_faces = determine_special_faces(dual, special_param)
    # Metamander around the partition by Sierpinskifying the special faces
    face_sierpinski_mesh(graph, special_faces)
    print("Made Metamander")
#    print("assigning districts to metamander")
#    total_pop = sum([graph.nodes[node]['population'] for node in graph])
#    cddict = recursive_tree_part(graph, range(num_dist), total_pop / num_dist, "population", .01, 1)
#    for node in graph.nodes():
#        graph.nodes[node]['part'] = cddict[node]
#    # sierp_partition = build_trivial_partition(g_sierpinsky)
#    print("assigned districts")
#    plt.figure()
#    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=1, width=1,
#            cmap=plt.get_cmap('jet'))
#    plt.savefig("./plots/large_sample/sierpinsky_plots/sierpinsky_mesh" + tag + ".png", format='png')
#    plt.close()
#    return graph, k

# TODO: Changed 10000 to 1000
def produce_sample(graph, k, tag, sample_size=500, chaintype='tree'):
    # Samples k partitions of the graph, stores the cut edges and records them graphically
    # Also stores vote histograms, and returns most extreme partitions.
    print("producing sample")
    updaters = {'population': Tally('population'),
                'cut_edges': cut_edges,
                'step_num': step_num,
                }
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['cut_times'] = 0

    for n in graph.nodes():
        # graph.nodes[n]["population"] = 1 #graph.nodes[n]["POP10"] #This is something gerrychain will refer to for checking population balance
        graph.nodes[n]["last_flipped"] = 0
        graph.nodes[n]["num_flips"] = 0

    print("set up chain")
    ideal_population = sum(graph.nodes[x]["population"] for x in graph.nodes()) / k
    initial_partition = Partition(graph, assignment='part', updaters=updaters)
    pop1 = .05
    print("popbound")
    popbound = within_percent_of_ideal_population(initial_partition, pop1)

    if chaintype == "tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=facefinder.my_mst_bipartition_tree_random)

    elif chaintype == "uniform_tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=facefinder.my_uu_bipartition_tree_random)
    else:
        print("Chaintype used: ", chaintype)
        raise RuntimeError("Chaintype not recongized. Use 'tree' or 'uniform_tree' instead")

    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept, initial_state=initial_partition,
                            total_steps=sample_size)

    z = 0
    num_cuts_list = []
    seats_won_table = []
    best_left = np.inf
    best_right = -np.inf
    print("begin chain")
    for part in exp_chain:

        # if z % 100 == 0:
        z += 1
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
            seats_won += total_seats
        # total seats won by rep
        seats_won_table.append(seats_won)
        # save gerrymandered partitionss
        if seats_won < best_left:
            best_left = seats_won
            left_mander = copy.deepcopy(part.parts)
        if seats_won > best_right:
            best_right = seats_won
            right_mander = copy.deepcopy(part.parts)
        # print("finished round"

    print("max", best_right, "min:", best_left)

    plt.figure()
    plt.hist(seats_won_table, bins=10)
    mean = sum(seats_won_table) / len(seats_won_table)
    std = np.std(seats_won_table)
    # plt.close()
    # title = 'mean: ' + str(mean) + ' standard deviation: ' + str(std)
    # plt.title(title)
    # name = "./plots/seats_hist/seats_histogram" + tag + ".png"
    # plt.savefig(name)
    # plt.close()

    # edge_colors = [graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()]
    #
    # plt.figure()
    # nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=0,
    #         edge_color=edge_colors, node_shape='s',
    #         cmap='magma', width=1)
    # plt.savefig("./plots/edges_plots/edges" + tag + ".png")
    # plt.close()

    return mean, std, graph

def run_chain(init_part, chaintype, length, ideal_population):
    """Runs a Recom chain, and saves the seats won histogram to a file and
       returns the most Gerrymandered plans for both PartyA and PartyB

    Args:
        init_part (Gerrychain Partition): initial partition of chain
        chaintype (String): indicates which proposal to be used to generate
        spanning tree during Recom. Must be either "tree" or "uniform_tree"
        length (int): total steps of chain

    Raises:
        RuntimeError: If chaintype is not "tree" nor 'uniform_tree"

    Returns:
        left_mander [Gerrymander Partition]: the most gerrymandered plan for
        PartyB generated by the chain
        right_mander [Gerrymander Partition]: the most gerrymandered plan for
        PartyA generated by the chain
    """
    for edge in init_part.graph.edges():
        init_part.graph.edges[edge[0], edge[1]]['cut_times'] = 0

    popbound = within_percent_of_ideal_population(init_part, config['EPSILON'])

    # Determine proposal for generating spanning tree based upon parameter
    if chaintype == "tree":
        tree_proposal = partial(recom, pop_col=config["POP_COL"], pop_target=ideal_population,
                           epsilon=config['EPSILON'], node_repeats=config['NODE_REPEATS'],
                           method=facefinder.my_mst_bipartition_tree_random)

    elif chaintype == "uniform_tree":
        tree_proposal = partial(recom, pop_col=config["POP_COL"], pop_target=ideal_population,
                           epsilon=config['EPSILON'], node_repeats=config['NODE_REPEATS'],
                           method=facefinder.my_uu_bipartition_tree_random)
    else:
        print("Chaintype used: ", chaintype)
        raise RuntimeError("Chaintype not recognized. Use 'tree' or 'uniform_tree' instead")

    # Chain to be run
    chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept, initial_state=init_part,
                            total_steps=length)

    # Run chain while saving each partition and counting how many times each
    # edge is cut.
    arr = []
    for i, partition in enumerate(chain):
        if i % 10 == 0:
            print(i)
        for edge in partition["cut_edges"]:
            init_part.graph[edge[0]][edge[1]]["cut_times"] += 1
        arr.append(partition)
    return arr

def saveHistogram(partitions, bins, tag):
    """Saves a list of numbers as a histogram in the appropriate location

    Args:
        partitions (list): list of Gerrychain Partitions
        bins (int): number of bins of histogram
        tag (String): tag to add to filename
    """
    print('entered function')
    # Determine table of data to save
    table = [partition[config['ELECTION_NAME']].seats('PartyA') for partition in samples]
    mean = sum(table) / len(table)
    std = np.std(table, ddof=1)
    # Save histogram to appropriate file
    plt.figure()
    plt.hist(table, bins=bins)
    title = 'mean: ' + str(mean) + ' standard deviation: ' + str(std)
    plt.title(title)
    name = "./plots/seats_hist/seats_histogram" + tag + ".png"
    plt.savefig(name)
    plt.close()

def drawGraph(graph, tag):
    edge_colors = [graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()]

    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=0,
            edge_color=edge_colors, node_shape='s',
            cmap='magma', width=1)
    plt.savefig("./plots/edges_plots/edges" + tag + ".png")
    plt.close()

def main():

    # left_mander, right_mander = produce_gerrymanders(graph, 12, '_nc', 100, 'tree')
    graph, dual = preprocessing(config["INPUT_GRAPH_FILENAME"])
    parts = list(set([graph.nodes[node][config['ASSIGN_COL']] for node in graph.nodes()]))
    ideal_pop = sum([graph.nodes[node][config['POP_COL']] for node in graph.nodes()]) / len(parts)
    election = Election(
                        config['ELECTION_NAME'],
                        {'PartyA': config['PARTY_A_COL'],
                        'PartyB': config['PARTY_B_COL']}
                        )

    updaters = {'population': Tally(config['POP_COL']),
                'cut_edges': cut_edges,
                'step_num': step_num,
                config['ELECTION_NAME'] : election
                }
    mean_table = []
    std_table = []
    # metamander, k = metamander_around_partition(graph, dual, left_mander, '_nc' + "LEFTMANDER", num_dist, False)
    #
    # produce_sample(metamander, k, '_nc')
    max_mean = 0
    min_mean = math.inf
    # TODO: Changed 500 to 15
    for i in range(config['NUM_EXPERIMENTS']):
        graph_copy = copy.deepcopy(graph)
        dual_copy = copy.deepcopy(dual)
        print("Searching for initial partition")
        # Note: doing this instead of using default assignment slows things down
        partition = Partition(graph=graph_copy, assignment=config['ASSIGN_COL'], updaters=updaters)
        run = run_chain(partition, config['CHAIN_TYPE'], config['FIND_GERRY_LENGTH'], ideal_pop)
        firstTable = [partition[config['ELECTION_NAME']].seats('PartyA') for partition in run]
        print('Ran chain')
        left_mander = min(run, key=lambda x: x[config['ELECTION_NAME']].seats('PartyA'))
        print('Found left mander')
        metamander_around_partition(partition, dual_copy, False, 2)
        # Refresh partition
        assign = recursive_tree_part(graph_copy, parts, ideal_pop, config['POP_COL'],
                                     config['EPSILON'], config['NODE_REPEATS'])
        updaters = {'population': Tally(config['POP_COL']),
                    'cut_edges': cut_edges,
                    'step_num': step_num,
                    config['ELECTION_NAME'] : election
                    }
        print('Found new intial partition')
        partition = Partition(graph=graph_copy, assignment=assign, updaters=updaters)
        samples = run_chain(partition, config['CHAIN_TYPE'], config['SAMPLE_META_LENGTH'], ideal_pop)
        print('Ran chain')
        table = [partition[config['ELECTION_NAME']].seats('PartyA') for partition in samples]
        saveHistogram(samples, range(len(parts)+1), config['HISTOGRAM_TAG'])
        drawGraph(partition.graph, config['GRAPH_TAG'])
        print('Saved images')

        edge_colors = [partition.graph[edge[0]][edge[1]]["cut_times"] for edge in graph_copy.edges()]

        mean = sum(table) / len(table)
        std = np.std(table, ddof=1)

        title = 'mean: ' + str(mean) + ' standard deviation: ' + str(std) + " Number:" + str(i)
        plt.title(title)
        if mean > max_mean:
            name = "./plots/extreme_shift/large_sample/seats_histogram" + "MaxMean_Left_random" + ".png"
            plt.savefig(name)
            plt.close()

            plt.figure()
            nx.draw(partition.graph, pos=nx.get_node_attributes(partition.graph, 'pos'), node_size=0,
                    edge_color=edge_colors, node_shape='s',
                    cmap='magma', width=1)
            plt.savefig("./plots/extreme_shift/large_sample/edges" + "MaxMean_Left_random" + ".png")
            plt.close()
            max_mean = mean
        elif mean < min_mean:
            name = "./plots/extreme_shift/large_sample/seats_histogram" + "MinMean_Left_random" + ".png"
            plt.savefig(name)
            plt.close()

            plt.figure()
            nx.draw(partition.graph, pos=nx.get_node_attributes(partition.graph, 'pos'), node_size=0,
                    edge_color=edge_colors, node_shape='s',
                    cmap='magma', width=1)
            plt.savefig("./plots/extreme_shift/large_sample/edges" + "MinMean_Left_random" + ".png")
            plt.close()
            min_mean = mean

        mean_table.append(mean)
        std_table.append(std)
        print("Finished " + str(i) + " sample")

    plt.figure()
    plt.subplot(121)
    plt.hist(mean_table, bins=10)
    mean_table.sort()
    plt.title("min mean: " + str(float("{:.2f}".format(mean_table[0]))) + " max mean: " + str(float("{:.2f}".format(mean_table[-1]))))

    plt.subplot(122)
    plt.hist(std_table, bins=10)
    std_table.sort()
    plt.title("min std: " + str(float("{:.2f}".format(std_table[0]))) + " max std: " + str(float("{:.2f}".format(std_table[-1]))))
    plt.savefig("./plots/extreme_shift/tables_Left_random.png", format='png')
    plt.close()

if __name__ == '__main__':
    # Loads JSON config file
    if len(sys.argv) > 2:
        print("Provide single filename")
        sys.exit()
    elif len(sys.argv) == 2:
        configFileName = sys.argv[1]
    else:
        configFileName = './config.json'

    with open(configFileName, 'r') as json_file:
        try:
            global config
            config = json.load(json_file)
        except:
            print("Unable to load JSON file")
            sys.exit()

    main()