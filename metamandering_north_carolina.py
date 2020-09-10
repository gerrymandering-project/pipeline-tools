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
        graph: graph object that edges will be added to
        special_faces: list of faces that we want to add node/edges to

    Returns:
        graph: The original graph but with nodes and edges added to the special faces

    TODO:
        Add parameter for depth of sierpinskification
    """

    # Get maximum node label.
    max_label = max(list(graph.nodes()))

    for face in special_faces:
        graph.add_node(face)
        neighbor_list = [] #  Neighbors of face
        locations = [] # Relative position of each neighbor
        connections = [] # List of all new nodes added
        locationCount = np.array([0,0]).astype("float64")

        # For each face, add to neighbor_list and add to location count
        for vertex in face:
            neighbor_list.append(vertex)
            locationCount += np.array(graph.nodes[vertex]["pos"]).astype("float64")
        # Set position of face to be average of all of its vertices
        graph.nodes[face]["pos"] = locationCount / len(face)
        # In order, append the relative position of each vertex to the position of the face
        for vertex in face:
            locations.append(graph.nodes[vertex]["pos"] - graph.nodes[face]["pos"])
        # Sort neighbor_list according to each node's angle with the center of the face
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)

        # For each consecutive pair of nodes of, remove their edge, create a new
        # node at their average position, and connect edge node to the new node:
        for vertex in range(len(neighbor_list)):
            next_index = (vertex+1) % len(neighbor_list) # Index of next vertex counter-clockwise
            # Average position of consecutive nodes
            avgPos = (np.array(graph.nodes[neighbor_list[vertex]]["pos"]) +
                      np.array(graph.nodes[neighbor_list[next_index]]["pos"])) * 0.5

            # Determine new label, and increment max_label
            label = max_label + 1
            max_label += 1

            # Add new node to graph with corresponding label at avgPos
            graph.add_node(label)
            graph.nodes[label]['pos'] = avgPos

            # Remove edge between consecutive nodes
            remove_undirected_edge(graph, neighbor_list[vertex], neighbor_list[next_index])
            # Add edge between both of the original nodes and the new node
            graph.add_edge(neighbor_list[vertex],label)
            graph.add_edge(label,neighbor_list[next_index])
            # Add node to connections
            connections.append(label)
        # Add an edge between each consecutive new node
        for vertex in range(0,len(connections)):
            graph.add_edge(connections[vertex],connections[(vertex+1) % len(connections)])
        # Remove the original face
        graph.remove_node(face)

    # Return the altered graph
    return graph


def remove_undirected_edge(graph, v, u):
    """Removes a single directed edge between the two nodes from the graph, if
    such an edge exists

    Args:
        graph: graph to remove edge from
        v: first node
        u: second node

    TODO:
        Verify that both directions should not exist/should only remove a single
        direction
    """
    if (v, u) in graph.edges():
        graph.remove_edge(v, u)
        return

    if (u, v) in graph.edges():
        graph.remove_edge(u, v)
        return


def preprocessing(path_to_json):
    """Takes file path to JSON graph, and returns the appropriate 

    Args:
        path_to_json (String): path to graph in JSON format

    Returns:
        graph (Gerrychain Graph): graph in JSON file following cleaning
        dual (Gerrychain Graph): planar dual of graph
    """
    graph = Graph.from_json(path_to_json)
    # For each node in graph, set 'pos' parameter to position
    for node in graph.nodes():
        graph.nodes[node]['pos'] = (graph.nodes[node][config['X_POSITION']],
                                    graph.nodes[node][config['Y_POSITION']])

    save_fig(graph, config['UNDERLYING_GRAPH_FILE'], config['WIDTH'])

    # Cleans graph to be able to find planar dual
    # TODO: why is this necessary?
    #graph = duality_cleaning(graph)

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
            graph = smooth_node(graph, v)

        # Exit loop if there are no more degree 1 or 2 nodes
        if (not any([(graph.degree(node) == 1 or graph.degree(node) == 2) for node in graph.nodes()])):
            return graph


def smooth_node(graph, v):
    neighbors = list(graph.neighbors(v))
    graph.remove_node(v)
    try:
        graph.add_edge(neighbors[0], neighbors[1])
    except:
        print(neighbors)
    return graph


def step_num(partition):
    """Determines the step in a chain a given partion is. Used as an updated.

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
        graph (Gerrychain graph): graph to be saved
        path (String): path to file location
        size (int): width of image
    """
    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=1, width=size, cmap=plt.get_cmap('jet'))
    # Gets format from end of filename
    plt.savefig(path, format=path.split('.')[-1])
    plt.close()


def produce_gerrymanders(graph, k, tag, sample_size, chaintype):
    # Samples k-partitions of the graph
    # stores vote histograms, and returns most extreme partitions.
    for node in graph.nodes():
        graph.nodes[node]["last_flipped"] = 0
        graph.nodes[node]["num_flips"] = 0

    ideal_population = sum(graph.nodes[x][config["POPULATION_COLUMN"]] for x in graph.nodes()) / k
    election = Election(
        config['ELECTION_NAME'],
        {'PartyA': config['PARTY_A_COL'], 'PartyB': config['PARTY_B_COL']},
        alias=config['ELECTION_ALIAS']
    )
    updaters = {'population': Tally(config['POPULATION_COLUMN']),
                'cut_edges': cut_edges,
                'step_num': step_num,
                config['ELECTION_ALIAS'] : election
                }
    initial_partition = Partition(graph, assignment=config['ASSIGNMENT_COLUMN'], updaters=updaters)
    popbound = within_percent_of_ideal_population(initial_partition, config['POPULATION_EPSILON'])

    if chaintype == "tree":
        tree_proposal = partial(recom, pop_col=config["POPULATION_COLUMN"], pop_target=ideal_population,
                           epsilon=config['POPULATION_EPSILON'], node_repeats=config['NODE_REPEATS'],
                           method=facefinder.my_mst_bipartition_tree_random)

    elif chaintype == "uniform_tree":
        tree_proposal = partial(recom, pop_col=config["POPULATION_COLUMN"], pop_target=ideal_population,
                           epsilon=config['POPULATION_EPSILON'], node_repeats=config['NODE_REPEATS'],
                           method=facefinder.my_uu_bipartition_tree_random)
    else:
        print("Chaintype used: ", chaintype)
        raise RuntimeError("Chaintype not recognized. Use 'tree' or 'uniform_tree' instead")

    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept, initial_state=initial_partition,
                            total_steps=sample_size)

    seats_won_table = []
    best_left = np.inf
    best_right = -np.inf
    for ctr, part in enumerate(exp_chain):
        seats_won = 0

        if ctr % 100 == 0:
            print("step ", ctr)
        for i in range(k):
            rep_votes = 0
            dem_votes = 0
            for node in graph.nodes():
                if part.assignment[node] == i:
                    rep_votes += graph.nodes[node]["EL16G_PR_R"]
                    dem_votes += graph.nodes[node]["EL16G_PR_D"]
            total_seats = int(rep_votes > dem_votes)
            seats_won += total_seats
        # total seats won by rep
        seats_won_table.append(seats_won)
        # save gerrymandered partitions
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

    name = "./plots/large_sample/seats_hist/seats_histogram_orig" + tag + ".png"
    plt.savefig(name)
    plt.close()
    return left_mander, right_mander


def assign_special_faces(graph, k):
    special_faces = []
    for node in graph.nodes():
        if graph.nodes[node]['distance'] >= k:
            special_faces.append(node)
    return special_faces


def metamander_around_partition(graph, dual, target_partition, tag,num_dist):


def metamander_around_partition(graph, dual, target_partition, tag, num_dist, secret):
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

    target_partition = Partition(graph, assignment, updaters=updaters)

    facefinder.viz(graph, set([]), target_partition.parts)
    plt.savefig("./plots/large_sample/target_maps/target_map" + tag + ".png", format='png')
    plt.close()

    print("made partition")
    crosses = facefinder.compute_cross_edge(graph, target_partition)

    k = len(target_partition.parts)

    dual_crosses = []
    for edge in dual.edges:
        if dual.edges[edge]["original_name"] in crosses:
            dual_crosses.append(edge)

    print("making dual distances")
    dual = facefinder.distance_from_partition(dual, dual_crosses)
    print('finished making dual distances')
    if secret:
        special_faces = assign_special_faces_random(dual)
        # special_faces = assign_special_faces_sqrt(dual)
    else:
        special_faces = assign_special_faces(dual, 2)
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
    nx.draw(g_sierpinsky, pos=nx.get_node_attributes(g_sierpinsky, 'pos'), node_size=1, width=1,
            cmap=plt.get_cmap('jet'))
    plt.savefig("./plots/large_sample/sierpinsky_plots/sierpinsky_mesh" + tag + ".png", format='png')
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
                                node_repeats=1, method=facefinder.my_mst_bipartition_tree_random)

    elif chaintype == "uniform_tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=facefinder.my_uu_bipartition_tree_random)
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
            seats_won += total_seats
        # total seats won by rep
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


def main():
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

    # left_mander, right_mander = produce_gerrymanders(graph, 12, '_nc', 100, 'tree')
    graph, dual = preprocessing(config["INPUT_GRAPH_FILENAME"])
    hold_graph = copy.deepcopy(graph)
    hold_dual = copy.deepcopy(dual)
    num_dist = 13

    metamander , k = metamander_around_partition(graph, dual, left_mander, '_nc' + "LEFTMANDER",num_dist)


            plt.figure()
            nx.draw(hold_graph, pos=nx.get_node_attributes(hold_graph, 'pos'), node_size=0,
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
            nx.draw(hold_graph, pos=nx.get_node_attributes(hold_graph, 'pos'), node_size=0,
                    edge_color=edge_colors, node_shape='s',
                    cmap='magma', width=1)
            plt.savefig("./plots/extreme_shift/large_sample/edges" + "MinMean_Left_random" + ".png")
            plt.close()
            min_mean = mean

        hold_graph = copy.deepcopy(graph)
        hold_dual = copy.deepcopy(dual)
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
    main()