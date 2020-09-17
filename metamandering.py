import facefinder
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, Value
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
import networkx as nx
import numpy as np
import copy
import random
import math
import json
import sys
import os
import traceback

from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain import accept
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population, UpperBound)

from gerrychain.updaters import Election, Tally, cut_edges

from gerrychain.partition import Partition
from gerrychain.proposals import recom

from gerrychain.tree import recursive_tree_part


def face_sierpinski_mesh(partition, special_faces):
    """'Sierpinskifies' certain faces of the graph by adding nodes and edges to
    certain faces.

    Args:
        partition (Gerrychain Partition): partition object which contain assignment
        and whose graph will have edges and nodes added to
        special_faces (list): list of faces that we want to add node/edges to

    Raises:
        RuntimeError if SIERPINSKI_POP_STYLE of config file is neither 'uniform'
        nor 'zero'
    """

    graph = partition.graph
    # Get maximum node label.
    label = max(list(graph.nodes()))
    # Assign each node to its district in partition
    for node in graph.nodes():
        graph.nodes[node][config['ASSIGN_COL']] = partition.assignment[node]

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

        newNodes = []
        newEdges = []
        # For each consecutive pair of nodes, remove their edge, create a new
        # node at their average position, and connect edge node to the new node:
        for vertex, next_vertex in zip(neighbors, neighbors[1:] + [neighbors[0]]):
            label += 1
            # Add new node to graph with corresponding label at the average position
            # of vertex and next_vertex, and with 0 population and 0 votes
            graph.add_node(label)
            avgPos = (np.array(graph.nodes[vertex]['pos']) +
                      np.array(graph.nodes[next_vertex]['pos'])) / 2
            graph.nodes[label]['pos'] = avgPos
            graph.nodes[label][config['X_POSITION']] = avgPos[0]
            graph.nodes[label][config['Y_POSITION']] = avgPos[1]
            graph.nodes[label][config['POP_COL']] = 0
            graph.nodes[label][config['PARTY_A_COL']] = 0
            graph.nodes[label][config['PARTY_B_COL']] = 0

            # For each new node, 'move' a third of the population, Party A votes,
            # and Party B votes from its two adjacent nodes which previously exists
            # to itself (so that each previously existing node equally shares
            # its statistics with the two new nodes adjacent to it)
            if config['SIERPINSKI_POP_STYLE'] == 'uniform':
                for vert in [vertex, next_vertex]:
                    for keyword, orig_keyword in zip(['POP_COL', 'PARTY_A_COL', 'PARTY_B_COL'],
                                                     ['orig_pop', 'orig_A', 'orig_B']):
                        # Save original values if not done already
                        if orig_keyword not in graph.nodes[vert]:
                            graph.nodes[vert][orig_keyword] = graph.nodes[vert][config[keyword]]

                        # Increment values of new node and decrement values of old nodes
                        # by the appropriate amount.
                        graph.nodes[label][config[keyword]] += graph.nodes[vert][orig_keyword] // 3
                        graph.nodes[vert][config[keyword]] -= graph.nodes[vert][orig_keyword] // 3

                # Assign new node to same district as neighbor. Note that intended
                # behavior is that special_faces do not correspond to cut edges,
                # and therefore both vertex and next_vertex will be of the same
                # district.
                graph.nodes[label][config['ASSIGN_COL']] = graph.nodes[vertex][config['ASSIGN_COL']]

            # Set the population and votes of the new nodes to zero. Do not change
            # previously existing nodes. Assign to random neighbor.
            elif config['SIERPINSKI_POP_STYLE'] == 'zero':
                graph.nodes[label][config['ASSIGN_COL']] =\
                random.choice([graph.nodes[vertex][config['ASSIGN_COL']],
                               graph.nodes[next_vertex][config['ASSIGN_COL']]]
                             )
            else:
                raise RuntimeError('SIERPINSKI_POP_STYLE must be "uniform" or "zero"')

            # Remove edge between consecutive nodes if it exists
            if graph.has_edge(vertex, next_vertex):
                graph.remove_edge(vertex, next_vertex)

            # Add edge between both of the original nodes and the new node
            graph.add_edge(vertex, label)
            newEdges.append((vertex, label))
            graph.add_edge(label, next_vertex)
            newEdges.append((label, next_vertex))
            # Add node to connections
            newNodes.append(label)

        # Add an edge between each consecutive new node
        for vertex in range(len(newNodes)):
            graph.add_edge(newNodes[vertex], newNodes[(vertex+1) % len(newNodes)])
            newEdges.append((newNodes[vertex], newNodes[(vertex+1) % len(newNodes)]))
        # For each new edge of the face, set sibilings to be the tuple of all
        # new edges
        siblings = tuple(newEdges)
        for edge in newEdges:
            graph.edges[edge]['siblings'] = siblings


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

    dual = facefinder.restricted_planar_dual(graph)

    return graph, dual


def save_fig(graph, path, size):
    """Saves graph to file in desired formed

    Args:
        graph (Gerrychain Graph): graph to be saved
        path (String): path to file location
        size (int): width of image
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=0.01, node_color='none', width=size,
            cmap=plt.get_cmap('jet'))
    plt.savefig(path, format=path.split('.')[-1])
    plt.close()


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


def metamander_around_partition(partition, dual, tag, secret=False, special_param=2):
    """Metamanders around a partition by determining the set of special faces,
    and then sierpinskifying them.

    Args:
        partition (Gerrychain Partition): Partition to metamander around
        dual (Networkx Graph): planar dual of partition's graph
        secret (Boolean): whether to metamander 'in secret'. If True, determines
        special faces randomly, else not.
        special_param (numeric): additional parameter passed to special faces function
    """

    # do we need graph in here, will this target_map update every time?
    graph = partition.graph
    facefinder.viz(partition, set([]))
    plt.savefig("./plots/large_sample/target_maps/target_map" + tag + ".svg", format='svg')
    plt.close()

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
    face_sierpinski_mesh(partition, special_faces)


def run_chain(init_part, chaintype, length, ideal_population, id):
    """Runs a Recom chain, and saves the seats won histogram to a file and
       returns the most Gerrymandered plans for both PartyA and PartyB

    Args:
        init_part (Gerrychain Partition): initial partition of chain
        chaintype (String): indicates which proposal to be used to generate
        spanning tree during Recom. Must be either "tree" or "uniform_tree"
        length (int): total steps of chain
        id (String): id of experiment, used when printing progress

    Raises:
        RuntimeError: If chaintype is not "tree" nor 'uniform_tree"

    Returns:
        list of partitions generated by chain
    """
    graph = init_part.graph
    for edge in graph.edges():
        graph.edges[edge]['cut_times'] = 0
        graph.edges[edge]['sibling_cuts'] = 0
        if 'siblings' not in graph.edges[edge]:
            graph.edges[edge]['siblings'] = tuple([edge])

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

    # Run chain, save each desired statistic, and keep track of cuts. Save most
    # left gerrymandered partition
    # should we save the rep and dem voting data in here?
    partitions = []
    for i, partition in enumerate(chain):
        for edge in partition["cut_edges"]:
            graph.edges[edge]['cut_times'] += 1
            for sibling in graph.edges[edge]['siblings']:
                graph.edges[sibling]['sibling_cuts'] += 1
        partitions.append(partition)
        if i % 500 == 0:
            print('{}: {}'.format(id, i))
    return partitions


def getGerry(partitions):
    """Returns the most gerrymandered partition of a given iterable of partitions.
    'The most gerrymandered' indicates the partition which takes the minimum value
    of the first Gerry Statistic. If there is a tie, then the second Gerry Statistic
    is used, and so on.

    Args:
        partitions (Iterable): Iterable of partitions to find most gerrymandered
        partition from
    """
    electionDict = {
        'seats' : (lambda x: x[config['ELECTION_NAME']].seats('PartyA')),
        'won' : (lambda x: x[config['ELECTION_NAME']].seats('PartyA')),
        'efficiency_gap' : (lambda x: x[config['ELECTION_NAME']].efficiency_gap()),
        'mean_median' : (lambda x: x[config['ELECTION_NAME']].mean_median()),
        'mean_thirdian' : (lambda x: x[config['ELECTION_NAME']].mean_thirdian()),
        'partisan_bias' : (lambda x: x[config['ELECTION_NAME']].partisan_bias()),
        'partisan_gini' : (lambda x: x[config['ELECTION_NAME']].partisan_gini())
    }
    return min(partitions, key=lambda x: [electionDict[statistic](x) for statistic in config['GERRY_STATISTICS']])


def drawGraph(graph, property, tag):
    """Draws graph with edges colored according to the value of their chosen
    property. Saves to file.

    Args:
        graph (Networkx Graph): graph to draw and save
        property (String): property of edges to use for edge colormap
        tag (String): tag added to filename to identify graph
    """
    edge_colors = [graph.edges[edge][property] for edge in graph.edges()]
    vmin = min(edge_colors)
    vmax = max(edge_colors)
    cmap = plt.get_cmap(config['COLORMAP1'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=0,
            edge_color=edge_colors, node_shape='s',
            edge_cmap=cmap, width=0.1, edge_vmin=vmin, edge_vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm, shrink=0.8)
    plt.savefig('./plots/edges_plots/edges_{}.svg'.format(tag))
    plt.close()


def drawDoubleGraph(graph, property, tag):
    """Draws graph with edges colored according to the value of their chosen
    property and whether or not they are from a special face. Saves to file.

    Identical to drawGraph() excepts uses two different edge colormaps for whether
    or not an edge comes from a special face.

    Args:
        graph (Networkx Graph): graph to draw and save
        property (String): property of edges to use for edge colormap
        tag (String): tag added to filename to identify graph
    """
    special_edges = [edge for edge in graph.edges() if len(graph.edges[edge]['siblings']) > 1]
    orig_edges = [edge for edge in graph.edges() if len(graph.edges[edge]['siblings']) == 1]

    G_special = graph.edge_subgraph(special_edges)
    G_orig = graph.edge_subgraph(orig_edges)

    special_edge_colors = [graph.edges[edge][property] for edge in special_edges]
    orig_edge_colors = [graph.edges[edge][property] for edge in orig_edges]

    vmin = min(min(special_edge_colors), min(orig_edge_colors))
    vmax = max(max(special_edge_colors), max(orig_edge_colors))

    cmap1 = plt.get_cmap(config['COLORMAP1'])
    cmap2 = plt.get_cmap(config['COLORMAP2'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    nx.draw(G_orig, pos=nx.get_node_attributes(graph, 'pos'), node_size=0,
            edge_color=orig_edge_colors, node_shape='s',
            edge_cmap=cmap1, width=0.1, edge_vmin=vmin, edge_vmax=vmax)
    nx.draw(G_special, pos=nx.get_node_attributes(graph, 'pos'), node_size=0,
            edge_color=special_edge_colors, node_shape='s',
            edge_cmap=cmap2, width=0.1, edge_vmin=vmin, edge_vmax=vmax)

    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm1._A = []
    clb_orig = plt.colorbar(sm1, shrink=0.8)
    clb_orig.ax.set_title('Original')

    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm2._A = []
    clb_special = plt.colorbar(sm2, shrink=0.8)
    clb_special.ax.set_title('Special')

    plt.savefig('./plots/edges_plots/edges_{}.svg'.format(tag))
    plt.close()


def saveRunStatistics(partitions, tag):
    """Saves the election statistics of a given list of partitions to a JSON file

    Args:
        partitions (Iterable): Iterable of partitions to save election statistics of
        tag (String): tag added to filename to identify run
    """
    electionDict = {
        'seats' : (lambda x: x[config['ELECTION_NAME']].seats('PartyA')),
        'won' : (lambda x: x[config['ELECTION_NAME']].seats('PartyA')),
        'efficiency_gap' : (lambda x: x[config['ELECTION_NAME']].efficiency_gap()),
        'mean_median' : (lambda x: x[config['ELECTION_NAME']].mean_median()),
        'mean_thirdian' : (lambda x: x[config['ELECTION_NAME']].mean_thirdian()),
        'partisan_bias' : (lambda x: x[config['ELECTION_NAME']].partisan_bias()),
        'partisan_gini' : (lambda x: x[config['ELECTION_NAME']].partisan_gini())
    }
    statistics = {statistic : [electionDict[statistic](partition) for partition in partitions]
                  for statistic in config['ELECTION_STATISTICS']}

    with open('generated_data/run_statistics_{}.json'.format(tag), 'w') as outfile:
        try:
            json.dump(statistics, outfile)
        except:
            track = traceback.format_exc()
            print(track)
            print('Unable to save run statistics to file.')


def saveGraphStatistics(graph, tag):
    """Saves the statistics of a graph to JSON file.

    Args:
        graph (Networkx Graph): graph to have data saved
        tag ([type]): tag added to filename to identify graph
    """
    data = [(edge, graph.edges[edge]['cut_times'], graph.edges[edge]['sibling_cuts']) for edge in graph.edges()]
    with open('generated_data/graph_statistics_{}.json'.format(tag), 'w') as outfile:
        try:
            json.dump(data, outfile)
        except:
            track = traceback.format_exc()
            print(track)
            print('Unable to save graph statistics to file.')


def savePartition(partition, tag):
    """Saves a partition to a JSON file

    Args:
        partition (Gerrychain Partition): partition to save
        tag (String): tag added to filename to identify partition
    """
    with open('generated_data/partition_{}.json'.format(tag), 'w') as outfile:
        try:
            json.dump(partition.assignment.to_dict(), outfile)
        except:
            track = traceback.format_exc()
            print(track)
            print('Unable to save partition to file')


def main(config_data, id):
    """Runs a single experiment with the given config file. Loads a graph,
    runs a Chain to search for a Gerrymander, metamanders around that partition,
    runs another chain, and then saves the generated data.

    Args:
        config_data (Object): configuration of experiment loaded from JSON file
        id (String): id of experiment, used in tags to differentiate between
        experiments
    """
    try:
        timeBeg = time.time()
        print('Experiment', id, 'has begun')
        # Save configuration into global variable
        global config
        config = config_data

        # Get graph and dual graph
        graph, dual = preprocessing(config["INPUT_GRAPH_FILENAME"])
        # List of districts in original graph
        parts = list(set([graph.nodes[node][config['ASSIGN_COL']] for node in graph.nodes()]))
        # Ideal population of districts
        ideal_pop = sum([graph.nodes[node][config['POP_COL']] for node in graph.nodes()]) / len(parts)
        # Initialize partition
        election = Election(
                            config['ELECTION_NAME'],
                            {'PartyA': config['PARTY_A_COL'],
                            'PartyB': config['PARTY_B_COL']}
                            )

        updaters = {'population': Tally(config['POP_COL']),
                    'cut_edges': cut_edges,
                    config['ELECTION_NAME'] : election
                    }

        partition = Partition(graph=graph, assignment=config['ASSIGN_COL'], updaters=updaters)
        # Run Chain to search for a gerrymander, and get it
        mander = getGerry(run_chain(partition, config['CHAIN_TYPE'],
                          config['FIND_GERRY_LENGTH'], ideal_pop, id + 'a'))
        savePartition(mander, config['LEFT_MANDER_TAG'] + id)
        # Metamanders around the found gerrymander
        metamander_around_partition(mander, dual, config['TARGET_TAG'] + id, config['SECRET'], config['META_PARAM'])
        # Refresh assignment and election of partition
        updaters[config['ELECTION_NAME']] = Election(
                                                     config['ELECTION_NAME'],
                                                     {'PartyA': config['PARTY_A_COL'],
                                                      'PartyB': config['PARTY_B_COL']}
                                                    )
        partition = Partition(graph=graph, assignment=config['ASSIGN_COL'], updaters=updaters)
        # Run chain again
        metamandered_run = run_chain(partition, config['CHAIN_TYPE'], config['SAMPLE_META_LENGTH'], ideal_pop, id + 'b')
        # Save data from experiment to JSON files
        drawGraph(partition.graph, 'cut_times', config['GRAPH_TAG'] + '_single_raw_' + id)
        drawGraph(partition.graph, 'sibling_cuts', config['GRAPH_TAG'] + '_single_adjusted_' + id)
        drawDoubleGraph(partition.graph, 'cut_times', config['GRAPH_TAG'] + '_double_raw_' + id)
        drawDoubleGraph(partition.graph, 'sibling_cuts', config['GRAPH_TAG'] + '_double_adjusted_' + id)
        saveRunStatistics(metamandered_run, config['RUN_STATISTICS_TAG'] + id)
        saveGraphStatistics(partition.graph, config['GRAPH_STATISTICS_TAG'] + id)

        print('Experiment {} completed in {:.2f} seconds'.format(id, time.time() - timeBeg))
    except Exception as e:
        # Print notification if any experiment fails to complete
        track = traceback.format_exc()
        print(track)
        print('Experiment {} failed to complete after {:.2f} seconds'.format(id, time.time() - timeBeg))

if __name__ == '__main__':
    timeStart = time.time()
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
            config = json.load(json_file)
        except:
            print("Unable to load JSON file")
            sys.exit()

    # Run NUM_EXPERIMENTS experiments using NUM_PROCESSORS processors
    pool = Pool(config['NUM_PROCESSORS'])
    for i in range(config['NUM_EXPERIMENTS']):
        pool.apply_async(main, args = (config, str(i)))
    pool.close()
    pool.join()
    print('All experiments completed in {:.2f} seconds'.format(time.time() - timeStart))