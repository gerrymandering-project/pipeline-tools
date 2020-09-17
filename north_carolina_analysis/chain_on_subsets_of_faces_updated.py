## This script will perform a markov chain on the  subset faces of the north carolina graph, picking a face UAR and sierpinskifying or de-sierpinskifying it, then running gerrychain on the graph, recording central seat tendencies.
import facefinder
import random
import statistics
import math
import gerrychain
import networkx
import matplotlib.pyplot as plt
import networkx as nx
from functools import partial
from gerrychain.tree import bipartition_tree as bpt
from gerrychain import Graph, MarkovChain
from gerrychain import accept
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

def face_sierpinski_mesh(graph, special_faces):
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

    # Get maximum node label.
    label = max(list(graph.nodes()))
    # Assign each node to its district in partition
    #for node in graph.nodes():
    #    graph.nodes[node][config['ASSIGN_COL']] = partition.assignment[node]

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
    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=1, width=size, cmap=plt.get_cmap('jet'))
    # Gets format from end of filename
    plt.savefig(path, format=path.split('.')[-1])
    plt.close()

def main():
    #gerrychain parameters
    #num districts
    k = 12
    epsilon = .05
    updaters = {'population': Tally('population'),
                            'cut_edges': cut_edges,
                            }
    graph, dual = preprocessing("json/NC.json")
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

    temperature = 1
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


        ##Makes the Markov chain lazy -- this just makes the chain aperiodic.
        if random.random() > .5:
            if not face in special_faces:
                special_faces.append(face)
            else:
                special_faces.remove(face)


        face_sierpinski_mesh(graph, special_faces)

        initial_partition = Partition(graph, assignment=config['ASSIGN_COL'], updaters=updaters)


        # Sets up Markov chain
        popbound = within_percent_of_ideal_population(initial_partition, epsilon)
        tree_proposal = partial(recom, pop_col=config['POP_COL'], pop_target=ideal_population, epsilon=epsilon,
                                    node_repeats=1, method=facefinder.my_mst_bipartition_tree_random)


        #make new function -- this computes the energy of the current map
        exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept,
                                initial_state=initial_partition, total_steps=gerrychain_steps)
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

        score = statistics.mean(seats_won_for_republicans)


        ##This is the acceptance step of the Metropolis-Hasting's algorithm.
        if random.random() < min(1, (math.exp(score) / chain_output['score'][z - 1])**(1/temperature) ):
             #if code acts weird, check if sign is wrong, unsure
             #rand < min(1, P(x')/P(x))
            chain_output['dem_seat_data'].append(seats_won_for_democrats)
            chain_output['rep_seat_data'].append(seats_won_for_republicans)
            chain_output['score'].append(math.exp(statistics.mean(seats_won_for_republicans)))
        else:
            chain_output['dem_seat_data'].append(chain_output['dem_seat_data'][z-1])
            chain_output['rep_seat_data'].append(chain_output['rep_seat_data'][z-1])
            chain_output['score'].append(chain_output['score'][z-1])

if __name__ ==  '__main__':
    global config
    config = {
        "INPUT_GRAPH_FILENAME" : "./json/NC.json",
        "X_POSITION" : "C_X",
        "Y_POSITION" : "C_Y",
        "UNDERLYING_GRAPH_FILE" : "./plots/UnderlyingGraph.png",
        "WIDTH" : 1,
        "ASSIGN_COL" : "part",
        "POP_COL" : "population",
    }
    main()
