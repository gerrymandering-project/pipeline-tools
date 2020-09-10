import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

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


def compute_rotation_system(graph):
    #Graph nodes must have "pos" and is promised that the embedding is a straight line embedding
    #The graph will be returned in a way that every node has a dictionary that gives you the next edge clockwise around that node
    #The rotation system is  clockwise (0,2) -> (1,1) -> (0,0) around (0,1)
    for v in graph.nodes():
        graph.nodes[v]["pos"] = np.array(graph.nodes[v]["pos"])

    for v in graph.nodes():
        locations = []
        neighbor_list = list(graph.neighbors(v))
        for w in neighbor_list:
            locations.append(graph.nodes[w]["pos"] - graph.nodes[v]["pos"])
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)
        #sorted_neighbors = [x for _,x in sorted(zip(angles, neighbor_list))]
        rotation_system = {}
        for i in range(len(neighbor_list)):
            rotation_system[neighbor_list[i]] = neighbor_list[(i + 1) % len(neighbor_list)]
        graph.nodes[v]["rotation"] = rotation_system
    return graph

def transform(x):
    #takes x from [-pi, pi] and puts it in [0,pi]
    if x >= 0:
        return x
    if x < 0:
        return 2 * np.pi + x



def is_clockwise(graph,face, average):
    #given a face (with respect to the rotation system computed), determine if it belongs to a the orientation assigned to bounded faces
    angles = [transform(float(np.arctan2(graph.nodes[x]["pos"][0] - average[0], graph.nodes[x]["pos"][1] - average[1])))  for x in face]
    first = min(angles)
    rotated = [x - first for x in angles]
    next_smallest = min([x for x in rotated if x != 0])
    ind = rotated.index(0)
    if rotated[(ind + 1)% len(rotated)] == next_smallest:
        return False
    else:
        return True

def cycle_around_face(graph, e):
    #Faces are being stored as the list of vertices
    face = list([e[0], e[1]])
    #starts off with the two vertices of the edge e
    last_point = e[1]
    current_point = graph.nodes[e[1]]["rotation"][e[0]]
    next_point = current_point
    while next_point != e[0]:
        face.append(current_point)
        next_point = graph.nodes[current_point]["rotation"][last_point]
        last_point = current_point
        current_point = next_point
    return face


def compute_face_data(graph):
    #graph must already have a rotation_system
    faces = []
    #faces will stored as sets of vertices

    for e in graph.edges():
        #need to make sure you get both possible directions for each edge..

        face = cycle_around_face(graph, e)
        faces.append(tuple(face))
        #has to get passed to a tuple because networkx wants the names of vertices to be frozen
        face = cycle_around_face(graph, [ e[1], e[0]])
        #also cycle in the other direction
        faces.append(tuple(face))
    #detect the unbounded face based on orientation
    bounded_faces = []
    for face in faces:
        run_sum = np.array([0,0]).astype('float64')
        for x in face:
            run_sum += np.array(graph.nodes[x]["pos"]).astype('float64')
        average = run_sum / len(face)
        #associate each face to the average of the vertices
        if is_clockwise(graph,face, average):
            #figures out whether a face is bounded or not based on clockwise orientation
            bounded_faces.append(face)
    faces_set = [frozenset(face) for face in bounded_faces]
    graph.graph["faces"] = set(faces_set)
    return graph

def compute_all_faces(graph):
        #graph must already have a rotation_system
    faces = []
    #faces will stored as sets of vertices

    for e in graph.edges():
        #need to make sure you get both possible directions for each edge..

        face = cycle_around_face(graph, e)
        faces.append(tuple(face))
        face = cycle_around_face(graph, [ e[1], e[0]])
        faces.append(tuple(face))

    #This overcounts, have to delete cyclic repeats now:

    sorted_faces = list(set([tuple(canonical_order(graph,x)) for x in faces]))
    cleaned_faces = [ tuple([ y for y in F]) for F in sorted_faces]
    graph.graph["faces"] = cleaned_faces
    return graph

def canonical_order(graph, face):
    '''
    Outputs the coordinates of the nodes of the face in a canonical order
    in particular, the first one is the lex-min.
    You need to use the graph structure to make this work
    '''

    lex_sorted_nodes = sorted(face)
    first_node = lex_sorted_nodes[0]
    cycle_sorted_nodes = [first_node]
    local_cycle = nx.subgraph( graph, face)

    #Compute the second node locally based on angle orientation

    v = first_node
    locations = []
    neighbor_list = list(local_cycle.neighbors(v))
    for w in neighbor_list:
        locations.append(graph.nodes[w]["pos"] - graph.nodes[v]["pos"])
    angles = [float(np.arctan2(x[1], x[0])) for x in locations]
    neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)

    second_node = neighbor_list[0]
    cycle_sorted_nodes.append(second_node)
    ##Now compute a canonical ordering of local_cycle, clockwise, starting
    ##from first_node


    while len(cycle_sorted_nodes) < len(lex_sorted_nodes):

        v = cycle_sorted_nodes[-1]
        neighbor_list = list(local_cycle.neighbors(v))
        neighbor_list.remove(cycle_sorted_nodes[-2])
        cycle_sorted_nodes.append(neighbor_list[0])

    return cycle_sorted_nodes


def delete_copies_up_to_permutation(array):
    '''
    Given an array of tuples, return an array consisting of one representative
    for each element in the orbit of the reordering action.
    '''

    cleaned_array = list(set([tuple(canonical_order(x)) for x in array]))

    return cleaned_array

def face_refine(graph):
    #graph must already have the face data computed
    #this adds a vetex in the middle of each face, and connects that vertex to the edges of that face...

    for face in graph.graph["faces"]:
        graph.add_node(face)
        location = np.array([0,0]).astype("float64")
        for v in face:
            graph.add_edge(face, v)
            location += graph.nodes[v]["pos"].astype("float64")
        graph.nodes[face]["pos"] = location / len(face)
    return graph

def edge_refine(graph):
    edge_list = list(graph.edges())
    for e in edge_list:
        graph.remove_edge(e[0],e[1])
        graph.add_node(str(e))
        location = np.array([0,0]).astype("float64")
        for v in e:
            graph.add_edge(str(e), v)
            location += graph.nodes[v]["pos"].astype("float64")
        graph.nodes[str(e)]["pos"] = location / 2
    return graph

def refine(graph):
    graph = compute_rotation_system(graph)
    graph = compute_face_data(graph)
    graph = face_refine(graph)
    return graph

def depth_k_refine(graph,k):
    graph.name = graph.name + str("refined_depth") + str(k)
    for i in range(k):
        graph = refine(graph)
    return graph

def depth_k_barycentric(graph, k):
    graph.name = graph.name + str("refined_depth") + str(k)
    for i in range(k):
        graph = barycentric_subdivision(graph)
    return graph

def barycentric_subdivision(graph):
    #graph must already have the face data computed
    #this adds a vetex in the middle of each face, and connects that vertex to the edges of that face...
    graph = edge_refine(graph)
    graph = refine(graph)
    return graph


def restricted_planar_dual(graph):
    #computes dual without unbounded face
    graph = compute_rotation_system(graph)
    graph = compute_face_data(graph)
    dual_graph = nx.Graph()
    for face in graph.graph["faces"]:
        dual_graph.add_node(face)
        location = np.array([0,0]).astype("float64")
        for v in face:
            location += graph.nodes[v]["pos"].astype("float64")
        dual_graph.nodes[face]["pos"] = location / len(face)
    ##handle edges
    #Construct incidence table --
    #We use this to efficiently construc the edges in the dual.
    incidence = {}
    for v in graph.nodes():
        incidence[v] = set()
        
    for face in graph.graph["faces"]:
        for v in face:
            incidence[v].add(face)
    
    #print(incidence)
    
    
    for e in graph.edges():
        v = e[0]
        for face1 in incidence[v]:
            for face2 in incidence[v]:
                if face1 != face2:
                    if (e[0] in face1) and (e[1] in face1) and (e[0] in face2) and (e[1] in face2):
                        dual_graph.add_edge(face1, face2)
                        dual_graph.edges[ (face1, face2) ]["original_name"] = e
    return dual_graph



def draw_with_location(graph):
    '''
    draws graph with 'pos' as the xy coordinate of each nodes
    initialized by something like graph.nodes[x]["pos"] = np.array([x[0], x[1]])
    '''
#    for x in graph.nodes():
#        graph.nodes[x]["pos"] = [graph.nodes[x]["X"], graph.nodes[x]["Y"]]

    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 20, width = .5, cmap=plt.get_cmap('jet'))
  
    
def test():  
    graph = nx.grid_graph([4,4])
    for x in graph.nodes():
        graph.nodes[x]["pos"] = x
    dual = restricted_planar_dual(graph)
    draw_with_location(graph)
    draw_with_location(dual)
def always_true(proposal):
    return True
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
def get_spanning_tree_mst(graph):
    for edge in graph.edges:
        graph.edges[edge]["weight"] = random.random()

    spanning_tree = nx.tree.maximum_spanning_tree(
        graph, algorithm="kruskal", weight="weight"
    )
    return spanning_tree
def viz(graph, edge_set, partition):
    values = [1 - int(x in edge_set) for x in graph.edges()]
    color_dictionary = {}
    for x in graph.nodes():
        color = 0
        for block in partition.keys():
            if x in partition[block]:
                color_dictionary[x] = color
            color += 1
        
    node_values = [color_dictionary[x] for x in graph.nodes()]
    f = plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_color = node_values, edge_color = values, width = 4, node_size= 65, font_size = 7)
def distance_from_partition(graph, boundary_edges):
    #General Idea:
    #Goal: Given any face of the original graph, want to calculate its distance to the boundary of the partition.
    #Method: Treat that face as a vertex v of the dual graph D, and then (using above) treat the boundary of the partition of a set of vertices S of the dual graph. 
    # Then calculate distance from v to S in D, using:
    #d = infinity
#   For each s in S:
 #    d = min ( distance_in_D(v,s), d)
    boundary_nodes = set( [x[0] for x in boundary_edges] + [x[1] for x in boundary_edges] )
    for node in graph.nodes():
        if node in boundary_nodes:
            graph.nodes[node]["distance"] = 0
        else:
            graph.nodes[node]["distance"] = np.inf
    for step in range(len(graph.nodes())):
        for node in graph.nodes():
            neighbor_distance = min([graph.nodes[x]["distance"] for x in graph.neighbors(node)]) + 1
            new_distance = min(neighbor_distance, graph.nodes[node]["distance"])
            graph.nodes[node]["distance"] = new_distance
    return graph

def compute_cross_edge(graph,partition):
    cross_list = []
    for n in graph.edges:
        if Partition.crosses_parts(partition,n):
            cross_list.append(n)
    return cross_list
def face_sierpinski_mesh(graph, special_faces):
    #parameters: 
    #graph: graph object that edges will be added to
    #special_faces: list of faces that we want to add node/edges to
    #TODO:k: integer depth parameter for depth of face refinement
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

def remove_undirected_edge(graph, v, u):
    if (v,u) in graph.edges():
        graph.remove_edge(v,u)
        return 
    
    if (u,v) in graph.edges():
        graph.remove_edge(u,v)
        return 
def preprocessing(path_to_json):
    graph = Graph.from_json(path_to_json)
    for node in graph.nodes():
        graph.nodes[node]['pos'] = [ graph.nodes[node]["C_X"], graph.nodes[node]["C_Y"] ]
    graph = duality_cleaning(graph)
    print("making dual")
    dual = restricted_planar_dual(graph)
    print("made dual")
    
    save_fig(graph,"./plots/UnderlyingGraph.png",1)
    
    for node in graph.nodes():
        graph.nodes[node]["population"] = graph.nodes[node]["TOTPOP"]


    return graph, dual

def duality_cleaning(graph):
    #Have to remove bad nodes in order for the duality thing to work properly
    cleanup = True
    while cleanup:
        print("clean up phase")
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
        
        bad_nodes = []
        for v in graph.nodes():
            if graph.degree(v) == 1 or graph.degree(v) == 2:
                bad_nodes.append(v)
        if len(bad_nodes) > 0:
            cleanup = True
        else:
            cleanup = False
    return graph
def smooth_node(graph, v):
    neighbors = list(graph.neighbors(v))
    graph.remove_node(v)
    try:
        graph.add_edge(neighbors[0], neighbors[1])
    except:
        print(neighbors)
    return graph
def save_fig(graph,path, size):
    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 1, width = size, cmap=plt.get_cmap('jet'))
    plt.savefig(path, format='png')
    plt.close()