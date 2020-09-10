import random
import math


def find_max_dist(graph):
    max_dist = 0
    for node in graph.nodes():
        dist = graph.nodes[node]['distance']
        if dist > max_dist:
            max_dist = dist
    return max_dist


def assign_special_faces_random(graph):
    max_dist = max(graph.nodes[node]["distance"] for node in graph.nodes())
    special_faces = []
    for node in graph.nodes():
        prob = graph.nodes[node]['distance'] / max_dist
        if random.uniform(0, 1) < prob:
            special_faces.append(node)
    return special_faces


def assign_special_faces_sqrt(graph):
    max_dist = max(graph.nodes[node]['distance'] for node in graph.nodes())
    special_faces = []
    for node in graph.nodes():
        prob = math.sqrt(graph.nodes[node]['distance'] / max_dist)
        if random.uniform(0, 1) < prob:
            special_faces.append(node)
    return special_faces


def assign_special_faces_quadratic(graph):
    max_dist = max(graph.nodes[node]['distance'] for node in graph.nodes())
    special_faces = []
    for node in graph.nodes():
        prob = math.pow(graph.nodes[node]['distance'] / max_dist, 2)
        if random.uniform(0, 1) < prob:
            special_faces.append(node)
    return special_faces


