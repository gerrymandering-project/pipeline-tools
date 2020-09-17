import time
import json
import random
import traceback
import networkx as nx

from multiprocessing import Pool, Value
from networkx.algorithms import tree
from collections import deque, namedtuple
from functools import partial

import facefinder

from gerrychain import MarkovChain
from gerrychain import Graph
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain import accept
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population, UpperBound)

Cut = namedtuple("Cut", "edge subset")


class PopulatedGraph:
    def __init__(self, graph, populations, ideal_pop, epsilon):
        self.graph = graph
        self.subsets = {node: {node} for node in graph}
        self.population = populations.copy()
        self.ideal_pop = ideal_pop
        self.epsilon = epsilon
        self._degrees = {node: graph.degree(node) for node in graph}

    def __iter__(self):
        return iter(self.graph)

    def degree(self, node):
        return self._degrees[node]

    def contract_node(self, node, parent):
        self.population[parent] += self.population[node]
        self.subsets[parent] |= self.subsets[node]
        self._degrees[parent] -= 1

    def has_ideal_population(self, node):
        return (
            abs(self.population[node] - self.ideal_pop) < self.epsilon * self.ideal_pop
        )

def predecessors(h, root):
    return {a: b for a, b in nx.bfs_predecessors(h, root)}

def find_balanced_edge_cuts(h, choice=random.choice):
    # this used to be greater than 2 but failed on small grids:(
    root = choice([x for x in h if h.degree(x) > 1])
    # BFS predecessors for iteratively contracting leaves
    pred = predecessors(h.graph, root)

    cuts = []
    leaves = deque(x for x in h if h.degree(x) == 1)
    while len(leaves) > 0:
        leaf = leaves.popleft()
        if h.has_ideal_population(leaf):
            cuts.append(Cut(edge=(leaf, pred[leaf]), subset=h.subsets[leaf].copy()))
        # Contract the leaf:
        parent = pred[leaf]
        h.contract_node(leaf, parent)
        if h.degree(parent) == 1 and parent != root:
            leaves.append(parent)
    return cuts

def random_spanning_tree(graph):
    for edge in graph.edges:
        graph.edges[edge]["weight"] = random.random()

    spanning_tree = tree.maximum_spanning_tree(
        graph, algorithm="kruskal", weight="weight"
    )
    return spanning_tree

def bipartition_tree_random(
    graph,
    pop_col,
    pop_target,
    epsilon,
    node_repeats=1,
    spanning_tree=None,
    choice=random.choice,
):
    """This is like :func:`bipartition_tree` except it chooses a random balanced
    cut, rather than the first cut it finds.
    This function finds a balanced 2 partition of a graph by drawing a
    spanning tree and finding an edge to cut that leaves at most an epsilon
    imbalance between the populations of the parts. If a root fails, new roots
    are tried until node_repeats in which case a new tree is drawn.
    Builds up a connected subgraph with a connected complement whose population
    is ``epsilon * pop_target`` away from ``pop_target``.
    Returns a subset of nodes of ``graph`` (whose induced subgraph is connected).
    The other part of the partition is the complement of this subset.
    :param graph: The graph to partition
    :param pop_col: The node attribute holding the population of each node
    :param pop_target: The target population for the returned subset of nodes
    :param epsilon: The allowable deviation from  ``pop_target`` (as a percentage of
        ``pop_target``) for the subgraph's population
    :param node_repeats: A parameter for the algorithm: how many different choices
        of root to use before drawing a new spanning tree.
    :param spanning_tree: The spanning tree for the algorithm to use (used when the
        algorithm chooses a new root and for testing)
    :param choice: :func:`random.choice`. Can be substituted for testing.
    """
    populations = {node: graph.nodes[node][pop_col] for node in graph}

    possible_cuts = []
    if spanning_tree is None:
        spanning_tree = random_spanning_tree(graph)
    restarts = 0
    while len(possible_cuts) == 0:
        if restarts == node_repeats:
            spanning_tree = random_spanning_tree(graph)
            restarts = 0
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = find_balanced_edge_cuts(h, choice=choice)
        restarts += 1

    return choice(possible_cuts).subset


def recursive_tree_part(
    graph, parts, pop_target, pop_col, epsilon, node_repeats=1, method=bipartition_tree_random
):
    """Uses :func:`~gerrychain.tree.bipartition_tree` recursively to partition a tree into
    ``len(parts)`` parts of population ``pop_target`` (within ``epsilon``). Can be used to
    generate initial seed plans or to implement ReCom-like "merge walk" proposals.
    :param graph: The graph
    :param parts: Iterable of part labels (like ``[0,1,2]`` or ``range(4)``
    :param pop_target: Target population for each part of the partition
    :param pop_col: Node attribute key holding population data
    :param epsilon: How far (as a percentage of ``pop_target``) from ``pop_target`` the parts
        of the partition can be
    :param node_repeats: Parameter for :func:`~gerrychain.tree_methods.bipartition_tree` to use.
    :return: New assignments for the nodes of ``graph``.
    :rtype: dict
    """
    flips = {}
    remaining_nodes = set(graph.nodes)
    # We keep a running tally of deviation from ``epsilon`` at each partition
    # and use it to tighten the population constraints on a per-partition
    # basis such that every partition, including the last partition, has a
    # population within +/-``epsilon`` of the target population.
    # For instance, if district n's population exceeds the target by 2%
    # with a +/-2% epsilon, then district n+1's population should be between
    # 98% of the target population and the target population.
    debt = 0

    for part in parts[:-1]:
        min_pop = max(pop_target * (1 - epsilon), pop_target * (1 - epsilon) - debt)
        max_pop = min(pop_target * (1 + epsilon), pop_target * (1 + epsilon) - debt)
        nodes = method(
            graph.subgraph(remaining_nodes),
            pop_col=pop_col,
            pop_target=(min_pop + max_pop) / 2,
            epsilon=(max_pop - min_pop) / (2 * pop_target),
            node_repeats=node_repeats,
        )

        part_pop = 0
        for node in nodes:
            flips[node] = part
            part_pop += graph.nodes[node][pop_col]
        debt += part_pop - pop_target
        remaining_nodes -= nodes

    # All of the remaining nodes go in the last part
    for node in remaining_nodes:
        flips[node] = parts[-1]

    return flips

def saveRunStatistics(statistics, tag):
    """Saves the election statistics of a given list of partitions to a JSON file

    Args:
        statistics (Iterable): Iterable of the election statistics of the partition
        of a run
        tag (String): tag added to filename to identify run
    """
    with open('correlation_exps/run_statistics_{}.json'.format(tag), 'w') as outfile:
        try:
            json.dump(statistics, outfile, indent=2)
        except:
            track = traceback.format_exc()
            print(track)
            print('Unable to save run statistics to file.')

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

    electionDict = {
        'seats' : (lambda x: x[config['ELECTION_NAME']].seats('PartyA')),
        'won' : (lambda x: x[config['ELECTION_NAME']].seats('PartyA')),
        'efficiency_gap' : (lambda x: x[config['ELECTION_NAME']].efficiency_gap()),
        'mean_median' : (lambda x: x[config['ELECTION_NAME']].mean_median()),
        'mean_thirdian' : (lambda x: x[config['ELECTION_NAME']].mean_thirdian()),
        'partisan_bias' : (lambda x: x[config['ELECTION_NAME']].partisan_bias()),
        'partisan_gini' : (lambda x: x[config['ELECTION_NAME']].partisan_gini())
    }

    # Run chain, save each desired statistic, and keep track of cuts. Save most
    # left gerrymandered partition
    statistics = {statistic : [] for statistic in config['ELECTION_STATISTICS']}
    for i, partition in enumerate(chain):
        # Save statistics of partition
        for statistic in config['ELECTION_STATISTICS']:
            statistics[statistic].append(electionDict[statistic](partition))
        if i % 500 == 0:
            print('{}: {}'.format(id, i))
    saveRunStatistics(statistics, id)

def main(config_data, id):
    try:
        timeBeg = time.time()
        print('Experiment', id, 'has begun')
        # Save configuration into global variable
        global config
        config = config_data

        graph = Graph.from_json(config['INPUT_GRAPH_FILENAME'])

        # List of districts in original graph
        parts = list(set([graph.nodes[node][config['ASSIGN_COL']] for node in graph.nodes()]))
        # Ideal population of districts
        ideal_pop = sum([graph.nodes[node][config['POP_COL']] for node in graph.nodes()]) / len(parts)

        election = Election(
                            config['ELECTION_NAME'],
                            {'PartyA': config['PARTY_A_COL'],
                            'PartyB': config['PARTY_B_COL']}
                            )

        updaters = {'population': Tally(config['POP_COL']),
                    'cut_edges': cut_edges,
                    config['ELECTION_NAME'] : election
                    }

        partDict = recursive_tree_part(graph, parts, ideal_pop, config['POP_COL'],
                                       config['EPSILON'], config['NODE_REPEATS'])
        for node in graph.nodes():
            graph.nodes[node][config['ASSIGN_COL']] = partDict[node]
        part = Partition(graph=graph, assignment=config['ASSIGN_COL'], updaters=updaters)
        for len_ in config['RUN_LENGTHS']:
            for num in range(config['RUNS_PER_LEN']):
                run_chain(part, config['CHAIN_TYPE'], len_, ideal_pop, '{}_{}_{}_{}'.format(config['TAG'], id, len_, num))

        print('Experiment {} completed in {} seconds'.format(id, time.time() - timeBeg))

    except Exception as e:
        # Print notification if any experiment fails to complete
        track = traceback.format_exc()
        print(track)
        print('Experiment {} failed to complete after {:.2f} seconds'.format(id, time.time() - timeBeg))

if __name__ == '__main__':
    timeStart = time.time()

    config = {
        "POP_COL" : "population",
        "ASSIGN_COL" : "part",
        "INPUT_GRAPH_FILENAME" : "./json/NC.json",
        "EPSILON" : 0.01,
        "ELECTION_NAME" : "2016_Presidential",
        "PARTY_A_COL" : "EL16G_PR_R",
        "PARTY_B_COL" : "EL16G_PR_D",
        "NODE_REPEATS" : 1,
        "CHAIN_TYPE" : "tree",
        "ELECTION_STATISTICS" : ["seats", "efficiency_gap", "mean_median"],
        "NUM_PROCESSORS" : 3,
        "NUM_EXPERIMENTS" : 10,
        "RUN_LENGTHS" : [10, 25, 50, 100, 150, 250, 500, 750, 1000, 1500, 2500,
                         5000, 7500, 10000, 15000, 20000, 25000, 37500, 50000],
        "RUNS_PER_LEN" : 4,
        "TAG" : "TEST"
    }

    # Run NUM_EXPERIMENTS experiments using NUM_PROCESSORS processors
    pool = Pool(config['NUM_PROCESSORS'])
    for i in range(config['NUM_EXPERIMENTS']):
        pool.apply_async(main, args = (config, str(i)))
    pool.close()
    pool.join()
    print('All experiments completed in {:.2f} seconds'.format(time.time() - timeStart))