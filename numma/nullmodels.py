"""
The null models module contains functions for constructing permutations of input networks.
Generation of null models is done on the adjacency matrix for speed;
the NetworkX representation is unfortunately slower.

The functions can either change (random model) or preserve (degree model) the degree distribution.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import networkx as nx
from scipy.special import binom
from random import sample
import logging.handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_null(networks, n, share, mode, core=False):
    """
    This function takes a list of networks.
    For each network, a list with length n is generated,
    with each item in the list being a permutation of the original network.
    This is returned as a list of lists with this structure:
    ---List corresponding to each original network (length networks)
        ---List of permutations per original network (length n)
    :param networks: List of input NetworkX objects
    :param n: Number of randomized networks per input network
    :param mode: random or degree; whether to preserve the degree distribution
    :param share: Fraction of conserved interactions
    :param core: Prevalence of core. If provided, null models have conserved interactions.
    :return: List of lists with randomized networks
    """
    nulls = list()
    for i in range(len(networks)):
        network = networks[i]
        nulls.append(list())
        for j in range(n):
            if mode == 'random':
                nulls[i].append(randomize_network(network, keep=[]))
            elif mode == 'degree':
                nulls[i].append(randomize_dyads(network, keep=[]))
            else:
                logger.error("The null model mode is not recognized.", exc_info=True)
    return nulls


def generate_core(networks, mode, share, core):
    """
    This function takes a list of networks.
    For each network, a list with length n is generated,
    with each item in the list being a permutation of the original network.
    This is returned as a list of lists with this structure:
    ---List corresponding to each original network (length networks)
        ---List of permutations per original network (length networks)
    :param networks: List of input NetworkX objects
    :param mode: random or degree; whether to preserve the degree distribution
    :param share: Fraction of conserved interactions
    :param core: Prevalence of core. If provided, null models have conserved interactions.
    :return: List of lists with randomized networks
    """
    nulls = list()
    for i in range(len(networks)):
        network = networks[i]
        nulls.append(list())
        # all null models need to preserve the same edges
        keep = sample(network.edges, round(len(network.edges) * share))
        # create lists to distribute edges over according to core prevalence
        keep_subsets = [[] for x in networks]
        occurrence = round(core * len(networks))
        for edge in keep:
            indices = sample(range(len(networks)), occurrence)
            for k in indices:
                keep_subsets[k].append(edge)
        for j in range(len(networks)):
            if mode == 'random':
                nulls[i].append(randomize_network(network, keep_subsets[j]))
            elif mode == 'degree':
                nulls[i].append(randomize_dyads(network, keep_subsets[j]))
            else:
                logger.error("The null model mode is not recognized.", exc_info=True)
    return nulls


def randomize_network(network, keep):
    """
    This function returns a network with the same nodes and edge number as the input network.
    However, each edge is placed randomly.

    :param network: NetworkX object
    :param keep: List of conserved edges
    :return: Randomized network
    """
    null = nx.Graph().to_undirected()
    null.add_nodes_from(network.nodes)
    if keep:
        null.add_edges_from(keep)
        nx.set_edge_attributes(null, nx.get_edge_attributes(network, 'weight'), 'weight')
    num = len(network.edges) - len(null.edges)
    randomized_weights = nx.get_edge_attributes(network, 'weight')
    if len(randomized_weights) > 0:
        for edge in null.edges:
            randomized_weights.pop(edge, None)
    randomized_weights = sample(list(randomized_weights.values()), len(randomized_weights))
    for edge in range(num):
        created = False
        while not created:
            new_edge = sample(null.nodes, 2)
            if new_edge not in null.edges:
                null.add_edge(new_edge[0], new_edge[1], weight=randomized_weights[edge])
                created = True
    return null


def randomize_dyads(network, keep):
    """
    This function returns a network with the same nodes and edge number as the input network.
    Each edge is swapped rather than moved, so the degree distribution is preserved.

    :param network: NetworkX object
    :param keep: List of conserved edges
    :return: Randomized network with preserved degree distribution
    """
    null = nx.Graph().to_undirected()
    null.add_nodes_from(network.nodes)
    null.add_edges_from(network.edges)
    nx.set_edge_attributes(null, nx.get_edge_attributes(network, 'weight'), 'weight')
    # we should carry out twice the number of swaps than the number of nodes with swappable edges
    # this should usually fully randomize the network
    swaps = 2 * len(network.edges)
    # creates a list of lists with each of the sublists containing nodes with same degree
    timeout = False
    for swap in range(swaps):
        success = False
        count = 0
        while not success and not timeout:
            # samples a set of nodes with swappable edges
            if count > 100:
                timeout = True
                logger.warning('Could not create good degree-preserving models!')
            dyad = sample(null.edges, 2)
            # samples two nodes that could have edges swapped
            if (dyad[0][0], dyad[1][0]) in null.edges or (dyad[1][0], dyad[0][0]) in null.edges:
                count += 1
                continue
            elif (dyad[1][1], dyad[0][1]) in null.edges or (dyad[0][1], dyad[1][1]) in null.edges:
                count += 1
                continue
            elif dyad[0][0] == dyad[1][0] or dyad[0][1] == dyad[1][1]:
                count += 1
                continue
            elif dyad[0] in keep or dyad[1] in keep:
                count += 1
                continue
            else:
                null.add_edge(dyad[0][0], dyad[1][0], weight=null.edges[dyad[0]]['weight'])
                null.add_edge(dyad[0][1], dyad[1][1], weight=null.edges[dyad[1]]['weight'])
                null.remove_edge(dyad[0][0], dyad[0][1])
                null.remove_edge(dyad[1][0], dyad[1][1])
                success = True
    return null
