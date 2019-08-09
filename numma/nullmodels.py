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


def generate_null(networks, n, share, mode):
    """
    This function takes a list of networks.
    For each network, a list with length n is generated,
    with each item in the list being a permutation of the original network.
    This is returned as a list of lists with this structure:
    ---List corresponding to each original network (length networks)
        ---List of permutations per original network (length n)
    :param networks: List of input NetworkX objects
    :param n: Number of randomized networks per input network
    :param share: Fraction of conserved interactions
    :param mode: random or degree; whether to preserve the degree distribution
    :return: List of lists with randomized networks
    """
    nulls = list()
    for i in range(len(networks)):
        network = networks[i]
        nulls.append(list())
        # all null models need to preserve the same edges
        keep = sample(network.edges, int(len(network.edges) * share))
        for j in range(n):
            if mode == 'random':
                nulls[i].append(randomize_network(network, keep))
            elif mode == 'degree':
                nulls[i].append(randomize_dyads(network, keep))
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
    Each edge is swapped via a dyad, so the degree distribution is preserved.

    :param network: NetworkX object
    :param keep: List of conserved edges
    :return: Randomized network with preserved degree distribution
    """
    null = nx.Graph().to_undirected()
    null.add_nodes_from(network.nodes)
    null.add_edges_from(network.edges)
    nx.set_edge_attributes(null, nx.get_edge_attributes(network, 'weight'), 'weight')
    # create dictionary of dyad pairs
    deg = {degree: list() for degree in dict(network.degree).values()}
    for node in network.nodes:
        deg[network.degree[node]].append(node)
    # sanity check for enough dyads
    total = 0
    for val in deg:
        total += binom(len(deg[val]), 2)
    if total < len(network) * 0.1:
        logger.warning("The network has too few dyads to generate a useful degree-preserving model!\n"
                       "It may be better to only use the fully randomized model. ")
    # we should carry out twice the number of swaps than the number of nodes with swappable edges
    # this should usually fully randomize the network
    swaps = 2 * (len(network.edges) - len(keep))
    # creates a list of lists with each of the sublists containing nodes with same degree
    swappable_deg = [deg[x] for x in deg if len(deg[x]) > 1]
    timeout = False
    for swap in range(swaps):
        success = False
        count = 0
        while not success and not timeout:
            # samples a set of nodes with swappable edges
            if count > 100:
                logger.warning("The network has too few dyads with distinct edge partners to \n"
                               "generate a useful degree-preserving model!\n"
                               "It may be better to only use the fully randomized model. ")
            dyadset = sample(swappable_deg, 1)[0]
            # samples two nodes that could have edges swapped
            dyad = sample(dyadset, 2)
            n0 = list(nx.neighbors(null, dyad[0]))
            n1 = list(nx.neighbors(null, dyad[0]))
            if dyad[1] in n0:
                n0.remove(dyad[1])
            if dyad[0] in n1:
                n1.remove(dyad[0])
            if len(n0) == 0 or len(n1) == 1:
                count += 1
                continue
            edge_0 = sample(n0, 1)[0]
            edge_1 = sample(n1, 1)[0]
            if (edge_0, dyad[0]) in keep or (dyad[0], edge_0) in keep:
                count += 1
                continue
            elif (edge_1, dyad[1]) in keep or (dyad[1], edge_1) in keep:
                count += 1
                continue
            # check if edge already exists
            elif (edge_1, dyad[0]) in null.edges:
                count += 1
                continue
            elif (edge_0, dyad[1]) in null.edges:
                count += 1
                continue
            elif edge_0 == edge_1:
                count += 1
                continue
            else:
                null.remove_edge(edge_0, dyad[0])
                null.remove_edge(edge_1, dyad[1])
                null.add_edge(edge_1, dyad[0], weight=null.edges[edge_0, dyad[0]])
                null.add_edge(edge_0, dyad[1], weight=null.edges[edge_1, dyad[1]])
                success = True
    return null
