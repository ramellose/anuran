"""
The utils module contains functions used by other modules for multiprocessing.

"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import networkx as nx
from scipy.special import binom
from random import sample
import numpy as np
import logging.handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _generate_null_parallel(values):
    """
    This function takes a list of networks.
    For each network, a list with length n is generated,
    with each item in the list being a permutation of the original network.
    This is returned as a list of lists with this structure:
    ---List corresponding to each original network (length networks)
        ---List of permutations per original network (length n)
    :param values: Dictionary containing values for generating null models
    :return: Dictionaries containing null models
    """
    try:
        networks = values['networks']
        name = values['name']
        fraction = values['fraction']
        prev = values['prev']
        n = values['n']
        mode = values['mode']
    except KeyError:
        logger.error('Could not unpack dictionary!', exc_info=True)
    nulls = list()
    for i in range(len(networks)):
        network = networks[i]
        nulls.append(list())
        if fraction:
            # all null models need to preserve the same edges
            keep = sample(network.edges, round(len(network.edges) * float(fraction)))
            # create lists to distribute edges over according to core prevalence
            keep_subsets = [[] for x in networks]
            occurrence = round(float(prev) * len(networks))
            for edge in keep:
                indices = sample(range(len(networks)), occurrence)
                for k in indices:
                    keep_subsets[k].append(edge)
            for j in range(len(networks)):
                if mode == 'random':
                    nulls[i].append(_randomize_network(network, keep_subsets[j]))
                elif mode == 'degree':
                    nulls[i].append(_randomize_dyads(network, keep_subsets[j]))
        else:
            for j in range(n):
                if mode == 'random':
                    nulls[i].append(_randomize_network(network, keep=[]))
                elif mode == 'degree':
                    nulls[i].append(_randomize_dyads(network, keep=[]))
    # nested dict with a single entry can be combined into a dict after multiprocessing
    if fraction:
        params = (mode, name, 'core', fraction, prev)
    else:
        params = (mode, name, mode)
    return params, nulls


def _randomize_network(network, keep):
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


def _randomize_dyads(network, keep):
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
            if (dyad[0][0], dyad[1][0]) in null.edges:
                count += 1
                continue
            elif (dyad[1][1], dyad[0][1]) in null.edges:
                count += 1
                continue
            elif dyad[0][0] == dyad[1][0]:
                # if there is a triplet, we can just do a single swap
                if (dyad[0][1], dyad[1][1]) not in null.edges:
                    null.add_edge(dyad[0][1], dyad[1][1], weight=null.edges[dyad[0]]['weight'])
                    null.remove_edge(dyad[0][0], dyad[0][1])
                else:
                    count += 1
                    continue
            elif dyad[0][1] == dyad[1][1]:
                if (dyad[0][0], dyad[1][0]) not in null.edges:
                    null.add_edge(dyad[0][0], dyad[1][0], weight=null.edges[dyad[0]]['weight'])
                    null.remove_edge(dyad[0][0], dyad[0][1])
                else:
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


def _generate_rows(values):
    """
    Generates dictionaries with necessary data for the pandas dataframes.
    While this function should be in set.py, the multiprocessing
    function needs it to be imported from here.

    :param values: Dictionary containing values for new pandas rows
    :return: Pandas dataframe with new rows
    """
    try:
        name = values['name']
        networks = values['networks']
        group = values['group']
        set_operation = values['set operation']
        fraction = values['fraction']
        prev = values['prev']
        sign = values['sign']
        sizes = values['sizes']
    except KeyError:
        logger.error('Could not unpack dictionary!', exc_info=True)
    full_name = name + ' networks'
    if fraction:
        name += ' size: ' + str(fraction) + ' prev:' + str(prev)
    data = list()
    if 'difference' in set_operation:
        data.append({'Network': name,
                     'Group': group,
                     'Network type': full_name,
                     'Conserved fraction': fraction,
                     'Prevalence of conserved fraction': prev,
                     'Set type': 'Difference',
                     'Set size': _difference(networks, sign),
                     'Samples': len(networks)})
    if 'intersection' in set_operation:
        for size in sizes:
            data.append({'Network': name,
                         'Group': group,
                         'Network type': full_name,
                         'Conserved fraction': fraction,
                         'Prevalence of conserved fraction': prev,
                         'Set type': 'Intersection ' + str(size),
                         'Set size': _intersection(networks, float(size), sign),
                         'Set type (absolute)': str(len(networks) * float(size)),
                         'Samples': len(networks)})
    return data


def _difference(networks, sign):
    """
    This function returns the size of the difference for a list of networks.
    If sign is true, edges with unique edge weights are part of the difference
    even though there are other edges with the same interaction partners.

    :param networks:
    :param sign: If true, the difference take sign information into account.
    :return: Size of difference
    """
    diff = list()
    for network in networks:
        for edge in network.edges:
            if sign:
                weight = np.sign(network.edges[edge]['weight'])
                diff.append(edge + (np.sign(network.edges[edge]['weight']),))
            else:
                diff.append(edge)
    unique_edges = 0
    for edge in set(diff):
        if sign:
            count = diff.count(edge) + diff.count((edge[1], edge[0], edge[2]))
        else:
            count = diff.count(edge) + diff.count((edge[1], edge[0]))
        # handles occurrence of reversed edges in list
        if count == 1:
            unique_edges += 1
    return unique_edges


def _intersection(networks, size, sign):
    """
    This function returns a network with the same nodes and edge number as the input network.
    Each edge is swapped via a dyad, so the degree distribution is preserved.
    If sign is true, edges with unique edge weights are counted separately to check
    whether they are part of the intersection.
    I.e. when taking the unsigned intersection of 2 networks,
    edges between the same partners but different weights are included.
    For the signed interaction, such edges are only kept if each network
    contains both the positive and negative edges.

    :param networks: List of input networks
    :param size: Number of networks that an edge needs to be a part of
    :param sign: If true, the difference take sign information into account.
    :return: Randomized network
    """
    matches = list()
    for network in networks:
        for edge in network.edges:
            if sign:
                matches.append(edge + (np.sign(network.edges[edge]['weight']),))
            else:
                matches.append(edge)
    shared_edges = 0
    # remove swapped edges
    edges = set(matches)
    for edge in set(matches):
        if (edge[1], edge[0]) in edges:
            edges.remove((edge[1], edge[0]))
    for edge in edges:
        if sign:
            count = matches.count(edge) + matches.count((edge[1], edge[0], edge[2]))
        else:
            count = matches.count(edge) + matches.count((edge[1], edge[0]))
        # handles occurrence of reversed edges in list
        if count >= round(size * len(networks)) > 1:
            # The edges should be present in a fraction of networks bigger than 0,
            # otherwise intersection size is identical to the difference
            # Should also be bigger than 1 otherwise there is not really an intersection
            shared_edges += 1
    return shared_edges