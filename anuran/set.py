"""
The functions in this module calculate intersections or differences of networks.
The first function is a wrapper that
subsamples networks from a list of null models to output a dataframe of set sizes.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import pandas as pd
from random import sample
from itertools import combinations
from scipy.special import binom
import numpy as np
import os


def generate_sizes(networks, random, degree, sign,
                   set_operation, fractions, core, perm, sizes):
    """
    This function carries out set operations on all networks provided in
    the network, random and degree lists.
    The random and degree lists are structured as follows:
    ---List corresponding to each original network (length networks)
        ---List of permutations per original network (length n in generate_null)
    The core list is structured as follows:
    ---List of all shared fractions (length fractions)
        ---List corresponding to core prevalence(length core)
            ---List of permutations per original network (length networks)
    The function returns a pandas dataframe with the size of the intersection,
    the type of model and the shared fraction as a separate column.
    The length of the dataset is equal to the number of original networks,
    the number of permuted sets for the random models and the number of permuted sets
    for the degree-preserving model.
    :param networks: List of input networks
    :param random: Dictionary with permuted input networks without preserved degree distribution
    :param degree: Dictionary with permuted input networks with preserved degree distribution
    :param sign: If true, sets take sign information into account.
    :param set_operation: Type of set operation to carry out
    :param fractions: List with fractions of shared interactions
    :param core: List with prevalence of shared interactions
    :param perm: Number of sets to take from null models
    :param sizes: Size of intersection to calculate. By default 1 (edge should be in all networks).
    :return: List of lists with set sizes
    """
    # Create empty pandas dataframe
    results = pd.DataFrame(columns=['Network', 'Group', 'Network type', 'Conserved fraction',
                                    'Prevalence of conserved fraction',
                                    'Set type', 'Set size', 'Set type (absolute)'])
    for x in networks:
        group = os.path.basename(x)
        results = _generate_rows(name='Input', data=results, group=group,
                                networks=networks[x], set_operation=set_operation,
                                sizes=sizes, sign=sign, fraction=None, prev=None)
        for j in range(perm):
            degreeperm = [sample(degree[x]['degree'][r], 1)[0] for r in range(len(degree[x]['degree']))]
            results = _generate_rows(name='Degree', data=results, group=group,
                                    networks=degreeperm, set_operation=set_operation,
                                    sizes=sizes, sign=sign, fraction=None, prev=None)
            randomperm = [sample(random[x]['random'][r], 1)[0] for r in range(len(random[x]['random']))]
            results = _generate_rows(name='Random', data=results, group=group,
                                    networks=randomperm, set_operation=set_operation,
                                    sizes=sizes, sign=sign, fraction=None, prev=None)
        if fractions:
            for frac in fractions:
                for c in core:
                    for network in range(len(networks[x])):
                        degreeperm = degree[x]['core'][frac][c][network]
                        randomperm = random[x]['core'][frac][c][network]
                        results = _generate_rows(name='Degree', data=results, group=group,
                                                networks=degreeperm, set_operation=set_operation,
                                                sizes=sizes, sign=sign, fraction=frac, prev=c)
                        results = _generate_rows(name='Random', data=results, group=group,
                                                networks=randomperm, set_operation=set_operation,
                                                sizes=sizes, sign=sign, fraction=frac, prev=c)
    return results


def generate_sample_sizes(networks, random,
                          degree, sign,
                          set_operation, fractions, core, perm, sizes, limit, number):
    """
    This function wraps the the generate_sizes function
    but it only gives a random subset of the input networks and null models.
    This shows the effect of increasing sample number on set size.
    :param networks: List of input networks
    :param random: List of permuted input networks without preserved degree distribution
    :param degree: List of permuted input networks with preserved degree distribution
    :param sign: If true, sets take sign information into account.
    :param set_operation: Type of set operation to carry out
    :param fractions: List with fractions of shared interactions
    :param core: List with prevalence of shared interactions
    :param perm: Number of sets to take from null models
    :param sizes: Size of intersection to calculate. By default 1 (edge should be in all networks).
    :param limit: Maximum number of resamples.
    :param number: Sample number to test.
    :return: List of lists with set sizes
    """
    results = pd.DataFrame(columns=['Network', 'Network type', 'Conserved fraction',
                                    'Set type', 'Set size', 'Set type (absolute)', 'Samples', ])
    subnetworks = dict()
    subrandom = dict()
    subdegree = dict()
    for x in networks:
        if number:
            seq = [int(x) for x in number]
        else:
            seq = range(1, len(networks[x])+1)
        for i in seq:
            n = binom(len(networks[x]), i)
            if type(limit) == int:
                if limit < n:
                    n = limit
            combos = combinations(range(len(networks[x])), i)
            combos = sample(list(combos), int(n))
            for item in combos:
                subnetworks[x] = [networks[x][y] for y in item]
                subrandom[x] = {'random': [random[x]['random'][y] for y in item]}
                subdegree[x] = {'degree': [degree[x]['degree'][y] for y in item]}
            subrandom[x]['core'] = {}
            subdegree[x]['core'] = {}
            if fractions:
                for frac in fractions:
                    subrandom[x]['core'][frac] = dict()
                    subdegree[x]['core'][frac] = dict()
                    for c in core:
                        subrandom[x]['core'][frac][c] = list()
                        subdegree[x]['core'][frac][c] = list()
                        for n in range(len(item)):
                            selection = [random[x]['core'][frac][c][n][y] for y in item]
                            subrandom[x]['core'][frac][c].append(selection)
                            selection = [degree[x]['core'][frac][c][n][y] for y in item]
                            subdegree[x]['core'][frac][c].append(selection)
            subresults = generate_sizes(networks=subnetworks, random=subrandom, degree=subdegree,
                                        sign=sign, set_operation=set_operation,
                                        fractions=fractions, core=core, perm=perm, sizes=sizes)
            subresults['Samples'] = i
            results = results.append(subresults)
    return results


def _generate_rows(name, data, group, networks, set_operation, sizes, sign, fraction=None, prev=None):
    """
    Generates dictionaries with necessary data for the pandas dataframes.
    :param name: Network name
    :param data: Pandas data
    :param group: Name for grouping NetworkX objects
    :param networks: List of NetworkX objects
    :param set_operation: Difference and/or intersection
    :param sizes: Intersection sizes
    :param sign: If true, sets take sign information into account.
    :param fraction: Fraction of conserved interactions.
    :param prev: Prevalence of conserved interactions.
    :return: Pandas dataframe with new rows
    """
    full_name = name + ' networks'
    if fraction:
        name += ' size: ' + str(fraction) + ' prev:' + str(prev)
    if 'difference' in set_operation:
        data = data.append({'Network': name,
                            'Group': group,
                            'Network type': full_name,
                            'Conserved fraction': fraction,
                            'Prevalence of conserved fraction': prev,
                            'Set type': 'Difference',
                            'Set size': difference(networks, sign)},
                           ignore_index=True)
    if 'intersection' in set_operation:
        for size in sizes:
            data = data.append({'Network': name,
                                'Group': group,
                                'Network type': full_name,
                                'Conserved fraction': fraction,
                                'Prevalence of conserved fraction': prev,
                                'Set type': 'Intersection ' + str(size),
                                'Set size': intersection(networks, float(size), sign),
                                'Set type (absolute)': str(len(networks) * float(size))},
                               ignore_index=True)
    return data


def difference(networks, sign):
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


def intersection(networks, size, sign):
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