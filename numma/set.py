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


def generate_sizes(networks, random, degree, fractions, perm, sizes):
    """
    This function carries out set operations on all networks provided in
    the network, random and degree lists.
    The random and degree lists are structured as follows:
    ---List of all shared fractions (length fractions)
        ---List corresponding to each original network (length networks)
            ---List of permutations per original network (length n in generate_null)
    The function returns a pandas dataframe with the size of the intersection,
    the type of model and the shared fraction as a separate column.
    The length of the dataset is equal to the number of original networks,
    the number of permuted sets for the random models and the number of permuted sets
    for the degree-preserving model.
    :param networks: List of input networks
    :param random: List of permuted input networks without preserved degree distribution
    :param degree: List of permuted input networks with preserved degree distribution
    :param fractions: List with fractions of shared interactions
    :param perm: Number of sets to take from null models
    :param sizes: Size of intersection to calculate. By default 1 (edge should be in all networks).
    :return: List of lists with set sizes
    """
    # Create empty pandas dataframe
    results = pd.DataFrame(columns=['Network', 'Network type', 'Conserved fraction',
                                    'Set type', 'Set size'])
    results = results.append({'Network': 'Input networks',
                              'Network type': 'Input networks',
                              'Set type': 'Difference',
                              'Set size': difference(networks)}, ignore_index=True)
    for size in sizes:
        results = results.append({'Network': 'Input networks',
                                  'Network type': 'Input networks',
                                  'Set type': 'Intersection ' + str(size),
                                  'Set size': intersection(networks, size)}, ignore_index=True)
    for i in range(len(fractions)):
        for j in range(perm):
            randomperm = [sample(random[i][r], 1)[0] for r in range(len(random[i]))]
            results = results.append({'Network': 'Random ' + str(fractions[i]),
                                      'Network type': 'Random networks',
                                      'Conserved fraction': fractions[i],
                                      'Set type': 'Difference',
                                      'Set size': difference(randomperm)}, ignore_index=True)
            for size in sizes:
                results = results.append({'Network': 'Random ' + str(fractions[i]),
                                          'Network type': 'Random networks',
                                          'Conserved fraction': fractions[i],
                                          'Set type': 'Intersection ' + str(size),
                                          'Set size': difference(randomperm)}, ignore_index=True)
            degreeperm = [sample(degree[i][r], 1)[0] for r in range(len(degree[i]))]
            results = results.append({'Network': 'Degree ' + str(fractions[i]),
                                      'Network type': 'Degree networks',
                                      'Conserved fraction': fractions[i],
                                      'Set type': 'Difference',
                                      'Set size': difference(degreeperm)}, ignore_index=True)
            for size in sizes:
                results = results.append({'Network': 'Degree ' + str(fractions[i]),
                                          'Network type': 'Degree networks',
                                          'Conserved fraction': fractions[i],
                                          'Set type': 'Intersection ' + str(size),
                                          'Set size': difference(degreeperm)}, ignore_index=True)
    return results


def generate_sample_sizes(networks, random, degree, fractions, perm, sizes, limit):
    """
    This function wraps the the generate_sizes function
    but it only gives a random subset of the input networks and null models.
    This shows the effect of increasing sample number on set size.
    :param networks: List of input networks
    :param random: List of permuted input networks without preserved degree distribution
    :param degree: List of permuted input networks with preserved degree distribution
    :param fractions: List with fractions of shared interactions
    :param perm: Number of sets to take from null models
    :param sizes: Size of intersection to calculate. By default 1 (edge should be in all networks).
    :param limit: Maximum number of resamples.
    :return: List of lists with set sizes
    """
    results = pd.DataFrame(columns=['Network', 'Network type', 'Conserved fraction',
                                    'Set type', 'Set size', 'Samples'])
    for i in range(1, len(networks)+1):
        n = binom(len(networks), i)
        if limit:
            if limit < n:
                n = limit
        combos = combinations(range(len(networks)), i)
        combos = sample(list(combos), int(n))
        for item in combos:
            subnetworks = [networks[x] for x in item]
            subrandom = list()
            for frac in range(len(fractions)):
                subrandom.append([random[frac][x] for x in item])
            subdegree = list()
            for frac in range(len(fractions)):
                subdegree.append([degree[frac][x] for x in item])
            subresults = generate_sizes(subnetworks, subrandom, subdegree,
                                        fractions, perm, sizes)
            subresults['Samples'] = i
            results = results.append(subresults)
    return results


def difference(networks):
    """
    This function returns the size of the difference for a list of networks.

    :param networks:
    :return: Size of difference
    """
    diff = list()
    for network in networks:
        diff.extend(list(network.edges))
    unique_edges = 0
    for edge in set(diff):
        count = diff.count(edge) + diff.count((edge[1], edge[0]))
        # handles occurrence of reversed edges in list
        if count == 1:
            unique_edges += 1
    return unique_edges


def intersection(networks, size):
    """
    This function returns a network with the same nodes and edge number as the input network.
    Each edge is swapped via a dyad, so the degree distribution is preserved.

    :param networks: List of input networks
    :param size: Number of networks that an edge needs to be a part of
    :return: Randomized network
    """
    intersection = list()
    for network in networks:
        intersection.extend(list(network.edges))
    shared_edges = 0
    # remove swapped edges
    edges = set(intersection)
    for edge in set(intersection):
        if (edge[1], edge[0]) in edges:
            edges.remove((edge[1], edge[0]))
    for edge in edges:
        count = intersection.count(edge) + intersection.count((edge[1], edge[0]))
        # handles occurrence of reversed edges in list
        if count >= (size * len(networks)):
            shared_edges += 1
    return shared_edges
