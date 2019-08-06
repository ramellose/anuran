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
import networkx as nx
import numpy as np


def generate_sizes(networks, random, degree, fractions, perm, size):
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
    :param fractions:
    :return: List of lists with set sizes
    """


def generate_difference(networks):
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


def generate_intersection(networks, size):
    """
    This function returns a network with the same nodes and edge number as the input network.
    Each edge is swapped via a dyad, so the degree distribution is preserved.

    :param network:
    :return: Randomized network
    """
    pass
