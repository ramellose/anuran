"""
The functions in this module visualize set sizes and contain utilities for set.py.
Draw_sets visualizes the 95% confidence intervals of null models and
shows whether the input networks lie outside or inside these confidence intervals.
Draw_samples shows the distribution of set sizes as the number of networks increases,
for both null models and the input networks.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import seaborn as sns
import logging.handlers
import pandas as pd
from random import sample
from itertools import combinations
from scipy.special import binom
import os
import multiprocessing as mp
from anuran.utils import _generate_rows

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_sizes(networks, random, degree, sign,
                   set_operation, core, fractions, prev, perm, sizes, combos=None):
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
    :param set_operation: Type of set operation to carry outo
    :param core: Number of processor cores
    :param fractions: List with fractions of shared interactions
    :param prev: List with prevalence of shared interactions
    :param perm: Number of sets to take from null models
    :param sizes: Size of intersection to calculate. By default 1 (edge should be in all networks).
    :param combos: Dictionary of networks to combine per network
    :return: List of lists with set sizes
    """
    # Create empty pandas dataframe
    all_results = pd.DataFrame(columns=['Network', 'Group', 'Network type', 'Conserved fraction',
                                        'Prevalence of conserved fraction',
                                        'Set type', 'Set size', 'Set type (absolute)'])
    for x in networks:
        if combos:
            c = combos[x]
        else:
            c = None
        combined_networks = _sample_combinations(combos=c, networks=networks, random=random, degree=degree,
                                                 group=x, fractions=fractions, prev=prev, perm=perm, sign=sign,
                                                 set_operation=set_operation, sizes=sizes)
        # run size inference in parallel
        pool = mp.Pool(core)
        results = pool.map(_generate_rows, combined_networks)
        pool.close()
        for result in results:
            all_results = all_results.append(result, ignore_index=True)
    return all_results


def generate_sample_sizes(networks, random,
                          degree, sign,
                          set_operation, core, fractions, prev, perm, sizes, limit, number):
    """
    This function wraps the the generate_sizes function
    but it only gives a random subset of the input networks and null models.
    This shows the effect of increasing sample number on set size.

    :param networks: List of input networks
    :param random: List of permuted input networks without preserved degree distribution
    :param degree: List of permuted input networks with preserved degree distribution
    :param sign: If true, sets take sign information into account.
    :param set_operation: Type of set operation to carry out
    :param core: Number of processor cores
    :param fractions: List with fractions of shared interactions
    :param prev: List with prevalence of shared interactions
    :param perm: Number of sets to take from null models
    :param sizes: Size of intersection to calculate. By default 1 (edge should be in all networks).
    :param limit: Maximum number of resamples.
    :param number: Sample number to test.
    :return: List of lists with set sizes
    """
    all_combinations = dict.fromkeys(networks.keys())
    for x in networks:
        if number:
            seq = [int(x) for x in number]
        else:
            seq = range(1, len(networks[x])+1)
        all_combinations[x] = []
        for i in seq:
            n = binom(len(networks[x]), i)
            if type(limit) == int:
                if limit < n:
                    n = limit
            combos = combinations(range(len(networks[x])), i)
            all_combinations[x].extend(sample(list(combos), int(n)))
            # generate list of combinations
        # iterables are all groups of networks
        # this can be a huge file in memory, so be careful!
        # maybe run in separate iterations
    results = generate_sizes(networks=networks, random=random, degree=degree, sign=sign,
                             set_operation=set_operation, core=core, fractions=fractions,
                             prev=prev, perm=perm, sizes=sizes, combos=all_combinations)
    return results


def _sample_combinations(networks, random, degree, group, fractions, prev, perm, sign, set_operation, sizes, combos=None):
    """
    This function generates an iterable containing all information required
    to add new rows to a pandas dataframe.
    This iterable can then be supplied to a multiprocessing pool to speed up calculations.

    :param combos: List of networks to combine
    :param networks: List of networks belonging to group x
    :param random: List of randomized null models belonging to group x
    :param degree: List of degree-preserving null models belonging to group x
    :param group: Name of group x
    :param fractions: Size of core
    :param prev: Prevalence of core
    :param perm: Number of sets to take from null models
    :param sign: If true, sets take sign information into account.
    :param set_operation: Type of set operation to carry out
    :param sizes: Size of intersection to calculate. By default 1 (edge should be in all networks).
    :return:
    """
    all_networks = list()
    if not combos:
        combos = [tuple([network for network in range(len(networks[group]))])]
    for item in combos:
        subnetworks = [networks[group][y] for y in item]
        all_networks.append({'networks': subnetworks,
                             'name': 'Input',
                             'group': group,
                             'set operation': set_operation,
                             'sizes': sizes,
                             'sign': sign,
                             'fraction': None,
                             'prev': None})
        subrandom = {'random': [random[group]['random'][y] for y in item]}
        subdegree = {'degree': [degree[group]['degree'][y] for y in item]}
        for j in range(perm):
            degreeperm = [sample(subdegree['degree'][r], 1)[0] for r in range(len(subdegree['degree']))]
            all_networks.append({'networks': degreeperm,
                                 'name': 'Degree',
                                 'group': os.path.basename(group),
                                 'set operation': set_operation,
                                 'sizes': sizes,
                                 'sign': sign,
                                 'fraction': None,
                                 'prev': None})
            randomperm = [sample(subrandom['random'][r], 1)[0] for r in range(len(subrandom['random']))]
            all_networks.append({'networks': randomperm,
                                 'name': 'Random',
                                 'group': group,
                                 'set operation': set_operation,
                                 'sizes': sizes,
                                 'sign': sign,
                                 'fraction': None,
                                 'prev': None})
        subrandom['core'] = {}
        subdegree['core'] = {}
        if fractions:
            for frac in fractions:
                subrandom['core'][frac] = dict()
                subdegree['core'][frac] = dict()
                for c in prev:
                    subrandom['core'][frac][c] = list()
                    subdegree['core'][frac][c] = list()
                    for n in range(len(networks[group])):
                        degreeperm = [degree[group]['core'][frac][c][n][y] for y in item]
                        all_networks.append({'networks': degreeperm,
                                             'name': 'Degree',
                                             'group': group,
                                             'set operation': set_operation,
                                             'sizes': sizes,
                                             'sign': sign,
                                             'fraction': frac,
                                             'prev': c})
                        randomperm = [random[group]['core'][frac][c][n][y] for y in item]
                        all_networks.append({'networks': randomperm,
                                             'name': 'Random',
                                             'group': group,
                                             'set operation': set_operation,
                                             'sizes': sizes,
                                             'sign': sign,
                                             'fraction': frac,
                                             'prev': c})
    return all_networks


def draw_sets(data, fp):
    """
    This function accepts a pandas dataframe
    with 5 columns:
    Network, Network type, Conserved fraction, Set type, Set size
    For every combination of set type a faceted box and whiskers plot is generated
    that visualizes the distribution of set sizes per network type.

    :param data: Pandas data frame
    :param fp: Filepath with prefix for name
    :return:
    """
    data['Set size'] = data['Set size'].astype(float)
    sns.set_style(style="whitegrid")
    fig = sns.catplot(x='Network', y='Set size', col='Set type',
                      data=data, kind='strip')
    fig.set_xticklabels(rotation=30)
    fig.savefig(fp + "_setsizes.png")
    fig.fig.clf()


def draw_centralities(data, fp):
    """
    This function accepts a pandas dataframe
    with 5 columns:
    Node, Network, Network type, Conserved fraction, Centrality, Upper limit, Lower limit
    For every centrality a scatter plot is generated with the upper- and lower limits
    on the x and y axes respectively.

    :param data: Pandas data frame
    :param fp: Filepath with prefix for name
    :return:
    """
    sns.set_style(style="whitegrid")
    degree = data[data['Centrality'] == 'Degree']
    fig = sns.relplot(x='Lower limit', y='Upper limit', col='Network',
                      hue='Network', data=degree)
    fig.set(ylim=(0, 1), xlim=(0, 1))
    fig.savefig(fp + "_degree.png")
    fig.fig.clf()
    degree = data[data['Centrality'] == 'Betweenness']
    fig = sns.relplot(x='Lower limit', y='Upper limit', col='Network',
                      hue='Network', data=degree)
    fig.set(ylim=(0, 1), xlim=(0, 1))
    fig.savefig(fp + "_betweenness.png")
    fig.fig.clf()
    degree = data[data['Centrality'] == 'Closeness']
    fig = sns.relplot(x='Lower limit', y='Upper limit', col='Network',
                      hue='Network', data=degree)
    fig.set(ylim=(0, 1), xlim=(0, 1))
    fig.savefig(fp + "_closeness.png")
    fig.fig.clf()


def draw_samples(data, fp):
    """
    This function accepts a pandas dataframe
    with 6 columns:
    Network, Network type, Conserved fraction, Set type, Set size
    For every combination of set type a faceted box and whiskers plot is generated
    that visualizes the distribution of set sizes per network type.

    :param data: Pandas data frame
    :param fp: Filepath with prefix for name
    :return:
    """
    data['Set size'] = data['Set size'].astype(float)
    data['Samples'] = data['Samples'].astype(int)
    for val in set(data['Set type']):
        subdata = data[data['Set type'] == val]
        sns.set_style(style="whitegrid")
        fig = sns.lineplot(x='Samples', y='Set size', hue='Network',
                           data=subdata)
        fig.set_xticks(range(1, max(subdata['Samples']) + 1))
        fig.figure.savefig(fp + "_" + val.replace(' ', '_') + "_samples.png")
        fig.clear()
