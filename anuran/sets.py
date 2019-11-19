"""
The functions in this module construct Pandas dataframes with set sizes for different operations.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import logging.handlers
import pandas as pd
import random
from itertools import combinations
from scipy.special import binom
import os
import multiprocessing as mp
from anuran.utils import _generate_rows

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_sizes(networks, random_models, degree_models, sign,
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
    :param random_models: Dictionary with permuted input networks without preserved degree distribution
    :param degree_models: Dictionary with permuted input networks with preserved degree distribution
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
        combined_networks = _sample_combinations(combos=c, networks=networks,
                                                 random_models=random_models, degree_models=degree_models,
                                                 group=x, fractions=fractions, prev=prev, perm=perm, sign=sign,
                                                 set_operation=set_operation, sizes=sizes)
        # run size inference in parallel
        pool = mp.Pool(core)
        results = pool.map(_generate_rows, combined_networks)
        pool.close()
        for result in results:
            all_results = all_results.append(result, ignore_index=True)
    return all_results


def generate_sample_sizes(networks, random_models,
                          degree_models, sign,
                          set_operation, core, fractions, prev, perm, sizes, limit, number):
    """
    This function wraps the the generate_sizes function
    but it only gives a random subset of the input networks and null models.
    This shows the effect of increasing sample number on set size.

    :param networks: List of input networks
    :param random_models: List of permuted input networks without preserved degree distribution
    :param degree_models: List of permuted input networks with preserved degree distribution
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
            max_num = n
            if type(limit) == int:
                if limit < n:
                    n = limit
            # for sampling random numbers
            # if the number of combinations is small enough,
            # we can use the iterator
            # if not, better to sample networks randomly
            # gives possibility of duplicates
            # however, better than having to store huge iterator as list
            if max_num == n:
                combos = list(combinations(range(len(networks[x])), i))
            else:
                combos = list()
                for j in range(n):
                    combos.append(random.sample(range(len(networks[x])), i))
            all_combinations[x].extend(combos)
    results = generate_sizes(networks=networks, random_models=random_models, degree_models=degree_models, sign=sign,
                             set_operation=set_operation, core=core, fractions=fractions,
                             prev=prev, perm=perm, sizes=sizes, combos=all_combinations)
    return results


def _sample_combinations(networks, random_models, degree_models, group, fractions, prev, perm, sign, set_operation, sizes, combos=None):
    """
    This function generates an iterable containing all information required
    to add new rows to a pandas dataframe.
    This iterable can then be supplied to a multiprocessing pool to speed up calculations.

    :param combos: List of networks to combine
    :param networks: List of networks belonging to group x
    :param random_models: List of randomized null models belonging to group x
    :param degree_models: List of degree-preserving null models belonging to group x
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
        subrandom = {'random': [random_models[group]['random'][y] for y in item]}
        subdegree = {'degree': [degree_models[group]['degree'][y] for y in item]}
        for j in range(perm):
            degreeperm = [random.sample(subdegree['degree'][r], 1)[0] for r in range(len(subdegree['degree']))]
            all_networks.append({'networks': degreeperm,
                                 'name': 'Degree',
                                 'group': os.path.basename(group),
                                 'set operation': set_operation,
                                 'sizes': sizes,
                                 'sign': sign,
                                 'fraction': None,
                                 'prev': None})
            randomperm = [random.sample(subrandom['random'][r], 1)[0] for r in range(len(subrandom['random']))]
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
                        degreeperm = [degree_models[group]['core'][frac][c][n][y] for y in item]
                        all_networks.append({'networks': degreeperm,
                                             'name': 'Degree',
                                             'group': group,
                                             'set operation': set_operation,
                                             'sizes': sizes,
                                             'sign': sign,
                                             'fraction': frac,
                                             'prev': c})
                        randomperm = [random_models[group]['core'][frac][c][n][y] for y in item]
                        all_networks.append({'networks': randomperm,
                                             'name': 'Random',
                                             'group': group,
                                             'set operation': set_operation,
                                             'sizes': sizes,
                                             'sign': sign,
                                             'fraction': frac,
                                             'prev': c})
    return all_networks


