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
from scipy.stats import sem, t
import numpy as np
import os
import multiprocessing as mp
from anuran.utils import _generate_centralities_parallel


def generate_ci_frame(networks, random, degree, fractions, prev, perm, core):
    """
    This function estimates centralities from all networks provided in
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

    'None' values reflect that the species in question was not found in a network.

    :param networks: List of input networks
    :param random: Dictionary with permuted input networks without preserved degree distribution
    :param degree: Dictionary with permuted input networks with preserved degree distribution
    :param fractions: List with fractions of shared interactions
    :param prev: List with prevalence of shared interactions
    :param perm: Number of sets to take from null models
    :param core: Number of processor cores
    :return: List of lists with set sizes
    """
    # Create empty pandas dataframe
    results = pd.DataFrame(columns=['Node', 'Network', 'Group', 'Network type', 'Conserved fraction',
                                    'Prevalence of conserved fraction',
                                    'Centrality', 'Upper limit', 'Lower limit', 'Values'])
    for x in networks:
        group = os.path.basename(x)
        obs_networks = _generate_centralities_parallel(networks[x])
        results = _generate_ci_rows(name='Input', data=results, group=group,
                                    networks=obs_networks, fraction=None, prev=None)
        # to reduce computational time, use lookup table for centralities
        # we add a third value to the tuple: centrality scores for the networks
        # run centrality calculations in parallel
        pool = mp.Pool(core)
        degree_centralities = pool.map(_generate_centralities_parallel, degree[x]['degree'])
        pool.close()
        pool = mp.Pool(core)
        random_centralities = pool.map(_generate_centralities_parallel, random[x]['random'])
        pool.close()
        for i in range(perm):
            degreeperm = [sample(degree_centralities[r], 1)[0] for r in range(len(degree_centralities))]
            results = _generate_ci_rows(name='Degree', data=results, group=group,
                                        networks=degreeperm, fraction=None, prev=None)
            randomperm = [sample(random_centralities[r], 1)[0] for r in range(len(random_centralities))]
            results = _generate_ci_rows(name='Random', data=results, group=group,
                                        networks=randomperm, fraction=None, prev=None)
        if fractions:
            for frac in fractions:
                for c in prev:
                    for network in range(len(networks)):
                        degreeperm = degree[x]['core'][frac][c][network]
                        degreeperm = _generate_centralities_parallel(degreeperm)
                        randomperm = random[x]['core'][frac][c][network]
                        randomperm = _generate_centralities_parallel(randomperm)
                        results = _generate_ci_rows(name='Degree', data=results, group=group,
                                                    networks=degreeperm, fraction=frac, prev=c)
                        results = _generate_ci_rows(name='Random', data=results, group=group,
                                                    networks=randomperm, fraction=frac, prev=c)
    return results


def _generate_ci_rows(data, name, group, networks, fraction, prev):
    """
    Generates Pandas rows with all centrality measures for a list of networks.

    :param data: Pandas dataframe
    :param name: Name for the list of NetworkX objects
    :param group: Name for grouping NetworkX objects
    :param networks: List of NetworkX objects as tuples:
    first item is network name, second NetworkX, third centrality dictionary
    :param fraction: If a null model with core is provided, adds the core fraction to the row
    :param prev: If a null model with core is provided, adds the core prevalence to the row
    :return: Pandas dataframe with added rows
    """
    full_name = name + ' networks'
    if fraction:
        name += ' size: ' + str(fraction) + ' prev:' + str(prev)
    properties = networks[0][2].keys() # gets all centrality names
    for centrality in properties:
        centrality_scores = [(networks[i][0], networks[i][2][centrality]) for i in range(len(networks))]
        ci = generate_confidence_interval(centrality_scores)
        for node in ci:
            data = data.append({'Node': node,
                                'Network': name,
                                'Group': group,
                                'Network type': full_name,
                                'Conserved fraction': fraction,
                                'Prevalence of conserved fraction': prev,
                                'Centrality': centrality,
                                'Upper limit': ci[node][1],
                                'Lower limit': ci[node][0],
                                'Values': [(x[0], _catch(x[1], node)) for
                                           x in centrality_scores if _catch(x[1], node)]},
                               ignore_index=True)
    return data


def generate_confidence_interval(ranking):
    """
    Given a list with centrality rankings calculated from multiple networks,
    this function calculates the confidence interval.

    :param ranking: List of centrality rankings for each network
    :return: Dictionary with nodes as keys and tuples of confidence intervals as values
    """
    nodes = list()
    for x in ranking:
        if x:
            nodes.extend(x[1].keys())
    ci = dict.fromkeys(set(nodes))
    for node in ci:
        # first constuct array of ranking
        ranks = [_catch(x[1], node) for x in ranking if _catch(x[1], node)]
        if len(ranks) == 1:
            ci[node] = (np.nan, np.nan)
        else:
            mean, se, m = np.mean(ranks), sem(ranks), t.ppf((1 + 0.95) / 2., len(ranks) - 1)
            interval = (mean - m*se, mean + m*se)
            # confidence intervals below 0 or above 1 are meaningless
            if interval[0] < 0:
                interval = (0, interval[1])
            if interval[1] > 1:
                interval = (interval[0], 1)
            ci[node] = interval
    return ci



def _catch(dictionary, key):
    try:
        return dictionary[key]
    except Exception:
        return None


