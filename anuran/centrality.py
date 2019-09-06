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
from random import sample
from scipy.stats import sem, t
import numpy as np
import os


def generate_ci_frame(networks, random, degree, fractions, core, perm):
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
    :param networks: List of input networks
    :param random: Dictionary with permuted input networks without preserved degree distribution
    :param degree: Dictionary with permuted input networks with preserved degree distribution
    :param fractions: List with fractions of shared interactions
    :param core: List with prevalence of shared interactions
    :param perm: Number of sets to take from null models
    :return: List of lists with set sizes
    """
    # Create empty pandas dataframe
    results = pd.DataFrame(columns=['Node', 'Network', 'Group', 'Network type', 'Conserved fraction',
                                    'Prevalence of conserved fraction',
                                    'Centrality', 'Upper limit', 'Lower limit'])
    for x in networks:
        group = os.path.basename(x)
        results = generate_ci_rows(name='Input', data=results, group=group,
                                   networks=networks[x], fraction=None, prev=None)
        for j in range(perm):
            degreeperm = [sample(degree[x]['degree'][r], 1)[0] for r in range(len(degree[x]['degree']))]
            results = generate_ci_rows(name='Degree', data=results, group=group,
                                       networks=degreeperm, fraction=None, prev=None)
            randomperm = [sample(random[x]['random'][r], 1)[0] for r in range(len(random[x]['random']))]
            results = generate_ci_rows(name='Random', data=results, group=group,
                                       networks=randomperm, fraction=None, prev=None)
        if fractions:
            for frac in fractions:
                for c in core:
                    for network in range(len(networks)):
                        degreeperm = degree[x]['core'][frac][c][network]
                        randomperm = random[x]['core'][frac][c][network]
                        results = generate_ci_rows(name='Degree', data=results, group=group,
                                                   networks=degreeperm, fraction=frac, prev=c)
                        results = generate_ci_rows(name='Random', data=results, group=group,
                                                   networks=randomperm, fraction=frac, prev=c)
        return results


def generate_ci_rows(data, name, group, networks, fraction, prev):
    """
    Generates Pandas rows with all centrality measures for a list of networks.

    :param data: Pandas dataframe
    :param name: Name for the list of NetworkX objects
    :param group: Name for grouping NetworkX objects
    :param networks: List of NetworkX objects
    :param fraction: If a null model with core is provided, adds the core fraction to the row
    :param prev: If a null model with core is provided, adds the core prevalence to the row
    :return: Pandas dataframe with added rows
    """
    full_name = name + ' networks'
    if fraction:
        name += ' size: ' + str(fraction) + ' prev:' + str(prev)
    properties = generate_centralities(networks)
    for property in properties:
        ci = generate_confidence_interval(properties[property])
        for node in ci:
            data = data.append({'Node': node,
                                'Network': name,
                                'Group': group,
                                'Network type': full_name,
                                'Conserved fraction': fraction,
                                'Prevalence of conserved fraction': prev,
                                'Centrality': property,
                                'Upper limit': ci[node][1],
                                'Lower limit': ci[node][0]},
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
            nodes.extend(x.keys())
    ci = dict.fromkeys(set(nodes))
    for node in ci:
        # first constuct array of ranking
        ranks = [_catch(x, node) for x in ranking if _catch(x, node)]
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


def generate_centralities(networks):
    """
    This function constructs lists with centrality rankings of nodes in multiple networks.
    Instead of using the absolute degree or betweenness centrality, this takes metric bias into account.

    :param networks: List of input networks
    :return: Pandas dataframe with rankings
    """
    properties = {x: list() for x in ['Degree', 'Closeness', 'Betweenness']}
    for network in networks:
        properties['Degree'].append(centrality_percentile(nx.degree_centrality(network)))
        properties['Closeness'].append(centrality_percentile(nx.closeness_centrality(network)))
        properties['Betweenness'].append(centrality_percentile(nx.betweenness_centrality(network)))
    return properties


def centrality_percentile(centrality):
    """
    Given a dictionary of centralities, this function returns the percentile score of
    nodes in a graph.

    :param centrality: Dictionary with nodes as keys and centralities as values.
    :return:
    """
    if len(centrality) > 0:
        ranking = pd.DataFrame.from_dict(centrality, orient='index')
        ranking['rank'] = ranking[0].rank(pct=True)
        ranking = ranking['rank'].to_dict()
    else:
        ranking = None
    return ranking


def _catch(dict, key):
    try:
        return dict[key]
    except Exception as e:
        return None

