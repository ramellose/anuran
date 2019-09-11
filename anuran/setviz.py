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
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def _generate_rows(values):
    """
    Generates dictionaries with necessary data for the pandas dataframes.
    While this function should be in set.py, the multiprocessing
    function needs it to be imported from here.

    :param values: Dictionary containing values for new pandas rows
    :return: Pandas dataframe with new rows
    """
    try:
        name = values['Name']
        networks = values['Networks']
        group = values['Group']
        set_operation = values['Set operation']
        fraction = values['Fraction']
        prev = values['Prev']
        sign = values['Sign']
        sizes = values['Sizes']
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