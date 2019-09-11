"""
The functions in this module visualize set sizes.
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
from anuran.set import difference, intersection
import logging.handlers

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
                     'Set size': difference(networks, sign),
                     'Samples': len(networks)})
    if 'intersection' in set_operation:
        for size in sizes:
            data.append({'Network': name,
                         'Group': group,
                         'Network type': full_name,
                         'Conserved fraction': fraction,
                         'Prevalence of conserved fraction': prev,
                         'Set type': 'Intersection ' + str(size),
                         'Set size': intersection(networks, float(size), sign),
                         'Set type (absolute)': str(len(networks) * float(size)),
                         'Samples': len(networks)})
    return data
