"""
The functions in this module visualize set sizes and other anuran outputs.
Draw_sets visualizes the set sizes of the  null models and original networks.
Draw_samples shows the distribution of set sizes as the number of networks increases,
for both null models and the input networks.
Draw_centralities plots the upper limit of the confidence interval against the lower limit.
Draw_graphs shows the graph properties for each of the networks used by anuran.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import seaborn as sns


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


def draw_set_differences(data, fp):
    """
    This function accepts a pandas dataframe with 4 columns:
    Interval, Set size, Group, Network.
    The interval is the difference of the intersections.

    The interval is the median for the null model networks.

    The function writes a bar plot of the intervals to path.

    :param data:
    :param fp:
    :return:
    """
    sns.set_style(style="whitegrid")
    interval_dict = {}
    for val in set(data['Interval']):
        startnum = float(val.split('->')[0])
        interval_dict[startnum] = val
    keys = list(interval_dict.keys())
    keys.sort(reverse=False)
    order_intervals = [interval_dict[x] for x in keys]
    fig = sns.catplot(x='Interval', y='Set size', col='Network', color='Gray',
                      data=data, kind='bar', order=order_intervals)
    fig.set_xticklabels(rotation=30)
    fig.savefig(fp + "_setdifferences.png")
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


def draw_graphs(data, fp):
    """
    This function accepts a pandas dataframe
    with 5 columns:
    Network, Name, Group, Network type, Conserved fraction, Property, Value

    :param data: Pandas data frame
    :param fp: Filepath with prefix for name
    :return:
    """
    sns.set_style(style="whitegrid")
    fig = sns.catplot(x='Network', y='Value', col='Property',
                      data=data, kind='strip')
    fig.set_xticklabels(rotation=30)
    fig.savefig(fp + "_graph_properties.png")
    fig.fig.clf()
