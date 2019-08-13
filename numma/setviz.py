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

