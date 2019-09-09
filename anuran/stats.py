"""
The functions in this module take previously calculated properties
and assesses whether it is likely that there are differences between these properties
by reporting p-values.

Three properties can be assessed for significance:
1. Set sizes
2. Centrality scores
3. Graph properties

For all properties, the properties are compared to both the null models
and different groups of networks (if included).
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import pandas as pd
from scipy.stats import normaltest, norm, mannwhitneyu
import numpy as np
from warnings import catch_warnings, simplefilter
from itertools import combinations
from statsmodels.sandbox.stats.multicomp import multipletests

import logging.handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compare_centralities(centralities, mc):
    """
    The centralities dataframe contains a list of all centrality ranks measured across a group of networks.

    The dataframe should have the following columns:
    Node, Network, Group, Network type, Conserved fraction,
    Prevalence of conserved fraction, Centrality, Upper limit,
    Lower limit, Values.

    This function carries out a Mann-Whitney test to test whether the ranks are different across
    the two groups that are being compared.
    This means that it can compare groups with different n.
    Consequently, a group is compared to all networks from a specific group.
    Since it is possible to generate more than one network per original network,
    this means that it is possible to compare ranks for 6 networks to ranks of 6*10 networks.

    :param centralities: Dataframe with centralities
    :param mc: Method for multiple-testing correction
    :return: pandas dataframe with p-values for comparisons
    """
    statsframe = pd.DataFrame(columns=['Node', 'Group', 'Comparison', 'Measure', 'P', 'P.type'])
    # first do comparison to null models
    for group in set(centralities['Group']):
        groupset = centralities[centralities['Group'] == group]
        # split up set in input and null models
        orig = groupset[groupset['Network type'] == 'Input networks']
        nulls = groupset[groupset['Network type'] != 'Input networks']
        for op in set(orig['Centrality']):
            orig_values = orig[orig['Centrality'] == op]
            all_null_values = nulls[nulls['Centrality'] == op]
            # we construct a value range from each network type
            for nulltype in set(all_null_values['Network']):
                null_values = all_null_values[all_null_values['Network'] == nulltype]
                for node in orig_values['Node']:
                    range_1 = [x for x in orig_values[orig_values['Node'] == node]['Values'].iloc[0] if x]
                    range_2 = list()
                    if len(null_values[null_values['Node'] == node]) > 0:
                        range_2 = [x for x in null_values[null_values['Node'] == node]['Values'].iloc[0] if x]
                        # nodes that are in original networks may not be in randomized networks
                    if len(range_1) > 5 and len(range_2) > 5:
                        # comparison is likely to return strange results if there are not enough observations
                        # 5 is actually still too small but warning is in main.py
                        with catch_warnings():
                            simplefilter("ignore")
                            p = mannwhitneyu(range_1, range_2)
                        statsframe = _generate_stat_rows(statsframe, node=node, group=group, comparison=nulltype,
                                                         operation=op, p=p[1], ptype='Mann-Whitney')
    combos = combinations(set(centralities['Group']), 2)
    for combo in combos:
        group1 = centralities[centralities['Group'] == combo[0]]
        group1 = group1[group1['Network'] == 'Input']
        group2 = centralities[centralities['Group'] == combo[1]]
        group2 = group2[group2['Network'] == 'Input']
        for op in set(group1['Centrality']):
            group1_values = group1[group1['Centrality'] == op]
            group2_values = group2[group2['Centrality'] == op]
            for node in group1_values['Node']:
                range_1 = [x for x in group1_values[group1_values['Node'] == node]['Values'].iloc[0] if x]
                range_2 = list()
                if len(group2_values[group2_values['Node'] == node]) > 0:
                    range_2 = [x for x in group2_values[group2_values['Node'] == node]['Values'].iloc[0] if x]
                    # nodes that are in original networks may not be in randomized networks
                if len(range_1) > 5 and len(range_2) > 5:
                    # comparison is likely to return strange results if there are not enough observations
                    # 5 is actually still too small but warning is in main.py
                    with catch_warnings():
                        simplefilter("ignore")
                        p = mannwhitneyu(range_1, range_2)
                    statsframe = _generate_stat_rows(statsframe, node=node, group=combo[0], comparison=combo[1],
                                                     operation=op, p=p[1], ptype='Mann-Whitney')
    # multiple testing correction
    if mc:
        p_adjusted = multipletests(statsframe['P'], method=mc)[1]
        statsframe['P.adj'] = p_adjusted
    statsframe = statsframe.sort_values('P')
    return statsframe


def compare_graph_properties(graph_properties, mc):
    """
    This function takes a dataframe of graph properties.
    Each graph property is compared to other groups with the Mann-Whitney test.

    Takes a pandas dataframe of graph properties with the following columns:
    Network, Group, Network type, Conserved fraction, Prevalence of conserved fraction,
    Property, Value.

    :param graph_properties: Dataframe with graph properties
    :param mc: Method for multiple-testing correction
    :return: pandas dataframe with p-values for comparisons
    """
    statsframe = pd.DataFrame(columns=['Group', 'Comparison', 'Measure', 'P', 'P.type'])
    # first do comparison to null models
    for group in set(graph_properties['Group']):
        groupset = graph_properties[graph_properties['Group'] == group]
        # split up set in input and null models
        orig = groupset[groupset['Network type'] == 'Input networks']
        nulls = groupset[groupset['Network type'] != 'Input networks']
        for op in set(orig['Property']):
            orig_values = orig[orig['Property'] == op]
            all_null_values = nulls[nulls['Property'] == op]
            # we construct a value range from each network type
            for nulltype in set(all_null_values['Network']):
                null_values = all_null_values[all_null_values['Network'] == nulltype]
                range_1 = [x for x in orig_values['Value'] if x]
                range_2 = [x for x in null_values['Value'] if x]
                if len(range_1) > 5 and len(range_2) > 5:
                    # comparison is likely to return strange results if there are not enough observations
                    # 5 is actually still too small but warning is in main.py
                    with catch_warnings():
                        simplefilter("ignore")
                        p = mannwhitneyu(range_1, range_2)
                    statsframe = _generate_stat_rows(statsframe, group=group, comparison=nulltype,
                                                     operation=op, p=p[1], ptype='Mann-Whitney')
    combos = combinations(set(graph_properties['Group']), 2)
    for combo in combos:
        group1 = graph_properties[graph_properties['Group'] == combo[0]]
        group1 = group1[group1['Network'] == 'Input']
        group2 = graph_properties[graph_properties['Group'] == combo[1]]
        group2 = group2[group2['Network'] == 'Input']
        for op in set(group1['Property']):
            group1_values = group1[group1['Property'] == op]
            group2_values = group2[group2['Property'] == op]
            range_1 = [x for x in group1_values['Value'] if x]
            range_2 = [x for x in group2_values['Value'] if x]
            if len(range_1) > 5 and len(range_2) > 5:
                # comparison is likely to return strange results if there are not enough observations
                # 5 is actually still too small but warning is in main.py
                with catch_warnings():
                    simplefilter("ignore")
                    p = mannwhitneyu(range_1, range_2)
                statsframe = _generate_stat_rows(statsframe, group=combo[0], comparison=combo[1],
                                                 operation=op, p=p[1], ptype='Mann-Whitney')
    # multiple testing correction
    if mc:
        p_adjusted = multipletests(statsframe['P'], method=mc)[1]
        statsframe['P.adj'] = p_adjusted
    statsframe = statsframe.sort_values('P')
    return statsframe


def compare_set_sizes(set_sizes, mc):
    """
    This function takes a dataframe of set sizes.
    Each set size is compared against a group (generated from null models);
    the p-value is computed by assessing how many standard deviations the set size is outside
    the distribution calculated from the null models.

    Only the random and degree models, without a core,
    are likely to follow a normal distribution and therefore meet conditions for this test.


    Takes a pandas dataframe of set sizes with the following columns:
    Network, Group, Network type, Conserved fraction, Prevalence of conserved fraction,
    Set type, Set size, Set type (absolute)

    The
    :param set_sizes: Dataframe with set sizes
    :param mc: Method for multiple-testing correction
    :return: pandas dataframe with p-values for comparisons
    """
    statsframe = pd.DataFrame(columns=['Group', 'Comparison', 'Measure', 'P', 'P.type'])
    # first do comparison to null models
    for group in set(set_sizes['Group']):
        groupset = set_sizes[set_sizes['Group'] == group]
        # split up set in input and null models
        orig = groupset[groupset['Network type'] == 'Input networks']
        nulls = groupset[groupset['Network type'] != 'Input networks']
        for op in set(orig['Set type']):
            size = orig[orig['Set type'] == op]['Set size'].iloc[0]
            op_nulls = nulls[nulls['Set type'] == op]
            # we construct a value range from each network type
            for nulltype in set(op_nulls['Network']):
                vals = op_nulls[op_nulls['Network'] == nulltype]['Set size']
                if not any(nulls[nulls['Network'] == nulltype]['Conserved fraction']) and not np.all(vals == 0):
                    # usually, core models do not follow a normal distribution
                    with catch_warnings():
                        simplefilter("ignore")
                        test = normaltest(vals)
                        if test[1] < 0.05:
                            logger.warning('The values do not appear to follow a normal distribution for: ' + nulltype)
                p = _value_outside_range(size, vals)
                statsframe = _generate_stat_rows(statsframe, group=group, comparison=nulltype,
                                                 operation=op, p=p, ptype='Set sizes')
    # multiple testing correction
    if mc:
        p_adjusted = multipletests(statsframe['P'], method=mc)[1]
        statsframe['P.adj'] = p_adjusted
    statsframe = statsframe.sort_values('P')
    return statsframe


def _generate_stat_rows(data, group, comparison, operation, p, ptype, node=None):
    """
    Generates dictionaries with necessary data for the pandas dataframes.
    :param data: Pandas data
    :param group: Name for grouping NetworkX objects
    :param comparison: Network name of comparison
    :param operation: Difference and/or intersection
    :param p: p value
    :param ptype: Type of graph property that is being compared
    :param node: Name of node
    :return: Pandas dataframe with new rows
    """
    new_row = {'Group': group,
               'Comparison': comparison,
               'Measure': operation,
               'P': p,
               'P.type': ptype}
    if node:
        new_row['Node'] = node
    data = data.append(new_row, ignore_index=True)
    return data


def _value_outside_range(value, values):
    """
    Tests whether a value is farther away than 2 standard deviations from the mean (95% CI)
    This assumes a normal distribution and sufficient networks to estimate a CI.


    :param value: Value to test
    :param values: List of values
    :return: P
    """
    if not np.all(values == 0):
        std = np.std(values)
        z = (value - np.mean(values)) / std
        pval = norm.sf(abs(z))**2
    else:
        pval = 1
    return pval
