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
import networkx as nx
from random import sample
from scipy.stats import sem, t, normaltest
import numpy as np
import os

import logging.handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compare_set_sizes(set_sizes):
    """
    Takes a pandas data frame of set sizes with the following columns:
    Network, Group, Network type, Conserved fraction, Prevalence of conserved fraction,
    Set type, Set size, Set type (absolute)
    :param set_sizes:
    :return: pandas dataframe with p-values for comparisons
    """
    statsframe = pd.DataFrame(columns=['Group', 'Comparison', 'Set type', 'P', 'P.type'])
    # first do comparison to null models
    for group in set(set_sizes['Group']):
        groupset = set_sizes[set_sizes['Group'] == group]
        # split up set in input and null models
        input = groupset[groupset['Network type'] == 'Input networks']
        nulls = groupset[groupset['Network type'] != 'Input networks']
        if len(input) != len(set(input['Set type'])):
            logger.error('Problem with internal data integrity!', exc_info=True)
            exit()
        for op in input['Set type']:
            size = input[input['Set type'] == op]['Set size'][0]
            op_nulls = nulls[nulls['Set type'] == op]
            # we construct a value range from each network type
            for nulltype in set(op_nulls['Network']):
                vals = op_nulls[op_nulls['Network'] == nulltype]['Set size']

    # next do between-group comparison


def value_outside_range(value, values):
    """
    Tests whether a value is farther away than 2 standard deviations from the mean (95% CI)
    The test always uses a ran

    :param value: Value to test
    :param values: List of values
    :return: P
    """
