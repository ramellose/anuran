"""
The null models module contains functions for constructing permutations of input networks.
Generation of null models is done on the adjacency matrix for speed;
the NetworkX representation is unfortunately slower.
The functions can either change (random model) or preserve (degree model) the degree distribution.

The functions in this module also calculate intersections or differences of networks.
The first function is a wrapper that
subsamples networks from a list of null models to output a dataframe of set sizes.

These functions run operations in parallel. utils.py contains the operations they carry out.

"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

from anuran.utils import _generate_null_parallel, _get_union
import multiprocessing as mp

import logging.handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_null(networks, n, npos, core, fraction=False, prev=False):
    """
    This function takes a list of networks.
    For each network, a list with length n is generated,
    with each item in the list being a permutation of the original network.
    This is returned as a list of lists with this structure:
    ---List corresponding to each original network (length networks)
        ---List of permutations per original network (length n)
    For the positive controls, this list is inverted:
    ---List of permutations across networks (length n)
        ---List corresponding to a single permuted group of networks
    To generate the list through multiprocessing,
    a dictionary with arguments is generated
    and provided to a utility function.

    :param networks: List of input NetworkX objects
    :param n: Number of randomized networks per input network
    :param npos: Number of positive control randomized networks per group
    :param core: Number of processor cores
    :param fraction: Fraction of conserved interactions
    :param prev: Prevalence of core. If provided, null models have conserved interactions.
    :return: List of lists with randomized networks
    """
    all_results = {'random': {x: {'random': [], 'core': {}} for x in networks},
                   'degree': {x: {'degree': [], 'core': {}} for x in networks}}
    # firt generate list of network models that need to be generated
    all_models = list()
    for x in networks:
        for y in networks[x]:
            all_models.append({'network': y,
                               'networks': len(networks[x]),
                               'name': x,
                               'fraction': None,
                               'prev': None,
                               'n': n,
                               'mode': 'random'})
            all_models.append({'network': y,
                               'networks': len(networks[x]),
                               'name': x,
                               'fraction': None,
                               'prev': None,
                               'n': n,
                               'mode': 'degree'})
        if fraction:
            for frac in fraction:
                all_results['random'][x]['core'][frac] = dict()
                all_results['degree'][x]['core'][frac] = dict()
                # report in logger the edge numbers
                all_edges = _get_union(networks[x])
                core_num = round(len(all_edges) * float(frac))
                logger.info("The " + str(frac) + "core for network group " + x +
                            " contains " + str(core_num) + " core edges out of " + str(all_edges) + "total.")
                for p in prev:
                    all_results['random'][x]['core'][frac][p] = list()
                    all_results['degree'][x]['core'][frac][p] = list()
                    all_models.append({'networks': networks[x],
                                       'network': None,
                                       'name': x,
                                       'fraction': frac,
                                       'prev': p,
                                       'n': npos,
                                       'mode': 'random'})
                    all_models.append({'networks': networks[x],
                                       'network': None,
                                       'name': x,
                                       'fraction': frac,
                                       'prev': p,
                                       'n': npos,
                                       'mode': 'degree'})
    # run size inference in parallel
    pool = mp.Pool(core)
    results = pool.map(_generate_null_parallel, all_models)
    pool.close()
    for result in results:
        # the first tuple in the result section
        # contains the settings:
        # model type, group name, core or not, prevalence and fraction of core
        if len(result[0]) == 3:
            # dict: null model, group name, null type
            all_results[result[0][0]][result[0][1]][result[0][2]].append(result[1])
        else:
            # dict: null model, group name, null type, frac, prev
            all_results[result[0][0]][result[0][1]][result[0][2]][result[0][3]][result[0][4]] = result[1]
    return all_results['random'], all_results['degree']


