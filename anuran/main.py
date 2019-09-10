#!/usr/bin/env python

"""
anuran: Null models for replicate networks.
The script takes a network as input and uses this to generate null models.
The output of the null models is presented as a csv of set sizes
and a t-test is used to assess whether set sizes are different than expected from the null model.
Detailed explanations are available in the headers of each file.

anuran uses the file extension to import networks.
Generation of null models is done on the adjacency matrix for speed;
the NetworkX representation is unfortunately slower.

The demo data for anuran was downloaded from the following publication:
Meyer, K. M., Memiaghe, H., Korte, L., Kenfack, D., Alonso, A., & Bohannan, B. J. (2018).
Why do microbes exhibit weak biogeographic patterns?. The ISME journal, 12(6), 1404.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import networkx as nx
import sys
import os
import argparse
import glob

import anuran
from anuran.nullmodels import generate_null, generate_core
from anuran.set import generate_sizes, generate_sample_sizes
from anuran.centrality import generate_ci_frame
from anuran.graphvals import generate_graph_frame
from anuran.setviz import draw_sets, draw_samples, draw_centralities
from anuran.stats import compare_set_sizes, compare_centralities, compare_graph_properties
import logging.handlers
from pbr.version import VersionInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# handler to sys.stdout
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


def set_anuran():
    """This parser gets input settings for running anuran.
    It requires an input format that can be read by NetworkX.
    Make sure to include the extension in the input filename
    as this is used to infer the file type."""
    parser = argparse.ArgumentParser(
        description='Run the microbial association network clustering algorithm.'
                    'If --central is added, centrality is calculated. '
                    'Exporting as .cyjs allows for import into Cytoscape with '
                    'a cluster- and phylogeny-informed layout.')
    parser.add_argument('-i', '--input_graphs',
                        dest='graph',
                        help='Locations of input network files. The format is detected based on the extension; \n'
                             'at the moment, .graphml, .txt (weighted edgelist), .gml and .cyjs are accepted. \n'
                             'If you set -i to "demo", a demo dataset will be loaded. \n'
                             'If you want to compare different sets of networks, \n'
                             'specify this by including multiple locations. ',
                        default=None,
                        required=False,
                        nargs='+')
    parser.add_argument('-o', '--output',
                        dest='fp',
                        help='Output filename. Specify full file path without extension.',
                        default=None, required=False)
    parser.add_argument('-set', '--set_type',
                        dest='set',
                        required=False,
                        help='Types of sets to compare. \n'
                             'By default, both the difference and intersection are calculated.',
                        choices=['difference', 'intersection'],
                        nargs='+',
                        default=['difference', 'intersection'])
    parser.add_argument('-size', '--intersection_size',
                        dest='size',
                        required=False,
                        nargs='+',
                        default=[1],
                        help='If specified, associations only shared by a number of networks '
                             'times the specified size fraction are included. \n'
                             'You can specify multiple numbers. By default, the full intersection is calculated.')
    parser.add_argument('-sign', '--edge_sign',
                        dest='sign',
                        required=False,
                        help='If flagged, signs of edge weights are not taken into account. \n'
                             'The set difference then includes edges that have a unique edge sign in one network. \n'
                             'The set intersection then only includes edges that have the same sign across networks.',
                        default=True,
                        action='store_false')
    parser.add_argument('-sample', '--resample',
                        dest='sample',
                        required=False,
                        type=int,
                        help='Resample your networks to observe the impact of increasing sample number. \n'
                             'when you increase the network number up until the total. \n'
                             'Specify an upper limit of resamples, or True if you want all possible resamples. \n'
                             'By default, the upper limit equal to the binomial coefficient of the input networks. \n'
                             'If the limit is higher than this coefficient, all possible combinations are resampled.',
                        default=False)
    parser.add_argument('-n', '--sample_number',
                        dest='number',
                        required=False,
                        nargs='+',
                        default=None,
                        help='If you have a lot of samples, specify the sample numbers to test here. \n'
                             'For example: -n 4 8 12 will test the effect of acquiring 4, 8, and 12 samples. \n'
                             'By default, all sample numbers are tested.')
    parser.add_argument('-cs', '--core_size',
                        dest='cs',
                        required=False,
                        nargs='+',
                        default=False,
                        help='If specified, true positive null models '
                             ' include a set fraction of shared interactions. \n'
                             'You can specify multiple fractions. '
                             'By default, null models have no shared interactions and '
                             'sets are computed for all randomized networks.\n. ')
    parser.add_argument('-prev', '--core_prevalence',
                        dest='prev',
                        required=False,
                        nargs='+',
                        help='Specify the prevalence of the core. \n'
                             'By default, 1; each interaction is present in all models.',
                        default=[1])
    parser.add_argument('-perm', '--permutations',
                        dest='perm',
                        type=int,
                        required=False,
                        help='Number of null models to generate for each input network. \n'
                             'Default: 10. ',
                        default=10)
    parser.add_argument('-nperm', '--permutationsets',
                        dest='nperm',
                        required=False,
                        type=int,
                        help='Number of sets to calculate from the null models. \n'
                             'The total number of possible sets is equal to \n'
                             'the number of null models raised to the number of networks.\n '
                             'This value becomes huge quickly, so a random subset of possible sets is taken.\n '
                             'Default: 50. ',
                        default=50)
    parser.add_argument('-c', '--centrality',
                        dest='centrality',
                        required=False,
                        action='store_true',
                        help='If true, extracts centrality ranking from networks \n'
                             'and compares these to rankings extracted from null models. ',
                        default=False)
    parser.add_argument('-net', '--network',
                        dest='network',
                        required=False,
                        action='store_true',
                        help='If true, extracts network-level properties \n'
                             'and compares these to properties of randomized networks. ',
                        default=False)
    parser.add_argument('-compare', '--compare_networks',
                        dest='comparison',
                        required=False,
                        help='If true, networks in the folders specified by the input parameter \n'
                             'are compared for different emergent properties. ',
                        default=False)
    parser.add_argument('-draw', '--draw_figures',
                        dest='draw',
                        required=False,
                        help='If flagged, draws figures showing the set sizes.',
                        action='store_true',
                        default=False)
    parser.add_argument('-stats', '--statistics',
                        dest='stats',
                        required=False,
                        help='Specify True or a multiple testing correction method to \n'
                             'calculate p-values for comparisons. \n'
                             'The available methods are listed in the docs for statsmodels.stats.multitest, \n'
                             'and include bonferroni, sidak, simes-hochberg, fdr_bh and others. ',
                        choices=[False, True, 'bonferroni', 'sidak', 'holm-sidak', 'holm',
                                 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by',
                                 'fdr_tsbh', 'fdr_tsbky'],
                        default=False)
    parser.add_argument('-version', '--version',
                        dest='version',
                        required=False,
                        help='Version number.',
                        action='store_true',
                        default=False)
    return parser


def main():
    args = set_anuran().parse_args(sys.argv[1:])
    args = vars(args)
    if args['version']:
        info = VersionInfo('anuran')
        logger.info('Version ' + info.version_string())
        exit(0)
    if not args['graph']:
        logger.info('Please give an input location.')
    if args['graph'] != ['demo']:
        networks = {x: list() for x in args['graph']}
        # code for importing from multiple folders
        for location in args['graph']:
            files = [f for f in glob.glob(location + "**/*.graphml", recursive=True)]
            files.extend([f for f in glob.glob(location + "**/*.txt", recursive=True)])
            files.extend([f for f in glob.glob(location + "**/*.gml", recursive=True)])
            for file in files:
                filename = file.split(sep=".")
                extension = filename[len(filename)-1]
                try:
                    if extension == 'graphml':
                        network = nx.read_graphml(file)
                    elif extension == 'txt':
                        network = nx.read_weighted_edgelist(file)
                    elif extension == 'gml':
                        network = nx.read_gml(file)
                    else:
                        logger.warning('Format not accepted. '
                                       'Please specify the filename including extension (e.g. test.graphml).', exc_info=True)
                        exit()
                    # need to make sure the graphml function does not arbitrarily assign node ID
                    try:
                        if 'name' in network.nodes[list(network.nodes)[0]]:
                            if network.nodes[list(network.nodes)[0]]['name'] != list(network.nodes)[0]:
                                network = nx.relabel_nodes(network, nx.get_node_attributes(network, 'name'))
                    except IndexError:
                        logger.warning('One of the imported networks contains no nodes.', exc_info=True)
                    networks[location].append(nx.to_undirected(network))
                except Exception:
                    logger.error('Could not import network file!', exc_info=True)
                    exit()
    elif args['graph'] == ['demo']:
        networks = {'demo': list()}
        path = os.path.dirname(anuran.__file__)
        networks['demo'].append(nx.read_graphml(path + '//data//conet_family_a.graphml'))
        networks['demo'].append(nx.read_graphml(path + '//data//conet_family_b.graphml'))
        networks['demo'].append(nx.read_graphml(path + '//data//conet_family_c.graphml'))
    logger.info('Imported ' + str(len(networks)) + ' group(s) of networks.')
    for network in networks:
        if len(networks[network]) < 20:
            logger.warning('One of the groups (' + network +
                           ') does not contain enough networks '
                           'to generate robust tests for centralities or set sizes. \n'
                           'Suppressing warnings, but please be careful with the statistics! \n'
                           'Preferably use groups with at least 20 networks. ')
    # first generate null models
    random = {x: {'random': [], 'core': {}} for x in networks}
    try:
        for x in networks:
            random[x]['random'] = generate_null(networks[x], n=args['perm'], share=0, mode='random')
            if args['cs']:
                for frac in args['cs']:
                    random[x]['core'][frac] = dict()
                    for core in args['prev']:
                        random[x]['core'][frac][core] = generate_core(networks[x],
                                                                      share=float(frac), mode='random',
                                                                      core=float(core))
                logger.info('Finished constructing all randomized networks.')
    except Exception:
        logger.error('Could not generate randomized null models!', exc_info=True)
        exit()
    degree = {x: {'degree': [], 'core': {}} for x in networks}
    try:
        for x in networks:
            degree[x]['degree'] = generate_null(networks[x], n=args['perm'], share=0, mode='degree')
            if args['cs']:
                for frac in args['cs']:
                    degree[x]['core'][frac] = dict()
                    for core in args['prev']:
                        degree[x]['core'][frac][core] = generate_core(networks[x],
                                                                      share=float(frac), mode='degree',
                                                                      core=float(core))
            logger.info('Finished constructing all degree-preserving randomized networks.')
    except Exception:
        logger.error('Could not generate degree-preserving null models! '
                     'Try increasing the conserved fraction. ', exc_info=True)
        exit()
    set_sizes = None
    try:
        set_sizes = generate_sizes(networks, random, degree,
                                   sign=args['sign'], set_operation=args['set'],
                                   fractions=args['cs'], core=args['prev'],
                                   perm=args['nperm'], sizes=args['size'])
        set_sizes.to_csv(args['fp'] + '_sets.csv')
        logger.info('Set sizes exported to: ' + args['fp'] + '_sets.csv')
    except Exception:
        logger.error('Failed to calculate set sizes!', exc_info=True)
        exit()
    if args['centrality']:
        try:
            centralities = generate_ci_frame(networks, random, degree,
                                             fractions=args['cs'], core=args['prev'])
            centralities.to_csv(args['fp'] + '_centralities.csv')
        except Exception:
            logger.error('Could not rank centralities!', exc_info=True)
            exit()
    if args['network']:
        try:
            graph_properties = generate_graph_frame(networks, random, degree,
                                                    fractions=args['cs'], core=args['prev'])
            graph_properties.to_csv(args['fp'] + 'graph_properties.csv')
        except Exception:
            logger.error('Could not estimate graph properties!', exc_info=True)
            exit()
    samples = None
    if args['sample']:
        try:
            samples = generate_sample_sizes(networks, random, degree,
                                            sign=args['sign'], set_operation=args['set'],
                                            fractions=args['cs'], perm=args['nperm'], core=args['prev'],
                                            sizes=args['size'], limit=args['sample'], number=args['number'])
            samples.to_csv(args['fp'] + '_subsampled_sets.csv')
            logger.info('Subsampled set sizes exported to: ' + args['fp'] + '_subsampled_sets.csv')
        except Exception:
            logger.error('Failed to subsample networks!', exc_info=True)
            exit()
    if args['stats']:
        # add code for pvalue estimation
        set_stats = compare_set_sizes(set_sizes, mc=args['stats'])
        set_stats.to_csv(args['fp'] + '_set_stats.csv')
        if args['centrality']:
            central_stats = compare_centralities(centralities, mc=args['stats'])
            central_stats.to_csv(args['fp'] + '_centrality_stats.csv')
        if args['network']:
            graph_stats = compare_graph_properties(graph_properties, mc=args['stats'])
            graph_stats.to_csv(args['fp'] + '_graph_stats.csv')
    if args['draw']:
        try:
            for x in networks:
                subset_sizes = set_sizes[set_sizes['Group'] == x]
                draw_sets(set_sizes, args['fp'] + '_' + x)
                if args['centrality']:
                    subset_centralities = centralities[centralities['Group'] == x]
                    draw_centralities(subset_centralities, args['fp'] + '_' + x)
                if args['sample']:
                    subset_samples = samples[samples['Group'] == x]
                    draw_samples(subset_samples, args['fp'] + '_' + x)
        except Exception:
            logger.error('Could not draw data!', exc_info=True)
            exit()
    logger.info('anuran completed all tasks.')
    exit(0)


if __name__ == '__main__':
    main()
