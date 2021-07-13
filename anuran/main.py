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
from multiprocessing import cpu_count
from pbr.version import VersionInfo
import logging.handlers

import anuran
from anuran.utils import _intersection, _construct_intersection
from anuran.nulls import generate_null
from anuran.sets import generate_sizes, generate_sample_sizes, generate_size_differences
from anuran.draw import draw_sets, draw_samples, draw_centralities, \
    draw_graphs, draw_set_differences
from anuran.centrality import generate_ci_frame
from anuran.graphvals import generate_graph_frame
from anuran.stats import compare_set_sizes, compare_centralities, compare_graph_properties, \
    correlate_centralities, correlate_graph_properties

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
                        help='Number of negative control networks to generate for each input network. \n'
                             'Default: 10. ',
                        default=20)
    parser.add_argument('-gperm', '--group_permutations',
                        dest='gperm',
                        type=int,
                        required=False,
                        help='Number of positive control networks to generate for each group of networks. \n'
                             'Default: 10. ',
                        default=10)
    parser.add_argument('-nperm', '--permutationsets',
                        dest='nperm',
                        required=False,
                        type=int,
                        help='Number of sets, centralities and graph values to calculate from the null models. \n'
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
                             'and compares these to rankings extracted from null models. \n'
                             'WARNING: For larger numbers of permutations, the centrality calculations can \n'
                             'become very slow! ',
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
                        choices=['True', 'bonferroni', 'sidak', 'holm-sidak', 'holm',
                                 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by',
                                 'fdr_tsbh', 'fdr_tsbky'],
                        default='fdr_bh')
    parser.add_argument('-core', '-processor_cores',
                        dest='core',
                        type=int,
                        required=False,
                        help='Number of processing cores to use. \n '
                             'By default, CPU count - 1. ',
                        default=cpu_count()-1)
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
        sys.exit(0)
    if not args['graph']:
        logger.info('Please give an input location.')
    if not args['fp']:
        logger.info('No file path given, writing to current directory.')
        args['fp'] = os.getcwd() + '/'
    if args['graph'] != ['demo']:
        networks = {}
        for x in args['graph']:
            if len(os.path.basename(x)) == 0:
                name = 'anuran'
            else:
                name = os.path.basename(x)
            networks[name] = list()
        new_graph = []
        for location in args['graph']:
            if not os.path.isdir(location):
                if os.path.isdir(os.getcwd() + '/' + location):
                    new_graph.append(os.getcwd() + '/' + location)
                else:
                    logger.error('Could not find the specified directory. Is your file path correct?')
                    sys.exit()
            else:
                new_graph.append(location)
        args['graph'] = new_graph
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
                        logger.warning('Ignoring file with wrong format.', exc_info=True)
                        network = False
                    # need to make sure the graphml function does not arbitrarily assign node ID
                    if network:
                        try:
                            if 'name' in network.nodes[list(network.nodes)[0]]:
                                if network.nodes[list(network.nodes)[0]]['name'] != list(network.nodes)[0]:
                                    network = nx.relabel_nodes(network, nx.get_node_attributes(network, 'name'))
                        except IndexError:
                            logger.warning('One of the imported networks contains no nodes.', exc_info=True)
                        networks[os.path.basename(location)].append((os.path.basename(file), nx.to_undirected(network)))
                except Exception:
                    logger.error('Could not import network file!', exc_info=True)
                    sys.exit()
    elif args['graph'] == ['demo']:
        networks = {'demo': list()}
        path = os.path.dirname(anuran.__file__)
        networks['demo'].append(('conet_family_a.graphml', nx.read_graphml(path + '//data//conet_family_a.graphml')))
        networks['demo'].append(('conet_family_b.graphml', nx.read_graphml(path + '//data//conet_family_b.graphml')))
        networks['demo'].append(('conet_family_c.graphml', nx.read_graphml(path + '//data//conet_family_c.graphml')))
    logger.info('Imported ' + str(len(networks)) + ' group(s) of networks.')
    for network in networks:
        if len(networks[network]) < 20:
            logger.warning('One of the groups (' + network +
                           ') does not contain enough networks '
                           'to generate robust tests for centralities or set sizes. \n'
                           'Suppressing warnings, but please be careful with the statistics! \n'
                           'Preferably use groups with at least 20 networks. ')
    model_calcs(networks, args)
    logger.info('anuran completed all tasks.')
    exit(0)


def model_calcs(networks, args):
    """
    Function for generating null models and carrying out calculations.
    :param networks: Dictionary with folder name as key and values as tuples (name, network object).
    :param args: Settings for running anuran
    :return:
    """
    if args['core'] < 1:
        args['core'] = 1
        logger.info("Setting cores for multiprocessing to 1.")
    # export intersections
    for size in args['size']:
        for group in networks:
            shared_edges = _intersection(networks[group], float(size), sign=args['sign'], edgelist=True)
            g = _construct_intersection(networks[group], shared_edges)
            nx.write_graphml(g, args['fp'] + '_' + group + '_' + str(size) + '_intersection.graphml')
    # first generate null models
    try:
        random, degree = generate_null(networks, n=args['perm'], npos=args['gperm'], core=args['core'], fraction=args['cs'],
                                       prev=args['prev'])
    except Exception:
        logger.error('Could not generate null models!', exc_info=True)
        sys.exit()
    set_sizes = None
    try:
        set_sizes = generate_sizes(networks, random, degree, core=args['core'],
                                   sign=args['sign'],
                                   fractions=args['cs'], prev=args['prev'],
                                   perm=args['nperm'], sizes=args['size'])
        set_sizes.to_csv(args['fp'] + '_sets.csv')
        set_differences = generate_size_differences(set_sizes, sizes=args['size'])
        set_differences.to_csv(args['fp'] + '_set_differences.csv')
        logger.info('Set sizes exported to: ' + args['fp'] + '_sets.csv')
    except Exception:
        logger.error('Failed to calculate set sizes!', exc_info=True)
        sys.exit()
    centralities = None
    if args['centrality']:
        try:
            centralities = generate_ci_frame(networks, random, degree,
                                             fractions=args['cs'], prev=args['prev'],
                                             perm=args['nperm'], core=args['core'])
            centralities.to_csv(args['fp'] + '_centralities.csv')
            logger.info('Centralities exported to: ' + args['fp'] + '_centralities.csv')
        except Exception:
            logger.error('Could not rank centralities!', exc_info=True)
            sys.exit()
    if args['network']:
        try:
            graph_properties = generate_graph_frame(networks, random, degree,
                                                    fractions=args['cs'], core=args['prev'],
                                                    perm=args['nperm'])
            graph_properties.to_csv(args['fp'] + '_graph_properties.csv')
            logger.info('Graph properties exported to: ' + args['fp'] + '_graph_properties.csv')
        except Exception:
            logger.error('Could not estimate graph properties!', exc_info=True)
            sys.exit()
    samples = None
    if args['sample']:
        try:
            samples = generate_sample_sizes(networks, random, degree,
                                            sign=args['sign'], core=args['core'],
                                            fractions=args['cs'], perm=args['nperm'], prev=args['prev'],
                                            sizes=args['size'], limit=args['sample'], number=args['number'])
            samples.to_csv(args['fp'] + '_subsampled_sets.csv')
            logger.info('Subsampled set sizes exported to: ' + args['fp'] + '_subsampled_sets.csv')
        except Exception:
            logger.error('Failed to subsample networks!', exc_info=True)
            sys.exit()
    central_stats = None
    if args['stats']:
        if args['stats'] == 'True':
            args['stats'] = True
        # add code for pvalue estimation
        set_stats = compare_set_sizes(set_sizes)
        set_stats.to_csv(args['fp'] + '_set_stats.csv')
        difference_stats = compare_set_sizes(set_differences)
        difference_stats.to_csv(args['fp'] + '_difference_stats.csv')
        if args['centrality'] and centralities is not None:
            central_stats = compare_centralities(centralities, mc=args['stats'])
            central_stats.to_csv(args['fp'] + '_centrality_stats.csv')
        if args['network']:
            graph_stats = compare_graph_properties(graph_properties)
            graph_stats.to_csv(args['fp'] + '_graph_stats.csv')
    # check if there is an order in the filenames
    for group in networks:
        prefixes = [x[0].split('_')[0] for x in networks[group]]
        try:
            prefixes = [int(x) for x in prefixes]
        except ValueError:
            pass
        if all(isinstance(x, int) for x in prefixes):
            centrality_correlation = correlate_centralities(group, centralities, mc=args['stats'])
            centrality_correlation.to_csv(args['fp'] + '_centrality_correlation.csv')
            graph_correlation = correlate_graph_properties(group, graph_properties)
            graph_correlation.to_csv(args['fp'] + '_centrality_correlation.csv')
    if args['draw']:
        try:
            for x in networks:
                subset_sizes = set_sizes[set_sizes['Group'] == x]
                draw_sets(subset_sizes, args['fp'] + '_' + x)
                subset_differences = set_differences[set_differences['Group'] == x]
                draw_set_differences(subset_differences, args['fp'] + '_' + x)
                if args['centrality']:
                    subset_centralities = centralities[centralities['Group'] == x]
                    draw_centralities(subset_centralities, args['fp'] + '_' + x)
                if args['sample']:
                    subset_samples = samples[samples['Group'] == x]
                    draw_samples(subset_samples, args['fp'] + '_' + x)
                if args['network']:
                    subset_graphs = graph_properties[graph_properties['Group'] == x]
                    draw_graphs(subset_graphs, args['fp'] + '_' + x)
        except Exception:
            logger.error('Could not draw data!', exc_info=True)
            sys.exit()
    if central_stats is not None:
        return central_stats


if __name__ == '__main__':
    main()
