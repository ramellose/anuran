#!/usr/bin/env python

"""
numma: Null models for replicate networks.
The script takes a network as input and uses this to generate null models.
The output of the null models is presented as a csv of set sizes
and a t-test is used to assess whether set sizes are different than expected from the null model.
Detailed explanations are available in the headers of each file.

numma uses the file extension to import networks.
Generation of null models is done on the adjacency matrix for speed;
the NetworkX representation is unfortunately slower.

The demo data for numma was downloaded from the following publication:
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

import numma
from numma.nullmodels import generate_null
from numma.set import generate_sizes, generate_sample_sizes
from numma.setviz import draw_sets, draw_samples
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


def set_numma():
    """This parser gets input settings for running numma.
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
                        help='Input network file. The format is detected based on the extension; \n'
                             'at the moment, .graphml, .txt (weighted edgelist), .gml and .cyjs are accepted. \n'
                             'If you set -i to "demo", a demo dataset will be loaded.',
                        default=None,
                        required=False,
                        nargs='+')
    parser.add_argument('-o', '--output',
                        dest='fp',
                        help='Output filename. Specify full file path without extension.',
                        default=None, required=False)
    parser.add_argument('-n', '--null_model',
                        dest='null',
                        required=False,
                        help='Types of null models to generate (degree-preserving or random rewiring). '
                             '\n By default, both models are generated.',
                        choices=['degree', 'random'],
                        nargs='+',
                        default=['degree', 'random'])
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
                        help='If flagged, signs of edge weights are taken into account. \n'
                             'The set difference then includes edges that have a unique edge sign in one network. \n'
                             'The set intersection then only includes edges that have the same sign across networks.',
                        default=True)
    parser.add_argument('-sample', '--resample',
                        dest='sample',
                        required=False,
                        type=int,
                        help='Resample your networks to generate changes in the set sizes \n'
                             'when you increase the network number up until the total. \n'
                             'Specify an upper limit of resamples, or True if you want all possible resamples. \n'
                             'By default, the upper limit equal to the binomial coefficient of the input networks. \n'
                             'If the limit is higher than this coefficient, all possible combinations are resampled.',
                        default=False)
    parser.add_argument('-share', '--shared_interactions',
                        dest='share',
                        required=False,
                        nargs='+',
                        default=False,
                        help='If specified, randomized null models (not the degree-preserving models)'
                             ' include a set fraction of shared interactions. \n'
                             'You can specify multiple fractions. '
                             'By default, null models have no shared interactions and '
                             'sets are computed for all randomized networks.\n'
                             'When the fraction is larger than 0, sets are only computed '
                             'between models generated from a single network.')
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
    parser.add_argument('-draw', '--draw_figures',
                        dest='draw',
                        required=False,
                        help='If flagged, draws figures showing the set sizes.',
                        action='store_true',
                        default=False)
    parser.add_argument('-version', '--version',
                        dest='version',
                        required=False,
                        help='Version number.',
                        action='store_true',
                        default=False)
    return parser


def main():
    args = set_numma().parse_args(sys.argv[1:])
    args = vars(args)
    if args['version']:
        info = VersionInfo('numma')
        logger.info('Version ' + info.version_string())
        exit(0)
    networks = list()
    if not args['graph']:
        logger.info('Please give an input file.')
    if args['graph'] != ['demo']:
        for file in args['graph']:
            filename = file.split(sep=".")
            extension = filename[len(filename)-1]
            try:
                if extension == 'graphml':
                    network = nx.read_graphml(args['graph'])
                elif extension == 'txt':
                    network = nx.read_weighted_edgelist(args['graph'])
                elif extension == 'gml':
                    network = nx.read_gml(args['graph'])
                else:
                    logger.warning('Format not accepted. '
                                   'Please specify the filename including extension (e.g. test.graphml).', exc_info=True)
                    exit()
            except Exception:
                logger.error('Could not import network file!', exc_info=True)
                exit()
    elif args['graph'] == ['demo']:
        path = os.path.dirname(numma.__file__)
        networks.append(nx.read_graphml(path + '//data//conet_family_a.graphml'))
        networks.append(nx.read_graphml(path + '//data//conet_family_b.graphml'))
        networks.append(nx.read_graphml(path + '//data//conet_family_c.graphml'))
    logger.info('Imported ' + str(len(networks)) + ' networks.')
    # first generate null models
    random = None
    degree = None
    if 'random' in args['null']:
        random = []
        random_fractions = []
        try:
            random = generate_null(networks, n=args['perm'], share=0, mode='random')
            if args['share']:
                for frac in args['share']:
                        random_fractions.append(generate_null(networks, n=args['perm'], share=frac, mode='random'))
                logger.info('Finished constructing all randomized networks.')
        except Exception:
            logger.error('Could not generate randomized null models!', exc_info=True)
            exit()
    if 'degree' in args['null']:
        degree = []
        try:
            degree = generate_null(networks, n=args['perm'], share=0, mode='degree')
            logger.info('Finished constructing all degree-preserving randomized networks.')
        except Exception:
            logger.error('Could not generate degree-preserving null models!', exc_info=True)
            exit()
    set_sizes = None
    try:
        set_sizes = generate_sizes(networks, random, random_fractions, degree,
                                   sign=args['sign'], set_operation=args['set'],
                                   fractions=args['share'], perm=args['nperm'], sizes=args['size'])
        set_sizes.to_csv(args['fp'] + '_sets.csv')
        logger.info('Set sizes exported to: ' + args['fp'] + '_sets.csv')
    except Exception:
        logger.error('Failed to calculate set sizes!', exc_info=True)
        exit()
    samples = None
    if args['sample']:
        try:
            samples = generate_sample_sizes(networks, random, random_fractions,
                                            degree, sign=args['sign'], set_operation=args['set'],
                                            fractions=args['share'], perm=args['nperm'],
                                            sizes=args['size'], limit=args['sample'])
            samples.to_csv(args['fp'] + '_subsampled_sets.csv')
        except Exception:
            logger.error('Failed to subsample networks!', exc_info=True)
            exit()
    if args['draw']:
        try:
            draw_sets(set_sizes, args['fp'])
            if args['sample']:
                draw_samples(samples, args['fp'])
        except Exception:
            logger.error('Could not draw data!', exc_info=True)
            exit()
    logger.info('numma completed all tasks.')
    exit(0)


if __name__ == '__main__':
    main()
