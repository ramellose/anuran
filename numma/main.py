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

"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import networkx as nx
import sys
import os
import argparse
import numpy as np
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

# handler to file
# only handler with 'w' mode, rest is 'a'
# once this handler is started, the file writing is cleared
# other handlers append to the file
logpath = "\\".join(os.getcwd().split("\\")[:-1]) + '\\numma.log'
# filelog path is one folder above manta
# pyinstaller creates a temporary folder, so log would be deleted
fh = logging.handlers.RotatingFileHandler(maxBytes=500,
                                          filename=logpath, mode='a')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


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
                        choices=['deg', 'random'],
                        nargs='+',
                        default=['deg', 'random'])
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
                        help='If specified, associations only shared by a number of networks are included. \n'
                             'You can specify multiple numbers. ')
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
                        help='Resamples your networks to generate a figure demonstrating changes in the set sizes \n'
                             'when you increase the network number up until the total.'
                        default=False)
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        required=False,
                        action='store_true',
                        help='If flagged, rovides additional details on progress. ',
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
    if args['graph'] != 'demo':
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
        logger.info('Wrote clustered network to ' + args['fp'] + '.')
    exit(0)


if __name__ == '__main__':
    main()
