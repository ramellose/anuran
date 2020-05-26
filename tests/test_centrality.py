"""
This file contains a testing function + resources for testing whether the centrality values are calculated
and reported correctly with centrality.py.
"""

__author__ = 'Lisa Rottjers'
__maintainer__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import unittest
import networkx as nx
import numpy as np
from anuran.centrality import generate_ci_frame, generate_confidence_interval, \
    _generate_ci_rows
from anuran.utils import _generate_centralities_parallel, _centrality_percentile
import pandas as pd


# generate three alternative networks with first 4 edges conserved but rest random
nodes = ["OTU_1", "OTU_2", "OTU_3", "OTU_4", "OTU_5"]
one = [("OTU_1", "OTU_2"), ("OTU_1", "OTU_3"),
       ("OTU_2", "OTU_5"), ("OTU_3", "OTU_4"),
       ("OTU_2", "OTU_3"), ("OTU_2", "OTU_4")]
two = [("OTU_1", "OTU_2"), ("OTU_1", "OTU_3"),
       ("OTU_2", "OTU_5"), ("OTU_3", "OTU_4"),
       ("OTU_3", "OTU_5"), ("OTU_4", "OTU_5")]
three = [("OTU_1", "OTU_2"), ("OTU_1", "OTU_3"),
         ("OTU_2", "OTU_5"), ("OTU_3", "OTU_4"),
         ("OTU_1", "OTU_4"), ("OTU_4", "OTU_5")]
weights = dict()
weights[("OTU_1", "OTU_2")] = float(1.0)
weights[("OTU_1", "OTU_3")] = float(1.0)
weights[("OTU_2", "OTU_5")] = float(1.0)
weights[("OTU_3", "OTU_4")] = float(-1.0)
weights[("OTU_2", "OTU_3")] = float(-1.0)
weights[("OTU_2", "OTU_4")] = float(-1.0)
weights[("OTU_3", "OTU_5")] = float(-1.0)
weights[("OTU_4", "OTU_5")] = float(-1.0)
weights[("OTU_1", "OTU_4")] = float(-1.0)

a = nx.Graph()
a.add_nodes_from(nodes)
a.add_edges_from(one)
nx.set_edge_attributes(a, values=weights, name='weight')
a = a.to_undirected()

b = nx.Graph()
b.add_nodes_from(nodes)
b.add_edges_from(two)
nx.set_edge_attributes(b, values=weights, name='weight')
b = b.to_undirected()

weights[("OTU_1", "OTU_2")] = float(-1.0)
c = nx.Graph()
c.add_nodes_from(nodes)
c.add_edges_from(three)
nx.set_edge_attributes(c, values=weights, name='weight')
c = c.to_undirected()

networks = {'a': [('a', a)], 'b': [('b', b)], 'c': [('c', c)]}


class TestMain(unittest.TestCase):
    """"
    Tests whether the main clustering function properly assigns cluster IDs.
    """

    def test_generate_ci_frame(self):
        """
        When given dictionaries with networks, this function should return a
        dataframe reporting the centrality values over groups of networks.
        """
        random = {x: {'random': [], 'core': {}} for x in networks}
        degree = {x: {'degree': [], 'core': {}} for x in networks}
        results = generate_ci_frame(networks, random=random, degree=degree,
                                    fractions=None, prev=None, perm=0, core=1)
        totalnodes = np.sum([len(networks[x][0][1].nodes) for x in networks])
        self.assertEqual(len(results), totalnodes*3)

    def test_generate_centralities(self):
        """
        Tests whether the generate_centralities function returns a ranking of centralities.
        """
        new = {'a': [networks['a'][0], networks['b'][0], networks['c'][0]]}
        ranking = _generate_centralities_parallel(new['a'])
        self.assertEqual(len(ranking), 3)

    def test_generate_confidence_interval(self):
        """
        Since there are only 3 networks, this function should
        return confidence intervals of (0, 1).
        """
        new = [networks['a'][0], networks['b'][0], networks['c'][0]]
        ranking = _generate_centralities_parallel(new)
        centrality_scores = [(ranking[i][0], ranking[i][2]['Betweenness']) for i in range(len(ranking))]
        CI = generate_confidence_interval(centrality_scores)
        self.assertEqual(CI['OTU_1'], (0, 1))

    def test__generate_ci_rows(self):
        """
        Tests whether a dataframe is returned with a length equal to
        3 times the number of nodes.
        :return:
        """
        new = {'a': [('a', networks['a'][0][1]),
                     ('b', networks['b'][0][1]),
                     ('c', networks['c'][0][1])]}
        results = pd.DataFrame(columns=['Node', 'Network', 'Group', 'Network type', 'Conserved fraction',
                                        'Prevalence of conserved fraction',
                                        'Centrality', 'Upper limit', 'Lower limit', 'Values'])
        central_new = _generate_centralities_parallel(new['a'])
        results = _generate_ci_rows(data=results, name='a', group='a', networks=central_new, fraction=None, prev=None)
        nodes = np.sum(len(x[1].nodes) for x in networks['a'])
        self.assertEqual(nodes*3, len(results))

    def centrality_percentile(self):
        """
        When given centrality scores, this function should return a ranking from 0 to 1.
        """
        deg = nx.degree_centrality(networks['a'][0])
        ranking = _centrality_percentile(deg)
        self.assertEqual(np.max(list(ranking.values())), 1)


if __name__ == '__main__':
    unittest.main()
