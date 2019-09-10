"""
This file contains a testing function + resources for testing whether the statistical tests in stats.py
are being carried out and returned.
"""

__author__ = 'Lisa Rottjers'
__maintainer__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import unittest
import networkx as nx
import numpy as np
from anuran.nullmodels import generate_null
from anuran.set import generate_sizes
from anuran.centrality import generate_ci_frame
from anuran.graphvals import generate_graph_frame
from anuran.stats import compare_centralities, compare_graph_properties, \
    compare_set_sizes, _generate_stat_rows, _value_outside_range

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

networks = {'a': [a, b, c, a, b, c]} # at least 5 nodes necessary for most tests
random = {x: {'random': [], 'core': {}} for x in networks}
degree = {x: {'degree': [], 'core': {}} for x in networks}
for x in networks:
    degree[x]['degree'] = generate_null(networks[x], n=10, share=0, mode='degree')
    random[x]['random'] = generate_null(networks[x], n=10, share=0, mode='random')


class TestMain(unittest.TestCase):
    """"
    Tests whether the main clustering function properly assigns cluster IDs.
    """

    def test_compare_centralities(self):
        """
        Given a pandas dataframe with centralities for different nodes in groups of networks,
        this function should return a dataframe with statistics on these centralities.
        """
        centralities = generate_ci_frame(networks, random, degree, fractions=None, core=None)
        results = compare_centralities(centralities, mc=None)
        otu_1 = results[results['Node'] == 'OTU_1']
        self.assertGreater(0.5, otu_1[otu_1['Comparison'] == 'Random']['P'].iloc[0])

    def test_compare_graph_properties(self):
        """
        Given a pandas dataframe with graph properties for different nodes in groups of networks,
        this function should return a dataframe with statistics on these graph properties.
        """
        graph_values = generate_graph_frame(networks, random, degree, fractions=None, core=None)
        # values are identical for most properties
        graph_values = graph_values[graph_values['Property'] == 'Assortativity']
        graph_values = graph_values[graph_values['Network'] != 'Degree']
        results = compare_graph_properties(graph_values, mc=None)
        self.assertGreater(0.5, results['P'].iloc[0])

    def test_compare_set_sizes(self):
        """
        Given a pandas dataframe with set sizes across groups of networks,
        this function should return a dataframe with statistics on these set sizes.
        """
        set_values = generate_sizes(networks, random, degree, fractions=None, core=None,
                                    sign=True, perm=10, sizes=[0.6, 1], set_operation=['difference', 'intersection'])
        results = compare_set_sizes(set_values, mc=None)
        results = results[results['Comparison'] == 'Random']
        results = results[results['Measure'] == 'Intersection 0.6']
        self.assertGreater(0.1, results['P'].iloc[0])

    def test_generate_stat_rows(self):
        """
        Given a pandas dataframe with results, this function should add a row.
        """
        set_values = generate_sizes(networks, random, degree, fractions=None, core=None,
                                    sign=True, perm=10, sizes=[0.6, 1], set_operation=['difference', 'intersection'])
        results = compare_set_sizes(set_values, mc=None)
        new_results = _generate_stat_rows(results, group='b', comparison='test',
                                          operation='test', p='0.05', ptype='test', node=None)
        self.assertGreater(len(new_results), len(results))

    def test_value_outside_range(self):
        """
        Given a list of values and a single value,
        this function calculates the p-value using the z-score.
        """
        values = [5, 7, 8, 4, 4, 9, 3]
        target = 10
        p = _value_outside_range(target, values)
        self.assertGreater(0.0005, p)


if __name__ == '__main__':
    unittest.main()
