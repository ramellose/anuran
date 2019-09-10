"""
This file contains a testing function + resources for testing whether the graph value estimations
from graphvals.py return the correct estimates.
"""

__author__ = 'Lisa Rottjers'
__maintainer__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import unittest
import networkx as nx
import pandas as pd
from anuran.graphvals import generate_graph_frame, generate_graph_properties, _generate_graph_rows

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

networks = {'a': [a], 'b': [b], 'c': [c]}


class TestMain(unittest.TestCase):
    """"
    Tests whether the main clustering function properly assigns cluster IDs.
    """

    def test_generate_graph_frame(self):
        """
        Tests whether the graph frame function returns results for all networks.
        """
        random = {x: {'random': [], 'core': {}} for x in networks}
        degree = {x: {'degree': [], 'core': {}} for x in networks}
        frame = generate_graph_frame(networks, random=random,
                                     degree=degree,
                                     fractions=None, core=None)
        # check that length of dataframe is equal to the set of properties times
        # the number of networks
        self.assertEqual(len(frame), len(set(frame['Property']))*len(networks))

    def test_generate_graph_properties(self):
        """
        Tests whether the graph frame function returns a dictionary of graph properties
        with each property key referring to a list of values for all graphs.
        """
        results = generate_graph_properties(networks['a'])
        properties = ['Assortativity', 'Connectivity', 'Diameter', 'Radius', 'Average shortest path length']
        self.assertTrue(all(prop in results for prop in properties))

    def test_generate_graph_rows(self):
        """
        Tests whether this function adds a new row if supplied the correct parameters.
        """
        frame = pd.DataFrame(columns=['Network', 'Group', 'Network type', 'Conserved fraction',
                                      'Prevalence of conserved fraction',
                                      'Property', 'Value'])
        frame = _generate_graph_rows(data=frame, name='test', group='a',
                                     networks=networks['a'], fraction=None, prev=None)
        self.assertEqual(len(frame), 5)


if __name__ == '__main__':
    unittest.main()
