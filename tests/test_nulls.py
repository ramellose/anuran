"""
This file contains a testing function + resources for testing
the null model generation in nulls.py.
"""

__author__ = 'Lisa Rottjers'
__maintainer__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import unittest
import networkx as nx
import numpy as np
from anuran.nulls import generate_null
from anuran.utils import _randomize_network, _randomize_dyads

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

    def test_generate_null(self):
        """
        Checks whether the specified number of randomized models is returned.
        generate_null should generate a list of lists with each of the lists
        containing all permuted networks for one original network.
        """
        perm = 10
        random, degree = generate_null(networks, n=perm, core=2)
        self.assertEqual(len(random['a']['random'][0]), perm)
        self.assertEqual(len(random['a']['random']), len(networks['a']))

    def test_generate_core(self):
        """
        Checks whether the specified number of randomized models is returned.
        generate_core should generate null models with fractions conserved.
        """
        a_core = generate_null(networks, n=1, core=2, fraction=[1], prev=[1])[0]['a']['core'][1][1][0]
        b_core = generate_null(networks, n=1, core=2, fraction=[1], prev=[1])[0]['a']['core'][1][1][0]
        a = list(a_core[0][1].edges)
        a.sort()
        b = list(b_core[0][1].edges)
        b.sort()
        self.assertEqual(a[0], b[0])

    def test_randomize_network(self):
        """
        Checks whether a randomized network is returned.
        """
        random = _randomize_network(a, keep=[])
        orig_deg = np.sort(nx.degree(a))
        new_deg = np.sort(nx.degree(random))
        self.assertFalse((orig_deg == new_deg).all())

    def test_randomize_dyads(self):
        """
        Checks whether a network with swapped dyads is returned.
        """
        random = _randomize_dyads(a, keep=[])
        orig_deg = np.sort(nx.degree(a))
        new_deg = np.sort(nx.degree(random))
        self.assertTrue((orig_deg == new_deg).all())

    def test_generate_core_random(self):
        """
        Checks whether a number associations occurs
        a certain number of times given a core size and prevalence.
        """
        nets = {'a': [('a', a), ('b', b), ('c', c)]}  # at least 5 nodes necessary for most tests
        random, degree = generate_null(nets, n=1, core=2, fraction=[0.3], prev=[0.6])
        core = random['a']['core'][0.3][0.6][0]
        all_edges = list()
        for network in core:
            all_edges.extend(network[1].edges)
        counts = {x: 0 for x in list(set(all_edges))}
        for edge in all_edges:
            counts[edge] += 1
        num_shared = 0
        for edge in counts:
            if counts[edge] > (0.6 * len(core)):
                num_shared += 1
        self.assertGreater(num_shared, 0.3 * len(core[0][1].edges))


if __name__ == '__main__':
    unittest.main()
