"""
This file contains a testing function + resources for testing the clustering algorithm.
"""

__author__ = 'Lisa Rottjers'
__maintainer__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import unittest
import networkx as nx
from scipy.special import binom
import numpy as np
from numma.nullmodels import generate_random, generate_degree, randomize_network, randomize_dyads
from numma.set import generate_sizes, difference, intersection
from numma.resample import generate_sample_sizes, draw_samples

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

networks = [a, b, c]

class TestMain(unittest.TestCase):
    """"
    Tests whether the main clustering function properly assigns cluster IDs.
    """

    def generate_random(self):
        """Checks whether the specified number of randomized models is returned.
         generate_random should generate a list of lists with each of the lists
         containing all permuted networks for one original network. """
        perm = 10
        results = generate_random(networks, perm)
        self.assertEqual(len(results[0]), perm)
        self.assertEqual(len(results), networks)

    def generate_random_distr(self):
        """Checks whether the degree distribution of the random models is different. """
        perm = 1
        results = generate_random(networks, perm)
        orig_deg = nx.degree(networks[0])
        new_deg = nx.degree(results[0][0])
        self.assertNotEqual(orig_deg, new_deg)

    def generate_degree(self):
        """Checks whether the specified number of randomized models is returned.
         generate_random should generate a list of lists with each of the lists
         containing all permuted networks for one original network. """
        perm = 10
        results = generate_degree(networks, perm)
        self.assertEqual(len(results[0]), perm)
        self.assertEqual(len(results), networks)

    def randomize_network(self):
        """Checks whether a randomized network is returned. """
        random = randomize_network(a)
        orig_deg = np.sort(nx.degree(a))
        new_deg = np.sort(nx.degree(random))
        self.assertNotEqual(orig_deg, new_deg)

    def randomize_dyads(self):
        """Checks whether a network with swapped dyads is returned. """
        random = randomize_dyads(a)
        orig_deg = np.sort(nx.degree(a))
        new_deg = np.sort(nx.degree(random))
        self.assertEqual(orig_deg, new_deg)

    def generate_sizes(self):
        """Checks whether the set sizes are correctly returned. """
        perm = 10
        nperm = 10
        random = generate_random(networks, perm)
        degree = generate_degree(networks, perm)
        results = generate_sizes(networks, random, degree)
        self.assertEqual(len(results['Type']), 23)

    def intersection(self):
        """Checks whether the intersection set size is correctly returned. """
        results = intersection(networks)
        self.assertEqual(len(results['Type']), 23)

    def intersection_size(self):
        """Checks whether the intersection set size is correctly returned. """
        results = intersection(networks, 2)
        self.assertEqual(len(results['Type']), 23)

    def difference(self):
        """Checks whether the difference set size is correctly returned. """
        results = difference(networks)
        self.assertEqual(len(results['Type']), 23)

    def generate_sample_sizes(self):
        """Checks whether the subsampled set sizes are correctly returned. """
        perm = 10
        nperm = 10
        random = generate_random(networks, perm)
        degree = generate_degree(networks, perm)
        results = generate_sample_sizes(networks, random, degree)
        num = 23 * (len(networks) - 1)
        self.assertEqual(len(results['Networks']), num)

if __name__ == '__main__':
    unittest.main()
