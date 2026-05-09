from unittest import TestCase

from project_hilbert.clustering import cluster_neighbors
from project_hilbert.salience import compute_salience
from project_hilbert.vector_math import normalize


class ClusteringAndSalienceTests(TestCase):
    def test_cluster_neighbors_finds_two_clusters(self) -> None:
        vectors = [
            normalize([1.0, 0.0, 0.0]),
            normalize([0.98, 0.02, 0.0]),
            normalize([0.0, 1.0, 0.0]),
            normalize([0.0, 0.98, 0.02]),
        ]
        labels = cluster_neighbors(vectors, eps=0.05, min_samples=2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_cluster_neighbors_handles_too_few_points(self) -> None:
        labels = cluster_neighbors([normalize([1.0, 0.0])], eps=0.1, min_samples=2)
        self.assertEqual(labels, [None])

    def test_salience_is_bounded(self) -> None:
        vectors = [
            normalize([1.0, 0.0, 0.0]),
            normalize([0.98, 0.02, 0.0]),
            normalize([0.0, 1.0, 0.0]),
        ]
        values = compute_salience(vectors, query_vector=normalize([1.0, 0.0, 0.0]))
        self.assertEqual(len(values), 3)
        for value in values:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_single_salience_is_one(self) -> None:
        values = compute_salience([normalize([1.0, 0.0])])
        self.assertEqual(values, [1.0])

