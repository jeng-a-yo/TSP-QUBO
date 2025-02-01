import random
import numpy as np
import pandas as pd

class TSP:
    def __init__(self, n, weight_range=(1, 100)):
        """
        Initialize the TSP problem with n cities.
        
        Args:
            n (int): Number of cities.
            weight_range (tuple): Range of edge weights (default: 1 to 100).
        """
        self.n = n
        self.graph = self.generate_graph(n, weight_range)
    
    @staticmethod
    def generate_graph(num_nodes, weight_range):
        """
        Generate a random weighted adjacency matrix for the TSP graph.

        Args:
            num_nodes (int): Number of nodes (cities).
            weight_range (tuple): Range of weights (min, max).

        Returns:
            np.array: Symmetric adjacency matrix.
        """
        rng = np.random.default_rng()
        graph = rng.integers(weight_range[0], weight_range[1] + 1, size=(num_nodes, num_nodes))
        np.fill_diagonal(graph, 0)
        return (graph + graph.T) // 2
    

    def show_graph(self):
        """
        Display the TSP graph as a well-formatted table.
        """
        df = pd.DataFrame(self.graph, index=[f"C{i}" for i in range(self.n)], columns=[f"C{i}" for i in range(self.n)])
        print("TSP Distance Matrix:")
        print(df)

    