from itertools import permutations
from tsp import TSP

def calculate_distance(graph, path):
    """
    Calculate the total distance of a given path.
    
    Args:
        path (tuple): A sequence of cities representing a path.

    Returns:
        int: Total distance of the path.
    """
    total_distance = sum(graph[path[i], path[i + 1]] for i in range(len(path) - 1))
    total_distance += graph[path[-1], path[0]]  # Returning to the start
    return total_distance

def solve_tsp_brute_force(num_nodes, graph):
        """
        Solve the TSP problem using brute force (permutations).

        Returns:
            tuple: (best_path, min_distance)
        """
        cities = list(range(num_nodes))
        min_distance = float('inf')
        best_path = None

        for perm in permutations(cities):
            current_distance = calculate_distance(graph, perm)
            if current_distance < min_distance:
                min_distance = current_distance
                best_path = perm

        return min_distance, best_path

if __name__ == "__main__":
    # Constant setup
    num_nodes = 5
    weight_range = (1, 100)



    # Generate random graph for TSP
    # np.random.seed(42)
    tsp_instance = TSP(num_nodes, weight_range)
    graph = tsp_instance.graph
    min_distance, best_path = solve_tsp_brute_force(num_nodes, graph)
    print("Best path:", best_path)
    print("Minimum distance:", min_distance)