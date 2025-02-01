from tsp import TSP
from functools import lru_cache

def solve_tsp_dp(n, graph):
    """
    Solves the TSP using dynamic programming with bit masking (Held-Karp algorithm).
    
    Args:
        tsp_instance (TSP): An instance of the TSP problem.

    Returns:
        tuple: (minimum_cost, optimal_path)
    """

    
    @lru_cache(None)
    def dp(mask, pos):
        """
        Recursively computes the minimum cost using memoization.
        
        Args:
            mask (int): Bitmask representing visited cities.
            pos (int): Current city position.
        
        Returns:
            int: Minimum travel cost.
        """
        if mask == (1 << n) - 1:  # All cities visited
            return graph[pos][0]  # Return to start
        
        min_cost = float('inf')
        for city in range(n):
            if not (mask & (1 << city)):  # If city is unvisited
                new_cost = graph[pos][city] + dp(mask | (1 << city), city)
                min_cost = min(min_cost, new_cost)
        return min_cost
    
    def get_path():
        """Reconstructs the path from memoized results."""
        mask, pos = 1, 0  # Start at city 0
        path = [0]
        while len(path) < n:
            next_city = min(
                (city for city in range(n) if not (mask & (1 << city))),
                key=lambda city: graph[pos][city] + dp(mask | (1 << city), city)
            )
            mask |= (1 << next_city)
            path.append(next_city)
            pos = next_city
        path.append(0)  # Return to start
        return path
    
    min_cost = dp(1, 0)  # Start from city 0 with only it visited
    optimal_path = get_path()
    return min_cost, optimal_path

if __name__ == '__main__':
    # Example Usage
    tsp_instance = TSP(5, (1, 20))
    min_cost, optimal_path = solve_tsp_dp(tsp_instance.n, tsp_instance.graph)
    tsp_instance.show_graph()
    print("Minimum cost: ", min_cost)
    print("Optimal path: ", optimal_path)
