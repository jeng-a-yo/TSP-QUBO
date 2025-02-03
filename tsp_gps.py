from pyqubo import Array, Constraint, Placeholder, Binary
import dimod
import numpy as np
from pprint import pprint

from tsp_dp import solve_tsp_dp
from tsp_brute_force import solve_tsp_brute_force
from tsp import TSP

def generate_tsp_qubo(graph, num_nodes, feed_dict):
    """
    Generate QUBO for TSP using the GPS formulation.
    """
    x = {(i, j, r): Binary(f"x_{i}_{j}_{r}") for i in range(num_nodes) for j in range(num_nodes) if i != j for r in range(3)}
    
    # Objective: Minimize travel distance
    objective = sum(graph[i][j] * x[i, j, 1] for i in range(num_nodes) for j in range(num_nodes) if i != j)
    
    # Constraints
    constraints = [
        Constraint((sum(x[i, j, r] for r in range(3)) - 1) ** 2, label=f"one_r_{i}_{j}")
        for i in range(num_nodes) for j in range(num_nodes) if i != j
    ]
    
    constraints += [
        Constraint((sum(x[i, j, 1] for j in range(num_nodes) if i != j) - 1) ** 2, label=f"exit_once_{i}")
        for i in range(num_nodes)
    ]
    
    constraints += [
        Constraint((sum(x[i, j, 1] for i in range(num_nodes) if i != j) - 1) ** 2, label=f"reach_once_{j}")
        for j in range(num_nodes)
    ]
    
    constraints += [
        Constraint((x[i, j, 2] + x[j, i, 2] - 1) ** 2, label=f"no_cycle_{i}_{j}")
        for i in range(num_nodes) for j in range(num_nodes) if i != j
    ]
    
    constraint_5 = [
        Constraint(((x[j, i, 2] * x[k, j, 2]) - (x[j, i, 2] * x[k, i, 2]) - (x[k, j, 2] * x[k, i, 2]) + x[k, i, 2]) ** 2,
                   label=f"prevent_cycle_{i}_{j}_{k}")
        for i in range(num_nodes) for j in range(num_nodes) for k in range(num_nodes) if i != j and i != k and j != k
    ]
    
    # Combine constraints and objective function
    H = objective + sum(constraints) + Placeholder("lambda") * sum(constraint_5)
    model = H.compile()
    qubo, offset = model.to_qubo(feed_dict=feed_dict)
    return model, qubo

def solve_tsp_with_dimod(qubo, num_reads=100):
    """
    Solve the QUBO using a Simulated Annealing sampler from Dimod.
    """
    sampler = dimod.SimulatedAnnealingSampler()
    return sampler.sample_qubo(qubo, num_reads=num_reads)

def decode_solution(model, sampleset, feed_dict):
    """
    Decode the best solution from the Dimod sampleset.
    """
    best_solution = sampleset.first.sample
    return model.decode_sample(best_solution, vartype='BINARY', feed_dict=feed_dict)

def extract_tsp_path(decoded_solution, num_nodes):
    """
    Extracts the optimal TSP path from the decoded binary solution.
    """
    selected_edges = []

    for key, val in decoded_solution.sample.items():
        if val == 1 and key.startswith("x_"):
            parts = key.split("_")  # Example: "x_0_1_1" -> ["x", "0", "1", "1"]
            if len(parts) == 4:
                i, j, r = int(parts[1]), int(parts[2]), int(parts[3])
                if r == 1:  # We only take edges with r=1
                    selected_edges.append((i, j))

    if not selected_edges:
        print("No valid tour found!")
        return None

    # Build the tour by chaining edges
    path = []
    current_node = 0  # Start from node 0
    visited = set()

    while len(path) < num_nodes:
        path.append(current_node)
        visited.add(current_node)
        next_nodes = [j for i, j in selected_edges if i == current_node and j not in visited]

        if not next_nodes:
            break  # No further connections

        current_node = next_nodes[0]  # Move to the next node

    path.append(path[0])  # Complete the cycle
    return path



def print_solution(decoded_solution, num_nodes):
    """
    Pretty print the decoded solution and constraints, including the optimal path.
    """
    print("\nDecoded Binary Solution:")
    for key, value in decoded_solution.sample.items():
        print(f"{key}: {value}")

    print("\nConstraint Values:")
    for label, value in decoded_solution.constraints().items():
        print(f"{label}: {value}")

    # Extract and print the optimal TSP path
    optimal_path = extract_tsp_path(decoded_solution, num_nodes)
    if optimal_path:
        print("\nOptimal Path (QUBO Solution):", " -> ".join(map(str, optimal_path)))


# Modify the main function call
if __name__ == "__main__":
    num_nodes = 4
    weight_range = (1, 100)
    num_reads = 200
    feed_dict = {"lambda": 1}

    # Generate TSP instance
    tsp_instance = TSP(num_nodes, weight_range)
    graph = tsp_instance.graph

    print("Graph (Distance Matrix):")
    pprint(graph)

    # Generate QUBO and solve
    model, qubo = generate_tsp_qubo(graph, num_nodes, feed_dict)
    sampleset = solve_tsp_with_dimod(qubo, num_reads)

    # Decode and print the solution
    decoded_solution = decode_solution(model, sampleset, feed_dict)
    print_solution(decoded_solution, num_nodes)

    # Compute optimal solutions using classical methods
    min_cost_dp, optimal_path_dp = solve_tsp_dp(num_nodes, graph)
    print("\nMinimum cost (Dynamic Programming):", min_cost_dp)
    print("Optimal path (Dynamic Programming):", optimal_path_dp)

    min_cost_bf, optimal_path_bf = solve_tsp_brute_force(num_nodes, graph)
    print("\nMinimum cost (Brute Force):", min_cost_bf)
    print("Optimal path (Brute Force):", optimal_path_bf)
