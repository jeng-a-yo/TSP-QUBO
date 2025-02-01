from pyqubo import Array, Constraint, Placeholder, Binary
import dimod
import numpy as np

from tsp_dp import solve_tsp_dp
from tsp import TSP


def generate_tsp_qubo_gps(graph, num_nodes, feed_dict):
    """
    Generate QUBO for GPS formulation of the TSP.

    Args:
        graph (list of list): Distance matrix where graph[i][j] is the distance between node i and j.
        num_nodes (int): Number of nodes (cities) in the TSP.

    Returns:
        model: Compiled PyQUBO model.
        qubo: QUBO dictionary.
    """
    # Binary variables
    x = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                for r in range(3):
                    x[i, j, r] = Binary(f"x_{i}_{j}_{r}")

    # Objective: Minimize travel distance
    objective = sum(graph[i][j] * x[i, j, 1] for i in range(num_nodes) for j in range(num_nodes) if i != j)

    # Constraints
    constraints = []

    # Constraint 1: Exactly one of r=0,1,2 per edge (i, j)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                constraints.append(Constraint(
                    (sum(x[i, j, r] for r in range(3)) - 1) ** 2, label=f"one_r_{i}_{j}"
                ))

    # Constraint 2: Each node is exited exactly once
    for i in range(num_nodes):
        constraints.append(Constraint(
            (sum(x[i, j, 1] for j in range(num_nodes) if i != j) - 1) ** 2, label=f"exit_once_{i}"
        ))

    # Constraint 3: Each node is reached exactly once
    for j in range(num_nodes):
        constraints.append(Constraint(
            (sum(x[i, j, 1] for i in range(num_nodes) if i != j) - 1) ** 2, label=f"reach_once_{j}"
        ))

    # Constraint 4: Prevent sub-tours
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                constraints.append(Constraint(
                    (x[i, j, 2] + x[j, i, 2] - 1) ** 2, label=f"no_cycle_{i}_{j}"
                ))

    # Constraint 5: Prevent cycles
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_nodes):
                if i != j and i != k and j != k:
                    constraints.append(
                        Constraint(
                            (x[j, i, 2] * x[k, j, 2] - 
                             x[j, i, 2] * x[k, i, 2] - 
                             x[k, j, 2] * x[k, i, 2] + 
                             x[k, i, 2] ** 2),
                            label=f"no_cycle_{i}_{j}_{k}"
                        )
                    )

    # Combine objective and constraints
    H = objective + Placeholder("lambda") * sum(constraints)
    # Compile QUBO with a feed_dict
    model = H.compile()
    qubo, offset = model.to_qubo(feed_dict=feed_dict)

    return model, qubo


def solve_tsp_with_dimod(qubo, num_reads=100):
    """
    Solve the QUBO for TSP using a Dimod solver.

    Args:
        qubo (dict): The QUBO dictionary.

    Returns:
        sampleset: Solution set from the Dimod solver.
    """
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo, num_reads=num_reads)
    return sampleset


def decode_solution(model, sampleset, feed_dict):
    """
    Decode the solution from the QUBO sampleset.

    Args:
        model: Compiled PyQUBO model.
        sampleset: Solution set from the Dimod solver.
        feed_dict: Feed dictionary for the PyQUBO model.

    Returns:
        dict: Decoded solution.
    """
    best_solution = sampleset.first.sample
    decoded_solution = model.decode_sample(best_solution, vartype='BINARY', feed_dict=feed_dict)
    return decoded_solution


if __name__ == "__main__":
    # Constant setup
    num_nodes = 5
    weight_range = (1, 100)
    num_reads = 200
    feed_dict = {"lambda": 30.0}


    # Generate random graph for TSP
    # np.random.seed(42)
    tsp_instance = TSP(num_nodes, weight_range)
    graph = tsp_instance.graph

    print(graph)

    # Generate QUBO and solve
    model, qubo = generate_tsp_qubo_gps(graph, num_nodes, feed_dict)
    sampleset = solve_tsp_with_dimod(qubo, num_reads)

    # Decode the solution
    decoded_solution = decode_solution(model, sampleset, feed_dict)
    print("Decoded solution: ", decoded_solution)

    min_cost, optimal_path = solve_tsp_dp(num_nodes, graph)
    print("Minimum cost: ", min_cost)
    print("Optimal path: ", optimal_path)
