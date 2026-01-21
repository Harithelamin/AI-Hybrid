# Multi-Objective Evolutionary Algorithms
# Key Concepts
# Pareto dominance
# Pareto optimal set
# Pareto front
# Trade-offs instead of a single “best” solution

# Why NSGA-II??
# Fast non-dominated sorting
# Elitism
# Diversity preservation via crowding distance

# The algortithm Workflow
# 1. Combine parent + offspring
# 2. Non-dominated sorting
# 3. Crowding distance
# 4. Truncation

# Why NSGA-II instead of weighted sum?
# 1. Weighted sum fails on non-convex Pareto fronts
# 2. NSGA-II finds multiple trade-off solutions in one run

# NSGA-II main steps:
# 1. Initialize population
# 2. Evaluate objectives
# 3. Non-dominated sorting
# 4. Crowding distance calculation
# 5. Selection (elitism)
# 6. Crossover
# 7. Mutation
# 8 .Loop over generations

import numpy as np
import random

# -----------------------------
# Parameters
# -----------------------------
POP_SIZE = 40
NUM_GENERATIONS = 50
GENE_LENGTH = 3          # number of decision variables
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9

# -----------------------------
# Objective Functions
# (Example: 2-objective minimization)
# -----------------------------
def objectives(individual):
    """
    Returns a tuple of objective values
    Example:
    f1 = minimize sum of genes
    f2 = minimize variance of genes
    """
    f1 = np.sum(individual)
    f2 = np.var(individual)
    return (f1, f2)

# -----------------------------
# Initialize Population
# -----------------------------
def initialize_population():
    return [np.random.rand(GENE_LENGTH) for _ in range(POP_SIZE)]

# -----------------------------
# Dominance Check
# -----------------------------
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

# -----------------------------
# Non-dominated Sorting
# -----------------------------
def non_dominated_sort(population, obj_values):
    fronts = []
    S = [[] for _ in range(len(population))]
    n = [0] * len(population)
    rank = [0] * len(population)

    for p in range(len(population)):
        for q in range(len(population)):
            if dominates(obj_values[p], obj_values[q]):
                S[p].append(q)
            elif dominates(obj_values[q], obj_values[p]):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0

    front = [i for i in range(len(population)) if n[i] == 0]
    fronts.append(front)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]

# -----------------------------
# Crowding Distance
# -----------------------------
def crowding_distance(front, obj_values):
    distance = {i: 0 for i in front}
    num_objectives = len(obj_values[0])

    for m in range(num_objectives):
        front_sorted = sorted(front, key=lambda i: obj_values[i][m])
        distance[front_sorted[0]] = float('inf')
        distance[front_sorted[-1]] = float('inf')

        min_val = obj_values[front_sorted[0]][m]
        max_val = obj_values[front_sorted[-1]][m]

        if max_val == min_val:
            continue

        for i in range(1, len(front_sorted) - 1):
            prev_val = obj_values[front_sorted[i - 1]][m]
            next_val = obj_values[front_sorted[i + 1]][m]
            distance[front_sorted[i]] += (next_val - prev_val) / (max_val - min_val)

    return distance

# -----------------------------
# Selection
# -----------------------------
def selection(population, fronts, obj_values):
    new_population = []

    for front in fronts:
        if len(new_population) + len(front) > POP_SIZE:
            distances = crowding_distance(front, obj_values)
            sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
            new_population.extend([population[i] for i in sorted_front[:POP_SIZE - len(new_population)]])
            break
        else:
            new_population.extend([population[i] for i in front])

    return new_population

# -----------------------------
# Crossover
# -----------------------------
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENE_LENGTH - 1)
        c1 = np.concatenate((p1[:point], p2[point:]))
        c2 = np.concatenate((p2[:point], p1[point:]))
        return c1, c2
    return p1.copy(), p2.copy()

# -----------------------------
# Mutation
# -----------------------------
def mutation(individual):
    for i in range(GENE_LENGTH):
        if random.random() < MUTATION_RATE:
            individual[i] = random.random()
    return individual

# -----------------------------
# NSGA-II Main Loop
# -----------------------------
population = initialize_population()

for gen in range(NUM_GENERATIONS):

    # Evaluate objectives
    obj_values = [objectives(ind) for ind in population]

    # Non-dominated sorting
    fronts = non_dominated_sort(population, obj_values)

    # Selection (elitism)
    selected = selection(population, fronts, obj_values)

    # Generate offspring
    offspring = []
    while len(offspring) < POP_SIZE:
        p1, p2 = random.sample(selected, 2)
        c1, c2 = crossover(p1, p2)
        offspring.append(mutation(c1))
        offspring.append(mutation(c2))

    # Combine parent + offspring
    population = offspring[:POP_SIZE]

    print(f"Generation {gen} completed")

# -----------------------------
# Final Pareto Front
# -----------------------------
obj_values = [objectives(ind) for ind in population]
fronts = non_dominated_sort(population, obj_values)
pareto_front = fronts[0]

print("\nPareto-optimal solutions:")
for i in pareto_front:
    print(population[i], obj_values[i])
