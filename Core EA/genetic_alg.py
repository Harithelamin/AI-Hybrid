# Evolutionary algorithms are population-based metaheuristics
#  inspired by natural evolution. 
# They are effective for non-linear, constrained, 
# and multi-objective problems where gradient-based methods fail. 
# Multi-objective algorithms like NSGA-II 
# allow discovering diverse trade-off solutions in a single 
# run while preserving convergence and diversity.

# A Genetic Algorithm is an optimization algorithm 
# inspired by natural evolution.
# It searches for the best solution by 
# evolving a population of candidate solutions over generations.

# 1. Initialize
# 2. Evaluate
# 3. Selection
# 4. Crossover
# 5. Mutation
# 6. Replacement

# The Genetic Algorithm concept
# 1. Population-based (global search)
# 2. Works with non-linear, non-differentiable problems
# 3. Stochastic to avoids local minima

# parameters. 
# Population size, 
# Crossover rate, 
# Mutation rate, 
# Selection method (tournament, roulette)


import random
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
POP_SIZE = 20
NUM_GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
GENE_LENGTH = 5

# -----------------------------
# Helper Functions
# -----------------------------

def initialize_population():
    return [np.random.rand(GENE_LENGTH) for _ in range(POP_SIZE)]

def fitness(individual):
    # Example: maximize sum of genes
    return np.sum(individual)

def selection(population, fitness_values):
    # Tournament selection
    selected = []
    for _ in range(POP_SIZE):
        i, j = random.sample(range(POP_SIZE), 2)
        winner = population[i] if fitness_values[i] > fitness_values[j] else population[j]
        selected.append(winner)
    return selected

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENE_LENGTH - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutation(individual):
    for i in range(GENE_LENGTH):
        if random.random() < MUTATION_RATE:
            individual[i] = random.random()
    return individual

# -----------------------------
# GA Main Loop
# -----------------------------

population = initialize_population()

for generation in range(NUM_GENERATIONS):

    # Evaluate fitness
    fitness_values = [fitness(ind) for ind in population]

    # Best individual
    best_idx = np.argmax(fitness_values)
    best_fitness = fitness_values[best_idx]

    print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")

    # Selection
    selected = selection(population, fitness_values)

    # Create next generation
    new_population = []

    for i in range(0, POP_SIZE, 2):
        parent1 = selected[i]
        parent2 = selected[i + 1]

        child1, child2 = crossover(parent1, parent2)

        new_population.append(mutation(child1))
        new_population.append(mutation(child2))

    population = new_population

# Final result
fitness_values = [fitness(ind) for ind in population]
best_idx = np.argmax(fitness_values)

print("\nBest solution:", population[best_idx])
print("Best fitness:", fitness_values[best_idx])
