import numpy as np
import pandas as pd

from population import Population
from operations import *

X = pd.DataFrame({
    "x0": np.random.uniform(low=-1000, high=1000, size=1000),
    "x1": np.random.uniform(low=-1000, high=1000, size=1000)
})

y_values = (X["x0"].values**2 - X["x1"].values) / X["x1"].values
y_values = X["x0"].values**2 - 1
y_values = (X["x0"].values**2 - 1) / X["x1"].values ** 2
y = pd.Series(y_values)

variables = list(X.columns)
#unary_operators = ["exp", "Abs", "log", "sin", "cos", "tan", "**0", "**2", "**3", "**-1", "**-2", "**-3"]
unary_operators = ["exp", "abs", "log", "sin", "cos", "tan", "**0", "**2", "**3", "**-1", "**-2", "**-3"]
binary_operators = ["+", "-", "*", "**", "/"]

population = Population(
    num_individuals_per_epoch=10000,
    max_individual_depth=15,
    variables=variables,
    unary_operators=unary_operators,
    binary_operators=binary_operators,
    prob_node_mutation=0.005,
    tournament_ratio=0.5,
    survival_rate=0.7,
    elitism_rate=0.01
)
print("CALCULATE FITNESS...")
population.calculate_fitness(X, y)
print("SORT BY FITNESS...")
population.sort_by_fitness()

import time

times_per_epoch = list()

for epoch in range(1, 50):
    start_time = time.time()

    epoch_fitness_scores = [individual.fitness for individual in population.population]
    best_individual_index = np.argmin(epoch_fitness_scores)
    best_individual = population.population[best_individual_index]

    print("\n==============================================================")
    print(f"BEST INDIVIDUAL FITNESS: {best_individual.fitness}")
    print(f"BEST INDIVIDUAL TREE DEPTH: {best_individual.depth}")
    print(f"BEST INDIVIDUAL EQUATION COMPLEXITY: {best_individual.complexity}")
    print(f"BEST INDIVIDUAL EQUATION: {best_individual.equation}")

    if best_individual.fitness < 1e-6:
        best_individual.visualize_binary_tree()
        break

    print(f"EPOCH={epoch}\n")
    

    print("TOURNAMENT...")
    population.roulette_wheel_selection()
    print("CROSSOVER...")
    population.perform_crossover()
    print("MUTATION...")
    population.perform_mutation()
    print("CALCULATE FITNESS...")
    population.calculate_fitness(X, y)
    print("SORT BY FITNESS...")
    population.sort_by_fitness()
    print("PREPARE NEXT EPOCH POPULATION...")
    population.prepare_next_epoch_population()
    print("CALCULATE FITNESS...")
    population.calculate_fitness(X, y)
    print("SORT BY FITNESS...")
    population.sort_by_fitness()
    times_per_epoch.append(time.time() - start_time)


print(f"MEAN TIME PER EPOCH {np.mean(times_per_epoch)}")



