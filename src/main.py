import random

import numpy as np
import pandas as pd

from population import Population
from binary_tree import BinaryTree
from operations import *

X = pd.DataFrame({
    "x0": np.random.uniform(low=-100, high=100, size=100),
    "x1": np.random.uniform(low=-100, high=100, size=100)
})

y = pd.Series((X["x0"].values**2 - X["x1"].values) / X["x1"].values)


variables = list(X.columns)
unary_operators = ["exp", "abs", "log", "sin", "cos", "tan", "**0", "**2", "**3", "**-1", "**-2", "**-3"]
binary_operators = ["+", "-", "*", "**", "/"]

population = Population(
    num_individuals=10000,
    max_individual_depth=3,
    variables=variables,
    unary_operators=unary_operators,
    binary_operators=binary_operators,
    prob_node_mutation = 0.01,
    prob_crossover = 0.5
)
population.calculate_fitness(X, y)
population.sort_by_fitness()
for individual in population.population:
    print("PREVIOUS TO MUTATION")
    individual.print_tree_level_order()
    individual.mutate_node()
    print("AFTER MUTATION")
    individual.print_tree_level_order()
    print("====================================")