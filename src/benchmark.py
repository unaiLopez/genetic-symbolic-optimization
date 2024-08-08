import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd

from src.operations import *
from src.genetic_symbolic_regressor import GeneticSymbolicRegressor


#hubble's law
def generate_hubbles_law_data():
    BENCHMARK_PATH = os.path.join(os.path.abspath("."), "data/benchmark")
    df = pd.read_csv(os.path.join(BENCHMARK_PATH, "hubble.csv"))
    H0 = 73.3
    print(df)
    print(H0 * df["D"].values)

#keepler's third law
def generate_keeplers_third_law_data():
    BENCHMARK_PATH = os.path.join(os.path.abspath("."), "data/benchmark")
    df = pd.read_csv(os.path.join(BENCHMARK_PATH, "keepler.csv"))
    print(df)

generate_hubbles_law_data()
generate_keeplers_third_law_data()
raise Exception



unary_operators =  ["exp", "abs", "log", "sin", "cos", "tan", "**0", "**2", "**3", "**-1", "**-2", "**-3"]   #CUANDO SOLO HAY UN OPERADOR FALLA LA MUTACION
binary_operators = ["+", "-", "*", "**", "/"]

for i in range(10):
    model = GeneticSymbolicRegressor(
        num_individuals_per_epoch=1000,
        max_individual_depth=5,
        variables=variables,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
        prob_node_mutation=0.02,
        prob_crossover=0.8,
        crossover_retries=3,
        tournament_size=100,
        elitism_ratio=0.01,
        timeout=600,
        stop_score=0.99,
        max_generations=1000,
        verbose=1,
        loss_name="mse",
        score_name="r2",
        random_state=None
    )
    model.fit(X, y)

    with open(f"best_individual_newtons_law_{i}.txt", "w") as f:
        f.write(str(model.search_results.best_individual))
        