import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd

from src.genetic_symbolic_regressor import GeneticSymbolicRegressor


def generate_hooks_law_data():
    k = np.random.uniform(low=1, high=100, size=50)
    original_length = np.random.uniform(low=1, high=100, size=50)
    current_length = original_length + np.random.uniform(low=1, high=25, size=50)

    f = k * (current_length - original_length)

    X = pd.DataFrame({
        "k": k,
        "original_length": original_length,
        "current_length": current_length
    })

    return list(X.columns), X.to_numpy(), f

def generate_newtons_law_data():
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    m_planets = np.array([3.3011e23, 4.8675e24, 5.97237e24, 6.4171e23, 1.8982e27, 5.6834e26, 8.6810e25, 1.02413e26])
    M_sun =  np.array([1.989e30] * len(planets))
    G =  np.array([6.674e-11] * len(planets))
    r_distance_from_sun =  np.array([5.791e10, 1.082e11, 1.496e11, 2.279e11, 7.785e11, 1.433e12, 2.877e12, 4.503e12])
    
    F = G * ((m_planets * M_sun) / (r_distance_from_sun ** 2))

    X = pd.DataFrame({
        "G": G,
        "m_planets": m_planets,
        "M_sun": M_sun,
        "r_distance_from_sun": r_distance_from_sun
    })
    
    return list(X.columns), X.to_numpy(), F

X = pd.DataFrame({
    "x0": np.random.uniform(low=-100, high=100, size=10),
    "x1": np.random.uniform(low=-100, high=100, size=10)
})

#y_values = (X["x0"].values**2 - X["x1"].values) / X["x1"].values
#y_values = X["x0"].values**2 - 1
#y_values = (X["x0"].values**2 - 1) / X["x1"].values ** 2
#y = pd.Series(y_values)


#

unary_operators =  ["exp", "abs", "log", "sin", "cos", "tan", "**0", "**2", "**3", "**-1", "**-2", "**-3"]   #CUANDO SOLO HAY UN OPERADOR FALLA LA MUTACION
binary_operators = ["+", "-", "*", "**", "/"]

df_benchmarking = pd.DataFrame({
    "formula": [],
    "trial": [],
    "score": [],
    "loss": [],
    "equation": [],
    "search_duration": []
})

for formula_name in ["hooks_law", "newtons_universal_law_of_gravity"]:
    if formula_name == "newtons_universal_law_of_gravity":
        variables, X, y = generate_newtons_law_data()
    else:
        variables, X, y = generate_hooks_law_data()
    for i in range(10):
        start_time = time.time()

        model = GeneticSymbolicRegressor(
            num_individuals_per_epoch=2000,
            max_individual_depth=6,
            variables=variables,
            unary_operators=unary_operators,
            binary_operators=binary_operators,
            prob_node_mutation=0.05,
            prob_crossover=0.7,
            crossover_retries=3,
            tournament_size=10,
            elitism_ratio=0.01,
            timeout=600,
            stop_score=0.999,
            max_generations=1000,
            verbose=1,
            loss_name="mse",
            score_name="r2",
            random_state=None
        )
        model.fit(X, y)
        best_individual = model.search_results.best_individual
        df_result = pd.DataFrame({
            "formula": [formula_name],
            "trial": [i],
            "score": [best_individual[2]],
            "loss": [best_individual[1]],
            "equation": [best_individual[5]],
            "search_duration": [time.time() - start_time]
        })
        df_benchmarking = pd.concat([df_benchmarking, df_result], axis=0)
df_benchmarking.to_csv("df_benchmarking.csv", index=False)