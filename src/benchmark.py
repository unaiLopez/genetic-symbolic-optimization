import os
import sys
import time

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd

from src.operations import *
from src.gradient_descent_symbolic_regressor import GradientDescentSymbolicRegressor

def generate_hubbles_law_data():
    BENCHMARK_PATH = os.path.join(os.path.abspath("."), "data/benchmark")
    df = pd.read_csv(os.path.join(BENCHMARK_PATH, "hubble.csv"))
    H0 = 73.3
    print(df)
    print(H0 * df["D"].values)

def generate_keeplers_third_law_data():
    BENCHMARK_PATH = os.path.join(os.path.abspath("."), "data/benchmark")
    df = pd.read_csv(os.path.join(BENCHMARK_PATH, "keepler.csv"))
    print(df)

def generate_newtons_law_data_2():
    m = np.random.uniform(low=1e15, high=1e35, size=50)
    M = np.random.uniform(low=1e40, high=1e50, size=50)
    r_distance = np.random.uniform(low=1e6, high=1e15, size=50)
    G =  np.array([6.674e-11] * len(M))

    F = G * ((m * M) / (r_distance ** 2))

    X = pd.DataFrame({
        "G": G,
        "m": m,
        "M": M,
        "r_distance": r_distance
    })
    
    return list(X.columns), X.to_numpy(), F



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

def generate_hooks_law_data():
    k = np.random.uniform(low=1, high=14, size=50)
    original_length = np.random.uniform(low=1, high=1000, size=50)
    current_length = original_length + np.random.uniform(low=1, high=25, size=50)

    f = k * (current_length - original_length)

    X = pd.DataFrame({
        "k": k,
        "original_length": original_length,
        "current_length": current_length
    })

    return list(X.columns), X.to_numpy(), f

if __name__ == "__main__":
    unary_operators =  ["exp", "abs", "log", "sin", "cos", "tan", "**0", "**2", "**3", "**-1", "**-2", "**-3"]
    binary_operators = ["+", "-", "*", "**", "/"]
    df_benchmarking = pd.DataFrame({
        "formula": [],
        "trial": [],
        "score": [],
        "loss": [],
        "equation": [],
        "search_duration": []
    })
    num_trials = 10

    for formula_name in ["newtons_universal_law_of_gravity", "hooks_law"]:
        if formula_name == "newtons_universal_law_of_gravity":
            variables, X, y = generate_newtons_law_data()
        else:
            variables, X, y = generate_hooks_law_data()

        for i in range(num_trials):
            start_time = time.time()

            model = GradientDescentSymbolicRegressor(
                num_individuals_per_sample=500,
                max_individual_depth=4,
                variables=variables,
                unary_operators=unary_operators,
                binary_operators=binary_operators,
                timeout=600,
                stop_score=0.999,
                max_iterations=1000,
                probs_learning_rate=1e-4,
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
    print(df_benchmarking)
