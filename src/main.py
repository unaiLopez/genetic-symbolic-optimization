import os
import sys

sys.path.append(os.path.abspath(os.curdir))
print(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd

from src.operations import *
from src.genetic_symbolic_regressor import GeneticSymbolicRegressor


def generate_hooks_law_data():
    k = np.random.uniform(low=1, high=100, size=50)
    original_length = np.random.uniform(low=1, high=100, size=50)
    current_length = original_length + np.random.uniform(low=1, high=25, size=50)

    f = k * (current_length - original_length)

    X = pd.DataFrame({
        "k": k,
        "original_lenght": original_length,
        "current_length": current_length
    })
    y = pd.Series(f)

    return X, y

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
    y = pd.Series(F)
    
    return X, y

X = pd.DataFrame({
    "x0": np.random.uniform(low=-1000, high=1000, size=1000),
    "x1": np.random.uniform(low=-1000, high=1000, size=1000)
})

y_values = (X["x0"].values**2 - X["x1"].values) / X["x1"].values

#y_values = X["x0"].values**2 - 1
y_values = (X["x0"].values**2 - 1) / X["x1"].values ** 2
y = pd.Series(y_values)

#X, y = generate_hooks_law_data()
#X, y = generate_newtons_law_data()

variables = list(X.columns)
unary_operators = ["exp", "abs", "log", "sin", "cos", "tan", "**0", "**2", "**3", "**-1", "**-2", "**-3"]   #CUANDO SOLO HAY UN OPERADOR FALLA LA MUTACION
binary_operators = ["+", "-", "*", "**", "/"]

model = GeneticSymbolicRegressor(
    num_individuals_per_epoch=250,
    max_initialization_individual_depth=5, #ESTO HAY QUE REVISAR A VECES LA PROFUNDIDAD DEL ARBOL ES MAYOR. AUNQUE ES POR EL CROSSOVER. AÑADIR RESTRICCIONES.
    variables=variables,
    unary_operators=unary_operators,
    binary_operators=binary_operators,
    prob_node_mutation=0.05,
    tournament_ratio=0.75,
    elitism_ratio=0.01,
    timeout=600,
    stop_loss=1e-6,
    max_generations=2500,
    verbose=1,
    loss_function="mae",    #THIS IS NOT IMPLEMENTED YET
    random_state=None
)
model.fit(X, y)