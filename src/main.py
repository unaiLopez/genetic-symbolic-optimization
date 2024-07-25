import numpy as np
import pandas as pd

from operations import *
from genetic_symbolic_regressor import GeneticSymbolicRegressor


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
    G = [6.67428e-11] * 50
    m1 = np.random.randint(low=1e10, high=1e11, size=50)
    m2 = np.random.randint(low=1e10, high=1e12, size=50)
    r = np.random.randint(low=1e7, high=1e9, size=50)

    F = G * ((m1 * m2) / (r ** 2))

    X = pd.DataFrame({
        "G": G,
        "m1": m1,
        "m2": m2,
        "r": r
    })
    y = pd.Series(F)
    
    return X, y

X = pd.DataFrame({
    "x0": np.random.uniform(low=-1000, high=1000, size=1000),
    "x1": np.random.uniform(low=-1000, high=1000, size=1000)
})

y_values = (X["x0"].values**2 - X["x1"].values) / X["x1"].values

#y_values = X["x0"].values**2 - 1
#y_values = (X["x0"].values**2 - 1) / X["x1"].values ** 2
y = pd.Series(y_values)

X, y = generate_hooks_law_data()
X, y = generate_newtons_law_data()
print(X)
print(y)

variables = list(X.columns)
unary_operators = []#["exp", "abs", "log", "sin", "cos", "tan", "**0", "**2", "**3", "**-1", "**-2", "**-3"]
binary_operators = ["+", "-", "*", "**", "/"]

model = GeneticSymbolicRegressor(
    num_individuals_per_epoch=1000,
    max_individual_depth=6, #ESTO HAY QUE REVISAR A VECES LA PROFUNDIDAD DEL ARBOLES ES MAYOR. AUNQUE ES POR EL CROSSOVER. AHI REVISAR.
    variables=variables,
    unary_operators=unary_operators,
    binary_operators=binary_operators,
    prob_node_mutation=0.015,
    tournament_ratio=0.7,
    elitism_ratio=0.01,
    timeout=120,
    stop_loss=1e-15,
    max_generations=100,
    verbose=1,
    loss_function="mae",    #THIS IS NOT IMPLEMENTED YET
    random_state=None
)
model.fit(X, y)