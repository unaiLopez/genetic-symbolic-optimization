import numpy as np

def power_0(val):
    return 1

def power_2(val):
    return val**2

def power_3(val):
    return val**3

def power_minus1(val):
    return np.power(float(val), -1)

def power_minus2(val):
    return np.power(float(val), -2)

def power_minus3(val):
    return np.power(float(val), -3)

OPERATIONS = {
    "-": "np.subtract",
    "+": "np.add",
    "/": "np.divide",
    "*": "np.multiply",
    "**": "np.power",
    "sin": "np.sin",
    "cos": "np.cos",
    "tan": "np.tan",
    "abs": "np.abs",
    "log": "np.log",
    "exp": "np.exp",
    "**0": "power_0",
    "**2": "power_2",
    "**3": "power_3",
    "**-1": "power_minus1",
    "**-2": "power_minus2",
    "**-3": "power_minus3"
}