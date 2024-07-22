import random
import numpy as np

from binary_tree import BinaryTree
from operations import *

variables = ["1.3", "2.1", "3.7"]
unary_operators = ["exp", "abs", "log", "sin", "cos", "tan", "**0", "**2", "**3", "**-1", "**-2", "**-3"]
binary_operators = ["+", "-", "*", "**", "/"]

if __name__ == "__main__":
    max_depth = 10
    for _ in range(100_000):
        tree = BinaryTree(
            max_depth=random.randint(0, max_depth),
            variables=variables,
            unary_operators=unary_operators,
            binary_operators=binary_operators
        )
        print(f"DEPTH={random.randint(0, max_depth)}")
        print(f"EQUATION={tree.equation}")
        print(f"RESULT={eval(tree.equation)}")
        print()
        print("=======================================")
        print()
        