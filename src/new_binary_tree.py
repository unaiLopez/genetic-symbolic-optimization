import io
import re
import os
import sys
import math
import uuid
import random
import warnings
import graphviz

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from numba import jit
from PIL import Image
from typing import List, Callable
from src.operations import *
from collections import deque
from src.node import Node, NodeType

warnings.filterwarnings('ignore', category=RuntimeWarning)

def build_tree(
    depth: int,
    parent: dict,
    variables: List[str],
    unary_operators: List[str],
    binary_operators: List[str]) -> dict:

    node = {}
    if depth < 1:
        raise ValueError("Depth must be at least 1")

    if depth == 1:
        node_value = random.choice(variables)
        node[node_value] = {
            "left": None,
            "right": None
        }
        return node
    else:
        if depth > 2:
            node_value = random.choice(binary_operators + unary_operators + variables)
        elif depth == 2:
            node_value = random.choice(unary_operators + variables)

        node[node_value] = {
            "left": None,
            "right": None
        }
                
        if node_value in unary_operators:
            node[node_value]["right"] = build_tree(depth - 1, node, variables, unary_operators, binary_operators)
        elif node_value in binary_operators:
            node[node_value]["left"] = build_tree(depth - 1, node, variables, unary_operators, binary_operators)
            node[node_value]["right"] = build_tree(depth - 1, node, variables, unary_operators, binary_operators)

        return node    

def build_full_binary_tree(
    max_initialization_depth: int,
    variables: List[str],
    unary_operators: List[str],
    binary_operators: List[str]) -> dict:

    full_binary_tree = {
        "max_initialization_depth": max_initialization_depth,
        "loss": None,
        "score": None,
        "max_depth": random.randint(1, max_initialization_depth),
        "depth": None,
        "executable_equation": None,
        "complexity": None
    }

    tree = build_tree(
        full_binary_tree["max_depth"],
        {},
        variables,
        unary_operators,
        binary_operators
    )
    full_binary_tree["tree"] = tree

    return full_binary_tree

def build_equation(node: dict, operators: List[str]) -> str:
    node_value = list(node.keys())[0]
    node_value_children = node[node_value]
    if node_value in operators:
        operation = str(OPERATIONS[node_value])
    else:
        operation = str(node_value)
    
    if not node_value_children["left"] and not node_value_children["right"]:
        return operation
    elif node_value_children["right"] and not node_value_children["left"]:
        right_expr = build_equation(node_value_children["right"], operators)
        return f"({operation}({right_expr}))"
    elif node_value_children["left"] and node_value_children["right"]:
        left_expr = build_equation(node_value_children["left"], operators)
        right_expr = build_equation(node_value_children["right"], operators)
        return f"({operation}({left_expr}, {right_expr}))"
    
def build_executable_equation(equation: str, variables: List[str]) -> str:
    substitutions = {}
    for i, col in enumerate(variables):
        substitutions[col] = f"X[:, {i}]"

    executable_equation = equation
    for var in variables:
        if var in substitutions:
            executable_equation = re.sub(r'\b' + re.escape(var) + r'\b', substitutions[var], executable_equation)
    
    return executable_equation

def calculate_loss(
    X: np.ndarray,
    y: np.ndarray,
    loss_function: Callable,
    executable_equation: str) -> float:
    
    try:
        loss = loss_function(y, eval(executable_equation))
    except:
        loss = np.inf
    if math.isinf(loss) or math.isnan(loss):
        loss = np.inf
    return float(loss)
    
def calculate_score(
    X: np.ndarray,
    y: np.ndarray,
    score_function: Callable,
    executable_equation: str) -> float:

    try:
        score = score_function(y, eval(executable_equation))
    except:
        score = np.inf
    if math.isinf(score) or math.isnan(score):
        score = 0
    return float(score)

def get_complexity(node: dict) -> int:
    if node is None:
        return 0
    
    count = 1
    left_child = list(node.values())[0]["left"]
    right_child = list(node.values())[0]["right"]
    
    count += get_complexity(left_child)
    count += get_complexity(right_child)
    
    return count

def calculate_max_depth(tree: dict) -> int:
    if tree == {}:
        return 0

    node = tree[list(tree.keys())[0]]
    left_depth = calculate_max_depth(node["left"]) if node["left"] else 0
    right_depth = calculate_max_depth(node["right"]) if node["right"] else 0
    return max(left_depth, right_depth) + 1

def update_tree_info(
    tree: dict,
    operators: List[str],
    variables: List[str]) -> dict:

    tree["complexity"] = get_complexity(tree["tree"])
    tree["depth"] = calculate_max_depth(tree["tree"])
    tree["equation"] = build_equation(tree["tree"], operators)
    tree["executable_equation"] = build_executable_equation(tree["equation"], variables)

def visualize_binary_tree(node: dict, variables: List[str]) -> None:
    graph = graphviz.Digraph()
    parent_node_id = str(uuid.uuid1())
    graph.node(name=parent_node_id, label=str(list(node.keys())[0]), style="filled", fillcolor="red")

    def add_nodes_edges(node: dict, node_id: str) -> None:
        node_left_id = str(uuid.uuid1())
        node_right_id = str(uuid.uuid1())
        
        node_value = list(node.keys())[0]
        node_value_children = node[node_value]
        if node_value_children["left"]:
            node_value_left = list(node_value_children["left"].keys())[0]
            if node_value_left not in variables + ["const"]:
                left_fill_color = "red"
                shape = None
            else:
                left_fill_color = "green"
                shape = "rectangle"


            graph.node(name=node_left_id, label=node_value_left, shape=shape, style="filled", fillcolor=left_fill_color)
            graph.edge(node_id, node_left_id)
            add_nodes_edges(node_value_children["left"], node_left_id)
        if node_value_children["right"]:
            node_value_right = list(node_value_children["right"].keys())[0]
            if node_value_right not in list(variables) + ["const"]:
                right_fill_color = "red"
                shape = None
            else:
                right_fill_color = "green"
                shape = "rectangle"

            graph.node(name=node_right_id, label=node_value_right, shape=shape, style="filled", fillcolor=right_fill_color)
            graph.edge(node_id, node_right_id)
            add_nodes_edges(node_value_children["right"], node_right_id)

    add_nodes_edges(node, parent_node_id)

    #graph_path = "equation_tree.png"
    #graph.render(graph_path, format="png", cleanup=True)
    graph_data = graph.pipe(format="png")
    img = Image.open(io.BytesIO(graph_data))
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    from pprint import pprint

    variables = ["x0", "x1"]
    unary_operators = ["sin", "cos", "tan", "log", "abs", "exp"]
    binary_operators = ["+", "-", "*", "**", "/"]

    for _ in range(10):
        my_binary_tree = build_full_binary_tree(7, variables, unary_operators, binary_operators)
        print("THIS IS THE TREE")
        pprint(my_binary_tree)

        print("THIS IS THE NUMPY EQUATION")
        equation = build_equation(my_binary_tree["tree"], unary_operators + binary_operators)
        print(equation)

        print("THIS IS THE EXECUTABLE NUMPY EQUATION")
        executable_equation = build_executable_equation(equation, variables)
        print(executable_equation)

        print("TREE DEPTH")
        print(calculate_max_depth(my_binary_tree["tree"]))

        print("COMPLEXITY")
        print(get_complexity(my_binary_tree["tree"]))

        print("VISUALIZE TREE")
        visualize_binary_tree(my_binary_tree["tree"], variables)

        