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

from PIL import Image
from typing import List, Callable, Any
from src.operations import *

warnings.filterwarnings('ignore', category=RuntimeWarning)

def _get_complexity(node: List[Any]) -> int:
    if node is None:
        return 0
    
    count = 1
    count += _get_complexity(node[1])
    count += _get_complexity(node[2])
    
    return count

def _calculate_max_depth(node: List[Any]) -> int:
    if node == []:
        return 0

    left_depth = _calculate_max_depth(node[1]) if node[1] else 0
    right_depth = _calculate_max_depth(node[2]) if node[2] else 0
    return max(left_depth, right_depth) + 1

def _build_equation(node: List[Any], operators: List[str]) -> str:
    if node[0] in operators:
        operation = str(OPERATIONS[node[0]])
    else:
        operation = str(node[0])
    
    if not node[1] and not node[2]:
        return operation
    elif node[2] and not node[1]:
        right_expr = _build_equation(node[2], operators)
        return f"({operation}({right_expr}))"
    elif node[1] and node[2]:
        left_expr = _build_equation(node[1], operators)
        right_expr = _build_equation(node[2], operators)
        return f"({operation}({left_expr}, {right_expr}))"

def _build_executable_equation(equation: str, variables: List[str]) -> str:
    substitutions = {}
    for i, col in enumerate(variables):
        substitutions[col] = f"X[:, {i}]"

    executable_equation = equation
    for var in variables:
        if var in substitutions:
            executable_equation = re.sub(r'\b' + re.escape(var) + r'\b', substitutions[var], executable_equation)
    
    return executable_equation

def update_tree_info(
    tree: List[Any],
    operators: List[str],
    variables: List[str]) -> dict:

    tree[4] = _calculate_max_depth(tree[-1])                      #update depth
    tree[5] = _build_equation(tree[-1], operators)                #update equation
    tree[6] = _build_executable_equation(tree[5], variables)      #update executable equation
    tree[7] = _get_complexity(tree[-1])                           #update complexity

    return tree

def _calculate_index(depth: int, position: int) -> int:
    # Calculate the index for a node based on its depth and position within the tree
    return (2 ** (depth - 1)) - 1 + (position - 1)

def _build_tree(
    depth: int,
    parent: list,
    variables: List[str],
    unary_operators: List[str],
    binary_operators: List[str],
    unary_operators_probs: np.ndarray,
    binary_operators_probs: np.ndarray,
    variables_probs: np.ndarray,
    current_depth: int = 1,
    position: int = 1) -> List[Any]:

    if depth < 1:
        raise ValueError("Depth must be at least 1")

    index = _calculate_index(current_depth, position)
    if depth == 1:
        node_value = np.random.choice(variables, p=(variables_probs[index] / variables_probs[index].sum()))
        return [str(node_value), None, None, index]
    else:
        node_value = np.random.choice(binary_operators + unary_operators + variables, p=np.concatenate((unary_operators_probs[index], binary_operators_probs[index], variables_probs[index])))
        node = [str(node_value), None, None, index]
                
        if node_value in unary_operators:
            node[2] = _build_tree(
                depth - 1,
                node,
                variables,
                unary_operators,
                binary_operators,
                unary_operators_probs,
                binary_operators_probs,
                variables_probs,
                current_depth + 1,
                position * 2
            )
        elif node_value in binary_operators:
            node[1] = _build_tree(
                depth - 1,
                node,
                variables,
                unary_operators,
                binary_operators,
                unary_operators_probs,
                binary_operators_probs,
                variables_probs,
                current_depth + 1,
                position * 2 - 1
            )
            node[2] = _build_tree(
                depth - 1,
                node,
                variables,
                unary_operators,
                binary_operators,
                unary_operators_probs,
                binary_operators_probs,
                variables_probs,
                current_depth + 1,
                position * 2
            )
        return node    

def build_full_binary_tree(
    max_initialization_depth: int,
    variables: List[str],
    unary_operators: List[str],
    binary_operators: List[str],
    unary_operators_probs: np.ndarray,
    binary_operators_probs: np.ndarray,
    variables_probs: np.ndarray) -> List[Any]:

    full_binary_tree = [
        max_initialization_depth,                       #max_initialization_depth
        None,                                           #loss
        None,                                           #score
        random.randint(1, max_initialization_depth),    #max_depth
        None,                                           #depth
        None,                                           #equation
        None,                                           #executable_equation
        None                                            #complexity
    ]

    tree = _build_tree(
        full_binary_tree[3],
        [],
        variables,
        unary_operators,
        binary_operators,
        unary_operators_probs,
        binary_operators_probs,
        variables_probs
    )
    full_binary_tree.append(tree)
    full_binary_tree = update_tree_info(
        full_binary_tree,
        unary_operators + binary_operators,
        variables
    )

    return full_binary_tree


def calculate_loss(
    X: np.ndarray,
    y: np.ndarray,
    loss_function: Callable,
    executable_equation: str,
    worst_loss: float) -> float:
    
    try:
        loss = loss_function(y, eval(executable_equation))
    except:
        loss = np.nan
    loss = np.nan_to_num(loss, nan=worst_loss, posinf=worst_loss, neginf=worst_loss)

    return float(loss)

def calculate_score(
    X: np.ndarray,
    y: np.ndarray,
    score_function: Callable,
    executable_equation: str,
    worst_score: float) -> float:

    try:
        score = score_function(y, eval(executable_equation))
    except:
        score = np.nan
    score = np.nan_to_num(score, nan=worst_score, posinf=worst_score, neginf=worst_score)
    
    return float(score)

def visualize_binary_tree(node: List[Any], variables: List[str]) -> None:
    graph = graphviz.Digraph()
    parent_node_id = str(uuid.uuid1())
    graph.node(name=parent_node_id, label=node[0], style="filled", fillcolor="red")

    def _add_nodes_edges(node: List[Any], node_id: str) -> None:
        node_left_id = str(uuid.uuid1())
        node_right_id = str(uuid.uuid1())
        if node[1]:
            if node[1][0] not in variables + ["const"]:
                left_fill_color = "red"
                shape = None
            else:
                left_fill_color = "green"
                shape = "rectangle"

            graph.node(name=node_left_id, label=node[1][0], shape=shape, style="filled", fillcolor=left_fill_color)
            graph.edge(node_id, node_left_id)
            _add_nodes_edges(node[1], node_left_id)

        if node[2]:
            if node[2][0] not in list(variables) + ["const"]:
                right_fill_color = "red"
                shape = None
            else:
                right_fill_color = "green"
                shape = "rectangle"

            graph.node(name=node_right_id, label=node[2][0], shape=shape, style="filled", fillcolor=right_fill_color)
            graph.edge(node_id, node_right_id)
            _add_nodes_edges(node[2], node_right_id)

    _add_nodes_edges(node, parent_node_id)

    #graph_path = "equation_tree.png"
    #graph.render(graph_path, format="png", cleanup=True)
    graph_data = graph.pipe(format="png")
    img = Image.open(io.BytesIO(graph_data))
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.show()