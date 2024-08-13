import io
import re
import math
import uuid
import random
import warnings
import graphviz

import numpy as np
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

def count_symbols_frequency(node: List[Any], symbols: List[str], symbol_counts: dict = None) -> dict:
    if symbol_counts is None:
        symbol_counts = {symbol: 0 for symbol in symbols}
    symbol_counts[node[0]] += 1
    if node[1]:
        symbol_counts = count_symbols_frequency(node[1], symbols, symbol_counts)
    if node[2]:
        symbol_counts = count_symbols_frequency(node[2], symbols, symbol_counts)
    
    return symbol_counts

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

def _build_tree(
    depth: int,
    parent: list,
    variables: List[str],
    unary_operators: List[str],
    binary_operators: List[str],
    unary_operators_frequencies: List[float] = None,
    binary_operators_frequencies: List[float] = None,
    variables_frequencies: List[float] = None) -> List[Any]:

    if unary_operators_frequencies is None or binary_operators_frequencies is None or variables_frequencies is None:
        has_weights = False
    else:
        has_weights = True
        
    if depth < 1:
        raise ValueError("Depth must be at least 1")

    if depth == 1:
        if has_weights:
            node_value = random.choices(variables, weights=variables_frequencies)[0]
        else:
            node_value = random.choices(variables)[0]

        return [node_value, None, None]
    else:
        if has_weights:
            node_value = random.choices(binary_operators + unary_operators + variables, weights=unary_operators_frequencies + binary_operators_frequencies + variables_frequencies)[0]
        else:
            node_value = random.choices(binary_operators + unary_operators + variables)[0]

        node = [node_value, None, None]
                
        if node_value in unary_operators:
            node[2] = _build_tree(depth - 1, node, variables, unary_operators, binary_operators, unary_operators_frequencies, binary_operators_frequencies, variables_frequencies)
        elif node_value in binary_operators:
            node[1] = _build_tree(depth - 1, node, variables, unary_operators, binary_operators, unary_operators_frequencies, binary_operators_frequencies, variables_frequencies)
            node[2] = _build_tree(depth - 1, node, variables, unary_operators, binary_operators, unary_operators_frequencies, binary_operators_frequencies, variables_frequencies)

        return node    

def build_full_binary_tree(
    max_initialization_depth: int,
    variables: List[str],
    unary_operators: List[str],
    binary_operators: List[str],
    unary_operators_frequencies: List[float] = None,
    binary_operators_frequencies: List[float] = None,
    variables_frequencies: List[float] = None) -> dict:

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
        unary_operators_frequencies,
        binary_operators_frequencies,
        variables_frequencies
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
        score = -1.0
    if math.isinf(score) or math.isnan(score):
        score = -1.0
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