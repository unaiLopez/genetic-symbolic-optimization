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

class BinaryTree:
    def __init__(self, max_initialization_depth: int, variables: List[str], unary_operators: List[str], binary_operators: List[str]):
        self.max_initialization_depth = max_initialization_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.operators = unary_operators + binary_operators

        self.loss = None
        self.score = None
        self.depth = None

        self.max_depth = random.randint(1, max_initialization_depth)
        self.root = self._build_tree(self.max_depth)
        self.executable_equation = None
        self.nodes = None
        self.complexity = None

        self.update_tree_info()

    def _update_nodes(self):
        self.nodes = self._collect_nodes()
    
    def _update_complexity(self):
        self.complexity = len(self.nodes)

    def _build_tree(self, depth: int, parent: Node = None) -> Node:
        if depth < 1:
            raise ValueError("Depth must be at least 1")

        if depth == 1:
            node_value = random.choice(self.variables)
            if node_value == "const":
                node_type = NodeType.CONSTANT
            else:
                node_type = NodeType.VARIABLE

            return Node(
                id=uuid.uuid1(),
                node_type=node_type,
                value=node_value,
                parent=parent
            )
        else:
            if depth > 2:
                node_value = random.choice(self.operators + self.variables)
            elif depth == 2:
                node_value = random.choice(self.unary_operators + self.variables)

            if node_value in self.unary_operators:
                node_type = NodeType.UNARY_OPERATOR
            elif node_value in self.binary_operators:
                node_type = NodeType.BINARY_OPERATOR
            elif node_value == "const":
                node_type = NodeType.CONSTANT
            else:
                node_type = NodeType.VARIABLE
                 
            node = Node(
                id=uuid.uuid1(),
                node_type=node_type,
                value=node_value,
                parent=parent
            )

            if node_value in self.unary_operators:
                node.right = self._build_tree(depth - 1, node)
            elif node_value in self.binary_operators:
                node.left = self._build_tree(depth - 1, node)
                node.right = self._build_tree(depth - 1, node)
        
            return node

    def _collect_nodes(self):
        if not self.root:
            return []
        
        nodes = []
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            nodes.append(node)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return nodes

    def evaluate(self, X: np.ndarray) -> float:
        return self.root.evaluate(X)

    def _build_equation(self, node: Node) -> None:
        if node.value in self.operators:
            operation = str(OPERATIONS[node.value])
        else:
            operation = str(node.value)
        
        if not node.left and not node.right:
            return operation
        elif node.right and not node.left:
            right_expr = self._build_equation(node.right)
            return f"({operation}({right_expr}))"
        elif node.left and node.right:
            left_expr = self._build_equation(node.left)
            right_expr = self._build_equation(node.right)
            return f"({operation}({left_expr}, {right_expr}))"

    def perform_mutation(self) -> None:
        node_to_mutate = random.choice(self.nodes)
        if node_to_mutate.node_type == NodeType.UNARY_OPERATOR:
            temp_operators = self.unary_operators.copy()
            temp_operators.remove(node_to_mutate.value)
        elif node_to_mutate.node_type == NodeType.BINARY_OPERATOR:
            temp_operators = self.binary_operators.copy()
            temp_operators.remove(node_to_mutate.value)
        elif node_to_mutate.node_type == NodeType.VARIABLE:
            temp_operators = self.variables.copy()
            temp_operators.remove(node_to_mutate.value)

        mutated_node_value = random.choice(temp_operators)
        node_to_mutate.value = mutated_node_value

    def perform_subtree_mutation(self):
        #TODO
        """
        Selection of Mutation Point:
        Randomly select a node in the binary tree. This node will be the root of the subtree that will be replaced.

        Generation of New Subtree:
        Generate a new random subtree. This subtree can be generated using the same rules and functions that were used to create the initial population of trees.

        Replacement:
        Replace the selected node and its subtree with the new randomly generated subtree.
        """
        pass
        

    def perform_node_mutation(self):
        #TODO
        """
        Randomly select a node in the tree and replace its value with another valid value (e.g., replace an operator with another operator or a variable with another variable).
        """
        pass
    
    def perform_hoist_mutation(self):
        #TODO
        """
        Select a random subtree and replace the entire tree with this subtree, effectively "hoisting" it up to the root.
        """
        pass

    def perform_shrink_mutation(self):
        #TODO
        """
        Select a subtree and replace it with one of its subtrees, effectively shrinking the tree.
        """
        pass

    def _build_executable_equation(self) -> None:
        substitutions = {}
        for i, col in enumerate(self.variables):
            substitutions[col] = f"X[:, {i}]"

        executable_equation = self.equation
        for var in self.variables:
            if var in substitutions:
                executable_equation = re.sub(r'\b' + re.escape(var) + r'\b', substitutions[var], executable_equation)
        
        self.executable_equation = executable_equation

    def update_tree_info(self) -> None:
        self._update_nodes()
        self._update_complexity()
        self.depth = self.root.calculate_max_depth()
        self.equation = self._build_equation(self.root)
        self._build_executable_equation()

    def calculate_loss(self, X: np.ndarray, y: np.ndarray, loss_function: Callable) -> None:
        try:
            loss = loss_function(y, eval(self.executable_equation))
        except:
            loss = np.inf
        if math.isinf(loss) or math.isnan(loss):
            loss = np.inf
        self.loss = float(loss)
    
    def calculate_score(self, X: np.ndarray, y: np.ndarray, score_function: Callable) -> None:
        try:
            score = score_function(y, eval(self.executable_equation))
        except:
            score = np.inf
        if math.isinf(score) or math.isnan(score):
            score = 0
        self.score = float(score)

    def visualize_binary_tree(self) -> None:
        graph = graphviz.Digraph()
        graph.node(name=str(self.root.id), label=str(self.root.value), style="filled", fillcolor="red")

        def add_nodes_edges(node):
            if node.left:
                if node.left.value not in list(self.variables) + ["const"]:
                    left_fill_color = "red"
                    shape = None
                else:
                    left_fill_color = "green"
                    shape = "rectangle"

                graph.node(name=str(node.left.id), label=str(node.left.value), shape=shape, style="filled", fillcolor=left_fill_color)
                graph.edge(str(node.id), str(node.left.id))
                add_nodes_edges(node.left)
            if node.right:
                if node.right.value not in list(self.variables) + ["const"]:
                    right_fill_color = "red"
                    shape = None
                else:
                    right_fill_color = "green"
                    shape = "rectangle"

                graph.node(name=str(node.right.id), label=str(node.right.value), shape=shape, style="filled", fillcolor=right_fill_color)
                graph.edge(str(node.id), str(node.right.id))
                add_nodes_edges(node.right)

        add_nodes_edges(self.root)

        #graph_path = 'equation_tree.png'
        #graph.render(graph_path, format='png', cleanup=True)
        graph_data = graph.pipe(format='png')
        img = Image.open(io.BytesIO(graph_data))
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()