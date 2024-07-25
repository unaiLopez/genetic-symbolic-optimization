import io
import re
import math
import uuid
import random
import warnings
import graphviz

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from node import Node
from typing import List, Optional
from PIL import Image
from operations import *
from node import NodeType
from collections import deque

warnings.filterwarnings('ignore', category=RuntimeWarning)

class BinaryTree:
    def __init__(self, max_possible_depth: int, variables: List[str], unary_operators: List[str], binary_operators: List[str]):
        self.max_possible_depth = max_possible_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.operators = unary_operators + binary_operators

        self.loss = None
        self.depth = None

        self.max_depth = random.randint(1, max_possible_depth)
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
        
    def _calculate_max_depth(self, node: Optional[Node]) -> int:
        if node is None:
            return 0
        left_depth = self._calculate_max_depth(node.left)
        right_depth = self._calculate_max_depth(node.right)
        return max(left_depth, right_depth) + 1

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
    
    def _build_executable_equation(self) -> None:
        substitutions = {}
        for col in self.variables:
            substitutions[col] = f"X['{col}'].values"

        executable_equation = self.equation
        for var in self.variables:
            if var in substitutions:
                executable_equation = re.sub(r'\b' + re.escape(var) + r'\b', substitutions[var], executable_equation)
        
        self.executable_equation = executable_equation

    def update_tree_info(self) -> None:
        self._update_nodes()
        self._update_complexity()
        self.depth = self._calculate_max_depth(self.root)
        self.equation = self._build_equation(self.root)
        self._build_executable_equation()

    def calculate_loss(self, X: pd.DataFrame, y: pd.Series) -> float:
        try:
            loss = np.mean(np.abs(y.values - eval(self.executable_equation)))
        except:
            loss = np.inf
        if math.isinf(loss) or math.isnan(loss):
            loss = np.inf
        self.loss = float(loss)

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