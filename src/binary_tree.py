import re
import math
import random

import numpy as np
import pandas as pd

from node import Node
from typing import List
from operations import *
from node import NodeType
from collections import deque

class BinaryTree:
    def __init__(self, max_possible_depth: int, variables: List[str], unary_operators: List[str], binary_operators: List[str]):
        self.max_possible_depth = max_possible_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.operators = unary_operators + binary_operators

        self.index = 0
        self.depth = None
        self.fitness = None
        
        self.max_depth = random.randint(1, max_possible_depth)
        self.root = self._build_tree(self.max_depth)
        self.equation = self.build_equation(self.root)

        

    def _build_tree(self, depth: int) -> Node:
        if depth == 1:
            node_value = random.choice(self.variables)
            if node_value == "const":
                node_type = NodeType.CONSTANT
            else:
                node_type = NodeType.VARIABLE

            node_id = f'values_depth_{self.max_depth - depth}_node_{self.index}'
            self.index += 1
            self.depth = self.max_possible_depth

            return Node(
                id=node_id,
                node_type=node_type,
                value=node_value
            )
        else:
            if depth > 2:
                node_id = f"any_operator_and_values_depth_{self.max_depth - depth}_node_{self.index}"
                node_value = random.choice(self.operators + self.variables)
            elif depth == 2:
                node_id = f"unary_operator_and_values_depth_{self.max_depth - depth}_node_{self.index}"
                node_value = random.choice(self.unary_operators + self.variables)

            if node_value in self.operators:
                node_type = NodeType.OPERATOR
            elif node_value == "const":
                node_type = NodeType.CONSTANT
            else:
                node_type = NodeType.VARIABLE
                 
            node = Node(
                id=node_id,
                node_type=node_type,
                value=node_value
            )
            self.index += 1

            if node_value in self.unary_operators:
                node.right = self._build_tree(depth - 1)
            elif node_value in self.binary_operators:
                node.left = self._build_tree(depth - 1)
                node.right = self._build_tree(depth - 1)
            else:
                self.depth = self.max_possible_depth - depth + 1
            
            return node

    def collect_nodes(self):
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

    def build_equation(self, node: Node) -> str:
        if not node:
            return ""
        
        if node.value in self.operators:
            operation = str(OPERATIONS[node.value])
        else:
            operation = str(node.value)
        
        if not node.left and not node.right:
            return operation
        elif node.right and not node.left:
            right_expr = self.build_equation(node.right)
            return f"({operation}({right_expr}))"
        elif node.left and node.right:
            left_expr = self.build_equation(node.left)
            right_expr = self.build_equation(node.right)
            return f"({operation}({left_expr}, {right_expr}))"

    def _mutate_until_success(self, node_to_mutate: Node, operators: List[str]):
        continue_mutation = True
        while(continue_mutation):
            mutated_node_value = random.choice(operators)
            if mutated_node_value != node_to_mutate.value:
                node_to_mutate.value = mutated_node_value
                continue_mutation = False

    def mutate_node(self):
        nodes = self.collect_nodes()

        node_to_mutate = random.choice(nodes)
        if node_to_mutate.value in self.unary_operators:
            self._mutate_until_success(node_to_mutate, self.unary_operators)
        elif node_to_mutate.value in self.binary_operators:
            self._mutate_until_success(node_to_mutate, self.binary_operators)
        elif node_to_mutate.value in self.variables:
            self._mutate_until_success(node_to_mutate, self.variables)

    def print_tree_level_order(self):
        if not self.root:
            return
        
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            print(node.value, end=" ")
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        print()
        
    
    def build_executable_equation(self, X: pd.DataFrame) -> None:
        substitutions = {}
        for col in self.variables:
            substitutions[col] = f"X['{col}'].values"

        executable_equation = self.equation
        for var in self.variables:
            if var in substitutions:
                executable_equation = re.sub(r'\b' + re.escape(var) + r'\b', substitutions[var], executable_equation)
        self.executable_equation = executable_equation

    def calculate_fitness(self, X: pd.DataFrame, y: pd.Series) -> float:
        self.build_executable_equation(X)
        self.fitness = np.mean(np.abs(y.values - eval(self.executable_equation)))