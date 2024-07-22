import random
import operations

import numpy as np

from node import Node
from typing import List
from node import NodeType

class BinaryTree:
    def __init__(self, max_depth: int, variables: List[str], unary_operators: List[str], binary_operators: List[str]):
        self.max_depth = max_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.operators = unary_operators + binary_operators

        self.index = 0
        self.root = self._build_tree(max_depth)
        self.equation = self.build_equation(self.root)

    def _build_tree(self, depth: int) -> Node:
        if depth == 0:
            node_value = random.choices(self.variables, k=1)[0]
            if node_value == "const":
                node_type = NodeType.CONSTANT
            else:
                node_type = NodeType.VARIABLE

            node_id = f'values_depth_{self.max_depth - depth}_node_{self.index}'
            self.index += 1

            return Node(
                id=node_id,
                node_type=node_type,
                value=node_value
            )
        else:
            if depth > 1:
                node_id = f"any_operator_and_values_depth_{self.max_depth - depth}_node_{self.index}"
                node_value = random.choices(self.operators + self.variables, k=1)[0]
            elif depth == 1:
                node_id = f"unary_operator_and_values_depth_{self.max_depth - depth}_node_{self.index}"
                node_value = random.choices(self.unary_operators + self.variables)[0]

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
            
            return node

    def build_equation(self, node: Node) -> str:
        if not node:
            return ""
        
        if node.value in self.operators:
            operation = str(operations.OPERATIONS[node.value])
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