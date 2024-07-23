from enum import Enum
from typing import Optional

class NodeType:
    UNARY_OPERATOR = "unary_operator"
    BINARY_OPERATOR = "binary_operator"
    VARIABLE = "variable"
    CONSTANT = "constant"

class Node:
    def __init__(self, id: str, node_type: NodeType, value: str, left: Optional["Node"] = None, right: Optional["Node"] = None):
        self.id = id
        self.node_type = node_type 
        self.value = value
        self.left = left
        self.right = right