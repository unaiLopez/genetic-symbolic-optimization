from enum import Enum
from collections import deque
from typing import Optional, List

class NodeType:
    UNARY_OPERATOR = "unary_operator"
    BINARY_OPERATOR = "binary_operator"
    VARIABLE = "variable"
    CONSTANT = "constant"

class Node:
    def __init__(self, id: str, node_type: NodeType, value: str, parent: Optional["Node"], left: Optional["Node"] = None, right: Optional["Node"] = None):
        self.id = id
        self.node_type = node_type 
        self.value = value
        self.parent = parent
        self.left = left
        self.right = right

    def parent_side(self):
        if self.parent is None:
            return None
        if self.parent.left is self:
            return 'left'
        if self.parent.right is self:
            return 'right'
        return None
    
    def calculate_max_depth(self) -> int:
        if self is None:
            return 0
        left_depth = self.left.calculate_max_depth() if self.left else 0
        right_depth = self.right.calculate_max_depth() if self.right else 0
        return max(left_depth, right_depth) + 1