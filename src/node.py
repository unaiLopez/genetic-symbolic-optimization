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
        """Returns 'left' if this node is the left child of its parent,
        'right' if this node is the right child, and None if it has no parent."""
        if self.parent is None:
            return None
        if self.parent.left is self:
            return 'left'
        if self.parent.right is self:
            return 'right'
        return None