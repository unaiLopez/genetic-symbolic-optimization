import os
import sys
import unittest

sys.path.append(os.path.abspath(os.curdir))

from src.node import Node
from src.node import NodeType

class TestNode(unittest.TestCase):
    
    def test_node_creation_with_no_children(self):
        """Test creating a node with no children."""
        node = Node(id="1", node_type=NodeType.CONSTANT, value="5", parent=None)
        self.assertEqual(node.id, "1")
        self.assertEqual(node.node_type, NodeType.CONSTANT)
        self.assertEqual(node.value, "5")
        self.assertIsNone(node.parent)
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)
        
    def test_node_creation_with_children(self):
        """Test creating a node with left and right children."""
        parent_node = Node(id="1", node_type=NodeType.BINARY_OPERATOR, value="+", parent=None)
        left_child = Node(id="2", node_type=NodeType.VARIABLE, value="x", parent=parent_node)
        right_child = Node(id="3", node_type=NodeType.VARIABLE, value="y", parent=parent_node)
        parent_node.left = left_child
        parent_node.right = right_child

        self.assertEqual(parent_node.left, left_child)
        self.assertEqual(parent_node.right, right_child)
        self.assertEqual(left_child.parent, parent_node)
        self.assertEqual(right_child.parent, parent_node)

    def test_parent_side_left(self):
        """Test the parent_side method when the node is the left child."""
        parent_node = Node(id="1", node_type=NodeType.BINARY_OPERATOR, value="+", parent=None)
        left_child = Node(id="2", node_type=NodeType.VARIABLE, value="x", parent=parent_node)
        parent_node.left = left_child
        self.assertEqual(left_child.parent_side(), "left")

    def test_parent_side_right(self):
        """Test the parent_side method when the node is the right child."""
        parent_node = Node(id="1", node_type=NodeType.BINARY_OPERATOR, value="+", parent=None)
        right_child = Node(id="3", node_type=NodeType.VARIABLE, value="y", parent=parent_node)
        parent_node.right = right_child
        self.assertEqual(right_child.parent_side(), "right")

    def test_parent_side_no_parent(self):
        """Test the parent_side method when the node has no parent."""
        node = Node(id="1", node_type=NodeType.CONSTANT, value="10", parent=None)
        self.assertIsNone(node.parent_side())

    def test_parent_side_unrelated(self):
        """Test the parent_side method when the node is not a child of the parent."""
        parent_node = Node(id="1", node_type=NodeType.BINARY_OPERATOR, value="+", parent=None)
        unrelated_node = Node(id="4", node_type=NodeType.CONSTANT, value="10", parent=parent_node)
        self.assertIsNone(unrelated_node.parent_side())
    
    def test_calculate_max_depth_single_node(self):
        """Test calculate_max_depth on a single node."""
        single_node = Node(id="5", node_type=NodeType.CONSTANT, value="7", parent=None)
        self.assertEqual(single_node.calculate_max_depth(), 1)

    def test_calculate_max_depth_two_levels(self):
        """Test calculate_max_depth on a tree with two levels."""
        root = Node(id="1", node_type=NodeType.BINARY_OPERATOR, value="+", parent=None)
        left_child = Node(id="2", node_type=NodeType.CONSTANT, value="5", parent=root)
        root.left = left_child
        self.assertEqual(root.calculate_max_depth(), 2)
    
    def test_calculate_max_depth_three_levels(self):
        """Test calculate_max_depth on a tree with three levels."""
        root = Node(id="1", node_type=NodeType.BINARY_OPERATOR, value="+", parent=None)
        left_child = Node(id="2", node_type=NodeType.CONSTANT, value="5", parent=root)
        right_child = Node(id="3", node_type=NodeType.VARIABLE, value="x", parent=root)
        left_left_child = Node(id="4", node_type=NodeType.CONSTANT, value="3", parent=left_child)
        root.left = left_child
        root.right = right_child
        left_child.left = left_left_child
        self.assertEqual(root.calculate_max_depth(), 3)
        
    def test_calculate_max_depth_no_children(self):
        """Test calculate_max_depth on a node with no children."""
        node = Node(id="1", node_type=NodeType.UNARY_OPERATOR, value="-", parent=None)
        self.assertEqual(node.calculate_max_depth(), 1)
        
    def test_calculate_max_depth_explicit_none_children(self):
        """Test calculate_max_depth on a node with explicitly None children."""
        node = Node(id="2", node_type=NodeType.BINARY_OPERATOR, value="+", parent=None, left=None, right=None)
        self.assertEqual(node.calculate_max_depth(), 1)
    
if __name__ == "__main__":
    unittest.main()