import os
import sys
import unittest

sys.path.append(os.path.abspath(os.curdir))

import pandas as pd

from src.node import Node
from src.binary_tree import BinaryTree

class TestBinaryTree(unittest.TestCase):
    def setUp(self):
        self.variables = ['x', 'y', 'const']
        self.unary_operators = ['sin', 'cos']
        self.binary_operators = ['+', '-', '*', '/']
        self.max_initialization_depth = 3
        self.tree = BinaryTree(self.max_initialization_depth, self.variables, self.unary_operators, self.binary_operators)

    def test_tree_initialization(self):
        """Test that a BinaryTree object is initialized correctly."""
        self.assertIsInstance(self.tree, BinaryTree)
        self.assertEqual(self.tree.max_initialization_depth, self.max_initialization_depth)
        self.assertEqual(self.tree.variables, self.variables)
        self.assertEqual(self.tree.unary_operators, self.unary_operators)
        self.assertEqual(self.tree.binary_operators, self.binary_operators)

    def test_update_nodes(self):
        """Test that _update_nodes correctly collects all nodes in the tree."""
        initial_node_count = len(self.tree.nodes)
        self.tree._update_nodes()
        self.assertEqual(len(self.tree.nodes), initial_node_count)
    
    def test_update_complexity(self):
        """Test that _update_complexity correctly calculates the tree's complexity."""
        initial_complexity = len(self.tree.nodes)
        self.tree._update_complexity()
        self.assertEqual(self.tree.complexity, initial_complexity)
    
    def test_build_tree(self):
        """Test that _build_tree correctly builds the tree structure."""
        tree = BinaryTree(self.max_initialization_depth, self.variables, self.unary_operators, self.binary_operators)
        self.assertIsNotNone(tree.root)
        self.assertIsInstance(tree.root, Node)
    
    def test_collect_nodes(self):
        """Test that _collect_nodes correctly collects all nodes in the tree."""
        nodes = self.tree._collect_nodes()
        self.assertGreater(len(nodes), 0)
        self.assertIsInstance(nodes[0], Node)
    
    def test_build_equation(self):
        """Test that _build_equation constructs the correct equation string."""
        equation = self.tree._build_equation(self.tree.root)
        self.assertIsInstance(equation, str)
    
    def test_perform_mutation(self):
        """Test that perform_mutation correctly mutates a node."""
        initial_nodes = self.tree._collect_nodes()
        initial_values = [node.value for node in initial_nodes]
        self.tree.perform_mutation()
        mutated_nodes = self.tree._collect_nodes()
        mutated_values = [node.value for node in mutated_nodes]
        self.assertNotEqual(initial_values, mutated_values)

    def test_build_executable_equation(self):
        """Test that _build_executable_equation constructs the correct executable equation."""
        self.tree.update_tree_info()
        self.tree._build_executable_equation()
        self.assertIsInstance(self.tree.executable_equation, str)
    
    def test_update_tree_info(self):
        """Test that update_tree_info updates nodes, complexity, depth, and equation."""
        self.tree.update_tree_info()
        self.assertIsNotNone(self.tree.nodes)
        self.assertIsNotNone(self.tree.complexity)
        self.assertIsNotNone(self.tree.depth)
        self.assertIsNotNone(self.tree.equation)

    def test_calculate_loss(self):
        """Test that calculate_loss correctly calculates the loss given input data."""
        X = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        y = pd.Series([5, 7, 9])
        self.tree.update_tree_info()
        self.tree._build_executable_equation()
        self.tree.calculate_loss(X, y)
        self.assertIsInstance(self.tree.loss, float)

if __name__ == "__main__":
    unittest.main()