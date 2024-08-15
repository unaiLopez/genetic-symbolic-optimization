import random

from typing import List, Any

from src.binary_tree import build_full_binary_tree

def perform_subtree_mutation():
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
        
def perform_node_mutation(
    node: List[Any],
    prob_mutation: float,
    unary_operators: List[str],
    binary_operators: List[str],
    variables: List[str]) -> List[Any]:
    
    # Helper function to perform mutation
    def _do_node_mutation(node: List[Any]) -> None:
        if node is None:
            return
        
        # Randomly decide whether to mutate this node or descend further
        if random.random() < prob_mutation:
            if node[0] in unary_operators:
                # Replace with another operator
                new_operator = random.choice(unary_operators)
                while new_operator == node[0]:
                    new_operator = random.choice(unary_operators)
                node[0] = new_operator
            elif node[0] in binary_operators:
                # Replace with another operator
                new_operator = random.choice(binary_operators)
                while new_operator == node[0]:
                    new_operator = random.choice(binary_operators)
                node[0] = new_operator
            elif node[0] in variables:
                # Replace with another terminal
                new_terminal = random.choice(variables)
                while new_terminal == node[0]:
                    new_terminal = random.choice(variables)
                node[0] = new_terminal
        else:
            # Recursively apply mutation to left and right children
            if node[1] is not None:
                _do_node_mutation(node[1])
            if node[1] is not None:
                _do_node_mutation(node[2])

    # Start mutation from the root node
    _do_node_mutation(node)
    return node
    
def perform_hoist_mutation(
    node: List[Any],
    max_initialization_depth: int,
    unary_operators: List[str],
    binary_operators: List[str],
    variables: List[str]) -> List[Any]:

    """
    Select a random subtree and replace the entire tree with this subtree, effectively "hoisting" it up to the root.
    """

    tree = build_full_binary_tree(
        max_initialization_depth,
        variables,
        unary_operators,
        binary_operators,
        None,
        None,
        None
    )
    return tree
    
def perform_shrink_mutation():
    #TODO
    """
    Select a subtree and replace it with one of its subtrees, effectively shrinking the tree.
    """
    pass