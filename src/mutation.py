import random

from typing import List

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
    node: dict,
    prob_mutation: float,
    unary_operators: List[str],
    binary_operators: List[str],
    variables: List[str]):
    
    # Helper function to perform mutation
    def _mutate_node(node: dict):
        if node is None:
            return
        
        current_node_key = list(node.keys())[0]
        children = node[current_node_key]

        # Randomly decide whether to mutate this node or descend further
        if random.random() < prob_mutation:
            if current_node_key in unary_operators:
                # Replace with another operator
                new_operator = random.choice(unary_operators)
                while new_operator == current_node_key:
                    new_operator = random.choice(unary_operators)
                node[new_operator] = children
                del node[current_node_key]
            elif current_node_key in binary_operators:
                # Replace with another operator
                new_operator = random.choice(binary_operators)
                while new_operator == current_node_key:
                    new_operator = random.choice(binary_operators)
                node[new_operator] = children
                del node[current_node_key]
            elif current_node_key in variables:
                # Replace with another terminal
                new_terminal = random.choice(variables)
                while new_terminal == current_node_key:
                    new_terminal = random.choice(variables)
                node[new_terminal] = children
                del node[current_node_key]
        else:
            # Recursively apply mutation to left and right children
            if children["left"] is not None:
                _mutate_node(children["left"])
            if children["right"] is not None:
                _mutate_node(children["right"])

    # Start mutation from the root node
    _mutate_node(node)
    return node
    
    def perform_hoist_mutation():
        #TODO
        """
        Select a random subtree and replace the entire tree with this subtree, effectively "hoisting" it up to the root.
        """
        pass

    def perform_shrink_mutation():
        #TODO
        """
        Select a subtree and replace it with one of its subtrees, effectively shrinking the tree.
        """
        pass