import copy
import random

from typing import List, Tuple, Any, Dict, Optional
from binary_tree import update_tree_info, _calculate_max_depth

def _collect_all_nodes(tree: List[Any], path: List[int], nodes: List[Tuple[List[int], Any]]) -> None:
    if tree is None or not isinstance(tree, list) or not tree:
        return

    # Record the current node and its path
    nodes.append((path.copy(), tree[0]))

    # Recur on left and right children if they exist
    if len(tree) > 1 and tree[1] is not None:
        _collect_all_nodes(tree[1], path + [1], nodes)
    if len(tree) > 2 and tree[2] is not None:
        _collect_all_nodes(tree[2], path + [2], nodes)

def _select_random_node(tree: List[Any]) -> List[int]:
    nodes = []
    _collect_all_nodes(tree, [], nodes)
    if not nodes:
        return []  # No nodes to select
    selected_path, _ = random.choice(nodes)
    return selected_path

def _get_subtree(tree: List[Any], path: List[int]) -> Optional[List[Any]]:
    current = tree
    for p in path:
        if not (0 <= p < len(current)) or current[p] is None:
            return None
        current = current[p]
    return current

def _set_subtree(tree: List[Any], path: List[int], new_subtree: List[Any]) -> None:
    if not path:
        raise ValueError("Path cannot be empty when setting a subtree.")

    current = tree
    # Traverse the path except for the last element
    for p in path[:-1]:
        if not (0 <= p < len(current)) or current[p] is None:
            raise IndexError(f"Path segment {p} does not exist in the tree structure.")
        current = current[p]

    last_index = path[-1]
    if 0 <= last_index < len(current):
        current[last_index] = new_subtree
    else:
        raise IndexError(f"Final path index {last_index} does not exist in the tree structure. Current state: {current}")

def perform_crossover(
    tree1: List[Any],
    tree2: List[Any],
    operators: List[str],
    variables: List[str],
    max_individual_depth: int,
    max_retries: int) -> Tuple[List[Any], List[Any]]:
    """
    Perform crossover between two binary trees and return two new trees.
    """
    new_tree1 = copy.deepcopy(tree1)
    new_tree2 = copy.deepcopy(tree2)

    current_retries = 0
    while(current_retries < max_retries):
        # Select random crossover points in both trees
        path1 = _select_random_node(new_tree1[-1])
        path2 = _select_random_node(new_tree2[-1])

        if not path1 or not path2:
            # If either path is empty, return the original trees as they are
            return tree1, tree2
        
        # Get the subtrees at the selected paths
        subtree1 = _get_subtree(new_tree1[-1], path1)
        subtree2 = _get_subtree(new_tree2[-1], path2)

        # Check if depth after crossover will be bigger than max permitted lenght    
        if ((len(path1) + _calculate_max_depth(subtree2)) > max_individual_depth or
            (len(path2) + _calculate_max_depth(subtree1)) > max_individual_depth):
            current_retries += 1
        else:
            if subtree1 is None or subtree2 is None:
                # If either subtree is invalid, return trees as they are
                return tree1, tree2

            # Swap the subtrees
            _set_subtree(new_tree1[-1], path1, subtree2)
            _set_subtree(new_tree2[-1], path2, subtree1)

            # Example placeholder for updating tree information
            # Update this function as per your specific requirements
            new_tree1 = update_tree_info(new_tree1, operators, variables)
            new_tree2 = update_tree_info(new_tree2, operators, variables)

            return new_tree1, new_tree2
    
    return tree1, tree2