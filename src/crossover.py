import copy
import random

from src.binary_tree import update_tree_info
from typing import List, Tuple, Any, Dict

def _collect_all_nodes(node: Dict[str, Any], path: List[Tuple[str, str]], nodes: List[Tuple[List[Tuple[str, str]], str]]) -> None:
    if node is None or not isinstance(node, dict) or not node:
        return
    
    # Extract the single key of the current node
    current_node_key = list(node.keys())[0]
    children = node[current_node_key]

    # Record the current node and its path
    nodes.append((path.copy(), current_node_key))

    # Recur on left and right children if they exist
    if children.get('left'):
        _collect_all_nodes(children['left'], path + [(current_node_key, 'left')], nodes)
    if children.get('right'):
        _collect_all_nodes(children['right'], path + [(current_node_key, 'right')], nodes)

def _select_random_node(node: Dict[str, Any]) -> List[Tuple[str, str]]:
    nodes = []
    _collect_all_nodes(node, [], nodes)
    if not nodes:
        return []  # No nodes to select
    selected_path, _ = random.choice(nodes)
    return selected_path

def _get_subtree(node: Dict[str, Any], path: List[Tuple[str, str]]) -> Dict[str, Any]:
    current = node
    for key, direction in path:
        current = current[key][direction]
        if current is None:
            return None
    return current

def _set_subtree(node: Dict[str, Any], path: List[Tuple[str, str]], subtree: Dict[str, Any]) -> None:
    if not path:
        raise ValueError("Path cannot be empty when setting a subtree.")

    current = node
    # Traverse the path except for the last element
    for key, direction in path[:-1]:
        if key not in current or direction not in current[key]:
            raise KeyError(f"Path segment ({key}, {direction}) does not exist in the tree structure.")
        current = current[key][direction]

    last_key, last_direction = path[-1]
    if last_key in current and last_direction in current[last_key]:
        current[last_key][last_direction] = subtree
    else:
        raise KeyError(f"Final path segment ({last_key}, {last_direction}) does not exist in the tree structure. "
                       f"Current state: {current}")

def perform_crossover(tree1: dict, tree2: dict, operators: List[str], variables: List[str]) -> Tuple[dict, dict]:
    """
    Perform crossover between two binary trees and return two new trees.
    """
    new_tree1 = copy.deepcopy(tree1)
    new_tree2 = copy.deepcopy(tree2)

    # Select random crossover points in both trees
    path1 = _select_random_node(new_tree1["tree"])
    path2 = _select_random_node(new_tree2["tree"])
    
    if not path1 or not path2:
        # If either path is empty, return the original trees as they are
        return tree1, tree2
    
    # Get the subtrees at the selected paths
    subtree1 = _get_subtree(new_tree1["tree"], path1)
    subtree2 = _get_subtree(new_tree2["tree"], path2)

    if subtree1 is None or subtree2 is None:
        # If either subtree is invalid, return trees as they are
        return tree1, tree2

    # Swap the subtrees
    _set_subtree(new_tree1["tree"], path1, subtree2)
    _set_subtree(new_tree2["tree"], path2, subtree1)

    new_tree1 = update_tree_info(new_tree1, operators, variables)
    new_tree2 = update_tree_info(new_tree2, operators, variables)

    return new_tree1, new_tree2