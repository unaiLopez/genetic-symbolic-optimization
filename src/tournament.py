import random

from typing import List, Tuple, Any

def _get_best_in_tournament(individuals: List[Any], tournament_size: int) -> List[Any]:
    tournament = random.sample(individuals, tournament_size)
    return max(tournament, key=lambda individual: individual[2])    #BY SCORE

def _select_parents(individuals: List[Any], tournament_size: int, num_parents: int) -> List[Any]:
    parents = []
    for _ in range(num_parents):
        parent = _get_best_in_tournament(individuals, tournament_size)
        parents.append(parent)
    return parents

def _create_parent_pairs(individuals: List[Any]):
    parent_pairs = []
    random.shuffle(individuals)
    for i in range(0, len(individuals), 2):
        if i + 1 < len(individuals):
            parent_pairs.append((individuals[i], individuals[i+1]))
    return parent_pairs

def perform_tournament_selection(individuals: List[Any], tournament_size: int) -> List[Tuple[List[Any], List[Any]]]:
    selected_parents = _select_parents(individuals, tournament_size, len(individuals))
    parent_pairs = _create_parent_pairs(selected_parents)
    return parent_pairs