import os
import sys
import copy
import time
import random
import logging

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd

from src.loss import Loss
from src.node import Node
from src.score import Score
from src.binary_tree import BinaryTree
from src.search_results import SearchResults
from typing import List, Tuple, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeneticSymbolicRegressor')

class GeneticSymbolicRegressor:
    def __init__(
        self, 
        num_individuals_per_epoch: int,
        max_initialization_individual_depth: int,
        variables: List[str],
        unary_operators: List[str],
        binary_operators: List[str],
        prob_node_mutation: float,
        prob_crossover: float,
        tournament_ratio: float,
        elitism_ratio: float,
        timeout: Optional[int],
        stop_score: Optional[float],
        max_generations: Optional[int] = 50,
        verbose: Optional[int] = 1,
        loss_name: Optional[str] = "mae",
        score_name: Optional[str] = "r2",
        random_state: Optional[int] = None):

        self.num_individuals_per_epoch = num_individuals_per_epoch
        self.max_initialization_individual_depth = max_initialization_individual_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.prob_node_mutation = prob_node_mutation
        self.prob_crossover = prob_crossover
        self.tournament_ratio = tournament_ratio
        self.elitism_ratio = elitism_ratio
        self.loss_name = loss_name
        self.score_name = score_name
        self.loss_function = Loss().get_loss_function(loss_name)
        self.score_function = Score().get_score_function(score_name)
        self.max_generations = max_generations
        self.timeout = timeout
        self.stop_score = stop_score
        self.verbose = verbose
        self.search_results = SearchResults()
        self.df_results = None

        random.seed(random_state)
        np.random.seed(random_state)

        if self.prob_node_mutation < 0.0 or self.prob_node_mutation > 1.0:
            raise ValueError("Mutation probability should be between 0.0 and 1.0")
        if self.tournament_ratio < 0.0 or self.tournament_ratio > 1.0:
            raise ValueError("Tournament ratio should be between 0.0 and 1.0")
        if self.elitism_ratio < 0.0 or self.elitism_ratio > 1.0:
            raise ValueError("Elitism ratio should be between 0.0 and 1.0")
        if (self.elitism_ratio + tournament_ratio) < 0.0 or (self.elitism_ratio + tournament_ratio) > 1.0:
            raise ValueError("Elitism ratio and tournament ratio combined should be between 0.0 and 1.0")
        if self.max_generations is not None:
            if self.max_generations <= 0:
                raise ValueError("Max generations should be bigger than 0")
        if self.timeout is not None:
            if self.timeout < 0.0:
                raise ValueError("Timeout should be bigger than 0")
        if self.stop_score is not None:
            if self.stop_score < -1.0 or self.stop_score > 1.0:
                raise ValueError("Stop score should be between -1.0 and 1.0")

    def _create_individuals(self, num_individuals: int) -> List[BinaryTree]:
        individuals = list()
        for _ in range(num_individuals):
            individuals.append(BinaryTree(
                max_initialization_depth=self.max_initialization_individual_depth,
                variables=self.variables,
                unary_operators=self.unary_operators,
                binary_operators=self.binary_operators
            ))
        return individuals
    
    def _sort_by_loss(self, individuals: List[BinaryTree]) -> None:
        individuals.sort(key=lambda individual: individual.loss, reverse=False)
        return individuals

    def _perform_tournament_selection(self, individuals, k: int = 2) -> List[Tuple[BinaryTree, BinaryTree]]:
        parents = list()
        continue_tournament = True
        num_individuals_in_tournament = int(
            ((len(individuals) * self.tournament_ratio) - (len(individuals) * self.tournament_ratio) % k) / k
        )
        individuals_to_select = [individual for individual in individuals if individual.loss is not np.inf]
        i = 0
        while continue_tournament:
            if random.random() < self.prob_crossover:
                parents.append(individuals_to_select[i])
                individuals_to_select.pop(i)
                i = 0
            else:
                if i == num_individuals_in_tournament - 1:
                    i = 0
                else:
                    i += 1
            
            if len(parents) == num_individuals_in_tournament:
                continue_tournament = False
        
        return parents

    def _depth_within_limits_after_crossover(self, parent: BinaryTree, parent_subtree_parent, parent_to_cross: Node, side: str):
        if side == "left":
            parent_subtree_parent_depth = parent_subtree_parent.left.calculate_max_depth()
        else:
            parent_subtree_parent_depth = parent_subtree_parent.right.calculate_max_depth()
        parent_depth = parent.depth
        parent_to_cross_node_depth = parent_to_cross.calculate_max_depth()

        depth_after_crossover = parent_depth - parent_subtree_parent_depth + parent_to_cross_node_depth
        if parent.max_depth >= depth_after_crossover:
            return True
        else:
            return False


    def ___perform_crossover(self, parents: List[Tuple[BinaryTree, BinaryTree]]) -> List[BinaryTree]: #ESTO ES UNA PRUEBA PARA EVITAR QUE EL MAX DEPTH SE PASE
        offsprings = list()
        num_crossover_tries = 1
        for parent1, parent2 in parents:
            tries = 0
            keep_trying = True
            while(keep_trying):
                tries += 1
                
                parent1_node = random.choice(parent1.nodes)
                parent2_node = random.choice(parent2.nodes)

                aux_parent1 = copy.deepcopy(parent1)
                aux_parent2 = copy.deepcopy(parent2)
                aux_parent1_node = copy.deepcopy(parent1_node)
                aux_parent2_node = copy.deepcopy(parent2_node)

                parent1_subtree_parent, parent1_subtree_side = aux_parent1_node.parent, aux_parent1_node.parent_side()
                parent2_subtree_parent, parent2_subtree_side = aux_parent2_node.parent, aux_parent2_node.parent_side()

                if parent1_subtree_parent:
                    if parent1_subtree_side == "left":
                        if self._depth_within_limits_after_crossover(aux_parent1, parent1_subtree_parent, aux_parent2_node, parent1_subtree_side):
                            parent1_subtree_parent.left = aux_parent2_node
                            keep_trying = False
                    else:
                        if self._depth_within_limits_after_crossover(aux_parent1, parent1_subtree_parent, aux_parent2_node, parent1_subtree_side):
                            parent1_subtree_parent.right = aux_parent2_node
                            keep_trying = False
                else:
                    aux_parent1.root = aux_parent2_node
                    keep_trying = False

                if parent2_subtree_parent:
                    if parent2_subtree_side == "left":
                        if self._depth_within_limits_after_crossover(aux_parent2, parent2_subtree_parent, parent1_node, parent2_subtree_side):
                            parent2_subtree_parent.left = aux_parent1_node
                            keep_trying = False
                    else:
                        if self._depth_within_limits_after_crossover(aux_parent2, parent2_subtree_parent, parent1_node, parent2_subtree_side):
                            parent2_subtree_parent.right = aux_parent1_node
                            keep_trying = False
                else:
                    aux_parent2.root = aux_parent1_node
                    keep_trying = False
                
                if num_crossover_tries <= tries:
                    offsprings.append(parent1)
                    offsprings.append(parent2)
                    #print(f"HE PROBADO {tries} VECES Y NO LO HE CONSEGUIDO")
                    break

                if not keep_trying:
                    parent1_node.parent = parent2_subtree_parent
                    parent2_node.parent = parent1_subtree_parent

                    parent1.update_tree_info()
                    parent2.update_tree_info()

                    offsprings.append(parent1)
                    offsprings.append(parent2)
                    
        return offsprings
    
    def _perform_crossover(self, parents: List[Tuple[BinaryTree, BinaryTree]]) -> List[BinaryTree]:
        offsprings = list()
        for parent1, parent2 in parents:
            aux_parent1 = copy.deepcopy(parent1)
            aux_parent2 = copy.deepcopy(parent2)

            aux_parent1_node = copy.deepcopy(random.choice(parent1.nodes))
            aux_parent2_node = copy.deepcopy(random.choice(parent2.nodes))

            parent1_subtree_parent, parent1_subtree_side = aux_parent1_node.parent, aux_parent1_node.parent_side()
            parent2_subtree_parent, parent2_subtree_side = aux_parent2_node.parent, aux_parent2_node.parent_side()

            if parent1_subtree_parent:
                if parent1_subtree_side == "left":
                    parent1_subtree_parent.left = aux_parent2_node
                else:
                    parent1_subtree_parent.right = aux_parent2_node
            else:
                aux_parent1.root = aux_parent2_node

            if parent2_subtree_parent:
                if parent2_subtree_side == "left":
                    parent2_subtree_parent.left = aux_parent1_node
                else:
                    parent2_subtree_parent.right = aux_parent1_node
            else:
                aux_parent2.root = aux_parent1_node
            
            aux_parent1_node.parent = parent2_subtree_parent
            aux_parent2_node.parent = parent1_subtree_parent

            aux_parent1.update_tree_info()
            aux_parent2.update_tree_info()

            offsprings.append(aux_parent1)
            offsprings.append(aux_parent2)
                
        return offsprings

    def _perform_mutation(self, offsprings: List[BinaryTree]) -> None:
        for i in range(len(offsprings)):
            if random.random() <= self.prob_node_mutation:
                offsprings[i].perform_mutation()
                offsprings[i].update_tree_info()
        return offsprings

    def _perform_elitism(self, individuals: List[BinaryTree]) -> List[BinaryTree]:
        num_elite_individuals = int(len(individuals) * self.elitism_ratio)
        elite_individuals = individuals[:num_elite_individuals]
        return [copy.deepcopy(individual) for individual in elite_individuals]

    def _calculate_loss(self, individuals: List[BinaryTree], X: pd.DataFrame, y: pd.Series) -> List[BinaryTree]:
        for i in range(len(individuals)):
            individuals[i].calculate_loss(X, y, self.loss_function)

        return individuals

    def _calculate_score(self, individuals: List[BinaryTree], X: pd.DataFrame, y: pd.Series) -> List[BinaryTree]:
        for i in range(len(individuals)):
            individuals[i].calculate_score(X, y, self.score_function)

        return individuals
    def _prepare_next_epoch_individual(self, offsprings: List[BinaryTree], elite_individuals: List[BinaryTree]) -> List[BinaryTree]:
        new_individuals = self._create_individuals(self.num_individuals_per_epoch - len(offsprings) - len(elite_individuals))
        return (
            elite_individuals +
            offsprings +
            new_individuals
        )
        
    def _check_stop_timeout(self, start_time: float) -> bool:
        if self.timeout != None:
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout:
                return True
            else:
                return False
        else:
            return False
    
    def _check_stop_score(self, best_score: Union[float, int]) -> bool:
        if self.stop_score != None:
            if self.stop_score <= best_score:
                return True
            else:
                return False
        else:
            return False
    
    def _check_max_generations_criteria(self, generation: int) -> bool:
        if self.max_generations != None:
            if generation >= self.max_generations + 1:
                return True
            else:
                return False
        else:
            return False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if X.empty or y.empty:
            raise ValueError(f"X and y shouldn't be empty.")
        
        start_time = time.time()

        individuals = self._create_individuals(self.num_individuals_per_epoch)
        individuals = self._calculate_loss(individuals, X, y)
        individuals = self._calculate_score(individuals, X, y)
        individuals = self._sort_by_loss(individuals)
        self.search_results.add_best_individuals_by_loss_and_complexity(individuals, 0)
        #self.search_results.extract_summary_statistics_from_individuals(individuals, 0)
        self.search_results.visualize_best_in_generation()

        best_individual = individuals[0]
        for generation in range(1, self.max_generations + 1):
            stop_timeout_criteria = self._check_stop_timeout(start_time)
            stop_score_criteria = self._check_stop_score(best_individual.score)
            stop_max_generations_criteria = self._check_max_generations_criteria(generation)
            if self.verbose >= 1:
                if stop_timeout_criteria:
                    logger.info('TIMEOUT STOP CRITERIA SATISFIED.')
                if stop_score_criteria:
                    logger.info('SCORE STOP CRITERIA SATISFIED.')
                if stop_max_generations_criteria:
                    logger.info('NUM GENERATIONS CRITERIA SATISFIED.')
            if stop_timeout_criteria or stop_score_criteria or stop_max_generations_criteria:
                if self.verbose >= 1: logger.info('STOPPING OPTIMIZATION...')
                break

            elite_individuals = self._perform_elitism(individuals)
            parents1 = self._perform_tournament_selection(individuals)
            parents2 = self._perform_tournament_selection(individuals)

            offsprings = self._perform_crossover(zip(parents1, parents2))
            offsprings = self._perform_mutation(offsprings)
            individuals = self._prepare_next_epoch_individual(offsprings, elite_individuals)
            individuals = self._calculate_loss(individuals, X, y)
            individuals = self._calculate_score(individuals, X, y)
            individuals = self._sort_by_loss(individuals)
            best_individual = individuals[0]

            self.search_results.add_best_individuals_by_loss_and_complexity(individuals, generation)
            #self.search_results.extract_summary_statistics_from_individuals(individuals, generation)
            self.search_results.visualize_best_in_generation()

        #self.search_results.plot_evolution_per_complexity()
        #self.search_results.plot_evolution()