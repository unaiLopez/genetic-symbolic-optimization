import os
import sys
import copy
import time
import random
import logging
import platform

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd

from src.node import Node
from src.binary_tree import BinaryTree
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
        tournament_ratio: float,
        elitism_ratio: float,
        timeout: Optional[int],
        stop_loss: Optional[float],
        max_generations: Optional[int] = 50,
        verbose: Optional[int] = 1,
        loss_function: Optional[str] = "mae",
        random_state: Optional[int] = None):

        self.num_individuals_per_epoch = num_individuals_per_epoch
        self.max_initialization_individual_depth = max_initialization_individual_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.prob_node_mutation = prob_node_mutation
        self.tournament_ratio = tournament_ratio
        self.elitism_ratio = elitism_ratio
        self.loss_function = loss_function
        self.max_generations = max_generations
        self.timeout = timeout
        self.stop_loss = stop_loss
        self.verbose = verbose

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
        if self.stop_loss is not None:
            if self.stop_loss < 0.0:
                raise ValueError("Stop loss should be bigger than 0")





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
    
    def _perform_selection(self, individuals, k: int = 2) -> List[Tuple[BinaryTree, BinaryTree]]:
        num_individuals_in_tournament = int(
            ((len(individuals) * self.tournament_ratio) - (len(individuals) * self.tournament_ratio) % k) / k
        )
        individuals_to_select = [individual for individual in individuals if individual.loss is not np.inf]
        inverted_loss = [1.0 / individual.loss if individual.loss != 0 else float(np.inf) for individual in individuals_to_select]
        total_inverted_loss = sum(inverted_loss)
        selection_probs = [loss / total_inverted_loss for loss in inverted_loss]

        return [tuple(random.choices(individuals_to_select, weights=selection_probs, k=k)) for _ in range(num_individuals_in_tournament)]

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
            individuals[i].calculate_loss(X, y)
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
    
    def _check_stop_loss(self, best_loss: Union[float, int]) -> bool:
        if self.stop_loss != None:
            if self.stop_loss >= best_loss:
                return True
            else:
                return False
        else:
            return False
    
    def _check_max_generations_criteria(self, generation: int) -> bool:
        if self.max_generations != None:
            if generation >= self.max_generations - 1:
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
        individuals = self._sort_by_loss(individuals)

        best_individual = individuals[0]

        for generation in range(1, self.max_generations):
            stop_timeout_criteria = self._check_stop_timeout(start_time)
            stop_loss_criteria = self._check_stop_loss(best_individual.loss)
            stop_max_generations_criteria = self._check_max_generations_criteria(generation)
            if self.verbose >= 1:
                if stop_timeout_criteria:
                    logger.info('TIMEOUT STOP CRITERIA SATISFIED.')
                if stop_loss_criteria:
                    logger.info('LOSS STOP CRITERIA SATISFIED.')
                if stop_max_generations_criteria:
                    logger.info('NUM GENERATIONS CRITERIA SATISFIED.')
            if stop_timeout_criteria or stop_loss_criteria or stop_max_generations_criteria:
                if self.verbose >= 1: logger.info('STOPPING OPTIMIZATION...')
                break

            elite_individuals = self._perform_elitism(individuals)
            parents = self._perform_selection(individuals)
            offsprings = self._perform_crossover(parents)
            offsprings = self._perform_mutation(offsprings)
            individuals = self._prepare_next_epoch_individual(offsprings, elite_individuals)
            individuals = self._calculate_loss(individuals, X, y)
            individuals = self._sort_by_loss(individuals)

            best_individual = individuals[0]

            epoch_results = []
            for individual in individuals:
                epoch_results.append({
                    "best": "",
                    "generation": generation,
                    "loss": individual.loss,
                    "complexity": individual.complexity,
                    "equation": individual.equation
                })
            
            df_epoch_results = pd.DataFrame(epoch_results)
            df_epoch_results.sort_values(by=["complexity", "loss"], ascending=True, inplace=True)
            df_epoch_results = df_epoch_results.groupby("complexity").first().reset_index()
            df_epoch_results = df_epoch_results[["best", "generation", "loss", "complexity", "equation"]]

            if self.df_results is None:
                self.df_results = df_epoch_results.copy()
            else:
                self.df_results = pd.concat([self.df_results, df_epoch_results], axis=0)
                self.df_results.sort_values(by=["complexity", "loss"], ascending=True, inplace=True)
                self.df_results = self.df_results.groupby("complexity").first().reset_index()
                self.df_results = self.df_results[["best", "generation", "loss", "complexity", "equation"]]
                self.df_results.loc[:, "best"] = ""
                min_index = self.df_results["loss"].idxmin()
                self.df_results.loc[min_index, "best"] = ">>>"
                self.df_results.loc[:, "current_generation"] = generation

            
            if platform.system() == "Windows":
                os.system("cls")
            else:
                os.system("clear")
            print(self.df_results)
            


            
            