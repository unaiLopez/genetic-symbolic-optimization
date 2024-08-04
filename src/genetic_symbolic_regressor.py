import os
import sys
import time
import random
import logging

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd

from numba import jit
from typing import List, Tuple, Optional, Union

from src.loss import get_loss_function
from src.score import get_score_function
from src.crossover import perform_crossover
from src.mutation import perform_node_mutation
from src.binary_tree import build_full_binary_tree, calculate_loss, calculate_score

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
        self.loss_function = get_loss_function(loss_name)
        self.score_function = get_score_function(score_name)
        self.max_generations = max_generations
        self.timeout = timeout
        self.stop_score = stop_score
        self.verbose = verbose

        random.seed(random_state)
        np.random.seed(random_state)

        if not isinstance(self.variables, list):
            raise TypeError("Variables should be of type List[str]")
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


    def _create_individuals(self, num_individuals: int) -> List[dict]:
        individuals = list()
        for _ in range(num_individuals):
            individuals.append(
                build_full_binary_tree(
                    max_initialization_depth=self.max_initialization_individual_depth,
                    variables=self.variables,
                    unary_operators=self.unary_operators,
                    binary_operators=self.binary_operators
                )
            )
        return individuals
    
    def _sort_by_loss(self, individuals: List[dict]) -> None:
        individuals.sort(key=lambda individual: individual["loss"], reverse=False)
        return individuals

    def _perform_tournament_selection(self, individuals, k: int = 2) -> List[Tuple[dict, dict]]:
        parents = list()
        num_individuals_in_tournament = int(
            ((len(individuals) * self.tournament_ratio) - (len(individuals) * self.tournament_ratio) % k) / k
        )
        individuals_to_select = [individual for individual in individuals if individual["loss"] is not np.inf]
        i = 0
        while True:
            if random.random() < self.prob_crossover:
                parents.append(individuals_to_select[i])
                individuals_to_select.pop(i)
                i = 0
            else:
                if i == len(individuals_to_select) - 1:
                    i = 0
                else:
                    i += 1
            if len(parents) == num_individuals_in_tournament:
                break
        
        return parents

    def _perform_elitism(self, individuals: List[dict]) -> List[dict]:
        num_elite_individuals = int(len(individuals) * self.elitism_ratio)
        elite_individuals = individuals[:num_elite_individuals]
        return [individual for individual in elite_individuals]

    def _perform_mutation(self, individuals: List[dict]) -> List[dict]:
        for i in range(len(individuals)):
            individuals[i]["tree"] = perform_node_mutation(
                individuals[i]["tree"],
                self.prob_node_mutation,
                self.unary_operators,
                self.binary_operators,
                self.variables
            )
        return individuals
    
    def _perform_crossover(self, parents: Tuple[List[dict], List[dict]]) -> List[dict]:
        offsprings = list()
        operators = self.unary_operators + self.binary_operators
        for parent1, parent2 in parents:
            offspring1, offspring2 = perform_crossover(
                parent1,
                parent2,
                operators,
                self.variables
            )
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        return offsprings

    def _calculate_loss(self, individuals: List[dict], X: np.ndarray, y: np.ndarray) -> List[dict]:
        for i in range(len(individuals)):
            individuals[i]["loss"] = calculate_loss(X, y, self.loss_function, individuals[i]["executable_equation"])
        return individuals

    def _calculate_score(self, individuals: List[dict], X: np.ndarray, y: np.ndarray) -> List[dict]:
        for i in range(len(individuals)):
            individuals[i]["score"] = calculate_score(X, y, self.score_function, individuals[i]["executable_equation"])
        return individuals
    
    def _prepare_next_epoch_individual(self, offsprings: List[dict], elite_individuals: List[dict]) -> List[dict]:
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
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0 or y.size == 0:
            raise ValueError(f"X and y shouldn't be empty.")
        
        start_time = time.time()

        individuals = self._create_individuals(self.num_individuals_per_epoch)
        individuals = self._calculate_loss(individuals, X, y)
        individuals = self._calculate_score(individuals, X, y)
        individuals = self._sort_by_loss(individuals)
        print("GENERATION 0")
        print(f"LOSS = {individuals[0]['loss']}")
        print(f"SCORE = {individuals[0]['score']}")
        print(f"EQUATION = {individuals[0]['equation']}")

        best_individual = individuals[0]
        for generation in range(1, self.max_generations + 1):
            stop_timeout_criteria = self._check_stop_timeout(start_time)
            stop_score_criteria = self._check_stop_score(best_individual["score"])
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
            print(f"GENERATION {generation}")
            print(f"LOSS = {best_individual['loss']}")
            print(f"SCORE = {best_individual['score']}")
            print(f"EQUATION = {best_individual['equation']}")
            print("\n================================================================\n")

