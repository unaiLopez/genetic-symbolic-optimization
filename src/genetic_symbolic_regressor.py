import os
import sys
import math
import time
import random
import logging

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd

from numba import jit
from typing import List, Tuple, Optional, Union, Any

from src.loss import get_loss_function
from src.score import get_score_function
from src.crossover import perform_crossover
from src.search_results import SearchResults
from src.mutation import perform_node_mutation, perform_hoist_mutation
from src.tournament import perform_tournament_selection
from src.diversity import unique_individuals_ratio
from src.binary_tree import build_full_binary_tree, calculate_loss, calculate_score, count_symbols_frequency

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeneticSymbolicRegressor')

class GeneticSymbolicRegressor:
    def __init__(
        self, 
        num_individuals_per_epoch: int,
        max_individual_depth: int,
        variables: List[str],
        unary_operators: List[str],
        binary_operators: List[str],
        prob_node_mutation: float,
        prob_hoist_mutation: float,
        prob_crossover: float,
        crossover_retries: int,
        tournament_size: int,
        elitism_ratio: float,
        timeout: Optional[int],
        stop_score: Optional[float],
        max_generations: Optional[int] = 50,
        frequencies_learning_rate: Optional[float] = 0.3,
        warmup_generations: int = 100,
        verbose: Optional[int] = 1,
        loss_name: Optional[str] = "mae",
        score_name: Optional[str] = "r2",
        random_state: Optional[int] = None):

        self.num_individuals_per_epoch = num_individuals_per_epoch
        self.max_individual_depth = max_individual_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.prob_node_mutation = prob_node_mutation
        self.prob_hoist_mutation = prob_hoist_mutation
        self.prob_crossover = prob_crossover
        self.crossover_retries = crossover_retries
        self.tournament_size = tournament_size
        self.elitism_ratio = elitism_ratio
        self.loss_name = loss_name
        self.score_name = score_name
        self.loss_function = get_loss_function(loss_name)
        self.score_function = get_score_function(score_name)
        self.frequencies_learning_rate = frequencies_learning_rate
        self.warmup_generations = warmup_generations
        self.max_generations = max_generations
        self.timeout = timeout
        self.stop_score = stop_score
        self.verbose = verbose
        self.search_results = SearchResults()
        self.symbol_frequencies = [1 / (len(self.unary_operators) + len(self.binary_operators) + len(self.variables))] * (len(self.unary_operators) + len(self.binary_operators) + len(self.variables))
        random.seed(random_state)
        np.random.seed(random_state)

        if not isinstance(self.variables, list):
            raise TypeError("Variables should be of type List[str]")
        if self.prob_node_mutation < 0.0 or self.prob_node_mutation > 1.0:
            raise ValueError("Mutation probability should be between 0.0 and 1.0")
        if self.max_generations is not None:
            if self.max_generations <= 0:
                raise ValueError("Max generations should be bigger than 0")
        if self.timeout is not None:
            if self.timeout < 0.0:
                raise ValueError("Timeout should be bigger than 0")

    def _create_individuals(
        self,
        num_individuals: int,
        unary_operators_frequencies: List[float] = None,
        binary_operators_frequencies: List[float] = None,
        variables_frequencies: List[float] = None) -> List[dict]:

        individuals = list()
        for _ in range(num_individuals):
            individuals.append(
                build_full_binary_tree(
                    max_initialization_depth=self.max_individual_depth,
                    variables=self.variables,
                    unary_operators=self.unary_operators,
                    binary_operators=self.binary_operators,
                    unary_operators_frequencies=unary_operators_frequencies,
                    binary_operators_frequencies=binary_operators_frequencies,
                    variables_frequencies=variables_frequencies
                )
            )
        return individuals
    
    def _perform_elitism(self, individuals: List[dict]) -> List[dict]:
        num_elite_individuals = int(len(individuals) * self.elitism_ratio)
        elite_individuals = individuals[:num_elite_individuals]
        return [individual for individual in elite_individuals]

    def _perform_mutation(self, individuals: List[List[Any]]) -> List[List[Any]]:
        for i in range(len(individuals)):
            if random.random() <= self.prob_node_mutation:
                individuals[i][-1] = perform_node_mutation(
                    individuals[i][-1],
                    self.prob_node_mutation,
                    self.unary_operators,
                    self.binary_operators,
                    self.variables
                )
            if random.random() <= self.prob_hoist_mutation:
                individuals[i][-1] = perform_hoist_mutation(
                    individuals[i][-1],
                    individuals[i][0],
                    self.unary_operators,
                    self.binary_operators,
                    self.variables
                )

        return individuals
    
    def _perform_crossover(self, parents: Tuple[List[dict], List[dict]]) -> List[dict]:
        offsprings = list()
        operators = self.unary_operators + self.binary_operators
        for parent1, parent2 in parents:
            if random.random() < self.prob_crossover:
                offspring1, offspring2 = perform_crossover(
                    parent1,
                    parent2,
                    operators,
                    self.variables,
                    self.max_individual_depth,
                    self.crossover_retries
                )
                offsprings.append(offspring1)
                offsprings.append(offspring2)
        return offsprings

    def _calculate_loss(self, individuals: List[dict], X: np.ndarray, y: np.ndarray) -> List[dict]:
        for i in range(len(individuals)):
            individuals[i][1] = calculate_loss(X, y, self.loss_function, individuals[i][6])
        return individuals

    def _calculate_score(self, individuals: List[dict], X: np.ndarray, y: np.ndarray) -> List[dict]:
        for i in range(len(individuals)):
            individuals[i][2] = calculate_score(X, y, self.score_function, individuals[i][6])
        return individuals
    
    def _sort_by_score(self, individuals: List[Any]) -> List[Any]:
        individuals.sort(key=lambda individual: individual[2], reverse=True)

        return individuals
    
    def _prepare_next_epoch_individual(
        self,
        offsprings: List[dict],
        elite_individuals: List[dict],
        unary_operators_frequencies: List[float] = None,
        binary_operators_frequencies: List[float] = None,
        variables_frequencies: List[float] = None) -> List[dict]:

        new_individuals = self._create_individuals(self.num_individuals_per_epoch - len(offsprings) - len(elite_individuals), unary_operators_frequencies, binary_operators_frequencies, variables_frequencies)
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
    
    def _check_warmup_generations_finished(self, generation: int) -> bool:
        if generation > self.warmup_generations:
            return True
        else:
            return False

    def _get_updated_frequencies(self, individuals: List[Any], best_k_individuals: int) -> Tuple[List[float], List[float], List[float]]:
        symbols = self.unary_operators + self.binary_operators + self.variables
        total_symbols_frequency = None
        for individual in individuals[:best_k_individuals]:
            total_symbols_frequency = count_symbols_frequency(individual[-1], symbols, total_symbols_frequency)

        frequencies = []
        for key, value in total_symbols_frequency.items():
            frequencies.append(value)
        frequencies = np.divide(frequencies, np.sum(frequencies)).tolist()
        gradients = np.subtract(self.symbol_frequencies, frequencies)
        new_frequencies = np.subtract(self.symbol_frequencies, np.multiply(self.frequencies_learning_rate, gradients)).tolist()

        # CALCULATE FREQUENCIES AND UPDATE THEM AT NODE LEVEL
        # TRY JUST GETTING THE UNIQUE INDIVIDUALS FREQUENCIES
        # ADD SOME WARMUP GENERATIONS BEFORE STARTING TO UPDATE THE FREQUENCIES
        # DO CLIPING OF THE FREQUENCIES IN A LOT OF CASES THE FREQUENCIES HAVE VANISHING GRADIENTS AND ARE PRACTICALLY 0 (E.g. 1e-100)
        self.symbol_frequencies = new_frequencies
        print(symbols)
        print(self.symbol_frequencies)

        unary_operators_frequencies = new_frequencies[:len(self.unary_operators)]
        binary_operators_frequencies = new_frequencies[len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)]
        variables_frequencies = new_frequencies[:len(self.variables)]

        return unary_operators_frequencies, binary_operators_frequencies, variables_frequencies
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0 or y.size == 0:
            raise ValueError(f"X and y shouldn't be empty.")
        
        start_time = time.time()

        individuals = self._create_individuals(self.num_individuals_per_epoch)
        individuals = self._calculate_loss(individuals, X, y)
        individuals = self._calculate_score(individuals, X, y)
        individuals = self._sort_by_score(individuals)

        self.search_results.add_best_individuals_by_loss_and_complexity(individuals, 0)
        #self.search_results.extract_summary_statistics_from_individuals(individuals, 0)
        self.search_results.visualize_best_in_generation()

        best_individual = individuals[0]
        for generation in range(1, self.max_generations + 1):
            stop_timeout_criteria = self._check_stop_timeout(start_time)
            stop_score_criteria = self._check_stop_score(best_individual[2])
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
            
            if self._check_warmup_generations_finished(generation):
                ratio_unique_individuals = unique_individuals_ratio(individuals)
                print("RATIO")
                print(ratio_unique_individuals)
                if ratio_unique_individuals <= 0.2:
                    print("NO HAY DIVERSIDAD")
                    self.prob_node_mutation = 0.2
                else:
                    self.prob_node_mutation = 0.025
                print(self.prob_node_mutation)

                unary_operators_frequencies, binary_operators_frequencies, variables_frequencies = self._get_updated_frequencies(individuals, 100)
            else:
                unary_operators_frequencies = None
                binary_operators_frequencies = None 
                variables_frequencies = None

            elite_individuals = self._perform_elitism(individuals)
            parent_pairs = perform_tournament_selection(individuals, self.tournament_size)
            offsprings = self._perform_crossover(parent_pairs)
            offsprings = self._perform_mutation(offsprings)
            individuals = self._prepare_next_epoch_individual(offsprings, elite_individuals, unary_operators_frequencies, binary_operators_frequencies, variables_frequencies)
            individuals = self._calculate_loss(individuals, X, y)
            individuals = self._calculate_score(individuals, X, y)
            individuals = self._sort_by_score(individuals)

            best_individual = individuals[0]

            self.search_results.add_best_individuals_by_loss_and_complexity(individuals, generation)
            #self.search_results.extract_summary_statistics_from_individuals(individuals, generation)
            self.search_results.visualize_best_in_generation()
        #self.search_results.plot_evolution_per_complexity()
        #self.search_results.plot_evolution()

