import os
import sys
import time
import random
import logging

sys.path.append(os.path.abspath(os.curdir))

import numpy as np

from typing import List, Tuple, Optional, Union, Any

from src_local.loss import get_loss_function
from src_local.optimizers import AdamOptimizer
from src_local.score import get_score_function
from src_local.search_results import SearchResults
from src_local.binary_tree import build_full_binary_tree, calculate_loss, calculate_score, count_symbols_frequency

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
        prob_crossover: float,
        crossover_retries: int,
        tournament_size: int,
        elitism_ratio: float,
        timeout: Optional[int],
        stop_score: Optional[float],
        max_generations: Optional[int] = 50,
        frequencies_learning_rate: Optional[float] = 0.001,
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

        self.num_symbols = (len(self.unary_operators) + len(self.binary_operators) + len(self.variables))
        self.frequency_t = np.full((self.num_individuals_per_epoch, self.num_symbols), 1.0 / self.num_symbols)
        self.frequency_t_minus_1 = None
        self.optimizer = AdamOptimizer(frequencies=self.frequency_t, lr=frequencies_learning_rate)
        
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
        unary_operators_frequencies: List[List[float]] = None,
        binary_operators_frequencies: List[List[float]] = None,
        variables_frequencies: List[List[float]] = None) -> List[dict]:

        individuals = list()
        for i in range(num_individuals):
            if unary_operators_frequencies is None or binary_operators_frequencies is None or variables_frequencies is None:
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
            else:
                individuals.append(
                    build_full_binary_tree(
                        max_initialization_depth=self.max_individual_depth,
                        variables=self.variables,
                        unary_operators=self.unary_operators,
                        binary_operators=self.binary_operators,
                        unary_operators_frequencies=unary_operators_frequencies[i],
                        binary_operators_frequencies=binary_operators_frequencies[i],
                        variables_frequencies=variables_frequencies[i]
                    )
                )
        return individuals
    
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
        
    def _calculate_policy_gradient(self, fitness: np.ndarray) -> np.ndarray:
        return fitness[:, np.newaxis] * (self.frequency_t - self.frequency_t_minus_1)
    
    def _optimize_frequencies(self, individuals: List[Any]) -> Tuple[List[float], List[float], List[float]]:
        if self.frequency_t_minus_1 is None:
            symbols = self.unary_operators + self.binary_operators + self.variables

            population_frequency = []
            population_frequency_symbols = [count_symbols_frequency(individual[-1], symbols, None) for individual in individuals]
            for individual_symbol_frequency in population_frequency_symbols:
                frequencies = np.array(list(individual_symbol_frequency.values()), dtype=np.float64)
                frequencies = (frequencies - frequencies.max()) / (frequencies.max() - frequencies.min())
                frequencies /= frequencies.sum()
                population_frequency.append(frequencies)
            population_frequency = np.array(population_frequency)

            self.frequency_t_minus_1 = self.frequency_t.copy()
            self.frequency_t = population_frequency.copy()
        else:
            population_fitness = np.array([individual[2] for individual in individuals])
            gradients = self._calculate_policy_gradient(population_fitness)

            """
            print(population_fitness)
            print("CURRENT")
            print(self.frequency_t)
            print("OLD")
            print(self.frequency_t_minus_1)

            
            """
            print("GRADIENTS")
            print(gradients)
            print("PROBS")
            print(self.frequency_t)
            self.frequency_t_minus_1 = self.frequency_t.copy()
            self.optimizer.step(gradients)
            self.frequency_t = self.optimizer.frequencies.copy()

        unary_operators_frequencies = []
        binary_operators_frequencies = []
        variables_frequencies = []
        for my_new_frequency in self.frequency_t.tolist():
            unary_operators_frequencies.append(my_new_frequency[:len(self.unary_operators)])
            binary_operators_frequencies.append(my_new_frequency[len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)])
            variables_frequencies.append(my_new_frequency[-len(self.variables):])
        return unary_operators_frequencies, binary_operators_frequencies, variables_frequencies
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0 or y.size == 0:
            raise ValueError(f"X and y shouldn't be empty.")
        
        start_time = time.time()

        individuals = self._create_individuals(self.num_individuals_per_epoch)
        individuals = self._calculate_loss(individuals, X, y)
        individuals = self._calculate_score(individuals, X, y)

        self.search_results.add_best_individuals_by_loss_and_complexity(individuals, 0)
        #self.search_results.extract_summary_statistics_from_individuals(individuals, 0)
        self.search_results.visualize_best_in_generation()

        best_individual = max(individuals, key=lambda individual: individual[2])
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
            
            unary_operators_frequencies, binary_operators_frequencies, variables_frequencies = self._optimize_frequencies(individuals)

            individuals = self._create_individuals(self.num_individuals_per_epoch, unary_operators_frequencies, binary_operators_frequencies, variables_frequencies)
            individuals = self._calculate_loss(individuals, X, y)
            individuals = self._calculate_score(individuals, X, y)
            best_individual = max(individuals, key=lambda individual: individual[2])

            self.search_results.add_best_individuals_by_loss_and_complexity(individuals, generation)
            #self.search_results.extract_summary_statistics_from_individuals(individuals, generation)
            self.search_results.visualize_best_in_generation()
        #self.search_results.plot_evolution_per_complexity()
        #self.search_results.plot_evolution()

