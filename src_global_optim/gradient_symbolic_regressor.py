import os
import sys
import time
import random
import logging

sys.path.append(os.path.abspath(os.curdir))

import numpy as np

from typing import List, Optional, Union, Any

from src_global_optim.loss import get_loss_function
from src_global_optim.score import get_score_function
from src_global_optim.search_results import SearchResults
from src_global_optim.binary_tree import build_full_binary_tree, calculate_loss, calculate_score, count_symbols_frequency

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeneticSymbolicRegressor')

class GradientSymbolicRegressor:
    def __init__(
        self, 
        num_individuals_per_epoch: int,
        max_individual_depth: int,
        variables: List[str],
        unary_operators: List[str],
        binary_operators: List[str],
        timeout: Optional[int],
        stop_score: Optional[float],
        max_generations: Optional[int] = 50,
        learning_rate: Optional[float] = 0.01,
        verbose: Optional[int] = 1,
        loss_name: Optional[str] = "mae",
        score_name: Optional[str] = "r2",
        random_state: Optional[int] = None):

        self.num_individuals_per_epoch = num_individuals_per_epoch
        self.max_individual_depth = max_individual_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.loss_name = loss_name
        self.score_name = score_name
        self.loss_function = get_loss_function(loss_name)
        self.score_function = get_score_function(score_name)
        
        self.learning_rate = learning_rate
        self.max_generations = max_generations
        self.timeout = timeout
        self.stop_score = stop_score
        self.verbose = verbose
        self.search_results = SearchResults()

        self.num_symbols = (len(self.unary_operators) + len(self.binary_operators) + len(self.variables))
        self.probabilities = np.full(self.num_symbols, 1.0 / self.num_symbols)
        
        random.seed(random_state)
        np.random.seed(random_state)

        if not isinstance(self.variables, list):
            raise TypeError("Variables should be of type List[str]")
        if self.max_generations is not None:
            if self.max_generations <= 0:
                raise ValueError("Max generations should be bigger than 0")
        if self.timeout is not None:
            if self.timeout < 0.0:
                raise ValueError("Timeout should be bigger than 0")

    def _create_individuals(
        self,
        num_individuals: int,
        unary_operators_probs: List[float] = None,
        binary_operators_probs: List[float] = None,
        variables_probs: List[float] = None) -> List[dict]:

        individuals = list()
        for i in range(num_individuals):
            individuals.append(
                build_full_binary_tree(
                    max_initialization_depth=self.max_individual_depth,
                    variables=self.variables,
                    unary_operators=self.unary_operators,
                    binary_operators=self.binary_operators,
                    unary_operators_probs=unary_operators_probs,
                    binary_operators_probs=binary_operators_probs,
                    variables_probs=variables_probs
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
        
    # Gradient descent-like update for probabilities
    def _update_probabilities(self, probabilities, gradient):
        probabilities += self.learning_rate * gradient
        probabilities = (probabilities - probabilities.max()) / (probabilities.max() - probabilities.min())
        #probabilities = np.maximum(probabilities, 0)  # Ensure non-negative
        return probabilities / probabilities.sum()  # Normalize to sum to 1
        
    def _create_new_individual_with_optimized_probabilities(self, individual: List[Any], X: np.ndarray, y: np.ndarray) -> List[Any]:
        # Estimate gradients via perturbation
        gradient = np.zeros_like(individual[8])
        epsilon = 1e-2

        for i in range(len(individual[8])):
            perturbed_probabilities = np.array(individual[8].copy())
            perturbed_probabilities[i] += epsilon
            perturbed_probabilities = (perturbed_probabilities / perturbed_probabilities.sum()).tolist()  # Normalize
            
            perturbed_unary_operators_probs = perturbed_probabilities[:len(self.unary_operators)]
            perturbed_binary_operators_probs = perturbed_probabilities[len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)]
            perturbed_variables_probs = perturbed_probabilities[-len(self.variables):]

            # PARA REDUCIR LA INCERTIDUMBRE REPETIR ESTE PROCESO N VECES Y CON LOS SCORES HACER BOOTSTRAPING PARA CONSEGUIR LA MEDIA
            perturbed_tree = build_full_binary_tree(
                max_initialization_depth=self.max_individual_depth,
                variables=self.variables,
                unary_operators=self.unary_operators,
                binary_operators=self.binary_operators,
                unary_operators_probs=perturbed_unary_operators_probs,
                binary_operators_probs=perturbed_binary_operators_probs,
                variables_probs=perturbed_variables_probs
            )
            perturbed_tree_score = calculate_score(X, y, self.score_function, perturbed_tree[6])
            gradient[i] = (perturbed_tree_score - individual[2]) / epsilon
        
        # Update probabilities using the gradient
        optimized_probabilities = self._update_probabilities(individual[8], gradient)
        print(optimized_probabilities)
                
        optimized_probabilities = optimized_probabilities.tolist()
        perturbed_unary_operators_probs = optimized_probabilities[:len(self.unary_operators)]
        perturbed_binary_operators_probs = optimized_probabilities[len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)]
        perturbed_variables_probs = optimized_probabilities[-len(self.variables):]

        return build_full_binary_tree(
            max_initialization_depth=self.max_individual_depth,
            variables=self.variables,
            unary_operators=self.unary_operators,
            binary_operators=self.binary_operators,
            unary_operators_probs=perturbed_unary_operators_probs,
            binary_operators_probs=perturbed_binary_operators_probs,
            variables_probs=perturbed_variables_probs
        )
    
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
            
            for i in range(len(individuals)):
                individuals[i] = self._create_new_individual_with_optimized_probabilities(individuals[i], X, y)
                
            individuals = self._calculate_loss(individuals, X, y)
            individuals = self._calculate_score(individuals, X, y)
            best_individual = max(individuals, key=lambda individual: individual[2])

            self.search_results.add_best_individuals_by_loss_and_complexity(individuals, generation)
            #self.search_results.extract_summary_statistics_from_individuals(individuals, generation)
            self.search_results.visualize_best_in_generation()
        #self.search_results.plot_evolution_per_complexity()
        #self.search_results.plot_evolution()

