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
from src.search_results import SearchResults
from src.binary_tree import build_full_binary_tree, calculate_loss, calculate_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeneticSymbolicRegressor')

class GradientDescentSymbolicRegressor:
    def __init__(
        self,
        num_individuals_per_sample: int,
        max_individual_depth: int,
        variables: List[str],
        unary_operators: List[str],
        binary_operators: List[str],
        timeout: Optional[int],
        stop_score: Optional[float],
        max_iterations: Optional[int] = 100,
        probs_learning_rate: Optional[float] = 1e-3,
        verbose: Optional[int] = 1,
        loss_name: Optional[str] = "mse",
        score_name: Optional[str] = "r2",
        random_state: Optional[int] = None):

        self.num_individuals_per_sample = num_individuals_per_sample
        self.max_individual_depth = max_individual_depth
        self.max_individual_nodes = (2 ** (max_individual_depth + 1)) - 1
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.loss_function = get_loss_function(loss_name)
        self.score_function = get_score_function(score_name)
        self.probs_learning_rate = probs_learning_rate
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.stop_score = stop_score
        self.verbose = verbose
        self.search_results = SearchResults()

        self.symbols = self.unary_operators + self.binary_operators + self.variables
        self.num_symbols = len(self.symbols)
        self.symbol_probs_t = np.full((self.max_individual_nodes, self.num_symbols), 1 / self.num_symbols)
        self.symbol_probs_minus_t = np.random.uniform(low=1.0, high=1000.0, size=(self.max_individual_nodes, self.num_symbols))
        self.symbol_probs_minus_t /= self.symbol_probs_minus_t.sum(axis=1).reshape(-1, 1)

        self.best_score = -999
        self.best_individual = None

        random.seed(random_state)
        np.random.seed(random_state)

        if not isinstance(self.variables, list):
            raise TypeError("Variables should be of type List[str]")
        if self.max_iterations is not None:
            if self.max_iterations <= 0:
                raise ValueError("Max generations should be bigger than 0")
        if self.timeout is not None:
            if self.timeout < 0.0:
                raise ValueError("Timeout should be bigger than 0")

    def _check_stop_timeout(self, start_time: float) -> bool:
        if self.timeout != None:
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout:
                return True
            else:
                return False
        else:
            return False
    
    def _check_max_iterations_criteria(self, iteration: int) -> bool:
        if self.max_iterations != None:
            if iteration >= self.max_iterations + 1:
                return True
            else:
                return False
        else:
            return False
    
    def _generate_individual(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unary_operators_probs: List[List[float]],
        binary_operators_probs: List[List[float]],
        variables_probs: List[List[float]]) -> List[Any]:


        individual = build_full_binary_tree(
            max_initialization_depth=self.max_individual_depth,
            variables=self.variables,
            unary_operators=self.unary_operators,
            binary_operators=self.binary_operators,
            unary_operators_probs=unary_operators_probs,
            binary_operators_probs=binary_operators_probs,
            variables_probs=variables_probs
        )
        individual[1] = calculate_loss(X, y, self.loss_function, individual[6])
        individual[2] = calculate_score(X, y, self.score_function, individual[6])
        
        return individual

    def _generate_n_individuals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unary_operators_probs: List[List[float]],
        binary_operators_probs: List[List[float]],
        variables_probs: List[List[float]]) -> List[List[Any]]:

        individuals = []
        for _ in range(self.num_individuals_per_sample):
            individuals.append(
                self._generate_individual(
                    X,
                    y,
                    unary_operators_probs,
                    binary_operators_probs,
                    variables_probs
                )
            )
        
        return individuals

    def _check_stop_score_criteria(self, individuals: List[Any], scores: np.ndarray) -> None:
        max_score = np.max(scores)
        if max_score > self.best_score:
            self.best_score = max_score
            self.best_individual = individuals[np.argmax(scores)]

            if max_score >= self.stop_score:
                self.search_results.visualize_best_in_generation()
                print(f"WITH A SCORE OF {self.best_score}, THE BEST INDIVIDUAL IS {self.best_individual}")
                #sys.exit(f"WITH A SCORE OF {self.best_score}, THE BEST INDIVIDUAL IS {self.best_individual}")

    def _calculate_forward_difference_step(self, X, y, probabilities, node_index, prob_index, iteration, epsilon) -> float:
        # Perturb one parameter slightly
        probabilities_calibration = epsilon / (probabilities[node_index].shape[0] - 1)
        probabilities[node_index] -= probabilities_calibration
        probabilities[node_index][prob_index] += (epsilon + probabilities_calibration)
        probabilities[node_index] = np.where(probabilities[node_index] < 0, 0, probabilities[node_index])
        probabilities[node_index] /= probabilities[node_index].sum()

        perturbed_individuals = self._generate_n_individuals(
            X,
            y,
            probabilities[:, :len(self.unary_operators)].tolist(),
            probabilities[:, len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)].tolist(),
            probabilities[:, -len(self.variables):].tolist()
        )

        perturbed_scores = np.array([perturbed_individual[2] for perturbed_individual in perturbed_individuals])
        self.search_results.add_best_individuals_by_loss_and_complexity(perturbed_individuals, iteration)
        self._check_stop_score_criteria(perturbed_individuals, perturbed_scores)
        
        return perturbed_scores.mean()

    def _calculate_backward_difference_step(self, X, y, probabilities, node_index, prob_index, iteration, epsilon) -> float:
        # Perturb one parameter slightly
        probabilities_calibration = epsilon / (probabilities[node_index].shape[0] - 1)
        probabilities[node_index] += probabilities_calibration
        probabilities[node_index][prob_index] -= (epsilon + probabilities_calibration)
        probabilities[node_index] = np.where(probabilities[node_index] < 0, 0, probabilities[node_index])
        probabilities[node_index] /= probabilities[node_index].sum()

        perturbed_individuals = self._generate_n_individuals(
            X,
            y,
            probabilities[:, :len(self.unary_operators)].tolist(),
            probabilities[:, len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)].tolist(),
            probabilities[:, -len(self.variables):].tolist()
        )
        perturbed_scores = np.array([perturbed_individual[2] for perturbed_individual in perturbed_individuals])
        self.search_results.add_best_individuals_by_loss_and_complexity(perturbed_individuals, iteration)
        self._check_stop_score_criteria(perturbed_individuals, perturbed_scores)
        
        return perturbed_scores.mean()
    
    def _compute_gradients_central_difference(self, X, y, probabilities, iteration, epsilon=1e-2) -> np.ndarray:
        gradients = np.zeros((self.max_individual_nodes, self.num_symbols))
        for node_index in range(self.max_individual_nodes):
            for prob_index in range(probabilities.shape[1]):
                forward_difference_value = self._calculate_forward_difference_step(X, y, probabilities.copy(), node_index, prob_index, iteration, epsilon)
                backward_difference_value = self._calculate_backward_difference_step(X, y, probabilities.copy(), node_index, prob_index, iteration, epsilon)

                # Compute the central difference step
                gradients[node_index][prob_index] = (forward_difference_value - backward_difference_value) / (2 * epsilon)
        
        return gradients

    # EN VEZ DE EVITAR QUE SEA MENOR QUE 0, TRATARLO COMO UNA DISTRIBUCUION DE PROBABILIDADES Y APLICAR SOFTMAX (REVISAR ESTE APPROACH)
    def _update_probabilities(self, gradients: np.ndarray, min_prob: float = 1e-5) -> None:
        self.symbol_probs_minus_t = self.symbol_probs_t.copy()
        self.symbol_probs_t += self.probs_learning_rate * gradients
        self.symbol_probs_t = np.where(self.symbol_probs_t < 0, 0, self.symbol_probs_t)
        self.symbol_probs_t /= self.symbol_probs_t.sum(axis=1).reshape(-1, 1)
        self.symbol_probs_t = np.clip(self.symbol_probs_t, min_prob, 1)
        self.symbol_probs_t /= self.symbol_probs_t.sum(axis=1).reshape(-1, 1)

    def _check_all_stop_criterias(self, iteration: int, score: float, start_time: float) -> bool:
        stop_timeout_criteria = self._check_stop_timeout(start_time)
        stop_score_criteria = self._check_stop_score(score)
        stop_max_generations_criteria = self._check_max_iterations_criteria(iteration)
        if self.verbose >= 1:
            if stop_timeout_criteria:
                logger.info('TIMEOUT STOP CRITERIA SATISFIED.')
            if stop_score_criteria:
                logger.info('SCORE STOP CRITERIA SATISFIED.')
            if stop_max_generations_criteria:
                logger.info('NUM GENERATIONS CRITERIA SATISFIED.')
        if stop_timeout_criteria or stop_score_criteria or stop_max_generations_criteria:
            if self.verbose >= 1: logger.info('STOPPING OPTIMIZATION...')
            return True
        else:
            return False

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0 or y.size == 0:
            raise ValueError(f"X and y shouldn't be empty.")
        
        start_time = time.time()
        for iteration in range(self.max_iterations):
            scores = []
            individuals = []
            individuals = self._generate_n_individuals(
                X,
                y,
                self.symbol_probs_t[:, :len(self.unary_operators)].tolist(),
                self.symbol_probs_t[:, len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)].tolist(),
                self.symbol_probs_t[:, -len(self.variables):].tolist()
            )
            scores = np.array([individual[2] for individual in individuals])
            self.search_results.add_best_individuals_by_loss_and_complexity(individuals, iteration)
            self._check_stop_score_criteria(individuals, scores)
            
            gradients = self._compute_gradients_central_difference(X, y, self.symbol_probs_t, iteration)
            self._update_probabilities(gradients)
            
            self.search_results.visualize_best_in_generation()
            print(f"Iteration {iteration}:\n\tMax Training Fitness = {self.best_individual[2]}")
            for node_index, symbol_probs_t in enumerate(self.symbol_probs_t):
                node_probs_string = f"NODE {node_index} PROBABILITIES:\n"
                for symbol, prob in zip(self.symbols, symbol_probs_t):
                    node_probs_string += f"\t{symbol} = {np.round(prob, 4)}\n"
                print(node_probs_string)
                    