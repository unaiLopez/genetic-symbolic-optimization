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
        probs_learning_rate: Optional[float] = 0.1,
        verbose: Optional[int] = 1,
        loss_name: Optional[str] = "mse",
        score_name: Optional[str] = "r2",
        random_state: Optional[int] = None):

        self.num_individuals_per_sample = num_individuals_per_sample
        self.max_individual_depth = max_individual_depth
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
        self.symbol_probs_t = np.full(self.num_symbols, 1 / self.num_symbols)
        self.symbol_probs_minus_t = np.random.uniform(low=1.0, high=1000.0, size=self.num_symbols)
        self.symbol_probs_minus_t /= self.symbol_probs_minus_t.sum()

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
        unary_operators_probs: List[float],
        binary_operators_probs: List[float],
        variables_probs: List[float]) -> List[Any]:

        individual = build_full_binary_tree(
            max_initialization_depth=self.max_individual_depth,
            variables=self.variables,
            unary_operators=self.unary_operators,
            binary_operators=self.binary_operators,
            unary_operators_frequencies=unary_operators_probs,
            binary_operators_frequencies=binary_operators_probs,
            variables_frequencies=variables_probs
        )
        individual[1] = calculate_loss(X, y, self.loss_function, individual[6])
        individual[2] = calculate_score(X, y, self.score_function, individual[6])
        
        return individual

    def _generate_n_individuals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unary_operators_probs: List[float],
        binary_operators_probs: List[float],
        variables_probs: List[float]) -> List[List[Any]]:

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
                sys.exit(f"WITH A SCORE OF {self.best_score}, THE BEST INDIVIDUAL IS {self.best_individual}")


    def _compute_gradients(self, X, y, probabilities, mean_score, iteration, epsilon=5e-5):
        gradients = np.zeros_like(probabilities)
        
        for i in range(len(probabilities)):
            # Perturb one parameter slightly
            probabilities[i] += epsilon

            unary_operators_probs = probabilities[:len(self.unary_operators)]
            binary_operators_probs = probabilities[len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)]
            variables_probs = probabilities[-len(self.variables):]

            perturbed_individuals = self._generate_n_individuals(
                X,
                y,
                unary_operators_probs,
                binary_operators_probs,
                variables_probs
            )
            perturbed_scores = np.array([perturbed_individual[2] for perturbed_individual in perturbed_individuals])
            self.search_results.add_best_individuals_by_loss_and_complexity(perturbed_individuals, iteration)
            self._check_stop_score_criteria(perturbed_individuals, perturbed_scores)

            

            
            # Compute the gradient (finite difference)
            #gradients[i] = (np.mean(perturbed_scores) - mean_score) / epsilon
            gradients[i] = (perturbed_scores.mean() - mean_score)
            
            # Reset the probability
            probabilities[i] -= epsilon
        
        return gradients

    def _update_probabilities(self, gradients: np.ndarray, min_prob: float = 1e-5) -> None:
        self.symbol_probs_minus_t = self.symbol_probs_t.copy()
        self.symbol_probs_t += self.probs_learning_rate * gradients
        self.symbol_probs_t = np.clip(self.symbol_probs_t, min_prob, 1)
        self.symbol_probs_t /= self.symbol_probs_t.sum()
    
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
            
            symbol_probs = self.symbol_probs_t.tolist()
            unary_operators_probs = symbol_probs[:len(self.unary_operators)]
            binary_operators_probs = symbol_probs[len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)]
            variables_probs = symbol_probs[-len(self.variables):]

            scores = []
            individuals = []
            individuals = self._generate_n_individuals(
                X,
                y,
                unary_operators_probs,
                binary_operators_probs,
                variables_probs
            )
            scores = np.array([individual[2] for individual in individuals])
            self.search_results.add_best_individuals_by_loss_and_complexity(individuals, iteration)
            self._check_stop_score_criteria(individuals, scores)
            
            
            gradients = self._compute_gradients(X, y, symbol_probs, scores.mean(), iteration)
            self._update_probabilities(gradients)
            
            
            self.search_results.visualize_best_in_generation()

            print(f"Iteration {iteration}:\n\tMax Training Fitness = {self.best_individual[2]}\n\tSymbols' Probabilities = {[(symbol, float(np.round(prob, 4))) for symbol, prob in zip(self.symbols, self.symbol_probs_t)]}")