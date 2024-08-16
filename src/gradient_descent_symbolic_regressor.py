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
        max_individual_depth: int,
        variables: List[str],
        unary_operators: List[str],
        binary_operators: List[str],
        timeout: Optional[int],
        stop_score: Optional[float],
        max_iterations: Optional[int] = 100,
        probs_learning_rate: Optional[float] = 0.2,
        verbose: Optional[int] = 1,
        loss_name: Optional[str] = "mse",
        score_name: Optional[str] = "r2",
        random_state: Optional[int] = None):

        self.max_individual_depth = max_individual_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.loss_name = loss_name
        self.score_name = score_name
        self.loss_function = get_loss_function(loss_name)
        self.score_function = get_score_function(score_name)
        self.probs_learning_rate = probs_learning_rate
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.stop_score = stop_score
        self.verbose = verbose
        self.search_results = SearchResults()

        self.num_symbols = len(self.unary_operators) + len(self.binary_operators) + len(self.variables)
        self.symbol_probs_t = np.full(self.num_symbols, 1 / self.num_symbols)
        self.symbol_probs_minus_t = np.random.uniform(low=1.0, high=1000.0, size=self.num_symbols)
        self.symbol_probs_minus_t /= self.symbol_probs_minus_t.sum()
        print(self.symbol_probs_t)
        print(self.symbol_probs_minus_t)
        raise Exception
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
    
    def _update_probabilities(self, individual: List[Any]) -> Tuple[List[float], List[float], List[float]]:
        symbols = self.unary_operators + self.binary_operators + self.variables



        #unary_operators_frequencies = new_frequencies[:len(self.unary_operators)]
        #binary_operators_frequencies = new_frequencies[len(self.unary_operators):len(self.unary_operators) + len(self.binary_operators)]
        #variables_frequencies = new_frequencies[:len(self.variables)]

        #return unary_operators_frequencies, binary_operators_frequencies, variables_frequencies
        return None, None, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0 or y.size == 0:
            raise ValueError(f"X and y shouldn't be empty.")
        
        start_time = time.time()

        for _ in range(self.max_iterations):
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
                
            unary_operators_frequencies, binary_operators_frequencies, variables_frequencies = self.update_probabilities(individual)

