import random

import numpy as np
import pandas as pd

from typing import List
from binary_tree import BinaryTree

class Population:
    def __init__(
        self, 
        num_individuals: int,
        max_individual_depth: int,
        variables: List[str],
        unary_operators: List[str],
        binary_operators: List[str],
        prob_node_mutation: float,
        tournament_size: int):

        self.num_individuals = num_individuals
        self.max_individual_depth = max_individual_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.prob_node_mutation = prob_node_mutation
        self.tournament_size = tournament_size

        self.population = self._initialize_population()
    
    def _initialize_population(self) -> List[BinaryTree]:
        population = list()
        for _ in range(self.num_individuals):
            population.append(BinaryTree(
                max_possible_depth=self.max_individual_depth,
                variables=self.variables,
                unary_operators=self.unary_operators,
                binary_operators=self.binary_operators
            ))
        return population
    
    def sort_by_fitness(self) -> None:
        self.population.sort(key=lambda individual: individual.fitness, reverse=False)

    def roulette_wheel_selection(self) -> None:
        total_fitness = sum(individual.fitness for individual in self.population)
        print(total_fitness)
        selection_probs = [individual.fitness / total_fitness for individual in self.population]
        print(selection_probs)
        return random.choices(self.population, weights=selection_probs, k=2)

    def perform_simple_node_mutation(self) -> None:
        for i in range(len(self.population)):
            if random.random() <= self.prob_simple_node_mutation:
                self.population[i].perform_simple_node_mutation()
    
    def calculate_fitness(self, X: pd.DataFrame, y: pd.Series) -> List[BinaryTree]:
        for i in range(len(self.population)):
            self.population[i].calculate_fitness(X, y)