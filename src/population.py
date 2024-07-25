import random

import numpy as np
import pandas as pd

from numba import njit
from typing import List
from binary_tree import BinaryTree

class Population:
    def __init__(
        self, 
        num_individuals_per_epoch: int,
        max_individual_depth: int,
        variables: List[str],
        unary_operators: List[str],
        binary_operators: List[str],
        prob_node_mutation: float,
        tournament_ratio: float,
        elitism_rate: float,
        survival_rate):

        self.num_individuals_per_epoch = num_individuals_per_epoch
        self.max_individual_depth = max_individual_depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.prob_node_mutation = prob_node_mutation
        self.tournament_ratio = tournament_ratio
        self.elitism_rate = elitism_rate
        self.survival_rate = survival_rate

        self.population = self._create_population(num_individuals_per_epoch)
        self.epoch_parents = list()
    
    def _create_population(self, num_individuals: int) -> List[BinaryTree]:
        population = list()
        for _ in range(num_individuals):
            population.append(BinaryTree(
                max_possible_depth=self.max_individual_depth,
                variables=self.variables,
                unary_operators=self.unary_operators,
                binary_operators=self.binary_operators
            ))
        return population
    
    def sort_by_fitness(self) -> None:
        self.population.sort(key=lambda individual: individual.fitness, reverse=False)

    def roulette_wheel_selection(self, k: int = 2) -> None:
        num_individuals_in_tournament = int(
            (len(self.population) * self.tournament_ratio) - (len(self.population) * self.tournament_ratio) % k
        )
        population_to_select = [individual for individual in self.population if individual.fitness is not np.inf]
        inverted_fitness = [1.0 / individual.fitness if individual.fitness != 0 else float(np.inf) for individual in population_to_select]
        total_inverted_fitness = sum(inverted_fitness)
        selection_probs = [fitness / total_inverted_fitness for fitness in inverted_fitness]

        self.epoch_parents = [tuple(random.choices(population_to_select, weights=selection_probs, k=k)) for _ in range(num_individuals_in_tournament)]

    def perform_crossover(self):
        for parent1, parent2 in self.epoch_parents:
            parent1_node = random.choice(parent1.nodes)
            parent2_node = random.choice(parent2.nodes)
            
            parent1_subtree_parent, parent1_subtree_side = parent1_node.parent, parent1_node.parent_side()
            parent2_subtree_parent, parent2_subtree_side = parent2_node.parent, parent2_node.parent_side()

            if parent1_subtree_parent:
                if parent1_subtree_side == 'left':
                    parent1_subtree_parent.left = parent2_node
                else:
                    parent1_subtree_parent.right = parent2_node
            else:
                parent1.root = parent2_node

            if parent2_subtree_parent:
                if parent2_subtree_side == 'left':
                    parent2_subtree_parent.left = parent1_node
                else:
                    parent2_subtree_parent.right = parent1_node
            else:
                parent2.root = parent1_node

            parent1_node.parent = parent2_subtree_parent
            parent2_node.parent = parent1_subtree_parent

            parent1.update_tree_info()
            parent2.update_tree_info()

            self.population.append(parent1)
            self.population.append(parent2)

    def perform_mutation(self) -> None:
        for i in range(len(self.population)):
            my_random = random.random()
            if my_random <= self.prob_node_mutation:
                self.population[i].perform_mutation()
                self.population[i].update_tree_info()

    def calculate_fitness(self, X: pd.DataFrame, y: pd.Series) -> List[BinaryTree]:
        for i in range(len(self.population)):
            self.population[i].calculate_fitness(X, y)
    
    def prepare_next_epoch_population(self):
        num_elite_individuals = int((len(self.population) * self.elitism_rate))
        num_surviving_individuals = int((self.num_individuals_per_epoch - num_elite_individuals) * self.survival_rate)
        
        population_to_select = [individual for individual in self.population if individual.fitness is not np.inf]
        inverted_fitness = [1.0 / individual.fitness if individual.fitness != 0 else float(np.inf) for individual in population_to_select]
        total_inverted_fitness = sum(inverted_fitness)
        selection_probs = [fitness / total_inverted_fitness for fitness in inverted_fitness]
        survived_individuals = random.choices(population_to_select, weights=selection_probs, k=num_surviving_individuals)

        elite_individuals = self.population[:num_elite_individuals]
        new_individuals = self._create_population(self.num_individuals_per_epoch - len(survived_individuals) - len(elite_individuals))
        self.population = elite_individuals + survived_individuals + new_individuals