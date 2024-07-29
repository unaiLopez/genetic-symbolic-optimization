import os
import sys
import time
import unittest

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd
from src.genetic_symbolic_regressor import GeneticSymbolicRegressor
from src.binary_tree import BinaryTree
from unittest.mock import patch

class TestGeneticSymbolicRegressor(unittest.TestCase):
    def setUp(self):
        """Initialize a basic GeneticSymbolicRegressor for use in tests."""
        self.regressor = GeneticSymbolicRegressor(
            num_individuals_per_epoch=5,
            max_initialization_individual_depth=3,
            variables=["x1", "x2"],
            unary_operators=["sin", "cos"],
            binary_operators=["+", "-"],
            prob_node_mutation=0.1,
            tournament_ratio=0.5,
            elitism_ratio=0.1,
            timeout=None,
            stop_loss=None,
            max_generations=10,
            verbose=0,
            loss_function="mae",
            random_state=42
        )

    def test_initialization(self):
        """Test the initialization of GeneticSymbolicRegressor."""
        self.assertEqual(self.regressor.num_individuals_per_epoch, 5)
        self.assertEqual(self.regressor.max_initialization_individual_depth, 3)
        self.assertEqual(self.regressor.variables, ["x1", "x2"])
        self.assertEqual(self.regressor.unary_operators, ["sin", "cos"])
        self.assertEqual(self.regressor.binary_operators, ["+", "-"])
        self.assertEqual(self.regressor.prob_node_mutation, 0.1)
        self.assertEqual(self.regressor.tournament_ratio, 0.5)
        self.assertEqual(self.regressor.elitism_ratio, 0.1)
        self.assertIsNone(self.regressor.timeout)
        self.assertIsNone(self.regressor.stop_loss)
        self.assertEqual(self.regressor.max_generations, 10)
        self.assertEqual(self.regressor.verbose, 0)
        self.assertEqual(self.regressor.loss_function, "mae")

    def test_create_individuals(self):
        """Test the creation of individuals."""
        individuals = self.regressor._create_individuals(3)
        self.assertEqual(len(individuals), 3)
        for individual in individuals:
            self.assertIsInstance(individual, BinaryTree)
    
    def test_sort_by_loss(self):
        """Test sorting individuals by loss."""
        individuals = self.regressor._create_individuals(3)
        for individual in individuals:
            individual.loss = np.random.rand()  # Assign random loss values
        sorted_individuals = self.regressor._sort_by_loss(individuals)
        self.assertEqual(sorted_individuals, sorted(individuals, key=lambda x: x.loss))
    
    def test_perform_crossover(self):
        """Test the crossover process."""
        individuals = self.regressor._create_individuals(2)
        for individual in individuals:
            individual.loss = np.random.rand()
        individuals = self.regressor._calculate_loss(individuals, pd.DataFrame(), pd.Series())
        parents = [(individuals[0], individuals[1])]
        offsprings = self.regressor._perform_crossover(parents)
        self.assertEqual(len(offsprings), 2)
        self.assertTrue(all(isinstance(offspring, BinaryTree) for offspring in offsprings))
    
    def test_perform_mutation(self):
        """Test the mutation process."""
        individuals = self.regressor._create_individuals(2)
        for individual in individuals:
            individual.loss = np.random.rand()
        individuals = self.regressor._calculate_loss(individuals, pd.DataFrame(), pd.Series())
        offsprings = self.regressor._perform_crossover([(individuals[0], individuals[1])])
        offsprings = self.regressor._perform_mutation(offsprings)
        self.assertTrue(all(isinstance(offspring, BinaryTree) for offspring in offsprings))
    
    def test_check_stop_timeout(self):
        """Test the timeout stopping criteria."""
        regressor_with_timeout = GeneticSymbolicRegressor(
            num_individuals_per_epoch=5,
            max_initialization_individual_depth=3,
            variables=["x1"],
            unary_operators=["sin"],
            binary_operators=["+"],
            prob_node_mutation=0.1,
            tournament_ratio=0.5,
            elitism_ratio=0.1,
            timeout=1,  # 1 second timeout
            stop_loss=None,
            max_generations=10,
            verbose=0,
            loss_function="mae",
            random_state=42
        )
        start_time = time.time()
        self.assertFalse(regressor_with_timeout._check_stop_timeout(start_time))
        time.sleep(1)
        self.assertTrue(regressor_with_timeout._check_stop_timeout(start_time))
    
    def test_check_stop_loss(self):
        """Test the loss stopping criteria."""
        regressor_with_stop_loss = GeneticSymbolicRegressor(
            num_individuals_per_epoch=5,
            max_initialization_individual_depth=3,
            variables=["x1"],
            unary_operators=["sin"],
            binary_operators=["+"],
            prob_node_mutation=0.1,
            tournament_ratio=0.5,
            elitism_ratio=0.1,
            timeout=None,
            stop_loss=10,
            max_generations=10,
            verbose=0,
            loss_function="mae",
            random_state=42
        )
        self.assertTrue(regressor_with_stop_loss._check_stop_loss(0.05))
        self.assertFalse(regressor_with_stop_loss._check_stop_loss(15))

    
    def test_check_max_generations_criteria(self):
        """Test the max generations stopping criteria."""
        self.assertTrue(self.regressor._check_max_generations_criteria(20))
        self.assertFalse(self.regressor._check_max_generations_criteria(7))
    
    @patch('builtins.print')
    def test_fit(self, mock_print):
        """Test the fit method."""
        X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
        y = pd.Series([7, 8, 9])
        self.regressor.fit(X, y)
        self.assertIsInstance(self.regressor.df_results, pd.DataFrame)
        self.assertGreater(len(self.regressor.df_results), 0)

    def test_empty_input_data(self):
        """Test behavior with empty input data."""
        X_empty = pd.DataFrame()
        y_empty = pd.Series()
        with self.assertRaises(ValueError):
            self.regressor.fit(X_empty, y_empty)
    
    def test_invalid_initialization_parameters(self):
        """Test handling of invalid initialization parameters."""
        with self.assertRaises(ValueError):
            GeneticSymbolicRegressor(
                num_individuals_per_epoch=-5,
                max_initialization_individual_depth=3,
                variables=["x1"],
                unary_operators=["sin"],
                binary_operators=["+"],
                prob_node_mutation=1.1,
                tournament_ratio=0.75,
                elitism_ratio=0.5,
                timeout=None,
                stop_loss=None,
                max_generations=10,
                verbose=0,
                loss_function="mae",
                random_state=42
            )

    def test_loss_function_selection(self):
        """Test the regressor with different loss functions."""
        for loss_function in ["mae", "mse"]:
            regressor = GeneticSymbolicRegressor(
                num_individuals_per_epoch=5,
                max_initialization_individual_depth=3,
                variables=["x1", "x2"],
                unary_operators=["sin", "cos"],
                binary_operators=["+", "-"],
                prob_node_mutation=0.1,
                tournament_ratio=0.5,
                elitism_ratio=0.1,
                timeout=None,
                stop_loss=None,
                max_generations=10,
                verbose=0,
                loss_function=loss_function,
                random_state=42
            )
            X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
            y = pd.Series([7, 8, 9])
            regressor.fit(X, y)
            self.assertIsInstance(regressor.df_results, pd.DataFrame)

    def test_tournament_ratio_effect(self):
        """Test the effect of different tournament ratios on selection."""
        tournament_ratios = [0.1, 0.5, 0.9]
        for tournament_ratio in tournament_ratios:
            regressor = GeneticSymbolicRegressor(
                num_individuals_per_epoch=5,
                max_initialization_individual_depth=3,
                variables=["x1", "x2"],
                unary_operators=["sin", "cos"],
                binary_operators=["+", "-"],
                prob_node_mutation=0.1,
                tournament_ratio=tournament_ratio,
                elitism_ratio=0.1,
                timeout=None,
                stop_loss=None,
                max_generations=10,
                verbose=0,
                loss_function="mae",
                random_state=42
            )
            X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
            y = pd.Series([7, 8, 9])
            regressor.fit(X, y)
            self.assertIsInstance(regressor.df_results, pd.DataFrame)

    def test_mutation_probability(self):
        """Test that mutation occurs according to the specified probability."""
        mutation_probabilities = [0.0, 0.1, 1.0]
        for prob in mutation_probabilities:
            regressor = GeneticSymbolicRegressor(
                num_individuals_per_epoch=5,
                max_initialization_individual_depth=3,
                variables=["x1", "x2"],
                unary_operators=["sin", "cos"],
                binary_operators=["+", "-"],
                prob_node_mutation=prob,
                tournament_ratio=0.5,
                elitism_ratio=0.1,
                timeout=None,
                stop_loss=None,
                max_generations=10,
                verbose=0,
                loss_function="mae",
                random_state=42
            )
            X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
            y = pd.Series([7, 8, 9])
            regressor.fit(X, y)
            # You would need to mock or inspect internal mutation to verify
            # that mutation occurred with the expected probability

    def test_max_generations_zero(self):
        """Test the behavior when max_generations is set to zero."""
        with self.assertRaises(ValueError):
            GeneticSymbolicRegressor(
                num_individuals_per_epoch=5,
                max_initialization_individual_depth=3,
                variables=["x1", "x2"],
                unary_operators=["sin", "cos"],
                binary_operators=["+", "-"],
                prob_node_mutation=0.1,
                tournament_ratio=0.5,
                elitism_ratio=0.1,
                timeout=None,
                stop_loss=None,
                max_generations=0,  # Zero generations
                verbose=0,
                loss_function="mae",
                random_state=42
            )

if __name__ == "__main__":
    unittest.main()