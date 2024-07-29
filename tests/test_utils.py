import os
import sys
import unittest

sys.path.append(os.path.abspath(os.curdir))

from src.utils import check_lists_equal

class TestUtils(unittest.TestCase):
    def test_equal_lists(self):
        """Test that the function returns True for lists with the same elements in the same order."""
        list1 = ["a", "b", "c"]
        list2 = ["a", "b", "c"]
        self.assertTrue(check_lists_equal(list1, list2))

    def test_equal_lists_different_order(self):
        """Test that the function returns True for lists with the same elements in different orders."""
        list1 = ["a", "b", "c"]
        list2 = ["c", "b", "a"]
        self.assertTrue(check_lists_equal(list1, list2))

    def test_lists_with_different_elements(self):
        """Test that the function returns False for lists with different elements."""
        list1 = ["a", "b", "c"]
        list2 = ["a", "b", "d"]
        self.assertFalse(check_lists_equal(list1, list2))

    def test_lists_with_different_lengths(self):
        """Test that the function returns False for lists with different lengths."""
        list1 = ["a", "b", "c"]
        list2 = ["a", "b"]
        self.assertFalse(check_lists_equal(list1, list2))

    def test_empty_lists(self):
        """Test that the function returns True for two empty lists."""
        list1 = []
        list2 = []
        self.assertTrue(check_lists_equal(list1, list2))

    def test_one_empty_list(self):
        """Test that the function returns False when one list is empty and the other is not."""
        list1 = ["a", "b", "c"]
        list2 = []
        self.assertFalse(check_lists_equal(list1, list2))

if __name__ == "__main__":
    unittest.main()