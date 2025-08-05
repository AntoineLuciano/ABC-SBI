import unittest

# --- Your package's add function would typically be in a separate file/module ---
# For demonstration, we'll define it here.
def add(x, y):
    """
    Adds two numbers together.
    """
    return x + y

# --- Unit Test Class ---
class TestAddFunction(unittest.TestCase):
    """
    A collection of unit tests for the 'add' function.
    """

    def test_add_positive_integers(self):
        """
        Test adding two positive integers.
        """
        # Test case 1: Basic addition
        self.assertEqual(add(2, 3), 5, "Should be 5")
        # Test case 2: Another pair of positive integers
        self.assertEqual(add(10, 20), 30, "Should be 30")

    def test_add_negative_integers(self):
        """
        Test adding two negative integers.
        """
        # Test case 1: Adding two negative numbers
        self.assertEqual(add(-2, -3), -5, "Should be -5")
        # Test case 2: Adding a negative and a positive number resulting in negative
        self.assertEqual(add(-10, 5), -5, "Should be -5")
        # Test case 3: Adding a negative and a positive number resulting in positive
        self.assertEqual(add(-5, 10), 5, "Should be 5")

    def test_add_floats(self):
        """
        Test adding floating-point numbers.
        """
        # Test case 1: Basic float addition
        self.assertEqual(add(2.5, 3.5), 6.0, "Should be 6.0")
        # Test case 2: Float with negative
        self.assertEqual(add(-1.5, 2.0), 0.5, "Should be 0.5")

    def test_add_zero(self):
        """
        Test adding numbers with zero.
        """
        # Test case 1: Adding zero to a positive number
        self.assertEqual(add(5, 0), 5, "Should be 5")
        # Test case 2: Adding zero to a negative number
        self.assertEqual(add(-5, 0), -5, "Should be -5")
        # Test case 3: Adding zero to zero
        self.assertEqual(add(0, 0), 0, "Should be 0")

    def test_add_large_numbers(self):
        """
        Test adding large numbers to ensure no overflow issues (for standard integer types).
        """
        self.assertEqual(add(1000000, 2000000), 3000000, "Should be 3000000")

# --- How to run the tests ---
if __name__ == '__main__':
    # This line discovers and runs all tests in the current file.
    unittest.main()
