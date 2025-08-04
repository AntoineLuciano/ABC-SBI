# examples/configs/diagnostics/test_config.py

"""
Unit tests for SBC configuration handling.
"""

import unittest
import tempfile
import yaml
from pathlib import Path
import sys

# Add the src directory to the path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from abcnre.diagnostics.sbc_not_ready import SBCConfig
from config_loader import load_sbc_config_from_yaml, list_available_variants


class TestSBCConfig(unittest.TestCase):
    """Test cases for SBCConfig class."""

    def test_valid_config_creation(self):
        """Test creation of valid SBC configuration."""
        config = SBCConfig(
            num_sbc_rounds=100,
            num_posterior_samples=500,
            use_correction="both",
            initial_bounds=(0.0, 10.0),
            estimator_path="test_estimator.yaml",
            output_results_path="test_results.csv",
            seed=42,
        )

        self.assertEqual(config.num_sbc_rounds, 100)
        self.assertEqual(config.num_posterior_samples, 500)
        self.assertEqual(config.use_correction, "both")
        self.assertEqual(config.initial_bounds, (0.0, 10.0))
        self.assertEqual(config.seed, 42)
        self.assertTrue(config.save_results)
        self.assertTrue(config.save_plots)

    def test_invalid_correction_value(self):
        """Test that invalid correction values raise errors."""
        with self.assertRaises(ValueError):
            SBCConfig(
                num_sbc_rounds=100,
                num_posterior_samples=500,
                use_correction="maybe",  # Invalid
                initial_bounds=(0.0, 10.0),
                estimator_path="test.yaml",
                output_results_path="test.csv",
            )

    def test_invalid_negative_rounds(self):
        """Test that negative rounds raise errors."""
        with self.assertRaises(ValueError):
            SBCConfig(
                num_sbc_rounds=-10,  # Invalid
                num_posterior_samples=500,
                use_correction="yes",
                initial_bounds=(0.0, 10.0),
                estimator_path="test.yaml",
                output_results_path="test.csv",
            )

    def test_invalid_negative_samples(self):
        """Test that negative samples raise errors."""
        with self.assertRaises(ValueError):
            SBCConfig(
                num_sbc_rounds=100,
                num_posterior_samples=-500,  # Invalid
                use_correction="yes",
                initial_bounds=(0.0, 10.0),
                estimator_path="test.yaml",
                output_results_path="test.csv",
            )

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = SBCConfig(
            num_sbc_rounds=100,
            num_posterior_samples=500,
            use_correction="yes",
            initial_bounds=(0.0, 10.0),
            estimator_path="test_estimator.yaml",  # String
            output_results_path="test_results.csv",  # String
            output_plots_path="test_plots.png",  # String
        )

        self.assertIsInstance(config.estimator_path, Path)
        self.assertIsInstance(config.output_results_path, Path)
        self.assertIsInstance(config.output_plots_path, Path)


class TestConfigLoader(unittest.TestCase):
    """Test cases for config loader functions."""

    def setUp(self):
        """Set up temporary YAML file for testing."""
        self.test_config = {
            "test_variant": {
                "num_sbc_rounds": 50,
                "num_posterior_samples": 200,
                "use_correction": "no",
                "initial_bounds": [0.0, 5.0],
                "estimator_path": "test_estimator.yaml",
                "output_results_path": "test_results.csv",
                "save_results": True,
                "save_plots": False,
                "seed": 123,
            },
            "another_variant": {
                "num_sbc_rounds": 100,
                "num_posterior_samples": 1000,
                "use_correction": "both",
                "initial_bounds": [0.0, 10.0],
                "estimator_path": "another_estimator.yaml",
                "output_results_path": "another_results.csv",
                "save_results": True,
                "save_plots": True,
                "seed": 456,
            },
        }

        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(self.test_config, self.temp_file)
        self.temp_file.close()
        self.temp_path = Path(self.temp_file.name)

    def tearDown(self):
        """Clean up temporary file."""
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        config = load_sbc_config_from_yaml(self.temp_path, "test_variant")

        self.assertEqual(config.num_sbc_rounds, 50)
        self.assertEqual(config.num_posterior_samples, 200)
        self.assertEqual(config.use_correction, "no")
        self.assertEqual(config.initial_bounds, (0.0, 5.0))
        self.assertEqual(config.seed, 123)
        self.assertFalse(config.save_plots)

    def test_load_nonexistent_variant(self):
        """Test loading non-existent variant raises error."""
        with self.assertRaises(ValueError):
            load_sbc_config_from_yaml(self.temp_path, "nonexistent_variant")

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            load_sbc_config_from_yaml("nonexistent_file.yaml", "test_variant")

    def test_list_available_variants(self):
        """Test listing available variants."""
        variants = list_available_variants(self.temp_path)

        self.assertIn("test_variant", variants)
        self.assertIn("another_variant", variants)
        self.assertEqual(len(variants), 2)


class TestCalibrationYAML(unittest.TestCase):
    """Test cases for the actual calibration.yaml file."""

    def setUp(self):
        """Set up path to calibration.yaml."""
        self.config_path = Path(__file__).parent / "calibration.yaml"

    def test_calibration_yaml_exists(self):
        """Test that calibration.yaml file exists."""
        self.assertTrue(
            self.config_path.exists(),
            f"calibration.yaml not found at {self.config_path}",
        )

    def test_calibration_yaml_structure(self):
        """Test that calibration.yaml has expected structure."""
        if not self.config_path.exists():
            self.skipTest("calibration.yaml not found")

        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Check expected variants exist
        expected_variants = ["quick_test", "defaults", "heavy"]
        for variant in expected_variants:
            self.assertIn(
                variant,
                config_data,
                f"Variant '{variant}' not found in calibration.yaml",
            )

        # Check each variant has required fields
        required_fields = [
            "num_sbc_rounds",
            "num_posterior_samples",
            "use_correction",
            "initial_bounds",
            "estimator_path",
            "output_results_path",
            "save_results",
            "save_plots",
            "seed",
        ]

        for variant in expected_variants:
            variant_config = config_data[variant]
            for field in required_fields:
                self.assertIn(
                    field,
                    variant_config,
                    f"Field '{field}' missing in variant '{variant}'",
                )

    def test_load_all_variants(self):
        """Test loading all variants from calibration.yaml."""
        if not self.config_path.exists():
            self.skipTest("calibration.yaml not found")

        variants = ["quick_test", "defaults", "heavy"]

        for variant in variants:
            try:
                config = load_sbc_config_from_yaml(self.config_path, variant)

                # Basic validation
                self.assertGreater(config.num_sbc_rounds, 0)
                self.assertGreater(config.num_posterior_samples, 0)
                self.assertIn(config.use_correction, ["yes", "no", "both"])
                self.assertEqual(len(config.initial_bounds), 2)

                print(f"Variant '{variant}' loaded successfully")

            except Exception as e:
                self.fail(f"Failed to load variant '{variant}': {e}")


class TestConfigValidation(unittest.TestCase):
    """Test cases for configuration validation."""

    def test_correction_values(self):
        """Test all valid correction values."""
        valid_corrections = ["yes", "no", "both"]

        for correction in valid_corrections:
            config = SBCConfig(
                num_sbc_rounds=100,
                num_posterior_samples=500,
                use_correction=correction,
                initial_bounds=(0.0, 10.0),
                estimator_path="test.yaml",
                output_results_path="test.csv",
            )
            self.assertEqual(config.use_correction, correction)

    def test_bounds_handling(self):
        """Test different bounds formats."""
        # Tuple bounds
        config1 = SBCConfig(
            num_sbc_rounds=100,
            num_posterior_samples=500,
            use_correction="yes",
            initial_bounds=(0.0, 10.0),
            estimator_path="test.yaml",
            output_results_path="test.csv",
        )
        self.assertEqual(config1.initial_bounds, (0.0, 10.0))

        # List bounds (should work too)
        config2 = SBCConfig(
            num_sbc_rounds=100,
            num_posterior_samples=500,
            use_correction="yes",
            initial_bounds=[0.0, 10.0],
            estimator_path="test.yaml",
            output_results_path="test.csv",
        )
        self.assertEqual(config2.initial_bounds, [0.0, 10.0])


def run_tests():
    """Run all tests and display results."""

    print("🧪 RUNNING SBC CONFIGURATION TESTS")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestSBCConfig,
        TestConfigLoader,
        TestCalibrationYAML,
        TestConfigValidation,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
