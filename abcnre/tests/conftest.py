"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return np.random.randn(100, 5)

@pytest.fixture  
def temp_config_file(tmp_path):
    """Temporary config file for testing."""
    config_content = """
    model: gauss_gauss
    n_samples: 1000
    tolerance: 0.1
    """
    config_file = tmp_path / "test_config.yml"
    config_file.write_text(config_content)
    return config_file
