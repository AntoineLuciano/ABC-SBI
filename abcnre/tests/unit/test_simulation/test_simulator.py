"""Tests for ABC simulator."""

import pytest
import numpy as np
from abcnre.simulation import ABCSimulator

class TestABCSimulator:
    """Test class for ABCSimulator."""
    
    def test_init(self):
        """Test simulator initialization."""
        simulator = ABCSimulator()
        assert simulator.config == {}
        assert simulator.model is None
        
    def test_init_with_config(self):
        """Test simulator initialization with config."""
        config = {"model": "gauss_gauss", "n_samples": 1000}
        simulator = ABCSimulator(config=config)
        assert simulator.config == config
