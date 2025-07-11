"""Basic ABCNRE workflow example."""

from abcnre import ABCSimulator, NeuralRatioEstimator, PosteriorValidator

def main():
    """Run basic ABCNRE workflow."""
    print("ABCNRE Basic Workflow Example")
    
    # Step 1: Simulation
    print("1. Setting up ABC simulation...")
    simulator = ABCSimulator()
    
    # Step 2: Training  
    print("2. Training neural ratio estimator...")
    estimator = NeuralRatioEstimator()
    
    # Step 3: Diagnostics
    print("3. Running posterior validation...")
    validator = PosteriorValidator(simulator, estimator)
    
    print("Workflow complete!")

if __name__ == "__main__":
    main()
