import numpy as np
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt

from quant_mc_framework.simulation.engine import LongShortSimulation
from quant_mc_framework.analysis.visualization import plot_portfolio_values


def setup_parameters() -> Dict[str, Any]:
    """
    Define simulation parameters for a single run.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of parameters for the simulation
    """
    params = {
        # Market parameters
        'factor_autocorrelation': 0.1,
        'information_coefficient': 0.05,
        'annual_expected_return': 0.0,
        
        # Long-short strategy parameters
        'n_stocks': 100,
        'initial_cash': 10000000,
        'long_weight': 2.5,
        'short_weight': 1.5,
        'max_turnover': 0.5,
        'risk_aversion': 0.5,
        'single_asset_bound': 0.05,
        
        # Options overlay parameters
        'otm_percentage': 0.0,
    }
    
    return params


def run_single_simulation(seed: int = 42) -> Dict[str, Any]:
    """
    Run a single simulation with fixed parameters.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, Any]
        Simulation results including portfolio values and metrics
    """
    # Get parameters
    params = setup_parameters()

    # Create and run simulator
    simulator = LongShortSimulation(params)
    results = simulator.run(seed=seed)

    return results


def display_results(results: Dict[str, Any]) -> None:
    """
    Display simulation results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Simulation results from run_single_simulation
    """
    # Extract portfolio values
    portfolio_value = results['portfolio_value']
    portfolio_with_options = results['portfolio_value_with_options_overlay']

    # Print metrics
    print("\nPerformance Metrics:")
    print("-" * 50)
    print("Long-Short Strategy:")
    for metric, value in results['metrics']['long_short'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nWith Options Overlay:")
    for metric, value in results['metrics']['options_overlay'].items():
        print(f"  {metric}: {value:.4f}")

    # Plot portfolio values
    fig = plot_portfolio_values(portfolio_value, portfolio_with_options)
    fig.savefig('single_sim_results.png')
    print("\nPlot saved as 'single_sim_results.png'")
    

def main() -> None:
    """
    Main function to run a single simulation and display results.
    """
    print("Running single simulation...")
    results = run_single_simulation()
    display_results(results)

if __name__ == "__main__":
    main()