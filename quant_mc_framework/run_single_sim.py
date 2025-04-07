from typing import Dict, Any

from quant_mc_framework.config import get_parameters
from quant_mc_framework.simulation.engine import LongShortSimulation
from quant_mc_framework.analysis.visualization import plot_portfolio_values


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
    params, _ = get_parameters()

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