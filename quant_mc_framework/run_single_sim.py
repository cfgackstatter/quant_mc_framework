from quant_mc_framework.simulation.engine import LongShortSimulation
from quant_mc_framework.analysis.visualization import plot_portfolio_values
import matplotlib.pyplot as plt


def run_single_simulation():
    """Run a single simulation and plot results"""
    # Define parameters
    params = {
        'n_stocks': 500,
        'target_ic': 0.05,
        'target_leverage': 1.5,
        'max_turnover': 1,
        'initial_cash': 10000000,
        'autocorrelation': 0.1,
        'otm_percentage': 0.00,
    }

    # Create and run simulation
    simulator = LongShortSimulation(params)
    results = simulator.run(seed=42)

    # Extract results
    portfolio_value = results['portfolio_value']
    portfolio_with_options = results['portfolio_with_options']
    metrics = results['metrics']

    # Plot results
    fig = plot_portfolio_values(portfolio_value, portfolio_with_options)
    plt.savefig('single_sim_results.png')
   
    # Print metrics
    print("\nBase Strategy Metrics:")
    for key, value in metrics['base_strategy'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nOptions Overlay Metrics:")
    for key, value in metrics['options_overlay'].items():
        print(f"  {key}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    print("Running single simulation...")
    results = run_single_simulation()
    print("\nSimulation completed successfully!")