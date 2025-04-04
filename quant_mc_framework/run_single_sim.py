from quant_mc_framework.simulation.engine import LongShortSimulation
from quant_mc_framework.analysis.visualization import plot_portfolio_values
import matplotlib.pyplot as plt


def run_single_simulation():
    """Run a single simulation and plot results"""
    # Define parameters
    params = {
        # Market parameters
        'factor_autocorrelation': 0.2,
        'information_coefficient': 0.025,
        'annual_expected_return': 0.0,

        # Long-Short strategy parameters
        'n_stocks': 100,
        'initial_cash': 10000000,
        'long_weight': 1.3,
        'short_weight': 0.3,
        'max_turnover': 1.0,
        'risk_aversion': 0.5,
        'single_asset_bound': 0.05,
        
        # Options overlay parameters
        'otm_percentage': 0.05,
    }

    # Create and run simulation
    simulator = LongShortSimulation(params)
    results = simulator.run(seed=42)

    # Extract results
    portfolio_value = results['portfolio_value']
    portfolio_with_options = results['portfolio_value_with_options_overlay']
    metrics = results['metrics']

    # Plot results
    fig = plot_portfolio_values(portfolio_value, portfolio_with_options)
    plt.savefig('single_sim_results.png')
   
    # Print metrics
    print("\nLong-Short Strategy Metrics:")
    for key, value in metrics['base_strategy'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nLong-Short Strategy w/ Options Overlay Metrics:")
    for key, value in metrics['options_overlay'].items():
        print(f"  {key}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    print("Running single simulation...")
    results = run_single_simulation()
    print("\nSimulation completed successfully!")