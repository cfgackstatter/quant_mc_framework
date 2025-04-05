import os
import argparse
from typing import Dict, Any, List
import warnings

from quant_mc_framework.simulation.monte_carlo import MonteCarloManager
from quant_mc_framework.analysis.visualization import plot_parameter_boxplots, plot_heatmap
from synced_collections.numpy_utils import NumpyConversionWarning

# Suppress warnings when NumPy types are automatically converted to Python types for storage in signac's job documents
warnings.filterwarnings("ignore", category=NumpyConversionWarning)


def setup_parameters() -> tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """
    Define simulation parameters and parameter ranges.
    
    Returns
    -------
    tuple[Dict[str, Any], Dict[str, List[Any]]]
        Base parameters and parameter ranges for Monte Carlo simulations
    """
    # Define structured parameters
    market_params = {
        'factor_autocorrelation': 0.1,
        'information_coefficient': 0.05,
        'annual_expected_return': 0.0,
    }

    long_short_params = {
        'n_stocks': 100,
        'initial_cash': 10000000,
        'long_weight': 2.5,
        'short_weight': 1.5,
        'max_turnover': 1.0,
        'risk_aversion': 0.5,
        'single_asset_bound': 0.05,
    }

    options_overlay_params = {
        'otm_percentage': 0.0
    }

    # Define parameter ranges (only for otm_percentage for now)
    param_ranges = {
        'otm_percentage': [-0.05, 0.0, 0.05]
    }

    # Combine all parameters into base_params for the MonteCarloManager
    base_params = {**market_params, **long_short_params, **options_overlay_params}
    
    return base_params, param_ranges


def analyze_results(mc_manager: MonteCarloManager) -> None:
    """
    Analyze simulation results and generate plots.
    
    Parameters
    ----------
    mc_manager : MonteCarloManager
        Monte Carlo manager with completed simulations
    """
    print("Analyzing results...")
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Generate plots for different metrics
    metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'volatility']
    
    for metric in metrics:
        print(f"Generating plots for {metric}...")
        
        # Get results for this metric
        results = mc_manager.get_results(metric_name=metric)
        
        # Create box plots
        fig = plot_parameter_boxplots(results, metric_name=metric)
        fig.savefig(f'plots/{metric}_boxplots.png')
        
        # Create heatmap
        try:
            fig = plot_heatmap(mc_manager, metric_name=metric)
            fig.savefig(f'plots/{metric}_heatmap.png')
        except Exception as e:
            print(f"Error creating heatmap: {e}")
    
    print("Analysis completed. Plots saved to 'plots' directory.")


def main() -> None:
    """
    Main function to run Monte Carlo simulations for trading strategy.
    """
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulations for trading strategies')
    parser.add_argument('--setup', action='store_true', help='Setup jobs')
    parser.add_argument('--run', action='store_true', help='Run simulations')
    parser.add_argument('--analyze', action='store_true', help='Analyze results')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--simulations', type=int, default=100, help='Number of simulations per parameter set')

    args = parser.parse_args()

    # Get parameters
    base_params, param_ranges = setup_parameters()

    # Create Monte Carlo manager
    mc_manager = MonteCarloManager(
        base_params=base_params,
        param_ranges=param_ranges,
        n_simulations=args.simulations
    )

    # Setup jobs if requested
    if args.setup:
        print("Setting up simulation jobs...")
        mc_manager.setup_jobs()
        print("Jobs created successfully.")

    # Run simulations if requested
    if args.run:
        print(f"Running simulations with {args.workers if args.workers else 'default'} workers...")
        mc_manager.run_all_jobs(max_workers=args.workers)
        print("Simulations completed.")

    # Analyze results if requested
    if args.analyze:
        analyze_results(mc_manager)


if __name__ == "__main__":
    main()