import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def plot_portfolio_values(portfolio_value: pd.Series, portfolio_with_options: pd.Series,
                          title: str = "Portfolio Performance") -> plt.Figure:
    """
    Plot portfolio values over time.

    Parameters
    ----------
    portfolio_value : pd.Series
        Series of portfolio values for the base strategy
    portfolio_with_options : pd.Series
        Series of portfolio values with options overlay
    title : str, default="Portfolio Performance"
        Title of the plot

    Returns
    -------
    plt.Figure
        Matplotlib figure object

    Notes
    -----
    This function creates a line plot comparing the performance of the base strategy
    and the strategy with options overlay over time.
    """
    # Create a figure and axis for the plot
    plt.figure(figsize=(12, 6))

    # Plot the base strategy portfolio values
    plt.plot(portfolio_value, label='Base Strategy', color='blue')

    # Plot the portfolio values with options overlay
    plt.plot(portfolio_with_options, label='With Options Overlay', color='orange')

    # Add title and labels
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Add legend to distinguish the lines
    plt.legend()

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Return the figure object
    return plt.gcf()


def plot_parameter_boxplots(results: Dict[str, Dict[Any, List[float]]],
                            metric_name: str = 'sharpe_ratio') -> plt.Figure:
    """
    Create box plots showing the distribution of metrics by parameter.

    Parameters
    ----------
    results : Dict[str, Dict[Any, List[float]]]
        Dictionary mapping parameter names to dictionaries of parameter values and metric lists
    metric_name : str, default='sharpe_ratio'
        Name of the metric to plot

    Returns
    -------
    plt.Figure
        Matplotlib figure object

    Notes
    -----
    This function creates box plots for each parameter, showing the distribution
    of the specified metric across different parameter values.
    """
    # Create subplots for each parameter
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 5*len(results)))

    # Ensure axes is iterable even if there's only one parameter
    if len(results) == 1:
        axes = [axes]

    for i, (param, param_data) in enumerate(results.items()):
        ax = axes[i]

        # Convert parameter data to a DataFrame for easier plotting
        data_list = []
        for value, metrics in param_data.items():
            for metric in metrics:
                data_list.append({'Parameter Value': value, 'Metric': metric})
        df = pd.DataFrame(data_list)

        # Check if DataFrame is empty
        if df.empty:
            ax.text(0.5, 0.5, "No data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(f'Impact of {param} on {metric_name}')
            continue

        # Create box plot
        sns.boxplot(x='Parameter Value', y='Metric', data=df, ax=ax)

        # Add title and labels
        ax.set_title(f'Impact of {param} on {metric_name}')
        ax.set_xlabel(param)
        ax.set_ylabel(metric_name)

        # Add mean values as text
        for j, value in enumerate(sorted(param_data.keys())):
            if param_data[value]:
                mean_val = np.mean(param_data[value])
                ax.text(j, mean_val, f'{mean_val:.3f}',
                        horizontalalignment='center', size='small',
                        color='black', weight='semibold')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()

    return fig


def plot_heatmap(mc_manager, metric_name: str = 'sharpe_ratio',
                 strategy: str = 'options_overlay') -> Optional[plt.Figure]:
    """
    Create a heatmap for two parameters' impact on a metric.
    
    Parameters
    ----------
    mc_manager : MonteCarloManager
        Monte Carlo manager with simulation results
    metric_name : str, default='sharpe_ratio'
        Name of the metric to visualize
    strategy : str, default='options_overlay'
        Strategy to extract metrics from
        
    Returns
    -------
    plt.Figure or None
        Matplotlib figure object or None if insufficient parameters
        
    Raises
    ------
    ValueError
        If fewer than two parameters are available for the heatmap
    """
    # Get all paramter keys
    param_keys = list(mc_manager.param_ranges.keys())

    if len(param_keys) < 2:
        raise ValueError("Need at least two parameters for a heatmap")
    
    # Select the first two parameters
    param1, param2 = param_keys[0], param_keys[1]
    param1_values = sorted(mc_manager.param_ranges[param1])
    param2_values = sorted(mc_manager.param_ranges[param2])

    # Create a matrix to store average metric values
    heatmap_data = np.zeros((len(param1_values), len(param2_values)))

    # Fill the matrix with average metric values
    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            # Find jobs with these parameter values
            jobs = mc_manager.project.find_jobs({param1: val1, param2: val2})

            metrics = []
            for job in jobs:
                if job.doc.get('status') == 'completed':
                    for sim in job.doc.get('simulations', []):
                        if strategy in sim and metric_name in sim[strategy]:
                            metric_value = sim[strategy][metric_name]
                            if not np.isnan(metric_value):
                                metrics.append(metric_value)

            if metrics:
                heatmap_data[i, j] = np.mean(metrics)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", 
               xticklabels=param2_values, yticklabels=param1_values,
               cmap="viridis")
    
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.title(f'Impact of {param1} and {param2} on {metric_name}')

    plt.tight_layout()
    
    return plt.gcf()