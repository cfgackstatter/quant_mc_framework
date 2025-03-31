import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_portfolio_values(portfolio_value, portfolio_with_options, title="Portfolio Performance"):
    """Plot portfolio values over time"""
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, label='Base Strategy')
    plt.plot(portfolio_with_options, label='With Options Overlay')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_parameter_boxplots(results, metric_name='sharpe_ratio'):
    """Create box plots showing the distribution of metrics by parameter"""
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 5*len(results)))

    if len(results) == 1:
        axes = [axes]

    for i, (param, param_data) in enumerate(results.items()):
        ax = axes[i]

        # Convert to DataFrame for easier plotting
        data_list = []
        for value, metrics in param_data.items():
            for metric in metrics:
                data_list.append({'Parameter Value': value, 'Metric': metric})

        df = pd.DataFrame(data_list)

        # Create box plot
        sns.boxplot(x='Parameter Value', y='Metric', data=df, ax=ax)
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
                
    plt.tight_layout()
    return fig


def plot_heatmap(mc_manager, metric_name='sharpe_ratio', strategy='options_overlay'):
    """Create a heatmap for two parameters' impact on a metric"""
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
                            metrics.append(sim[strategy][metric_name])

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
