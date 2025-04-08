# Long-Short Monte Carlo Simulator
A comprehensive framework for simulating and analyzing long-short equity strategies with options overlays, including realistic financing costs.

## Overview
This framework provides tools for:

- Simulating long-short equity strategies with customizable parameters
- Adding options overlay strategies to enhance returns
- Incorporating realistic financing costs, including cash interest, margin costs, and stock borrow fees
- Running Monte Carlo simulations to analyze parameter sensitivity
- Visualizing and analyzing simulation results

## Features
### Stock Market Generation
- Generate factor scores and stock returns
- Set factor autocorrelation and information coefficient

### Long-Short Strategy Implementation
- Customizable long and short exposure levels (e.g. 130/30, 100/0, 250/150)
- Factor-based stock selection with alpha generation
- Quadratic optimization with turnover constraints
- Realistic cash management and financing

### Options Overlay
- Covered call writing on long positions
- Covered put writing on short positions
- Black-Scholes option pricing
- Customizable out-of-the-money percentages

### Financing Costs
- Interest earned on cash balances including sale proceeds
- Margin costs for leveraged positions
- Stock borrow fees with different rates for easy and hard-to-borrow stocks

### Monte Carlo Simulation
- Parameter sensitivity analysis
- Parallel processing for efficient computation
- Job management with signac

### Analysis Tools
- Performance metrics calculation (Sharpe ratio, returns, drawdowns, etc.)
- Visualization of results with matplotlib and seaborn

## Installation
```console
# Clone the repository
git clone https://github.com/cfgackstatter/quant_mc_framework.git
cd quant_mc_framework

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e
```

## Usage
### Running a Single Simulation
```python quant_mc_framework/run_single_sim.py```

This will run a single simulation with the default parameters and save a plot of the portfolio values to `single_sim_results.png`.

### Running Monte Carlo Simulations

```console
# Set up simulation jobs
python quant_mc_framework/main.py --setup

# Run simulations (optionally specify number of worker processes)
python quant_mc_framework/main.py --run --workers 4

# Analyze results and generate plots
python quant_mc_framework/main.py --analyze
```

### Customizing Parameters
Edit the config.py file to customize simulation parameters:

```python
# Example: Change to a 130/30 strategy
long_short_params = {
    'n_stocks': 100,
    'initial_cash': 10000000,
    'long_weight': 1.3,
    'short_weight': 0.3,
    'max_turnover': 0.5,
    'risk_aversion': 1.0,
    'single_asset_bound': 0.05,
}
```

## Key Components
- `engine.py`: Core simulation engine for running strategies
- `strategies.py`: Portfolio optimization and alpha generation
- `monte_carlo.py`: Monte Carlo simulation management
- `generators.py`: Data generation for simulations
- `metrics.py`: Performance metric calculations
- `visualization.py`: Plotting and visualization tools
- `config.py`: Parameter configuration

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.