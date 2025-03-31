# Monte Carlo Framework for Long-Short Investment Strategy

A framework for running Monte Carlo simulations of quantitative trading strategies with parameter variations.

## Installation
```
git clone https://github.com/cfgackstatter/quant_mc_framework.git
cd quant_mc_framework
pip install -e .
```

## Usage

### Run a single simulation
```python quant_mc_framework/run_single_sim.py```

### Run Monte Carlo simulations

#### Setup jobs
```python quant_mc_framework/main.py --setup```

#### Run simulations
```python quant_mc_framework/main.py --run --workers 4```

#### Analyze results
```python quant_mc_framework/main.py --analyze```

## Features

- Long-short equity strategy with factor-based stock selection
- Options overlay with covered calls and cash-secured puts
- Turnover constraints and leverage targeting
- Monte Carlo simulations with parameter variations
- Performance metrics and visualization