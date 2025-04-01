# Monte Carlo Framework for Long-Short Investment Strategy

A framework for running and analyzing Monte Carlo simulations of a quantitative trading strategy with parameter variations.

The strategy is a long-short factor strategy with an options overlay.

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

- Generate random stock price and factor data with factor autocorrelation and information coefficient
- Long-short equity strategy with factor-based stock selection
- Portfolio optimizer with turnover and leverage constraints
- Options overlay with covered calls and puts
- Monte Carlo simulations with parameter variations
- Performance metrics and visualization