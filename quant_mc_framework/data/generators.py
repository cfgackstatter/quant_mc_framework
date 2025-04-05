import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional, Dict

def generate_factor_scores(n_stocks: int, n_periods: int, factor_autocorrelation: float,
                           seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate random factor scores with autocorrelation using an AR(1) process.
    
    Parameters
    ----------
    n_stocks : int
        Number of stocks to generate factor scores for
    n_periods : int
        Number of time periods to generate
    factor_autocorrelation : float
        Autocorrelation coefficient between 0 and 1
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping stock names to arrays of factor scores
    """
    if seed is not None:
        np.random.seed(seed)

    factor_scores_dict = {}
    for stock in range(n_stocks):
        stock_name = f'Stock_{stock+1}'
        # Initialize the first factor score
        factor_scores_dict[stock_name] = [np.random.normal(0, 1)]

        # Generate subsequent factor scores using AR(1) process
        for t in range(1, n_periods):
            new_score = (factor_autocorrelation * factor_scores_dict[stock_name][-1] +
                         np.sqrt(1 - factor_autocorrelation**2) * np.random.normal(0, 1))
            factor_scores_dict[stock_name].append(new_score)
        
        # Convert list to numpy array
        factor_scores_dict[stock_name] = np.array(factor_scores_dict[stock_name])
    
    return factor_scores_dict


def generate_lognormal_returns(factor_scores: Dict[str, np.ndarray], information_coefficient: float,
                               annual_expected_return: float, volatilities: Dict[str, float], seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate log-normal returns correlated with factor scores.
    
    Parameters
    ----------
    factor_scores : Dict[str, np.ndarray]
        Dictionary mapping stock names to factor score arrays
    information_coefficient : float
        Target correlation between factor scores and returns
    annual_expected_return : float
        Expected annual return (drift) for the log-normal process
    volatilities : Dict[str, float]
        Dictionary mapping stock names to annual volatility values
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping stock names to arrays of returns
    """
    if seed is not None:
        np.random.seed(seed)

    returns_data = {}
    for stock, volatility in volatilities.items():
        noise = np.random.normal(0, 1, len(factor_scores[stock]))
        sigma = volatility / np.sqrt(12)
        mu = annual_expected_return / 12 - 0.5 * sigma**2
        correlated_normal = information_coefficient * factor_scores[stock] + np.sqrt(1 - information_coefficient**2) * noise
        raw_returns = np.exp(mu + sigma * correlated_normal) - 1
        returns_data[stock] = raw_returns

    return returns_data


def generate_stock_prices(returns_data: Dict[str, np.ndarray], initial_prices: Dict[str, float], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generate stock prices from returns data.
    
    Parameters
    ----------
    returns_data : Dict[str, np.ndarray]
        Dictionary mapping stock names to arrays of returns
    initial_prices : Dict[str, float]
        Dictionary mapping stock names to initial prices
    dates : pd.DatetimeIndex
        DatetimeIndex of dates for the price series
        
    Returns
    -------
    pd.DataFrame
        DataFrame of stock prices indexed by date
    """
    stocks = list(returns_data.keys())
    stock_prices = pd.DataFrame(index=dates, columns=stocks, dtype=float)

    # Set initial prices
    stock_prices.iloc[0] = [initial_prices[stock] for stock in stocks]
    
    # Vectorized calculation instead of loops
    for stock in stocks:
        stock_prices.loc[dates[1:], stock] = initial_prices[stock] * np.cumprod(1 + returns_data[stock])

    return stock_prices


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Calculate Black-Scholes option price.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Option strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying stock
    option_type : str, default='call'
        Type of option ('call' or 'put')
        
    Returns
    -------
    float
        Option price
    
    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    return option_price