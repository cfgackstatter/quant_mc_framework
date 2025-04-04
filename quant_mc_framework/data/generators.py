import numpy as np
import pandas as pd
from scipy.stats import norm

def generate_factor_scores(n_stocks, n_periods, factor_autocorrelation, seed=None):
    """Generate random factor scores with autocorrelation"""
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


def generate_lognormal_returns(factor_scores, information_coefficient, annual_expected_return, volatilities, seed=None):
    """Generate log-normal returns correlated with factor scores"""
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


def generate_stock_prices(returns_data, initial_prices, dates):
    """Generate stock prices from returns"""
    stocks = list(returns_data.keys())
    stock_prices = pd.DataFrame(index=dates, columns=stocks, dtype=float)

    #Set initial prices
    stock_prices.iloc[0] = initial_prices

    # Generate stock prices based on returns
    for t in range(1, len(dates)):
        for stock in stocks:
            stock_prices.iloc[t, stock_prices.columns.get_loc(stock)] = (
                stock_prices.iloc[t-1, stock_prices.columns.get_loc(stock)] *
                (1 + returns_data[stock][t-1])
            )
    
    return stock_prices


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    return option_price