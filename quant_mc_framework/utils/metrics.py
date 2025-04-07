import pandas as pd
import numpy as np


def calculate_total_return(timeseries: pd.Series) -> float:
    """
    Calculate the total return of a timeseries.
    
    Parameters
    ----------
    timeseries : pd.Series
        Values over time
        
    Returns
    -------
    float
        Total return of the timeseries
    """
    return (timeseries.iloc[-1] / timeseries.iloc[0]) - 1

def calculate_annualized_return(timeseries: pd.Series) -> float:
    """
    Calculate the annualized return of a timeseries.
    
    Parameters
    ----------
    timeseries : pd.Series
        Values over time
        
    Returns
    -------
    float
        Annualized return of the timeseries
    """
    total_return = calculate_total_return(timeseries)
    years = (timeseries.index[-1] - timeseries.index[0]).days / 365.25
    return (1 + total_return) ** (1 / years) - 1


def calculate_sharpe_ratio(timeseries: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio of a timeseries.
    
    Parameters
    ----------
    timeseries : pd.Series
        Values over time
    risk_free_rate : float, default=0.0
        Risk-free rate for the Sharpe ratio calculation
        
    Returns
    -------
    float
        Sharpe ratio of the timeseries
    """
    annualized_return = calculate_annualized_return(timeseries)
    excess_return = annualized_return - risk_free_rate
    volatility = calculate_volatility(timeseries)

    return excess_return / volatility


def calculate_max_drawdown(timeseries: pd.Series) -> float:
    """
    Calculate the maximum drawdown of a timeseries.
    
    Parameters
    ----------
    values : pd.Series
        Values over time
        
    Returns
    -------
    float
        Maximum drawdown of the timeseries
    """
    return (timeseries / timeseries.cummax() - 1).min()


def calculate_volatility(timeseries: pd.Series) -> float:
    """
    Calculate the annualized volatility of a timeseries.
    
    Parameters
    ----------
    values : pd.Series
        Values over time
        
    Returns
    -------
    float
        Annualized volatility of the timeseries
    """
    returns = timeseries.pct_change().dropna()

    # Calculate annualization factor based on actual observation frequency
    years = (timeseries.index[-1] - timeseries.index[0]).days / 365.25
    periods_per_year = len(returns) / years

    return returns.std() * np.sqrt(periods_per_year)