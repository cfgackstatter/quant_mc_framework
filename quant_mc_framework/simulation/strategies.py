import numpy as np
import pandas as pd
import cvxpy as cp


def calculate_alphas(factor_scores: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Calculate alpha signals based on factor scores and risk model.
    
    Parameters
    ----------
    factor_scores : pd.Series
        Factor scores for each stock
    cov_matrix : pd.DataFrame
        Covariance matrix of stock returns (annualized)
        
    Returns
    -------
    pd.Series
        Alpha signals for portfolio optimization
        
    Notes
    -----
    This function transforms factor scores into alpha signals by:
    1. Ranking the factor scores
    2. Demeaning and standardizing the ranks
    3. Risk-adjusting the standardized ranks
    4. Transforming to alpha space using the covariance matrix
    """
    # Convert factor scores to ranks (higher factor score = higher rank)
    ranked_stocks = factor_scores.rank(method='first', ascending=True)

    # Demean the ranks to center around zero
    demeaned_ranks = ranked_stocks - ranked_stocks.mean()

    # Standardize to unit variance
    standardized_ranks = demeaned_ranks / demeaned_ranks.std()

    # Calculate portfolio volatility of the standardized ranks long-short portfolio
    portfolio_volatility = np.sqrt(standardized_ranks.dot(cov_matrix).dot(standardized_ranks))

    # Risk-adjust the standardized ranks long-short portfolio
    risk_adjusted_ranks = standardized_ranks / portfolio_volatility

    # Transform to alpha space by multiplying by covariance matrix
    # This converts factor signals into expected return space
    alphas = cov_matrix.dot(risk_adjusted_ranks)

    return alphas


def optimize_weights(
        alphas: np.ndarray,
        old_weights: np.ndarray,
        old_cash_weight: float,
        max_turnover: float,
        long_weight: float,
        short_weight: float,
        cov_matrix: np.ndarray,
        risk_aversion: float,
        single_asset_bound: float
    ) -> np.ndarray:
    """
    Optimize portfolio weights with turnover constraint and long and short target.
    
    Parameters
    ----------
    alphas : np.ndarray
        Alpha signals for each stock
    old_weights : np.ndarray
        Previous portfolio weights
    old_cash_weight : float
        Previous cash weight
    max_turnover : float
        Maximum allowed turnover (sum of absolute weight changes)
    long_weight : float
        Target long exposure (e.g., 1.0 for 100%, 1.3 for 130%)
    short_weight : float
        Target short exposure (e.g., 0.0 for long-only, 0.3 for 130/30)
    cov_matrix : np.ndarray
        Covariance matrix of stock returns
    risk_aversion : float
        Risk aversion parameter (higher = more risk-averse)
    single_asset_bound : float
        Maximum allowed weight for any single asset (e.g., 0.05 for 5%)
        
    Returns
    -------
    np.ndarray
        Optimized portfolio weights
        
    Notes
    -----
    This function solves a quadratic optimization problem to maximize:
    alpha return - risk_aversion * portfolio_variance
    
    Subject to constraints on:
    - Long and short exposures
    - Position size limits
    - Portfolio turnover
    """
    n_stocks = len(alphas)

    # Regularize the covariance matrix to ensure numerical stability
    cov_matrix_reg = cov_matrix + np.eye(n_stocks) * 1e-6

    # Set up the optimization problem with split variables for long and short positions
    w_long = cp.Variable(n_stocks)      # Long positions (always positive)
    w_short = cp.Variable(n_stocks)     # Short positions (always positive)
    w = w_long - w_short                # Net positions (can be positive or negative)

    # Objective: maximize alpha return minus risk penalty
    # The risk term uses a quadratic form to represent portfolio variance
    objective = cp.Maximize(w @ alphas - risk_aversion * cp.quad_form(w, cov_matrix_reg))

    # Constraints
    constraints = [
        # Non-negativity constraints for long and short positions
        w_long >= 0,
        w_short >= 0,

        # Asset size constraints (limit concentration risk)
        w <= single_asset_bound,   # Cap on long positions
        w >= -single_asset_bound,  # Cap on short positions

        # Turnover constraint (limit transaction costs)
        cp.norm((w - old_weights), 1) + np.abs(1.0 - long_weight + short_weight - old_cash_weight) <= max_turnover,

        # Total exposure constraints
        cp.sum(w_long) == long_weight,              # Target long exposure
        cp.sum(w_short) == short_weight,            # Target short exposure
    ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    try:
        # Use a more robust solver if available
        problem.solve(solver=cp.OSQP)
    except cp.SolverError:
        # Fall back to default solver
        problem.solve()

    # Return optimized weights if solution is optimal, otherwise return old weights
    if problem.status == "optimal":
        return w_long.value - w_short.value
    else:
        print(f"Warning: Optimization failed with status {problem.status}")
        return old_weights