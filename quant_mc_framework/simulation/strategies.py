import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.stats import norm


def calculate_alphas(factor_scores, cov_matrix):
    """Calculate portfolio weights based on factor scores and covariance matrix"""
    ranked_stocks = factor_scores.rank(method='first', ascending=True)
    demeaned_ranks = ranked_stocks - ranked_stocks.mean()
    standardized_ranks = demeaned_ranks / demeaned_ranks.std()

    # Calculate portfolio volatility
    portfolio_volatility = np.sqrt(standardized_ranks.dot(cov_matrix).dot(standardized_ranks))

    # Risk-adjust the standardized ranks
    risk_adjusted_ranks = standardized_ranks / portfolio_volatility

    # Transform to alpha space
    alphas = cov_matrix.dot(risk_adjusted_ranks)

    return alphas


def optimize_weights(alphas, old_weights, max_turnover, long_weight, short_weight, cov_matrix, risk_aversion, single_asset_bound):
    """Optimize weights with turnover constraint and leverage target"""
    n_stocks = len(alphas)

    # Set up the optimization problem with split variables
    w_long = cp.Variable(n_stocks)
    w_short = cp.Variable(n_stocks)
    w = w_long - w_short

    # Objective: minimize squared difference to target weights
    objective = cp.Maximize(w @ alphas - risk_aversion * cp.quad_form(w, cov_matrix))

    # Constraints
    constraints = [
        # Non-negativity constraints
        w_long >= 0,
        w_short >= 0,

        # Asset size constraints
        w <= single_asset_bound,
        w >= -single_asset_bound,

        # Turnover constraint
        cp.norm(w - old_weights, 1) <= max_turnover,

        # Total exposure constraints
        cp.sum(w_long) == long_weight,
        cp.sum(w_short) == short_weight,
    ]

    #Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status == "optimal":
        return w_long.value - w_short.value
    else:
        return old_weights