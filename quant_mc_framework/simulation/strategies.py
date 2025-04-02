import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.stats import norm


def calculate_factor_weights(factor_scores, target_leverage=1.0):
    """Calculate portfolio weights based on factor scores"""
    ranked_stocks = factor_scores.rank(method='first', ascending=True)
    demeaned_ranks = ranked_stocks - ranked_stocks.mean()
    standardized_ranks = demeaned_ranks / demeaned_ranks.std()
    return standardized_ranks / standardized_ranks.abs().sum() * (2 * target_leverage)


def optimize_weights(target_weights, drifted_weights, max_turnover, target_leverage):
    """Optimize weights with turnover constraint and leverage target"""
    n_stocks = len(target_weights)

    # Set up the optimization problem with split variables
    w_long = cp.Variable(n_stocks)
    w_short = cp.Variable(n_stocks)

    # Objective: minimize squared difference to target weights
    objective = cp.Minimize(cp.sum_squares(w_long - w_short - target_weights))

    # Constraints
    constraints = [
        # Non-negativity constraints
        w_long >= 0,
        w_short >= 0,

        # Turnover constraint
        cp.norm(w_long - w_short - drifted_weights, 1) <= max_turnover,

        # Cash neutral constraint
        cp.sum(w_long - w_short) == 0,

        # Leverage constraint
        cp.sum(w_long + w_short) == 2 * target_leverage,
    ]

    #Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status == "optimal":
        return w_long.value - w_short.value
    else:
        return drifted_weights