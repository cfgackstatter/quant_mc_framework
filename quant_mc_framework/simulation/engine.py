import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any

from ..data.generators import (
    generate_factor_scores, generate_lognormal_returns,
    generate_stock_prices, black_scholes
)

from ..simulation.strategies import (
    calculate_alphas, optimize_weights
)

from ..utils.metrics import (
    calculate_total_return, calculate_annualized_return,
    calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_volatility
)


class LongShortSimulation:
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize simulation with parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of simulation parameters including market parameters,
            strategy parameters, and options overlay parameters
        """
        self.params = params
        self.results = {}


    def run(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single simulation with the specified parameters.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing simulation results:
            - portfolio_value: Series of portfolio values over time
            - portfolio_value_with_options_overlay: Series with options overlay
            - metrics: Performance metrics for both strategies
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            # Generate different seeds for different components
            factor_seed = seed
            returns_seed = seed + 1000

        # Extract parameters from different categories
        # Market parameters
        factor_autocorrelation = self.params.get('factor_autocorrelation')
        information_coefficient = self.params.get('information_coefficient')
        annual_expected_return = self.params.get('annual_expected_return')

        # Financing parameters
        cash_interest_rate = self.params.get('cash_interest_rate')
        margin_interest_rate = self.params.get('margin_interest_rate')
        borrow_fee_base = self.params.get('borrow_fee_base')
        borrow_fee_hard = self.params.get('borrow_fee_hard')
        hard_to_borrow_pct = self.params.get('hard_to_borrow_pct')

        # Long-short strategy parameters
        n_stocks = self.params.get('n_stocks')
        initial_cash = self.params.get('initial_cash')
        long_weight = self.params.get('long_weight')
        short_weight = self.params.get('short_weight')
        max_turnover = self.params.get('max_turnover')
        risk_aversion = self.params.get('risk_aversion')
        single_asset_bound = self.params.get('single_asset_bound')

        # Options overlay parameters
        otm_percentage = self.params.get('otm_percentage')

        # Generate dates
        start_date = pd.Timestamp('2019-12-01')
        end_date = pd.Timestamp('2025-01-01')
        months = pd.date_range(start=start_date, end=end_date, freq='ME')

        # Generate stock names
        stocks = [f'Stock_{i}' for i in range(1, n_stocks+1)]

        # Generate factor scores with autocorrelation
        factor_scores_dict = generate_factor_scores(
            n_stocks, len(months)-1, factor_autocorrelation, factor_seed
        )

        factor_scores_df = pd.DataFrame(factor_scores_dict, index=months[:-1])

        # Generate volatilities
        volatilities = np.random.uniform(0.15, 0.75, n_stocks)
        volatilities_dict = dict(zip(stocks, volatilities))

        # Generate returns, stock prices and covariance matrix (annualized)
        returns_data = generate_lognormal_returns(
            factor_scores_dict, information_coefficient, annual_expected_return,
            volatilities_dict, returns_seed
        )

        initial_prices = np.random.uniform(10, 150, n_stocks)
        initial_prices_dict = dict(zip(stocks, initial_prices))
        
        stock_prices = generate_stock_prices(returns_data, initial_prices_dict, months)
        
        cov_matrix = stock_prices.pct_change().cov() * 12

        # Run long-short strategy
        portfolio_value, weights, shares = self.run_long_short_strategy(
            factor_scores_df, stock_prices, months,
            long_weight, short_weight, max_turnover, initial_cash,
            cov_matrix, risk_aversion, single_asset_bound, cash_interest_rate,
            margin_interest_rate, hard_to_borrow_pct, borrow_fee_base,
            borrow_fee_hard
        )

        # Run options overlay
        portfolio_value_with_options = self.run_options_overlay(
            factor_scores_df, stock_prices, shares,
            volatilities_dict, months, otm_percentage, portfolio_value
        )

        # Calculate performance metrics
        metrics = self.calculate_metrics(portfolio_value, portfolio_value_with_options)

        return {
            'portfolio_value': portfolio_value,
            'portfolio_value_with_options_overlay': portfolio_value_with_options,
            'metrics': metrics,
        }
    

    def run_long_short_strategy(
            self, factor_scores_df: pd.DataFrame, stock_prices: pd.DataFrame, months: pd.DatetimeIndex,
            long_weight: float, short_weight: float, max_turnover: float, initial_cash: float,
            cov_matrix: pd.DataFrame, risk_aversion: float, single_asset_bound: float,
            cash_interest_rate: float, margin_interest_rate: float, hard_to_borrow_pct: float, borrow_fee_base: float,
            borrow_fee_hard: float
        ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Run long-short strategy with turnover constraint and cash tracking.
        
        Parameters
        ----------
        factor_scores_df : pd.DataFrame
            Factor scores for each stock at each date
        stock_prices : pd.DataFrame
            Stock prices for each stock at each date
        months : pd.DatetimeIndex
            DatetimeIndex of months for the simulation
        long_weight : float
            Target long exposure (e.g., 1.0 for 100%, 1.3 for 130%)
        short_weight : float
            Target short exposure (e.g., 0.0 for long-only, 0.3 for 130/30)
        max_turnover : float
            Maximum allowed turnover (sum of absolute weight changes)
        initial_cash : float
            Initial cash amount
        cov_matrix : pd.DataFrame
            Annualized covariance matrix of stock returns
        risk_aversion : float
            Risk aversion parameter (higher = more risk-averse)
        single_asset_bound : float
            Maximum allowed weight for any single asset
            
        Returns
        -------
        Tuple[pd.Series, pd.DataFrame, pd.DataFrame]
            - portfolio_value: Series of portfolio values over time
            - weights: DataFrame of portfolio weights at each rebalance date
            - shares: DataFrame of shares held at each rebalance date
        """
        stocks = stock_prices.columns

        # Initialize Series for tracking portfolio values
        cash = pd.Series(index=months, dtype=float)
        portfolio_value = pd.Series(index=months, dtype=float)
        stock_position_value = 0

        # Set initial cash and portfolio value
        cash[months[0]] = initial_cash
        portfolio_value[months[0]] = cash[months[0]]

        # Initialize DataFrames for alphas, weights and shares
        alphas = pd.DataFrame(index=factor_scores_df.index, columns=stocks)
        weights = pd.DataFrame(index=factor_scores_df.index, columns=stocks)
        shares = pd.DataFrame(index=factor_scores_df.index, columns=stocks)

        # Loop through months with factor scores
        for i, date in enumerate(factor_scores_df.index):
            # 1. Calculate target weights from factor scores
            alphas.loc[date] = calculate_alphas(factor_scores_df.loc[date], cov_matrix)

            # 2. Optimize weights with constraints
            if i == 0:
                # First period: no previous weights, allow full turnover
                weights.loc[date] = optimize_weights(
                    alphas.loc[date].values, np.zeros(len(stocks), dtype=object), 1.0,
                    2.01 * (long_weight + short_weight), long_weight, short_weight,
                    cov_matrix.values, risk_aversion, single_asset_bound
                )
            else:
                # Subsequent periods: apply turnover constraint
                prev_date = factor_scores_df.index[i-1]
                # Calculate drifted weights based on price changes and drifted cash weight
                drifted_weights = shares.loc[prev_date] * stock_prices.loc[date] / portfolio_value[date]
                drifted_cash_weight = cash.loc[date] / portfolio_value[date]

                weights.loc[date] = optimize_weights(
                    alphas.loc[date].values, drifted_weights.values, drifted_cash_weight,
                    max_turnover, long_weight, short_weight,
                    cov_matrix.values, risk_aversion, single_asset_bound
                )
            
            # Set optimized cash weight (target cash weight)
            cash.loc[date] = (1.0 - long_weight + short_weight) * portfolio_value.loc[date]
            
            # 3. Calculate shares using current portfolio value
            shares.loc[date] = (weights.loc[date] * portfolio_value[date]) / stock_prices.loc[date]
            long_value = (weights.loc[date].clip(lower=0) * portfolio_value[date]).sum()
            short_value = (weights.loc[date].clip(upper=0).abs() * portfolio_value[date]).sum()

            # 4. Calculate portfolio value at t+1 using shares from t and prices at t+1 and add financing costs
            next_date = months[i+1]

            # Calculate financing costs
            cash_interest = cash.loc[date] * cash_interest_rate / 12
            short_rebate = short_value * cash_interest_rate / 12
            margin_cost = max(0, long_value - portfolio_value[date]) * margin_interest_rate / 12

            # Stock borrow fees
            easy_to_borrow = short_value * (1 - hard_to_borrow_pct) * borrow_fee_base / 12
            hard_to_borrow = short_value * hard_to_borrow_pct * borrow_fee_hard / 12
            borrow_fees = easy_to_borrow + hard_to_borrow

            # Net financing impact
            net_financing = cash_interest + short_rebate - margin_cost - borrow_fees

            # Update cash with financing impact
            cash[next_date] = cash[date] + net_financing

            # Update stock position value with new prices
            stock_position_value = (shares.loc[date] * stock_prices.loc[next_date]).sum()
            
            # Portfolio value is cash plus stock position value
            portfolio_value[next_date] = cash.loc[next_date] + stock_position_value
        
        return portfolio_value, weights, shares
    

    def run_options_overlay(
            self, factor_scores_df: pd.DataFrame, stock_prices: pd.DataFrame, shares: pd.DataFrame,
            volatilities_dict: Dict[str, float], months: pd.DatetimeIndex,otm_percentage: float,
            portfolio_value: pd.Series
        ) -> pd.Series:
        """
        Run options overlay strategy on top of the long-short portfolio.
        
        Parameters
        ----------
        factor_scores_df : pd.DataFrame
            Factor scores for each stock at each date
        stock_prices : pd.DataFrame
            Stock prices for each stock at each date
        shares : pd.DataFrame
            Number of shares held at each rebalance date
        volatilities_dict : Dict[str, float]
            Dictionary mapping stock names to volatility values
        months : pd.DatetimeIndex
            DatetimeIndex of months for the simulation
        otm_percentage : float
            Percentage out-of-the-money for option strikes
        portfolio_value : pd.Series
            Base portfolio value series from long-short strategy
            
        Returns
        -------
        pd.Series
            Portfolio value series with options overlay
        """
        stocks = stock_prices.columns

        # Initialize DataFrames for covered calls and puts and their PnL
        covered_calls = pd.DataFrame(0, index=factor_scores_df.index, columns=stocks)
        covered_puts = pd.DataFrame(0, index=factor_scores_df.index, columns=stocks)
        call_return = pd.DataFrame(0.0, index=factor_scores_df.index, columns=stocks)
        put_return = pd.DataFrame(0.0, index=factor_scores_df.index, columns=stocks)

        # Loop through factor score index dates
        for i, date in enumerate(factor_scores_df.index):
            next_date = months[i+1]

            # Identify long positions with negative factor scores for covered calls
            # Selling calls against stocks we expect to underperform
            call_mask = (shares.loc[date] >= 100) & (factor_scores_df.loc[date] < 0)

            # Identify short positions with positive factor scores for covered puts
            # Selling puts against stocks we expect to outperform
            put_mask = (shares.loc[date] <= -100) & (factor_scores_df.loc[date] > 0)

            current_prices = stock_prices.loc[date]
            final_prices = stock_prices.loc[next_date]

            # Process covered calls
            if call_mask.any():
                covered_calls.loc[date, call_mask] = shares.loc[date, call_mask] // 100
                call_strikes = current_prices[call_mask] * (1 + otm_percentage)

                # Calculate call premiums using Black-Scholes
                call_premiums = np.vectorize(black_scholes, otypes=[float])(
                    current_prices[call_mask],
                    call_strikes,
                    1/12,  # 1 month to expiration
                    0.0,   # Risk-free rate
                    [volatilities_dict[s] for s in call_mask[call_mask].index],
                    'call'
                )

                # Calculate call payoffs at expiration
                final_call_prices = np.maximum(0, final_prices[call_mask] - call_strikes)

                # Calculate net return from calls (premium received - final payoff)
                call_return.loc[next_date, call_mask] = (
                    (call_premiums - final_call_prices) * 100 * covered_calls.loc[date, call_mask]
                )

            # Process covered puts
            if put_mask.any():
                covered_puts.loc[date, put_mask] = abs(shares.loc[date, put_mask]) // 100
                put_strikes = current_prices[put_mask] / (1 + otm_percentage)

                # Calculate put premiums using Black-Scholes
                put_premiums = np.vectorize(black_scholes, otypes=[float])(
                    current_prices[put_mask],
                    put_strikes,
                    1/12,  # 1 month to expiration
                    0.0,   # Risk-free rate
                    [volatilities_dict[s] for s in put_mask[put_mask].index],
                    'put'
                )

                # Calculate put payoffs at expiration
                final_put_prices = np.maximum(0, put_strikes - final_prices[put_mask])

                # Calculate net return from puts (premium received - final payoff)
                put_return.loc[next_date, put_mask] = (
                    (put_premiums - final_put_prices) * 100 * covered_puts.loc[date, put_mask]
                )

        # Calculate cumulative options strategy performance
        call_strategy = call_return.sum(axis=1).cumsum()
        put_strategy = put_return.sum(axis=1).cumsum()
        options_overlay = call_strategy + put_strategy

        # Add options PnL to portfolio value
        portfolio_value_with_options_overlay = portfolio_value + options_overlay

        return portfolio_value_with_options_overlay
    

    def calculate_metrics(
            self, portfolio_value: pd.Series, portfolio_value_with_options_overlay: pd.Series
        ) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for both strategies.
        
        Parameters
        ----------
        portfolio_value_base : pd.Series
            Portfolio value series for the base long-short strategy
        portfolio_value_with_options : pd.Series
            Portfolio value series with options overlay
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of performance metrics for each strategy
        """
        # Define metric functions
        metric_functions = {
            'total_return': calculate_total_return,
            'annualized_return': calculate_annualized_return,
            'sharpe_ratio': calculate_sharpe_ratio,
            'max_drawdown': calculate_max_drawdown,
            'volatility': calculate_volatility
        }

        # Define portfolio values to analyze
        portfolios = {
            'long_short': portfolio_value,
            'options_overlay': portfolio_value_with_options_overlay
        }

        
        # Calculate metrics for each portfolio
        metrics = {}
        for portfolio_name, portfolio_data in portfolios.items():
            metrics[portfolio_name] = {}
            for metric_name, metric_func in metric_functions.items():
                metrics[portfolio_name][metric_name] = metric_func(portfolio_data)
        
        return metrics