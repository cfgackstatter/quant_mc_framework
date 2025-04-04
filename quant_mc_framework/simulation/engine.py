import numpy as np
import pandas as pd
from ..data.generators import (
    generate_factor_scores, generate_lognormal_returns,
    generate_stock_prices, black_scholes
)
from ..simulation.strategies import (
    calculate_alphas, optimize_weights
)


class LongShortSimulation:
    def __init__(self, params):
        """Initialize simulation with parameters"""
        self.params = params
        self.results = {}


    def run(self, seed=None):
        """Run a single simulation"""
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
        start_date = pd.Timestamp('2023-12-01')
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

        # Generate returns, stock prices and covariance matrix
        returns_data = generate_lognormal_returns(
            factor_scores_dict, information_coefficient, annual_expected_return, volatilities_dict, returns_seed
        )
        initial_prices = np.random.uniform(10, 150, n_stocks)
        initial_prices_dict = dict(zip(stocks, initial_prices))
        
        stock_prices = generate_stock_prices(returns_data, initial_prices_dict, months)
        
        cov_matrix = stock_prices.pct_change().cov() * 12

        # Run long-short strategy
        portfolio_value, weights, shares = self.run_long_short_strategy(
            factor_scores_df, stock_prices, months,
            long_weight, short_weight, max_turnover, initial_cash,
            cov_matrix, risk_aversion, single_asset_bound
        )

        # Run options overlay
        portfolio_value_with_options_overlay = self.run_options_overlay(
            factor_scores_df, stock_prices, shares,
            volatilities_dict, months, otm_percentage, portfolio_value
        )

        # Calculate performance metrics
        metrics = self.calculate_metrics(portfolio_value, portfolio_value_with_options_overlay)

        return {
            'portfolio_value': portfolio_value,
            'portfolio_value_with_options_overlay': portfolio_value_with_options_overlay,
            'metrics': metrics,
        }
    

    def run_long_short_strategy(self, factor_scores_df, stock_prices, months,
                                long_weight, short_weight, max_turnover, initial_cash,
                                cov_matrix, risk_aversion, single_asset_bound):
        """Run long-short strategy with turnover constraint"""
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
                weights.loc[date] = optimize_weights(
                    alphas.loc[date].values, np.zeros(len(stocks), dtype=object),
                    2 * (long_weight + short_weight), long_weight, short_weight,
                    cov_matrix.values, risk_aversion, single_asset_bound
                )
            else:
                prev_date = factor_scores_df.index[i-1]
                drifted_weights = shares.loc[prev_date] * stock_prices.loc[date] / portfolio_value[date]

                weights.loc[date] = optimize_weights(
                    alphas.loc[date].values, drifted_weights.values,
                    max_turnover, long_weight, short_weight,
                    cov_matrix.values, risk_aversion, single_asset_bound
                )
            
            # 3. Calculate shares using current portfolio value
            shares.loc[date] = (weights.loc[date] * portfolio_value[date]) / stock_prices.loc[date]

            # 4. Update positions
            cash.loc[date] = cash.loc[date] - ((shares.loc[date] * stock_prices.loc[date]).sum() - stock_position_value)

            # 5. Calculate portfolio value at t+1 using shares from t and prices at t+1
            next_date = months[i+1]
            cash[next_date] = cash[date]
            stock_position_value = (shares.loc[date] * stock_prices.loc[next_date]).sum()
            portfolio_value[next_date] = cash.loc[next_date] + stock_position_value

        return portfolio_value, weights, shares
    

    def run_options_overlay(self, factor_scores_df, stock_prices, shares,
                            volatilities_dict, months, otm_percentage, portfolio_value):
        """Run options overlay strategy"""
        stocks = stock_prices.columns

        # Initialize DataFrmaes for covered cals and puts and their PnL
        covered_calls = pd.DataFrame(0, index=factor_scores_df.index, columns=stocks)
        covered_puts = pd.DataFrame(0, index=factor_scores_df.index, columns=stocks)
        call_return = pd.DataFrame(0.0, index=factor_scores_df.index, columns=stocks)
        put_return = pd.DataFrame(0.0, index=factor_scores_df.index, columns=stocks)

        # Loop through factor score index dates
        for i, date in enumerate(factor_scores_df.index):
            next_date = months[i+1]

            # Identify long positions with negative factor scores for covered calls
            call_mask = (shares.loc[date] >= 100) & (factor_scores_df.loc[date] < 0)

            # Identify short positions with positive factor scores for covered puts
            put_mask = (shares.loc[date] <= -100) & (factor_scores_df.loc[date] > 0)

            current_prices = stock_prices.loc[date]
            final_prices = stock_prices.loc[next_date]

            # Process covered calls
            covered_calls.loc[date, call_mask] = shares.loc[date, call_mask] // 100
            call_strikes = current_prices[call_mask] * (1 + otm_percentage)
            call_premiums = np.vectorize(black_scholes, otypes=[float])(
                    current_prices[call_mask], call_strikes, 1/12, 0.0, 
                    [volatilities_dict[s] for s in call_mask[call_mask].index], 'call')
            final_call_prices = np.maximum(0, final_prices[call_mask] - call_strikes)
            call_return.loc[next_date, call_mask] = (call_premiums - final_call_prices) * 100 * covered_calls.loc[date, call_mask]

            # Process covered puts
            covered_puts.loc[date, put_mask] = abs(shares.loc[date, put_mask]) // 100
            put_strikes = current_prices[put_mask] / (1 + otm_percentage)
            put_premiums = np.vectorize(black_scholes, otypes=[float])(
                    current_prices[put_mask], put_strikes, 1/12, 0.0, 
                    [volatilities_dict[s] for s in put_mask[put_mask].index], 'put')
            final_put_prices = np.maximum(0, put_strikes - final_prices[put_mask])
            put_return.loc[next_date, put_mask] = (put_premiums - final_put_prices) * 100 * covered_puts.loc[date, put_mask]

        # Calculate cumulative options strategy performance
        call_strategy = call_return.sum(axis=1).cumsum()
        put_strategy = put_return.sum(axis=1).cumsum()
        options_overlay = call_strategy + put_strategy

        # Add options PnL to portfolio value
        portfolio_value_with_options_overlay = portfolio_value + options_overlay

        return portfolio_value_with_options_overlay
    

    def calculate_metrics(self, portfolio_value, portfolio_with_options):
        """Calculate performance metrics for the strategies"""
        # Calculate returns
        portfolio_returns = portfolio_value.pct_change().dropna()
        options_returns = portfolio_with_options.pct_change().dropna()
        
        # Calculate metrics
        metrics = {
            'base_strategy': {
                'total_return': (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1,
                'annualized_return': ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** 
                                     (12 / len(portfolio_returns))) - 1,
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(12),
                'max_drawdown': (portfolio_value / portfolio_value.cummax() - 1).min(),
                'volatility': portfolio_returns.std() * np.sqrt(12)
            },
            'options_overlay': {
                'total_return': (portfolio_with_options.iloc[-1] / portfolio_with_options.iloc[0]) - 1,
                'annualized_return': ((portfolio_with_options.iloc[-1] / portfolio_with_options.iloc[0]) ** 
                                     (12 / len(options_returns))) - 1,
                'sharpe_ratio': options_returns.mean() / options_returns.std() * np.sqrt(12),
                'max_drawdown': (portfolio_with_options / portfolio_with_options.cummax() - 1).min(),
                'volatility': options_returns.std() * np.sqrt(12)
            }
        }
        
        return metrics