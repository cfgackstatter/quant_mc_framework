import numpy as np
import pandas as pd
from ..data.generators import (
    generate_factor_scores, generate_lognormal_returns,
    generate_stock_prices, black_scholes
)
from ..simulation.strategies import (
    calculate_factor_weights, optimize_weights
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

        # Extract parameters
        n_stocks = self.params.get('n_stocks')
        target_ic = self.params.get('target_ic')
        target_leverage = self.params.get('target_leverage')
        max_turnover = self.params.get('max_turnover')
        initial_cash = self.params.get('initial_cash')
        autocorrelation = self.params.get('autocorrelation')
        otm_percentage = self.params.get('otm_percentage')

        # Generate dates
        start_date = pd.Timestamp('2023-12-01')
        end_date = pd.Timestamp('2025-01-01')
        months = pd.date_range(start=start_date, end=end_date, freq='ME')

        # Generate stock names
        stocks = [f'Stock_{i}' for i in range(1, n_stocks+1)]

        # Generate factor scores with autocorrelation
        factor_scores_dict = generate_factor_scores(
            n_stocks, len(months)-1, autocorrelation, factor_seed
        )
        factor_scores_df = pd.DataFrame(factor_scores_dict, index=months[:-1])

        # Generate volatilities
        volatilities = np.random.uniform(0.15, 0.75, n_stocks)
        volatilities_dict = dict(zip(stocks, volatilities))

        # Generate returns and stock prices
        returns_data = generate_lognormal_returns(
            factor_scores_dict, target_ic, volatilities_dict, returns_seed
        )
        initial_prices = np.random.uniform(10, 150, n_stocks)
        initial_prices_dict = dict(zip(stocks, initial_prices))
        stock_prices = generate_stock_prices(returns_data, initial_prices_dict, months)

        # Run long-short strategy
        portfolio_value, weights, shares = self.run_long_short_strategy(
            factor_scores_df, stock_prices, months,
            target_leverage, max_turnover, initial_cash
        )

        # Run options overlay
        portfolio_with_options = self.run_options_overlay(
            factor_scores_df, stock_prices, shares,
            volatilities_dict, months, otm_percentage, portfolio_value
        )

        # Calculate performance metrics
        metrics = self.calculate_metrics(portfolio_value, portfolio_with_options)

        return {
            'portfolio_value': portfolio_value,
            'portfolio_with_options': portfolio_with_options,
            'metrics': metrics,
            'weights': weights,
            'shares': shares,
            'stock_prices': stock_prices,
        }
    

    def run_long_short_strategy(self, factor_scores_df, stock_prices, months,
                                target_leverage, max_turnover, initial_cash):
        """Run long-short strategy with turnover constraint"""
        stocks = stock_prices.columns

        # Initialize Series for tracking portfolio values
        portfolio_value = pd.Series(index=months, dtype=float)
        portfolio_value[months[0]] = initial_cash

        # Initialize DataFrames for weights and shares
        target_weights = pd.DataFrame(index=factor_scores_df.index, columns=stocks)
        weights = pd.DataFrame(index=factor_scores_df.index, columns=stocks)
        shares = pd.DataFrame(index=factor_scores_df.index, columns=stocks)

        # Loop through months with factor scores
        for i, date in enumerate(factor_scores_df.index):
            # 1. Calculate target weights from factor scores
            target_weights.loc[date] = calculate_factor_weights(
                factor_scores_df.loc[date], target_leverage
            )

            # 2. Apply turnover constraint
            if i == 0:
                weights.loc[date] = target_weights.loc[date]
            else:
                prev_date = factor_scores_df.index[i-1]
                drifted_weights = shares.loc[prev_date] * stock_prices.loc[date] / portfolio_value[date]

                # Optimize weights with constraints
                weights.loc[date] = optimize_weights(
                    target_weights.loc[date].values,
                    drifted_weights.values,
                    max_turnover,
                    target_leverage
                )
            
            # 3. Calculate shares using current portfolio value
            shares.loc[date] = (weights.loc[date] * portfolio_value[date]) / stock_prices.loc[date]

            # 4. Calculate portfolio value at t+1 using shares from t and prices at t+1
            next_date = months[i+1]
            portfolio_value[next_date] = portfolio_value[date] + (
                shares.loc[date] * stock_prices.loc[next_date]
            ).sum()

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

        # Add options PnL to portfolio value
        portfolio_with_options = portfolio_value + call_strategy + put_strategy

        return portfolio_with_options
    

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