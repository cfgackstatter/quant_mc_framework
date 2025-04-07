from typing import Dict, List, Any, Tuple

def get_parameters() -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """
    Define simulation parameters and parameter ranges.
    
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, List[Any]]]
        Base parameters and parameter ranges for Monte Carlo simulations
    """
    # Define market parameters
    market_params = {
        'factor_autocorrelation': 0.1,
        'information_coefficient': 0.05,
        'annual_expected_return': 0.0,
    }

    # Define long-short strategy parameters
    long_short_params = {
        'n_stocks': 100,
        'initial_cash': 10000000,
        'long_weight': 2.5,
        'short_weight': 1.5,
        'max_turnover': 0.5,
        'risk_aversion': 0.5,
        'single_asset_bound': 0.05,
    }

    # Define options overlay parameters
    options_overlay_params = {
        'otm_percentage': 0.0
    }

    # Combine all parameters into base_params
    base_params = {**market_params, **long_short_params, **options_overlay_params}
    
    # Define parameter ranges
    param_ranges = {
        'otm_percentage': [-0.05, 0.0, 0.05]
    }
    
    return base_params, param_ranges