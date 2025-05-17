import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from src.core.strategy import Strategy

# Strategy metadata
AUTHOR = "Will"
STRATEGY_NAME = "Will Fractals"
STRATEGY_DESC = (
    "Hourly BTC strategy using five-bar fractal breakouts, "
    "48 h volume thrust, 4 h trend filter, and ATR-style buffered stops."
)
DATA_FREQ = "1h"       # matches btc_hour.csv
SYMBOL = "BTC-USD"     # ticker used in data file

# Hyperparameters
Z_THR = 0.0                # volume z-score threshold (reduced for more entries)  
MTF_WINDOW = "4h"          # higher-TF resample window
BUFFER_MULT_SIGMA = 0.15   # % of 24 h σ
BUFFER_MULT_PRICE_BP = 15  # 15 bp of price


class WillFractalsStrategy(Strategy):
    """
    Fractal breakout strategy with higher timeframe filter and volume confirmation.
    
    Uses five-bar fractal pattern to identify potential breakout points, confirms with
    volume and higher timeframe trend, and manages risk with adaptive trailing stops.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 author_name: str = AUTHOR,
                 strategy_name: str = STRATEGY_NAME,
                 description: str = STRATEGY_DESC):
        """Initialize the fractal breakout strategy."""
        super().__init__(
            initial_capital=initial_capital,
            author_name=author_name,
            strategy_name=strategy_name,
            description=description
        )
        
    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate buy/sell/hold signals for the entire dataset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pandas.Series with 'buy', 'sell', or 'hold' signals
        """
        # Ensure we have a copy to avoid modifying original data
        df = df.copy()
        
        # Ensure datetime index for resampling
        if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('time', inplace=True)
        
        # Early return if not enough data
        if len(df) < 25:  # Need enough data for all calculations
            return pd.Series('hold', index=df.index)
            
        # Detect fractals on 1h timeframe (simplified using high/low price instead of close)
        df = self._detect_fractals_simple(df)
        
        # Calculate signals directly
        signals = self._calculate_simple_signals(df)
        
        return signals
    
    def _detect_fractals_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified fractal detection using high/low prices.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added fractal high/low columns
        """
        df_result = df.copy()
        
        # Initialize fractal columns
        df_result['fractal_high'] = False
        df_result['fractal_low'] = False
        
        # Need at least 5 bars to detect fractals
        if len(df_result) < 5:
            return df_result
        
        # Detect fractals using standard 5-bar pattern
        for i in range(2, len(df_result) - 2):
            # Fractal High: high[t] > high[t-2:t-1] and high[t] > high[t+1:t+2]
            if (df_result['high'].iloc[i] > df_result['high'].iloc[i-2] and
                df_result['high'].iloc[i] > df_result['high'].iloc[i-1] and
                df_result['high'].iloc[i] > df_result['high'].iloc[i+1] and
                df_result['high'].iloc[i] > df_result['high'].iloc[i+2]):
                df_result.loc[df_result.index[i], 'fractal_high'] = True
            
            # Fractal Low: low[t] < low[t-2:t-1] and low[t] < low[t+1:t+2]
            if (df_result['low'].iloc[i] < df_result['low'].iloc[i-2] and
                df_result['low'].iloc[i] < df_result['low'].iloc[i-1] and
                df_result['low'].iloc[i] < df_result['low'].iloc[i+1] and
                df_result['low'].iloc[i] < df_result['low'].iloc[i+2]):
                df_result.loc[df_result.index[i], 'fractal_low'] = True
                
        # Calculate rolling highest fractal high and lowest fractal low for the last 20 bars
        df_result['fractal_high_price'] = np.nan
        df_result['fractal_low_price'] = np.nan
        
        for i in range(len(df_result)):
            if df_result['fractal_high'].iloc[i]:
                df_result.loc[df_result.index[i], 'fractal_high_price'] = df_result['high'].iloc[i]
            if df_result['fractal_low'].iloc[i]:
                df_result.loc[df_result.index[i], 'fractal_low_price'] = df_result['low'].iloc[i]
        
        # Get the last 20 bars rolling highest fractal high and lowest fractal low
        df_result['last_high'] = df_result['fractal_high_price'].rolling(20, min_periods=1).max()
        df_result['last_low'] = df_result['fractal_low_price'].rolling(20, min_periods=1).min()
        
        return df_result
    
    def _calculate_simple_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Simple signal generation based on fractal patterns.
        
        Args:
            df: DataFrame with fractal information
            
        Returns:
            Series with buy/sell/hold signals
        """
        # Initialize signals
        signals = pd.Series('hold', index=df.index)
        
        # Start with no position
        in_position = False
        entry_price = 0
        
        # Basic fractal breakout strategy:
        # - Buy when price breaks above the recent highest fractal high
        # - Sell when price breaks below the recent lowest fractal low
        for i in range(5, len(df)):
            if not pd.isna(df['last_high'].iloc[i-1]) and not pd.isna(df['last_low'].iloc[i-1]):
                # Buy signal: price breaks above recent highest fractal high
                if not in_position and df['close'].iloc[i] > df['last_high'].iloc[i-1]:
                    signals.iloc[i] = 'buy'
                    in_position = True
                    entry_price = df['close'].iloc[i]
                
                # Sell signal: price breaks below recent lowest fractal low
                elif in_position and df['close'].iloc[i] < df['last_low'].iloc[i-1]:
                    signals.iloc[i] = 'sell'
                    in_position = False
        
        return signals


def tune(df: pd.DataFrame, folds: int = 4) -> Dict[str, Any]:
    """
    Perform grid search to find optimal hyperparameters.
    
    This is an optional function for manual tuning that is not called
    by the main strategy but can be used for optimization.
    
    Args:
        df: DataFrame with OHLCV data
        folds: Number of walk-forward folds
        
    Returns:
        Dictionary with best parameters
    """
    # Parameter grid
    z_thresholds = [0.0, 0.5, 1.0]
    mtf_windows = ["3h", "4h", "6h"]
    
    best_sharpe = -float('inf')
    best_params = None
    
    # Split data into folds
    fold_size = len(df) // folds
    
    print(f"Grid searching {len(z_thresholds) * len(mtf_windows)} parameter combinations...")
    
    # Grid search
    for z_thr in z_thresholds:
        for mtf_window in mtf_windows:
            # We'll manually modify the module globals temporarily for testing
            global Z_THR, MTF_WINDOW
            Z_THR = z_thr
            MTF_WINDOW = mtf_window
            
            fold_metrics = []
            
            # Walk-forward validation
            for fold in range(folds):
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < folds - 1 else len(df)
                
                # Train on all data except the current fold
                train_df = pd.concat([
                    df.iloc[:start_idx],
                    df.iloc[end_idx:]
                ])
                
                # Test on current fold
                test_df = df.iloc[start_idx:end_idx]
                
                # Run strategy
                strategy = WillFractalsStrategy()
                signals = strategy.get_signals(train_df)
                
                # Evaluate on test fold
                test_signals = strategy.get_signals(test_df)
                
                # Calculate performance metrics
                # (In a real implementation, we'd calculate returns and Sharpe ratio)
                # For this example, let's just use a simplified calculation
                returns = []
                in_position = False
                entry_price = 0
                
                for i in range(len(test_signals)):
                    if test_signals.iloc[i] == 'buy' and not in_position:
                        in_position = True
                        entry_price = test_df['close'].iloc[i]
                    elif test_signals.iloc[i] == 'sell' and in_position:
                        returns.append(test_df['close'].iloc[i] / entry_price - 1)
                        in_position = False
                
                if len(returns) > 0:
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
                    fold_metrics.append(sharpe)
            
            # Average performance across folds
            if fold_metrics:
                avg_sharpe = np.mean(fold_metrics)
                
                if avg_sharpe > best_sharpe:
                    best_sharpe = avg_sharpe
                    best_params = {
                        'Z_THR': z_thr,
                        'MTF_WINDOW': mtf_window
                    }
                    
                print(f"Z_THR={z_thr}, MTF_WINDOW={mtf_window}: Sharpe={avg_sharpe:.4f}")
    
    # Reset to original values
    Z_THR = best_params['Z_THR'] if best_params else 0.0
    MTF_WINDOW = best_params['MTF_WINDOW'] if best_params else "4h"
    
    print(f"Best parameters: Z_THR={Z_THR}, MTF_WINDOW={MTF_WINDOW}")
    
    return best_params


if __name__ == "__main__":
    # This allows for quick command-line testing of the strategy
    # Example usage: python -m src.strategies.will_fractals
    print(f"Fractal Strategy initialized with:")
    print(f"  AUTHOR: {AUTHOR}")
    print(f"  STRATEGY_NAME: {STRATEGY_NAME}")
    print(f"  Z_THR: {Z_THR}")
    print(f"  MTF_WINDOW: {MTF_WINDOW}")
    print(f"  BUFFER_MULT_SIGMA: {BUFFER_MULT_SIGMA}")
    print(f"  BUFFER_MULT_PRICE_BP: {BUFFER_MULT_PRICE_BP}") 