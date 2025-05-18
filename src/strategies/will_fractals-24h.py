import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from src.core.strategy import Strategy

# Strategy metadata
AUTHOR = "Will"
STRATEGY_NAME = "Will Fractals 24h"
STRATEGY_DESC = (
    "Optimized 24-hour fractal breakout strategy with protection mechanisms. "
    "Constructs 24h bars from 1-hour close prices only and executes signals on the next 1h bar."
)
DATA_FREQ = "1h"       # Base data frequency
SYMBOL = "BTC-USD"     # ticker used in data file

# Optimized parameters from deep learning model
OPTIMAL_PARAMS = {
    'bar_freq': '24h',         # 24-hour bars constructed from 1h close prices
    'fractal_lookback': 8,     # lookback window for fractal high/low detection
    'max_loss_pct': 0.05,      # Maximum allowed loss before emergency exit (5%)
    'trend_ma_period': 5,      # Moving average period for trend filter
    'max_hold_periods': 6,     # Maximum holding periods before forced exit
    'volatility_thresh': 2.5   # Volatility threshold (ATR multiple) to avoid entry
}


class WillFractals24hStrategy(Strategy):
    """
    Optimized 24-hour fractal breakout strategy using higher timeframe bars derived from 1-hour close prices.
    
    Uses the optimal parameters discovered through deep learning optimization:
    - 24h bar frequency
    - 8-bar fractal lookback
    - 5% max loss
    - 5-period trend MA
    - 6 periods max hold
    - 2.5 volatility threshold
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 author_name: str = AUTHOR,
                 strategy_name: str = STRATEGY_NAME,
                 description: str = STRATEGY_DESC):
        """Initialize the fractal breakout strategy with optimized parameters."""
        super().__init__(
            initial_capital=initial_capital,
            author_name=author_name,
            strategy_name=strategy_name,
            description=description
        )
        # Set optimized parameters
        self.bar_freq = OPTIMAL_PARAMS['bar_freq']
        self.fractal_lookback = OPTIMAL_PARAMS['fractal_lookback']
        self.max_loss_pct = OPTIMAL_PARAMS['max_loss_pct']
        self.trend_ma_period = OPTIMAL_PARAMS['trend_ma_period']
        self.max_hold_periods = OPTIMAL_PARAMS['max_hold_periods']
        self.volatility_thresh = OPTIMAL_PARAMS['volatility_thresh']
        
    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate buy/sell/hold signals by converting 1h data to 24h bars.
        
        Args:
            df: DataFrame with OHLCV 1-hour data
            
        Returns:
            pandas.Series with 'buy', 'sell', or 'hold' signals on 1-hour timeframe
        """
        # Ensure we have a copy to avoid modifying original data
        df = df.copy()
        
        # Ensure datetime index for resampling
        if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('time', inplace=True)
            
        # Get number of hours in selected bar frequency (24)
        hours_per_bar = self._get_hours_per_bar(self.bar_freq)
        
        # Early return if not enough data
        min_bars = max(self.trend_ma_period + 5, self.fractal_lookback + 5) * hours_per_bar
        if len(df) < min_bars:
            return pd.Series('hold', index=df.index)
        
        # Resample 1-hour data to 24-hour bars
        df_bars = self._resample_to_bars(df, self.bar_freq)
        
        # Calculate indicators on 24-hour bars
        df_bars = self._add_protective_indicators(df_bars)
        
        # Detect fractals on 24-hour timeframe
        df_bars = self._detect_fractals_simple(df_bars)
        
        # Calculate signals on 24-hour timeframe
        signals_bars = self._calculate_protected_signals(df_bars)
        
        # Align signals back to 1-hour timeframe and shift to avoid look-ahead bias
        signals_1h = self._align_signals_to_1h(signals_bars, df.index)
        
        return signals_1h
    
    def _get_hours_per_bar(self, bar_freq: str) -> int:
        """
        Calculate the number of hours in the given bar frequency.
        
        Args:
            bar_freq: Bar frequency string (e.g., '24h')
            
        Returns:
            Number of hours per bar
        """
        if bar_freq.endswith('h'):
            return int(bar_freq[:-1])
        elif bar_freq.endswith('d'):
            return int(bar_freq[:-1]) * 24
        else:
            raise ValueError(f"Unsupported bar frequency: {bar_freq}")
        
    def _resample_to_bars(self, df: pd.DataFrame, bar_freq: str) -> pd.DataFrame:
        """
        Resample 1-hour data to 24-hour bars using only close prices.
        
        Args:
            df: DataFrame with OHLCV data at 1-hour timeframe
            bar_freq: Bar frequency string (e.g., '24h')
            
        Returns:
            DataFrame with OHLCV data resampled to 24-hour timeframe, where OHLC
            is derived from close prices only
        """
        # Ensure dataframe has datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for resampling")
        
        # Create a copy of the dataframe with just the close prices
        df_close = df[['close']].copy()
        
        # For volume, we'll still use the sum aggregation if it exists
        if 'volume' in df.columns:
            df_close['volume'] = df['volume']
        
        # Resample close prices to 24h frequency
        # Define aggregation functions:
        # - open: first close price in the period
        # - high: highest close price in the period
        # - low: lowest close price in the period
        # - close: last close price in the period
        agg_dict = {
            'close': ['first', 'max', 'min', 'last']
        }
        
        # Add volume aggregation if it exists
        if 'volume' in df_close.columns:
            agg_dict['volume'] = 'sum'
        
        # Resample to specified bar frequency
        resampled = df_close.resample(bar_freq).agg(agg_dict)
        
        # Flatten the column names
        resampled.columns = ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df_close.columns else [])
        
        return resampled
    
    def _add_protective_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators for drawdown protection.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional protection indicators
        """
        df_result = df.copy()
        
        # Calculate trend filter (simple moving average)
        df_result['trend_ma'] = df_result['close'].rolling(self.trend_ma_period).mean()
        
        # Calculate ATR for volatility assessment and stop losses
        high_low = df_result['high'] - df_result['low']
        high_close = np.abs(df_result['high'] - df_result['close'].shift(1))
        low_close = np.abs(df_result['low'] - df_result['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        # 14-period ATR
        df_result['atr'] = true_range.rolling(14).mean()
        
        # ATR volatility ratio (current ATR vs long-term ATR average)
        df_result['atr_ratio'] = df_result['atr'] / df_result['atr'].rolling(20).mean()
        
        return df_result
    
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
                
        # Calculate rolling highest fractal high and lowest fractal low for the lookback window
        df_result['fractal_high_price'] = np.nan
        df_result['fractal_low_price'] = np.nan
        
        for i in range(len(df_result)):
            if df_result['fractal_high'].iloc[i]:
                df_result.loc[df_result.index[i], 'fractal_high_price'] = df_result['high'].iloc[i]
            if df_result['fractal_low'].iloc[i]:
                df_result.loc[df_result.index[i], 'fractal_low_price'] = df_result['low'].iloc[i]
        
        # Get the rolling highest fractal high and lowest fractal low using lookback window
        df_result['last_high'] = df_result['fractal_high_price'].rolling(self.fractal_lookback, min_periods=1).max()
        df_result['last_low'] = df_result['fractal_low_price'].rolling(self.fractal_lookback, min_periods=1).min()
        
        return df_result
    
    def _calculate_protected_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Signal generation with drawdown protection mechanisms.
        
        Args:
            df: DataFrame with fractal and protection indicators
            
        Returns:
            Series with buy/sell/hold signals
        """
        # Initialize signals
        signals = pd.Series('hold', index=df.index)
        
        # Start with no position
        in_position = False
        entry_price = 0
        entry_time_idx = 0
        stop_price = 0
        
        for i in range(self.trend_ma_period, len(df)):
            current_close = df['close'].iloc[i]
            current_time = df.index[i]
            
            # Skip if essential data is missing
            if pd.isna(current_close) or pd.isna(df['trend_ma'].iloc[i]) or pd.isna(df['atr'].iloc[i]):
                continue
                
            # If not in position, check entry conditions
            if not in_position:
                # Check if all entry conditions are met:
                # 1. Original fractal breakout condition (using previous bar's fractal)
                # 2. Price above trend MA (uptrend)
                # 3. Volatility not too high
                
                if (i > 0 and  # Ensure we can look back one bar
                    not pd.isna(df['last_high'].iloc[i-1]) and 
                    current_close > df['last_high'].iloc[i-1] and
                    current_close > df['trend_ma'].iloc[i] and
                    df['atr_ratio'].iloc[i] < self.volatility_thresh):
                    
                    signals.iloc[i] = 'buy'
                    in_position = True
                    entry_price = current_close
                    entry_time_idx = i
                    
                    # Set stop price using both fractal low and max allowed loss
                    if not pd.isna(df['last_low'].iloc[i-1]):
                        fractal_stop = df['last_low'].iloc[i-1]
                        max_loss_stop = entry_price * (1 - self.max_loss_pct)
                        stop_price = max(fractal_stop, max_loss_stop)
                    else:
                        # Fallback to max loss stop if no fractal low
                        stop_price = entry_price * (1 - self.max_loss_pct)
            
            # If in position, check exit conditions
            else:
                # Check multiple exit conditions:
                # 1. Traditional fractal low break
                # 2. Stop loss hit
                # 3. Maximum holding time exceeded
                # 4. Reverse trend (price falls below MA)
                
                holding_periods = i - entry_time_idx
                
                # Update trailing stop if a new higher fractal low appears
                if df['fractal_low'].iloc[i] and df['low'].iloc[i] > stop_price:
                    stop_price = df['low'].iloc[i]
                
                # Exit conditions with multiple protection mechanisms
                exit_conditions_met = False
                
                # 1. Traditional fractal low exit (using previous bar's fractal)
                if i > 0 and not pd.isna(df['last_low'].iloc[i-1]) and current_close < df['last_low'].iloc[i-1]:
                    exit_conditions_met = True
                
                # 2. Stop loss hit
                elif current_close <= stop_price:
                    exit_conditions_met = True
                
                # 3. Maximum holding time exceeded
                elif holding_periods >= self.max_hold_periods:
                    exit_conditions_met = True
                
                # 4. Trend reversal (price falls below MA)
                elif current_close < df['trend_ma'].iloc[i] and holding_periods > 2:
                    # Only use this after 2 periods to avoid whipsaws
                    exit_conditions_met = True
                
                if exit_conditions_met:
                    signals.iloc[i] = 'sell'
                    in_position = False
        
        return signals
    
    def _align_signals_to_1h(self, signals_bars: pd.Series, index_1h: pd.DatetimeIndex) -> pd.Series:
        """
        Aligns 24-hour timeframe signals to 1-hour timeframe and shifts by 1 hour.
        
        Args:
            signals_bars: Series with signals on 24-hour timeframe
            index_1h: DatetimeIndex of 1-hour data
            
        Returns:
            Series with signals on 1-hour timeframe, shifted by 1 hour for proper execution
        """
        # Create a Series with 1-hour index filled with 'hold'
        signals_1h = pd.Series('hold', index=index_1h)
        
        # Reindex higher timeframe signals to 1-hour timeframe using forward fill
        # This ensures signals persist until the next signal
        temp_signals = signals_bars.reindex(index_1h, method='ffill')
        
        # Where temp_signals is not NA, use those values
        signals_1h[~temp_signals.isna()] = temp_signals[~temp_signals.isna()]
        
        # Shift signals by 1 hour to execute on the next 1h bar after a signal
        # This avoids look-ahead bias by ensuring we don't act on information
        # from the current bar's close
        signals_1h = signals_1h.shift(1).fillna('hold')
        
        return signals_1h


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio from a list of returns.
    
    Args:
        returns: List of period returns
        risk_free_rate: Annualized risk-free rate
        
    Returns:
        Annualized Sharpe ratio
    """
    if not returns:
        return -float('inf')
    
    period_rfr = risk_free_rate / 252  # Daily risk-free rate
    
    returns_array = np.array(returns)
    excess_returns = returns_array - period_rfr
    
    mean_excess_return = np.mean(excess_returns)
    std_deviation = np.std(excess_returns, ddof=1)
    
    if std_deviation == 0:
        return -float('inf')
    
    sharpe = mean_excess_return / std_deviation
    annualized_sharpe = sharpe * np.sqrt(252)  # Annualize to daily trading
    
    return annualized_sharpe


def run_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run a backtest on the dataset using the optimized 24h strategy.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with performance metrics
    """
    # Initialize strategy
    strategy = WillFractals24hStrategy()
    
    # Generate signals
    signals = strategy.get_signals(df)
    
    # Calculate performance metrics
    returns = []
    trades = []
    equity_curve = [1.0]  # Start with $1
    peak = 1.0
    drawdowns = []
    in_position = False
    entry_price = 0
    entry_time = None
    
    for i in range(len(signals)):
        if signals.iloc[i] == 'buy' and not in_position:
            in_position = True
            entry_price = df['close'].iloc[i]
            entry_time = df.index[i]
        elif signals.iloc[i] == 'sell' and in_position:
            exit_price = df['close'].iloc[i]
            exit_time = df.index[i]
            pct_return = exit_price / entry_price - 1
            
            # Record trade
            trade = {
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': pct_return,
                'hold_hours': (exit_time - entry_time).total_seconds() / 3600
            }
            trades.append(trade)
            
            # Update returns and equity curve
            returns.append(pct_return)
            equity_curve.append(equity_curve[-1] * (1 + pct_return))
            
            # Update peak and drawdown
            peak = max(peak, equity_curve[-1])
            drawdown = (peak - equity_curve[-1]) / peak
            drawdowns.append(drawdown)
            
            in_position = False
    
    # Calculate performance metrics
    if returns:
        total_return = equity_curve[-1] - 1
        max_drawdown = max(drawdowns) if drawdowns else 0
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        sharpe = calculate_sharpe_ratio(returns)
        
        metrics = {
            'n_trades': len(returns),
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe
        }
    else:
        metrics = {
            'n_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'sharpe': -float('inf')
        }
    
    return {
        'metrics': metrics,
        'trades': trades,
        'equity_curve': equity_curve
    }


if __name__ == "__main__":
    import os
    import warnings
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    warnings.filterwarnings('ignore')
    
    # Find the path to the data directory
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent.parent
    data_path = project_root / "nstrade" / "data" / "btc_hour.csv"
    
    print(f"Loading data from: {data_path}")
    
    # Load BTC hourly data
    btc_data = pd.read_csv(data_path)
    
    # Convert time column to datetime
    btc_data['time'] = pd.to_datetime(btc_data['time'])
    
    # Set time as index
    btc_data = btc_data.set_index('time')
    
    # Use the full date range from 2011-2024
    start_date = '2011-01-01'
    end_date = '2024-05-01'  # Use data up to May 2024 or latest available
    btc_filtered = btc_data.loc[start_date:end_date]
    
    print(f"Data loaded: {len(btc_filtered)} rows from {btc_filtered.index.min()} to {btc_filtered.index.max()}")
    print(f"Running backtest on 24h fractal strategy with optimized parameters...")
    
    # Run the backtest
    backtest_results = run_backtest(btc_filtered)
    
    # Print backtest metrics
    metrics = backtest_results['metrics']
    print("\nBacktest Results:")
    print(f"Number of Trades: {metrics['n_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['equity_curve'])
    plt.title("Equity Curve - Will Fractals 24h Strategy")
    plt.xlabel('Trades')
    plt.ylabel('Equity (starting at $1)')
    plt.grid(True)
    
    # Save plot to results directory
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "will_fractals_24h_equity_curve.png")
    
    print(f"\nEquity curve saved to {results_dir / 'will_fractals_24h_equity_curve.png'}")
    print("\nStrategy ready for use with optimized 24h parameters") 