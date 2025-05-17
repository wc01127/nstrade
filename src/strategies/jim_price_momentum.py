"""
Jim's Price Momentum Strategy.
This strategy buys when the current price is higher than the previous close,
sells when it's lower, and holds otherwise.
"""

from src.core.strategy import Strategy
import pandas as pd

class PriceMomentumStrategy(Strategy):
    def __init__(self, initial_capital=10000):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Jim",
            strategy_name="Price Momentum",
            description="Buys when price increases from previous close, sells when it decreases, and holds otherwise."
        )
        
        # Initialize variables to track price history
        self.previous_close = None
        self.last_signal = 'hold'

    def process_bar(self, bar):
        """
        Process each bar of data and generate trading signals based on price momentum.
        
        The strategy logic:
        1. If this is the first bar, just store the price and hold
        2. If price has increased from previous close, generate buy signal (if not already in position)
        3. If price has decreased from previous close, generate sell signal (if already in position)
        4. Otherwise, hold current position
        
        Args:
            bar: Dictionary containing 'time', 'close', and 'volume' data
        """
        self.current_bar = bar
        
        # If this is the first bar, we don't have a previous close to compare with
        if self.previous_close is None:
            self.previous_close = bar['close']
            self.last_signal = 'hold'
            return
        
        # Generate signal based on price movement
        if bar['close'] > self.previous_close and self.position == 0:
            # Price increased and we're not in a position, so buy
            self.last_signal = 'buy'
        elif bar['close'] < self.previous_close and self.position == 1:
            # Price decreased and we're in a position, so sell
            self.last_signal = 'sell'
        else:
            # Either price didn't change enough or we already have the appropriate position
            self.last_signal = 'hold'
            
        # Update previous close for next bar
        self.previous_close = bar['close']

    def get_signal(self):
        """
        Return the current trading signal.
        
        Returns:
            str: 'buy', 'sell', or 'hold'
        """
        return self.last_signal
        
    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized implementation that processes the entire dataset at once.
        
        This is more efficient for backtesting than processing each bar individually.
        
        Args:
            df: DataFrame with at least 'close' column
            
        Returns:
            pd.Series: Series of signals ('buy', 'sell', 'hold') aligned with df index
        """
        # Initialize signals as 'hold'
        signals = pd.Series('hold', index=df.index)
        
        # Calculate price changes
        price_change = df['close'].diff()
        
        # Generate buy signals where price increased
        buy_signals = (price_change > 0)
        signals[buy_signals] = 'buy'
        
        # Generate sell signals where price decreased
        sell_signals = (price_change < 0)
        signals[sell_signals] = 'sell'
        
        # First bar should be 'hold' as we don't have a previous price
        signals.iloc[0] = 'hold'
        
        # Shift signals by 1 to avoid lookahead bias
        # (ensure we're only using information available at the time)
        signals = signals.shift(1).fillna('hold')
        
        return signals
