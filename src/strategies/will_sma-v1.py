"""
SMA Crossover 1 strategy for Bitcoin.
Goes long when the fast SMA crosses above the slow SMA, and exits when it crosses below.
"""

from src.core.strategy import Strategy
import pandas as pd

class SMACrossoverStrategy(Strategy):
    def __init__(self, initial_capital=10000, fast=18, slow=80):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Will",
            strategy_name="SMA Crossover 1",
            description="Goes long when the 18-period SMA crosses above the 80-period SMA, and exits when it crosses below."
        )
        self.prices = []
        self.fast = fast  # The fast SMA period (18 by default)
        self.slow = slow  # The slow SMA period (80 by default)
        self.last_signal = 'hold'

    def process_bar(self, bar):
        """
        Process a single price bar and determine the trading signal.
        This implementation calculates SMAs on each update for real-time trading.
        """
        self.current_bar = bar
        self.prices.append(bar['close'])
        if len(self.prices) < self.slow:
            self.last_signal = 'hold'
            return

        fast_ma = pd.Series(self.prices).rolling(self.fast).mean().iloc[-1]
        slow_ma = pd.Series(self.prices).rolling(self.slow).mean().iloc[-1]

        if fast_ma > slow_ma and self.position == 0:
            self.last_signal = 'buy'
        elif fast_ma < slow_ma and self.position == 1:
            self.last_signal = 'sell'
        else:
            self.last_signal = 'hold'

    def get_signal(self):
        return self.last_signal

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized implementation for backtesting.
        This is more efficient than calling process_bar() and get_signal() for each bar.
        The shift(1) operation avoids look-ahead bias by ensuring signals are based on prior data.
        """
        fast_ma = df['close'].rolling(self.fast).mean()
        slow_ma = df['close'].rolling(self.slow).mean()
        signals = pd.Series('hold', index=df.index)
        signals[fast_ma > slow_ma] = 'buy'
        signals[fast_ma < slow_ma] = 'sell'
        signals.iloc[:self.slow-1] = 'hold'
        signals = signals.shift(1).fillna('hold')
        return signals