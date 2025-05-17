"""
SMA Crossover 1 strategy for Bitcoin.
Goes long when the fast SMA crosses above the slow SMA, and exits when it crosses below.
"""

from src.core.strategy import Strategy
import pandas as pd
import logging

class VolumeSMAConfirmationStrategy(Strategy):
    def __init__(self, initial_capital=10000, window=240, res_window=5, sup_window=5):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Yuan",
            strategy_name="Volume SMA Confirmation Strategy",
            description="Goes long when the price breaks above the resistance with high volume, and exits when it breaks below the support with high volume."
        )
        self.prices = []
        self.volumes = []
        self.window = window
        self.res_window = res_window
        self.sup_window = sup_window
        self.last_signal = 'hold'
    
    def get_signals(self, df):
        """
        Vectorized version of signal generation.
        Returns a pandas Series of signals: 'buy', 'sell', or 'hold' for each row.
        """
        resistance = df['close'].shift(1).rolling(self.res_window).max()
        support = df['close'].shift(1).rolling(self.sup_window).min()
        vol_sma = df['volumefrom'].shift(1).rolling(self.window).mean()

        buy = (df['close'].shift(1) <= resistance) & \
              (df['close'] > resistance) & \
              (df['volumefrom']  > 2.5 * vol_sma)

        sell = (df['close'].shift(1) >= support) & \
               (df['close'] < support) & \
               (df['volumefrom'] > 2.5 *vol_sma)
        
        signals = pd.Series('hold', index=df.index)
        signals[buy] = 'buy'
        signals[sell] = 'sell'
        # Set initial period to 'hold' where rolling windows are not valid
        min_period = max(self.window, self.res_window, self.sup_window)
        signals.iloc[:min_period - 1] = 'hold'
        signals = signals.shift(1).fillna('hold')
        return signals
        
