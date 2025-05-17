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

    def process_bar(self, bar):
        self.current_bar = bar
        self.prices.append(bar['close'])
        self.volumes.append(bar['volume'])
        if len(self.prices) < max(self.window, self.res_window, self.sup_window):
            self.last_signal = 'hold'
            return
        # Calculate resistance and support as recent max/min close
        resistance = pd.Series(self.prices).rolling(self.res_window).max().iloc[-2]
        support = pd.Series(self.prices).rolling(self.sup_window).min().iloc[-2]
        # Calculate SMA of volume
        vol_sma = pd.Series(self.volumes).rolling(self.window).mean().iloc[-2]

        # Print some log to debug the resistance vs the self.price[-2], self.price[-1], support
        logging.debug(f"resistance={resistance}, prev_close={self.prices[-2]}, curr_close={self.prices[-1]}, support={support}")

        # Buy: price breaks above resistance with high volume
        if self.prices[-2] <= resistance and self.prices[-1] > resistance and bar['volume'] > vol_sma and self.position == 0:
            self.last_signal = 'buy'
        # Sell: price breaks below support with high volume
        elif self.prices[-2] >= support and self.prices[-1] < support and bar['volume'] > vol_sma and self.position == 1:
            self.last_signal = 'sell'
        else:
            self.last_signal = 'hold'

    def get_signal(self):
        return self.last_signal
    
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
        