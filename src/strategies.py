import pandas as pd
from strategy import Strategy

class BuyAndHoldStrategy(Strategy):
    def __init__(self, initial_capital=10000):
        super().__init__(initial_capital)
        self.has_bought = False

    def process_bar(self, bar):
        self.current_bar = bar

    def get_signal(self):
        if not self.has_bought:
            self.has_bought = True
            return 'buy'
        return 'hold'
        
    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized version - returns 'buy' for first bar, 'hold' for all others
        """
        signals = pd.Series('hold', index=df.index)
        signals.iloc[0] = 'buy'
        return signals

class SMACrossoverStrategy(Strategy):
    def __init__(self, initial_capital=10000, fast=20, slow=100):
        super().__init__(initial_capital)
        self.prices = []
        self.fast = fast
        self.slow = slow
        self.last_signal = 'hold'

    def process_bar(self, bar):
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
        Vectorized version - calculates all signals at once, and shifts by 1 to avoid lookahead bias
        """
        # Calculate moving averages
        fast_ma = df['close'].rolling(self.fast).mean()
        slow_ma = df['close'].rolling(self.slow).mean()
        
        # Initialize signals as 'hold'
        signals = pd.Series('hold', index=df.index)
        
        # Generate signals based on crossover
        signals[fast_ma > slow_ma] = 'buy'
        signals[fast_ma < slow_ma] = 'sell'
        
        # First slow-1 bars should be 'hold' as we don't have enough data
        signals.iloc[:self.slow-1] = 'hold'
        
        # Shift signals forward by 1 bar to avoid lookahead bias
        signals = signals.shift(1).fillna('hold')
        
        return signals 