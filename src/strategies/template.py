"""
Template for creating a new trading strategy.
Copy this file and modify it to implement your strategy.
"""

from src.core.strategy import Strategy

class TemplateStrategy(Strategy):
    def __init__(self, initial_capital=10000):
        super().__init__(
            initial_capital=initial_capital,
            author_name="YOUR_NAME",  # Replace with your name
            strategy_name="YOUR_STRATEGY_NAME",  # Replace with your strategy name
            description="DESCRIBE YOUR STRATEGY HERE"  # Replace with your strategy description
        )
        
        # Add any strategy-specific initialization here
        # For example:
        # self.lookback_period = 20
        # self.threshold = 0.02

    def process_bar(self, bar):
        """
        Process each bar of data.
        This is where you implement your strategy logic.
        
        Args:
            bar: Dictionary containing 'time', 'close', and 'volume' data
        """
        self.current_bar = bar
        
        # Add your strategy logic here
        # For example:
        # if self.current_bar['close'] > self.previous_close * (1 + self.threshold):
        #     self.last_signal = 'buy'
        # elif self.current_bar['close'] < self.previous_close * (1 - self.threshold):
        #     self.last_signal = 'sell'
        # else:
        #     self.last_signal = 'hold'

    def get_signal(self):
        """
        Return the current trading signal.
        Must return one of: 'buy', 'sell', 'hold'
        """
        # Add your signal generation logic here
        return 'hold'

    def get_signals(self, df):
        """
        Vectorized version of signal generation.
        Override this if you want to implement a more efficient vectorized version.
        """
        return super().get_signals(df) 