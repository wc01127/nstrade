"""
Jim's Enhanced Price Momentum Strategy.

This improved strategy builds on the original Price Momentum Strategy with added:
1. Trend confirmation using moving averages
2. Price movement threshold filter to reduce excessive trading
3. Position sizing based on volatility
4. Stop-loss and take-profit mechanisms
5. Trading cooldown period to prevent overtrading
"""

from src.core.strategy import Strategy
import pandas as pd
import numpy as np

class EnhancedPriceMomentumStrategy(Strategy):
    def __init__(self, initial_capital=10000, 
                 fast_ma_period=20, slow_ma_period=50,
                 price_threshold_pct=0.5, 
                 volatility_window=20,
                 max_position_pct=0.5,
                 stop_loss_pct=5.0, take_profit_pct=10.0,
                 cooldown_periods=24):
        """
        Initialize the Enhanced Price Momentum Strategy.
        
        Args:
            initial_capital: Starting capital
            fast_ma_period: Period for fast moving average
            slow_ma_period: Period for slow moving average
            price_threshold_pct: Minimum price movement percentage to trigger a trade
            volatility_window: Window for calculating volatility
            max_position_pct: Maximum percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            cooldown_periods: Number of periods to wait after a trade
        """
        super().__init__(
            initial_capital=initial_capital,
            author_name="Jim",
            strategy_name="Enhanced Price Momentum",
            description="Enhanced momentum strategy with trend confirmation, volatility-based position sizing, and risk management."
        )
        
        # Strategy parameters
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.price_threshold_pct = price_threshold_pct / 100.0
        self.volatility_window = volatility_window
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct / 100.0
        self.take_profit_pct = take_profit_pct / 100.0
        self.cooldown_periods = cooldown_periods
        
        # Initialize state variables
        self.previous_close = None
        self.last_signal = 'hold'
        self.prices = []
        self.entry_price = None
        self.periods_since_last_trade = 0
        self.volatility = None
        
    def process_bar(self, bar):
        """
        Process each bar of data and generate trading signals based on enhanced price momentum.
        
        Args:
            bar: Dictionary containing 'time', 'close', and 'volume' data
        """
        self.current_bar = bar
        self.prices.append(bar['close'])
        self.periods_since_last_trade += 1
        
        # Default signal is hold
        self.last_signal = 'hold'
        
        # Wait until we have enough data
        if len(self.prices) < self.slow_ma_period + 1:
            self.previous_close = bar['close']
            return
        
        # Calculate moving averages for trend confirmation
        fast_ma = np.mean(self.prices[-self.fast_ma_period:])
        slow_ma = np.mean(self.prices[-self.slow_ma_period:])
        
        # Calculate volatility for position sizing
        self.volatility = np.std(self.prices[-self.volatility_window:]) / np.mean(self.prices[-self.volatility_window:])
        
        # Calculate price change percentage
        price_change_pct = (bar['close'] - self.previous_close) / self.previous_close
        
        # Check for stop loss or take profit if in a position
        if self.position == 1 and self.entry_price is not None:
            current_return = (bar['close'] - self.entry_price) / self.entry_price
            
            # Check stop loss
            if current_return <= -self.stop_loss_pct:
                self.last_signal = 'sell'
                self.entry_price = None
                self.periods_since_last_trade = 0
                self.previous_close = bar['close']
                return
            
            # Check take profit
            if current_return >= self.take_profit_pct:
                self.last_signal = 'sell'
                self.entry_price = None
                self.periods_since_last_trade = 0
                self.previous_close = bar['close']
                return
        
        # Only consider trading if we're past the cooldown period
        if self.periods_since_last_trade >= self.cooldown_periods:
            # Generate signal based on price movement and trend confirmation
            if (abs(price_change_pct) >= self.price_threshold_pct and  # Price moved enough
                fast_ma > slow_ma and                                  # Upward trend
                price_change_pct > 0 and                               # Price increased
                self.position == 0):                                   # Not in position
                
                self.last_signal = 'buy'
                self.entry_price = bar['close']
                self.periods_since_last_trade = 0
                
            elif (abs(price_change_pct) >= self.price_threshold_pct and  # Price moved enough
                  fast_ma < slow_ma and                                  # Downward trend
                  price_change_pct < 0 and                               # Price decreased
                  self.position == 1):                                   # In position
                
                self.last_signal = 'sell'
                self.entry_price = None
                self.periods_since_last_trade = 0
        
        # Update previous close for next bar
        self.previous_close = bar['close']
    
    def get_signal(self):
        """
        Return the current trading signal.
        
        Returns:
            str: 'buy', 'sell', or 'hold'
        """
        return self.last_signal
    
    def get_position_size(self):
        """
        Calculate position size based on volatility.
        Lower volatility = larger position, higher volatility = smaller position.
        
        Returns:
            float: Position size as a fraction of available capital
        """
        if self.volatility is None or self.volatility == 0:
            return self.max_position_pct
        
        # Inverse relationship with volatility, capped at max_position_pct
        position_pct = min(self.max_position_pct, 
                          self.max_position_pct / (self.volatility * 10))
        return position_pct
    
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
        
        # Calculate price changes and percentage changes
        price_change = df['close'].diff()
        price_change_pct = df['close'].pct_change()
        
        # Calculate moving averages
        fast_ma = df['close'].rolling(window=self.fast_ma_period).mean()
        slow_ma = df['close'].rolling(window=self.slow_ma_period).mean()
        
        # Calculate volatility
        volatility = df['close'].rolling(window=self.volatility_window).std() / \
                    df['close'].rolling(window=self.volatility_window).mean()
        
        # Generate signals based on enhanced rules
        trend_up = fast_ma > slow_ma
        trend_down = fast_ma < slow_ma
        
        # Price moved enough (threshold filter)
        significant_move = price_change_pct.abs() >= self.price_threshold_pct
        
        # Buy signals: significant upward move during uptrend
        buy_signals = (significant_move & 
                       trend_up & 
                       (price_change > 0))
        
        # Sell signals: significant downward move during downtrend
        sell_signals = (significant_move & 
                        trend_down & 
                        (price_change < 0))
        
        # Apply signals
        signals[buy_signals] = 'buy'
        signals[sell_signals] = 'sell'
        
        # Apply cooldown periods
        buy_indices = signals[signals == 'buy'].index
        sell_indices = signals[signals == 'sell'].index
        
        all_trade_indices = sorted(list(buy_indices) + list(sell_indices))
        
        # Implement cooldown by setting signals to 'hold' during cooldown periods
        for i in range(len(all_trade_indices) - 1):
            current_idx = all_trade_indices[i]
            next_idx = all_trade_indices[i + 1]
            
            # Calculate periods between trades
            periods_between = df.index.get_indexer([next_idx])[0] - df.index.get_indexer([current_idx])[0]
            
            # If next trade is within cooldown period, set it to 'hold'
            if periods_between < self.cooldown_periods:
                signals.loc[next_idx] = 'hold'
                
        return signals
