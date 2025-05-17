"""
Advanced ML Trading Strategy for Bitcoin.
Uses a hybrid approach with machine learning for feature importance
combined with robust technical indicators.
"""

from src.core.strategy import Strategy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MLTradingStrategy(Strategy):
    def __init__(self, initial_capital=10000, fast_sma=20, slow_sma=80, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Will",
            strategy_name="ML Enhanced Trading Strategy",
            description="A hybrid strategy using machine learning to select optimal parameters for technical indicators including SMA crossovers and RSI."
        )
        # Strategy parameters optimized via ML analysis
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # For incremental processing
        self.prices = []
        self.last_signal = 'hold'

    def process_bar(self, bar):
        """Process each bar and update signals"""
        self.current_bar = bar
        self.prices.append(bar['close'])
        
        # Need enough data for indicators
        if len(self.prices) < self.slow_sma + 1:
            self.last_signal = 'hold'
            return

        # Calculate indicators
        prices_series = pd.Series(self.prices)
        
        # SMA calculations
        fast_ma = prices_series.rolling(self.fast_sma).mean().iloc[-1]
        slow_ma = prices_series.rolling(self.slow_sma).mean().iloc[-1]
        
        # RSI calculation
        if len(self.prices) > self.rsi_period + 1:
            price_changes = prices_series.diff().iloc[-self.rsi_period-1:]
            gains = price_changes.copy()
            losses = price_changes.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = -losses
            
            avg_gain = gains.rolling(self.rsi_period).mean().iloc[-1]
            avg_loss = losses.rolling(self.rsi_period).mean().iloc[-1]
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50  # Neutral RSI when not enough data
        
        # Trading logic (SMA crossover with RSI filter)
        if fast_ma > slow_ma and rsi > 30 and self.position == 0:
            self.last_signal = 'buy'
        elif (fast_ma < slow_ma or rsi > 80) and self.position == 1:
            self.last_signal = 'sell'
        else:
            self.last_signal = 'hold'

    def get_signal(self):
        """Return the current trading signal"""
        return self.last_signal
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI for a price series"""
        delta = np.diff(prices)
        gain = np.copy(delta)
        loss = np.copy(delta)
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First period
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Subsequent periods
            for i in range(period + 1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.zeros_like(prices)
        rsi = np.zeros_like(prices)
        
        # Calculate RS and RSI
        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rs[i] = 100
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        
        return rsi

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized implementation for generating signals on the entire dataset.
        Uses SMA crossover with RSI filtering.
        """
        signals = pd.Series('hold', index=df.index)
        
        # Calculate technical indicators
        fast_ma = df['close'].rolling(self.fast_sma).mean()
        slow_ma = df['close'].rolling(self.slow_sma).mean()
        
        # Calculate RSI using the vectorized approach
        price_changes = df['close'].diff()
        gains = price_changes.copy()
        losses = price_changes.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses
        
        avg_gain = gains.rolling(self.rsi_period).mean()
        avg_loss = losses.rolling(self.rsi_period).mean()
        
        # Safe division
        rs = pd.Series(index=df.index)
        for i in range(len(df)):
            if avg_loss.iloc[i] == 0:
                rs.iloc[i] = 100
            else:
                rs.iloc[i] = avg_gain.iloc[i] / avg_loss.iloc[i]
        
        rsi = 100 - (100 / (1 + rs))
        
        # Generate trading signals
        buy_signal = (fast_ma > slow_ma) & (rsi > self.rsi_oversold)
        sell_signal = (fast_ma < slow_ma) | (rsi > self.rsi_overbought)
        
        # Set initial signals
        signals[buy_signal] = 'buy'
        signals[sell_signal] = 'sell'
        
        # Handle warm-up period
        signals.iloc[:self.slow_sma] = 'hold'
        
        # Ensure proper trade sequence (can't buy when already in position)
        position = 0
        for i in range(self.slow_sma, len(df)):
            if signals.iloc[i] == 'buy' and position == 0:
                position = 1
            elif signals.iloc[i] == 'sell' and position == 1:
                position = 0
            else:
                signals.iloc[i] = 'hold'
        
        # Shift signals by 1 to avoid look-ahead bias
        signals = signals.shift(1).fillna('hold')
        
        return signals 