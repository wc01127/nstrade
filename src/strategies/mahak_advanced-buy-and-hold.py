"""
Advanced Buy And Hold Strategy for Bitcoin.
This strategy aims to provide a robust, adaptive approach that can perform well across different market conditions while managing risk effectively.
"""

from src.core.strategy import Strategy
import pandas as pd
import ta

class AdvancedBuyAndHoldStrategy(Strategy):
    def __init__(
        self,
        initial_capital=10000,
        sma_fast=50,
        sma_slow=200,
        rsi_period=14,
        adx_period=14,
        atr_period=14,
        bb_window=20,
        trailing_atr_mult=2,
        min_hold_bars=6,
        cooldown_bars=24
    ):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Mahak",
            strategy_name="Advanced Buy And Hold Strategy",
            description="This strategy aims to provide a robust, adaptive approach that can perform well across different market conditions while managing risk effectively."
        )
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.bb_window = bb_window
        self.trailing_atr_mult = trailing_atr_mult
        self.min_hold_bars = min_hold_bars
        self.cooldown_bars = cooldown_bars

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        # Ensure required columns
        for col in ['high', 'low']:
            if col not in df.columns:
                df[col] = df['close']

        # Feature engineering
        sma_fast = df['close'].rolling(self.sma_fast).mean()
        sma_slow = df['close'].rolling(self.sma_slow).mean()
        rsi = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=self.adx_period).adx()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=self.atr_period).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'], window=self.bb_window)
        bb_low = bb.bollinger_lband()
        bb_high = bb.bollinger_hband()

        # Regime detection: trending if price > slow SMA, slow SMA is sloping up, and ADX > 20
        sma_slow_slope = sma_slow.diff(self.sma_slow // 10) > 0
        trending = (df['close'] > sma_slow) & sma_slow_slope & (adx > 20)
        sideways = ~trending

        signals = pd.Series('hold', index=df.index)
        position = False
        highest_close = None
        bars_held = 0
        cooldown = 0

        for i in range(len(df)):
            if cooldown > 0:
                cooldown -= 1
                signals.iloc[i] = 'hold'
                continue

            if not position:
                # Trend-following entry
                if trending.iloc[i]:
                    if (df['close'].iloc[i] > sma_fast.iloc[i]) and (rsi.iloc[i] > 50):
                        signals.iloc[i] = 'buy'
                        position = True
                        highest_close = df['close'].iloc[i]
                        bars_held = 0
                # Mean reversion entry
                elif sideways.iloc[i]:
                    if (df['close'].iloc[i] < bb_low.iloc[i]) and (rsi.iloc[i] < 30):
                        signals.iloc[i] = 'buy'
                        position = True
                        highest_close = df['close'].iloc[i]
                        bars_held = 0
            else:
                highest_close = max(highest_close, df['close'].iloc[i])
                trailing_stop = highest_close - self.trailing_atr_mult * atr.iloc[i]
                bars_held += 1

                # Trend-following exit
                if trending.iloc[i]:
                    if (
                        (df['close'].iloc[i] < sma_fast.iloc[i]) or
                        (rsi.iloc[i] < 45) or
                        (df['close'].iloc[i] < trailing_stop)
                    ) and (bars_held >= self.min_hold_bars):
                        signals.iloc[i] = 'sell'
                        position = False
                        highest_close = None
                        bars_held = 0
                        cooldown = self.cooldown_bars
                    else:
                        signals.iloc[i] = 'hold'
                # Mean reversion exit
                elif sideways.iloc[i]:
                    if (
                        (df['close'].iloc[i] > bb_high.iloc[i]) or
                        (rsi.iloc[i] > 50)
                    ) and (bars_held >= self.min_hold_bars):
                        signals.iloc[i] = 'sell'
                        position = False
                        highest_close = None
                        bars_held = 0
                        cooldown = self.cooldown_bars
                    else:
                        signals.iloc[i] = 'hold'
            # If not in position and no entry, hold
            if not position and signals.iloc[i] != 'buy':
                signals.iloc[i] = 'hold'

        signals = signals.shift(1).fillna('hold')
        return signals