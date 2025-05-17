"""
EMA Crossover + Adaptive ATR Filter strategy for Bitcoin.
Buy when fast EMA > slow EMA and ATR% exceeds its rolling quantile threshold.
Sell when EMA crosses down or ATR% falls below that adaptive threshold.
"""

import pandas as pd
from src.core.strategy import Strategy  # adjust import as needed

class EMACrossoverWithAdaptiveATRFilterStrategy(Strategy):
    def __init__(self,
                 initial_capital=10000,
                 fast=30,
                 slow=120,
                 atr_period=14,
                 atr_quantile_window=100,
                 atr_quantile=0.5):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Lucien",
            strategy_name="EMA Crossover + Adaptive ATR Filter",
            description=(
                "Buy when fast EMA > slow EMA and ATR% > rolling "
                "quantile threshold; sell on reversal or low adaptive volatility."
            )
        )
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_quantile_window = atr_quantile_window
        self.atr_quantile = atr_quantile

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series('hold', index=df.index)

        # 1. Compute EMAs
        ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()

        # 2. Compute ATR% (using close-only TR approximation)
        prev_close = df['close'].shift(1)
        tr = (df['close'] - prev_close).abs()
        atr = tr.rolling(window=self.atr_period).mean()
        atr_pct = atr / df['close']

        # 3. Build adaptive threshold: rolling quantile of ATR%
        atr_threshold = atr_pct.rolling(window=self.atr_quantile_window) \
                               .quantile(self.atr_quantile)

        # 4. Entry / Exit conditions
        buy_cond = (ema_fast > ema_slow) & (atr_pct > atr_threshold)
        sell_cond = (ema_fast < ema_slow) | (atr_pct < atr_threshold)

        signals[buy_cond] = 'buy'
        signals[sell_cond] = 'sell'

        # 5. Warm-up / prevent lookahead
        warmup = max(self.slow, self.atr_period, self.atr_quantile_window)
        signals.iloc[:warmup] = 'hold'

        # 6. Shift by one to avoid lookahead bias
        return signals.shift(1).fillna('hold')
