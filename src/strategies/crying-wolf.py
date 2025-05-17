from src.core.strategy import Strategy
import pandas as pd
import numpy as np


class FullMoonMeanRevertStrategy(Strategy):
    def __init__(
        self,
        initial_capital=10000,
        moon_phase_path='../data/moon_phases.csv',
        df=None,
        zentry_threshold=4,
        z_exit_threshold=1,
        max_hold_hours=6,
        sma_lookback_hours=24,
        aggressiveness_full=1.0,
        aggressiveness_empty=0.01,
        stop_loss_pct=0.1,  # 2% stop loss by default
    ):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Adi, Mikey, Alex",
            strategy_name="Z-Score Moon Mean Revert",
            description=(
                "Z-score mean reversion strategy, entry/exit thresholds and holding time modulated by moon phase."
            )
        )
        self.df = df
        self.moon_phase_path = moon_phase_path
        self.zentry_threshold = zentry_threshold
        self.z_exit_threshold = z_exit_threshold
        self.max_hold_hours = max_hold_hours
        self.sma_lookback_hours = sma_lookback_hours
        self.aggressiveness_full = aggressiveness_full
        self.aggressiveness_empty = aggressiveness_empty
        self.stop_loss_pct = stop_loss_pct

        self._prepare_moon_phases()

        self.position = 0
        self.entry_index = None
        self.entry_time = None
        self.entry_price = None
        self.last_signal = 'hold'

    def _prepare_moon_phases(self):
        # Load and clean moon phase data: only first and third columns, only categories 4 and 7
        moon = pd.read_csv(self.moon_phase_path, usecols=[0, 2])
        moon.columns = ['date', 'category']
        moon = moon[moon['category'].isin([4, 7])].copy()
        # Set time to 12:00 Beijing time (UTC+8), then convert to naive UTC
        moon['datetime'] = pd.to_datetime(
            moon['date']) + pd.Timedelta(hours=12)
        moon['datetime'] = moon['datetime'].dt.tz_localize(
            'Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)
        self.moon_dates = moon[['datetime', 'category']].reset_index(drop=True)

    def _get_aggressiveness(self, current_time):
        # Find the most recent moon phase
        moon = self.moon_dates[self.moon_dates['datetime'] <= current_time]
        if moon.empty:
            return 1.0
        last = moon.iloc[-1]
        if last['category'] == 4:
            return self.aggressiveness_full
        elif last['category'] == 7:
            return self.aggressiveness_empty
        return 1.0

    def process_bar(self, bar):
        if self.df is None:
            self.last_signal = 'hold'
            return

        # Ensure bar['time'] is timezone-naive
        bar_time = pd.to_datetime(bar['time'])
        if bar_time.tzinfo is not None:
            bar_time = bar_time.tz_localize(None)

        idxs = self.df.index[self.df['time'] == bar_time].tolist()
        idx = idxs[0] if idxs else None
        if idx is None or idx < self.sma_lookback_hours:
            self.last_signal = 'hold'
            return

        window = self.df.loc[idx-self.sma_lookback_hours+1:idx, 'close']
        mean = window.mean()
        std = window.std()
        z = (bar['close'] - mean) / std if std > 0 else 0
        z_prev = (self.df.loc[idx-1, 'close'] - window[:-1].mean()
                  ) / window[:-1].std() if window[:-1].std() > 0 else 0

        aggr = self._get_aggressiveness(bar_time)
        zentry = self.zentry_threshold * aggr
        zexit = self.z_exit_threshold * aggr

        if self.position == 0:
            if z < -zentry and z > z_prev:
                self.position = 1
                self.entry_index = idx
                self.entry_time = bar_time
                self.entry_price = bar['close']
                self.last_signal = 'buy'
            elif z > zentry and z < z_prev:
                self.position = -1
                self.entry_index = idx
                self.entry_time = bar_time
                self.entry_price = bar['close']
                self.last_signal = 'sell'
            else:
                self.last_signal = 'hold'
        else:
            hours_held = (bar_time - self.entry_time).total_seconds() / 3600.0
            stop_loss_triggered = False
            if self.position == 1 and bar['close'] <= self.entry_price * (1 - self.stop_loss_pct):
                stop_loss_triggered = True
            elif self.position == -1 and bar['close'] >= self.entry_price * (1 + self.stop_loss_pct):
                stop_loss_triggered = True

            if abs(z) < zexit or hours_held >= self.max_hold_hours or stop_loss_triggered:
                self.position = 0
                self.entry_index = None
                self.entry_time = None
                self.entry_price = None
                if stop_loss_triggered:
                    self.last_signal = 'stop_loss'
                else:
                    self.last_signal = 'sell' if self.position == 1 else 'buy'
            else:
                self.last_signal = 'hold'

    def get_signal(self):
        return self.last_signal

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df = df.sort_values('time').reset_index(drop=True)
        signals = pd.Series('hold', index=df.index)

        df['rolling_mean'] = df['close'].rolling(
            window=self.sma_lookback_hours, min_periods=1).mean()
        df['rolling_std'] = df['close'].rolling(
            window=self.sma_lookback_hours, min_periods=1).std()
        df['z'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
        df['z_prev'] = df['z'].shift(1)

        position = 0
        entry_time = None
        entry_price = None

        for i in range(1, len(df)):
            z = df['z'].iloc[i]
            z_prev = df['z_prev'].iloc[i]
            time = df['time'].iloc[i]
            price = df['close'].iloc[i]

            aggr = self._get_aggressiveness(time)
            zentry = self.zentry_threshold * aggr
            zexit = self.z_exit_threshold * aggr

            if position == 0:
                if z < -zentry and z > z_prev:
                    position = 1
                    entry_time = time
                    entry_price = price
                    signals.iloc[i] = 'buy'
                elif z > zentry and z < z_prev:
                    position = -1
                    entry_time = time
                    entry_price = price
                    signals.iloc[i] = 'sell'
            else:
                hours_held = (time - entry_time).total_seconds() / 3600.0
                stop_loss_triggered = False
                if position == 1 and price <= entry_price * (1 - self.stop_loss_pct):
                    stop_loss_triggered = True
                elif position == -1 and price >= entry_price * (1 + self.stop_loss_pct):
                    stop_loss_triggered = True

                if abs(z) < zexit or hours_held >= self.max_hold_hours or stop_loss_triggered:
                    if stop_loss_triggered:
                        signals.iloc[i] = 'stop_loss'
                    else:
                        signals.iloc[i] = 'sell' if position == 1 else 'buy'
                    position = 0
                    entry_time = None
                    entry_price = None

        return signals
