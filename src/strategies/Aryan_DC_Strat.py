"""
Donchian Channel Strategy for Bitcoin.
Trades breakouts based on Donchian Channels.
"""

from src.core.strategy import Strategy
import pandas as pd

class AryanDCStrategy(Strategy):
    def __init__(self, initial_capital=10000, dc_period=20):
        super().__init__(
            initial_capital=initial_capital,
            author_name="AryanBhargav",
            strategy_name="Donchian Channel Strategy",
            description=f"Trades breakouts based on a {dc_period}-period Donchian Channel."
        )
        self.dc_period = dc_period
        self.prices = []
        self.last_signal = 'hold'

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        # Print column names for debugging
        print(f"Available columns in DataFrame: {df.columns.tolist()}")
        
        # Check if we have the necessary price columns
        required_columns = ['high', 'low', 'close']
        
        # Handle case insensitivity (some datasets use 'High', 'Low', 'Close')
        column_mapping = {}
        for col in required_columns:
            # Check for exact match
            if col in df.columns:
                column_mapping[col] = col
            # Check for case-insensitive match
            elif col.upper() in [c.upper() for c in df.columns]:
                for df_col in df.columns:
                    if df_col.upper() == col.upper():
                        column_mapping[col] = df_col
                        break
            else:
                # For OHLC data, try using the 'close' column as fallback for high/low if needed
                if col in ['high', 'low'] and 'close' in df.columns:
                    column_mapping[col] = 'close'
                    print(f"Warning: Using 'close' as fallback for '{col}'")
        
        # Handle the case where required columns are not found
        missing_columns = [col for col in required_columns if col not in column_mapping]
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            # Create basic signals with just 'hold' as fallback
            return pd.Series('hold', index=df.index)
        
        # Calculate Donchian Channels using mapped column names
        try:
            high_col = column_mapping['high']
            low_col = column_mapping['low']
            close_col = column_mapping['close']
            
            df['dc_upper'] = df[high_col].rolling(window=self.dc_period).max()
            df['dc_lower'] = df[low_col].rolling(window=self.dc_period).min()
            
            # Use previous bar's channel for signals
            df['dc_upper_prev'] = df['dc_upper'].shift(1)
            df['dc_lower_prev'] = df['dc_lower'].shift(1)

            signals = pd.Series('hold', index=df.index)
            
            # Buy when close breaks above upper channel
            signals[df[close_col] > df['dc_upper_prev']] = 'buy'
            
            # Sell when close breaks below lower channel
            signals[df[close_col] < df['dc_lower_prev']] = 'sell'
            
            # No signals during initial period
            signals.iloc[:self.dc_period] = 'hold'
            
            return signals
            
        except Exception as e:
            print(f"Error calculating signals: {str(e)}")
            # Return a series of 'hold' signals as fallback
            return pd.Series('hold', index=df.index)
        
