from typing import Any, Dict

class Strategy:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.position = 0  # 0 = out, 1 = long
        self.entry_price = None
        self.trades = []  # List of dicts: {'entry': ..., 'exit': ..., 'pnl': ...}
        self.current_bar = None

    def process_bar(self, bar: Dict[str, Any]):
        """
        Called for each bar (dict with keys: 'time', 'close', 'volume').
        Should update internal state and decide on signal.
        """
        self.current_bar = bar

    def get_signal(self) -> str:
        """
        Return 'buy', 'sell', or 'hold'.
        """
        return 'hold' 