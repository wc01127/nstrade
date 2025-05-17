import pandas as pd
from metrics import (
    sharpe_ratio, total_return, n_trades, win_rate, max_drawdown, annualized_return,
    rolling_sharpe_ratio, unrealized_drawdown_series, realized_drawdown_series
)

def run_backtest(strategy_class, df, initial_capital=10000):

    # Local copy in memory for us to work with
    df = df.copy()    

    # Initialize strategy
    strat = strategy_class(initial_capital=initial_capital)

    equity_curve = [initial_capital]
    position = 0  # 0 = out, 1 = long
    entry_price = None
    coins = 0  # Track number of coins held

    for i, row in df.iterrows():
        bar = {'time': row['time'], 'close': row['close'], 'volume': row['volume']}
        strat.process_bar(bar)
        signal = strat.get_signal()

        # Enforce long-only, full portfolio, no re-buying, no shorting
        if signal == 'buy' and position == 0:
            position = 1
            entry_price = row['close']
            strat.position = 1
            strat.entry_price = entry_price
            coins = equity_curve[-1] / entry_price  # Buy as many coins as possible
            entry_idx = len(equity_curve) - 1
            print(f"BUY: time={row['time']}, entry_idx={entry_idx}, equity_curve_len={len(equity_curve)}, coins={coins}")
            trade = {'entry': row['time'], 'entry_idx': entry_idx, 'entry_price': entry_price}
        elif signal == 'sell' and position == 1:
            position = 0
            exit_price = row['close']
            equity = coins * exit_price  # Sell all coins
            pnl = equity - equity_curve[-1]
            exit_idx = len(equity_curve) - 1
            print(f"SELL: time={row['time']}, exit_idx={exit_idx}, equity_curve_len={len(equity_curve)}, coins={coins}")
            strat.trades.append({
                'entry': trade['entry'],
                'entry_idx': trade['entry_idx'],
                'entry_price': trade['entry_price'],
                'exit': row['time'],
                'exit_idx': exit_idx,
                'exit_price': exit_price,
                'pnl': pnl
            })
            equity_curve.append(equity)
            coins = 0
            entry_price = None
            strat.position = 0
            strat.entry_price = None
            continue  # Don't double-count equity update below

        # Update equity curve
        if position == 1:
            # Mark-to-market: value of held coins at current price
            equity = coins * row['close']
        else:
            equity = equity_curve[-1]
        equity_curve.append(equity)

    # If still in position at end, close it
    if position == 1:
        exit_price = df.iloc[-1]['close']
        equity = coins * exit_price
        pnl = equity - equity_curve[-1]
        exit_idx = len(equity_curve) - 1
        print(f"FINAL SELL: time={df.iloc[-1]['time']}, exit_idx={exit_idx}, equity_curve_len={len(equity_curve)}, coins={coins}")
        strat.trades.append({
            'entry': trade['entry'],
            'entry_idx': trade['entry_idx'],
            'entry_price': trade['entry_price'],
            'exit': df.iloc[-1]['time'],
            'exit_idx': exit_idx,
            'exit_price': exit_price,
            'pnl': pnl
        })
        equity_curve.append(equity)
        coins = 0

    # Calculate returns (hourly)
    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().fillna(0).values

    # Metrics
    results = {
        'sharpe': sharpe_ratio(returns),
        'total_return': total_return(equity_curve),
        'n_trades': n_trades(strat.trades),
        'win_rate': win_rate(strat.trades),
        'max_drawdown': max_drawdown(equity_curve),
        'annualized_return': annualized_return(equity_curve),
        'equity_curve': equity_curve,
        'trades': strat.trades,
        'rolling_sharpe': rolling_sharpe_ratio(equity_curve),
        'unrealized_drawdown': unrealized_drawdown_series(equity_curve),
        'realized_drawdown': realized_drawdown_series(equity_curve, strat.trades),
    }
    return results 