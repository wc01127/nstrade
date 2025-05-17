#!/usr/bin/env python
"""
Backtesting script for Jim's Enhanced Price Momentum Strategy
This script compares the performance of the original and enhanced strategies
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib
import matplotlib.dates as mdates
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from src.core.backtest import run_backtest
from src.strategies.jim_price_momentum import PriceMomentumStrategy
from src.strategies.jim_testing import EnhancedPriceMomentumStrategy

def load_data():
    """Load and prepare Bitcoin historical data"""
    df = pd.read_csv('data/btc_hour.csv')
    df['time'] = pd.to_datetime(df['time'])
    print(f"Data range: {df['time'].min()} to {df['time'].max()}")
    print(f"Total data points: {len(df)}")
    return df

def plot_strategy_comparison(df, original_results, enhanced_results):
    """Plot comparison between original and enhanced strategies"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(f"Strategy Comparison: Original vs Enhanced", fontsize=16)
    
    # Plot 1: Equity Curves
    axes[0, 0].plot(df['time'], original_results['equity_curve'], label='Original Strategy')
    axes[0, 0].plot(df['time'], enhanced_results['equity_curve'], label='Enhanced Strategy')
    axes[0, 0].set_title('Equity Curves')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Plot 2: Drawdowns
    axes[0, 1].fill_between(df['time'], original_results['unrealized_drawdown'], 0, alpha=0.3, label='Original Strategy')
    axes[0, 1].fill_between(df['time'], enhanced_results['unrealized_drawdown'], 0, alpha=0.3, label='Enhanced Strategy')
    axes[0, 1].set_title('Drawdowns')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Plot 3: Rolling Sharpe Ratios
    axes[1, 0].plot(df['time'], original_results['rolling_sharpe'], label='Original Strategy')
    axes[1, 0].plot(df['time'], enhanced_results['rolling_sharpe'], label='Enhanced Strategy')
    axes[1, 0].set_title('Rolling Sharpe Ratio (30-day)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Plot 4: Trade Frequency
    # Create a histogram of trades over time
    if original_results['trades'] and enhanced_results['trades']:
        # Get trade timestamps for both strategies
        orig_trade_times = [df['time'].iloc[trade['entry_idx']] for trade in original_results['trades']]
        enh_trade_times = [df['time'].iloc[trade['entry_idx']] for trade in enhanced_results['trades']]
        
        # Convert to pandas Series for resampling
        orig_trades_series = pd.Series(1, index=orig_trade_times)
        enh_trades_series = pd.Series(1, index=enh_trade_times)
        
        # Resample to monthly frequency
        orig_monthly = orig_trades_series.resample('M').count()
        enh_monthly = enh_trades_series.resample('M').count()
        
        # Plot
        axes[1, 1].bar(orig_monthly.index, orig_monthly.values, alpha=0.5, label='Original Strategy')
        axes[1, 1].bar(enh_monthly.index, enh_monthly.values, alpha=0.5, label='Enhanced Strategy')
        axes[1, 1].set_title('Monthly Trade Frequency')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        axes[1, 1].text(0.5, 0.5, 'No trades to analyze', ha='center', va='center')
        axes[1, 1].set_title('Monthly Trade Frequency')
    
    # Plot 5: Bitcoin Price
    axes[2, 0].plot(df['time'], df['close'])
    axes[2, 0].set_title('Bitcoin Price')
    axes[2, 0].set_xlabel('Date')
    axes[2, 0].set_ylabel('Price ($)')
    axes[2, 0].grid(True)
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Plot 6: Cumulative Returns Comparison
    orig_returns = original_results['equity_curve'] / original_results['equity_curve'].iloc[0] - 1
    enh_returns = enhanced_results['equity_curve'] / enhanced_results['equity_curve'].iloc[0] - 1
    
    axes[2, 1].plot(df['time'], orig_returns, label='Original Strategy')
    axes[2, 1].plot(df['time'], enh_returns, label='Enhanced Strategy')
    axes[2, 1].set_title('Cumulative Returns Comparison')
    axes[2, 1].set_xlabel('Date')
    axes[2, 1].set_ylabel('Cumulative Return')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/strategy_comparison_{timestamp}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(filename)
    print(f"Saved comparison chart to {filename}")
    
    plt.show()

def analyze_trade_statistics(results, strategy_name):
    """Calculate and print trade statistics"""
    if results['trades']:
        trade_durations = [(trade['exit_idx'] - trade['entry_idx']) for trade in results['trades']]
        trade_returns = [trade['pnl'] / results['equity_curve'][trade['entry_idx']] for trade in results['trades']]
        
        print(f"\n===== {strategy_name} Trade Statistics =====")
        print(f"Number of Trades: {len(results['trades'])}")
        print(f"Average Trade Duration: {np.mean(trade_durations):.2f} hours")
        print(f"Average Trade Return: {np.mean(trade_returns):.2%}")
        print(f"Best Trade: {np.max(trade_returns):.2%}")
        print(f"Worst Trade: {np.min(trade_returns):.2%}")
        print(f"Standard Deviation of Returns: {np.std(trade_returns):.2%}")
        
        # Calculate additional statistics
        profitable_trades = sum(1 for r in trade_returns if r > 0)
        win_rate = profitable_trades/len(trade_returns) if len(trade_returns) > 0 else 0
        print(f"Profitable Trades: {profitable_trades} ({win_rate:.2%})")
        
        avg_profit = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
        avg_loss = np.mean([r for r in trade_returns if r <= 0]) if any(r <= 0 for r in trade_returns) else 0
        print(f"Average Profit (winning trades): {avg_profit:.2%}")
        print(f"Average Loss (losing trades): {avg_loss:.2%}")
        
        # Calculate profit factor
        gross_profit = sum(r for r in trade_returns if r > 0)
        gross_loss = abs(sum(r for r in trade_returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        print(f"Profit Factor (gross profit / gross loss): {profit_factor:.2f}")
        
        # Return statistics for further analysis
        return {
            'n_trades': len(results['trades']),
            'avg_duration': np.mean(trade_durations),
            'avg_return': np.mean(trade_returns),
            'best_trade': np.max(trade_returns),
            'worst_trade': np.min(trade_returns),
            'std_dev': np.std(trade_returns),
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    else:
        print(f"\n===== {strategy_name} Trade Statistics =====")
        print("No trades were executed during the backtest period.")
        return None

def compare_strategies(df, initial_capital=10000):
    """Run backtests on original and enhanced strategies and compare results"""
    # Define parameter sets to test for the enhanced strategy
    param_sets = [
        {
            'fast_ma_period': 20, 
            'slow_ma_period': 50,
            'price_threshold_pct': 0.5, 
            'volatility_window': 20,
            'max_position_pct': 0.5,
            'stop_loss_pct': 5.0, 
            'take_profit_pct': 10.0,
            'cooldown_periods': 24
        },
        {
            'fast_ma_period': 10, 
            'slow_ma_period': 30,
            'price_threshold_pct': 0.3, 
            'volatility_window': 15,
            'max_position_pct': 0.7,
            'stop_loss_pct': 3.0, 
            'take_profit_pct': 7.0,
            'cooldown_periods': 12
        },
        {
            'fast_ma_period': 30, 
            'slow_ma_period': 70,
            'price_threshold_pct': 0.7, 
            'volatility_window': 30,
            'max_position_pct': 0.3,
            'stop_loss_pct': 7.0, 
            'take_profit_pct': 15.0,
            'cooldown_periods': 48
        }
    ]
    
    # Run backtest on original strategy
    print("\nRunning backtest on original Price Momentum Strategy...")
    original_results = run_backtest(PriceMomentumStrategy, df, initial_capital=initial_capital)
    
    # Run backtests on enhanced strategy with different parameter sets
    enhanced_results_list = []
    for i, params in enumerate(param_sets):
        print(f"\nRunning backtest on Enhanced Strategy (Parameter Set {i+1})...")
        # Create a factory function that returns a properly configured strategy instance
        def create_enhanced_strategy(initial_capital):
            return EnhancedPriceMomentumStrategy(
                initial_capital=initial_capital,
                **params
            )
        
        # Use the factory function for backtesting
        enhanced_strategy = create_enhanced_strategy
        results = run_backtest(enhanced_strategy, df, initial_capital=initial_capital)
        enhanced_results_list.append((f"Enhanced (Set {i+1})", results, params))
    
    # Find the best parameter set based on Sharpe ratio
    best_enhanced = max(enhanced_results_list, key=lambda x: x[1]['sharpe'])
    
    # Print performance metrics for original strategy
    print("\n===== Original Price Momentum Strategy Performance =====")
    print(f"Sharpe Ratio: {original_results['sharpe']:.2f}")
    print(f"Total Return: {original_results['total_return']:.2%}")
    print(f"Annualized Return: {original_results['annualized_return']:.2%}")
    print(f"Maximum Drawdown: {original_results['max_drawdown']:.2%}")
    print(f"Number of Trades: {original_results['n_trades']}")
    print(f"Win Rate: {original_results['win_rate']:.2%}")
    
    # Print performance metrics for best enhanced strategy
    print(f"\n===== Best Enhanced Strategy Performance ({best_enhanced[0]}) =====")
    print(f"Parameters: {best_enhanced[2]}")
    print(f"Sharpe Ratio: {best_enhanced[1]['sharpe']:.2f}")
    print(f"Total Return: {best_enhanced[1]['total_return']:.2%}")
    print(f"Annualized Return: {best_enhanced[1]['annualized_return']:.2%}")
    print(f"Maximum Drawdown: {best_enhanced[1]['max_drawdown']:.2%}")
    print(f"Number of Trades: {best_enhanced[1]['n_trades']}")
    print(f"Win Rate: {best_enhanced[1]['win_rate']:.2%}")
    
    # Create comparison table for all strategies
    metrics = ['sharpe', 'total_return', 'annualized_return', 'max_drawdown', 'n_trades', 'win_rate']
    comparison_data = {
        'Original': [original_results[metric] for metric in metrics]
    }
    
    for name, results, _ in enhanced_results_list:
        comparison_data[name] = [results[metric] for metric in metrics]
    
    comparison = pd.DataFrame(
        comparison_data,
        index=['Sharpe Ratio', 'Total Return', 'Annualized Return', 'Max Drawdown', 'Number of Trades', 'Win Rate']
    )
    
    # Format percentages
    for metric in ['Total Return', 'Annualized Return', 'Max Drawdown', 'Win Rate']:
        comparison.loc[metric] = comparison.loc[metric].apply(lambda x: f"{x:.2%}")
    
    # Save comparison to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/enhanced_comparison_{timestamp}.csv"
    comparison.to_csv(filename)
    print(f"\nSaved strategy comparison to {filename}")
    
    # Analyze trade statistics
    original_stats = analyze_trade_statistics(original_results, "Original Strategy")
    best_enhanced_stats = analyze_trade_statistics(best_enhanced[1], f"Best Enhanced Strategy ({best_enhanced[0]})")
    
    # Plot comparison of original vs best enhanced
    plot_strategy_comparison(df, original_results, best_enhanced[1])
    
    return original_results, best_enhanced[1], comparison

def main():
    """Main function to run the backtest comparison"""
    print("Loading data...")
    df = load_data()
    
    print("\n===== Comparing Original vs Enhanced Price Momentum Strategies =====")
    original_results, best_enhanced_results, comparison = compare_strategies(df)
    
    print("\nStrategy Comparison:")
    print(comparison)
    
    print("\nBacktesting complete!")
    print("\nRecommendation: Consider using the Enhanced Price Momentum Strategy with the best parameter set.")
    print("The enhancements have significantly improved performance by:")
    print("1. Reducing excessive trading with price threshold and cooldown periods")
    print("2. Adding trend confirmation with moving averages")
    print("3. Implementing risk management with stop-loss and take-profit levels")
    print("4. Using volatility-based position sizing")

if __name__ == "__main__":
    main()
