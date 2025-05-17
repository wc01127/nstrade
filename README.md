# nstrade

NS Learnathon - Trading for Bitcoin

## Setup

### Prerequisites

1. **Install Homebrew**
   - Visit [brew.sh](https://brew.sh)

2. **Install uv**
   ```bash
   # On macOS
   brew install uv
   ```
   > Note: `uv` is like `poetry` but better in every way. [Learn more](https://docs.astral.sh/uv/getting-started/installation/)

3. **Fork and Clone**
   ```bash
   # Fork the repo at https://github.com/athon-millane/nstrade.git
   git clone https://github.com/your-username/nstrade.git
   ```

4. **Python Setup**
   ```bash
   # Optional: Install Python 3.12 if needed
   uv python install 3.12
   
   # Initialize venv and install dependencies
   uv sync
   ```

### Syncing with Upstream

```bash
# Add upstream remote
git remote add upstream https://github.com/athon-millane/nstrade.git

# Pull from upstream (either method)
git pull upstream main

# OR
git fetch upstream
git merge upstream main
```

## Creating a New Strategy

1. **Copy the Template**
   ```bash
   cp src/strategies/template.py src/strategies/my_strategy.py
   ```

2. **Edit Your Strategy**
   - Set your name, strategy name, and description
   - Implement your strategy logic in `process_bar()` and `get_signal()`
   - Optionally implement a vectorized version in `get_signals()`

3. **Quick Test**
   ```bash
   python scripts/test_strategy.py src/strategies/my_strategy.py
   ```
   This will:
   - Validate your strategy implementation
   - Test on development period (2011-2024)
   - Test on holdout period (2025)

4. **Update leader board**
   ```bash
   python scripts/update_leaderboard.py src/strategies/my_strategy.py
   ```
   This will:
   - Update your strategy results to the leaderboard.json

5. **Detailed Evaluation**
   - Open `notebooks/evaluate.ipynb` in Jupyter
   - Change the `strategy_path` to point to your strategy file
   - Run all cells to see:
     - Detailed performance metrics
     - Interactive plots of price, volume, equity curve, drawdowns, and Sharpe ratio
     - Buy/sell points overlaid on the price chart
     - Trade-by-trade analysis

6. **Submit Your Strategy**
   - Create a pull request
   - Our CI will validate and backtest your strategy
   - If successful, it will be added to the leaderboard

## Project Status

### Completed
- [x] Speed up backtest (maybe vectorise, maybe parallelise)
- [x] Set up a standardised flow for creating, contributing and backtesting new `Strategy`
- [x] Capture user metadata in Strategy for leaderboard display

### To Do
- [ ] Set up CI/CD YAML file:
  - Validate strategy backtestability
  - Run backtests on our dataset
  - Store results in leaderboard file
  - Handle unique submissions (User + Strategy name)
  - Define development/holdout sets:
    - `development.csv`: Data up to end of 2024
    - `holdout.csv`: All of 2025
    - Strategies should only use `development.csv`

## Metrics

### Current Focus
- Sharpe ratio as primary metric
- Drawdown
- Win rate
- Number of trades

### Future Metrics
- Sortino ratio
- R² of equity curve (suggested by Barnabas)

## Notes

### Security
- Current focus: Ensure test data isn't used in strategy development
- More comprehensive security measures to be addressed later

### Observations
- SMA Crossover performance appears suspiciously good
- Most bootstrapped strategies perform poorly without fees and slippage
- Recommended: Include 0.3% fee+slippage per trade