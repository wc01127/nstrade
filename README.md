# nstrade
NS Learnathon - Trading for Bitcoin

# install brew
Go to `brew.sh`

# install uv
If on mac
`brew install uv`

Apparently `uv` is `poetry` but better in every way. To be seen.

[website here](https://docs.astral.sh/uv/getting-started/installation/)

# fork the repo
Go to [https://github.com/athon-millane/nstrade.git](https://github.com/athon-millane/nstrade.git) and fork the repo

# clone the repo
`git clone https://github.com/your-username/nstrade.git`

# Optional: install python if you don't have version 3.12
`uv python install 3.12`

# initialize venv and sync packages
`uv sync`

# Sync with changes to the upstream repo (the one at `athon-millane` repo)
Add upstream
`git remote add upstream https://github.com/athon-millane/nstrade.git`

Pull from upstream
`git pull upstream main`

Pull via fetch / merge (if you know what that means)
`git fetch upstream`
`git merge upstream main`

# To do
 - [ ] Speed up backtest (maybe vectorise, maybe parallelise)
 - [ ] Set up a standardised flow for creating, contributing and backtesting and new `Strategy`
 - [ ] Capture user metadata in that Strategy, so that we can show it on a leaderboard in our web UI (tbd)
 - [ ] Set up CI/CD YAML file which does the following:
   - Checks a users Strategy is legit (can be backtested). If not, reject, is so continue
   - Backtests the strategy on our dataset
   - Stores results (along with user metadata) in our leaderboard file. A unique submission will be based on User name + Strategy name. If the same user + strategy exists already, overwrite it.
   - We want to define our `development set` (essentially our training set), and also our `holdout set`
       - `development.csv` will have all data up to the end of 2024
       - `holdout.csv` will have all of 2025
       - Strategies should only leverage `development.csv`

# Future metrics
 - Let's start with `Sharpe` as our north star - acknowedge this isn't perfect
 - Drawdown, win rate, Sharpe ratio, Sortino, r^2 of equity
 - Barnabas suggest we use `r^2` of the equity curve

# Security
 - For this version, we will just check that test data isn't used in Strategy development
 - This is not the challenge we're here to solve right now

# Comment
 - SMA Crossover still looks potentially too good to be true.