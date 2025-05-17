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