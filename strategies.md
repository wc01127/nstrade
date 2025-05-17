## Strategies

1. Talk to your LLM
2. One option:
    - Try to optimise the simple moving average (SMA) crossover we just did
3. Maybe load up MiniAI (fastai thing) and try to do an ML based strategy
4. Feature engineering and features
     - Technical analysis - check out `talib` for TA features like SMA, EMA, MACD, RSI etc
     - Take interestesting combinations of these, across different timeframes
     - Get hundreds of them, and feed them to your favourite ML model
     - Catboost, XGBoost, etc are usually good places to start.
5. Train a 1D diffusion model to simulate thousands of fake price trajectories, then train your model on all of those
6. Reinforcement learning - train an RL agent that is able to look at a new trade, look at history, and decide what to do every hour
