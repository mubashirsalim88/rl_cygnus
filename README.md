# RL Cygnus

A comprehensive reinforcement learning system for cryptocurrency trading using Dueling Double Deep Q-Network (DDQN) algorithm.

## Features

- **Advanced Trading Environment**: Realistic market simulation with commission costs, slippage, and latency
- **Dueling DDQN Agent**: State-of-the-art value-based reinforcement learning algorithm for discrete action spaces
- **Feature Engineering**: Comprehensive technical indicators and market regime features
- **Data Pipeline**: Automated data collection from Binance API with feature processing
- **Training Infrastructure**: Complete training loop with experience replay and model checkpointing

## Quick Start

### 1. Setup Project
```bash
python setup_project.py
```

### 2. Generate Features
```bash
python scripts/run_feature_engineering.py --symbol BTCUSDT --interval 1h --start_date 2023-01-01 --end_date 2023-06-01
```

### 3. Train Agent
```bash
python src/main_train.py
```

### 4. Clean Project (Optional)
```bash
python scripts/clean_project.py --force
```

## Project Structure

```
rl_cygnus/
├── src/
│   ├── components/
│   │   ├── environment.py      # Trading environment with market frictions
│   │   ├── agent.py           # Dueling DDQN agent implementation
│   │   ├── feature_engineering.py  # Technical indicators and features
│   │   └── replay_buffer.py   # Experience replay buffer
│   ├── data_sourcing/
│   │   └── binance_loader.py  # Binance API data loader
│   └── main_train.py          # Main training script
├── scripts/
│   ├── run_feature_engineering.py  # Feature pipeline script
│   └── clean_project.py       # Project cleanup utility
├── data/
│   ├── raw/                   # Raw OHLCV data
│   └── processed/             # Processed features
├── models/                    # Trained model checkpoints
└── requirements.txt           # Python dependencies
```

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- pandas-ta (technical analysis)
- python-binance
- hurst (Hurst exponent calculation)

## Environment Features

- **Realistic Market Conditions**: Commission fees, price slippage, and data latency
- **Comprehensive State Space**: Technical indicators, market regime features, and price derivatives
- **Discrete Action Space**: Three actions for trading decisions (BUY, HOLD, SELL)
- **Portfolio Tracking**: Real-time portfolio value and performance metrics

## Agent Features

- **Dueling DDQN Algorithm**: Dueling architecture with double Q-learning for stable value estimation
- **Experience Replay**: Large replay buffer for stable learning
- **Model Management**: Automatic checkpointing and best model saving
- **Comprehensive Logging**: Training metrics and performance tracking

## Usage Notes

- Start with small date ranges for initial testing
- Monitor training logs for convergence
- Adjust hyperparameters based on market conditions
- Use dry-run mode for project cleaning to preview changes

## Contributing

This project implements academic research in reinforcement learning for financial markets. Ensure proper risk management when deploying in live trading environments.
