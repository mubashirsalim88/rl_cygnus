# RL Cygnus

A comprehensive reinforcement learning system for cryptocurrency trading using Proximal Policy Optimization (PPO) algorithm.

## Features

- **Advanced Trading Environment**: Realistic market simulation with commission costs, slippage, and latency
- **PPO Agent**: State-of-the-art policy gradient reinforcement learning algorithm with continuous action spaces
- **Feature Engineering**: Comprehensive technical indicators and market regime features
- **Data Pipeline**: Automated data collection from Binance API with feature processing
- **Training Infrastructure**: Complete training loop with MLflow experiment tracking and model checkpointing

## Quick Start

### 1. Setup Project
Install all required dependencies.
```bash
pip install -r requirements.txt
```

### 2. Generate Features
Create the dataset for the agent to train on.
```bash
python scripts/run_feature_engineering.py --symbol BTCUSDT --interval 5m --start_date 2023-01-01 --end_date 2023-12-31
```

### 3. Run Training with MLflow
First, start the MLflow UI to track your experiments.
```bash
# In one terminal, start the dashboard
mlflow ui
```

In another terminal, start a training run.
```bash
# Run a training job with default parameters
python src/main_train.py

# Override parameters from the command line for new experiments
python src/main_train.py agent.lr=0.001 training.num_episodes=100
```

## Project Structure

```
rl_cygnus/
├── src/
│   ├── components/
│   │   ├── environment.py      # Trading environment with market frictions
│   │   ├── agent_ppo.py       # PPO agent implementation
│   │   └── feature_engineering.py  # Technical indicators and features
│   ├── data_sourcing/
│   │   └── binance_loader.py  # Binance API data loader
│   └── main_train.py          # Main training script with MLflow integration
├── scripts/
│   ├── run_feature_engineering.py  # Feature pipeline script
│   └── clean_project.py       # Project cleanup utility
├── configs/                   # Hydra configuration files
├── data/
│   ├── raw/                   # Raw OHLCV data
│   └── processed/             # Processed features
├── models/                    # Trained model checkpoints
├── mlruns/                    # MLflow experiment tracking data
├── run_gcp_experiments.sh     # Batch experiment runner for GCP
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
- **Continuous Action Space**: Fine-grained position sizing from -1 (max short) to +1 (max long)
- **Portfolio Tracking**: Real-time portfolio value and performance metrics

## Agent Features

- **PPO Algorithm**: Proximal Policy Optimization with clipped surrogate objective for stable policy updates
- **Actor-Critic Architecture**: Separate policy and value networks for efficient learning
- **Model Management**: Automatic checkpointing and best model saving with MLflow tracking
- **Comprehensive Logging**: Training metrics, performance tracking, and experiment comparison via MLflow UI

## Usage Notes

- Start with small date ranges for initial testing
- Monitor training progress through MLflow UI at http://localhost:5000
- Adjust hyperparameters via command line or Hydra config files
- Use `run_gcp_experiments.sh` to launch batch experiments on remote servers

## Contributing

This project implements academic research in reinforcement learning for financial markets. Ensure proper risk management when deploying in live trading environments.
