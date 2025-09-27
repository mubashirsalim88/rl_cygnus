#!/usr/bin/env python3
"""
Comprehensive backtesting and visualization script for trained TD3 agent.

This script loads a trained TD3 agent and backtests it against historical market data,
providing detailed performance metrics and professional visualizations.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch

# Optional matplotlib imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization features will be disabled.")

from datetime import datetime

# Add src directory to path for imports
src_path = Path(__file__).parent
sys.path.append(str(src_path))

from components.environment import TradingEnvironment
from components.agent import TD3Agent


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate feature data from CSV file.

    Args:
        data_path: Path to the processed feature data CSV file

    Returns:
        DataFrame containing the loaded feature data

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if len(df) == 0:
        raise ValueError("Data file is empty")

    if 'close' not in df.columns:
        raise ValueError("Data must contain 'close' price column")

    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def initialize_environment(df: pd.DataFrame) -> TradingEnvironment:
    """
    Initialize the trading environment with the loaded data.

    Args:
        df: DataFrame containing market data

    Returns:
        Initialized TradingEnvironment instance
    """
    print("Initializing trading environment...")
    env = TradingEnvironment(
        df=df,
        initial_balance=10000.0,
        commission_rate=0.001,
        slippage_factor=0.0005,
        latency=2
    )
    print(f"Environment initialized with {env.n_features} features")
    return env


def load_trained_agent(model_path: str, state_dim: int, action_dim: int = 1) -> TD3Agent:
    """
    Load a trained TD3 agent from saved model file.

    Args:
        model_path: Path to the saved model file
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space (default: 1 for trading)

    Returns:
        Loaded TD3Agent instance in evaluation mode

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading trained agent from: {model_path}")

    # Initialize agent
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load model weights
    agent.load(model_path)

    # Set to evaluation mode for deterministic actions
    agent.set_eval_mode()

    print(f"Agent loaded successfully on device: {agent.device}")
    return agent


def run_backtest(env: TradingEnvironment, agent: TD3Agent) -> pd.DataFrame:
    """
    Run the backtest loop with the trained agent.

    Args:
        env: Trading environment
        agent: Trained TD3 agent in evaluation mode

    Returns:
        DataFrame containing complete trading history
    """
    print("Starting backtest simulation...")

    def continuous_to_discrete_action(continuous_action: float) -> int:
        """Converts a continuous action into a discrete action."""
        if continuous_action < -0.33:
            return 2  # SELL
        elif continuous_action > 0.33:
            return 1  # BUY
        else:
            return 0  # HOLD

    # Reset environment
    state = env.reset()
    done = False
    step_count = 0

    while not done:
        # Get deterministic action from agent (no exploration noise)
        continuous_action = agent.select_action(state, add_noise=False)

        # Convert numpy array to float for discrete conversion
        continuous_action_float = float(continuous_action[0]) if isinstance(continuous_action, np.ndarray) else float(continuous_action)

        # Convert continuous action to discrete action
        discrete_action = continuous_to_discrete_action(continuous_action_float)

        # Take action in environment
        next_state, reward, done, info = env.step(discrete_action)

        # Move to next state
        state = next_state
        step_count += 1

        # Print progress every 1000 steps
        if step_count % 1000 == 0:
            print(f"Step {step_count}, Portfolio Value: ${info['portfolio_value']:.2f}")

    print(f"Backtest completed after {step_count} steps")

    # Get trading history
    history_df = env.get_trading_history()
    return history_df


def calculate_performance_metrics(history_df: pd.DataFrame, initial_balance: float = 10000.0) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from trading history.

    Args:
        history_df: DataFrame containing trading history
        initial_balance: Initial portfolio balance

    Returns:
        Dictionary containing performance metrics
    """
    if len(history_df) == 0:
        return {}

    final_value = history_df['portfolio_value'].iloc[-1]
    returns = history_df['portfolio_value'].pct_change().dropna()

    # Calculate metrics
    total_return = (final_value - initial_balance) / initial_balance

    # Annualized Sharpe Ratio (assuming hourly data, 8760 hours per year)
    if returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(8760)
    else:
        sharpe_ratio = 0.0

    # Maximum Drawdown
    portfolio_values = history_df['portfolio_value']
    peak = portfolio_values.expanding().max()
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()

    # Sortino Ratio (downside deviation)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0 and negative_returns.std() > 0:
        sortino_ratio = (returns.mean() / negative_returns.std()) * np.sqrt(8760)
    else:
        sortino_ratio = 0.0

    # Win Rate and Trade Count
    # Count actual trades (non-HOLD actions: BUY=1, SELL=2)
    trades = history_df[history_df['action'] != 0]
    total_trades = len(trades)

    if total_trades > 0:
        # Calculate returns for each trade period
        trade_returns = []
        for i in range(len(trades) - 1):
            start_value = trades.iloc[i]['portfolio_value']
            end_value = trades.iloc[i + 1]['portfolio_value']
            trade_return = (end_value - start_value) / start_value
            trade_returns.append(trade_return)

        winning_trades = sum(1 for ret in trade_returns if ret > 0)
        win_rate = (winning_trades / len(trade_returns)) * 100 if trade_returns else 0
    else:
        win_rate = 0

    return {
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': abs(max_drawdown) * 100,
        'sortino_ratio': sortino_ratio,
        'win_rate_pct': win_rate,
        'total_trades': total_trades,
        'final_value': final_value,
        'initial_value': initial_balance
    }


def create_visualization(history_df: pd.DataFrame, df: pd.DataFrame, save_path: str = "backtest_results.png") -> None:
    """
    Create professional visualization of backtest results.

    Args:
        history_df: DataFrame containing trading history
        df: Original market data DataFrame
        save_path: Path to save the visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping visualization: matplotlib not available")
        return

    print("Creating visualization...")

    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Prepare data for plotting
    if 'timestamp' in history_df.columns:
        timestamps = pd.to_datetime(history_df['timestamp'])
    else:
        # Fallback to using step indices
        timestamps = df.index[:len(history_df)]

    prices = history_df['price']
    portfolio_values = history_df['portfolio_value']
    actions = history_df['action']

    # Top subplot: Price chart with buy/sell signals
    ax1.plot(timestamps, prices, label='Asset Price', color='black', linewidth=1, alpha=0.8)

    # Mark buy and sell actions (discrete actions: 0=HOLD, 1=BUY, 2=SELL)
    buy_signals = history_df[actions == 1]
    sell_signals = history_df[actions == 2]

    if len(buy_signals) > 0:
        buy_timestamps = pd.to_datetime(buy_signals['timestamp']) if 'timestamp' in buy_signals.columns else df.index[:len(buy_signals)]
        ax1.scatter(buy_timestamps, buy_signals['price'],
                   marker='^', color='green', s=60, label=f'BUY ({len(buy_signals)})', alpha=0.8, zorder=5)

    if len(sell_signals) > 0:
        sell_timestamps = pd.to_datetime(sell_signals['timestamp']) if 'timestamp' in sell_signals.columns else df.index[:len(sell_signals)]
        ax1.scatter(sell_timestamps, sell_signals['price'],
                   marker='v', color='red', s=60, label=f'SELL ({len(sell_signals)})', alpha=0.8, zorder=5)

    ax1.set_ylabel('Asset Price ($)', fontsize=12)
    ax1.set_title('TD3 Agent Backtest Results', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: Portfolio value over time
    ax2.plot(timestamps, portfolio_values, label='Portfolio Value', color='blue', linewidth=2)
    ax2.axhline(y=history_df['portfolio_value'].iloc[0], color='gray', linestyle='--',
                alpha=0.7, label='Initial Value')

    ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {save_path}")

    # Close the figure to free memory
    plt.close()


def print_performance_report(metrics: Dict[str, float]) -> None:
    """
    Print a formatted performance report to console.

    Args:
        metrics: Dictionary containing performance metrics
    """
    print("\n" + "="*60)
    print("              TRADING PERFORMANCE REPORT")
    print("="*60)
    print(f"Initial Portfolio Value:    ${metrics['initial_value']:>12,.2f}")
    print(f"Final Portfolio Value:      ${metrics['final_value']:>12,.2f}")
    print(f"Total Return:               {metrics['total_return_pct']:>12.2f}%")
    print(f"Annualized Sharpe Ratio:    {metrics['sharpe_ratio']:>12.2f}")
    print(f"Maximum Drawdown:           {metrics['max_drawdown_pct']:>12.2f}%")
    print(f"Sortino Ratio:              {metrics['sortino_ratio']:>12.2f}")
    print(f"Win Rate:                   {metrics['win_rate_pct']:>12.2f}%")
    print(f"Total Number of Trades:     {metrics['total_trades']:>12}")
    print("="*60)

    # Performance assessment
    if metrics['total_return_pct'] > 0:
        print(" Strategy generated positive returns")
    else:
        print(" Strategy generated negative returns")

    if metrics['sharpe_ratio'] > 1.0:
        print(" Good risk-adjusted returns (Sharpe > 1.0)")
    elif metrics['sharpe_ratio'] > 0.5:
        print("~ Moderate risk-adjusted returns (Sharpe > 0.5)")
    else:
        print(" Poor risk-adjusted returns (Sharpe < 0.5)")

    print()


def main() -> None:
    """Main function to run the backtest."""
    parser = argparse.ArgumentParser(
        description="Backtest a trained TD3 trading agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to processed feature data CSV file (e.g., ../data/processed/BTCUSDT-1h_features.csv)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved trained agent model file (e.g., ../models/TD3_best.pth)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save backtest results"
    )

    args = parser.parse_args()

    try:
        # Load data and initialize environment
        df = load_data(args.data_path)
        env = initialize_environment(df)

        # Load trained agent
        agent = load_trained_agent(args.model_path, state_dim=env.n_features)

        # Run backtest
        history_df = run_backtest(env, agent)

        # Calculate performance metrics
        metrics = calculate_performance_metrics(history_df)

        # Generate performance report
        print_performance_report(metrics)

        # Create and save visualization
        output_path = os.path.join(args.output_dir, "backtest_results.png")
        create_visualization(history_df, df, output_path)

        # Save trading history to CSV
        history_path = os.path.join(args.output_dir, "trading_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"Trading history saved to: {history_path}")

        print("\nBacktest completed successfully!")

    except Exception as e:
        print(f"Error during backtest: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()