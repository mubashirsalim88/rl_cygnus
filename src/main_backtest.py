import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from components.environment import TradingEnvironment
from components.agent import DuelingDDQNAgent

def calculate_performance_metrics(history_df: pd.DataFrame, initial_balance: float) -> dict:
    """Calculate and return key performance metrics."""
    final_value = history_df['portfolio_value'].iloc[-1]
    total_return = ((final_value - initial_balance) / initial_balance) * 100

    returns = history_df['portfolio_value'].pct_change().dropna()

    # Sharpe Ratio
    sharpe_ratio = 0
    if returns.std() > 0:
        # Assuming 252 trading days, and 288 5-min intervals per day
        annualization_factor = np.sqrt(252 * 288)
        sharpe_ratio = returns.mean() / returns.std() * annualization_factor

    # Max Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min() * 100

    # Sortino Ratio
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std()
    sortino_ratio = 0
    if downside_std > 0:
        sortino_ratio = returns.mean() / downside_std * annualization_factor

    # Win Rate and Trade Count
    trade_actions = history_df[history_df['action'].isin([1, 2])] # BUY=1, SELL=2
    total_trades = len(trade_actions)

    win_rate = 0.0
    if total_trades > 1:
        trade_returns = []
        for i in range(len(trade_actions) - 1):
            if trade_actions.iloc[i]['action'] == 1: # Start of a cycle is a BUY
                start_value = trade_actions.iloc[i]['portfolio_value']
                end_value = trade_actions.iloc[i + 1]['portfolio_value']
                trade_returns.append((end_value - start_value) / start_value)

        if trade_returns:
            winning_trades = sum(1 for ret in trade_returns if ret > 0)
            win_rate = (winning_trades / len(trade_returns)) * 100

    return {
        "Final Portfolio Value": final_value,
        "Total Return": total_return,
        "Annualized Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Sortino Ratio": sortino_ratio,
        "Win Rate": win_rate,
        "Total Number of Trades": total_trades
    }

def create_visualization(history_df: pd.DataFrame, output_path: str):
    """Create and save a visualization of the backtest results."""
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Plot 1: Price and Trades
    ax1.plot(history_df['timestamp'], history_df['price'], label='Asset Price', color='skyblue', alpha=0.8)
    buy_signals = history_df[history_df['action'] == 1]
    sell_signals = history_df[history_df['action'] == 2]
    ax1.scatter(buy_signals['timestamp'], buy_signals['price'], marker='^', color='green', label='Buy', s=100, edgecolors='k')
    ax1.scatter(sell_signals['timestamp'], sell_signals['price'], marker='v', color='red', label='Sell', s=100, edgecolors='k')
    ax1.set_title('Asset Price and Trading Signals', fontsize=16)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: Portfolio Value (Equity Curve)
    ax2.plot(history_df['timestamp'], history_df['portfolio_value'], label='Portfolio Value', color='darkorange')
    ax2.set_title('Portfolio Value Over Time (Equity Curve)', fontsize=16)
    ax2.set_xlabel('Timestamp', fontsize=12)
    ax2.set_ylabel('Portfolio Value (USD)', fontsize=12)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")

def run_backtest(data_path: str, model_path: str, output_dir: str = "."):
    """Run the backtest simulation."""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure timestamp is datetime
    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    print("Initializing trading environment...")
    env = TradingEnvironment(df, initial_balance=10000)
    state_dim = env.n_features
    action_dim = 3

    print(f"Loading trained agent from: {model_path}")
    agent = DuelingDDQNAgent(state_dim=state_dim, action_dim=action_dim)
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Starting backtest simulation...")
    state = env.reset()
    done = False
    history = []

    while not done:
        action = agent.select_action(state, epsilon=0.0)  # Use greedy policy for backtesting
        next_state, reward, done, info = env.step(action)
        history.append(info)
        state = next_state

        if info['step'] % 10000 == 0:
            print(f"Step {info['step']}, Portfolio Value: ${info['portfolio_value']:.2f}")

    print(f"Backtest completed after {len(history)} steps")
    history_df = pd.DataFrame(history)

    # --- PERFORMANCE REPORT ---
    metrics = calculate_performance_metrics(history_df, env.initial_balance)
    print("\n" + "="*60)
    print(" " * 18 + "TRADING PERFORMANCE REPORT")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<25} {value: >15.2f}" + ("%" if "Return" in key or "Rate" in key or "Drawdown" in key else ""))
        else:
            print(f"{key:<25} {value: >15}")
    print("="*60)

    # --- VISUALIZATION AND SAVE ---
    vis_path = f"{output_dir}/backtest_results_ddqn.png"
    hist_path = f"{output_dir}/trading_history_ddqn.csv"
    create_visualization(history_df, vis_path)
    history_df.to_csv(hist_path, index=False)
    print(f"Trading history saved to: {hist_path}")
    print("\nBacktest completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backtest a trained Dueling DDQN agent.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed feature data CSV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained agent model file (.pth).")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save backtest results.")
    args = parser.parse_args()

    run_backtest(args.data_path, args.model_path, args.output_dir)