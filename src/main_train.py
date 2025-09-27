#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3 Training Script for RL Cygnus

This script implements the complete training pipeline for the TD3 agent
in the cryptocurrency trading environment. It loads processed features,
initializes the environment and agent, and runs the full training loop
with experience replay and delayed policy updates.
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging

from components.environment import TradingEnvironment
from components.agent import TD3Agent
from components.replay_buffer import ReplayBuffer


def continuous_to_discrete_action(continuous_action: float) -> int:
    """
    Convert continuous action from TD3 to discrete action for environment.

    Args:
        continuous_action: Float value in range [-1, 1] from TD3 agent

    Returns:
        Discrete action: 0 (HOLD), 1 (BUY), or 2 (SELL)

    Mapping:
        [-1, -0.33] -> 2 (SELL)
        [-0.33, 0.33] -> 0 (HOLD)
        [0.33, 1] -> 1 (BUY)
    """
    if continuous_action <= -0.33:
        return 2  # SELL
    elif continuous_action >= 0.33:
        return 1  # BUY
    else:
        return 0  # HOLD


def setup_logging() -> logging.Logger:
    """Set up logging configuration for training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_processed_data(data_path: str) -> pd.DataFrame:
    """
    Load processed feature data from CSV file.

    Args:
        data_path: Path to the processed feature CSV file

    Returns:
        DataFrame containing processed features

    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the data is empty or invalid
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError("Loaded data is empty")

    # Ensure we have the required columns
    required_columns = ['close', 'volume', 'open', 'high', 'low']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def run_training(
    data_file: str = "BTCUSDT-1h_features.csv",
    num_episodes: int = 1000,
    batch_size: int = 256,
    start_timesteps: int = 10000,
    eval_freq: int = 100,
    save_freq: int = 500,
    max_timesteps: int = 1000000
) -> None:
    """
    Run the complete TD3 training pipeline.

    Args:
        data_file: Name of the processed data file in data/processed/
        num_episodes: Number of training episodes
        batch_size: Batch size for training
        start_timesteps: Number of random actions before training starts
        eval_freq: Frequency of evaluation episodes
        save_freq: Frequency of model saving
        max_timesteps: Maximum total timesteps for training
    """
    logger = setup_logging()

    logger.info("="*80)
    logger.info("STARTING TD3 TRAINING FOR RL CYGNUS")
    logger.info("="*80)

    # Load processed data
    logger.info(f"Loading processed data: {data_file}")
    data_path = Path("data/processed") / data_file
    df = load_processed_data(str(data_path))
    logger.info(f" Loaded {len(df)} timesteps with {len(df.columns)} features")
    logger.info(f" Data range: {df.index[0]} to {df.index[-1]}")

    # Initialize environment
    logger.info("Initializing trading environment...")
    env = TradingEnvironment(
        df=df,
        initial_balance=10000.0,
        commission_rate=0.001,
        slippage_factor=0.0005,
        latency=2
    )
    logger.info(f" Environment initialized with {env.n_features} features")

    # Get dimensions
    state_dim = env.n_features
    action_dim = 1  # Single continuous action

    # Initialize TD3 agent
    logger.info("Initializing TD3 agent...")
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        actor_lr=3e-5,
        critic_lr=3e-5,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2
    )

    agent_info = agent.get_network_info()
    logger.info(f" Agent initialized with {agent_info['total_parameters']} total parameters")
    logger.info(f" Device: {agent_info['device']}")

    # Initialize replay buffer
    logger.info("Initializing replay buffer...")
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=1000000
    )
    logger.info(f" Replay buffer initialized with capacity {replay_buffer.max_size}")

    # Training hyperparameters
    logger.info("Training configuration:")
    logger.info(f"  - Episodes: {num_episodes}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Start timesteps: {start_timesteps}")
    logger.info(f"  - Eval frequency: {eval_freq}")
    logger.info(f"  - Max timesteps: {max_timesteps}")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Training metrics
    total_timesteps = 0
    best_episode_reward = float('-inf')
    episode_rewards = []
    training_start_time = time.time()

    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING LOOP")
    logger.info("="*80)

    try:
        for episode in range(num_episodes):
            episode_start_time = time.time()
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done and total_timesteps < max_timesteps:
                # Action selection
                if total_timesteps < start_timesteps:
                    # Random exploration phase
                    action = np.random.uniform(-1, 1, size=action_dim)
                else:
                    # Agent policy with exploration noise
                    action = agent.select_action(state, add_noise=True, noise_scale=0.1)

                # Convert continuous action to discrete action
                discrete_action = continuous_to_discrete_action(action[0])

                # Execute action in environment
                next_state, reward, done, info = env.step(discrete_action)

                # Store transition in replay buffer
                replay_buffer.add(state, action, next_state, reward, done)

                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_timesteps += 1

                # Train agent (if buffer has enough samples)
                if total_timesteps >= start_timesteps and replay_buffer.is_ready(batch_size):
                    training_metrics = agent.train(replay_buffer, batch_size)

                    # Log training metrics occasionally
                    if total_timesteps % 1000 == 0:
                        logger.info(f"Step {total_timesteps}: "
                                   f"C1_Loss={training_metrics['critic1_loss']:.4f}, "
                                   f"C2_Loss={training_metrics['critic2_loss']:.4f}, "
                                   f"Q1_Mean={training_metrics['mean_q1']:.4f}")

            # Episode completion
            episode_time = time.time() - episode_start_time
            episode_rewards.append(episode_reward)

            # Get final portfolio information
            portfolio_summary = env.get_portfolio_summary()
            final_portfolio_value = portfolio_summary['portfolio_value']
            total_return_pct = portfolio_summary['total_return_pct']

            # Update best episode reward
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                # Save best model
                best_model_path = models_dir / "best_td3_model_5m.pth"
                agent.save(str(best_model_path))

            # Episode logging
            logger.info(f"Episode {episode + 1:4d}/{num_episodes} | "
                       f"Steps: {episode_steps:4d} | "
                       f"Reward: {episode_reward:8.4f} | "
                       f"Portfolio: ${final_portfolio_value:8.2f} | "
                       f"Return: {total_return_pct:6.2f}% | "
                       f"Time: {episode_time:.2f}s")

            # Evaluation episodes
            if (episode + 1) % eval_freq == 0:
                logger.info("\n" + "-"*60)
                logger.info("EVALUATION METRICS")
                logger.info("-"*60)

                # Calculate recent performance
                recent_episodes = min(eval_freq, len(episode_rewards))
                recent_rewards = episode_rewards[-recent_episodes:]
                avg_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards)

                logger.info(f"Last {recent_episodes} episodes:")
                logger.info(f"  - Average reward: {avg_reward:.4f} +/- {std_reward:.4f}")
                logger.info(f"  - Best episode reward: {best_episode_reward:.4f}")
                logger.info(f"  - Buffer size: {len(replay_buffer):,}")
                logger.info(f"  - Total timesteps: {total_timesteps:,}")

                # Replay buffer info
                buffer_info = replay_buffer.get_buffer_info()
                logger.info(f"  - Buffer fill: {buffer_info['fill_percentage']:.1f}%")
                logger.info("-"*60 + "\n")

            # Save model periodically
            if (episode + 1) % save_freq == 0:
                checkpoint_path = models_dir / f"td3_checkpoint_ep{episode + 1}_5m.pth"
                agent.save(str(checkpoint_path))
                logger.info(f" Model checkpoint saved: {checkpoint_path}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    # Training completion
    total_training_time = time.time() - training_start_time
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total episodes: {episode + 1}")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Training time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")
    logger.info(f"Best episode reward: {best_episode_reward:.4f}")

    if episode_rewards:
        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        logger.info(f"Final average reward (last 100 episodes): {final_avg_reward:.4f}")

    # Save final model
    final_model_path = models_dir / "final_td3_model_5m.pth"
    agent.save(str(final_model_path))
    logger.info(f" Final model saved: {final_model_path}")

    # Save training history
    history_df = pd.DataFrame({
        'episode': range(1, len(episode_rewards) + 1),
        'reward': episode_rewards
    })
    history_path = models_dir / "training_history_5m.csv"
    history_df.to_csv(history_path, index=False)
    logger.info(f" Training history saved: {history_path}")

    logger.info("Training pipeline completed successfully! <�")


if __name__ == '__main__':
    # Configuration for 5-minute timeframe training
    run_training(
        data_file="BTCUSDT-5m_2023-01-01_to_2023-12-31_features.csv",
        num_episodes=50,
        batch_size=256,
        start_timesteps=50000,  # MODIFIED
        eval_freq=5,
        save_freq=20,
        max_timesteps=500000
    )