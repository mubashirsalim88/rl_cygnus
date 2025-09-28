import pandas as pd
import numpy as np
import logging
import time
import os

from components.environment import TradingEnvironment
from components.agent import DuelingDDQNAgent
from components.replay_buffer import ReplayBuffer

# --- CONFIGURATION ---
LOG_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "filename": "logs/training.log"
}

def setup_logging():
    """Set up the logging environment."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(**LOG_CONFIG)
    return logging.getLogger(__name__)

def run_training(
    data_file: str,
    num_episodes: int = 50,
    batch_size: int = 256,
    start_timesteps: int = 25000,
    max_timesteps: int = 1_000_000,
    save_freq: int = 10,
    logger: logging.Logger = None
):
    """Main training loop for the Dueling DDQN agent."""
    if not logger:
        logger = setup_logging()

    logger.info("================================================================================")
    logger.info("STARTING NEW TRAINING RUN - ALGORITHM: DUELING DOUBLE DQN")
    logger.info("================================================================================")

    # --- ENVIRONMENT AND AGENT INITIALIZATION ---
    try:
        df = pd.read_csv(f"data/processed/{data_file}")
        logger.info(f"Loaded data: {data_file}, shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file}. Please run feature engineering script.")
        return

    env = TradingEnvironment(df)
    state_dim = env.n_features
    action_dim = 3  # BUY (1), HOLD (0), SELL (2)

    agent = DuelingDDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-4,
        discount=0.99,
        tau=0.005
    )

    # Replay buffer action_dim can be 1, as we store a single integer action
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=1, max_size=int(1e6))

    # --- EPSILON-GREEDY EXPLORATION SCHEDULE ---
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 500000

    # --- TRAINING LOOP ---
    start_time = time.time()
    total_timesteps = 0
    best_episode_reward = -np.inf
    training_history = []

    logger.info(f"Starting training for {num_episodes} episodes or {max_timesteps} timesteps.")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done and total_timesteps < max_timesteps:
            # Action selection
            if total_timesteps < start_timesteps:
                action = np.random.randint(action_dim)
            else:
                epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (total_timesteps / epsilon_decay_steps))
                action = agent.select_action(state, epsilon)

            # Environment step
            next_state, reward, done, info = env.step(action)

            # Store transition in replay buffer (action is an integer)
            replay_buffer.add(state, np.array([action]), next_state, reward, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1

            # Train agent
            if total_timesteps >= start_timesteps and replay_buffer.is_ready(batch_size):
                metrics = agent.train(replay_buffer, batch_size)
                if total_timesteps % 5000 == 0:
                    logger.info(f"Step {total_timesteps}: Training Loss={metrics['loss']:.4f}, Epsilon={epsilon:.3f}")

        # --- LOGGING AND SAVING ---
        portfolio_value = info.get('portfolio_value', 0)
        market_return = info.get('market_return', 0)
        agent_return = info.get('agent_return', 0)

        logger.info(
            f"Ep {episode}/{num_episodes} | Steps: {episode_steps:5d} | Reward: {episode_reward:8.2f} | "
            f"Portfolio: ${portfolio_value:10.2f} | Return: {agent_return:7.2f}% | Mkt Return: {market_return:7.2f}%"
        )
        training_history.append({
            "episode": episode, "reward": episode_reward, "steps": episode_steps,
            "portfolio_value": portfolio_value, "agent_return": agent_return,
            "total_timesteps": total_timesteps
        })

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            agent.save("models/best_ddqn_model.pth")
            logger.info(f"*** New best model saved with reward: {best_episode_reward:.2f} ***")

        if episode % save_freq == 0:
            agent.save(f"models/ddqn_checkpoint_ep{episode}.pth")

        if total_timesteps >= max_timesteps:
            logger.info(f"Reached max timesteps ({max_timesteps}). Terminating training.")
            break

    # --- END OF TRAINING ---
    end_time = time.time()
    total_duration_m = (end_time - start_time) / 60

    logger.info("================================================================================")
    logger.info(f"TRAINING COMPLETED in {total_duration_m:.2f} minutes")
    logger.info("================================================================================")

    agent.save("models/final_ddqn_model.pth")
    history_df = pd.DataFrame(training_history)
    history_df.to_csv("models/training_history_ddqn.csv", index=False)

    logger.info("Final model and training history saved successfully.")

if __name__ == '__main__':
    logger = setup_logging()
    run_training(
        data_file="BTCUSDT-5m_2023-01-01_to_2023-12-31_features.csv",
        num_episodes=100,
        batch_size=256,
        start_timesteps=50000,
        max_timesteps=1_000_000,
        save_freq=10,
        logger=logger
    )