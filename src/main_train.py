# In src/main_train.py

import pandas as pd
import logging
import time
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import git

from components.environment import TradingEnvironment
from components.agent_ppo import PPOAgent

# --- HYDRA + MLFLOW APP ---

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function orchestrated by Hydra.
    All parameters are sourced from the `cfg` object.
    """
    # --- MLFLOW SETUP ---
    mlflow.set_experiment(cfg.project.name)

    with mlflow.start_run() as run:
        # --- LOGGING SETUP ---
        # MLflow will manage logging, so we don't need a separate file handler
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__name__)
        logger.info("========================================================")
        logger.info("STARTING NEW TRAINING RUN - HYDRA + MLFLOW")
        logger.info(f"Experiment: {cfg.project.name}, Run ID: {run.info.run_id}")
        logger.info("========================================================")

        # Log all hyperparameters from the config file
        def flatten_dict(d, parent_key='', sep='.'):
            """Flatten a nested dictionary for MLflow logging."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        if isinstance(config_dict, dict):
            flat_params = flatten_dict(config_dict)
            mlflow.log_params(flat_params)
        else:
            logger.warning("Config could not be converted to dictionary for MLflow logging")

        # Log git commit hash for reproducibility
        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.object.hexsha
            mlflow.log_param("git_commit_hash", commit_hash)
            logger.info(f"Git Commit Hash: {commit_hash}")
        except git.InvalidGitRepositoryError:
            logger.warning("Not a git repository. Skipping commit hash logging.")
            mlflow.log_param("git_commit_hash", "N/A")

        # --- DATA & ENVIRONMENT SETUP ---
        data_path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'processed', cfg.environment.data_file)
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        env = TradingEnvironment(
            df,
            initial_balance=cfg.environment.initial_balance,
            commission_rate=cfg.environment.commission_rate,
            profit_bonus=cfg.environment.profit_bonus,
            drawdown_penalty=cfg.environment.drawdown_penalty,
            reward_profit_bonus_weight=cfg.environment.reward_profit_bonus_weight,
            reward_drawdown_penalty_weight=cfg.environment.reward_drawdown_penalty_weight
        )
        state_dim = env.n_features
        action_dim = int(env.action_space.n)  # Convert to Python int

        # --- AGENT INITIALIZATION ---
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=cfg.agent.lr,
            gamma=cfg.agent.gamma,
            K_epochs=cfg.agent.K_epochs,
            eps_clip=cfg.agent.eps_clip,
            device=cfg.training.device
        )

        # --- ON-POLICY TRAINING LOOP ---
        logger.info("Starting training loop...")
        start_time = time.time()
        timestep_counter = 0
        memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}

        for episode in range(1, cfg.training.num_episodes + 1):
            state = env.reset()
            episode_reward = 0

            while True:
                timestep_counter += 1
                action, log_prob = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                memory['states'].append(state)
                memory['actions'].append(action)
                memory['log_probs'].append(log_prob)
                memory['rewards'].append(reward)
                memory['dones'].append(done)

                state = next_state
                episode_reward += reward

                if timestep_counter % 500 == 0:
                    logger.info(
                        f"    Step {env.current_step} | "
                        f"Action: {action:.2f} | "
                        f"Portfolio Value: ${info.get('portfolio_value', 0):.2f}"
                    )

                if timestep_counter % cfg.training.update_timestep == 0:
                    agent.train(memory)
                    memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}

                if done:
                    break

            # Log metrics to MLflow
            mlflow.log_metric("episode_reward", episode_reward, step=episode)
            mlflow.log_metric("final_portfolio_value", info.get('portfolio_value', 0), step=episode)
            mlflow.log_metric("agent_return_percent", info.get('agent_return', 0), step=episode)

            logger.info(
                f"Ep {episode}/{cfg.training.num_episodes} | Reward: {episode_reward:8.2f} | "
                f"Portfolio: ${info.get('portfolio_value', 0):10.2f} | Return: {info.get('agent_return', 0):7.2f}%"
            )

            if episode % cfg.training.model_save_freq == 0:
                # Ensure the directory exists
                os.makedirs("checkpoints", exist_ok=True)
                checkpoint_path = f"checkpoints/ppo_checkpoint_ep{episode}.pth"
                agent.save(checkpoint_path)
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
                os.remove(checkpoint_path) # Clean up local file

        end_time = time.time()
        training_duration_minutes = (end_time - start_time) / 60
        mlflow.log_metric("training_duration_minutes", training_duration_minutes)
        logger.info(f"TRAINING COMPLETED in {training_duration_minutes:.2f} minutes")

        # Save and log the final model as an artifact
        final_model_path = "final_ppo_model.pth"
        agent.save(final_model_path)
        mlflow.log_artifact(final_model_path, artifact_path="model")
        os.remove(final_model_path)

        logger.info("="*60)
        logger.info("MLflow Run Completed. View results with 'mlflow ui'")
        logger.info("="*60)

if __name__ == '__main__':
    main()