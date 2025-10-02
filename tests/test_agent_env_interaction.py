import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from src.components.environment import TradingEnvironment
from src.components.agent_ppo import PPOAgent

@pytest.fixture
def sample_env():
    """Creates a sample TradingEnvironment for integration testing."""
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=20)),
        'close': [100.0] * 20,
        'feature1': [0.5] * 20
    }
    df = pd.DataFrame(data)
    return TradingEnvironment(df=df)

def test_agent_and_env_step_without_error(sample_env):
    """
    Tests that the agent can select an action and the environment can
    process it for one step without raising an error.
    This is a critical integration test.
    """
    env = sample_env
    state_dim = env.n_features
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim, device='cpu')

    state = env.reset()

    try:
        # Agent selects an action (returns int, numpy array)
        action, log_prob = agent.select_action(state)

        # Environment processes the action (expects int)
        env.step(action)
    except Exception as e:
        pytest.fail(f"Agent-Environment interaction failed with error: {e}")
