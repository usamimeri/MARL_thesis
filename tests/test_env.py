import torch
from env import EconomicEnv
import pytest
from utils import load_config


@pytest.fixture
def env():
    return EconomicEnv()


@pytest.fixture
def config():
    return load_config()


def test_compute_firm_labor_simple(env):
    # set number of firms and workers manually for the test
    env.num_firm_agents = 3
    env.worker_in_firm = torch.tensor([0, 1, 1, 2, 0], dtype=torch.long)
    env.worker_labor = torch.tensor([100.0, 200.0, 300.0, 400.0, 100.0])
    expected = torch.tensor([200.0, 500.0, 400.0])
    result = env.compute_firm_labor()
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_firm_labor_zero(env):
    env.num_firm_agents = 2
    env.worker_in_firm = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    env.worker_labor = torch.zeros(4)
    expected = torch.tensor([0.0, 0.0])
    result = env.compute_firm_labor()
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_worker_obs_shape(env, config):
    env.reset()
    obs = env.construct_worker_obs()
    assert obs.shape == (env.num_worker_agents, config["size"]["observation"]["worker"])


def test_firm_obs_shape(env, config):
    env.reset()
    obs = env.construct_firm_obs()
    assert obs.shape == (env.num_firm_agents, config["size"]["observation"]["firm"])


def test_government_obs_shape(env, config):
    env.reset()
    obs = env.construct_government_obs()
    assert obs.shape == (1, config["size"]["observation"]["government"])


@pytest.mark.market
def test_market_clearing(env):
    env.reset()
    env.worker_consumption = torch.tensor([20, 10, 5], dtype=torch.float32).to(env.device)  # num_worker
    env.worker_quote = torch.tensor([100, 90, 80], dtype=torch.float32).to(env.device)  # num_worker
    env.firm_production = torch.tensor([10, 15, 10], dtype=torch.float32).to(env.device)  # num_firm
    env.firm_quote = torch.tensor([70, 75, 120], dtype=torch.float32).to(env.device)  # num_firm
    env.market_clearing()
    print(env.firm_sales)
    print(env.worker_cost)
    print(env.worker_consumption)
    assert env.firm_sales.sum() == env.worker_cost.sum()
