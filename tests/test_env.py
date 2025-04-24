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
