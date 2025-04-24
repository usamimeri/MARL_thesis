import torch
from env import EconomicEnv
import pytest
from utils import load_config
from network import MultiHeadActorCritic


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


@pytest.mark.firm
def test_firm_settlement(env, config):
    env.reset()
    worker_net = MultiHeadActorCritic(config["size"]["observation"]["worker"],
                                      config["size"]["action"]["worker"]).to(env.device)
    firm_net = MultiHeadActorCritic(config["size"]["observation"]["firm"],
                                    config["size"]["action"]["firm"]).to(env.device)
    worker_obs = env.construct_worker_obs()
    worker_action = worker_net.get_logprob_and_action(worker_obs)[2]
    firm_obs = env.construct_firm_obs()
    firm_action = firm_net.get_logprob_and_action(firm_obs)[2]
    env.worker_settlement(worker_action)
    print("劳动者工作企业")
    print(env.worker_in_firm)
    print("劳动者消费量")
    print(env.worker_consumption)
    print("劳动者报价")
    print(env.worker_quote)
    env.firm_settlement(firm_action)
    print("企业生产量")
    print(env.firm_production)
    print("企业报价")
    print(env.firm_quote)
    print("====================================")
    print("企业销售额")
    print(env.firm_sales)
    print("劳动者成本")
    print(env.worker_cost)
    assert env.firm_sales.sum() == env.worker_cost.sum()
