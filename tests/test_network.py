import pytest
import torch
from network import Critic, MultiHeadActor, MultiHeadActorCritic
from env import EconomicEnv
from utils import load_config


@pytest.fixture
def env():
    return EconomicEnv()


@pytest.fixture
def config():
    return load_config()

def test_critic_output_shape():
    state_dim = 8
    batch_size = 4
    critic = Critic(state_dim)
    state = torch.randn(batch_size, state_dim)
    value = critic.get_value(state)
    assert isinstance(value, torch.Tensor)
    assert value.shape == (batch_size, 1)


def test_multiheadactor_single_action_dim():
    state_dim = 5
    action_dim = 3
    batch_size = 6
    actor = MultiHeadActor(state_dim, action_dim)
    state = torch.randn(batch_size, state_dim)
    logprobs, entropy, action = actor.get_logprob_and_action(state)
    # Check shapes
    assert action.shape == (batch_size, 1)
    assert logprobs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)


def test_multiheadactor_multiple_action_dim():
    state_dim = 4
    action_dims = [2, 3, 4]
    batch_size = 5
    actor = MultiHeadActor(state_dim, action_dims)
    state = torch.randn(batch_size, state_dim)
    logprobs, entropy, action = actor.get_logprob_and_action(state)
    assert action.shape == (batch_size, len(action_dims))
    assert logprobs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)


def test_critic_three_dim_input():
    state_dim = 8
    batch_size = 3
    num_agents = 4
    critic = Critic(state_dim)
    state = torch.randn(batch_size, num_agents, state_dim)
    value = critic.get_value(state)
    assert isinstance(value, torch.Tensor)
    assert value.shape == (batch_size, num_agents, 1)


def test_multiheadactor_three_dim_input_single_action_dim():
    state_dim = 5
    action_dim = 3
    batch_size = 4
    num_agents = 6
    actor = MultiHeadActor(state_dim, action_dim)
    state = torch.randn(batch_size, num_agents, state_dim)
    logprobs, entropy, action = actor.get_logprob_and_action(state)
    assert action.shape == (batch_size, num_agents, 1)
    assert logprobs.shape == (batch_size, num_agents)
    assert entropy.shape == (batch_size, num_agents)


def test_multiheadactor_three_dim_input_multiple_action_dim():
    state_dim = 10
    action_dims = [2, 3, 4]
    batch_size = 3
    num_agents = 10
    actor = MultiHeadActor(state_dim, action_dims)
    state = torch.randn(batch_size, num_agents, state_dim)
    logprobs, entropy, action = actor.get_logprob_and_action(state)
    print(entropy)
    assert action.shape == (batch_size, num_agents, len(action_dims))
    assert logprobs.shape == (batch_size, num_agents)
    assert entropy.shape == (batch_size, num_agents)


def test_multiheadactor_critic_worker_output_shape(env, config):
    """测试能否正确输入劳动者观测，输出动作、对数概率和熵"""
    env.reset()
    obs = env.construct_worker_obs()
    actor_critic = MultiHeadActorCritic(config["size"]["observation"]["worker"], 
                                        config["size"]["action"]["worker"]).to(env.device)
    logprobs, entropy, action = actor_critic.get_logprob_and_action(obs)
    assert action.shape == (config["num_worker_agents"], len(config["size"]["action"]["worker"]))
    assert logprobs.shape == (config["num_worker_agents"],)
    assert entropy.shape == (config["num_worker_agents"],)