from buffer import RolloutBuffer
from env import EconomicEnv
from network import MultiHeadActorCritic
from utils import load_config, seed_everything
import torch
# 尝试rollout
config = load_config()
env = EconomicEnv()
num_steps = 20
worker_obs_dim = config["size"]["observation"]["worker"]
firm_obs_dim = config["size"]["observation"]["firm"]
government_obs_dim = config["size"]["observation"]["government"]
worker_action_dim = config["size"]["action"]["worker"]
firm_action_dim = config["size"]["action"]["firm"]
government_action_dim = config["size"]["action"]["government"]
device = config["device"]
worker_buffer = RolloutBuffer(buffer_size=num_steps, obs_dim=worker_obs_dim,
                              action_dim=len(worker_action_dim), num_agents=env.num_worker_agents)
firm_buffer = RolloutBuffer(buffer_size=num_steps, obs_dim=firm_obs_dim,
                            action_dim=len(firm_action_dim), num_agents=env.num_firm_agents)
government_buffer = RolloutBuffer(buffer_size=num_steps, obs_dim=government_obs_dim,
                                  action_dim=1, num_agents=1)
worker_net = MultiHeadActorCritic(worker_obs_dim, worker_action_dim).to(device)
firm_net = MultiHeadActorCritic(firm_obs_dim, firm_action_dim).to(device)
government_net = MultiHeadActorCritic(government_obs_dim, government_action_dim).to(device)

seed_everything(config["train"]["seed"])
num_updates = config["train"]["total_timesteps"] // num_steps

for epoch in range(1):
    env.reset()
    for step in range(num_steps):
        worker_obs = env.construct_worker_obs()
        with torch.no_grad():
            worker_logprobs, worker_entropy, worker_action = worker_net.get_logprob_and_action(
                torch.from_numpy(worker_obs).to(device))
            worker_value = worker_net.get_value(torch.from_numpy(worker_obs).to(device))
        env.worker_settlement(worker_action.cpu().numpy())
        firm_obs = env.construct_firm_obs()
        with torch.no_grad():
            firm_logprobs, firm_entropy, firm_action = firm_net.get_logprob_and_action(
                torch.from_numpy(firm_obs).to(device))
            firm_value = firm_net.get_value(torch.from_numpy(firm_obs).to(device))
        env.firm_settlement(firm_action.cpu().numpy())
        government_obs = env.construct_government_obs()
        with torch.no_grad():
            government_logprobs, government_entropy, government_action = government_net.get_logprob_and_action(
                torch.from_numpy(government_obs).to(device))
            government_value = government_net.get_value(torch.from_numpy(government_obs).to(device))
        env.government_settlement(government_action.cpu().numpy())
        worker_reward = env.worker_utility
        firm_reward = env.firm_pre_tax_profit
        government_reward = env.government_reward
        worker_buffer.add(worker_obs, worker_action.cpu().numpy(), worker_reward, worker_value, worker_logprobs)
        firm_buffer.add(firm_obs, firm_action.cpu().numpy(), firm_reward, firm_value, firm_logprobs)
        government_buffer.add(government_obs, government_action.cpu().numpy(), government_reward,
                              government_value, government_logprobs)
    with torch.no_grad():
        worker_values = worker_net.get_value(torch.from_numpy(worker_obs).to(device))
        firm_values = firm_net.get_value(torch.from_numpy(firm_obs).to(device))
        government_values = government_net.get_value(torch.from_numpy(government_obs).to(device))
    worker_buffer.compute_returns_and_advantage(last_values=worker_values)
    firm_buffer.compute_returns_and_advantage(last_values=firm_values)
    government_buffer.compute_returns_and_advantage(last_values=government_values)
