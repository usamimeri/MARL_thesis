from buffer import RolloutBuffer
from env import EconomicEnv
from network import MultiHeadActorCritic
from utils import load_config, seed_everything, wandb_log, init_wandb
import torch
from ppo import PPO
import numpy as np
import wandb
import os
from datetime import datetime
init_wandb()
np.set_printoptions(suppress=True)

config = load_config()
env = EconomicEnv()
num_steps = config["train"]["num_steps"]
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

ppo_worker = PPO(config["train"]["worker_lr"], worker_buffer, worker_net)
ppo_firm = PPO(config["train"]["firm_lr"], firm_buffer, firm_net)
ppo_government = PPO(config["train"]["government_lr"], government_buffer, government_net)

seed_everything(config["train"]["seed"])
num_updates = config["train"]["total_timesteps"] // num_steps


for epoch in range(num_updates):
    print(f"epoch: {epoch}")
    env.reset()
    ppo_worker.buffer.reset()
    ppo_firm.buffer.reset()
    ppo_government.buffer.reset()
    # 经济模拟没有dones
    ppo_worker.buffer._last_episode_starts = np.zeros(env.num_worker_agents)
    ppo_firm.buffer._last_episode_starts = np.zeros(env.num_firm_agents)
    ppo_government.buffer._last_episode_starts = np.zeros(1)

    for step in range(num_steps):
        # 构造劳动者观测和获取动作
        worker_obs = env.construct_worker_obs()
        with torch.no_grad():
            worker_logprobs, worker_entropy, worker_action = worker_net.get_logprob_and_action(
                torch.from_numpy(worker_obs).to(device))
            worker_value = worker_net.get_value(torch.from_numpy(worker_obs).to(device))
            worker_action = worker_action.cpu().numpy()
        env.worker_settlement(worker_action)

        # 构造企业观测和获取动作
        firm_obs = env.construct_firm_obs()
        with torch.no_grad():
            firm_logprobs, firm_entropy, firm_action = firm_net.get_logprob_and_action(
                torch.from_numpy(firm_obs).to(device))
            firm_value = firm_net.get_value(torch.from_numpy(firm_obs).to(device))
            firm_action = firm_action.cpu().numpy()
        env.firm_settlement(firm_action)

        # 构造政府观测和获取动作
        government_obs = env.construct_government_obs()
        with torch.no_grad():
            government_logprobs, government_entropy, government_action = government_net.get_logprob_and_action(
                torch.from_numpy(government_obs).to(device))
            government_value = government_net.get_value(torch.from_numpy(government_obs).to(device))
            government_action = government_action.cpu().numpy()
        env.government_settlement(government_action)

        worker_reward = env.worker_utility
        firm_reward = env.firm_reward
        government_reward = env.government_reward

        ppo_worker.buffer.add(worker_obs,
                              worker_action,
                              worker_reward,
                              worker_value,
                              worker_logprobs,
                              np.zeros(env.num_worker_agents))
        ppo_firm.buffer.add(firm_obs,
                            firm_action,
                            firm_reward,
                            firm_value,
                            firm_logprobs,
                            np.zeros(env.num_firm_agents))
        ppo_government.buffer.add(government_obs,
                                  government_action,
                                  government_reward,
                                  government_value,
                                  government_logprobs,
                                  np.zeros(1))

    with torch.no_grad():
        worker_values = worker_net.get_value(torch.from_numpy(worker_obs).to(device))
        firm_values = firm_net.get_value(torch.from_numpy(firm_obs).to(device))
        government_values = government_net.get_value(torch.from_numpy(government_obs).to(device))
    worker_buffer.compute_returns_and_advantage(last_values=worker_values, dones=np.zeros(env.num_worker_agents))
    firm_buffer.compute_returns_and_advantage(last_values=firm_values, dones=np.zeros(env.num_firm_agents))
    government_buffer.compute_returns_and_advantage(last_values=government_values, dones=np.zeros(1))

    # 必须在这里log 不然get后会变化折叠buffer 就不代表最后一步了
    # 这里log的是最后一步的return
    wandb.log({"worker/worker_return": ppo_worker.buffer.returns[-1].mean(),
               "firm/firm_return": ppo_firm.buffer.returns[-1].mean(),
               "government/return": ppo_government.buffer.returns[-1][0]})

    # =====================================训练阶段=====================================
    worker_pg_loss_mean, worker_vf_loss_mean, worker_approx_kl_mean, worker_loss_mean = ppo_worker.update()
    firm_pg_loss_mean, firm_vf_loss_mean, firm_approx_kl_mean, firm_loss_mean = ppo_firm.update()
    government_pg_loss_mean, government_vf_loss_mean, government_approx_kl_mean, government_loss_mean = ppo_government.update()

    wandb_log("worker", worker_pg_loss_mean, worker_vf_loss_mean,
              worker_approx_kl_mean, worker_loss_mean)
    wandb_log("firm", firm_pg_loss_mean, firm_vf_loss_mean,
              firm_approx_kl_mean, firm_loss_mean)
    wandb_log("government", government_pg_loss_mean, government_vf_loss_mean,
              government_approx_kl_mean, government_loss_mean)
    wandb.log({"social_efficiency": env.social_efficiency, "equality": env.equality})
    for i in range(env.num_firm_agents):
        wandb.log({f"firm/firm_{i+1}_asset": env.firm_asset[i]})

    for i in range(env.num_worker_agents):
        wandb.log({f"worker/worker_{i+1}_asset": env.worker_asset[i]})

    wandb.log({"government/tax_rate": env.tax_rate, "government/total_transfer": env.total_transfer})
    # =====================================训练结束=====================================
# 最终模型
modelpath = f"models/{datetime.now().strftime('%m-%d_%H-%M')}"
os.makedirs(modelpath, exist_ok=True)
torch.save(worker_net.state_dict(), f"{modelpath}/worker_net_final.pth")
torch.save(firm_net.state_dict(), f"{modelpath}/firm_net_final.pth")
torch.save(government_net.state_dict(), f"{modelpath}/government_net_final.pth")
