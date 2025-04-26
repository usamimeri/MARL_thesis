from buffer import RolloutBuffer
from env import EconomicEnv
from network import MultiHeadActorCritic
from utils import load_config, seed_everything
import torch
import numpy as np
import pandas as pd

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

worker_net = MultiHeadActorCritic(worker_obs_dim, worker_action_dim).to(device)
firm_net = MultiHeadActorCritic(firm_obs_dim, firm_action_dim).to(device)
government_net = MultiHeadActorCritic(government_obs_dim, government_action_dim).to(device)


seed_everything(config["train"]["seed"])
env.logger.enabled = True
model_path = "/home/usamimeri/MARL_thesis/models/04-26_17-15"
worker_net.load_state_dict(torch.load(f"{model_path}/worker_net_final.pth"))
firm_net.load_state_dict(torch.load(f"{model_path}/firm_net_final.pth"))
government_net.load_state_dict(torch.load(f"{model_path}/government_net_final.pth"))
env.reset()

path = "./data"
social_efficiency = []
equality = []
government_reward = []
worker_reward = []
firm_reward = []
worker_asset = []
firm_asset = []
tax_rate = []
worker_in_firm = []
worker_labor = []

vector_data = [social_efficiency, equality, tax_rate, government_reward]


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

    social_efficiency.append(env.social_efficiency)
    equality.append(env.equality)
    government_reward.append(env.government_reward)
    worker_reward.append(env.worker_utility)
    firm_reward.append(env.firm_reward)
    worker_asset.append(env.worker_asset)
    firm_asset.append(env.firm_asset)
    tax_rate.append(env.tax_rate)
    worker_in_firm.append(env.worker_in_firm)
    worker_labor.append(env.worker_labor)


vector_data = np.stack(vector_data, axis=1)
pd.DataFrame(data=vector_data, columns=["social_efficiency", "equality", "tax_rate",
             "government_reward"]).to_csv(f"{path}/vector_data.csv", )


pd.DataFrame(data=worker_reward).to_csv(f"{path}/worker_reward.csv", )
pd.DataFrame(data=firm_reward).to_csv(f"{path}/firm_reward.csv", )
pd.DataFrame(data=worker_asset).to_csv(f"{path}/worker_asset.csv", )
pd.DataFrame(data=firm_asset).to_csv(f"{path}/firm_asset.csv", )
pd.DataFrame(data=worker_in_firm).to_csv(f"{path}/worker_in_firm.csv", )
pd.DataFrame(data=worker_labor).to_csv(f"{path}/worker_labor.csv", )
