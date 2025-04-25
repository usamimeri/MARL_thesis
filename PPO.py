from buffer import RolloutBuffer
from utils import load_config
from network import MultiHeadActorCritic
import torch
import torch.nn.functional as F
import numpy as np


class PPO:
    def __init__(self, learning_rate: float, buffer: RolloutBuffer, network: MultiHeadActorCritic):
        self.config = load_config()
        self.clip_param = self.config["train"]["clip_param"]
        self.vf_coef = self.config["train"]["vf_coef"]
        self.entropy_coef_start = self.config["train"]["entropy_coef_start"]
        self.entropy_coef_end = self.config["train"]["entropy_coef_end"]
        self.ppo_update = self.config["train"]["ppo_update"]
        self.max_grad_norm = self.config["train"]["max_grad_norm"]
        self.buffer = buffer
        self.batch_size = buffer.buffer_size  # 一般是num_agent*num_steps
        self.n_minibatches = self.config["train"]["n_minibatches"]
        self.minibatch_size = self.batch_size // self.n_minibatches
        self.network = network
        self.device = network.device
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        self.ent_coef = self.config["train"]["ent_coef"]

        # LOG
        self.ent_losses = []
        self.pg_losses = []
        self.vf_losses = []
        self.approx_kls = []
        self.losses = []

    def update_buffer(self, buffer: RolloutBuffer):
        assert buffer.buffer_size == self.batch_size, f"新buffer的尺度{buffer.buffer_size}不匹配{self.buffer.buffer_size}"
        self.buffer = buffer

    def update(self):
        entropy_loss = []
        pg_loss = []
        vf_loss = []
        approx_kl = []
        loss = []

        # 完成一轮epoch的训练迭代，在一次ppo_update中，遍历每个minibatch
        for update_epoch in range(self.ppo_update):
            # 遍历minibatch
            for data in self.buffer.get(self.minibatch_size):
                logprobs, entropy, actions = self.network.get_logprob_and_action(
                    data.observations)
                values = self.network.get_value(data.observations)
                # 标准化优势函数
                advantages = data.advantages
                advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)

                # 新旧策略
                ratio = torch.exp(logprobs-data.old_log_prob)

                # 策略损失
                pg_loss1 = -advantages*ratio
                pg_loss2 = -advantages*torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 价值损失
                vf_loss = F.mse_loss(values, data.returns)
                entropy_loss = -entropy.mean()
                # 总损失
                loss = pg_loss+self.vf_coef*vf_loss+self.ent_coef*entropy_loss

                with torch.no_grad():
                    log_ratio = logprobs-data.old_log_prob
                    approx_kl = ((ratio-1)-log_ratio).mean()

                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 每个minibatch的损失
                entropy_loss.append(entropy_loss.item())
                pg_loss.append(pg_loss.item())
                vf_loss.append(vf_loss.item())
                approx_kl.append(approx_kl.item())
                loss.append(loss.item())

        # 一轮epoch的平均损失
        self.ent_losses.append(np.mean(entropy_loss))
        self.pg_losses.append(np.mean(pg_loss))
        self.vf_losses.append(np.mean(vf_loss))
        self.approx_kls.append(np.mean(approx_kl))
        self.losses.append(np.mean(loss))
