import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
"""
TODO: 记得看下runningmean和runningstd 这里是要normalization state的
"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, state):
        """
        输出(num_agent,1)
        """
        return self.net(state)


class MultiHeadActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: list | int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim if isinstance(action_dim, list) else [action_dim]
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([layer_init(nn.Linear(64, a_dim), std=0.01) for a_dim in self.action_dim])

    def get_logprob_and_action(self, state, action=None):
        """
        num_action=len(action_dim)
        state: (num_agent,state_dim)
        action: (num_agent,num_action)

        输出：
        logprobs: (num_agent,)
        entropy: (num_agent,)
        action: (num_agent,num_action)

        其中action的顺序为：消费，劳动，报价，工作企业
        """
        x = self.backbone(state)
        probs = [Categorical(logits=head(x)) for head in self.heads]
        if action is None:
            # 例如一共四个动作，则一个智能体是[4]，一般输入是(num_agent,state_dim)
            # 一个头输出是(num_agent,1)，因此在最后一维堆叠
            # 输出为(num_agent,num_action)
            action = torch.stack([prob.sample() for prob in probs], dim=-1)
        # 单个头的prob是(num_agent,action_size),由于MultiDiscrete所以是例如
        # (5,10)，(5,10)，(5,10)，(5,4)
        # 这里得到对应每个动作的对数概率，输出为(num_agent,num_action)
        logprobs = torch.stack([prob.log_prob(action[..., i]) for i, prob in enumerate(probs)], dim=-1)
        entropy = torch.stack([prob.entropy() for prob in probs], dim=-1)
        # \log\pi_{a|s}=log\pi_{a_1|s}+log\pi_{a_2|s}+...+log\pi_{a_n|s}
        logprobs = logprobs.sum(dim=-1)
        entropy = entropy.sum(dim=-1)
        return logprobs, entropy, action


class MultiHeadActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: list | int):
        super().__init__()
        self.actor = MultiHeadActor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    def get_logprob_and_action(self, state, action=None):
        return self.actor.get_logprob_and_action(state, action)

    def get_value(self, state):
        return self.critic.get_value(state)
