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
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
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
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([layer_init(nn.Linear(128, a_dim), std=0.01) for a_dim in self.action_dim])

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
        multi_probs = [F.softmax(head(x), dim=-1) for head in self.heads]
        actions = []
        sum_logprobs = 0
        sum_entropy = 0
        if action is None:
            for prob in multi_probs:
                categorial_prob = Categorical(prob)
                action = categorial_prob.sample()
                actions.append(action)
            # 例如一共2类动作，3个智能体，每次采样时是[3]，代表每个智能体做的动作
            # stack就需要在最后一维堆叠，变成[3,2]，第一维是智能体，第二维是每类动作
            action = torch.stack(actions, dim=-1)
        # 计算累加的对数概率

        for i, prob in enumerate(multi_probs):
            categorial_prob = Categorical(prob)
            # 根据动作，从每一行概率分布中找到对应的对数概率
            if len(action.shape) == 1:
                logprobs = categorial_prob.log_prob(action)
            else:
                logprobs = categorial_prob.log_prob(action[:, i])
            sum_logprobs += logprobs
            # 后面算的时候直接取mean就行
            sum_entropy += categorial_prob.entropy()
        return sum_logprobs, sum_entropy, action


class MultiHeadActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: list | int):
        super().__init__()
        self.actor = MultiHeadActor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    def get_logprob_and_action(self, state, action=None):
        return self.actor.get_logprob_and_action(state, action)

    def get_value(self, state):
        return self.critic.get_value(state)
