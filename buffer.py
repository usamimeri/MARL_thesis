from typing import NamedTuple, Optional, Generator
import numpy as np
from utils import RunningMeanStd, load_config
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union
import os

import numpy as np
import torch as th
import pandas as pd

config = load_config()
mapping = {config["num_worker_agents"]: "worker",
           config["num_firm_agents"]: "firm",
           1: "government"}


class RolloutData(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        num_agents: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = load_config()
        self.pos = 0
        self.full = False
        self.device = self.config["device"]
        self.num_agents = num_agents
        self._last_episode_starts = None

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (num_agents)
        to convert shape from [n_steps, num_agents, ...] (when ... is the shape of the features)
        to [n_steps * num_agents, ...] (which maintain the order)

        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray
    ) -> Union[RolloutData]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(obs: np.ndarray):
        pass

    @staticmethod
    def _normalize_reward(reward: np.ndarray):
        pass


class RolloutBuffer(BaseBuffer):
    """要计算优势函数的话，需要在step运行完后，传入最后的值函数值
    即跳出for i in range(self.num_steps)后，进行↓
    with th.no_grad():
        # Compute value for the last timestep
        values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  

    rollout_buffer.compute_returns_and_advantage(last_values=values)

    """
    observations: np.ndarray  # (num_agent,state_dim)
    actions: np.ndarray  # (num_agent,action_dim)
    rewards: np.ndarray  # (num_agent)
    advantages: np.ndarray  # (num_agent)
    returns: np.ndarray  # (num_agent)
    log_probs: np.ndarray  # (num_agent)
    values: np.ndarray  # (num_agent)
    episode_starts: np.ndarray  # (num_agent) 标志是否为结束状态

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        num_agents: int = 1,
    ):
        super().__init__(buffer_size, obs_dim, action_dim, num_agents=num_agents)
        self.config = load_config()
        self.gae_lambda = self.config["train"]["lambda"]
        self.gamma = self.config["train"]["gamma"]
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_agents, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self._last_episode_starts = np.ones((self.num_agents,), dtype=bool)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor,  dones: np.ndarray) -> None:
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        episode_start: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param episode_start: 标志是否为结束状态
        """

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutData, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.num_agents)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.num_agents

        start_idx = 0
        while start_idx < self.buffer_size * self.num_agents:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> RolloutData:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),  # [num_agent,1]->[num_agent]
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutData(*tuple(map(self.to_torch, data)))

    def save(self, to_csv=True):
        """保存要用于训练的数据到本地，便于观察"""
        prefix = mapping.get(self.num_agents, "Unknown")
        os.makedirs(f"./data/{prefix}", exist_ok=True)
        path = f"./data/{prefix}"
        if to_csv:
            # 二维的数据专门放一个csv，TODO:由于obs和action三维，这里后面再实现
            # obs和action的数据目前先保存起始，中间，最终步的数据
            start_obs = pd.DataFrame(self.observations[0]).T.round(2)
            mid_obs = pd.DataFrame(self.observations[self.buffer_size//2]).T.round(2)
            final_obs = pd.DataFrame(self.observations[-1]).T.round(2)
            reward_df = pd.DataFrame(self.rewards).round(2)
            return_df = pd.DataFrame(self.returns).round(2)
            start_obs.to_csv(os.path.join(path, f"{prefix}_start_obs.csv"), index=False)
            mid_obs.to_csv(os.path.join(path, f"{prefix}_mid_obs.csv"), index=False)
            final_obs.to_csv(os.path.join(path, f"{prefix}_final_obs.csv"), index=False)
            reward_df.to_csv(os.path.join(path, f"{prefix}_reward.csv"), index=False)
            return_df.to_csv(os.path.join(path, f"{prefix}_return.csv"), index=False)
        else:
            raise NotImplementedError("暂时只支持保存为csv")
        return True


if __name__ == "__main__":
    buffer = RolloutBuffer(buffer_size=5, obs_dim=4, action_dim=2, num_agents=2)
    for _ in range(5):
        buffer.add(obs=np.random.randn(2, 4), action=np.random.randn(
            2, 2), reward=np.random.randint(0, 10, size=2), value=th.randn(2), log_prob=th.randn(2), episode_start=np.zeros(2))

    last_values = th.randn(2)
    buffer.compute_returns_and_advantage(last_values=last_values, dones=np.zeros(2))
    buffer.save(to_csv=True)
    # for data in buffer.get(batch_size=2):
    #     for j in [data.observations, data.actions, data.old_values, data.old_log_prob, data.advantages, data.returns]:
    #         print(j)
    #         print("-"*100)
    #     break
