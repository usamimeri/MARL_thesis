from typing import NamedTuple, Literal, Optional, Generator
import torch
from utils import RunningMeanStd, load_config


class RolloutData(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
    def __init__(self, agent_type: Literal["worker", "firm", "government"]) -> None:
        self.config = load_config()
        self.rms = RunningMeanStd()
        self.agent_type = agent_type
        self.buffer_size = self.config['experiment']['buffer_size']
        self.gae_lambda = self.config['experiment']['gae_lambda']
        self.gamma = self.config['experiment']['gamma']
        self.device = self.config['device']
        self.num_agent = 1 if self.agent_type == "government" else self.config[f"num_{self.agent_type}_agents"]
        self.reset()

    def reset(self) -> None:
        self.observations = torch.zeros((self.buffer_size, self.num_agent, *self.obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((self.buffer_size, self.num_agent, self.action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((self.buffer_size, self.num_agent), dtype=torch.float32)
        self.returns = torch.zeros((self.buffer_size, self.num_agent), dtype=torch.float32)
        self.episode_starts = torch.zeros((self.buffer_size, self.num_agent), dtype=torch.float32)
        self.values = torch.zeros((self.buffer_size, self.num_agent), dtype=torch.float32)
        self.log_probs = torch.zeros((self.buffer_size, self.num_agent), dtype=torch.float32)
        self.advantages = torch.zeros((self.buffer_size, self.num_agent), dtype=torch.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def compute_gae(self, last_value):
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def normalize_rewards(self):
        pass

    def normalize_obs(self):
        pass

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
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
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: torch.Tensor,
    ) -> RolloutData:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutData(*data)

    def sample(self, batch_size: int):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper_bound, size=(batch_size,))
        return self._get_samples(batch_inds)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutData, None, None]:
        assert self.full, ""
        indices = torch.randperm(self.buffer_size * self.num_agent)
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
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size
