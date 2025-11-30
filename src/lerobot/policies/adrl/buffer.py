#!/usr/bin/env python
"""Simple replay buffer for ADRL that stores base_action."""

import torch


class ResidualReplayBuffer:
    """Replay buffer storing (state, base_action, action, reward, next_state, next_base_action, done)."""

    def __init__(self, capacity: int, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        self.initialized = False

    def _init_storage(self, state: torch.Tensor, action: torch.Tensor):
        """Initialize storage on first add."""
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]

        self.states = torch.zeros(self.capacity, state_dim)
        self.base_actions = torch.zeros(self.capacity, action_dim)
        self.actions = torch.zeros(self.capacity, action_dim)
        self.rewards = torch.zeros(self.capacity)
        self.next_states = torch.zeros(self.capacity, state_dim)
        self.next_base_actions = torch.zeros(self.capacity, action_dim)
        self.dones = torch.zeros(self.capacity)

        self.initialized = True

    def __len__(self):
        return self.size

    def add(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        next_base_action: torch.Tensor,
        done: bool,
    ):
        if not self.initialized:
            self._init_storage(state, action)

        self.states[self.position] = state.squeeze().cpu()
        self.base_actions[self.position] = base_action.squeeze().cpu()
        self.actions[self.position] = action.squeeze().cpu()
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state.squeeze().cpu()
        self.next_base_actions[self.position] = next_base_action.squeeze().cpu()
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (min(batch_size, self.size),))

        return {
            "state": self.states[idx].to(self.device),
            "base_action": self.base_actions[idx].to(self.device),
            "action": self.actions[idx].to(self.device),
            "reward": self.rewards[idx].to(self.device),
            "next_state": self.next_states[idx].to(self.device),
            "next_base_action": self.next_base_actions[idx].to(self.device),
            "done": self.dones[idx].to(self.device),
        }
