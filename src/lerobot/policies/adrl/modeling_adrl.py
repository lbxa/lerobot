#!/usr/bin/env python
"""
ADRL (Adaptive Distributional Residual Learning) - Model Implementation.

This module implements the core ADRL components:
- GatedGaussianActor: Stochastic actor with learnable gate for residual learning
- QuantileCritic: Distributional critic using quantile regression
- BaseResidualAgent: Abstract interface for residual RL agents
- DSACGatedAgent: Distributional SAC with gated actor implementation
- ADRLPolicy: LeRobot policy wrapper for deployment
"""

import copy
import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from lerobot.policies.adrl.configuration_adrl import ADRLConfig
from lerobot.policies.pretrained import PreTrainedPolicy


# ==============================================================================
# Network Building Blocks
# ==============================================================================


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ==============================================================================
# GatedGaussianActor
# ==============================================================================


class GatedGaussianActor(nn.Module):
    """
    Gated Gaussian Actor for residual learning.

    Outputs:
    - mu: Mean of Gaussian distribution for the residual action
    - log_std: Log standard deviation of the Gaussian distribution
    - gate: Sigmoid value (0-1) indicating how much to intervene

    The gate allows the network to say "I am uncertain, let base handle this"
    (gate -> 0) or "I should correct the base policy" (gate -> 1).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Input: State + Base Action (for contextual residual)
        input_dim = state_dim + action_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: Tensor, base_action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor (batch, state_dim)
            base_action: Base policy action (batch, action_dim)

        Returns:
            mu: Mean of Gaussian (batch, action_dim)
            log_std: Log standard deviation (batch, action_dim)
            gate: Gate value in [0, 1] (batch, 1)
        """
        x = torch.cat([state, base_action], dim=-1)
        x = self.trunk(x)

        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        gate = torch.sigmoid(self.gate_head(x))

        return mu, log_std, gate

    def sample(
        self, state: Tensor, base_action: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Sample action from the policy.

        Args:
            state: State tensor
            base_action: Base policy action
            deterministic: If True, return mean; otherwise sample

        Returns:
            action: Tanh-squashed action
            log_prob: Log probability
            gate: Gate value
        """
        mu, log_std, gate = self.forward(state, base_action)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mu)
            log_prob = torch.zeros(mu.shape[0], device=mu.device)
        else:
            dist = Normal(mu, std)
            z = dist.rsample()  # Reparameterization trick
            action = torch.tanh(z)
            # Log prob with tanh correction
            log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)

        return action, log_prob, gate


# ==============================================================================
# QuantileCritic
# ==============================================================================


class QuantileCritic(nn.Module):
    """
    Quantile Critic for distributional RL.

    Instead of outputting a single Q-value, outputs `num_quantiles` values
    representing different quantiles of the return distribution.
    Critical for robotics where outcomes are often binary (success/fail).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_quantiles: int = 25,
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.net = MLP(state_dim + action_dim, hidden_dim, num_quantiles)

        # Precompute quantile midpoints (tau values)
        self.register_buffer(
            "tau",
            torch.linspace(0, 1, num_quantiles + 1)[1:] - 0.5 / num_quantiles,
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Returns quantile values (batch, num_quantiles)."""
        return self.net(torch.cat([state, action], dim=-1))


def quantile_huber_loss(pred: Tensor, target: Tensor, tau: Tensor) -> Tensor:
    """
    Quantile Huber loss for distributional RL.

    Combines asymmetric quantile loss with Huber loss for outlier robustness.
    """
    td_error = target.unsqueeze(1) - pred.unsqueeze(2)
    huber = torch.where(td_error.abs() <= 1, 0.5 * td_error.pow(2), td_error.abs() - 0.5)
    quantile_weight = torch.abs(tau.view(1, -1, 1) - (td_error < 0).float())
    return (quantile_weight * huber).sum(dim=2).mean()


# ==============================================================================
# Base Residual Agent Interface
# ==============================================================================


class BaseResidualAgent(ABC):
    """
    Abstract base class for residual RL agents.

    This interface allows seamless switching between different residual
    learning algorithms without changing robot interaction code.
    """

    def __init__(self, config: ADRLConfig):
        self.config = config

    @abstractmethod
    def get_action(
        self, state: Tensor, base_action: Tensor, eval_mode: bool = False
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """
        Get residual action given state and base action.

        Returns:
            residual: Scaled residual correction
            gate: Gate value indicating intervention magnitude
            info: Additional info (e.g., log_prob)
        """
        pass

    @abstractmethod
    def update(self, batch: dict[str, Tensor], step_count: int = 0) -> dict[str, float]:
        """Update agent given a batch of transitions. Returns loss metrics."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent state to file."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent state from file."""
        pass


# ==============================================================================
# DSAC Gated Agent
# ==============================================================================


class DSACGatedAgent(BaseResidualAgent):
    """
    Distributional SAC with Gated Actor for residual learning.

    Key features:
    - Quantile regression for distributional critic (handles sparse rewards)
    - Gated stochastic actor (learns when and how much to intervene)
    - Automatic entropy tuning
    """

    def __init__(self, config: ADRLConfig, device: str | None = None):
        super().__init__(config)
        self.device = device or config.device

        state_dim = config.state_dim
        action_dim = config.action_dim

        # Actor
        self.actor = GatedGaussianActor(
            state_dim,
            action_dim,
            config.hidden_dim,
            config.actor_config.log_std_min,
            config.actor_config.log_std_max,
        ).to(self.device)

        # Critics
        self.critic1 = QuantileCritic(state_dim, action_dim, config.hidden_dim, config.num_quantiles).to(
            self.device
        )
        self.critic2 = QuantileCritic(state_dim, action_dim, config.hidden_dim, config.num_quantiles).to(
            self.device
        )

        # Target critics
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Temperature
        self.log_alpha = torch.tensor([math.log(config.alpha)], device=self.device, requires_grad=True)
        self.target_entropy = config.target_entropy or -float(action_dim)

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.learning_rate,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)

        # Store config values
        self.gamma = config.gamma
        self.polyak_tau = config.polyak_tau
        self.max_residual_scale = config.max_residual_scale
        self.learnable_gate = config.learnable_gate

    @property
    def alpha(self) -> float:
        """Current temperature value."""
        return self.log_alpha.exp().item()

    def get_action(
        self, state: Tensor, base_action: Tensor, eval_mode: bool = False
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """
        Get residual action.

        The final residual is: gate * tanh(sampled_action) * max_residual_scale
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if base_action.dim() == 1:
            base_action = base_action.unsqueeze(0)

        state = state.to(self.device)
        base_action = base_action.to(self.device)

        with torch.no_grad():
            action, log_prob, gate = self.actor.sample(state, base_action, deterministic=eval_mode)
            if self.learnable_gate:
                residual = gate * action * self.max_residual_scale
            else:
                residual = action * self.max_residual_scale

        return residual, gate, {"log_prob": log_prob}

    def update(self, batch: dict[str, Tensor], step_count: int = 0) -> dict[str, float]:
        """Update actor, critics, and temperature."""
        state = batch["state"].to(self.device)
        base_action = batch["base_action"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device).unsqueeze(-1)
        next_state = batch["next_state"].to(self.device)
        next_base_action = batch["next_base_action"].to(self.device)
        done = batch["done"].to(self.device).unsqueeze(-1)

        # === Critic Update ===
        with torch.no_grad():
            next_action, next_log_prob, next_gate = self.actor.sample(next_state, next_base_action)
            if self.learnable_gate:
                next_residual = next_gate * next_action * self.max_residual_scale
            else:
                next_residual = next_action * self.max_residual_scale
            next_full_action = next_base_action + next_residual

            target_q1 = self.critic1_target(next_state, next_full_action)
            target_q2 = self.critic2_target(next_state, next_full_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
            target_q = reward + (1 - done) * self.gamma * target_q

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic_loss = quantile_huber_loss(q1, target_q, self.critic1.tau) + quantile_huber_loss(
            q2, target_q, self.critic2.tau
        )

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # === Actor Update ===
        new_action, new_log_prob, new_gate = self.actor.sample(state, base_action)
        if self.learnable_gate:
            new_residual = new_gate * new_action * self.max_residual_scale
        else:
            new_residual = new_action * self.max_residual_scale
        new_full_action = base_action + new_residual

        q1_new = self.critic1(state, new_full_action)
        q2_new = self.critic2(state, new_full_action)
        min_q = torch.min(q1_new, q2_new).mean(dim=-1)

        actor_loss = (self.alpha * new_log_prob - min_q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # === Alpha Update ===
        alpha_loss = -(self.log_alpha * (new_log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # === Target Update ===
        with torch.no_grad():
            for p, p_targ in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                p_targ.data.mul_(1 - self.polyak_tau).add_(self.polyak_tau * p.data)
            for p, p_targ in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                p_targ.data.mul_(1 - self.polyak_tau).add_(self.polyak_tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss.item(),
            "mean_gate": new_gate.mean().item(),
            "mean_q": min_q.mean().item(),
        }

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic1_target": self.critic1_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "log_alpha": self.log_alpha,
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic1.load_state_dict(data["critic1"])
        self.critic2.load_state_dict(data["critic2"])
        self.critic1_target.load_state_dict(data["critic1_target"])
        self.critic2_target.load_state_dict(data["critic2_target"])
        self.log_alpha.data.copy_(data["log_alpha"])
        self.actor_opt.load_state_dict(data["actor_opt"])
        self.critic_opt.load_state_dict(data["critic_opt"])
        self.alpha_opt.load_state_dict(data["alpha_opt"])


# ==============================================================================
# ADRLPolicy (LeRobot Policy Wrapper)
# ==============================================================================


class ADRLPolicy(PreTrainedPolicy):
    """
    LeRobot policy wrapper for ADRL.

    Wraps DSACGatedAgent to conform to the LeRobot policy interface.
    Requires a base policy to be set via set_base_policy() before use.
    """

    config_class = ADRLConfig
    name = "adrl"

    def __init__(self, config: ADRLConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.agent = DSACGatedAgent(config)
        self._base_policy: PreTrainedPolicy | None = None

    def set_base_policy(self, base_policy: PreTrainedPolicy) -> None:
        """Set the frozen base policy for residual learning."""
        self._base_policy = base_policy
        self._base_policy.eval()
        for param in self._base_policy.parameters():
            param.requires_grad = False

    @property
    def base_policy(self) -> PreTrainedPolicy:
        if self._base_policy is None:
            raise ValueError("Base policy not set. Call set_base_policy() first.")
        return self._base_policy

    def reset(self) -> None:
        if self._base_policy is not None:
            self._base_policy.reset()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference (base + residual)."""
        base_action = self.base_policy.select_action(batch)

        if "observation.state" in batch:
            state = batch["observation.state"]
        else:
            state = torch.zeros(base_action.shape[0], self.config.state_dim, device=base_action.device)

        residual, gate, _ = self.agent.get_action(state, base_action, eval_mode=True)
        return base_action + residual

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Not used - training uses agent.update() directly."""
        raise NotImplementedError("Use agent.update() for training")
