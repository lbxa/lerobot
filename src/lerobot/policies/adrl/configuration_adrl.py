#!/usr/bin/env python
"""
Configuration for ADRL (Adaptive Distributional Residual Learning) policy.

ADRL learns residual corrections on top of a frozen base policy (e.g., Groot)
using Distributional SAC with a gated stochastic actor.

Key innovations:
- Quantile regression for distributional critic (handles sparse rewards)
- Gated actor that learns WHEN and HOW MUCH to intervene
- Gate value (0-1) allows network to defer to base policy when uncertain
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass
class GatedActorConfig:
    """
    Configuration for the Gated Gaussian Actor network.

    The actor outputs:
    - mu: Mean of Gaussian distribution for residual action
    - log_std: Log standard deviation
    - gate: Sigmoid value (0-1) controlling intervention magnitude
    """

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    use_layer_norm: bool = True
    log_std_min: float = -20.0
    log_std_max: float = 2.0


@dataclass
class QuantileCriticConfig:
    """
    Configuration for the Quantile Critic network.

    Instead of outputting a single Q-value, outputs `num_quantiles` values
    representing different quantiles of the return distribution.
    Critical for robotics where outcomes are often binary (success/fail).
    """

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    use_layer_norm: bool = True
    num_quantiles: int = 25


@PreTrainedConfig.register_subclass("adrl")
@dataclass
class ADRLConfig(PreTrainedConfig):
    """
    ADRL (Adaptive Distributional Residual Learning) configuration.

    This configuration defines a residual learning agent that learns corrections
    on top of a frozen base policy using Distributional SAC with a gated actor.

    Architecture:
        base_action = frozen_policy(obs)
        residual, gate = adrl_agent(obs, base_action)
        full_action = base_action + gate * residual * max_residual_scale

    The gate allows the network to say "I am uncertain, let base handle this"
    (gate -> 0) or "I should correct the base policy" (gate -> 1).
    """

    # ===== Base Policy =====
    # Type of base policy to load (e.g., "groot", "sac", "diffusion")
    base_policy_type: str = "groot"

    # ===== Dimensions =====
    # Inferred from environment if None
    state_dim: int | None = None
    action_dim: int | None = None

    # ===== Architecture =====
    hidden_dim: int = 256
    num_quantiles: int = 25
    num_critics: int = 2

    # ===== Gating =====
    # If True, gate is learned; if False, gate is always 1.0
    learnable_gate: bool = True
    # Maximum scale for residual correction (clips the residual magnitude)
    max_residual_scale: float = 0.15

    # ===== RL Algorithm (DSAC / TQC style) =====
    gamma: float = 0.99
    polyak_tau: float = 0.005
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: float | None = None  # Default: -action_dim

    # ===== Training =====
    batch_size: int = 256
    learning_rate: float = 3e-4
    utd_ratio: int = 2  # Gradient updates per environment step
    grad_clip_norm: float = 40.0
    buffer_size: int = 100_000
    warmup_steps: int = 1000
    online_steps: int = 100_000

    # ===== Device =====
    device: str = "cuda"
    storage_device: str = "cpu"

    # ===== Network Configs =====
    actor_config: GatedActorConfig = field(default_factory=GatedActorConfig)
    critic_config: QuantileCriticConfig = field(default_factory=QuantileCriticConfig)

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self) -> AdamConfig:
        """Return optimizer configuration (used for compatibility, ADRL manages its own optimizers)."""
        return AdamConfig(lr=self.learning_rate)

    def get_scheduler_preset(self) -> None:
        """No scheduler - ADRL manages its own training."""
        return None

    def validate_features(self) -> None:
        """Validate and infer dimensions from features."""
        if ACTION in self.output_features and self.action_dim is None:
            self.action_dim = self.output_features[ACTION].shape[0]

        if OBS_STATE in self.input_features and self.state_dim is None:
            self.state_dim = self.input_features[OBS_STATE].shape[0]

        if self.target_entropy is None and self.action_dim is not None:
            self.target_entropy = -float(self.action_dim)

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None
