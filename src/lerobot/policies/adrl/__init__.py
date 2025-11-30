"""
ADRL: Adaptive Distributional Residual Learning.

A framework for learning residual corrections on top of frozen base policies
using Distributional SAC with a gated stochastic actor.
"""

from lerobot.policies.adrl.buffer import ResidualReplayBuffer
from lerobot.policies.adrl.configuration_adrl import (
    ADRLConfig,
    GatedActorConfig,
    QuantileCriticConfig,
)
from lerobot.policies.adrl.modeling_adrl import (
    ADRLPolicy,
    BaseResidualAgent,
    DSACGatedAgent,
    GatedGaussianActor,
    QuantileCritic,
)

__all__ = [
    "ADRLConfig",
    "GatedActorConfig",
    "QuantileCriticConfig",
    "ADRLPolicy",
    "BaseResidualAgent",
    "DSACGatedAgent",
    "GatedGaussianActor",
    "QuantileCritic",
    "ResidualReplayBuffer",
]
