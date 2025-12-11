#!/usr/bin/env python
"""
ADRL (Adaptive Distributional Residual Learning) Training Script.

Synchronous training loop for residual learning on top of a frozen base policy.

Usage:
    python -m lerobot.policies.adrl.train_adrl \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM1 \
        --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 1280, height: 720, fps: 30, fourcc: MJPG} }" \
        --robot.id=so101_follower_arm \
        --teleop.type=so101_leader \
        --teleop.port=/dev/ttyACM0 \
        --teleop.id=so101_leader_arm \
        --policy.pretrained_path=${HF_USER}/so101_gr00t_rubix_stack_v4 \
        --policy.device=cuda \
        --output_dir=outputs/adrl_training

Resume from checkpoint:
    python -m lerobot.policies.adrl.train_adrl \
        --resume=True \
        --output_dir=outputs/adrl_training
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser  # noqa: F401
from lerobot.configs.default import WandBConfig
from lerobot.envs.configs import HILSerlRobotEnvConfig, HILSerlProcessorConfig
from lerobot.policies.adrl.buffer import ResidualReplayBuffer
from lerobot.policies.adrl.configuration_adrl import ADRLConfig
from lerobot.policies.adrl.modeling_adrl import DSACGatedAgent
from lerobot.policies.factory import get_policy_class
from lerobot.processor import TransitionKey, create_transition
from lerobot.rl.gym_manipulator import make_processors, make_robot_env, step_env_and_process_transition
from lerobot.robots import RobotConfig, so100_follower, so101_follower  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig, so100_leader, so101_leader  # noqa: F401
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO)

# Optional WandB import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ADRLTrainConfig:
    """Configuration for ADRL training.

    Uses same CLI structure as lerobot-record for consistency.
    """

    # Robot and teleop (same structure as lerobot-record)
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None

    # Policy config (ADRL-specific settings)
    policy: ADRLConfig = field(default_factory=ADRLConfig)

    # Environment settings
    fps: int = 30

    # Output
    output_dir: Path = Path("outputs/adrl")
    seed: int = 42

    # Logging
    log_freq: int = 100
    save_freq: int = 5000

    # Resume
    resume: bool = False
    checkpoint_path: str | None = None  # If None, uses output_dir/checkpoint.pt

    # WandB
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def build_env_config(self) -> HILSerlRobotEnvConfig:
        """Build HILSerlRobotEnvConfig from flat robot/teleop configs."""
        return HILSerlRobotEnvConfig(
            robot=self.robot,
            teleop=self.teleop,
            fps=self.fps,
            processor=HILSerlProcessorConfig(),
        )


def load_base_policy(config: ADRLConfig, device: str):
    """Load and freeze the base policy."""
    logging.info(f"Loading base policy: {config.pretrained_path}")
    policy_cls = get_policy_class(config.base_policy_type)
    policy = policy_cls.from_pretrained(config.pretrained_path)
    policy.to(device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    logging.info(f"Base policy loaded: {config.base_policy_type}")
    return policy


def save_checkpoint(agent, buffer, step, episode_count, output_dir: Path):
    """Save training checkpoint."""
    checkpoint = {
        "step": step,
        "episode_count": episode_count,
        "buffer_size": len(buffer),
    }
    # Save agent state
    agent.save(str(output_dir / "agent.pt"))
    # Save checkpoint metadata
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    logging.info(f"Checkpoint saved at step {step}")


def load_checkpoint(agent, output_dir: Path, checkpoint_path: str | None = None):
    """Load training checkpoint. Returns (step, episode_count) or (0, 0) if not found."""
    ckpt_path = Path(checkpoint_path) if checkpoint_path else output_dir / "checkpoint.pt"
    agent_path = output_dir / "agent.pt"

    if not ckpt_path.exists() or not agent_path.exists():
        logging.info("No checkpoint found, starting from scratch")
        return 0, 0

    agent.load(str(agent_path))
    checkpoint = torch.load(ckpt_path)
    logging.info(f"Resumed from step {checkpoint['step']}, episode {checkpoint['episode_count']}")
    return checkpoint["step"], checkpoint["episode_count"]


@parser.wrap()
def train(cfg: ADRLTrainConfig):
    """Main ADRL training loop."""
    if cfg.robot is None:
        raise ValueError("robot config required (--robot.type=so101_follower --robot.port=...)")
    if cfg.teleop is None:
        raise ValueError("teleop config required (--teleop.type=so101_leader --teleop.port=...)")
    if cfg.policy.pretrained_path is None:
        raise ValueError("policy.pretrained_path required (--policy.path=...)")

    # Build env config from flat robot/teleop
    env_cfg = cfg.build_env_config()

    # Setup
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = cfg.policy.device
    logging.info(f"Device: {device}")

    # WandB
    use_wandb = WANDB_AVAILABLE and cfg.wandb.enable and cfg.wandb.project
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name or f"adrl_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "policy": cfg.policy.__dict__,
                "seed": cfg.seed,
            },
        )
        logging.info(f"WandB initialized: {cfg.wandb.project}")
    else:
        logging.info("WandB disabled")

    # Environment
    env, teleop = make_robot_env(env_cfg)
    env_proc, action_proc = make_processors(env, teleop, env_cfg, device)

    # Dimensions
    action_dim = env.action_space.shape[0]
    state_dim = cfg.policy.state_dim or action_dim
    cfg.policy.action_dim = action_dim
    cfg.policy.state_dim = state_dim
    logging.info(f"Dims: state={state_dim}, action={action_dim}")

    # Base policy
    base_policy = load_base_policy(cfg.policy, device)

    # ADRL agent
    agent = DSACGatedAgent(cfg.policy, device)
    logging.info(f"Agent params: {sum(p.numel() for p in agent.actor.parameters()):,} (actor)")

    # Buffer
    buffer = ResidualReplayBuffer(cfg.policy.buffer_size, device)

    # Resume
    start_step, episode_count = 0, 0
    if cfg.resume:
        start_step, episode_count = load_checkpoint(agent, cfg.output_dir, cfg.checkpoint_path)

    # Reset env
    obs, info = env.reset()
    env_proc.reset()
    action_proc.reset()
    transition = env_proc(create_transition(observation=obs, info=info))

    episode_reward = 0.0
    metrics = {}

    logging.info(f"Starting training: steps {start_step} -> {cfg.policy.online_steps}")

    for step in range(start_step, cfg.policy.online_steps):
        t0 = time.perf_counter()

        # Get state
        observation = transition[TransitionKey.OBSERVATION]
        state = observation.get(OBS_STATE, torch.zeros(1, state_dim))
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(device)

        # Base action
        with torch.no_grad():
            base_action = base_policy.select_action(observation)

        # Residual action
        if step < cfg.policy.warmup_steps:
            residual = torch.randn_like(base_action) * cfg.policy.max_residual_scale
            gate = torch.ones(1, 1, device=device)
        else:
            residual, gate, _ = agent.get_action(state, base_action, eval_mode=False)

        full_action = base_action + residual

        # Step env
        new_transition = step_env_and_process_transition(
            env, transition, full_action.squeeze(0), env_proc, action_proc
        )

        next_obs = new_transition[TransitionKey.OBSERVATION]
        next_state = next_obs.get(OBS_STATE, torch.zeros(1, state_dim))
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        next_state = next_state.to(device)

        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        # Next base action
        with torch.no_grad():
            next_base_action = base_policy.select_action(next_obs)

        # Store
        buffer.add(
            state, base_action, full_action, float(reward), next_state, next_base_action, done or truncated
        )

        episode_reward += float(reward)

        # Train
        if len(buffer) >= cfg.policy.warmup_steps:
            for _ in range(cfg.policy.utd_ratio):
                metrics = agent.update(buffer.sample(cfg.policy.batch_size))

        # Episode end
        if done or truncated:
            episode_count += 1
            ep_metrics = {
                "episode/reward": episode_reward,
                "episode/length": step,
                "episode/count": episode_count,
                "episode/gate": gate.mean().item(),
            }
            logging.info(f"Ep {episode_count}: reward={episode_reward:.2f}, gate={gate.mean():.3f}")

            if use_wandb:
                wandb.log(ep_metrics, step=step)

            episode_reward = 0.0
            obs, info = env.reset()
            env_proc.reset()
            action_proc.reset()
            transition = env_proc(create_transition(observation=obs, info=info))
        else:
            transition = new_transition

        # Logging
        if step % cfg.log_freq == 0 and step > cfg.policy.warmup_steps and metrics:
            log_metrics = {
                "train/critic_loss": metrics["critic_loss"],
                "train/actor_loss": metrics["actor_loss"],
                "train/alpha": metrics["alpha"],
                "train/gate": metrics["mean_gate"],
                "train/q_value": metrics.get("mean_q", 0),
                "train/buffer_size": len(buffer),
                "train/fps": 1.0 / (time.perf_counter() - t0 + 1e-6),
            }
            logging.info(
                f"Step {step}: critic={metrics['critic_loss']:.4f}, "
                f"actor={metrics['actor_loss']:.4f}, alpha={metrics['alpha']:.4f}"
            )

            if use_wandb:
                wandb.log(log_metrics, step=step)

        # Save checkpoint
        if step > 0 and step % cfg.save_freq == 0:
            save_checkpoint(agent, buffer, step, episode_count, cfg.output_dir)

        # FPS limit
        if cfg.fps:
            dt = time.perf_counter() - t0
            if dt < 1 / cfg.fps:
                time.sleep(1 / cfg.fps - dt)

    # Final save
    save_checkpoint(agent, buffer, cfg.policy.online_steps, episode_count, cfg.output_dir)
    logging.info(f"Training complete. Saved to {cfg.output_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
