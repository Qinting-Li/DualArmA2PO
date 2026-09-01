"""A2PO training utilities for the cooperative dual-Panda assembly task.

This implementation keeps the existing two-policy residual-control interface:

* Agent 1: 6-D trajectory / object-motion residual.
* Agent 2: adaptive impedance residual (7-D for the dynamic vertical task).

The policy-optimization stage implements the central mechanisms of
"Order Matters: Agent-by-agent Policy Optimization" (ICLR 2023):

1. one rollout is collected under a frozen behaviour joint policy;
2. agents are updated sequentially from the same rollout;
3. preceding-agent off-policy correction (PreOPC) is used to recompute the
   advantage after preceding policies have changed;
4. the practical A2PO objective clips the preceding joint-policy ratio first
   and then clips the full joint ratio (paper Eq. 6);
5. the clipping width is adapted by update position.

For this robotics task the update order is fixed to Agent 1 -> Agent 2 because
Agent 2 is explicitly conditioned on the sampled Agent-1 action.  This is a
valid fixed selection rule R(k), but it intentionally does not implement the
paper's optional semi-greedy ordering extension.

Important scientific note: the code reproduces the A2PO optimization structure,
but the paper's monotonic-improvement theorem was derived for its MARL setting.
It should not be claimed automatically for this task-specific autoregressive
residual controller without a separate proof.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .panda_dual_assembly import A2POCoordinator, DualPandaAssemblyEnv


@dataclass(frozen=True)
class A2POTrainConfig:
    seed: int = 20260824
    sanity_episodes: int = 100
    formal_episodes: int = 500
    hidden_size: int = 128
    ppo_epochs: int = 4
    learning_rate: float = 1e-4
    value_learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    # Paper-style adaptive clipping:
    # eps_k = eps * [c_eps + (1-c_eps) * k / n].
    # c_eps is exposed rather than hard-coded because the paper treats C(.,.)
    # as a tunable adapting function.
    adaptive_clip_floor: float = 0.5
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    residual_scale_trajectory: float = 0.08
    residual_scale_impedance: float = 0.05
    max_grad_norm: float = 1.0
    # Numerical guard only; this does not replace A2PO clipping.
    log_ratio_limit: float = 20.0
    checkpoint_every: int = 100
    device: str = "cpu"


@dataclass
class EpisodeRecord:
    episode: int
    reward_agent1: float
    reward_agent2: float
    reward_total: float
    success: bool
    peak_force_N: float
    peak_torque_Nm: float
    jamming: bool
    steps: int
    action1_min: float
    action1_max: float
    action2_min: float
    action2_max: float
    kp_min: float
    kp_max: float
    kd_min: float
    kd_max: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def trajectory_prior(obs: np.ndarray) -> np.ndarray:
    """Fallback 6-D trajectory prior.

    The dynamic tabletop environment should normally expose ``rl_action_prior``
    or ``control_target_pose``; :meth:`A2POTrainer._object_target_prior` overrides
    this fallback at sampling time.
    """
    x = np.asarray(obs, dtype=np.float32)
    if x.size < 6:
        return np.zeros(6, dtype=np.float32)
    target_delta = x[:3] + np.array([0.06, 0.0, 0.16], dtype=np.float32)
    action = np.r_[target_delta / 0.15, x[3:6] / 0.8]
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def impedance_prior(obs_with_agent1: np.ndarray) -> np.ndarray:
    """Fallback 12-D impedance prior used by the legacy assembly environment."""
    x = np.asarray(obs_with_agent1, dtype=np.float32)
    base = np.array(
        [0.80, 0.80, 0.90, 0.75, 0.75, 0.75,
         0.75, 0.75, 0.80, 0.80, 0.80, 0.80],
        dtype=np.float32,
    )
    if x.shape[0] >= 42 and x[22] > 0.5:
        base = np.array(
            [0.18, 0.18, 0.85, 0.15, 0.15, 0.15,
             0.90, 0.90, 0.95, 0.95, 0.95, 0.95],
            dtype=np.float32,
        )
    return base


def dynamic_impedance_prior(obs_with_agent1: np.ndarray) -> np.ndarray:
    """7-D grouped K/D and internal-force prior for the vertical task.

    Action semantics:
      [K_parallel, K_lateral, K_rotation,
       D_parallel, D_lateral, D_rotation, F_internal]
    """
    del obs_with_agent1
    return np.array([0.50, 0.65, 0.65, 0.85, 0.90, 0.90, 0.65], dtype=np.float32)


class ResidualActorCritic(nn.Module):
    """Residual Gaussian policy with a compatibility value head.

    The trainer uses a separate shared value network for A2PO/PreOPC.  The
    local value head is retained so existing rollout/render code that expects
    ``sample() -> (action, logp, value, raw)`` remains source-compatible.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        residual_scale: float,
        action_low: float,
        action_high: float,
        prior: Callable[[np.ndarray], np.ndarray],
        device: torch.device,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.residual_scale = float(residual_scale)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.prior_fn = prior
        self.device = device

        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.value = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

        # Start exactly from the validated controller prior; RL learns residuals.
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        self.to(device)

    def _distribution(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.body(obs)
        mean = self.mean(features)
        std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std), features

    def value_of(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(self.body(obs)).squeeze(-1)

    def sample(
        self,
        obs_np: np.ndarray,
        deterministic: bool = False,
        prior_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        obs_np = np.asarray(obs_np, dtype=np.float32)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, _ = self._distribution(obs)
        raw = dist.mean if deterministic else dist.rsample()
        latent = raw.tanh()

        prior_value = self.prior_fn(obs_np) if prior_override is None else prior_override
        prior = torch.as_tensor(prior_value, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = prior + self.residual_scale * latent
        action = action.clamp(self.action_low, self.action_high)

        # Ratios are evaluated on the sampled latent raw variable.  The tanh
        # transform Jacobian cancels in new/old ratios for the same stored raw
        # sample; residual scales are intentionally small so final action clamp
        # saturation should remain rare and is separately audited by action range.
        log_prob = dist.log_prob(raw).sum(-1)
        value = self.value_of(obs)
        return (
            action.squeeze(0).detach().cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
            raw.squeeze(0).detach().cpu().numpy(),
        )

    def log_prob_entropy(
        self,
        obs: torch.Tensor,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist, _ = self._distribution(obs)
        return dist.log_prob(raw).sum(-1), dist.entropy().sum(-1)

    def log_prob_value(
        self,
        obs: torch.Tensor,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy = self.log_prob_entropy(obs, raw)
        return log_prob, entropy, self.value_of(obs)


class SharedValueNetwork(nn.Module):
    """Single global value function V(s), as used by A2PO."""

    def __init__(self, obs_dim: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class A2POTrainer:
    """Two-agent A2PO trainer specialized to trajectory -> impedance control.

    Training order is fixed to Agent 1 then Agent 2.  Both use the same rollout
    and the same cooperative reward.  Agent-2's advantage is recomputed with
    PreOPC after Agent-1 has been updated.
    """

    AGENT_ORDER = ("agent1", "agent2")

    def __init__(self, env: DualPandaAssemblyEnv, config: A2POTrainConfig, output_dir: Path):
        set_seed(config.seed)
        self.env = env
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device)

        if not 0.0 < config.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in (0, 1]")
        if not 0.0 < config.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be in (0, 1)")
        if not 0.0 <= config.adaptive_clip_floor <= 1.0:
            raise ValueError("adaptive_clip_floor must be in [0, 1]")

        shapes = env.observation_space_shapes
        trajectory_dim = int(shapes["trajectory"][0])
        impedance_dim = int(shapes["impedance"][0])
        self.impedance_action_dim = int(getattr(env, "impedance_action_dim", 12))
        if self.impedance_action_dim not in (7, 12):
            raise ValueError(f"unsupported Agent 2 action width: {self.impedance_action_dim}")

        impedance_prior_fn = impedance_prior if self.impedance_action_dim == 12 else dynamic_impedance_prior
        self.agent1 = ResidualActorCritic(
            trajectory_dim,
            6,
            config.hidden_size,
            config.residual_scale_trajectory,
            -1.0,
            1.0,
            trajectory_prior,
            self.device,
        )
        self.agent2 = ResidualActorCritic(
            impedance_dim + 6,
            self.impedance_action_dim,
            config.hidden_size,
            config.residual_scale_impedance,
            0.0,
            1.0,
            impedance_prior_fn,
            self.device,
        )

        self.value_net = SharedValueNetwork(trajectory_dim, config.hidden_size, self.device)
        self.optim1 = torch.optim.Adam(self.agent1.parameters(), lr=config.learning_rate)
        self.optim2 = torch.optim.Adam(self.agent2.parameters(), lr=config.learning_rate)
        self.optim_value = torch.optim.Adam(self.value_net.parameters(), lr=config.value_learning_rate)

        self.records: list[EpisodeRecord] = []
        self.impedance_trace: list[dict[str, Any]] = []
        self.update_trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Action generation / behaviour rollout
    # ------------------------------------------------------------------

    def _object_target_prior(self) -> np.ndarray:
        """Return the next physical 6-D command; never teleports the workpiece."""
        if hasattr(self.env, "rl_action_prior") and not bool(getattr(self.env, "grasped", False)):
            return np.asarray(self.env.rl_action_prior(), dtype=np.float32)

        if hasattr(self.env, "control_target_pose"):
            target_pose = np.asarray(self.env.control_target_pose(), dtype=np.float64)
            target_pos, target_quat = target_pose[:3], target_pose[3:]
        elif hasattr(self.env, "object_target_pose"):
            target_pose = np.asarray(self.env.object_target_pose(), dtype=np.float64)
            target_pos, target_quat = target_pose[:3], target_pose[3:]
        else:
            target_pos = np.array([0.0, 0.0, 0.20], dtype=np.float64)
            target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        desired = self.env.desired_pose
        from .panda_dual_assembly import _quat_conj, _quat_mul, _rotvec

        translation = (target_pos - desired[:3]) / self.env.cfg.action_translation_limit
        rotation = (
            _rotvec(_quat_mul(_quat_conj(desired[3:]), target_quat))
            / self.env.cfg.action_rotation_limit_rad
        )
        return np.clip(np.r_[translation, rotation], -1.0, 1.0).astype(np.float32)

    def _coordinator(
        self,
        obs: Mapping[str, np.ndarray],
        deterministic: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        trace: dict[str, Any] = {}

        def policy1(value: np.ndarray) -> np.ndarray:
            action, logp, _local_value, raw = self.agent1.sample(
                value,
                deterministic=deterministic,
                prior_override=self._object_target_prior(),
            )
            trace["a1"] = action
            trace["logp1"] = logp
            trace["raw1"] = raw
            return action

        def policy2(value: np.ndarray) -> np.ndarray:
            action, logp, _local_value, raw = self.agent2.sample(value, deterministic=deterministic)
            trace["a2"] = action
            trace["logp2"] = logp
            trace["raw2"] = raw
            return action

        actions = A2POCoordinator(policy1, policy2).act(obs)
        return actions, trace

    def _shared_value_np(self, obs1: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.as_tensor(obs1, dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.value_net(x).item())

    def run_episode(
        self,
        episode: int,
        deterministic: bool = False,
        collect_training: bool = True,
    ) -> tuple[EpisodeRecord, dict[str, list[np.ndarray]]]:
        obs = self.env.reset(self.config.seed + episode)
        storage: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "obs1",
                "obs2",
                "raw1",
                "raw2",
                "logp1",
                "logp2",
                "value",
                "reward_joint",
                "done",
            )
        }

        rewards1_diag: list[float] = []
        rewards2_diag: list[float] = []
        rewards_joint_diag: list[float] = []

        initial_wrench = np.asarray(self.env._contact_wrench(), dtype=float)
        peak_force = float(np.linalg.norm(initial_wrench[:3]))
        peak_torque = float(np.linalg.norm(initial_wrench[3:]))
        any_jam = False

        action1_values: list[float] = []
        action2_values: list[float] = []
        kp_values: list[float] = []
        kd_values: list[float] = []

        for _step in range(self.env.cfg.max_steps):
            obs1 = np.asarray(obs["trajectory"], dtype=np.float32)
            obs2_base = np.asarray(obs["impedance"], dtype=np.float32)
            behaviour_value = self._shared_value_np(obs1)

            actions, trace = self._coordinator(obs, deterministic=deterministic)
            # This is exactly the input used by Agent 2 at rollout time.
            obs2 = np.r_[obs2_base, actions.trajectory].astype(np.float32)

            next_obs, reward, done, info = self.env.step(actions.trajectory, actions.impedance)
            effective_done = bool(done or _step == self.env.cfg.max_steps - 1)

            wrench = np.asarray(info.get("wrench", np.zeros(6)), dtype=float)
            kp = np.asarray(info.get("kp", np.zeros(6)), dtype=float)
            kd = np.asarray(info.get("kd", np.zeros(6)), dtype=float)
            peak_force = max(peak_force, float(np.linalg.norm(wrench[:3])))
            peak_torque = max(peak_torque, float(np.linalg.norm(wrench[3:])))
            any_jam = any_jam or bool(info.get("jamming", False))

            action1_values.extend(np.asarray(actions.trajectory, dtype=float).tolist())
            action2_values.extend(np.asarray(actions.impedance, dtype=float).tolist())
            kp_values.extend(kp.tolist())
            kd_values.extend(kd.tolist())

            self.impedance_trace.append(
                {
                    "episode": episode,
                    "step": len(rewards_joint_diag),
                    "stage": info.get("stage", ""),
                    "stage_index": int(info.get("stage_index", 0)),
                    "success": int(info.get("success", False)),
                    "contact": int(info.get("contact", False)),
                    "jamming": int(info.get("jamming", False)),
                    "Kx": float(kp[0]) if kp.size > 0 else 0.0,
                    "Ky": float(kp[1]) if kp.size > 1 else 0.0,
                    "Kz": float(kp[2]) if kp.size > 2 else 0.0,
                    "Krx": float(kp[3]) if kp.size > 3 else 0.0,
                    "Kry": float(kp[4]) if kp.size > 4 else 0.0,
                    "Krz": float(kp[5]) if kp.size > 5 else 0.0,
                    "Dx": float(kd[0]) if kd.size > 0 else 0.0,
                    "Dy": float(kd[1]) if kd.size > 1 else 0.0,
                    "Dz": float(kd[2]) if kd.size > 2 else 0.0,
                    "Drx": float(kd[3]) if kd.size > 3 else 0.0,
                    "Dry": float(kd[4]) if kd.size > 4 else 0.0,
                    "Drz": float(kd[5]) if kd.size > 5 else 0.0,
                    "internal_force_N": float(info.get("internal_force_N", 0.0)),
                    "grasp_capacity_N": float(info.get("grasp_capacity_N", 0.0)),
                    "grasp_load_N": float(info.get("grasp_load_N", 0.0)),
                    "grasp_margin_N": float(info.get("grasp_margin_N", 0.0)),
                    "Fx": float(wrench[0]),
                    "Fy": float(wrench[1]),
                    "Fz": float(wrench[2]),
                    "Tx": float(wrench[3]),
                    "Ty": float(wrench[4]),
                    "Tz": float(wrench[5]),
                    "peg1_lateral_error": float(info.get("peg1_lateral_error", 0.0)),
                    "peg2_lateral_error": float(info.get("peg2_lateral_error", 0.0)),
                    "peg1_depth": float(info.get("peg1_depth", 0.0)),
                    "peg2_depth": float(info.get("peg2_depth", 0.0)),
                    "relative_position_error": float(info.get("relative_position_error", 0.0)),
                    "relative_orientation_error": float(info.get("relative_orientation_error", 0.0)),
                    "agent1_action": np.asarray(actions.trajectory, dtype=float).tolist(),
                    "impedance_action": np.asarray(actions.impedance, dtype=float).tolist(),
                }
            )

            # Diagnostics may be decomposed by the environment, but A2PO must
            # optimize the same cooperative reward for both policies.
            reward1_diag = float(info.get("agent1_reward", reward))
            reward2_diag = float(info.get("agent2_reward", reward))
            joint_reward = float(reward)
            rewards1_diag.append(reward1_diag)
            rewards2_diag.append(reward2_diag)
            rewards_joint_diag.append(joint_reward)

            if collect_training:
                storage["obs1"].append(obs1)
                storage["obs2"].append(obs2)
                storage["raw1"].append(np.asarray(trace["raw1"], dtype=np.float32))
                storage["raw2"].append(np.asarray(trace["raw2"], dtype=np.float32))
                storage["logp1"].append(np.array(trace["logp1"], dtype=np.float32))
                storage["logp2"].append(np.array(trace["logp2"], dtype=np.float32))
                storage["value"].append(np.array(behaviour_value, dtype=np.float32))
                storage["reward_joint"].append(np.array(joint_reward, dtype=np.float32))
                storage["done"].append(np.array(float(effective_done), dtype=np.float32))

            obs = next_obs
            if done:
                break

        def _safe_min(values: list[float], default: float = 0.0) -> float:
            return float(np.min(values)) if values else default

        def _safe_max(values: list[float], default: float = 0.0) -> float:
            return float(np.max(values)) if values else default

        record = EpisodeRecord(
            episode=episode,
            reward_agent1=float(np.sum(rewards1_diag)),
            reward_agent2=float(np.sum(rewards2_diag)),
            # This is the actual cooperative return used for A2PO training.
            reward_total=float(np.sum(rewards_joint_diag)),
            success=bool(self.env.success),
            peak_force_N=peak_force,
            peak_torque_Nm=peak_torque,
            jamming=any_jam,
            steps=len(rewards_joint_diag),
            action1_min=_safe_min(action1_values),
            action1_max=_safe_max(action1_values),
            action2_min=_safe_min(action2_values),
            action2_max=_safe_max(action2_values),
            kp_min=_safe_min(kp_values),
            kp_max=_safe_max(kp_values),
            kd_min=_safe_min(kd_values),
            kd_max=_safe_max(kd_values),
        )

        if not collect_training:
            storage = {key: [] for key in storage}
        return record, storage

    # ------------------------------------------------------------------
    # A2PO / PreOPC
    # ------------------------------------------------------------------

    def _adaptive_clip(self, position: int, n_agents: int = 2) -> float:
        """C(eps, k) used by A2PO's adaptive clipping extension."""
        if not 1 <= position <= n_agents:
            raise ValueError("position must be in [1, n_agents]")
        c = self.config.adaptive_clip_floor
        return float(self.config.clip_ratio * (c + (1.0 - c) * position / n_agents))

    def _critic_values(self, obs1: np.ndarray) -> np.ndarray:
        if len(obs1) == 0:
            return np.empty(0, dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs1, dtype=torch.float32, device=self.device)
            return self.value_net(obs_t).detach().cpu().numpy().astype(np.float32)

    def _preopc_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        done: np.ndarray,
        preceding_joint_ratio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute A^{pi, pi_hat^{i-1}} using PreOPC.

        Paper Eq. (2):
            A_t = delta_t + sum_{k>=1} gamma^k
                  prod_{j=1}^k [lambda * min(1, rho_{t+j})] delta_{t+k}

        which has the backward recursion
            A_t = delta_t + gamma * lambda * min(1, rho_{t+1})
                  * (1-done_t) * A_{t+1}.

        ``preceding_joint_ratio`` is the likelihood ratio of the already-updated
        preceding joint policy to the rollout behaviour joint policy, evaluated
        on the same stored actions.  For the first agent it is all ones and the
        estimator reduces to ordinary GAE(lambda).
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)
        rho = np.asarray(preceding_joint_ratio, dtype=np.float32)

        if not (len(rewards) == len(values) == len(done) == len(rho)):
            raise ValueError("PreOPC arrays must have identical lengths")
        if len(rewards) == 0:
            return rewards.copy(), rewards.copy()

        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]
        deltas = rewards + self.config.gamma * next_values * (1.0 - done) - values

        advantages = np.zeros_like(rewards, dtype=np.float32)
        advantages[-1] = deltas[-1]
        for t in range(len(rewards) - 2, -1, -1):
            correction_next = self.config.gae_lambda * min(1.0, max(0.0, float(rho[t + 1])))
            advantages[t] = (
                deltas[t]
                + self.config.gamma
                * correction_next
                * (1.0 - done[t])
                * advantages[t + 1]
            )

        value_targets = advantages + values
        return advantages.astype(np.float32), value_targets.astype(np.float32)

    @staticmethod
    def _normalize_advantage(advantages: np.ndarray) -> np.ndarray:
        advantages = np.asarray(advantages, dtype=np.float32)
        if len(advantages) <= 1:
            return advantages.copy()
        return ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).astype(np.float32)

    def _policy_ratio(
        self,
        agent: ResidualActorCritic,
        obs: np.ndarray,
        raw: np.ndarray,
        old_logp: np.ndarray,
    ) -> np.ndarray:
        """Current/behaviour likelihood ratio on the stored rollout actions."""
        if len(obs) == 0:
            return np.empty(0, dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            raw_t = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
            old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
            new_logp, _ = agent.log_prob_entropy(obs_t, raw_t)
            log_ratio = (new_logp - old_logp_t).clamp(
                -self.config.log_ratio_limit,
                self.config.log_ratio_limit,
            )
            return log_ratio.exp().detach().cpu().numpy().astype(np.float32)

    def _update_value(self, obs1: np.ndarray, value_targets: np.ndarray) -> float:
        if len(obs1) == 0:
            return 0.0
        obs_t = torch.as_tensor(obs1, dtype=torch.float32, device=self.device)
        target_t = torch.as_tensor(value_targets, dtype=torch.float32, device=self.device)
        value = self.value_net(obs_t)
        loss = 0.5 * (value - target_t).pow(2).mean()
        self.optim_value.zero_grad(set_to_none=True)
        (self.config.value_coef * loss).backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
        self.optim_value.step()
        return float(loss.item())

    def _update_agent_a2po(
        self,
        agent: ResidualActorCritic,
        optimizer: torch.optim.Optimizer,
        obs: np.ndarray,
        raw: np.ndarray,
        old_logp: np.ndarray,
        advantages: np.ndarray,
        preceding_joint_ratio: np.ndarray,
        clip_epsilon: float,
    ) -> dict[str, float]:
        """Optimize the paper's practical A2PO surrogate (Eq. 6).

        For current agent i:
          g = clip(prod_{j in preceding} pi_new^j/pi_old^j,
                   1-eps_i/2, 1+eps_i/2)
          l = (pi_new^i/pi_old^i) * g
          L = E[min(l*A, clip(l,1-eps_i,1+eps_i)*A)]

        ``preceding_joint_ratio`` is detached: preceding policies have already
        completed their update at this stage.
        """
        if len(obs) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "entropy": 0.0, "mean_joint_ratio": 1.0}

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        raw_t = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        preceding_t = torch.as_tensor(preceding_joint_ratio, dtype=torch.float32, device=self.device).detach()

        g = preceding_t.clamp(1.0 - clip_epsilon / 2.0, 1.0 + clip_epsilon / 2.0)
        stats: dict[str, float] = {}

        for _ in range(self.config.ppo_epochs):
            logp, entropy = agent.log_prob_entropy(obs_t, raw_t)
            own_log_ratio = (logp - old_logp_t).clamp(
                -self.config.log_ratio_limit,
                self.config.log_ratio_limit,
            )
            own_ratio = own_log_ratio.exp()
            joint_ratio = own_ratio * g

            unclipped = joint_ratio * adv_t
            clipped = joint_ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_t
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss - self.config.entropy_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), self.config.max_grad_norm)
            optimizer.step()

            stats = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "entropy": float(entropy_mean.item()),
                "mean_joint_ratio": float(joint_ratio.detach().mean().item()),
                "min_joint_ratio": float(joint_ratio.detach().min().item()),
                "max_joint_ratio": float(joint_ratio.detach().max().item()),
            }
        return stats

    def train_episode(self, episode: int) -> EpisodeRecord:
        """Collect one rollout, then update Agent 1 and Agent 2 sequentially."""
        record, storage = self.run_episode(episode, deterministic=False, collect_training=True)
        if not storage["obs1"]:
            return record

        obs1 = np.asarray(storage["obs1"], dtype=np.float32)
        obs2 = np.asarray(storage["obs2"], dtype=np.float32)
        raw1 = np.asarray(storage["raw1"], dtype=np.float32)
        raw2 = np.asarray(storage["raw2"], dtype=np.float32)
        old_logp1 = np.asarray(storage["logp1"], dtype=np.float32)
        old_logp2 = np.asarray(storage["logp2"], dtype=np.float32)
        rewards = np.asarray(storage["reward_joint"], dtype=np.float32)
        done = np.asarray(storage["done"], dtype=np.float32)

        # Ratio of the already-updated preceding joint policy to the behaviour
        # policy.  Before the first agent update, pi_hat^0 == pi, so rho == 1.
        preceding_joint_ratio = np.ones(len(rewards), dtype=np.float32)

        agent_specs = {
            "agent1": (self.agent1, self.optim1, obs1, raw1, old_logp1),
            "agent2": (self.agent2, self.optim2, obs2, raw2, old_logp2),
        }

        for position, agent_name in enumerate(self.AGENT_ORDER, start=1):
            agent, optimizer, agent_obs, raw, old_logp = agent_specs[agent_name]

            # Re-evaluate the shared value function after preceding agent/value
            # updates, then perform PreOPC under the updated preceding policies.
            values = self._critic_values(obs1)
            advantages_raw, value_targets = self._preopc_advantages(
                rewards,
                values,
                done,
                preceding_joint_ratio,
            )
            advantages = self._normalize_advantage(advantages_raw)
            clip_epsilon = self._adaptive_clip(position, n_agents=len(self.AGENT_ORDER))

            policy_stats = self._update_agent_a2po(
                agent,
                optimizer,
                agent_obs,
                raw,
                old_logp,
                advantages,
                preceding_joint_ratio,
                clip_epsilon,
            )

            # Algorithm 1 also updates the single global value function at each
            # agent stage using v(s)=A^{pi,pi_hat^{i-1}}+V(s).
            value_loss = 0.0
            for _ in range(self.config.ppo_epochs):
                value_loss = self._update_value(obs1, value_targets)

            # This agent is now a preceding agent for the next update.  Use its
            # fully updated policy ratio, not a clipped proxy, in PreOPC; Eq. (2)
            # performs the truncation min(1,rho) itself.
            updated_own_ratio = self._policy_ratio(agent, agent_obs, raw, old_logp)
            preceding_joint_ratio = np.clip(
                preceding_joint_ratio * updated_own_ratio,
                np.exp(-self.config.log_ratio_limit),
                np.exp(self.config.log_ratio_limit),
            ).astype(np.float32)

            self.update_trace.append(
                {
                    "episode": episode,
                    "position": position,
                    "agent": agent_name,
                    "clip_epsilon": clip_epsilon,
                    "preopc_adv_mean": float(np.mean(advantages_raw)),
                    "preopc_adv_abs_mean": float(np.mean(np.abs(advantages_raw))),
                    "preceding_ratio_mean_after_update": float(np.mean(preceding_joint_ratio)),
                    "preceding_ratio_min_after_update": float(np.min(preceding_joint_ratio)),
                    "preceding_ratio_max_after_update": float(np.max(preceding_joint_ratio)),
                    "value_loss": value_loss,
                    **policy_stats,
                }
            )

        return record

    # ------------------------------------------------------------------
    # Evaluation / persistence
    # ------------------------------------------------------------------

    def sanity_run(self, episodes: int) -> dict[str, Any]:
        records = [
            self.run_episode(i, deterministic=True, collect_training=False)[0]
            for i in range(episodes)
        ]
        summary = self.summarize(records)
        summary["episodes"] = episodes
        summary["action_range_valid"] = bool(
            summary["action1_min"] >= -1.00001
            and summary["action1_max"] <= 1.00001
            and summary["action2_min"] >= -1e-6
            and summary["action2_max"] <= 1.00001
        )
        summary["impedance_positive"] = bool(summary["kp_min"] > 0 and summary["kd_min"] > 0)
        summary["finite_metrics"] = bool(
            all(
                math.isfinite(float(v))
                for k, v in summary.items()
                if k not in ("episodes", "action_range_valid", "impedance_positive", "finite_metrics")
            )
        )
        return summary

    @staticmethod
    def summarize(records: list[EpisodeRecord]) -> dict[str, Any]:
        if not records:
            return {}

        def values(key: str) -> np.ndarray:
            return np.asarray([getattr(r, key) for r in records], dtype=float)

        return {
            "mean_reward_agent1": float(values("reward_agent1").mean()),
            "mean_reward_agent2": float(values("reward_agent2").mean()),
            "mean_reward_total": float(values("reward_total").mean()),
            "success_rate": float(values("success").mean()),
            "peak_force_N": float(values("peak_force_N").max()),
            "peak_torque_Nm": float(values("peak_torque_Nm").max()),
            "jamming_rate": float(values("jamming").mean()),
            "mean_steps": float(values("steps").mean()),
            "action1_min": float(values("action1_min").min()),
            "action1_max": float(values("action1_max").max()),
            "action2_min": float(values("action2_min").min()),
            "action2_max": float(values("action2_max").max()),
            "kp_min": float(values("kp_min").min()),
            "kp_max": float(values("kp_max").max()),
            "kd_min": float(values("kd_min").min()),
            "kd_max": float(values("kd_max").max()),
        }

    def save_checkpoint(self, episode: int, path: Path | None = None) -> Path:
        path = path or self.output_dir / f"checkpoint_{episode:06d}.pt"
        torch.save(
            {
                "episode": episode,
                "algorithm": "A2PO-PreOPC-fixed-order",
                "config": asdict(self.config),
                "agent_update_order": list(self.AGENT_ORDER),
                "impedance_action_dim": self.impedance_action_dim,
                "observation_space_shapes": self.env.observation_space_shapes,
                "agent1": self.agent1.state_dict(),
                "agent2": self.agent2.state_dict(),
                "critic": self.value_net.state_dict(),
                "optimizer1": self.optim1.state_dict(),
                "optimizer2": self.optim2.state_dict(),
                "optimizer_value": self.optim_value.state_dict(),
                "records": [asdict(r) for r in self.records],
                "update_trace_tail": self.update_trace[-20:],
            },
            path,
        )
        return path

    def load_checkpoint(self, path: Path, load_optimizers: bool = True) -> dict[str, Any]:
        """Full training resume helper; evaluation code may load only actors."""
        payload = torch.load(path, map_location=self.device)
        self.agent1.load_state_dict(payload["agent1"])
        self.agent2.load_state_dict(payload["agent2"])
        if "critic" in payload:
            self.value_net.load_state_dict(payload["critic"])
        if load_optimizers:
            if "optimizer1" in payload:
                self.optim1.load_state_dict(payload["optimizer1"])
            if "optimizer2" in payload:
                self.optim2.load_state_dict(payload["optimizer2"])
            if "optimizer_value" in payload:
                self.optim_value.load_state_dict(payload["optimizer_value"])
        return payload

    def write_records(self, path: Path) -> None:
        if not self.records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(asdict(self.records[0]).keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

    def write_summary(self, path: Path) -> None:
        path.write_text(json.dumps(self.summarize(self.records), indent=2))

    def write_impedance_trace(self, path: Path) -> None:
        if not self.impedance_trace:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(self.impedance_trace[0].keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.impedance_trace)

    def write_update_trace(self, path: Path) -> None:
        if not self.update_trace:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(self.update_trace[0].keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.update_trace)
