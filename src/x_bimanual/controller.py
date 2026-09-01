"""Simulator-independent dual-agent Cartesian impedance controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _vector6(value: Array, name: str, *, positive: bool = False) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _arms6(value: Array, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2, 6):
        raise ValueError(f"{name} must have shape (2, 6), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _clip_vector_norm(vectors: Array, limit: float) -> Array:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scales = np.minimum(1.0, limit / np.maximum(norms, 1e-12))
    return vectors * scales


@dataclass(frozen=True)
class ImpedanceLimits:
    kp_min: Array
    kp_max: Array
    damping_ratio_min: float = 0.5
    damping_ratio_max: float = 1.5
    max_change_fraction: float = 0.05
    max_force: float = 80.0
    max_torque: float = 12.0

    def __post_init__(self) -> None:
        kp_min = _vector6(self.kp_min, "kp_min", positive=True)
        kp_max = _vector6(self.kp_max, "kp_max", positive=True)
        if np.any(kp_max <= kp_min):
            raise ValueError("every kp_max entry must exceed kp_min")
        if not 0.0 < self.damping_ratio_min <= self.damping_ratio_max:
            raise ValueError("invalid damping ratio bounds")
        if not 0.0 < self.max_change_fraction <= 1.0:
            raise ValueError("max_change_fraction must be in (0, 1]")
        if self.max_force <= 0.0 or self.max_torque <= 0.0:
            raise ValueError("wrench limits must be positive")
        object.__setattr__(self, "kp_min", kp_min)
        object.__setattr__(self, "kp_max", kp_max)


@dataclass(frozen=True)
class SafetyLimits:
    max_contact_force: float = 45.0
    max_sync_error: float = 0.008


@dataclass(frozen=True)
class ControllerOutput:
    wrench: Array
    kp: Array
    kd: Array
    stopped: bool
    stop_reason: str | None


class BimanualVariableImpedanceController:
    """Combine trajectory and impedance actions behind a fixed safety layer.

    The trajectory policy supplies a small correction to desired-minus-current
    Cartesian error. The impedance policy supplies normalized stiffness and
    damping-ratio scales. Rows are left and right arms; columns are xyz/rxyz.
    """

    def __init__(self, impedance: ImpedanceLimits, safety: SafetyLimits):
        self.impedance = impedance
        self.safety = safety
        midpoint = 0.5 * (impedance.kp_min + impedance.kp_max)
        self.kp = np.repeat(midpoint[None, :], 2, axis=0)
        initial_ratio = 0.5 * (impedance.damping_ratio_min + impedance.damping_ratio_max)
        self.kd = 2.0 * initial_ratio * np.sqrt(self.kp)

    def reset(self) -> None:
        midpoint = 0.5 * (self.impedance.kp_min + self.impedance.kp_max)
        self.kp[:] = midpoint
        ratio = 0.5 * (
            self.impedance.damping_ratio_min + self.impedance.damping_ratio_max
        )
        self.kd[:] = 2.0 * ratio * np.sqrt(self.kp)

    def step(
        self,
        pose_error: Array,
        twist: Array,
        trajectory_correction: Array,
        kp_scale: Array,
        damping_ratio_scale: Array,
        contact_force: Array,
        sync_error: float,
    ) -> ControllerOutput:
        pose_error = _arms6(pose_error, "pose_error")
        twist = _arms6(twist, "twist")
        correction = _arms6(trajectory_correction, "trajectory_correction")
        kp_scale = np.clip(_arms6(kp_scale, "kp_scale"), 0.0, 1.0)
        ratio_scale = np.clip(
            _arms6(damping_ratio_scale, "damping_ratio_scale"), 0.0, 1.0
        )
        forces = np.asarray(contact_force, dtype=np.float64)
        if forces.shape != (2, 3) or not np.all(np.isfinite(forces)):
            raise ValueError("contact_force must be finite with shape (2, 3)")

        stop_reason = self._safety_reason(forces, sync_error)
        if stop_reason is not None:
            return ControllerOutput(
                wrench=np.zeros((2, 6), dtype=np.float64),
                kp=self.kp.copy(),
                kd=self.kd.copy(),
                stopped=True,
                stop_reason=stop_reason,
            )

        kp_target = self.impedance.kp_min + kp_scale * (
            self.impedance.kp_max - self.impedance.kp_min
        )
        kp_step = self.impedance.max_change_fraction * (
            self.impedance.kp_max - self.impedance.kp_min
        )
        self.kp += np.clip(kp_target - self.kp, -kp_step, kp_step)

        ratio = self.impedance.damping_ratio_min + ratio_scale * (
            self.impedance.damping_ratio_max - self.impedance.damping_ratio_min
        )
        self.kd = 2.0 * ratio * np.sqrt(self.kp)

        wrench = self.kp * (pose_error + correction) - self.kd * twist
        wrench[:, :3] = _clip_vector_norm(wrench[:, :3], self.impedance.max_force)
        wrench[:, 3:] = _clip_vector_norm(wrench[:, 3:], self.impedance.max_torque)
        return ControllerOutput(
            wrench=wrench,
            kp=self.kp.copy(),
            kd=self.kd.copy(),
            stopped=False,
            stop_reason=None,
        )

    def _safety_reason(self, contact_force: Array, sync_error: float) -> str | None:
        if not np.isfinite(sync_error):
            return "non_finite_sync_error"
        if np.max(np.linalg.norm(contact_force, axis=1)) > self.safety.max_contact_force:
            return "contact_force_limit"
        if abs(sync_error) > self.safety.max_sync_error:
            return "synchronization_error_limit"
        return None


def make_default_controller() -> BimanualVariableImpedanceController:
    return BimanualVariableImpedanceController(
        impedance=ImpedanceLimits(
            kp_min=np.array([50, 50, 20, 5, 5, 2], dtype=np.float64),
            kp_max=np.array([1200, 1200, 500, 120, 120, 50], dtype=np.float64),
        ),
        safety=SafetyLimits(),
    )

