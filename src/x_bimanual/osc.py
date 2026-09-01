"""Simulator-independent operational-space torque mapping for two 7-DoF arms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _finite_array(value: Array, shape: tuple[int, ...], name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


@dataclass(frozen=True)
class OperationalSpaceLimits:
    """Limits and gains for wrench-to-joint-torque conversion."""

    max_joint_torque: Array
    nullspace_kp: float = 10.0
    nullspace_kd: float = 2.0
    damping: float = 0.05

    def __post_init__(self) -> None:
        limits = _finite_array(self.max_joint_torque, (7,), "max_joint_torque")
        if np.any(limits <= 0.0):
            raise ValueError("max_joint_torque must be strictly positive")
        if self.nullspace_kp < 0.0 or self.nullspace_kd < 0.0:
            raise ValueError("nullspace gains must be non-negative")
        if not np.isfinite(self.damping) or self.damping <= 0.0:
            raise ValueError("damping must be finite and positive")
        object.__setattr__(self, "max_joint_torque", limits)


@dataclass(frozen=True)
class OperationalSpaceOutput:
    joint_torque: Array
    task_torque: Array
    nullspace_torque: Array
    saturated: Array


class BimanualOperationalSpaceMapper:
    """Map Cartesian wrenches to bounded torques for two 7-DoF arms.

    The primary task uses ``J.T @ wrench``. A damped kinematic nullspace
    projector adds posture regulation without directly competing with the
    six-dimensional end-effector task. Gravity and Coriolis compensation can
    be supplied through ``feedforward_torque`` by the simulator or robot.
    """

    def __init__(self, limits: OperationalSpaceLimits):
        self.limits = limits

    def compute(
        self,
        wrench: Array,
        jacobian: Array,
        joint_position: Array,
        joint_velocity: Array,
        rest_position: Array,
        feedforward_torque: Array | None = None,
        *,
        stopped: bool,
    ) -> OperationalSpaceOutput:
        wrench = _finite_array(wrench, (2, 6), "wrench")
        jacobian = _finite_array(jacobian, (2, 6, 7), "jacobian")
        joint_position = _finite_array(joint_position, (2, 7), "joint_position")
        joint_velocity = _finite_array(joint_velocity, (2, 7), "joint_velocity")
        rest_position = _finite_array(rest_position, (2, 7), "rest_position")
        if feedforward_torque is None:
            feedforward = np.zeros((2, 7), dtype=np.float64)
        else:
            feedforward = _finite_array(
                feedforward_torque, (2, 7), "feedforward_torque"
            )

        if stopped:
            zeros = np.zeros((2, 7), dtype=np.float64)
            return OperationalSpaceOutput(
                joint_torque=zeros,
                task_torque=zeros.copy(),
                nullspace_torque=zeros.copy(),
                saturated=np.zeros((2, 7), dtype=np.bool_),
            )

        task_torque = np.einsum("aij,ai->aj", jacobian, wrench)
        posture_torque = (
            self.limits.nullspace_kp * (rest_position - joint_position)
            - self.limits.nullspace_kd * joint_velocity
        )
        nullspace_torque = np.empty((2, 7), dtype=np.float64)
        identity7 = np.eye(7, dtype=np.float64)
        damping_squared = self.limits.damping**2
        for arm in range(2):
            j = jacobian[arm]
            task_inverse = np.linalg.solve(
                j @ j.T + damping_squared * np.eye(6, dtype=np.float64), j
            )
            nullspace_projector = identity7 - j.T @ task_inverse
            nullspace_torque[arm] = nullspace_projector @ posture_torque[arm]

        commanded = task_torque + nullspace_torque + feedforward
        limits = self.limits.max_joint_torque[None, :]
        saturated = np.abs(commanded) > limits
        joint_torque = np.clip(commanded, -limits, limits)
        return OperationalSpaceOutput(
            joint_torque=joint_torque,
            task_torque=task_torque,
            nullspace_torque=nullspace_torque,
            saturated=saturated,
        )


def make_default_osc_mapper() -> BimanualOperationalSpaceMapper:
    return BimanualOperationalSpaceMapper(
        OperationalSpaceLimits(
            max_joint_torque=np.array([60, 60, 60, 60, 10, 10, 10], dtype=np.float64)
        )
    )
