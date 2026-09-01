"""Peg-in-hole task phases, termination logic, and shaped reward."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class InsertionPhase(Enum):
    APPROACH = auto()
    ALIGN = auto()
    INSERT = auto()
    HOLD = auto()
    DONE = auto()
    ABORT = auto()


class RelativeInsertionStage(Enum):
    """Stages for a target whose pose is not constant in world coordinates."""

    APPROACH = auto()
    ALIGNMENT = auto()
    CONTACT_SEARCH = auto()
    INSERTION = auto()
    JAM_RECOVERY = auto()
    SUCCESS = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class RelativeInsertionMetrics:
    """Only relative/contact quantities are used by the floating stage detector."""

    lateral_error_m: float
    angular_error_deg: float
    relative_approach_distance_m: float
    relative_speed_mps: float
    insertion_depth_m: float
    contact_force_N: float
    contact_torque_Nm: float
    contact: bool


@dataclass(frozen=True)
class RelativeTaskThresholds:
    approach_distance_m: float = 0.18
    lateral_error_m: float = 0.008
    angular_error_deg: float = 8.0
    target_depth_m: float = 0.030
    max_contact_force_N: float = 45.0
    max_contact_torque_Nm: float = 8.0
    jam_speed_mps: float = 0.001
    insertion_timeout_steps: int = 1200


class RelativeInsertionStateMachine:
    """Stage detector based on live peg-hole relative state and contact."""

    def __init__(self, thresholds: RelativeTaskThresholds | None = None):
        self.thresholds = thresholds or RelativeTaskThresholds()
        self.stage = RelativeInsertionStage.APPROACH
        self.steps = 0

    def reset(self) -> None:
        self.stage = RelativeInsertionStage.APPROACH
        self.steps = 0

    def update(self, metrics: RelativeInsertionMetrics) -> RelativeInsertionStage:
        if self.stage in (RelativeInsertionStage.SUCCESS, RelativeInsertionStage.FAILURE):
            return self.stage
        self.steps += 1
        t = self.thresholds
        if metrics.contact_force_N > t.max_contact_force_N or metrics.contact_torque_Nm > t.max_contact_torque_Nm:
            self.stage = RelativeInsertionStage.FAILURE
        elif self.steps > t.insertion_timeout_steps:
            self.stage = RelativeInsertionStage.FAILURE
        elif (
            metrics.insertion_depth_m >= t.target_depth_m
            and metrics.lateral_error_m < t.lateral_error_m
            and metrics.angular_error_deg < t.angular_error_deg
            and metrics.contact_force_N < t.max_contact_force_N
        ):
            self.stage = RelativeInsertionStage.SUCCESS
        elif self.stage is RelativeInsertionStage.APPROACH and metrics.relative_approach_distance_m <= t.approach_distance_m:
            self.stage = RelativeInsertionStage.ALIGNMENT
        elif self.stage is RelativeInsertionStage.ALIGNMENT and metrics.lateral_error_m <= t.lateral_error_m and metrics.angular_error_deg <= t.angular_error_deg:
            self.stage = RelativeInsertionStage.CONTACT_SEARCH
        elif self.stage is RelativeInsertionStage.CONTACT_SEARCH and metrics.contact:
            self.stage = RelativeInsertionStage.INSERTION
        elif self.stage is RelativeInsertionStage.INSERTION and metrics.contact and metrics.relative_speed_mps < t.jam_speed_mps:
            self.stage = RelativeInsertionStage.JAM_RECOVERY
        elif self.stage is RelativeInsertionStage.JAM_RECOVERY and not metrics.contact:
            self.stage = RelativeInsertionStage.INSERTION
        return self.stage


@dataclass(frozen=True)
class InsertionMetrics:
    approach_distance_m: float
    lateral_error_m: float
    angle_error_deg: float
    insertion_depth_m: float
    contact_force_N: float
    sync_error_m: float


@dataclass(frozen=True)
class TaskThresholds:
    approach_distance_m: float = 0.02
    lateral_error_m: float = 0.001
    angle_error_deg: float = 2.0
    target_depth_m: float = 0.035
    max_contact_force_N: float = 45.0
    max_sync_error_m: float = 0.008
    hold_steps: int = 25
    insertion_timeout_steps: int = 300


class InsertionStateMachine:
    def __init__(self, thresholds: TaskThresholds | None = None):
        self.thresholds = thresholds or TaskThresholds()
        self.phase = InsertionPhase.APPROACH
        self.phase_steps = 0

    def reset(self) -> None:
        self.phase = InsertionPhase.APPROACH
        self.phase_steps = 0

    def update(self, metrics: InsertionMetrics) -> InsertionPhase:
        if self.phase in (InsertionPhase.DONE, InsertionPhase.ABORT):
            return self.phase
        self.phase_steps += 1
        t = self.thresholds
        if metrics.contact_force_N > t.max_contact_force_N:
            self.phase = InsertionPhase.ABORT
        elif metrics.sync_error_m > t.max_sync_error_m:
            self.phase = InsertionPhase.ABORT
        elif self.phase is InsertionPhase.APPROACH:
            if metrics.approach_distance_m <= t.approach_distance_m:
                self._advance(InsertionPhase.ALIGN)
        elif self.phase is InsertionPhase.ALIGN:
            if (
                metrics.lateral_error_m <= t.lateral_error_m
                and metrics.angle_error_deg <= t.angle_error_deg
            ):
                self._advance(InsertionPhase.INSERT)
        elif self.phase is InsertionPhase.INSERT:
            if self.phase_steps > t.insertion_timeout_steps:
                self.phase = InsertionPhase.ABORT
            elif metrics.insertion_depth_m >= t.target_depth_m:
                self._advance(InsertionPhase.HOLD)
        elif self.phase is InsertionPhase.HOLD and self.phase_steps >= t.hold_steps:
            self._advance(InsertionPhase.DONE)
        return self.phase

    def _advance(self, phase: InsertionPhase) -> None:
        self.phase = phase
        self.phase_steps = 0


def compute_reward(metrics: InsertionMetrics, phase: InsertionPhase) -> float:
    reward = -5.0 * metrics.approach_distance_m
    reward -= 50.0 * metrics.lateral_error_m
    reward -= 0.02 * metrics.angle_error_deg
    reward += 20.0 * max(0.0, metrics.insertion_depth_m)
    reward -= 0.002 * metrics.contact_force_N**2
    reward -= 50.0 * metrics.sync_error_m
    if phase is InsertionPhase.HOLD:
        reward += 2.0
    elif phase is InsertionPhase.DONE:
        reward += 100.0
    elif phase is InsertionPhase.ABORT:
        reward -= 100.0
    return reward
