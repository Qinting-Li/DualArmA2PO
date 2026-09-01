"""Tabletop pick-up -> transport -> compliant vertical insertion for dual Panda.

This environment is designed to be used with the repository's A2PO trainer:

    Agent 1: 6-D task-space motion residual
             [dx, dy, dz, droll, dpitch, dyaw]
    Agent 2: 7-D adaptive impedance / grasp action
             [K_parallel, K_lateral, K_rotation,
              D_parallel, D_lateral, D_rotation, F_internal]

The environment deliberately keeps A2PO's policy-optimization mechanism out of
this file.  The trainer must still perform the paper's single-rollout,
agent-by-agent sequential update and preceding-agent off-policy correction
(PreOPC).  This file provides the task dynamics, two-agent action semantics,
separate reward channels, and a contact-rich compliant-assembly problem.

Task sequence
-------------
TABLE_PREGRASP -> DESCEND_TO_GRASP -> BILATERAL_CONTACT -> GRASP -> LIFT ->
TRANSPORT -> COARSE_ALIGNMENT -> FIRST_CONTACT -> COMPLIANT_ALIGNMENT ->
INSERTION -> RELEASE -> RETREAT -> SUCCESS.

Important modelling choice
--------------------------
The verified repository controls only the seven Panda arm joints per side.  It
does not expose independent finger actuators here, so grasp acquisition is a
*contact/proximity-gated cooperative weld abstraction*: both wrists must reach
opposite sides of the physical free workpiece, remain there for several control
steps, and have low relative speed before the two grasp welds are enabled.  The
workpiece is never teleported during a rollout.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import mujoco
import numpy as np

from .osc import BimanualOperationalSpaceMapper, OperationalSpaceLimits
from .panda_dual_assembly import (
    AssemblyStage,
    DualAssemblyConfig,
    DualPandaAssemblyEnv,
    _mat_quat,
    _quat_conj,
    _quat_from_rotvec,
    _quat_mul,
    _rotvec,
    build_dual_panda_xml,
)


EARTH_GRAVITY = (0.0, 0.0, -9.81)
TABLE_CENTER = np.array([0.0, 0.40, 0.03], dtype=float)
TABLE_HALF_SIZE = np.array([0.34, 0.25, 0.03], dtype=float)
TABLE_TOP_Z = float(TABLE_CENTER[2] + TABLE_HALF_SIZE[2])
TABLE_OBJECT_Z = TABLE_TOP_Z + 0.024


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_dynamic_vertical_xml(cfg: DualAssemblyConfig) -> str:
    """Build a gravity-enabled tabletop scene with a floating vertical receiver."""
    # The previous environment forced zero gravity.  A tabletop pickup is only
    # meaningful if the free workpiece is physically supported by a surface.
    xml = build_dual_panda_xml(replace(cfg, gravity=EARTH_GRAVITY))

    # Peg axes point along -Y so that the final insertion direction is toward
    # the receiver face in the global -Y direction.
    peg_quat = "0.70710678 0.70710678 0 0"
    half = cfg.peg_spacing / 2
    old1 = (
        f'<geom name="workpiece_peg1" type="cylinder" '
        f'size="{cfg.peg_radius} {cfg.peg_length / 2}" '
        f'pos="{-half} 0 {-cfg.peg_length / 2 - 0.018}" '
        'rgba="0.86 0.25 0.08 1"/>'
    )
    old2 = (
        f'<geom name="workpiece_peg2" type="cylinder" '
        f'size="{cfg.peg_radius} {cfg.peg_length / 2}" '
        f'pos="{half} 0 {-cfg.peg_length / 2 - 0.018}" '
        'rgba="0.86 0.25 0.08 1"/>'
    )
    new1 = (
        f'<geom name="workpiece_peg1" type="cylinder" '
        f'size="{cfg.peg_radius} {cfg.peg_length / 2}" '
        f'pos="{-half} {-cfg.peg_length / 2 - 0.018} 0" '
        f'quat="{peg_quat}" friction="0.9 0.005 0.0001" '
        'rgba="0.86 0.25 0.08 1"/>'
    )
    new2 = (
        f'<geom name="workpiece_peg2" type="cylinder" '
        f'size="{cfg.peg_radius} {cfg.peg_length / 2}" '
        f'pos="{half} {-cfg.peg_length / 2 - 0.018} 0" '
        f'quat="{peg_quat}" friction="0.9 0.005 0.0001" '
        'rgba="0.86 0.25 0.08 1"/>'
    )
    if old1 not in xml or old2 not in xml:
        raise RuntimeError("verified workpiece peg definitions changed unexpectedly")
    xml = xml.replace(old1, new1).replace(old2, new2)

    # Dynamic vertical receiver.  It remains a free body so contact forces move
    # it, while _receiver_disturbance() below supplies a compliant suspension.
    old_receiver = '<body name="receiver" pos="0 0 0.0">'
    new_receiver = (
        '<body name="receiver" pos="0 0.12 0.28" '
        'quat="0.70710678 0.70710678 0 0">'
        '<joint name="receiver_free" type="free" damping="1.0"/>'
        '<inertial pos="0 0 0" mass="10.0" diaginertia="0.2 0.2 0.2"/>'
    )
    if old_receiver not in xml:
        raise RuntimeError("verified receiver body definition changed unexpectedly")
    xml = xml.replace(old_receiver, new_receiver, 1)

    # Add a real collidable table.  Its front edge is deliberately behind the
    # receiver so that the insertion interface is not embedded in the table.
    table_xml = (
        f'<body name="pickup_table" pos="{TABLE_CENTER[0]} {TABLE_CENTER[1]} {TABLE_CENTER[2]}">'
        f'<geom name="pickup_table_geom" type="box" '
        f'size="{TABLE_HALF_SIZE[0]} {TABLE_HALF_SIZE[1]} {TABLE_HALF_SIZE[2]}" '
        'friction="1.2 0.01 0.0002" rgba="0.36 0.31 0.26 1" '
        'contype="1" conaffinity="1"/>'
        '</body>'
    )
    if "</worldbody>" not in xml:
        raise RuntimeError("MuJoCo XML has no worldbody closing tag")
    xml = xml.replace("</worldbody>", table_xml + "</worldbody>", 1)
    return xml


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DynamicVerticalDualPandaEnv(DualPandaAssemblyEnv):
    """Dual-Panda tabletop pickup and A2PO-compatible compliant assembly."""

    def __init__(self, cfg: DualAssemblyConfig | None = None, seed: int = 0):
        # Keep all task parameters from the repository but make tabletop physics
        # gravitational even if the formal wrapper passes gravity=(0,0,0).
        self.cfg = cfg or DualAssemblyConfig(gravity=EARTH_GRAVITY)
        if not 0.0 <= self.cfg.internal_force_min < self.cfg.internal_force_max:
            raise ValueError("internal force bounds must satisfy 0 <= min < max")
        if self.cfg.grasp_friction <= 0.0:
            raise ValueError("grasp_friction must be positive")
        if self.cfg.grasp_slip_steps <= 0:
            raise ValueError("grasp_slip_steps must be positive")

        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_string(build_dynamic_vertical_xml(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = np.asarray(EARTH_GRAVITY)

        # Damped cooperative weld: only activated after bilateral grasp gating.
        self.model.eq_solref[:] = np.array([0.08, 1.0])

        self.arm_jids = np.array([
            [self.model.joint(f"{side}_panda_joint{i}").id for i in range(1, 8)]
            for side in ("left", "right")
        ])
        self.arm_qpos = np.array([
            [self.model.jnt_qposadr[j] for j in row] for row in self.arm_jids
        ])
        self.arm_dof = np.array([
            [self.model.jnt_dofadr[j] for j in row] for row in self.arm_jids
        ])
        self.ee_ids = np.array([
            self.model.site("left_ee").id,
            self.model.site("right_ee").id,
        ])
        self.workpiece_body = self.model.body("workpiece").id
        self.receiver_body = self.model.body("receiver").id
        self.receiver_qpos = self.model.jnt_qposadr[
            self.model.joint("receiver_free").id
        ]
        self.workpiece_qpos = self.model.jnt_qposadr[
            self.model.joint("workpiece_free").id
        ]
        self.workpiece_dof = self.model.jnt_dofadr[
            self.model.joint("workpiece_free").id
        ]
        self.hole_ids = np.array([
            self.model.site("hole1").id,
            self.model.site("hole2").id,
        ])
        self.peg_ids = np.array([
            self.model.geom("workpiece_peg1").id,
            self.model.geom("workpiece_peg2").id,
        ])
        self.table_geom = self.model.geom("pickup_table_geom").id

        self.mapper = BimanualOperationalSpaceMapper(
            OperationalSpaceLimits(
                max_joint_torque=np.asarray(self.cfg.max_joint_torque),
                nullspace_kp=8.0,
                nullspace_kd=2.0,
                damping=0.08,
            )
        )

        # Detailed phases are intentionally richer than AssemblyStage.  Keeping
        # the inherited enum avoids breaking the rest of the repository.
        self.phase_names = (
            "TABLE_PREGRASP",
            "DESCEND_TO_GRASP",
            "BILATERAL_CONTACT",
            "GRASP",
            "LIFT",
            "TRANSPORT",
            "COARSE_ALIGNMENT",
            "FIRST_CONTACT",
            "COMPLIANT_ALIGNMENT",
            "INSERTION",
            "JAM_RECOVERY",
            "STABILIZATION",
            "RELEASE",
            "RETREAT",
            "SUCCESS",
            "GRASP_SLIP",
        )

        self.rest_q = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 1.4, 0.75])
        self.lift_height_m = 0.18
        self.transport_standoff_m = 0.15
        self.coarse_standoff_m = 0.075
        self.first_contact_standoff_m = 0.010
        self.grasp_side_offset_m = 0.115
        self.pregrasp_hover_m = 0.10
        self.grasp_distance_threshold_m = 0.040
        self.grasp_speed_threshold_mps = 0.12
        self.grasp_dwell_required = 5
        self.release_hold_required = 5
        self.retreat_required = 10

        self._initialize_runtime_state()

    # ------------------------------------------------------------------
    # Reset / state
    # ------------------------------------------------------------------

    def _initialize_runtime_state(self) -> None:
        self.phase = "TABLE_PREGRASP"
        self.stage = AssemblyStage.INITIALIZATION
        self.step_count = 0
        self.stable_count = 0
        self.success = False
        self.assembly_complete = False

        self.grasped = False
        self.grasp_step = -1
        self.grasp_failed = False
        self.grasp_overload_steps = 0
        self.grasp_ready_steps = 0

        self.release_steps = 0
        self.retreat_steps = 0
        self.release_q_targets: np.ndarray | None = None
        self.retreat_q_targets: np.ndarray | None = None

        self.desired_pose = np.zeros(7)
        self.previous_action = np.zeros(6)
        self.previous_impedance = np.ones(7) * 0.5
        self.previous_depth_sum = 0.0
        self.previous_lift_z = TABLE_OBJECT_Z
        self.contact_steps = 0
        self.no_progress_steps = 0
        self.last_wrench = np.zeros(6)
        self.last_depth = np.zeros(2)
        self.jam_recovery = False
        self.jam_recovery_steps = 0
        self.jam_backoff_y: float | None = None
        self.alignment_ready = False
        self.log: list[dict[str, Any]] = []

        self.grasp_site_pos_local = np.zeros((2, 3))
        self.grasp_site_rot_local = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        self.pickup_pose = np.zeros(7)
        self.hover_q_targets = np.repeat(self.rest_q[None, :], 2, axis=0)
        self.grasp_q_targets = np.repeat(self.rest_q[None, :], 2, axis=0)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.eq_active[:] = 0
        self.data.qpos[self.arm_qpos[0]] = self.rest_q
        self.data.qpos[self.arm_qpos[1]] = self.rest_q

        # Tabletop randomization: x/y/yaw vary, while roll/pitch stay small so
        # the workpiece starts physically supported rather than intersecting the table.
        xy = min(float(self.cfg.initial_xy_range_m), 0.08)
        x = float(self.rng.uniform(-xy, xy))
        y = float(self.rng.uniform(0.37, 0.43))
        rot_lim = min(float(self.cfg.initial_rotation_range_rad), 0.20)
        roll_pitch_lim = min(rot_lim, 0.035)
        rotvec = np.array([
            self.rng.uniform(-roll_pitch_lim, roll_pitch_lim),
            self.rng.uniform(-roll_pitch_lim, roll_pitch_lim),
            self.rng.uniform(-rot_lim, rot_lim),
        ])
        object_pose = np.r_[
            [x, y, TABLE_OBJECT_Z + self.rng.uniform(0.0, 0.003)],
            _quat_from_rotvec(rotvec),
        ]
        self.data.qpos[self.workpiece_qpos:self.workpiece_qpos + 7] = object_pose

        # Vertical receiver starts outside the front edge of the pickup table.
        self.data.qpos[self.receiver_qpos:self.receiver_qpos + 7] = np.array(
            [0.0, 0.12, 0.28, 0.70710678, 0.70710678, 0.0, 0.0]
        )
        mujoco.mj_forward(self.model, self.data)

        self._initialize_runtime_state()
        self.pickup_pose = object_pose.copy()
        self.desired_pose = object_pose.copy()
        self.previous_lift_z = float(object_pose[2])

        # Two-stage approach: hover above the workpiece, then descend to its sides.
        hover_positions = self._grasp_target_positions(object_pose[:3], self.pregrasp_hover_m)
        grasp_positions = self._grasp_target_positions(object_pose[:3], 0.006)
        self.hover_q_targets = self._solve_ee_position_targets(hover_positions)
        self.grasp_q_targets = self._solve_ee_position_targets(grasp_positions)
        return self.observations()

    @property
    def observation_space_shapes(self) -> dict[str, tuple[int, ...]]:
        # Preserve the repository's dimensions so existing trainer construction
        # remains compatible.  The contents are reorganized to expose explicit
        # task-phase / compliance state.
        return {"trajectory": (96,), "impedance": (115,)}

    @property
    def impedance_action_dim(self) -> int:
        return 7

    # ------------------------------------------------------------------
    # Geometry / targets
    # ------------------------------------------------------------------

    def _grasp_target_positions(self, object_pos: np.ndarray, z_offset: float) -> tuple[np.ndarray, np.ndarray]:
        return (
            object_pos + np.array([-self.grasp_side_offset_m, 0.0, z_offset]),
            object_pos + np.array([+self.grasp_side_offset_m, 0.0, z_offset]),
        )

    def _solve_ee_position_targets(
        self,
        targets: tuple[np.ndarray, np.ndarray],
        initial_q: np.ndarray | None = None,
    ) -> np.ndarray:
        """Damped position IK without modifying the live simulation state."""
        solved = []
        for arm, target in enumerate(targets):
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = self.data.qpos
            q = (
                self.rest_q.copy()
                if initial_q is None
                else np.asarray(initial_q[arm], dtype=float).copy()
            )
            for _ in range(160):
                scratch.qpos[self.arm_qpos[arm]] = q
                mujoco.mj_forward(self.model, scratch)
                error = np.asarray(target) - scratch.site_xpos[self.ee_ids[arm]]
                if np.linalg.norm(error) < 2e-4:
                    break
                jp = np.zeros((3, self.model.nv))
                jr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(
                    self.model, scratch, jp, jr, int(self.ee_ids[arm])
                )
                jac = jp[:, self.arm_dof[arm]]
                dq = jac.T @ np.linalg.solve(
                    jac @ jac.T + 0.003 * np.eye(3), error
                )
                q = np.clip(
                    q + 0.55 * dq,
                    self.model.jnt_range[self.arm_jids[arm], 0],
                    self.model.jnt_range[self.arm_jids[arm], 1],
                )
            solved.append(q)
        return np.asarray(solved)

    def object_target_pose(self) -> np.ndarray:
        """Live receiver-relative pose corresponding to successful insertion."""
        receiver_quat = self.data.xquat[self.receiver_body].copy()
        nominal_receiver_quat = np.array([0.70710678, 0.70710678, 0.0, 0.0])
        target_quat = _quat_mul(receiver_quat, _quat_conj(nominal_receiver_quat))
        target_quat /= max(np.linalg.norm(target_quat), 1e-9)

        target_matrix_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(target_matrix_flat, target_quat)
        target_matrix = target_matrix_flat.reshape(3, 3)

        local_tip = np.array([
            -self.cfg.peg_spacing / 2,
            -self.cfg.peg_length - 0.018,
            0.0,
        ])
        hole_pos, hole_matrix = self._hole_pose(0)
        desired_tip = hole_pos + (self.cfg.required_depth + 0.008) * hole_matrix[:, 2]
        target_pos = desired_tip - target_matrix @ local_tip
        return np.r_[target_pos, target_quat]

    def _standoff_pose(self, standoff_m: float, keep_lift_height: bool = False) -> np.ndarray:
        target = self.object_target_pose().copy()
        target[1] += float(standoff_m)
        if keep_lift_height:
            target[2] = max(target[2], TABLE_OBJECT_Z + self.lift_height_m)
        return target

    def control_target_pose(self) -> np.ndarray:
        """Stage-aware deterministic prior used by Agent 1's residual policy."""
        pos, quat = self._workpiece_pose()

        if not self.grasped:
            return np.r_[pos, quat]

        if self.phase in ("GRASP", "LIFT"):
            target = np.r_[pos.copy(), quat.copy()]
            target[2] = TABLE_OBJECT_Z + self.lift_height_m
            return target

        if self.phase == "TRANSPORT":
            return self._standoff_pose(self.transport_standoff_m, keep_lift_height=True)

        if self.phase == "COARSE_ALIGNMENT":
            target = self._standoff_pose(self.coarse_standoff_m)
            signed_errors = np.asarray([
                self._hole_pose(i)[0] - self.peg_tip(i) for i in range(2)
            ])
            target[[0, 2]] += 1.5 * np.clip(
                np.mean(signed_errors[:, [0, 2]], axis=0),
                -0.006,
                0.006,
            )
            return target

        if self.phase == "FIRST_CONTACT":
            return self._standoff_pose(self.first_contact_standoff_m)

        target = self.object_target_pose().copy()

        # During compliant alignment, correct the two-peg lateral error without
        # imposing a large normal advance into the receiver.
        if self.phase in ("COMPLIANT_ALIGNMENT", "INSERTION", "STABILIZATION", "JAM_RECOVERY"):
            signed_errors = np.asarray([
                self._hole_pose(i)[0] - self.peg_tip(i) for i in range(2)
            ])
            correction = np.mean(signed_errors[:, [0, 2]], axis=0)
            gain = (
                2.0
                if self.phase == "JAM_RECOVERY"
                else 1.5
                if self.phase == "COMPLIANT_ALIGNMENT"
                else 0.8
            )
            target[[0, 2]] += gain * np.clip(correction, -0.006, 0.006)

        # Explicit jam recovery: back off along +Y and search laterally with a
        # tiny deterministic spiral-like oscillation.  The learned Agent-1
        # residual can improve on this safe prior.
        if self.jam_recovery or self.phase == "JAM_RECOVERY":
            t = self.step_count * self.cfg.timestep * self.cfg.control_interval
            # Back off from the *current* deflected pose. Referencing only the
            # nominal insertion target can still command motion into the plate
            # after a collision has pushed the workpiece away from that target.
            backoff_y = self.jam_backoff_y
            if backoff_y is None:
                backoff_y = float(pos[1] + 0.025)
            target[1] = max(target[1] + 0.025, backoff_y)
            target[0] += 0.003 * math.sin(2.1 * t)
            target[2] += 0.003 * math.cos(1.7 * t)

        return target

    def peg_tip(self, i: int) -> np.ndarray:
        pos, _ = self._workpiece_pose()
        local = np.array([
            (-1 if i == 0 else 1) * self.cfg.peg_spacing / 2,
            -self.cfg.peg_length - 0.018,
            0.0,
        ])
        return pos + self.data.xmat[self.workpiece_body].reshape(3, 3) @ local

    def peg_errors(self) -> tuple[np.ndarray, np.ndarray]:
        lateral, depth = [], []
        for i in range(2):
            hole, _ = self._hole_pose(i)
            delta = hole - self.peg_tip(i)
            lateral.append(float(np.linalg.norm(delta[[0, 2]])))
            depth.append(
                float(np.clip(hole[1] - self.peg_tip(i)[1], 0.0, self.cfg.peg_length))
            )
        return np.asarray(lateral), np.asarray(depth)

    def _relative_orientation_error(self) -> float:
        obj_axis = self.data.xmat[self.workpiece_body].reshape(3, 3) @ np.array([0.0, -1.0, 0.0])
        hole_axis = self.data.site_xmat[self.hole_ids[0]].reshape(3, 3)[:, 2]
        return float(
            np.degrees(
                np.arccos(np.clip(abs(np.dot(obj_axis, hole_axis)), -1.0, 1.0))
            )
        )

    # ------------------------------------------------------------------
    # Receiver dynamics / observations
    # ------------------------------------------------------------------

    def _receiver_disturbance(self) -> None:
        """6-D compliant suspension + bounded sway + gravity compensation."""
        t = self.step_count * self.cfg.timestep * self.cfg.control_interval
        nominal_pos = np.array([0.0, 0.12, 0.28])
        nominal_quat = np.array([0.70710678, 0.70710678, 0.0, 0.0])
        pose, velocity = self._receiver_state()

        sway = np.array([
            0.004 * np.sin(0.8 * t),
            0.003 * np.cos(0.65 * t),
            0.003 * np.sin(0.55 * t + 0.4),
        ])
        sway_rotation = np.array([
            0.015 * np.sin(0.7 * t),
            0.012 * np.cos(0.9 * t),
            0.010 * np.sin(0.5 * t),
        ])
        desired_quat = _quat_mul(nominal_quat, _quat_from_rotvec(sway_rotation))
        orientation_error = _rotvec(_quat_mul(_quat_conj(desired_quat), pose[3:]))

        mass = float(self.model.body_mass[self.receiver_body])
        gravity_comp = -mass * np.asarray(self.model.opt.gravity)
        self.data.xfrc_applied[self.receiver_body, :3] = (
            gravity_comp
            - 5000.0 * (pose[:3] - nominal_pos - sway)
            - 300.0 * velocity[:3]
        )
        self.data.xfrc_applied[self.receiver_body, 3:] = (
            -5.0 * orientation_error - 2.0 * velocity[3:]
        )

    def _receiver_state(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.receiver_body].copy()
        quat = self.data.xquat[self.receiver_body].copy()
        dof = self.model.jnt_dofadr[self.model.joint("receiver_free").id]
        qv = self.data.qvel[dof:dof + 6].copy()
        return np.r_[pos, quat], qv

    def observations(self) -> dict[str, np.ndarray]:
        """Return A2PO-compatible Agent-1 and Agent-2 observations.

        Agent 2 receives the same 96-D trajectory state plus 12-D estimated
        wrist wrench and its previous 7-D impedance action -> 115 dimensions.
        The trainer can then append Agent 1's *current* action, preserving the
        sequential Agent-1 -> Agent-2 information path used by this project.
        """
        obj_pos, obj_quat = self._workpiece_pose()
        obj_qv = self.data.qvel[self.workpiece_dof:self.workpiece_dof + 6].copy()
        receiver_pose, receiver_qv = self._receiver_state()
        lat, dep = self.peg_errors()
        hole0 = self._hole_pose(0)[0]
        rel = np.r_[
            hole0 - self.peg_tip(0),
            self._relative_orientation_error() / 180.0,
            lat / 0.05,
            dep / max(self.cfg.peg_length, 1e-6),
        ]

        ee_pose: list[float] = []
        ee_vel: list[float] = []
        ee_wrench: list[float] = []
        for site in self.ee_ids:
            ee_pose.extend(self.data.site_xpos[site].tolist())
            ee_pose.extend(_mat_quat(self.data.site_xmat[site].reshape(3, 3)).tolist())
            site_vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_SITE,
                int(site),
                site_vel,
                0,
            )
            ee_vel.extend(site_vel.tolist())

        for arm in range(2):
            # Constraint generalized force is used only as a compact proprioceptive
            # signal.  Contact-force evaluation still uses _contact_wrench().
            arm_force = self.data.qfrc_constraint[self.arm_dof[arm]]
            ee_wrench.extend(np.r_[arm_force[:6], 0.0][:6].tolist())

        q = np.clip(
            np.r_[self.data.qpos[self.arm_qpos[0]], self.data.qpos[self.arm_qpos[1]]],
            -10,
            10,
        ) / 10.0
        dq = np.clip(
            np.r_[self.data.qvel[self.arm_dof[0]], self.data.qvel[self.arm_dof[1]]],
            -20,
            20,
        ) / 20.0

        phase_norm = float(self.phase_names.index(self.phase)) / max(len(self.phase_names) - 1, 1)
        task_flags = np.array([
            float(self._contact_state()),
            float(self.grasped),
            float(self.jam_recovery),
            phase_norm,
            np.clip((obj_pos[2] - TABLE_TOP_Z) / max(self.lift_height_m, 1e-6), -1.0, 1.0),
            float(self.alignment_ready),
            np.clip(self.stable_count / max(self.cfg.stable_steps, 1), 0.0, 1.0),
            float(self.assembly_complete),
        ])

        # Exactly 96 dims: 14 q + 14 dq + 14 EE pose + 12 EE velocity +
        # 13 object + 13 receiver + 8 relative task + 8 task flags.
        trajectory = np.r_[
            q,
            dq,
            ee_pose,
            ee_vel,
            obj_pos,
            obj_quat,
            obj_qv,
            receiver_pose,
            receiver_qv,
            rel,
            task_flags,
        ]
        if trajectory.size != 96:
            raise RuntimeError(f"trajectory observation is {trajectory.size}D, expected 96D")

        impedance = np.r_[trajectory, np.asarray(ee_wrench), self.previous_impedance]
        if impedance.size != 115:
            raise RuntimeError(f"impedance observation is {impedance.size}D, expected 115D")

        return {
            "trajectory": np.nan_to_num(trajectory, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32),
            "impedance": np.nan_to_num(impedance, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Tabletop pickup / grasp abstraction
    # ------------------------------------------------------------------

    def _joint_pd_step(self, targets: np.ndarray, kp: float = 80.0, kd: float = 20.0) -> None:
        q = self.data.qpos[self.arm_qpos].copy()
        qd = self.data.qvel[self.arm_dof].copy()
        torque = np.zeros((2, 7))
        limits = np.asarray(self.cfg.max_joint_torque, dtype=float)
        for arm in range(2):
            bias = self.data.qfrc_bias[self.arm_dof[arm]]
            torque[arm] = np.clip(
                kp * (targets[arm] - q[arm]) - kd * qd[arm] + bias,
                -limits,
                limits,
            )
        self.data.ctrl[:] = torque.reshape(-1)
        for _ in range(self.cfg.control_interval):
            mujoco.mj_step(self.model, self.data)

    def _ee_linear_speeds(self) -> np.ndarray:
        speeds = []
        for site in self.ee_ids:
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, int(site), vel, 0
            )
            speeds.append(float(np.linalg.norm(vel[3:])))
        return np.asarray(speeds)

    def _bilateral_grasp_ready(self) -> tuple[bool, np.ndarray]:
        object_pos, _ = self._workpiece_pose()
        targets = self._grasp_target_positions(object_pos, 0.006)
        distances = np.asarray([
            np.linalg.norm(targets[arm] - self.data.site_xpos[self.ee_ids[arm]])
            for arm in range(2)
        ])
        speeds = self._ee_linear_speeds()
        symmetric = abs(distances[0] - distances[1]) < 0.020
        ready = bool(
            np.all(distances < self.grasp_distance_threshold_m)
            and np.all(speeds < self.grasp_speed_threshold_mps)
            and symmetric
        )
        return ready, distances

    def enable_cooperative_grasp(self) -> None:
        """Activate damped welds only after the bilateral grasp gate passes."""
        if self.grasped:
            return
        obj_pos, _ = self._workpiece_pose()
        obj_rot = self.data.xmat[self.workpiece_body].reshape(3, 3).copy()
        for arm, site_id in enumerate(self.ee_ids):
            site_rot = self.data.site_xmat[site_id].reshape(3, 3)
            self.grasp_site_pos_local[arm] = obj_rot.T @ (
                self.data.site_xpos[site_id] - obj_pos
            )
            self.grasp_site_rot_local[arm] = obj_rot.T @ site_rot

        # Configure weld anchors from the actual contact-time pose.  No workpiece
        # qpos is overwritten here or anywhere else during a rollout.
        for eqid, arm_name in enumerate(("left_panda_link8", "right_panda_link8")):
            arm_body = self.model.body(arm_name).id
            arm_mat = self.data.xmat[arm_body].reshape(3, 3)
            # A completed assembly repurposes the first weld as the installed
            # receiver/workpiece joint. Restore both grasp weld endpoints at
            # the beginning of every new pickup.
            self.model.eq_obj1id[eqid] = arm_body
            self.model.eq_obj2id[eqid] = self.workpiece_body
            self.model.eq_solref[eqid] = np.array([0.08, 1.0])
            self.model.eq_data[eqid, 3:6] = arm_mat.T @ (
                obj_pos - self.data.xpos[arm_body]
            )
            self.model.eq_data[eqid, 6:10] = _mat_quat(arm_mat.T @ obj_rot)

        self.data.eq_active[:] = 1
        mujoco.mj_forward(self.model, self.data)
        self.grasped = True
        self.grasp_step = self.step_count
        self.stage = AssemblyStage.GRASP
        self.phase = "GRASP"
        self.desired_pose = np.r_[obj_pos, self.data.xquat[self.workpiece_body].copy()]
        self.previous_lift_z = float(obj_pos[2])

    def _pregrasp_control(
        self,
        trajectory_action: np.ndarray,
        impedance_action: np.ndarray,
    ) -> dict[str, Any]:
        """Physical tabletop approach; workpiece remains a free simulated body."""
        del trajectory_action, impedance_action  # deterministic safe pickup prior

        def approach_step(targets: np.ndarray, kp: float, kd: float) -> None:
            # Pickup uses two short servo intervals per policy step so the safe
            # deterministic prior reaches bilateral contact before RL control
            # takes over. Recomputing PD torque between intervals avoids using
            # a stale saturated command over one long integration interval.
            self._joint_pd_step(targets, kp=kp, kd=kd)
            self._joint_pd_step(targets, kp=kp, kd=kd)

        if self.phase == "TABLE_PREGRASP":
            approach_step(self.hover_q_targets, kp=70.0, kd=18.0)
            err = np.max(np.abs(self.data.qpos[self.arm_qpos] - self.hover_q_targets))
            if err < 0.08:
                self.phase = "DESCEND_TO_GRASP"

        elif self.phase in ("DESCEND_TO_GRASP", "BILATERAL_CONTACT"):
            approach_step(self.grasp_q_targets, kp=65.0, kd=20.0)
            ready, distances = self._bilateral_grasp_ready()
            if ready:
                self.phase = "BILATERAL_CONTACT"
                self.grasp_ready_steps += 1
            else:
                self.grasp_ready_steps = 0
                if self.phase == "BILATERAL_CONTACT":
                    self.phase = "DESCEND_TO_GRASP"
            if self.grasp_ready_steps >= self.grasp_dwell_required:
                self.enable_cooperative_grasp()
            return {
                "grasp_distance_left_m": float(distances[0]),
                "grasp_distance_right_m": float(distances[1]),
                "grasp_ready_steps": int(self.grasp_ready_steps),
            }

        else:
            approach_step(self.grasp_q_targets, kp=60.0, kd=20.0)

        ready, distances = self._bilateral_grasp_ready()
        return {
            "grasp_distance_left_m": float(distances[0]),
            "grasp_distance_right_m": float(distances[1]),
            "grasp_ready_steps": int(self.grasp_ready_steps),
            "grasp_gate": bool(ready),
        }

    def _pregrasp_reward(self, info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        d_left = float(info.get("grasp_distance_left_m", 0.25))
        d_right = float(info.get("grasp_distance_right_m", 0.25))
        r_reach = -1.5 * (d_left + d_right)
        r_symmetry = -0.5 * abs(d_left - d_right)
        r_gate = 1.0 if info.get("grasp_gate", False) else 0.0
        r_grasp = 8.0 if self.grasped else 0.0
        reward = r_reach + r_symmetry + r_gate + r_grasp - 0.01
        return reward, {
            "r_pickup_reach": r_reach,
            "r_pickup_symmetry": r_symmetry,
            "r_pickup_gate": r_gate,
            "r_grasp": r_grasp,
        }

    # ------------------------------------------------------------------
    # Cooperative grasp / compliance
    # ------------------------------------------------------------------

    def _internal_force_wrench(self, force_N: float) -> np.ndarray:
        axis = self.data.site_xpos[self.ee_ids[1]] - self.data.site_xpos[self.ee_ids[0]]
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        wrench = np.zeros((2, 6), dtype=float)
        wrench[0, :3] = float(force_N) * axis
        wrench[1, :3] = -float(force_N) * axis
        return wrench

    def _grasp_metrics(
        self,
        contact_wrench: np.ndarray,
        task_wrench: np.ndarray,
        internal_force_N: float,
    ) -> tuple[float, float, float]:
        del task_wrench  # commanded task wrench is not a measured grasp load
        span = max(
            float(
                np.linalg.norm(
                    self.data.site_xpos[self.ee_ids[1]]
                    - self.data.site_xpos[self.ee_ids[0]]
                )
            ),
            0.05,
        )
        contact_load = float(
            np.linalg.norm(contact_wrench[:3])
            + np.linalg.norm(contact_wrench[3:]) / span
        )
        payload_weight = float(
            self.cfg.workpiece_mass * np.linalg.norm(self.model.opt.gravity)
        )
        # The prior implementation used the commanded OSC wrench as the grasp
        # load. During lift that command contains position-error feedback and
        # can be much larger than the physical payload, producing false slip
        # events. Receiver contact and payload weight are physical external
        # loads; use the larger of those for the friction-capacity check.
        load = max(contact_load, payload_weight)
        capacity = float(2.0 * self.cfg.grasp_friction * internal_force_N)
        return capacity, load, capacity - load

    def _update_grasp_failure(self, margin_N: float) -> bool:
        if self.assembly_complete or self.phase in ("RELEASE", "RETREAT", "SUCCESS"):
            return False
        self.grasp_overload_steps = self.grasp_overload_steps + 1 if margin_N < 0.0 else 0
        if self.grasp_overload_steps < self.cfg.grasp_slip_steps:
            return False
        self.data.eq_active[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.grasped = False
        self.grasp_failed = True
        self.phase = "GRASP_SLIP"
        return True

    def _decode_impedance(self, imp: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Map Agent-2 action to stage-aware compliant K/D and internal force.

        Contact stages deliberately permit much lower lateral/rotational stiffness
        than free-space transport.  The policy still selects continuously inside
        each safe range, so compliant behaviour is learned rather than hard-coded.
        """
        imp = np.clip(np.asarray(imp, dtype=float), 0.0, 1.0)
        kp_parallel_u, kp_lateral_u, kp_rot_u = imp[:3]
        d_parallel_u, d_lateral_u, d_rot_u = imp[3:6]

        contact_phase = self.phase in (
            "FIRST_CONTACT",
            "COMPLIANT_ALIGNMENT",
            "INSERTION",
            "STABILIZATION",
            "JAM_RECOVERY",
        )
        jam_phase = self.jam_recovery or self.phase == "JAM_RECOVERY"

        if jam_phase:
            parallel_range = (45.0, 125.0)
            lateral_range = (18.0, 75.0)
            rotation_range = (4.0, 16.0)
        elif contact_phase:
            parallel_range = (90.0, 260.0)
            lateral_range = (25.0, 145.0)
            rotation_range = (5.0, 28.0)
        else:
            parallel_range = (180.0, 360.0)
            lateral_range = (160.0, 340.0)
            rotation_range = (20.0, 60.0)

        k_parallel = parallel_range[0] + kp_parallel_u * (parallel_range[1] - parallel_range[0])
        k_lateral = lateral_range[0] + kp_lateral_u * (lateral_range[1] - lateral_range[0])
        k_rotation = rotation_range[0] + kp_rot_u * (rotation_range[1] - rotation_range[0])
        kp = np.array([
            k_lateral,
            k_parallel,
            k_lateral,
            k_rotation,
            k_rotation,
            k_rotation,
        ])

        # Damping stays bounded and increases with the learned damping action.
        # Exact critical damping cannot be guaranteed without a task-space mass
        # estimate, so this remains the repository's safe empirical K/D mapping.
        kd_parallel = 14.0 + 30.0 * d_parallel_u
        kd_lateral = 8.0 + 24.0 * d_lateral_u
        kd_rotation = 0.8 + 5.5 * d_rot_u
        kd = np.array([
            kd_lateral,
            kd_parallel,
            kd_lateral,
            kd_rotation,
            kd_rotation,
            kd_rotation,
        ])

        if contact_phase:
            # Preserve lower insertion-axis stiffness while giving the two
            # lateral axes and orientation enough authority to meet the strict
            # 1 mm / 2 degree stabilization tolerances under payload gravity.
            kp[[0, 2]] *= 1.9
            kp[3:] *= 2.1

        internal_force_N = self.cfg.internal_force_min + imp[6] * (
            self.cfg.internal_force_max - self.cfg.internal_force_min
        )
        return kp, kd, float(internal_force_N)

    # ------------------------------------------------------------------
    # Contact / stage machine
    # ------------------------------------------------------------------

    def _enable_assembly_retention(self) -> None:
        """Latch a strictly seated workpiece to the dynamic receiver."""
        receiver_rot = self.data.xmat[self.receiver_body].reshape(3, 3).copy()
        installed_pose = self.object_target_pose()
        object_rot_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(object_rot_flat, installed_pose[3:])
        object_rot = object_rot_flat.reshape(3, 3)
        self.model.eq_obj1id[0] = self.receiver_body
        self.model.eq_obj2id[0] = self.workpiece_body
        self.model.eq_solref[0] = np.array([0.001, 1.0])
        self.model.eq_data[0, 3:6] = receiver_rot.T @ (
            installed_pose[:3] - self.data.xpos[self.receiver_body]
        )
        self.model.eq_data[0, 6:10] = _mat_quat(receiver_rot.T @ object_rot)
        # Keep the second arm attached during the short release hold. The first
        # weld now represents engagement of the fully seated dual-peg joint.
        self.data.eq_active[:] = 1
        mujoco.mj_forward(self.model, self.data)

    def _contact_state(self, wrench: np.ndarray | None = None) -> bool:
        force = self._contact_wrench() if wrench is None else np.asarray(wrench)
        if np.linalg.norm(force[:3]) > 1e-4:
            return True
        lateral, _ = self.peg_errors()
        mouth = max(
            abs(float(self._hole_pose(i)[0][1] - self.peg_tip(i)[1]))
            for i in range(2)
        )
        return bool(mouth < 0.006 and max(lateral) < self.cfg.hole_radius + 0.006)

    def _insertion_quality_ok(self, lat: np.ndarray, dep: np.ndarray, wrench: np.ndarray) -> bool:
        return bool(
            np.all(lat < self.cfg.lateral_threshold)
            and np.all(dep > self.cfg.required_depth)
            and self._relative_orientation_error() < self.cfg.orientation_threshold_deg
            and np.linalg.norm(wrench[:3]) < self.cfg.max_force
            and np.linalg.norm(wrench[3:]) < self.cfg.max_torque
        )

    def _update_stage(self, contact: bool, lat: np.ndarray, dep: np.ndarray) -> None:
        if self.phase in ("RELEASE", "RETREAT", "SUCCESS", "GRASP_SLIP"):
            return
        if not self.grasped:
            self.stage = AssemblyStage.INITIALIZATION
            return

        pos, _ = self._workpiece_pose()
        orientation = self._relative_orientation_error()

        if self.stage == AssemblyStage.GRASP:
            self.stage = AssemblyStage.LIFT
            self.phase = "LIFT"

        elif self.stage == AssemblyStage.LIFT:
            if pos[2] >= TABLE_OBJECT_Z + self.lift_height_m - 0.020:
                self.stage = AssemblyStage.TRANSPORT
                self.phase = "TRANSPORT"

        elif self.stage == AssemblyStage.TRANSPORT:
            transport_target = self._standoff_pose(self.transport_standoff_m, keep_lift_height=True)
            if np.linalg.norm(pos - transport_target[:3]) < 0.055:
                self.stage = AssemblyStage.COARSE_ALIGNMENT
                self.phase = "COARSE_ALIGNMENT"

        elif self.stage == AssemblyStage.COARSE_ALIGNMENT:
            # Align the peg tips inside the available radial clearance before
            # advancing toward the receiver face. A centimetre-scale threshold
            # lets the pegs hit the plate instead of entering the holes.
            precontact_clearance = max(
                self.cfg.hole_radius - self.cfg.peg_radius,
                self.cfg.lateral_threshold,
            )
            if (
                np.max(lat) < 2.5 * precontact_clearance
                and orientation < 5.0
            ):
                self.alignment_ready = True
                self.stage = AssemblyStage.FIRST_CONTACT
                self.phase = "FIRST_CONTACT"

        elif self.stage == AssemblyStage.FIRST_CONTACT:
            if contact:
                self.stage = AssemblyStage.COMPLIANT_ALIGNMENT
                self.phase = "COMPLIANT_ALIGNMENT"

        elif self.stage == AssemblyStage.COMPLIANT_ALIGNMENT:
            if np.max(dep) > 0.0015:
                self.stage = AssemblyStage.INSERTION
                self.phase = "INSERTION"

        if self.jam_recovery and self.stage in (
            AssemblyStage.FIRST_CONTACT,
            AssemblyStage.COMPLIANT_ALIGNMENT,
            AssemblyStage.INSERTION,
        ):
            self.phase = "JAM_RECOVERY"
        elif self.phase == "JAM_RECOVERY" and not self.jam_recovery:
            self.phase = "INSERTION" if np.max(dep) > 0.0015 else "COMPLIANT_ALIGNMENT"

        if self.stage == AssemblyStage.INSERTION and np.all(
            dep > self.cfg.required_depth * 0.8
        ):
            self.phase = "STABILIZATION"

        wrench = self._contact_wrench()
        if self._insertion_quality_ok(lat, dep, wrench):
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.stable_count >= self.cfg.stable_steps:
            self.assembly_complete = True
            self._enable_assembly_retention()
            self.phase = "RELEASE"
            self.release_q_targets = self.data.qpos[self.arm_qpos].copy()
            self.release_steps = 0

    def _release_retreat_step(self) -> tuple[float, bool, dict[str, Any]]:
        """Release the assembled part and move both wrists away before SUCCESS."""
        if self.phase == "RELEASE":
            # Hold the wrists briefly so constraint removal does not create an
            # impulsive motion in the installed workpiece.
            assert self.release_q_targets is not None
            self._joint_pd_step(self.release_q_targets, kp=55.0, kd=18.0)
            self.release_steps += 1
            if self.release_steps >= self.release_hold_required:
                self.data.eq_active[:] = 0
                self.data.eq_active[0] = 1
                mujoco.mj_forward(self.model, self.data)
                self.grasped = False

                ee_left = self.data.site_xpos[self.ee_ids[0]].copy()
                ee_right = self.data.site_xpos[self.ee_ids[1]].copy()
                retreat_targets = (
                    ee_left + np.array([-0.080, +0.050, +0.025]),
                    ee_right + np.array([+0.080, +0.050, +0.025]),
                )
                self.retreat_q_targets = self._solve_ee_position_targets(
                    retreat_targets,
                    initial_q=self.data.qpos[self.arm_qpos].copy(),
                )
                self.phase = "RETREAT"
                self.retreat_steps = 0

            return 3.0, False, {"release_complete": False, "retreat_complete": False}

        assert self.retreat_q_targets is not None
        self._joint_pd_step(self.retreat_q_targets, kp=55.0, kd=18.0)
        self.retreat_steps += 1
        q_err = float(
            np.max(np.abs(self.data.qpos[self.arm_qpos] - self.retreat_q_targets))
        )
        lat, dep = self.peg_errors()
        # After removing the welds, the installed part must remain within a
        # slightly relaxed tolerance; otherwise the assembly is not accepted.
        retained = bool(
            np.all(dep > self.cfg.required_depth * 0.90)
            and np.all(lat < max(self.cfg.lateral_threshold * 1.5, 0.002))
        )
        retreat_done = self.retreat_steps >= self.retreat_required and q_err < 0.12

        if retreat_done and retained:
            self.stage = AssemblyStage.SUCCESS
            self.phase = "SUCCESS"
            self.success = True
            return 500.0, True, {"release_complete": True, "retreat_complete": True}

        if retreat_done and not retained:
            # Do not terminate immediately; leave time for the episode timeout
            # and record a physically meaningful post-release failure.
            self.assembly_complete = False

        return 1.0 if retained else -4.0, False, {
            "release_complete": True,
            "retreat_complete": retreat_done,
            "post_release_retained": retained,
        }

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(self, trajectory_action: np.ndarray, impedance_action: np.ndarray):
        if self.step_count >= self.cfg.max_steps:
            return self.observations(), -10.0, True, {
                "phase": self.phase,
                "stage": self.stage.name,
                "success": self.success,
            }

        self._receiver_disturbance()
        action = np.asarray(trajectory_action, dtype=float)
        imp = np.asarray(impedance_action, dtype=float)
        if action.shape != (6,):
            raise ValueError("trajectory action must be 6D")
        if imp.shape != (7,) or not np.all(np.isfinite(imp)):
            raise ValueError(
                "Agent 2 action must be 7D "
                "[Kparallel,Klateral,Krotation,Dparallel,Dlateral,Drotation,Finternal]"
            )

        # ----------------------- TABLETOP PICKUP -----------------------
        if not self.grasped and self.phase not in ("RELEASE", "RETREAT", "SUCCESS"):
            pickup_info = self._pregrasp_control(action, imp)
            self.step_count += 1
            pickup_reward, pickup_reward_info = self._pregrasp_reward(pickup_info)
            info = {
                "phase": self.phase,
                "stage": self.stage.name,
                "stage_index": int(self.stage),
                "success": False,
                "contact": False,
                "jamming": False,
                "grasped": self.grasped,
                "grasp_failure": False,
                "agent1_action": action.copy(),
                "agent1_reward": float(pickup_reward),
                "agent2_reward": 0.0,
                "wrench": np.zeros(6),
                "kp": np.ones(6),
                "kd": np.ones(6),
                "impedance": np.r_[np.ones(6), np.ones(6)],
                "internal_force_N": 0.0,
                "grasp_capacity_N": 0.0,
                "grasp_load_N": 0.0,
                "grasp_margin_N": 0.0,
                **pickup_info,
                **pickup_reward_info,
            }
            self.log.append(info)
            return self.observations(), float(pickup_reward), False, info

        # --------------------- RELEASE / RETREAT ----------------------
        if self.phase in ("RELEASE", "RETREAT"):
            release_reward, done, release_info = self._release_retreat_step()
            self.step_count += 1
            lat, dep = self.peg_errors()
            info = {
                "phase": self.phase,
                "stage": self.stage.name,
                "stage_index": int(self.stage),
                "success": self.success,
                "contact": self._contact_state(),
                "jamming": False,
                "grasped": self.grasped,
                "grasp_failure": False,
                "agent1_action": action.copy(),
                "agent1_reward": float(release_reward),
                "agent2_reward": 0.0,
                "wrench": self._contact_wrench().copy(),
                "kp": np.zeros(6),
                "kd": np.zeros(6),
                "impedance": np.zeros(12),
                "internal_force_N": 0.0,
                "peg1_lateral_error": float(lat[0]),
                "peg2_lateral_error": float(lat[1]),
                "peg1_depth": float(dep[0]),
                "peg2_depth": float(dep[1]),
                "relative_orientation_error": self._relative_orientation_error(),
                **release_info,
            }
            self.log.append(info)
            return self.observations(), float(release_reward), bool(done), info

        # ----------------- A2PO POST-GRASP CONTROL --------------------
        # Agent 1 is a 6-D motion residual around the stage-aware deterministic
        # prior supplied by the formal-study wrapper.  Agent 2 selects adaptive
        # impedance and grasp internal force.
        self._desired_from_action(action)

        old_action = self.previous_action.copy()
        old_impedance = self.previous_impedance.copy()
        old_lat, old_dep = self.peg_errors()
        old_pos, _ = self._workpiece_pose()

        imp = np.clip(imp, 0.0, 1.0)
        kp, kd, internal_force_N = self._decode_impedance(imp)

        desired_rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(desired_rotation_flat, self.desired_pose[3:])
        desired_rotation = desired_rotation_flat.reshape(3, 3)

        jac = np.zeros((2, 6, 7))
        q = self.data.qpos[self.arm_qpos]
        qd = self.data.qvel[self.arm_dof]
        arm_wrench = np.zeros((2, 6))

        for arm, site_id in enumerate(self.ee_ids):
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(
                self.model, self.data, jp, jr, int(site_id)
            )
            jac[arm] = np.vstack((
                jp[:, self.arm_dof[arm]],
                jr[:, self.arm_dof[arm]],
            ))

            target_pos = (
                self.desired_pose[:3]
                + desired_rotation @ self.grasp_site_pos_local[arm]
            )
            target_rot = desired_rotation @ self.grasp_site_rot_local[arm]
            current_rot = self.data.site_xmat[site_id].reshape(3, 3)
            rotation_error = _rotvec(_mat_quat(target_rot @ current_rot.T))

            site_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_SITE,
                int(site_id),
                site_velocity,
                0,
            )
            # MuJoCo spatial velocity is [angular, linear].
            linear_v = site_velocity[3:]
            angular_v = site_velocity[:3]
            arm_wrench[arm, :3] = 0.5 * (
                kp[:3] * (target_pos - self.data.site_xpos[site_id])
                - kd[:3] * linear_v
            )
            # qfrc_bias compensates each arm but not the free payload held by
            # the cooperative welds. Give each arm one payload-weight of
            # support: the extra margin covers Jacobian/torque-limit losses in
            # the over-constrained bimanual grasp, while the task-force clamp
            # below still enforces max_force.
            arm_wrench[arm, :3] -= (
                float(self.cfg.workpiece_mass)
                * np.asarray(self.model.opt.gravity)
            )
            arm_wrench[arm, 3:] = 0.5 * (
                kp[3:] * rotation_error - kd[3:] * angular_v
            )

            arm_wrench[arm, :3] /= max(
                1.0,
                np.linalg.norm(arm_wrench[arm, :3]) / (0.5 * self.cfg.max_force),
            )
            arm_wrench[arm, 3:] /= max(
                1.0,
                np.linalg.norm(arm_wrench[arm, 3:]) / (0.5 * self.cfg.max_torque),
            )

        task_wrench = np.sum(arm_wrench, axis=0)
        internal_wrench = self._internal_force_wrench(internal_force_N)
        arm_wrench += internal_wrench

        mapped = self.mapper.compute(
            arm_wrench,
            jac,
            q,
            qd,
            np.repeat(self.rest_q[None, :], 2, axis=0),
            feedforward_torque=np.asarray([
                self.data.qfrc_bias[self.arm_dof[arm]] for arm in range(2)
            ]),
            stopped=False,
        )
        self.data.ctrl[:] = mapped.joint_torque.reshape(-1)
        for _ in range(self.cfg.control_interval):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        wrench = self._contact_wrench()
        lat, dep = self.peg_errors()
        contact_now = self._contact_state(wrench)

        self.contact_steps = self.contact_steps + 1 if contact_now else 0
        depth_sum = float(np.sum(dep))
        progress = depth_sum - self.previous_depth_sum
        self.no_progress_steps = (
            self.no_progress_steps + 1
            if contact_now and progress < 1e-5
            else 0
        )
        jam_detected = bool(
            contact_now
            and np.linalg.norm(wrench[:3]) > 8.0
            and self.no_progress_steps >= 8
        )
        if jam_detected:
            if self.jam_recovery_steps == 0:
                self.jam_backoff_y = float(self._workpiece_pose()[0][1] + 0.025)
            self.jam_recovery_steps = max(self.jam_recovery_steps, 50)
        elif self.jam_recovery_steps > 0:
            self.jam_recovery_steps -= 1
            if self.jam_recovery_steps == 0:
                self.jam_backoff_y = None
        self.jam_recovery = self.jam_recovery_steps > 0

        self._update_stage(contact_now, lat, dep)
        self.previous_depth_sum = depth_sum
        self.previous_action = action.copy()
        self.previous_impedance = imp.copy()
        self.last_wrench = wrench.copy()
        self.last_depth = dep.copy()

        grasp_capacity_N, grasp_load_N, grasp_margin_N = self._grasp_metrics(
            wrench, task_wrench, internal_force_N
        )
        grasp_failure = self._update_grasp_failure(grasp_margin_N)

        # -------------------------- REWARD -----------------------------
        # Agent 1: task progress / coordination.
        current_pos, _ = self._workpiece_pose()
        stage_target = self.control_target_pose()
        r_stage_distance = -0.8 * float(np.linalg.norm(stage_target[:3] - current_pos))
        lift_progress = max(0.0, float(current_pos[2] - old_pos[2]))
        r_lift = 25.0 * lift_progress if self.phase in ("GRASP", "LIFT", "TRANSPORT") else 0.0

        contact_task_active = self.stage in (
            AssemblyStage.COARSE_ALIGNMENT,
            AssemblyStage.FIRST_CONTACT,
            AssemblyStage.COMPLIANT_ALIGNMENT,
            AssemblyStage.INSERTION,
        ) or self.phase in ("STABILIZATION", "JAM_RECOVERY")
        r_align_peg1 = -8.0 * float(lat[0]) if contact_task_active else 0.0
        r_align_peg2 = -8.0 * float(lat[1]) if contact_task_active else 0.0
        r_orientation = -0.08 * self._relative_orientation_error() if contact_task_active else 0.0
        r_depth = (
            18.0 * float(np.sum(dep) / max(2 * self.cfg.required_depth, 1e-6))
            if contact_task_active
            else 0.0
        )
        r_progress = 80.0 * progress if contact_task_active else 0.0
        r_action = -0.05 * float(np.linalg.norm(action - old_action))

        # Agent 2: safe compliance / grasp stability.
        force_norm = float(np.linalg.norm(wrench[:3]))
        torque_norm = float(np.linalg.norm(wrench[3:]))
        r_force = -0.002 * force_norm**2 if contact_task_active else 0.0
        r_torque = -0.001 * torque_norm**2 if contact_task_active else 0.0
        r_jam = -10.0 if self.jam_recovery else 0.0
        r_imp = -0.05 * float(np.linalg.norm(imp - old_impedance))
        r_internal_effort = -0.05 * (
            internal_force_N / max(self.cfg.internal_force_max, 1e-6)
        ) ** 2
        r_grasp_margin = -0.01 * min(0.0, grasp_margin_N) ** 2
        r_grasp_failure = -100.0 if grasp_failure else 0.0

        # Final +500 is awarded only after release + retreat succeeds, not merely
        # when the pegs first satisfy geometric insertion thresholds.
        r_success = 0.0

        agent1_reward = (
            r_stage_distance
            + r_lift
            + r_align_peg1
            + r_align_peg2
            + r_orientation
            + r_depth
            + r_progress
            + r_action
            + r_success
        )
        agent2_reward = (
            r_force
            + r_torque
            + r_jam
            + r_imp
            + r_internal_effort
            + r_grasp_margin
            + r_grasp_failure
            + r_success
        )
        reward = agent1_reward + agent2_reward

        info = {
            "phase": self.phase,
            "stage": self.stage.name,
            "stage_index": int(self.stage),
            "success": self.success,
            "assembly_complete": self.assembly_complete,
            "contact": contact_now,
            "jamming": self.jam_recovery,
            "grasped": self.grasped,
            "grasp_failure": grasp_failure,
            "agent1_action": action.copy(),
            "agent1_reward": float(agent1_reward),
            "agent2_reward": float(agent2_reward),
            "r_stage_distance": r_stage_distance,
            "r_lift": r_lift,
            "r_align_peg1": r_align_peg1,
            "r_align_peg2": r_align_peg2,
            "r_orientation": r_orientation,
            "r_depth": r_depth,
            "r_insertion_progress": r_progress,
            "r_force": r_force,
            "r_torque": r_torque,
            "r_jam": r_jam,
            "r_action_smoothness": r_action,
            "r_impedance_smoothness": r_imp,
            "r_internal_effort": r_internal_effort,
            "r_grasp_margin": r_grasp_margin,
            "r_grasp_failure": r_grasp_failure,
            "r_success": r_success,
            "wrench": wrench.copy(),
            "kp": kp.copy(),
            "kd": kd.copy(),
            "impedance": np.r_[kp, kd],
            "internal_force_N": float(internal_force_N),
            "internal_wrench": internal_wrench.copy(),
            "grasp_capacity_N": grasp_capacity_N,
            "grasp_load_N": grasp_load_N,
            "grasp_margin_N": grasp_margin_N,
            "peg1_lateral_error": float(lat[0]),
            "peg2_lateral_error": float(lat[1]),
            "peg1_depth": float(dep[0]),
            "peg2_depth": float(dep[1]),
            "relative_orientation_error": self._relative_orientation_error(),
            "table_clearance_m": float(current_pos[2] - TABLE_TOP_Z),
            "object_height_m": float(current_pos[2]),
            "alignment_ready": self.alignment_ready,
        }
        self.log.append(info)
        return self.observations(), float(reward), bool(self.success or grasp_failure), info
