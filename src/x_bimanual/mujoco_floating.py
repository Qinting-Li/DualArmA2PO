"""MuJoCo dual-arm peg-in-hole environment with a dynamic 6-DoF target.

The target panel is a genuine rigid body in the floating modes.  Its pose and
velocity are initialized once per episode, then are advanced only by MuJoCo's
mass/inertia/contact solver.  The small carrier is the common grasp frame for
the left and right grippers; a weld transfers contact wrench from the peg to
the two-arm carrier while the six carrier joints are driven by a compliant
Cartesian controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

Array = np.ndarray


class ExperimentMode(str, Enum):
    FIXED = "fixed"
    FLOATING_ZERO_VELOCITY = "floating_zero_velocity"
    FLOATING_RANDOM_VELOCITY = "floating_random_velocity"


class RelativeStage(str, Enum):
    APPROACH = "APPROACH"
    ALIGNMENT = "ALIGNMENT"
    CONTACT_SEARCH = "CONTACT_SEARCH"
    INSERTION = "INSERTION"
    JAM_RECOVERY = "JAM_RECOVERY"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass(frozen=True)
class FloatingConfig:
    timestep: float = 0.002
    control_interval: int = 4
    target_mass: float = 1.2
    target_inertia: tuple[float, float, float] = (0.0018, 0.0018, 0.0028)
    position_offset: float = 0.015
    rotation_deg: float = 4.0
    linear_velocity: float = 0.012
    angular_velocity_dps: float = 1.5
    target_contact_damping: float = 0.8
    target_friction: float = 0.55
    peg_mass: float = 0.30
    peg_length: float = 0.160
    peg_radius: float = 0.018
    hole_half_width: float = 0.024
    panel_thickness: float = 0.040
    required_depth: float = 0.030
    lateral_threshold: float = 0.003
    angular_threshold_deg: float = 5.0
    safe_force: float = 45.0
    safe_torque: float = 8.0
    mass: tuple[float, ...] = (0.25, 0.25, 0.25, 0.015, 0.015, 0.015)
    kp: tuple[float, ...] = (110.0, 110.0, 140.0, 3.0, 3.0, 3.0)
    damping_ratio: float = 1.0
    max_force: float = 35.0
    max_torque: float = 5.0


def config_from_mapping(mapping: dict[str, Any]) -> FloatingConfig:
    """Create a config from the ``mujoco`` section of task.yaml."""
    m = mapping.get("mujoco", mapping)
    c = m.get("controller", {})
    return FloatingConfig(
        timestep=float(m.get("timestep", 0.002)),
        control_interval=int(m.get("control_interval", 4)),
        target_mass=float(m.get("target_mass_kg", 1.2)),
        target_inertia=tuple(float(v) for v in m.get("target_inertia", (0.0018, 0.0018, 0.0028))),
        position_offset=float(m.get("target_position_offset_m", 0.015)),
        rotation_deg=float(m.get("target_rotation_deg", 4.0)),
        linear_velocity=float(m.get("target_linear_velocity_mps", 0.012)),
        angular_velocity_dps=float(m.get("target_angular_velocity_dps", 1.5)),
        target_contact_damping=float(m.get("target_contact_damping", 0.8)),
        target_friction=float(m.get("target_friction", 0.55)),
        peg_mass=float(m.get("peg_mass_kg", 0.30)),
        peg_length=float(m.get("peg_length_m", 0.160)),
        peg_radius=float(m.get("peg_radius_m", 0.018)),
        hole_half_width=float(m.get("hole_half_width_m", 0.024)),
        panel_thickness=float(m.get("panel_thickness_m", 0.040)),
        required_depth=float(m.get("required_insertion_depth_m", 0.030)),
        lateral_threshold=float(m.get("success_lateral_error_m", 0.003)),
        angular_threshold_deg=float(m.get("success_angular_error_deg", 5.0)),
        safe_force=float(m.get("safe_contact_force_N", 45.0)),
        safe_torque=float(m.get("safe_contact_torque_Nm", 8.0)),
        mass=tuple(float(v) for v in c.get("mass", (0.25, 0.25, 0.25, 0.015, 0.015, 0.015))),
        kp=tuple(float(v) for v in c.get("kp", (110.0, 110.0, 140.0, 3.0, 3.0, 3.0))),
        damping_ratio=float(c.get("damping_ratio", 1.0)),
        max_force=float(c.get("max_force_N", 35.0)),
        max_torque=float(c.get("max_torque_Nm", 5.0)),
    )


def _xml(cfg: FloatingConfig, floating_target: bool) -> str:
    """Build a minimal, deterministic MuJoCo scene.

    In floating mode the target body contains ``<freejoint/>`` and an
    inertial.  In fixed mode it is a static body with the same collision
    geometry, keeping the baseline's geometry and controller identical.
    """
    target_joint = '<freejoint name="target_free" />' if floating_target else ""
    target_inertial = (
        f'<inertial pos="0 0 0" mass="{cfg.target_mass}" '
        f'diaginertia="{cfg.target_inertia[0]} {cfg.target_inertia[1]} {cfg.target_inertia[2]}" />'
        if floating_target
        else ""
    )
    hw = cfg.hole_half_width + 0.019
    wall = 0.019
    panel = cfg.panel_thickness
    return f'''<mujoco model="bimanual_floating_peg_hole">
  <compiler angle="radian" coordinate="local" />
  <option timestep="{cfg.timestep}" gravity="0 0 0" integrator="implicitfast" cone="elliptic" />
  <size njmax="2000" nconmax="400" />
  <default>
    <joint damping="0.15" armature="0.002" />
    <geom contype="1" conaffinity="1" friction="{cfg.target_friction} 0.04 0.01" solref="0.008 1" solimp="0.85 0.95 0.01" />
  </default>
  <worldbody>
    <light name="key" pos="1 -1 1.5" dir="-1 1 -1" diffuse="0.8 0.8 0.8" />
    <geom name="floor_visual" type="plane" size="2 2 0.01" pos="0 0 -0.12" contype="0" conaffinity="0" rgba="0.12 0.14 0.17 1" />
    <body name="grasp_carrier" pos="0 0 0">
      <joint name="carrier_x" type="slide" axis="1 0 0" range="-0.5 0.5" />
      <joint name="carrier_y" type="slide" axis="0 1 0" range="-0.5 0.5" />
      <joint name="carrier_z" type="slide" axis="0 0 1" range="-0.1 0.5" />
      <joint name="carrier_roll" type="hinge" axis="1 0 0" range="-1.0 1.0" />
      <joint name="carrier_pitch" type="hinge" axis="0 1 0" range="-1.0 1.0" />
      <joint name="carrier_yaw" type="hinge" axis="0 0 1" range="-1.0 1.0" />
      <inertial pos="0 0 0" mass="0.18" diaginertia="0.0008 0.0008 0.0008" />
      <geom name="left_gripper" type="box" size="0.012 0.028 0.018" pos="-0.030 0 0.120" rgba="0.15 0.35 0.85 1" />
      <geom name="right_gripper" type="box" size="0.012 0.028 0.018" pos="0.030 0 0.120" rgba="0.15 0.35 0.85 1" />
      <geom name="carrier_crossbar" type="box" size="0.045 0.012 0.012" pos="0 0 0.120" contype="0" conaffinity="0" rgba="0.18 0.22 0.30 1" />
      <body name="left_arm" pos="-0.16 0 0.12">
        <site name="left_upper_link" type="capsule" fromto="0 0 0 0.07 0 0" size="0.018" rgba="0.22 0.42 0.82 1" />
        <site name="left_forearm" type="capsule" fromto="0.07 0 0 0.13 0 0" size="0.014" rgba="0.25 0.48 0.90 1" />
      </body>
      <body name="right_arm" pos="0.16 0 0.12">
        <site name="right_upper_link" type="capsule" fromto="0 0 0 -0.07 0 0" size="0.018" rgba="0.22 0.42 0.82 1" />
        <site name="right_forearm" type="capsule" fromto="-0.07 0 0 -0.13 0 0" size="0.014" rgba="0.25 0.48 0.90 1" />
      </body>
      <site name="left_wrist" pos="-0.030 0 0.120" size="0.006" rgba="0.1 0.3 1 1" />
      <site name="right_wrist" pos="0.030 0 0.120" size="0.006" rgba="0.1 0.3 1 1" />
    </body>
    <body name="peg" pos="0 0 0.120">
      <freejoint name="peg_free" />
      <inertial pos="0 0 0" mass="{cfg.peg_mass}" diaginertia="0.0007 0.0007 0.00005" />
      <geom name="peg_geom" type="cylinder" size="{cfg.peg_radius} {cfg.peg_length / 2}" rgba="0.78 0.16 0.08 1" />
      <site name="peg_tip" pos="0 0 {-cfg.peg_length / 2}" size="0.004" rgba="1 0.2 0.1 1" />
    </body>
    <body name="target_panel" pos="0 0 0">
      {target_joint}
      {target_inertial}
      <geom name="target_left" type="box" size="{wall} {hw} {panel / 2}" pos="{-hw} 0 0" rgba="0.38 0.42 0.48 1" />
      <geom name="target_right" type="box" size="{wall} {hw} {panel / 2}" pos="{hw} 0 0" rgba="0.38 0.42 0.48 1" />
      <geom name="target_front" type="box" size="{hw} {wall} {panel / 2}" pos="0 {-hw} 0" rgba="0.38 0.42 0.48 1" />
      <geom name="target_back" type="box" size="{hw} {wall} {panel / 2}" pos="0 {hw} 0" rgba="0.38 0.42 0.48 1" />
      <site name="hole_center" pos="0 0 {panel / 2}" size="0.006" rgba="0.2 0.9 0.3 1" />
    </body>
  </worldbody>
  <equality>
    <weld name="dual_arm_grasp" body1="peg" body2="grasp_carrier" solref="0.004 1" solimp="0.9 0.98 0.01" />
  </equality>
  <actuator>
    <motor name="carrier_x_motor" joint="carrier_x" ctrlrange="-100 100" />
    <motor name="carrier_y_motor" joint="carrier_y" ctrlrange="-100 100" />
    <motor name="carrier_z_motor" joint="carrier_z" ctrlrange="-100 100" />
    <motor name="carrier_roll_motor" joint="carrier_roll" ctrlrange="-12 12" />
    <motor name="carrier_pitch_motor" joint="carrier_pitch" ctrlrange="-12 12" />
    <motor name="carrier_yaw_motor" joint="carrier_yaw" ctrlrange="-12 12" />
  </actuator>
</mujoco>'''


def _normalize(q: Array) -> Array:
    return np.asarray(q, dtype=float) / max(np.linalg.norm(q), 1e-12)


def quat_conj(q: Array) -> Array:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1: Array, q2: Array) -> Array:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def quat_to_rot(q: Array) -> Array:
    w, x, y, z = _normalize(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quat_to_rotvec(q: Array) -> Array:
    q = _normalize(q)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-10:
        return 2.0 * q[1:]
    angle = 2.0 * np.arctan2(s, np.clip(q[0], -1.0, 1.0))
    return angle * q[1:] / s


def rotvec_to_quat(v: Array) -> Array:
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return _normalize(np.array([1.0, v[0]/2, v[1]/2, v[2]/2]))
    return np.r_[np.cos(angle/2), np.sin(angle/2) * v / angle]


def quat_to_euler_xyz(q: Array) -> Array:
    w, x, y, z = _normalize(q)
    return np.array([
        np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
        np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
        np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)),
    ])


@dataclass
class EpisodeState:
    stage: RelativeStage
    contact: bool
    contact_force: float
    contact_torque: float
    insertion_depth: float
    lateral_error: float
    angular_error: float


class FloatingInsertionEnv:
    """A compact MuJoCo environment used by the reproducible evaluator."""

    def __init__(self, cfg: FloatingConfig, mode: ExperimentMode, seed: int = 0):
        self.cfg, self.mode = cfg, ExperimentMode(mode)
        self.rng = np.random.default_rng(seed)
        self.floating = self.mode is not ExperimentMode.FIXED
        self.model = mujoco.MjModel.from_xml_string(_xml(cfg, self.floating))
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = 0.0
        self.target_body = self.model.body("target_panel").id
        self.peg_body = self.model.body("peg").id
        self.carrier_jids = [self.model.joint(n).id for n in ("carrier_x", "carrier_y", "carrier_z", "carrier_roll", "carrier_pitch", "carrier_yaw")]
        self.carrier_qpos = np.array([self.model.jnt_qposadr[j] for j in self.carrier_jids])
        self.carrier_dof = np.array([self.model.jnt_dofadr[j] for j in self.carrier_jids])
        self.target_qpos = self.model.jnt_qposadr[self.model.joint("target_free").id] if self.floating else None
        self.target_dof = self.model.jnt_dofadr[self.model.joint("target_free").id] if self.floating else None
        if self.floating:
            # Viscous free-joint damping is a physical parameter, not a pose
            # clamp or return spring; contact remains the source of target
            # translation/rotation.
            self.model.dof_damping[self.target_dof:self.target_dof + 6] = cfg.target_contact_damping
        self.peg_qpos = self.model.jnt_qposadr[self.model.joint("peg_free").id]
        self.stage = RelativeStage.APPROACH
        self.step_count = 0
        self.last_depth = 0.0
        self.peak_force = 0.0
        self.peak_torque = 0.0
        self._last_step_force = 0.0
        self._last_step_torque = 0.0
        self._last_step_contact = False
        self._target_initial_pos = np.zeros(3)
        self._target_initial_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self, seed: int | None = None) -> dict[str, Array | float | str]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        # Carrier starts above the target.  Peg and carrier are coupled by the
        # dual-arm grasp weld; no target state is changed after this reset.
        self.data.qpos[self.carrier_qpos] = np.array([0.0, 0.0, 0.22, 0.0, 0.0, 0.0])
        # Match the weld's nominal 120 mm grasp offset.  This is initialization
        # only; after mj_forward the peg is advanced by contact dynamics.
        self.data.qpos[self.peg_qpos:self.peg_qpos + 7] = np.array(
            [0.0, 0.0, 0.34, 1.0, 0.0, 0.0, 0.0]
        )
        if self.floating:
            pos = np.array([self.rng.uniform(-self.cfg.position_offset, self.cfg.position_offset),
                            self.rng.uniform(-self.cfg.position_offset, self.cfg.position_offset), 0.0])
            r = np.deg2rad(self.cfg.rotation_deg)
            rotvec = self.rng.uniform(-r, r, size=3)
            quat = rotvec_to_quat(rotvec)
            self.data.qpos[self.target_qpos:self.target_qpos+3] = pos
            self.data.qpos[self.target_qpos+3:self.target_qpos+7] = quat
            if self.mode is ExperimentMode.FLOATING_RANDOM_VELOCITY:
                self.data.qvel[self.target_dof:self.target_dof+3] = self.rng.uniform(-self.cfg.linear_velocity, self.cfg.linear_velocity, 3)
                self.data.qvel[self.target_dof+3:self.target_dof+6] = self.rng.uniform(-np.deg2rad(self.cfg.angular_velocity_dps), np.deg2rad(self.cfg.angular_velocity_dps), 3)
            self._target_initial_pos = pos.copy()
            self._target_initial_quat = quat.copy()
        else:
            self._target_initial_pos = self.data.xpos[self.target_body].copy()
            self._target_initial_quat = self.data.xquat[self.target_body].copy()
        mujoco.mj_forward(self.model, self.data)
        self.stage = RelativeStage.APPROACH
        self.step_count = 0
        self.last_depth = 0.0
        self.peak_force = self.peak_torque = 0.0
        self._last_step_force = self._last_step_torque = 0.0
        self._last_step_contact = False
        return self.agent1_observation()

    def target_pose(self) -> tuple[Array, Array]:
        return self.data.xpos[self.target_body].copy(), self.data.xquat[self.target_body].copy()

    def peg_pose(self) -> tuple[Array, Array]:
        return self.data.xpos[self.peg_body].copy(), self.data.xquat[self.peg_body].copy()

    def target_velocity(self) -> tuple[Array, Array]:
        if self.floating:
            qv = self.data.qvel[self.target_dof:self.target_dof+6]
            return qv[:3].copy(), qv[3:6].copy()
        return np.zeros(3), np.zeros(3)

    def relative_state(self) -> dict[str, Array | float]:
        tp, tq = self.target_pose()
        pp, pq = self.peg_pose()
        rt = quat_to_rot(tq)
        rel_pos = rt.T @ (pp - tp)
        rel_quat = quat_mul(quat_conj(tq), pq)
        rel_rot = quat_to_rotvec(rel_quat)
        tv, tw = self.target_velocity()
        pv = self.data.cvel[self.peg_body][3:6].copy()
        pw = self.data.cvel[self.peg_body][:3].copy()
        rel_vel = rt.T @ (pv - tv)
        rel_ang_vel = rt.T @ (pw - tw)
        top = self.cfg.panel_thickness / 2.0
        depth = float(np.clip(top - (rel_pos[2] - self.cfg.peg_length / 2.0), 0.0, self.cfg.panel_thickness + self.cfg.required_depth))
        force, torque, contact = self.contact_wrench()
        force = max(force, self._last_step_force)
        torque = max(torque, self._last_step_torque)
        contact = contact or self._last_step_contact
        return {"peg_position": pp, "peg_orientation": pq, "target_position": tp,
                "target_orientation": tq, "relative_position": rel_pos,
                "relative_orientation": rel_rot, "peg_velocity": np.r_[pv, pw],
                "target_linear_velocity": tv, "target_angular_velocity": tw,
                "relative_linear_velocity": rel_vel, "relative_angular_velocity": rel_ang_vel,
                "insertion_depth": depth, "contact_force": np.array([0.0, 0.0, force]),
                "contact_torque": np.array([0.0, 0.0, torque]), "contact": contact}

    def contact_wrench(self) -> tuple[float, float, bool]:
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if {b1, b2} != {self.target_body, self.peg_body}:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            frame = np.asarray(c.frame).reshape(3, 3)
            f_world = frame @ wrench[:3]
            if b1 == self.target_body:
                f_world = -f_world
            total_force += f_world
            total_torque += np.cross(c.pos - self.data.xpos[self.target_body], f_world)
        fn, tn = float(np.linalg.norm(total_force)), float(np.linalg.norm(total_torque))
        self.peak_force = max(self.peak_force, fn)
        self.peak_torque = max(self.peak_torque, tn)
        return fn, tn, bool(fn > 1e-8)

    def agent1_observation(self) -> dict[str, Array | float | str]:
        """Realtime relative observation consumed by trajectory Agent 1."""
        s = self.relative_state()
        return {**s, "stage": self.stage.value, "step": float(self.step_count)}

    def agent1_action(self) -> Array:
        """Six-dimensional relative-motion action (dx,dy,dz,droll,dpitch,dyaw)."""
        s = self.relative_state()
        p = np.asarray(s["relative_position"])
        r = np.asarray(s["relative_orientation"])
        depth = float(s["insertion_depth"])
        contact = bool(s["contact"])
        if self.stage is RelativeStage.APPROACH:
            z_goal = 0.22
        elif self.stage in (RelativeStage.ALIGNMENT, RelativeStage.CONTACT_SEARCH):
            z_goal = self.cfg.panel_thickness / 2 + self.cfg.peg_length / 2 + 0.004
        else:
            z_goal = self.cfg.panel_thickness / 2 + self.cfg.peg_length / 2 - self.cfg.required_depth
        correction = np.r_[-p[:2], z_goal - p[2], -r]
        if self.stage is RelativeStage.CONTACT_SEARCH and not contact:
            # A bounded lateral search deliberately produces a light wall
            # touch when the clearance is small; the target then moves through
            # ordinary contact impulses rather than a scripted pose update.
            correction[:2] += 0.040 * np.array([
                np.sin(self.step_count * 0.025), np.cos(self.step_count * 0.019)
            ])
        return np.clip(correction, [-0.04, -0.04, -0.03, -0.15, -0.15, -0.15], [0.04, 0.04, 0.03, 0.15, 0.15, 0.15])

    def agent2_wrench(self, action: Array) -> tuple[Array, Array, Array]:
        """Variable-impedance Agent 2 using target motion and contact wrench."""
        s = self.relative_state()
        pose_error = np.r_[np.asarray(s["relative_position"]), np.asarray(s["relative_orientation"])]
        # Agent 1 action is a bounded correction in the relative frame.
        desired_error = -pose_error + np.asarray(action)
        rel_vel = np.r_[np.asarray(s["relative_linear_velocity"]), np.asarray(s["relative_angular_velocity"])]
        contact = bool(s["contact"])
        if self.stage in (RelativeStage.ALIGNMENT, RelativeStage.CONTACT_SEARCH):
            scale = 0.72
        elif contact or self.stage is RelativeStage.INSERTION:
            # Lower stiffness after impact so the carrier follows target motion.
            scale = 0.78
        else:
            scale = 1.0
        k = np.asarray(self.cfg.kp) * scale
        d = 2.0 * self.cfg.damping_ratio * np.sqrt(np.asarray(self.cfg.mass) * k)
        wrench = k * desired_error - d * rel_vel
        wrench[:3] = wrench[:3] * min(1.0, self.cfg.max_force / max(np.linalg.norm(wrench[:3]), 1e-12))
        wrench[3:] = wrench[3:] * min(1.0, self.cfg.max_torque / max(np.linalg.norm(wrench[3:]), 1e-12))
        return wrench, k, d

    def _update_stage(self) -> None:
        s = self.relative_state()
        p = np.asarray(s["relative_position"])
        r = np.asarray(s["relative_orientation"])
        depth = float(s["insertion_depth"])
        force, torque, contact = self.contact_wrench()
        force = max(force, self._last_step_force)
        torque = max(torque, self._last_step_torque)
        contact = contact or self._last_step_contact
        lateral = float(np.linalg.norm(p[:2]))
        angle = float(np.rad2deg(np.linalg.norm(r)))
        self.stage = RelativeStage(self.stage)
        if force > self.cfg.safe_force or torque > self.cfg.safe_torque:
            self.stage = RelativeStage.FAILURE
        # Approach is measured along the live hole axis, not by a fixed world
        # coordinate.  The 180 mm live-axis threshold leaves room for search.
        elif self.stage is RelativeStage.APPROACH and max(0.0, p[2] - 0.06) < 0.18:
            self.stage = RelativeStage.ALIGNMENT
        elif self.stage is RelativeStage.ALIGNMENT and lateral < self.cfg.lateral_threshold and angle < self.cfg.angular_threshold_deg:
            self.stage = RelativeStage.CONTACT_SEARCH
        elif self.stage is RelativeStage.CONTACT_SEARCH and contact:
            self.stage = RelativeStage.INSERTION
        elif self.stage is RelativeStage.INSERTION and force > 1.0 and depth + 0.0005 < self.last_depth and self.step_count > 80:
            self.stage = RelativeStage.JAM_RECOVERY
        elif self.stage is RelativeStage.JAM_RECOVERY and force < 1.0 and lateral < 2.0 * self.cfg.lateral_threshold:
            self.stage = RelativeStage.INSERTION
        if depth >= self.cfg.required_depth and lateral < self.cfg.lateral_threshold and angle < self.cfg.angular_threshold_deg and force < self.cfg.safe_force:
            self.stage = RelativeStage.SUCCESS
        self.last_depth = max(self.last_depth, depth)

    def step(self, action: Array, *, substeps: int | None = None) -> dict[str, Any]:
        action = np.asarray(action, dtype=float)
        if action.shape != (6,):
            raise ValueError("action must have shape (6,)")
        # Carrier qpos are expressed in its world parent.  Desired orientation
        # follows the live target quaternion; no cached world target is used.
        tp, tq = self.target_pose()
        target_euler = quat_to_euler_xyz(tq)
        depth_goal = self.cfg.panel_thickness / 2 + self.cfg.peg_length / 2 - min(self.cfg.required_depth, self.last_depth + 0.004)
        if self.stage is RelativeStage.APPROACH:
            depth_goal = 0.22
        elif self.stage in (RelativeStage.ALIGNMENT, RelativeStage.CONTACT_SEARCH):
            depth_goal = self.cfg.panel_thickness / 2 + self.cfg.peg_length / 2 + 0.004
        # The peg body is 120 mm ahead of the carrier frame in the grasp weld;
        # command the carrier pose that produces the desired peg pose.
        target_offset = quat_to_rot(tq) @ np.array([0.0, 0.0, depth_goal - 0.120])
        desired = np.r_[tp + target_offset, target_euler]
        # Agent 1's lateral/rotational correction is applied as a small
        # compliant offset.  The live target pose remains the reference; in
        # particular, no cached world-space hole coordinate is used.
        desired[:2] += 0.25 * action[:2]
        desired[3:] += 0.10 * action[3:]
        q = self.data.qpos[self.carrier_qpos].copy()
        v = self.data.qvel[self.carrier_dof].copy()
        wrench, k, d = self.agent2_wrench(action)
        # Use the same relative wrench for both gripper arms; their common
        # carrier applies the summed compliant wrench to the grasped peg.
        error = desired - q
        error[3:] = desired[3:] - q[3:]
        command = k * error - d * v
        command[:3] = np.clip(command[:3], -self.cfg.max_force, self.cfg.max_force)
        command[3:] = np.clip(command[3:], -self.cfg.max_torque, self.cfg.max_torque)
        self.data.ctrl[:] = command
        step_force = step_torque = 0.0
        step_contact = False
        for _ in range(substeps or self.cfg.control_interval):
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1
            force_i, torque_i, contact_i = self.contact_wrench()
            step_force = max(step_force, force_i)
            step_torque = max(step_torque, torque_i)
            step_contact = step_contact or contact_i
        self._last_step_force, self._last_step_torque, self._last_step_contact = step_force, step_torque, step_contact
        self._update_stage()
        s = self.relative_state()
        # A brief collision during the control interval remains observable to
        # the agents and evaluator even if the final substep has separated.
        s["contact_force"] = np.array([0.0, 0.0, step_force])
        s["contact_torque"] = np.array([0.0, 0.0, step_torque])
        s["contact"] = step_contact or bool(s["contact"])
        s.update({"stage": self.stage.value, "agent1_action": action.copy(), "wrench": wrench.copy(), "K": k.copy(), "D": d.copy()})
        return s

    def metrics(self) -> EpisodeState:
        s = self.relative_state()
        force, torque, contact = self.contact_wrench()
        return EpisodeState(self.stage, contact, force, torque, float(s["insertion_depth"]),
                            float(np.linalg.norm(np.asarray(s["relative_position"])[:2])),
                            float(np.rad2deg(np.linalg.norm(np.asarray(s["relative_orientation"])))) )


def target_displacement(env: FloatingInsertionEnv) -> tuple[float, float]:
    pos, quat = env.target_pose()
    translation = float(np.linalg.norm(pos - env._target_initial_pos))
    rotation = float(np.rad2deg(np.linalg.norm(quat_to_rotvec(quat_mul(quat_conj(env._target_initial_quat), quat)))))
    return translation, rotation
