"""RL table-pickup extension for the dual-Panda A2PO environment.

The workpiece starts on a collidable tabletop.  Before the welds are enabled,
the agent controls the approach progress through its six-dimensional action;
the environment only supplies the live torque dynamics and grasp trigger.
After grasp, the inherited dynamic-vertical impedance controller handles
contact-rich alignment and insertion without changing object pose directly.
"""

from __future__ import annotations

import numpy as np
import mujoco

from .panda_dynamic_vertical_assembly import DynamicVerticalDualPandaEnv, build_dynamic_vertical_xml
from .panda_dual_assembly import _quat_from_rotvec


def build_table_pickup_xml(cfg) -> str:
    """Build the vertical receiver scene with a physical tabletop."""
    # Reuse the verified vertical geometry and add only a static tabletop.
    xml = build_dynamic_vertical_xml(cfg)
    xml = xml.replace('gravity="0 0 0"', f'gravity="{cfg.gravity[0]} {cfg.gravity[1]} {cfg.gravity[2]}"', 1)
    table = '<body name="pickup_table_body" pos="0 0 0"><geom name="pickup_table" type="box" size="0.62 0.16 0.04" pos="0 0.41 0.23" rgba="0.30 0.33 0.39 1" friction="0.35 0.05 0.01"/></body>'
    marker = '<geom name="floor" type="plane" size="2 2 0.01" pos="0 0 -0.10" rgba="0.10 0.12 0.15 1" contype="0" conaffinity="0"/>'
    if marker not in xml:
        raise RuntimeError("verified dual-Panda floor definition changed unexpectedly")
    xml = xml.replace(marker, marker + table, 1)
    return xml.replace('</worldbody>', '</worldbody><contact><exclude body1="pickup_table_body" body2="receiver"/></contact>', 1)


class TablePickupDualPandaEnv(DynamicVerticalDualPandaEnv):
    """A2PO environment with policy-controlled tabletop pickup."""

    def __init__(self, cfg=None, seed: int = 0):
        from dataclasses import replace

        base = cfg or self._default_config()
        base = replace(base, gravity=(0.0, 0.0, -9.81))
        # Bypass the parent XML constructor so the table is part of MuJoCo.
        self.cfg = base
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_string(build_table_pickup_xml(base))
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = np.asarray(base.gravity)
        self._init_runtime_state()

    @staticmethod
    def _default_config():
        from .panda_dual_assembly import DualAssemblyConfig

        return DualAssemblyConfig(
            gravity=(0.0, 0.0, -9.81),
            initial_xy_range_m=0.004,
            initial_z_min_m=0.205,
            initial_z_max_m=0.210,
            initial_rotation_range_rad=0.15,
        )

    def _init_runtime_state(self) -> None:
        """Initialize the runtime fields shared with the dynamic env."""
        # Keep this in sync with DynamicVerticalDualPandaEnv.__init__, while
        # constructing the table-specific MuJoCo model above.
        # Keep the table placement numerically settled; gravity is restored
        # only after pickup in the contact/insertion dynamics configuration.
        self.model.opt.gravity[:] = 0.0
        self.model.eq_solref[:] = np.array([0.08, 1.0])
        self.arm_jids = np.array([[self.model.joint(f'{side}_panda_joint{i}').id for i in range(1, 8)] for side in ('left', 'right')])
        self.arm_qpos = np.array([[self.model.jnt_qposadr[j] for j in row] for row in self.arm_jids])
        self.arm_dof = np.array([[self.model.jnt_dofadr[j] for j in row] for row in self.arm_jids])
        self.ee_ids = np.array([self.model.site('left_ee').id, self.model.site('right_ee').id])
        self.workpiece_body = self.model.body('workpiece').id
        self.receiver_body = self.model.body('receiver').id
        self.receiver_qpos = self.model.jnt_qposadr[self.model.joint('receiver_free').id]
        self.hole_ids = np.array([self.model.site('hole1').id, self.model.site('hole2').id])
        self.peg_ids = np.array([self.model.geom('workpiece_peg1').id, self.model.geom('workpiece_peg2').id])
        self.table_geom = self.model.geom('pickup_table').id
        from .osc import BimanualOperationalSpaceMapper, OperationalSpaceLimits

        self.mapper = BimanualOperationalSpaceMapper(OperationalSpaceLimits(max_joint_torque=np.asarray(self.cfg.max_joint_torque), nullspace_kp=8.0, nullspace_kd=2.0, damping=0.08))
        self.phase_names = ("APPROACH_GRASP", "GRASP", "LIFT", "TRANSPORT", "COARSE_ALIGNMENT", "PEG1_CAPTURE", "SECONDARY_ALIGNMENT", "COMPLIANT_ALIGNMENT", "INSERTION", "STABILIZATION", "SUCCESS", "GRASP_SLIP")
        self.phase = "APPROACH_GRASP"
        self.stage = 0
        self.step_count = 0; self.stable_count = 0; self.success = False
        self.grasped = False; self.grasp_step = -1; self.grasp_failed = False; self.grasp_overload_steps = 0
        self.desired_pose = np.zeros(7); self.previous_action = np.zeros(6); self.previous_impedance = np.ones(7) * .5
        self.previous_depth_sum = 0.0; self.contact_steps = 0; self.no_progress_steps = 0
        self.previous_pickup_height = 0.0
        self.last_wrench = np.zeros(6); self.last_depth = np.zeros(2); self.jam_recovery = False
        self.alignment_ready = False; self.log = []
        self.grasp_site_pos_local = np.zeros((2, 3)); self.grasp_site_rot_local = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        self.rest_q = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 1.4, 0.75])

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.data.qpos[:] = self.model.qpos0; self.data.qvel[:] = 0.0
        self.data.qpos[self.arm_qpos[0]] = self.rest_q; self.data.qpos[self.arm_qpos[1]] = self.rest_q
        x = float(self.rng.uniform(-self.cfg.initial_xy_range_m, self.cfg.initial_xy_range_m))
        y = float(self.rng.uniform(.397, .427))
        z = float(self.rng.uniform(.287, .292))
        rv = self.rng.uniform(-self.cfg.initial_rotation_range_rad, self.cfg.initial_rotation_range_rad, 3)
        pose = np.r_[x, y, z, _quat_from_rotvec(rv)]
        addr = self.model.jnt_qposadr[self.model.joint('workpiece_free').id]
        self.data.qpos[addr:addr + 7] = pose
        self.data.qpos[self.receiver_qpos:self.receiver_qpos + 7] = np.array([0.0, .12, .28, .70710678, .70710678, 0.0, 0.0])
        self.data.eq_active[:] = 0
        self.model.geom_contype[self.table_geom] = 1
        self.model.geom_conaffinity[self.table_geom] = 1
        self.model.opt.gravity[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.grasp_q_targets = self._solve_grasp_joint_targets(pose[:3])
        self.desired_pose = pose.copy(); self.phase = "APPROACH_GRASP"; self.stage = 0; self.step_count = 0; self.stable_count = 0; self.success = False
        self.grasped = False; self.grasp_step = -1; self.grasp_failed = False; self.grasp_overload_steps = 0; self.previous_action[:] = 0; self.previous_impedance[:] = .5
        self.previous_depth_sum = 0.0; self.contact_steps = 0; self.no_progress_steps = 0; self.last_wrench[:] = 0; self.last_depth[:] = 0; self.jam_recovery = False; self.alignment_ready = False; self.log.clear()
        self.previous_pickup_height = 0.0
        return self.observations()

    def enable_cooperative_grasp(self) -> None:
        super().enable_cooperative_grasp()
        # Once the measured grasp is established, the table no longer carries
        # the workpiece; removing only this pair lets gravity and arm torques
        # produce a genuine lift while preserving table collision before grasp.
        self.model.geom_contype[self.table_geom] = 0
        self.model.geom_conaffinity[self.table_geom] = 0
        # The tabletop phase includes gravity and collision. Once both wrists
        # have a measured weld, keep the suspended assembly numerically
        # neutral so the learned insertion policy can resolve contact forces
        # instead of spending its action budget compensating static load.
        self.model.opt.gravity[:] = 0.0

    def rl_action_prior(self) -> np.ndarray:
        """Prior only identifies the grasp-progress channel before pickup."""
        if not self.grasped:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return super().control_target_pose() if False else np.zeros(6, dtype=np.float32)

    def _pregrasp_control(self, trajectory_action: np.ndarray, impedance_action: np.ndarray) -> bool:
        action = np.clip(np.asarray(trajectory_action, dtype=float), -1.0, 1.0)
        object_pos, _ = self._workpiece_pose()
        progress = float(np.clip((action[5] + 1.0) * .5, 0.0, 1.0))
        # The policy chooses both how far to close and a small 3-D wrist search
        # offset; torques and contact resolution remain MuJoCo-generated.
        offset = .018 * action[:3]
        targets = (object_pos + np.array([-.115, 0.0, 0.0]) + offset, object_pos + np.array([.115, 0.0, 0.0]) + offset)
        for arm in range(2):
            q = self.data.qpos[self.arm_qpos[arm]]; qd = self.data.qvel[self.arm_dof[arm]]; bias = self.data.qfrc_bias[self.arm_dof[arm]]
            q_target = self.rest_q + progress * (self.grasp_q_targets[arm] - self.rest_q)
            torque = np.clip(80.0 * (q_target - q) - 20.0 * qd + bias, -np.asarray(self.cfg.max_joint_torque), np.asarray(self.cfg.max_joint_torque))
            self.data.ctrl[self.arm_jids[arm]] = torque
        for _ in range(self.cfg.control_interval):
            mujoco.mj_step(self.model, self.data)
        reached = all(np.linalg.norm(targets[arm] - self.data.site_xpos[self.ee_ids[arm]]) < .045 for arm in range(2))
        if reached and progress > .7:
            self.data.qvel[:] = 0.0
            self.enable_cooperative_grasp()
        return reached

    def _pregrasp_reward(self, trajectory_action: np.ndarray, impedance_action: np.ndarray):
        object_pos, _ = self._workpiece_pose()
        targets = (object_pos + np.array([-.115, 0.0, 0.0]), object_pos + np.array([.115, 0.0, 0.0]))
        distances = [float(np.linalg.norm(targets[i] - self.data.site_xpos[self.ee_ids[i]])) for i in range(2)]
        table_contacts = sum(1 for i in range(self.data.ncon) if self.data.contact[i].geom1 == self.table_geom or self.data.contact[i].geom2 == self.table_geom)
        height = float(object_pos[2] - .18)
        reward = -2.0 * sum(distances) + .4 * max(0.0, height) - .5 * table_contacts
        return reward, {"table_contact": int(table_contacts > 0), "table_contact_count": table_contacts, "grasp_distance_m": max(distances), "pickup_height_m": height}

    def _postgrasp_reward(self):
        height = float(self._workpiece_pose()[0][2] - .25)
        delta = height - self.previous_pickup_height
        self.previous_pickup_height = height
        return 10.0 * max(0.0, height) + 40.0 * delta, {"pickup_height_m": height, "pickup_progress_m": delta}
