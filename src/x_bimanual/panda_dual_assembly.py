"""MuJoCo dual-Panda cooperative dual-peg / dual-hole assembly.

This environment is intentionally separate from ``mujoco_floating``.  It
loads the locally installed Franka Panda URDF mesh assets into an MJCF model,
uses two 7-DoF Panda arms, and attaches one rigid two-peg workpiece to the two
grippers with welds.  The task action is an object-level Cartesian increment;
the environment converts that command to a desired object pose, evaluates a
variable Cartesian impedance command, maps the two equal cooperative wrenches
through the live Panda Jacobians, and advances MuJoCo with motor torques.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from .osc import BimanualOperationalSpaceMapper, OperationalSpaceLimits

Array = np.ndarray


class AssemblyStage(IntEnum):
    INITIALIZATION = 0
    GRASP = 1
    LIFT = 2
    TRANSPORT = 3
    COARSE_ALIGNMENT = 4
    APPROACH = 5
    FIRST_CONTACT = 6
    COMPLIANT_ALIGNMENT = 7
    INSERTION = 8
    SUCCESS = 9


@dataclass(frozen=True)
class DualAssemblyConfig:
    timestep: float = 0.002
    control_interval: int = 4
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    arm_base_x: float = 0.48
    arm_base_z: float = -0.38
    workpiece_mass: float = 1.0
    peg_length: float = 0.18
    peg_radius: float = 0.020
    peg_spacing: float = 0.12
    hole_radius: float = 0.022
    hole_depth: float = 0.04
    receiver_thickness: float = 0.04
    required_depth: float = 0.035
    lateral_threshold: float = 0.003
    orientation_threshold_deg: float = 2.0
    stable_steps: int = 25
    max_steps: int = 1600
    action_translation_limit: float = 0.015
    action_rotation_limit_rad: float = math.radians(8.0)
    max_force: float = 80.0
    max_torque: float = 12.0
    internal_force_min: float = 8.0
    internal_force_max: float = 50.0
    grasp_friction: float = 0.7
    grasp_slip_steps: int = 5
    max_joint_torque: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    # Defaults preserve the verified environment.  Training can widen these
    # ranges without changing the robot, receiver, or solver geometry.
    initial_xy_range_m: float = 0.004
    initial_z_min_m: float = 0.328
    initial_z_max_m: float = 0.332
    initial_rotation_range_rad: float = 0.02


def dual_config_from_mapping(mapping: Mapping[str, Any]) -> DualAssemblyConfig:
    """Load the new environment from ``dual_panda_assembly`` YAML values."""
    section = mapping.get("dual_panda_assembly", {})
    geometry = mapping.get("geometry", {})
    success = mapping.get("success", {})
    grasp = mapping.get("grasp", {})
    return DualAssemblyConfig(
        timestep=float(section.get("timestep", 0.002)),
        control_interval=int(section.get("control_interval", 4)),
        gravity=tuple(float(v) for v in section.get("gravity", (0.0, 0.0, -9.81))),
        arm_base_x=float(section.get("arm_base_x_m", 0.48)),
        arm_base_z=float(section.get("arm_base_z_m", -0.38)),
        workpiece_mass=float(section.get("workpiece_mass_kg", 1.0)),
        peg_length=float(geometry.get("peg_length_m", section.get("peg_length_m", 0.18))),
        peg_radius=float(geometry.get("peg_radius_m", section.get("peg_radius_m", 0.020))),
        peg_spacing=float(section.get("peg_spacing_m", 0.12)),
        hole_radius=float(geometry.get("hole_radius_m", section.get("hole_radius_m", 0.022))),
        hole_depth=float(geometry.get("hole_depth_m", section.get("hole_depth_m", 0.04))),
        required_depth=float(success.get("insertion_depth_m", section.get("required_depth_m", 0.035))),
        lateral_threshold=float(success.get("lateral_error_m", section.get("lateral_threshold_m", 0.003))),
        orientation_threshold_deg=float(success.get("angle_error_deg", section.get("orientation_threshold_deg", 2.0))),
        stable_steps=int(success.get("hold_steps", section.get("stable_steps", 25))),
        max_steps=int(section.get("max_steps", 1600)),
        internal_force_min=float(grasp.get("internal_force_min_N", 8.0)),
        internal_force_max=float(grasp.get("internal_force_max_N", 50.0)),
        grasp_friction=float(grasp.get("friction_coefficient", 0.7)),
        grasp_slip_steps=int(grasp.get("slip_persistence_steps", 5)),
        initial_xy_range_m=float(section.get("initial_xy_range_m", 0.004)),
        initial_z_min_m=float(section.get("initial_z_min_m", 0.328)),
        initial_z_max_m=float(section.get("initial_z_max_m", 0.332)),
        initial_rotation_range_rad=float(section.get("initial_rotation_range_rad", 0.02)),
    )


def _panda_urdf_path() -> Path:
    try:
        import pybullet_data
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("pybullet_data is required for the local Panda mesh assets") from exc
    path = Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _rpy_quat(rpy: str | None) -> tuple[float, float, float, float]:
    vals = np.fromstring(rpy or "0 0 0", sep=" ", dtype=float)
    if vals.size != 3:
        raise ValueError(f"bad rpy: {rpy}")
    roll, pitch, yaw = vals
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _xyz(value: str | None) -> tuple[float, float, float]:
    vals = np.fromstring(value or "0 0 0", sep=" ", dtype=float)
    if vals.size != 3:
        raise ValueError(f"bad xyz: {value}")
    return tuple(float(v) for v in vals)


def _mesh_filename(urdf: Path, filename: str) -> str:
    # The pybullet URDF uses package://meshes/... paths.
    if filename.startswith("package://meshes/"):
        return str(urdf.parent / filename.removeprefix("package://"))
    return str((urdf.parent / filename).resolve())


def _panda_body_xml(urdf: Path, prefix: str, base_pos: tuple[float, float, float], quat: tuple[float, ...]) -> str:
    """Convert the Panda URDF's link/joint/mesh subset to one MJCF body tree."""
    root = ET.parse(urdf).getroot()
    links: dict[str, ET.Element] = {n.attrib["name"]: n for n in root.findall("link")}
    joints = list(root.findall("joint"))
    children = {j.find("child").attrib["link"]: j for j in joints if j.find("child") is not None}
    by_parent: dict[str, list[ET.Element]] = {}
    for joint in joints:
        parent = joint.find("parent")
        if parent is not None:
            by_parent.setdefault(parent.attrib["link"], []).append(joint)

    def link_elements(link_name: str) -> list[str]:
        link = links[link_name]
        out: list[str] = []
        inertial = link.find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").attrib["value"])
            origin = inertial.find("origin")
            pos = _xyz(origin.attrib.get("xyz") if origin is not None else None)
            iq = _rpy_quat(origin.attrib.get("rpy") if origin is not None else None)
            inertia = inertial.find("inertia")
            diag = (float(inertia.attrib.get("ixx", "0.01")), float(inertia.attrib.get("iyy", "0.01")), float(inertia.attrib.get("izz", "0.01")))
            out.append(f'<inertial pos="{pos[0]} {pos[1]} {pos[2]}" quat="{iq[0]} {iq[1]} {iq[2]} {iq[3]}" mass="{mass}" diaginertia="{diag[0]} {diag[1]} {diag[2]}"/>')
        for kind, color in (("visual", "0.86 0.86 0.86 1"), ("collision", "0.70 0.70 0.72 1")):
            element = link.find(kind)
            if element is None:
                continue
            geometry = element.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None:
                continue
            filename = _mesh_filename(urdf, mesh.attrib["filename"])
            origin = element.find("origin")
            pos = _xyz(origin.attrib.get("xyz") if origin is not None else None)
            q = _rpy_quat(origin.attrib.get("rpy") if origin is not None else None)
            asset_name = f"{prefix}_{link_name}_{kind}"
            out.append(f'<geom name="{asset_name}" type="mesh" mesh="{asset_name}_mesh" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}" rgba="{color}" contype="0" conaffinity="0"/>')
        return out

    def recurse(parent_link: str) -> str:
        chunks: list[str] = []
        for joint in by_parent.get(parent_link, []):
            child = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            pos = _xyz(origin.attrib.get("xyz") if origin is not None else None)
            q = _rpy_quat(origin.attrib.get("rpy") if origin is not None else None)
            name = joint.attrib["name"]
            jtype = joint.attrib.get("type", "fixed")
            attrs = f' name="{prefix}_{name}"'
            if jtype == "revolute":
                axis = _xyz(joint.find("axis").attrib.get("xyz"))
                limit = joint.find("limit")
                lo = float(limit.attrib.get("lower", "-3.14")) if limit is not None else -3.14
                hi = float(limit.attrib.get("upper", "3.14")) if limit is not None else 3.14
                chunks.append(f'<body name="{prefix}_{child}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}"><joint{attrs} type="hinge" axis="{axis[0]} {axis[1]} {axis[2]}" range="{lo} {hi}" damping="0.15"/><site name="{prefix}_{child}_frame" pos="0 0 0" size="0.004" rgba="0.2 0.6 1 1"/>{''.join(link_elements(child))}{recurse(child)}</body>')
            else:
                ee_site = f'<site name="{prefix}_ee" pos="0 0 0" size="0.008" rgba="0.2 0.9 0.2 1"/>' if child == "panda_link8" else ""
                chunks.append(f'<body name="{prefix}_{child}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">{"".join(link_elements(child))}{ee_site}{recurse(child)}</body>')
        return "".join(chunks)

    root_link = "panda_link0"
    root_body = f'<body name="{prefix}_panda_link0" pos="{base_pos[0]} {base_pos[1]} {base_pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">'
    root_body += "".join(link_elements(root_link))
    root_body += recurse(root_link)
    root_body += "</body>"
    return root_body


def _panda_asset_xml(urdf: Path, prefix: str) -> str:
    root = ET.parse(urdf).getroot()
    names: list[str] = []
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            element = link.find(kind)
            geometry = element.find("geometry") if element is not None else None
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is not None:
                name = f"{prefix}_{link.attrib['name']}_{kind}"
                path = _mesh_filename(urdf, mesh.attrib["filename"])
                names.append(f'<mesh name="{name}_mesh" file="{path}"/>')
    return "".join(names)


def _hole_ring_mesh(cfg: DualAssemblyConfig) -> str:
    """Create one annular collision mesh used for each physical cylindrical hole."""
    n = 32
    outer = cfg.hole_radius + 0.006
    inner = cfg.hole_radius
    z0, z1 = 0.0, cfg.hole_depth
    vertices: list[str] = []
    for z in (z0, z1):
        for radius in (outer, inner):
            for i in range(n):
                a = 2 * math.pi * i / n
                vertices.append(f"{radius * math.cos(a):.7g} {radius * math.sin(a):.7g} {z:.7g}")
    faces: list[str] = []
    # Ring indexing: bottom outer/inner, top outer/inner.
    for i in range(n):
        j = (i + 1) % n
        quads = ((i, j, 2*n+j, 2*n+i), (n+i, 3*n+i, 3*n+j, n+j),
                 (2*n+i, 2*n+j, 3*n+j, 3*n+i), (i, n+i, n+j, j))
        for a, b, c, d in quads:
            faces.extend((f"{a} {b} {c}", f"{a} {c} {d}"))
    vertex_values = " ".join(vertices)
    face_values = " ".join(faces)
    return f'<mesh name="hole_ring_mesh" vertex="{vertex_values}" face="{face_values}"/>'


def build_dual_panda_xml(cfg: DualAssemblyConfig) -> str:
    urdf = _panda_urdf_path()
    # Rotate the right arm around Z so both wrists face the common workpiece.
    left = _panda_body_xml(urdf, "left", (-cfg.arm_base_x, 0.0, cfg.arm_base_z), (1.0, 0.0, 0.0, 0.0))
    right = _panda_body_xml(urdf, "right", (cfg.arm_base_x, 0.0, cfg.arm_base_z), (0.0, 0.0, 0.0, 1.0))
    assets = _panda_asset_xml(urdf, "left") + _panda_asset_xml(urdf, "right")
    half_spacing = cfg.peg_spacing / 2
    receiver_frame = '''
      <geom name="receiver_back" type="box" size="0.24 0.012 0.02" pos="0 0.148 0" rgba="0.36 0.39 0.46 1"/>
      <geom name="receiver_front" type="box" size="0.24 0.012 0.02" pos="0 -0.148 0" rgba="0.36 0.39 0.46 1"/>
      <geom name="receiver_left" type="box" size="0.012 0.14 0.02" pos="-0.228 0 0" rgba="0.36 0.39 0.46 1"/>
      <geom name="receiver_right" type="box" size="0.012 0.14 0.02" pos="0.228 0 0" rgba="0.36 0.39 0.46 1"/>'''
    rings = []
    # MuJoCo collision meshes are convexified; a single annular mesh would
    # therefore close the hole.  Use sixteen convex wall segments instead,
    # which leaves a genuine open cylindrical passage with low facet error.
    ring_outer = cfg.hole_radius + 0.006
    ring_half_tangent = ring_outer * math.sin(math.pi / 16) * 0.82
    for hole_index, cx in enumerate((-half_spacing, half_spacing), 1):
        for segment in range(16):
            angle = 2 * math.pi * segment / 16
            qz = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
            rings.append(
                f'<geom name="hole_ring_{hole_index}_{segment}" type="box" size="0.003 {ring_half_tangent:.7g} {cfg.hole_depth / 2:.7g}" pos="{cx + ring_outer * math.cos(angle):.7g} {ring_outer * math.sin(angle):.7g} {cfg.hole_depth / 2:.7g}" quat="{qz[0]:.7g} {qz[1]:.7g} {qz[2]:.7g} {qz[3]:.7g}" rgba="0.12 0.13 0.16 1"/>'
            )
    receiver_frame += "".join(rings)
    return f'''<mujoco model="dual_panda_dual_peg_assembly">
  <compiler angle="radian" meshdir="/" coordinate="local"/>
  <option timestep="{cfg.timestep}" gravity="{cfg.gravity[0]} {cfg.gravity[1]} {cfg.gravity[2]}" integrator="implicitfast" cone="elliptic"/>
  <size njmax="4000" nconmax="800"/>
  <asset>{assets}</asset>
  <default><joint damping="0.2" armature="0.005"/><geom friction="0.7 0.05 0.01" solref="0.006 1" solimp="0.88 0.96 0.01"/></default>
  <worldbody>
    <light name="key" pos="1 -1 2" dir="-1 1 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="2 2 0.01" pos="0 0 -0.10" rgba="0.10 0.12 0.15 1" contype="0" conaffinity="0"/>
    {left}
    {right}
    <body name="workpiece" pos="0 0 0.32">
      <freejoint name="workpiece_free"/>
      <inertial pos="0 0 0" mass="{cfg.workpiece_mass}" diaginertia="0.004 0.004 0.006"/>
      <geom name="workpiece_bar" type="box" size="0.17 0.035 0.018" rgba="0.18 0.30 0.78 1"/>
      <geom name="workpiece_peg1" type="cylinder" size="{cfg.peg_radius} {cfg.peg_length / 2}" pos="{-half_spacing} 0 {-cfg.peg_length / 2 - 0.018}" rgba="0.86 0.25 0.08 1"/>
      <geom name="workpiece_peg2" type="cylinder" size="{cfg.peg_radius} {cfg.peg_length / 2}" pos="{half_spacing} 0 {-cfg.peg_length / 2 - 0.018}" rgba="0.86 0.25 0.08 1"/>
      <site name="workpiece_center" pos="0 0 0" size="0.006" rgba="0.1 0.9 0.2 1"/>
    </body>
    <body name="receiver" pos="0 0 0.0">
      {receiver_frame}
      <site name="hole1" pos="{-half_spacing} 0 {cfg.hole_depth}" size="0.006" rgba="0.1 0.9 0.2 1"/>
      <site name="hole2" pos="{half_spacing} 0 {cfg.hole_depth}" size="0.006" rgba="0.1 0.9 0.2 1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="left_grasp" body1="left_panda_link8" body2="workpiece" relpose="0.100934 -0.094030 -0.030044 0 -0.930508 0.366273 0" solref="0.008 1" solimp="0.88 0.96 0.01"/>
    <weld name="right_grasp" body1="right_panda_link8" body2="workpiece" relpose="0.100934 -0.094030 -0.030044 0 -0.366273 -0.930508 0" solref="0.008 1" solimp="0.88 0.96 0.01"/>
  </equality>
  <actuator>
    {''.join(f'<motor name="left_motor_{i+1}" joint="left_panda_joint{i+1}" ctrlrange="{-cfg.max_joint_torque[i]} {cfg.max_joint_torque[i]}"/>' for i in range(7))}
    {''.join(f'<motor name="right_motor_{i+1}" joint="right_panda_joint{i+1}" ctrlrange="{-cfg.max_joint_torque[i]} {cfg.max_joint_torque[i]}"/>' for i in range(7))}
  </actuator>
</mujoco>'''


def _quat_conj(q: Array) -> Array:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_mul(a: Array, b: Array) -> Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def _rotvec(q: Array) -> Array:
    q = np.asarray(q, dtype=float) / max(np.linalg.norm(q), 1e-12)
    if q[0] < 0: q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-10: return 2 * q[1:]
    return 2 * np.arctan2(s, np.clip(q[0], -1, 1)) * q[1:] / s


def _quat_from_rotvec(v: Array) -> Array:
    a = np.linalg.norm(v)
    if a < 1e-12: return np.array([1.0, v[0]/2, v[1]/2, v[2]/2])
    return np.r_[math.cos(a/2), math.sin(a/2) * np.asarray(v) / a]


def _mat_quat(mat: Array) -> Array:
    """Convert a rotation matrix to a normalized wxyz quaternion."""
    m = np.asarray(mat, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2
        return np.array([0.25*s, (m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s])
    axis = int(np.argmax(np.diag(m)))
    if axis == 0:
        s = math.sqrt(max(1e-12, 1 + m[0, 0] - m[1, 1] - m[2, 2])) * 2
        q = np.array([(m[2, 1]-m[1, 2])/s, 0.25*s, (m[0, 1]+m[1, 0])/s, (m[0, 2]+m[2, 0])/s])
    elif axis == 1:
        s = math.sqrt(max(1e-12, 1 - m[0, 0] + m[1, 1] - m[2, 2])) * 2
        q = np.array([(m[0, 2]-m[2, 0])/s, (m[0, 1]+m[1, 0])/s, 0.25*s, (m[1, 2]+m[2, 1])/s])
    else:
        s = math.sqrt(max(1e-12, 1 - m[0, 0] - m[1, 1] + m[2, 2])) * 2
        q = np.array([(m[1, 0]-m[0, 1])/s, (m[0, 2]+m[2, 0])/s, (m[1, 2]+m[2, 1])/s, 0.25*s])
    return q / max(np.linalg.norm(q), 1e-12)


@dataclass(frozen=True)
class AgentActions:
    trajectory: Array
    impedance: Array


class DualPandaAssemblyEnv:
    """Dual-Panda cooperative assembly environment with A2PO-compatible IO."""

    def __init__(self, cfg: DualAssemblyConfig | None = None, seed: int = 0):
        self.cfg = cfg or DualAssemblyConfig()
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_string(build_dual_panda_xml(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = self.cfg.gravity
        self.arm_jids = np.array([[self.model.joint(f'{side}_panda_joint{i}').id for i in range(1, 8)] for side in ('left', 'right')])
        self.arm_qpos = np.array([[self.model.jnt_qposadr[j] for j in row] for row in self.arm_jids])
        self.arm_dof = np.array([[self.model.jnt_dofadr[j] for j in row] for row in self.arm_jids])
        self.ee_ids = np.array([self.model.site('left_ee').id, self.model.site('right_ee').id])
        self.workpiece_body = self.model.body('workpiece').id
        self.hole_ids = np.array([self.model.site('hole1').id, self.model.site('hole2').id])
        self.peg_ids = np.array([self.model.geom('workpiece_peg1').id, self.model.geom('workpiece_peg2').id])
        self.mapper = BimanualOperationalSpaceMapper(OperationalSpaceLimits(max_joint_torque=np.asarray(self.cfg.max_joint_torque), nullspace_kp=8.0, nullspace_kd=2.0, damping=0.08))
        self.stage = AssemblyStage.INITIALIZATION
        self.step_count = 0
        self.stable_count = 0
        self.success = False
        self.desired_pose = np.zeros(7)
        self.previous_action = np.zeros(6)
        self.previous_impedance = np.ones(12) * 0.5
        self.last_wrench = np.zeros(6)
        self.last_depth = np.zeros(2)
        self.jam_recovery = False
        self.log: list[dict[str, Any]] = []
        self.rest_q = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 1.4, 0.75])

    @property
    def observation_space_shapes(self) -> dict[str, tuple[int, ...]]:
        return {'trajectory': (30,), 'impedance': (42,)}

    @property
    def impedance_action_dim(self) -> int:
        """Normalized Agent 2 action width used by the generic trainer."""
        return 12

    def reset(self, seed: int | None = None) -> dict[str, Array]:
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0
        self.data.qpos[self.arm_qpos[0]] = self.rest_q
        self.data.qpos[self.arm_qpos[1]] = self.rest_q
        # Randomized object pose while preserving the common rigid workpiece.
        xy = float(self.cfg.initial_xy_range_m)
        pose = np.r_[self.rng.uniform([-xy, -xy, self.cfg.initial_z_min_m], [xy, xy, self.cfg.initial_z_max_m]), _quat_from_rotvec(self.rng.uniform(-self.cfg.initial_rotation_range_rad, self.cfg.initial_rotation_range_rad, 3))]
        wj = self.model.joint('workpiece_free').id
        addr = self.model.jnt_qposadr[wj]
        self.data.qpos[addr:addr+7] = pose
        mujoco.mj_forward(self.model, self.data)
        self.desired_pose = pose.copy()
        self.stage = AssemblyStage.GRASP
        self.step_count = 0; self.stable_count = 0; self.success = False
        self.previous_action[:] = 0; self.previous_impedance[:] = 0.5; self.last_wrench[:] = 0; self.log.clear()
        self.last_depth[:] = 0.0; self.jam_recovery = False
        self.mapper.limits  # force construction validation
        return self.observations()

    def _workpiece_pose(self) -> tuple[Array, Array]:
        return self.data.xpos[self.workpiece_body].copy(), self.data.xquat[self.workpiece_body].copy()

    def _hole_pose(self, i: int) -> tuple[Array, Array]:
        return self.data.site_xpos[self.hole_ids[i]].copy(), self.data.site_xmat[self.hole_ids[i]].reshape(3, 3).copy()

    def peg_tip(self, i: int) -> Array:
        pos, quat = self._workpiece_pose()
        local = np.array([(-1 if i == 0 else 1) * self.cfg.peg_spacing / 2, 0.0, -self.cfg.peg_length - 0.018])
        rot = self.data.xmat[self.workpiece_body].reshape(3, 3)
        return pos + rot @ local

    def peg_errors(self) -> tuple[Array, Array]:
        lateral, depth = [], []
        for i in range(2):
            hole, _ = self._hole_pose(i)
            tip = self.peg_tip(i)
            d = hole - tip
            lateral.append(float(np.linalg.norm(d[:2])))
            # Insertion is measured from the annular opening plane, not from
            # the peg length: a peg above the receiver has zero depth.
            depth.append(float(np.clip(hole[2] - tip[2], 0.0, self.cfg.peg_length)))
        return np.asarray(lateral), np.asarray(depth)

    def _relative_orientation_error(self) -> float:
        mat = self.data.xmat[self.workpiece_body].reshape(3, 3)
        hole_mat = self.data.site_xmat[self.hole_ids[0]].reshape(3, 3)
        trace = np.clip((np.trace(hole_mat.T @ mat) - 1) / 2, -1, 1)
        return float(np.degrees(np.arccos(trace)))

    def _contact_wrench(self) -> Array:
        wrench = np.zeros(6)
        buf = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in self.peg_ids or contact.geom2 in self.peg_ids:
                mujoco.mj_contactForce(self.model, self.data, i, buf)
                wrench += buf
        return wrench

    def _contact_state(self, wrench: Array | None = None) -> bool:
        """Return solver or geometric first-contact state at the hole mouths."""
        force = self._contact_wrench() if wrench is None else np.asarray(wrench)
        if np.linalg.norm(force[:3]) > 1e-5:
            return True
        lateral, _ = self.peg_errors()
        mouth_dist = []
        for i in range(2):
            hole, _ = self._hole_pose(i)
            mouth_dist.append(abs(float(hole[2] - self.peg_tip(i)[2])))
        return bool(max(mouth_dist) < 0.004 and max(lateral) < self.cfg.hole_radius + 0.006)

    def observations(self) -> dict[str, Array]:
        pos, quat = self._workpiece_pose()
        hole, hole_mat = self._hole_pose(0)
        rel_pos = hole - pos
        rel_ori = _rotvec(_quat_mul(_quat_conj(quat), _mat_quat(hole_mat)))
        qvel = self.data.qvel[self.model.jnt_dofadr[self.model.joint('workpiece_free').id]:][:6]
        lat, dep = self.peg_errors()
        wrench = self._contact_wrench()
        contact = float(self._contact_state(wrench))
        common = np.r_[rel_pos, rel_ori, qvel, lat, dep, wrench, contact, float(self.stage), self.previous_action]
        trajectory = common.astype(float)
        impedance = np.r_[wrench, rel_pos, rel_ori, qvel, lat, dep, contact, float(self.stage), self.previous_action, self.previous_impedance]
        return {'trajectory': trajectory, 'impedance': impedance}

    def _update_stage(self, contact: bool, lat: Array, dep: Array) -> None:
        if self.success: return
        if self.stage is AssemblyStage.GRASP: self.stage = AssemblyStage.LIFT
        elif self.stage is AssemblyStage.LIFT and self._workpiece_pose()[0][2] > 0.25: self.stage = AssemblyStage.TRANSPORT
        elif self.stage is AssemblyStage.TRANSPORT and np.linalg.norm(self._workpiece_pose()[0][:2]) < 0.08: self.stage = AssemblyStage.COARSE_ALIGNMENT
        elif self.stage is AssemblyStage.COARSE_ALIGNMENT and np.max(lat) < 0.03: self.stage = AssemblyStage.APPROACH
        elif self.stage is AssemblyStage.APPROACH and np.max(lat) < 0.012: self.stage = AssemblyStage.FIRST_CONTACT if contact else self.stage
        elif self.stage is AssemblyStage.FIRST_CONTACT and (contact or np.any(dep > 0.0)): self.stage = AssemblyStage.COMPLIANT_ALIGNMENT
        elif self.stage is AssemblyStage.COMPLIANT_ALIGNMENT and (np.max(lat) < self.cfg.lateral_threshold or np.any(dep > 0.0)): self.stage = AssemblyStage.INSERTION
        if np.all(lat < self.cfg.lateral_threshold) and np.all(dep > self.cfg.required_depth) and self._relative_orientation_error() < self.cfg.orientation_threshold_deg:
            self.stable_count += 1
            if self.stable_count >= self.cfg.stable_steps:
                self.stage = AssemblyStage.SUCCESS; self.success = True
        else:
            self.stable_count = 0

    def _desired_from_action(self, action: Array) -> Array:
        a = np.asarray(action, dtype=float)
        if a.shape != (6,) or not np.all(np.isfinite(a)): raise ValueError('trajectory action must have shape (6,)')
        delta = np.r_[np.clip(a[:3], -1, 1) * self.cfg.action_translation_limit, np.clip(a[3:], -1, 1) * self.cfg.action_rotation_limit_rad]
        pos = self.desired_pose[:3] + delta[:3]
        quat = _quat_mul(self.desired_pose[3:], _quat_from_rotvec(delta[3:]))
        self.desired_pose = np.r_[pos, quat / np.linalg.norm(quat)]
        return self.desired_pose

    def _impedance(self, action: Array, contact: bool) -> tuple[Array, Array]:
        a = np.asarray(action, dtype=float)
        if a.shape != (12,) or not np.all(np.isfinite(a)): raise ValueError('impedance action must have shape (12,)')
        # Positive bounded physical gains; actions are normalized [0, 1].
        scale = np.clip(a, 0.0, 1.0)
        if contact:
            # This is a stability envelope, not a stage lookup policy: the
            # policy still supplies every parameter and can choose any value.
            scale[:3] = np.minimum(scale[:3], 0.65)
            scale[3:6] = np.minimum(scale[3:6], 0.55)
            scale[6:] = np.maximum(scale[6:], 0.45)
        if self.jam_recovery:
            # A physical jam is an event-driven safety envelope.  Agent 2
            # still supplies all parameters; this prevents the baseline from
            # continuing to drive the insertion axis into a stalled contact.
            scale[2] = min(scale[2], 0.25)
            scale[3:6] = np.minimum(scale[3:6], 0.35)
        kp = np.array([180, 180, 260, 30, 30, 22]) * (0.55 + 1.45 * scale[:6])
        kd = np.array([2.0, 2.0, 2.5, 0.7, 0.7, 0.5]) * (0.8 + 1.4 * scale[6:])
        self.previous_impedance = scale.copy()
        return kp, kd

    def step(self, trajectory_action: Array, impedance_action: Array) -> tuple[dict[str, Array], float, bool, dict[str, Any]]:
        if self.step_count >= self.cfg.max_steps: return self.observations(), -10.0, True, {'stage': self.stage.name, 'success': self.success}
        self._desired_from_action(trajectory_action)
        contact_wrench = self._contact_wrench()
        lat, dep = self.peg_errors()
        contact = self._contact_state(contact_wrench)
        impedance_prev = self.previous_impedance.copy()
        kp, kd = self._impedance(impedance_action, contact)
        pos, quat = self._workpiece_pose()
        pose_error = np.r_[self.desired_pose[:3] - pos, _rotvec(_quat_mul(_quat_conj(quat), self.desired_pose[3:]))]
        vel = self.data.qvel[self.model.jnt_dofadr[self.model.joint('workpiece_free').id]:][:6]
        # Contact wrench is part of Agent 2's observation.  The Cartesian
        # tracking command remains pose/velocity based so a contact normal
        # cannot reverse the trajectory command and drive a peg deeper into a
        # wall; compliance comes from the policy-selected K/D values.
        wrench6 = kp * pose_error - kd * vel
        wrench6[:3] = wrench6[:3] / max(1.0, np.linalg.norm(wrench6[:3]) / self.cfg.max_force)
        wrench6[3:] = wrench6[3:] / max(1.0, np.linalg.norm(wrench6[3:]) / self.cfg.max_torque)
        # Both arms receive the same object-level wrench, preserving one rigid
        # workpiece ownership and cooperative manipulation.
        jac = np.zeros((2, 6, 7)); q = self.data.qpos[self.arm_qpos]; qd = self.data.qvel[self.arm_dof]
        for arm, site_id in enumerate(self.ee_ids):
            jp = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv)); mujoco.mj_jacSite(self.model, self.data, jp, jr, site_id)
            jac[arm] = np.vstack((jp[:, self.arm_dof[arm]], jr[:, self.arm_dof[arm]]))
        gravity_comp = np.asarray([self.data.qfrc_bias[self.arm_dof[arm]] for arm in range(2)])
        mapped = self.mapper.compute(np.repeat(wrench6[None, :], 2, axis=0), jac, q, qd, np.repeat(self.rest_q[None, :], 2, axis=0), feedforward_torque=gravity_comp, stopped=False)
        self.data.ctrl[:] = mapped.joint_torque.reshape(-1)
        for _ in range(self.cfg.control_interval): mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        contact_wrench = self._contact_wrench(); lat, dep = self.peg_errors()
        contact_now = self._contact_state(contact_wrench)
        self.jam_recovery = bool(contact_now and np.max(dep - self.last_depth) < 1e-5 and np.linalg.norm(contact_wrench[:3]) > 8.0)
        self._update_stage(contact_now, lat, dep)
        self.last_depth = dep.copy()
        self.previous_action = np.asarray(trajectory_action, dtype=float).copy(); self.last_wrench = contact_wrench.copy()
        next_obs = self.observations()
        success = self.success
        reward = self.reward(lat, dep, contact_wrench, pose_error, success)
        force_term = -0.002 * np.linalg.norm(contact_wrench[:3])**2
        torque_term = -0.001 * np.linalg.norm(contact_wrench[3:])**2
        smooth_term = -0.1 * np.linalg.norm(np.asarray(impedance_action, dtype=float) - impedance_prev)
        success_term = 250.0 if success else 0.0
        contact_stability = 0.5 if contact_now and np.linalg.norm(contact_wrench[:3]) < self.cfg.max_force else 0.0
        jamming_term = -2.0 if self.jam_recovery else 0.0
        agent1_base = -4*np.linalg.norm(pose_error[:3]) - 1.5*np.linalg.norm(pose_error[3:]) - 8*np.sum(lat) + 8*np.sum(dep)/(2*self.cfg.required_depth)
        info = {'stage': self.stage.name, 'stage_index': int(self.stage), 'contact': contact_now, 'jamming': self.jam_recovery, 'agent1_action': np.asarray(trajectory_action).copy(), 'impedance': np.r_[kp, kd], 'kp': kp.copy(), 'kd': kd.copy(), 'wrench': contact_wrench.copy(), 'peg1_lateral_error': lat[0], 'peg2_lateral_error': lat[1], 'peg1_depth': dep[0], 'peg2_depth': dep[1], 'relative_position_error': float(np.linalg.norm(pose_error[:3])), 'relative_orientation_error': float(np.degrees(np.linalg.norm(pose_error[3:]))), 'r_force': float(force_term), 'r_torque': float(torque_term), 'r_jamming': float(jamming_term), 'r_impedance_smoothness': float(smooth_term), 'r_contact_stability': float(contact_stability), 'r_success': float(success_term), 'agent1_reward': float(agent1_base + success_term - 0.01), 'agent2_reward': float(force_term + torque_term + jamming_term + smooth_term + contact_stability + success_term), 'success': success}
        self.log.append(info)
        return next_obs, reward, success or self.stage is AssemblyStage.SUCCESS, info

    def reward(self, lateral: Array, depth: Array, wrench: Array, pose_error: Array, success: bool) -> float:
        progress = float(np.sum(depth) / (2 * self.cfg.required_depth))
        r1 = -4*np.linalg.norm(pose_error[:3]) - 1.5*np.linalg.norm(pose_error[3:]) - 8*float(np.sum(lateral)) + 8*progress + 2*float(np.sum(depth > 0))
        r2 = -0.002*np.linalg.norm(wrench[:3])**2 - 0.001*np.linalg.norm(wrench[3:])**2 - 0.1*np.linalg.norm(np.diff(self.previous_impedance))
        return r1 + r2 + (250.0 if success else 0.0) - 0.01

    def close(self) -> None:
        del self.data
        del self.model


class A2POCoordinator:
    """Small adapter defining the two-policy A2PO contract.

    The repository intentionally does not vendor an RL framework.  A trainer
    can call ``act`` with two policy callables; Agent 2 receives Agent 1's
    action in its observation, as required by the task specification.
    """

    def __init__(self, trajectory_policy: Any, impedance_policy: Any):
        self.trajectory_policy = trajectory_policy
        self.impedance_policy = impedance_policy

    def act(self, observations: Mapping[str, Array]) -> AgentActions:
        trajectory = np.asarray(self.trajectory_policy(observations['trajectory']), dtype=float)
        impedance_obs = np.r_[observations['impedance'], trajectory]
        impedance = np.asarray(self.impedance_policy(impedance_obs), dtype=float)
        return AgentActions(trajectory=trajectory, impedance=impedance)
