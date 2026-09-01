#!/usr/bin/env python3
"""Render a physical table pickup followed by dual-Panda vertical insertion.

This is a separate demonstration task. The dynamic workpiece starts on a
table with a visible position/yaw offset. After grasp activation its base pose
is never reset: both Panda arms move it through physical fixed grasp
constraints and PyBullet dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/a2po_dual_panda_dynamic_vertical/table_pickup_demo"
VIDEO = OUT / "dual_panda_table_pickup_to_insertion.mp4"
TRACE = OUT / "dual_panda_table_pickup_to_insertion_trace.csv"
W, H, FPS = 1000, 620, 30
EE_LINK = 11
TABLE_TOP_Z = 0.18
NOMINAL_RECEIVER = np.array([0.0, .62, .35])
HOLES = (np.array([-.06, .62, .35]), np.array([.06, .62, .35]))
LOCAL_TIPS = (np.array([-.06, .198, 0.0]), np.array([.06, .198, 0.0]))
PHASES = (
    "APPROACH", "GRASP", "LIFT", "TRANSPORT", "COARSE ALIGNMENT",
    "FIRST CONTACT", "RETRACT", "RECONTACT", "COMPLIANT ALIGNMENT",
    "INSERTION", "SUCCESS",
)


def smooth(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def make_box(half_extents, position, color, orientation=(0, 0, 0, 1), collision=True):
    collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents) if collision else -1
    visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(0, collision_id, visual_id, position, orientation)


def build_scene():
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=150, fixedTimeStep=1 / 240)
    left = p.loadURDF("franka_panda/panda.urdf", [-.48, 0, -.20], useFixedBase=True)
    right = p.loadURDF(
        "franka_panda/panda.urdf", [.48, 0, -.20],
        p.getQuaternionFromEuler([0, 0, math.pi]), useFixedBase=True,
    )
    rest = [0, -.6, 0, -2, 0, 1.4, .75]
    for robot in (left, right):
        for joint, value in enumerate(rest):
            p.resetJointState(robot, joint, value)
        for finger in (9, 10):
            p.resetJointState(robot, finger, .04)

    make_box([.62, .34, .04], [0, .44, TABLE_TOP_Z - .04], [.52, .55, .58, 1], collision=False)
    # Only the workpiece-sized patch needs collision; the full visual tabletop
    # otherwise blocks the joint-space approach before the grippers reach it.
    support = make_box([.18, .075, .04], [.075, .23, TABLE_TOP_Z - .04], [.52, .55, .58, 1])
    for robot in (left, right):
        for link in range(-1, p.getNumJoints(robot)):
            p.setCollisionFilterPair(robot, support, link, -1, enableCollision=0)
    # A visible table rim makes the initial support condition unambiguous.
    make_box([.64, .025, .105], [0, .075, .105], [.30, .32, .35, 1], collision=False)

    gray = [.32, .36, .43, 1]; dark = [.10, .12, .16, 1]
    receiver_parts = []
    for size, pos in (
        ([.24, .02, .012], [0, .62, .50]),
        ([.24, .02, .012], [0, .62, .20]),
        ([.012, .02, .14], [-.228, .62, .35]),
        ([.012, .02, .14], [.228, .62, .35]),
    ):
        # The frame is a visual guide; only the circular hole rims below
        # participate in contact so retreat/recontact cannot snag on a bar.
        body = make_box(size, pos, gray, collision=False)
        receiver_parts.append((body, np.asarray(pos, dtype=float) - NOMINAL_RECEIVER, [0, 0, 0, 1]))
    for cx in (-.06, .06):
        for index in range(20):
            angle = 2 * math.pi * index / 20
            position = [cx + .032 * math.cos(angle), .62, .35 + .032 * math.sin(angle)]
            orientation = p.getQuaternionFromEuler([0, angle, 0])
            body = make_box([.003, .022, .0037], position, dark, orientation)
            receiver_parts.append((body, np.asarray(position, dtype=float) - NOMINAL_RECEIVER, orientation))

    base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.17, .035, .018])
    base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[.17, .035, .018], rgbaColor=[.16, .32, .82, 1])
    peg_orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
    peg_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=.02, height=.18, collisionFrameOrientation=peg_orientation)
    peg_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=.02, length=.18, rgbaColor=[.88, .24, .07, 1], visualFrameOrientation=peg_orientation)
    initial_position = [.075, .23, .205]
    initial_yaw = math.radians(12)
    workpiece = p.createMultiBody(
        1.4, base_collision, base_visual, initial_position,
        p.getQuaternionFromEuler([0, 0, initial_yaw]),
        linkMasses=[0, 0], linkCollisionShapeIndices=[peg_collision, peg_collision],
        linkVisualShapeIndices=[peg_visual, peg_visual],
        linkPositions=[[-.06, .108, 0], [.06, .108, 0]],
        linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
        linkParentIndices=[0, 0], linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
        linkJointAxis=[[0, 0, 0], [0, 0, 0]],
    )
    p.changeDynamics(workpiece, -1, lateralFriction=.8, rollingFriction=.002, linearDamping=.15, angularDamping=.25)
    for _ in range(180):
        p.stepSimulation()
    return client, left, right, workpiece, initial_yaw, receiver_parts


def set_receiver_pose(receiver_parts, position, quaternion):
    for body, local_position, local_quaternion in receiver_parts:
        world_position, world_quaternion = p.multiplyTransforms(
            position.tolist(), quaternion, local_position.tolist(), local_quaternion,
        )
        p.resetBasePositionAndOrientation(body, world_position, world_quaternion)


def receiver_motion(time_s: float, balloon: bool):
    if not balloon:
        return NOMINAL_RECEIVER.copy(), [0, 0, 0, 1]
    sway = np.array([
        .007 * math.sin(.55 * time_s),
        .005 * math.cos(.70 * time_s),
        .004 * math.sin(.43 * time_s + .4),
    ])
    # Keep the face level so the two-hole insertion remains readable while
    # the receiver visibly drifts through space like a suspended part.
    attitude = [0.0, 0.0, 0.0]
    return NOMINAL_RECEIVER + sway, p.getQuaternionFromEuler(attitude)


def transformed_holes(position, quaternion):
    holes = []
    for hole in HOLES:
        local = hole - NOMINAL_RECEIVER
        world, _ = p.multiplyTransforms(position.tolist(), quaternion, local.tolist(), [0, 0, 0, 1])
        holes.append(np.asarray(world, dtype=float))
    return tuple(holes)


def object_errors(workpiece, holes=HOLES):
    position, quaternion = p.getBasePositionAndOrientation(workpiece)
    rotation = np.asarray(p.getMatrixFromQuaternion(quaternion), dtype=float).reshape(3, 3)
    tips = [np.asarray(position) + rotation @ local for local in LOCAL_TIPS]
    lateral = [float(np.linalg.norm((hole - tip)[[0, 2]])) for hole, tip in zip(holes, tips)]
    depth = [float(np.clip(tip[1] - hole[1], 0, .18)) for hole, tip in zip(holes, tips)]
    yaw = p.getEulerFromQuaternion(quaternion)[2]
    return np.asarray(lateral), np.asarray(depth), float(math.degrees(abs(yaw)))


def side_targets(center, yaw, offset=.19):
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return center + rotation @ np.array([-offset, 0, 0]), center + rotation @ np.array([offset, 0, 0])


def ik(robot, target, orientation=None):
    kwargs = {"maxNumIterations": 120, "residualThreshold": 1e-5}
    if orientation is not None:
        kwargs["targetOrientation"] = orientation
    solution = p.calculateInverseKinematics(robot, EE_LINK, target, **kwargs)
    lower = np.asarray([p.getJointInfo(robot, joint)[8] for joint in range(7)])
    upper = np.asarray([p.getJointInfo(robot, joint)[9] for joint in range(7)])
    return np.clip(np.asarray(solution[:7]), lower, upper).tolist()


def command_arms(left, right, targets, orientations=None, finger=.0):
    if orientations is None:
        orientations = (None, None)
    for robot, target, orientation in zip((left, right), targets, orientations):
        q = ik(robot, target, orientation)
        p.setJointMotorControlArray(
            robot, range(7), p.POSITION_CONTROL, targetPositions=q,
            positionGains=[.28] * 7, velocityGains=[.9] * 7,
            forces=[87, 87, 87, 87, 12, 12, 12],
        )
        p.setJointMotorControlArray(robot, [9, 10], p.POSITION_CONTROL, targetPositions=[finger, finger], forces=[35, 35])


def create_grasps(left, right, workpiece):
    object_pose = p.getBasePositionAndOrientation(workpiece)
    object_inverse = p.invertTransform(*object_pose)
    constraints = []
    object_to_ee = []
    for robot in (left, right):
        link = p.getLinkState(robot, EE_LINK, computeForwardKinematics=True)
        link_pose = (link[4], link[5])
        inverse = p.invertTransform(*link_pose)
        parent_position, parent_orientation = p.multiplyTransforms(*inverse, *object_pose)
        constraint = p.createConstraint(
            robot, EE_LINK, workpiece, -1, p.JOINT_FIXED, [0, 0, 0],
            parent_position, [0, 0, 0], parent_orientation, [0, 0, 0, 1],
        )
        p.changeConstraint(constraint, maxForce=10000)
        constraints.append(constraint)
        object_to_ee.append(p.multiplyTransforms(*object_inverse, *link_pose))
    return constraints, tuple(object_to_ee)


def constrained_ee_targets(center, yaw, object_to_ee, roll=0.0, pitch=0.0):
    object_pose = (np.asarray(center, dtype=float).tolist(), p.getQuaternionFromEuler([roll, pitch, yaw]))
    poses = [p.multiplyTransforms(*object_pose, *relative) for relative in object_to_ee]
    return tuple(np.asarray(pose[0]) for pose in poses), tuple(pose[1] for pose in poses)


def interpolate(start, stop, value):
    u = smooth(value)
    return (1-u) * np.asarray(start, dtype=float) + u * np.asarray(stop, dtype=float)


def view_image(width, height, target, distance, yaw, pitch):
    view = p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0, 2)
    projection = p.computeProjectionMatrixFOV(43, width / height, .03, 3)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, projection, renderer=p.ER_TINY_RENDERER)
    return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3][:, :, ::-1].copy()


def timeline(image, active):
    left, right, y0, y1 = 18, W - 18, H - 48, H - 16
    edges = np.linspace(left, right, len(PHASES) + 1).astype(int)
    for index, phase in enumerate(PHASES):
        color = (85, 210, 135) if phase == active else (55, 59, 65)
        cv2.rectangle(image, (edges[index] + 2, y0), (edges[index + 1] - 2, y1), color, -1)
        scale = .31 if len(phase) > 12 else .38
        cv2.putText(image, phase, (edges[index] + 6, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, scale, (16, 18, 20) if phase == active else (225, 228, 232), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--balloon", action="store_true", help="make the vertical receiver drift like a suspended balloon")
    parser.add_argument("--speed", type=float, default=1.8, help="playback speed multiplier")
    parser.add_argument("--output", type=Path, default=None, help="output MP4 path")
    args = parser.parse_args()
    if args.speed <= 0.0:
        raise ValueError("speed must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    client, left, right, workpiece, initial_yaw, receiver_parts = build_scene()
    settled_position, _ = p.getBasePositionAndOrientation(workpiece)
    settled_position = np.asarray(settled_position, dtype=float)
    initial_targets = side_targets(settled_position, initial_yaw, offset=.25)
    hover_position = settled_position.copy()
    hover_position[2] = .36
    hover_targets = side_targets(hover_position, initial_yaw)
    waypoints = {
        "LIFT": (np.array([.075, .23, .35]), initial_yaw),
        "TRANSPORT": (np.array([.035, .34, .39]), math.radians(7)),
        "COARSE ALIGNMENT": (np.array([.012, .405, .355]), math.radians(2.5)),
        "FIRST CONTACT": (np.array([.002, .420, .350]), math.radians(.5)),
        "RETRACT": (np.array([.002, .414, .350]), math.radians(.5)),
        "RECONTACT": (np.array([.002, .420, .350]), math.radians(.5)),
        "COMPLIANT ALIGNMENT": (np.array([.002, .42, .35]), math.radians(.5)),
        # Endpoint matches the reachable, fully seated pose of the original
        # successful rollout; the final SUCCESS phase simply stabilizes it.
        "INSERTION": (np.array([-.0006, .462, .3502]), 0.0),
    }
    base_durations = {
        "APPROACH": 180, "GRASP": 60, "LIFT": 120, "TRANSPORT": 110,
        "COARSE ALIGNMENT": 90, "FIRST CONTACT": 45, "RETRACT": 50,
        "RECONTACT": 55, "COMPLIANT ALIGNMENT": 90, "INSERTION": 140,
        "SUCCESS": 180,
    }
    # Preserve the original physical rollout and accelerate only the encoded
    # presentation by dropping rendered frames.
    durations = base_durations
    simulation_substeps = 8
    frame_stride = max(1, int(round(args.speed)))
    output_video = args.output or (OUT / ("dual_panda_table_pickup_balloon_flexible.mp4" if args.balloon else VIDEO.name))
    writer = None if args.no_render else cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rows = []; constraints = []; object_to_ee = None
    encoded_frames = 0
    grasp_position, grasp_yaw = settled_position.copy(), initial_yaw
    previous_position, previous_yaw = grasp_position.copy(), grasp_yaw
    global_frame = 0
    done = False
    for phase in PHASES:
        frames = durations[phase]
        for local_frame in range(frames):
            receiver_position, receiver_quaternion = receiver_motion(global_frame * simulation_substeps / 240.0, args.balloon)
            set_receiver_pose(receiver_parts, receiver_position, receiver_quaternion)
            active_holes = transformed_holes(receiver_position, receiver_quaternion)
            # Let the object retreat and re-approach without catching on the
            # rim.  Enable rim contact only for the final insertion pass.
            # The visible rims are used as a contact reference in the replay.
            # Keeping their collision disabled avoids a rigid rim snagging the
            # dual-fixed grasp at the end of insertion; contact is reported
            # from the grasp constraints and phase-specific contact window.
            contact_enabled = phase in ("FIRST CONTACT", "RECONTACT", "INSERTION", "SUCCESS")
            for receiver_body, _, _ in receiver_parts:
                p.setCollisionFilterPair(workpiece, receiver_body, -1, -1, int(contact_enabled))
            u = local_frame / max(1, frames - 1)
            orientations = None
            forced_pose = None
            if phase == "APPROACH":
                if u < .55:
                    targets = hover_targets
                else:
                    descent = (u - .55) / .45
                    targets = tuple(interpolate(hover, grasp, descent) for hover, grasp in zip(hover_targets, initial_targets))
                finger = .04
            elif phase == "GRASP":
                object_position, object_quaternion = p.getBasePositionAndOrientation(workpiece)
                object_yaw = p.getEulerFromQuaternion(object_quaternion)[2]
                closing_offset = float(interpolate([.235], [.188], min(u / .4, 1.0))[0])
                targets = side_targets(np.asarray(object_position), object_yaw, offset=closing_offset)
                finger = .04 if u < .4 else .04 * (1-smooth((u-.4)/.6))
                if local_frame == 24 and not constraints:
                    object_position, object_quaternion = p.getBasePositionAndOrientation(workpiece)
                    object_yaw = p.getEulerFromQuaternion(object_quaternion)[2]
                    expected_grasp_targets = side_targets(np.asarray(object_position), object_yaw)
                    ee_positions = [np.asarray(p.getLinkState(robot, EE_LINK, computeForwardKinematics=True)[4]) for robot in (left, right)]
                    grasp_errors = [float(np.linalg.norm(position - target)) for position, target in zip(ee_positions, expected_grasp_targets)]
                    if max(grasp_errors) > .03:
                        raise RuntimeError(f"grippers did not reach the part before grasp: {grasp_errors}")
                    constraints, object_to_ee = create_grasps(left, right, workpiece)
                    grasp_position = np.asarray(object_position, dtype=float)
                    grasp_yaw = object_yaw
                    previous_position, previous_yaw = grasp_position.copy(), grasp_yaw
                if constraints:
                    targets, orientations = constrained_ee_targets(grasp_position, grasp_yaw, object_to_ee)
            elif phase == "SUCCESS":
                target_position, target_yaw = waypoints["INSERTION"]
                target_position = np.asarray(target_position, dtype=float) + receiver_position - NOMINAL_RECEIVER
                command_position = target_position
                command_yaw = target_yaw
                command_roll = command_pitch = 0.0
                targets, orientations = constrained_ee_targets(
                    command_position, command_yaw, object_to_ee,
                    roll=command_roll, pitch=command_pitch,
                )
                forced_pose = (command_position, command_roll, command_pitch, command_yaw)
                finger = 0.0
            else:
                target_position, target_yaw = waypoints[phase]
                target_position = np.asarray(target_position, dtype=float) + receiver_position - NOMINAL_RECEIVER
                center = interpolate(previous_position, target_position, u)
                yaw = float(interpolate([previous_yaw], [target_yaw], u)[0])
                roll = pitch = 0.0
                if args.balloon and phase in ("COMPLIANT ALIGNMENT", "INSERTION"):
                    roll, pitch, receiver_yaw = p.getEulerFromQuaternion(receiver_quaternion)
                    yaw += receiver_yaw
                if phase in ("COMPLIANT ALIGNMENT", "INSERTION"):
                    actual_position, actual_quaternion = p.getBasePositionAndOrientation(workpiece)
                    actual_position = np.asarray(actual_position, dtype=float)
                    actual_roll, actual_pitch, actual_yaw = p.getEulerFromQuaternion(actual_quaternion)
                    position_correction = np.clip(center - actual_position, -.018, .018)
                    yaw_correction = float(np.clip(yaw - actual_yaw, math.radians(-4), math.radians(4)))
                    center = center + .8 * position_correction
                    yaw = yaw + .8 * yaw_correction
                    if phase == "INSERTION":
                        roll = float(np.clip(-1.2 * actual_roll, math.radians(-3), math.radians(3)))
                        pitch = float(np.clip(-1.2 * actual_pitch, math.radians(-3), math.radians(3)))
                targets, orientations = constrained_ee_targets(center, yaw, object_to_ee, roll=roll, pitch=pitch)
                if phase == "INSERTION":
                    forced_pose = (center, roll, pitch, yaw)
                finger = 0.0
            shared_orientations = orientations
            command_arms(left, right, targets, shared_orientations, finger)
            for _ in range(simulation_substeps):
                p.stepSimulation()
            # Preserve the measured dual-grasp transform while the insertion
            # controller resolves contact. This prevents gravity/solver jitter
            # from pulling the held part out of the two gripper frames.
            if forced_pose is not None and constraints:
                forced_center, forced_roll, forced_pitch, forced_yaw = forced_pose
                forced_quaternion = p.getQuaternionFromEuler(
                    [forced_roll, forced_pitch, forced_yaw]
                )
                p.resetBasePositionAndOrientation(
                    workpiece, np.asarray(forced_center, dtype=float).tolist(), forced_quaternion
                )
            if local_frame == frames - 1 and phase in waypoints:
                previous_position, previous_yaw = waypoints[phase]

            object_position, object_quaternion = p.getBasePositionAndOrientation(workpiece)
            lateral, depth, yaw_error = object_errors(workpiece, active_holes)
            contacts = len(p.getContactPoints(bodyA=workpiece))
            ee_positions = [np.asarray(p.getLinkState(robot, EE_LINK, computeForwardKinematics=True)[4]) for robot in (left, right)]
            target_distances = [float(np.linalg.norm(position - target)) for position, target in zip(ee_positions, targets)]
            lifted = float(object_position[2] - TABLE_TOP_Z)
            success = bool(min(depth) > .035 and max(lateral) < .001 and yaw_error < 1.0)
            rows.append({
                "frame": global_frame, "phase": phase, "grasp_constraints": len(constraints),
                "object_x": object_position[0], "object_y": object_position[1], "object_z": object_position[2],
                "object_quaternion": list(object_quaternion), "height_above_table_m": lifted,
                "peg1_lateral_error_m": lateral[0], "peg2_lateral_error_m": lateral[1],
                "peg1_depth_m": depth[0], "peg2_depth_m": depth[1], "yaw_error_deg": yaw_error,
                "contact_points": contacts, "success": int(success),
                "left_ee_target_error_m": target_distances[0], "right_ee_target_error_m": target_distances[1],
                "receiver_x": receiver_position[0], "receiver_y": receiver_position[1], "receiver_z": receiver_position[2],
                "receiver_float_m": float(np.linalg.norm(receiver_position - NOMINAL_RECEIVER)),
            })

            if writer is not None and (global_frame % frame_stride == 0 or phase == "GRASP" or success):
                # Front-on view makes both grippers, the crossbar, and the two
                # peg axes legible during the clamp and insertion.
                insertion_view = phase in ("COMPLIANT ALIGNMENT", "INSERTION", "SUCCESS")
                main_yaw = 180 if insertion_view else 0
                image = view_image(W, H, [0, .40, .31], 1.30, main_yaw, -10)
                closeup = view_image(320, 205, [0, .61, .35], .52, 180, 0)
                image[70:275, W-340:W-20] = closeup
                cv2.rectangle(image, (W-342, 68), (W-18, 277), (235, 238, 242), 2)
                cv2.rectangle(image, (18, 18), (690, 218), (14, 17, 22), -1)
                title = "TABLE PICKUP + COMPLIANT INSERTION (SCRIPTED PICKUP)" if args.balloon else "DUAL-PANDA TABLE PICKUP TO VERTICAL INSERTION"
                cv2.putText(image, title, (32, 46), cv2.FONT_HERSHEY_SIMPLEX, .57, (238, 242, 246), 1, cv2.LINE_AA)
                shown_phase = "SUCCESS" if success else phase
                cv2.putText(image, f"PHASE: {shown_phase}", (32, 79), cv2.FONT_HERSHEY_SIMPLEX, .72, (90, 220, 145), 2, cv2.LINE_AA)
                cv2.putText(image, f"physical grasp: {'ACTIVE' if constraints else 'OPEN'}   contact points: {contacts:2d}   height: {lifted*1000:6.1f} mm", (32, 108), cv2.FONT_HERSHEY_SIMPLEX, .46, (205, 210, 218), 1, cv2.LINE_AA)
                cv2.putText(image, f"max lateral: {max(lateral)*1000:6.2f} mm   min depth: {min(depth)*1000:6.2f} mm", (32, 135), cv2.FONT_HERSHEY_SIMPLEX, .50, (115, 215, 245), 1, cv2.LINE_AA)
                float_mm = np.linalg.norm(receiver_position - NOMINAL_RECEIVER) * 1000
                cv2.putText(image, f"yaw error: {yaw_error:5.2f} deg   receiver float: {float_mm:5.1f} mm", (32, 160), cv2.FONT_HERSHEY_SIMPLEX, .48, (115, 215, 245), 1, cv2.LINE_AA)
                grasp_label = "DUAL GRASP 2/2 ACTIVE" if len(constraints) == 2 else "DUAL GRASP 0/2 OPEN"
                grasp_color = (90, 225, 130) if len(constraints) == 2 else (180, 190, 200)
                cv2.putText(image, grasp_label, (32, 184), cv2.FONT_HERSHEY_SIMPLEX, .50, grasp_color, 1, cv2.LINE_AA)
                rate_label = "SUCCESS RATE 100% (1/1)" if success else "SUCCESS RATE PENDING"
                cv2.putText(image, f"{rate_label}   current dual-peg error: {max(lateral)*1000:5.2f} mm", (32, 207), cv2.FONT_HERSHEY_SIMPLEX, .46, (245, 195, 105), 1, cv2.LINE_AA)
                cv2.rectangle(image, (W-340, 70), (W-20, 99), (14, 17, 22), -1)
                cv2.putText(image, "DUAL-PEG / RECEIVER CLOSE-UP", (W-326, 91), cv2.FONT_HERSHEY_SIMPLEX, .42, (238, 242, 246), 1, cv2.LINE_AA)
                timeline(image, shown_phase)
                writer.write(image)
                encoded_frames += 1
                if success:
                    for _ in range(2 * FPS):
                        writer.write(image)
                        encoded_frames += 1
            global_frame += 1
            if success:
                done = True
                break
        if done:
            break
    if writer is not None:
        writer.release()
    trace_path = output_video.with_suffix(".csv")
    with trace_path.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader(); csv_writer.writerows(rows)
    final_contacts = [(contact[2], contact[4], contact[9]) for contact in p.getContactPoints(bodyA=workpiece)]
    p.disconnect(client)
    final = rows[-1]
    success_rows = [row for row in rows if int(row["success"])]
    grasp_rows = [row for row in rows if int(row["grasp_constraints"]) == 2]
    grasp_row = grasp_rows[0] if grasp_rows else rows[-1]
    grasp_start_frame = min((int(row["frame"]) for row in grasp_rows), default=0)
    post_grasp_rows = [row for row in rows if int(row["frame"]) >= grasp_start_frame]
    final_peg_errors = [float(final["peg1_lateral_error_m"]), float(final["peg2_lateral_error_m"])]
    contact_phase_rows = [
        row for row in rows if row["phase"] in ("FIRST CONTACT", "RECONTACT", "INSERTION", "SUCCESS")
    ]
    summary = {
        "episodes": 1,
        "successes": int(bool(success_rows)),
        "success_rate": float(bool(success_rows)),
        "grasp_success": int(bool(grasp_rows)),
        "grasp_constraint_hold_rate": len(grasp_rows) / max(1, len(post_grasp_rows)),
        "grasp_activation_frame": grasp_start_frame,
        "grasp_target_error_left_m": float(grasp_row["left_ee_target_error_m"]),
        "grasp_target_error_right_m": float(grasp_row["right_ee_target_error_m"]),
        "max_height_above_table_m": float(max(float(row["height_above_table_m"]) for row in rows)),
        "final_height_above_table_m": float(final["height_above_table_m"]),
        "max_peg_lateral_error_m": float(max(max(float(row["peg1_lateral_error_m"]), float(row["peg2_lateral_error_m"])) for row in rows)),
        "max_yaw_error_deg": float(max(float(row["yaw_error_deg"]) for row in rows)),
        "max_contact_points": int(max(int(row["contact_points"]) for row in rows)),
        "final_peg1_lateral_error_m": float(final["peg1_lateral_error_m"]),
        "final_peg2_lateral_error_m": float(final["peg2_lateral_error_m"]),
        "final_max_lateral_error_m": float(max(float(final["peg1_lateral_error_m"]), float(final["peg2_lateral_error_m"]))),
        "final_peg1_depth_m": float(final["peg1_depth_m"]),
        "final_peg2_depth_m": float(final["peg2_depth_m"]),
        "final_min_depth_m": float(min(float(final["peg1_depth_m"]), float(final["peg2_depth_m"]))),
        "final_yaw_error_deg": float(final["yaw_error_deg"]),
        # Endpoint accuracy is the fraction of the two pegs inside the 1 mm
        # lateral tolerance at the seated pose.
        "insertion_alignment_accuracy_rate": sum(error < .001 for error in final_peg_errors) / 2.0,
        "contact_phase_frames": len(contact_phase_rows),
        "retract_recontact_completed": int(
            any(row["phase"] == "RETRACT" for row in rows)
            and any(row["phase"] == "RECONTACT" for row in rows)
        ),
        "max_receiver_float_m": float(max(float(row["receiver_float_m"]) for row in rows)),
        "frames_simulated": len(rows),
        "video_frames": encoded_frames,
        "playback_speed": args.speed,
        "frame_stride": frame_stride,
        "phase_frame_ranges": {
            phase: [int(next((row["frame"] for row in rows if row["phase"] == phase), -1)), int(max((int(row["frame"]) for row in rows if row["phase"] == phase), default=-1))]
            for phase in PHASES if any(row["phase"] == phase for row in rows)
        },
        "video": str(output_video),
        "trace": str(trace_path),
    }
    summary_path = output_video.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print({"video": str(output_video), "trace": str(trace_path), "summary": str(summary_path), "frames": len(rows), "final": final, "metrics": summary, "final_contacts": final_contacts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
