#!/usr/bin/env python3
"""Render a dual-Panda cooperative grasp and vertical dynamic insertion replay.

The workpiece is fixed to both Panda wrists with two PyBullet fixed constraints
after the approach phase.  Subsequent object motion is produced by coordinated
arm joint targets; the object base is never reset during the grasp/transport/
insertion phases.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/a2po_dual_panda_dynamic_vertical"
W, H, FPS, FRAMES = 800, 520, 30, 360


def quat(axis, angle):
    axis = np.asarray(axis, float); axis = axis / max(np.linalg.norm(axis), 1e-12)
    return [math.cos(angle / 2), *(math.sin(angle / 2) * axis)]


def make_box(size, pos, color, mass=0.0, orn=(0, 0, 0, 1)):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
    return p.createMultiBody(mass, collision, visual, pos, orn)


def build_scene():
    cid = p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0, 0, 0)
    left = p.loadURDF("franka_panda/panda.urdf", [-.48, 0, -.38], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
    right = p.loadURDF("franka_panda/panda.urdf", [.48, 0, -.38], p.getQuaternionFromEuler([0, 0, math.pi]), useFixedBase=True)
    q_rest = np.array([0, -.6, 0, -2, 0, 1.4, .75])
    for robot in (left, right):
        for j, value in enumerate(q_rest): p.resetJointState(robot, j, value)
    # Vertical X-Z receiver with hole axes along -Y.  The receiver is moved by
    # a bounded six-DoF disturbance trajectory to represent microgravity sway.
    receiver_parts = []
    gray = [.35, .38, .46, 1]; dark = [.12, .14, .18, 1]
    for size, pos in [([.24, .012, .012], [0, .12, .43]), ([.24, .012, .012], [0, .12, .13]), ([.012, .012, .14], [-.228, .12, .28]), ([.012, .012, .14], [.228, .12, .28])]: receiver_parts.append(make_box(size, pos, gray))
    for cx in (-.06, .06):
        for i in range(16):
            a = 2 * math.pi * i / 16
            receiver_parts.append(make_box([.003, .02, .00448], [cx + .028 * math.cos(a), .12, .28 + .028 * math.sin(a)], dark, orn=p.getQuaternionFromEuler([0, a, 0])))
    # Object base and two pegs as fixed visual/collision links.
    base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.17, .035, .018])
    base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[.17, .035, .018], rgbaColor=[.18, .30, .78, 1])
    peg_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=.02, height=.18)
    peg_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=.02, length=.18, rgbaColor=[.86, .25, .08, 1], visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]))
    obj = p.createMultiBody(1.0, base_collision, base_visual, [0, .28, .23], linkMasses=[0, 0], linkCollisionShapeIndices=[peg_collision, peg_collision], linkVisualShapeIndices=[peg_visual, peg_visual], linkPositions=[[-.06, -.108, 0], [.06, -.108, 0]], linkOrientations=[p.getQuaternionFromEuler([math.pi/2,0,0])]*2, linkInertialFramePositions=[[0,0,0],[0,0,0]], linkInertialFrameOrientations=[[0,0,0,1],[0,0,0,1]], linkParentIndices=[0,0], linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED], linkJointAxis=[[0,0,0],[0,0,0]])
    return cid, left, right, obj, receiver_parts, q_rest


def ik(robot, target):
    solution = p.calculateInverseKinematics(robot, 7, target, maxNumIterations=100, residualThreshold=1e-4)
    return np.asarray(solution[:7], dtype=float)


def phase_pose(frame):
    t = frame / (FRAMES - 1)
    if t < .20: u = t / .20; phase = "APPROACH_GRASP"; pos = np.array([.0, .28, .23])
    elif t < .27: u = (t - .20) / .07; phase = "GRASP"; pos = np.array([.0, .28, .23])
    elif t < .42: u = (t - .27) / .15; phase = "LIFT"; pos = np.array([.0, .28, .23]) * (1-u) + np.array([.0, .32, .34]) * u
    elif t < .55: u = (t - .42) / .13; phase = "TRANSPORT"; pos = np.array([.0, .32, .34]) * (1-u) + np.array([.0, .31, .30]) * u
    elif t < .68: u = (t - .55) / .13; phase = "COARSE_ALIGNMENT"; pos = np.array([.0, .31, .30]) * (1-u) + np.array([.004, .29, .28]) * u
    elif t < .77: u = (t - .68) / .09; phase = "FIRST_CONTACT"; pos = np.array([.004, .29, .28]) * (1-u) + np.array([.003, .285, .28]) * u
    elif t < .88: u = (t - .77) / .11; phase = "COMPLIANT_ALIGNMENT"; pos = np.array([.003, .285, .28]) * (1-u) + np.array([0, .275, .28]) * u
    else: u = (t - .88) / .12; phase = "INSERTION"; pos = np.array([0, .275, .28]) * (1-u) + np.array([0, .245, .28]) * u
    # Small target-relative pose correction and persistent receiver sway.
    rot = np.array([.025 * math.sin(3 * t), .035 * math.sin(2 * t), .025 * math.cos(2.5 * t)])
    return pos, rot, phase


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cid, left, right, obj, receiver_parts, q_rest = build_scene()
    receiver_initial = [np.asarray(p.getBasePositionAndOrientation(part)[0], dtype=float) for part in receiver_parts]
    writer = cv2.VideoWriter(str(OUT / "dual_panda_dynamic_vertical_cooperative.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rows = []; grasp_constraints = None
    q_left = q_rest.copy(); q_right = q_rest.copy()
    for frame in range(FRAMES):
        target, rot, phase = phase_pose(frame)
        sway = np.array([.006 * math.sin(frame / 27), .004 * math.cos(frame / 31), .004 * math.sin(frame / 37)])
        # Dynamic target visualization: all receiver pieces move together with
        # a small translation and rotation; the workpiece is arm-constrained.
        receiver_orn = p.getQuaternionFromEuler([.02*math.sin(frame/41), .015*math.cos(frame/37), .01*math.sin(frame/29)])
        for part, initial in zip(receiver_parts, receiver_initial):
            p.resetBasePositionAndOrientation(part, initial + sway, receiver_orn)
        left_target = target + np.array([-.115, 0, 0]); right_target = target + np.array([.115, 0, 0])
        q_left = ik(left, left_target); q_right = ik(right, right_target)
        # Real motor position control, integrated for several physics substeps;
        # no per-frame joint or workpiece pose overwrite occurs here.
        p.setJointMotorControlArray(left, range(7), p.POSITION_CONTROL, targetPositions=q_left.tolist(), positionGains=[.25] * 7, velocityGains=[.8] * 7, forces=[87, 87, 87, 87, 12, 12, 12])
        p.setJointMotorControlArray(right, range(7), p.POSITION_CONTROL, targetPositions=q_right.tolist(), positionGains=[.25] * 7, velocityGains=[.8] * 7, forces=[87, 87, 87, 87, 12, 12, 12])
        if phase == "GRASP" and grasp_constraints is None:
            grasp_constraints = [p.createConstraint(left, 7, obj, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0, 1], [-.115, 0, 0], [0, 0, 0, 1]), p.createConstraint(right, 7, obj, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0, 1], [.115, 0, 0], [0, 0, 0, 1])]
        for _substep in range(8): p.stepSimulation()
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj)
        ee_l = p.getLinkState(left, 7)[4]; ee_r = p.getLinkState(right, 7)[4]
        contacts = len(p.getContactPoints(bodyA=obj))
        actual_left_q = [p.getJointState(left, j)[0] for j in range(7)]; actual_right_q = [p.getJointState(right, j)[0] for j in range(7)]
        obj_mat = np.asarray(p.getMatrixFromQuaternion(obj_orn), dtype=float).reshape(3, 3)
        depths = []; lateral = []
        for cx in (-.06, .06):
            tip = np.asarray(obj_pos) + obj_mat @ np.array([cx, -.198, 0.0]); mouth = np.array([cx + sway[0], .12 + sway[1], .28 + sway[2]])
            lateral.append(float(np.linalg.norm((mouth - tip)[[0, 2]]))); depths.append(float(np.clip(mouth[1] - tip[1], 0, .18)))
        actual_success = bool(min(depths) > .035 and max(lateral) < .003)
        logged_phase = "SUCCESS" if actual_success else phase
        rows.append({"frame": frame, "phase": logged_phase, "planned_phase": phase, "grasped": int(grasp_constraints is not None), "success": int(actual_success), "object_pos": list(obj_pos), "object_quat": list(obj_orn), "receiver_sway": sway.tolist(), "left_q": actual_left_q, "right_q": actual_right_q, "left_ee": list(ee_l), "right_ee": list(ee_r), "peg1_lateral": lateral[0], "peg2_lateral": lateral[1], "peg1_depth": depths[0], "peg2_depth": depths[1], "contact_count": contacts})
        view = p.computeViewMatrixFromYawPitchRoll([0, .10, .20], 1.15, 37, -21, 0, 2); proj = p.computeProjectionMatrixFOV(42, W / H, .03, 3)
        _, _, rgba, _, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER); image = np.asarray(rgba, dtype=np.uint8).reshape(H, W, 4)[:, :, :3][:, :, ::-1].copy()
        cv2.rectangle(image, (15, 15), (570, 115), (15, 18, 24), -1); cv2.putText(image, "DUAL PANDA COOPERATIVE DYNAMIC ASSEMBLY", (27, 43), cv2.FONT_HERSHEY_SIMPLEX, .62, (235,240,245), 1, cv2.LINE_AA); cv2.putText(image, f"phase: {logged_phase}", (27, 70), cv2.FONT_HERSHEY_SIMPLEX, .58, (100,220,170), 1, cv2.LINE_AA); cv2.putText(image, f"two-wrist grasp: {'ON' if grasp_constraints else 'approaching'} | receiver: floating/swaying", (27, 96), cv2.FONT_HERSHEY_SIMPLEX, .48, (245,190,100), 1, cv2.LINE_AA); writer.write(image)
    writer.release();
    with (OUT / "dynamic_vertical_cooperative_trace.csv").open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0])); writer_csv.writeheader(); writer_csv.writerows(rows)
    p.disconnect(cid); print(OUT / "dual_panda_dynamic_vertical_cooperative.mp4")


if __name__ == "__main__": main()
