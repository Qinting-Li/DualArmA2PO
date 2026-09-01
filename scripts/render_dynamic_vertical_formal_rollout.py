#!/usr/bin/env python3
"""Render an actual formal-evaluation rollout as a headless PyBullet replay."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from run_dynamic_vertical_formal_study import (
    A2POTrainConfig,
    A2POTrainer,
    DynamicVerticalDualPandaEnv,
    evaluation_config,
    load_base_config,
    load_weights,
    rollout_episode,
)

W, H, FPS = 800, 520, 30


def xyzw(q_wxyz: list[float]) -> list[float]:
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]


def make_visual(shape: int, *, size=None, radius=None, length=None, color=None, orientation=None) -> int:
    kwargs = {"shapeType": shape, "rgbaColor": color}
    if size is not None:
        kwargs["halfExtents"] = size
    if radius is not None:
        kwargs["radius"] = radius
    if length is not None:
        kwargs["length"] = length
    if orientation is not None:
        kwargs["visualFrameOrientation"] = orientation
    return p.createVisualShape(**kwargs)


def build_scene():
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    left = p.loadURDF("franka_panda/panda.urdf", [-.48, 0, -.38], useFixedBase=True)
    right = p.loadURDF(
        "franka_panda/panda.urdf", [.48, 0, -.38],
        p.getQuaternionFromEuler([0, 0, math.pi]), useFixedBase=True,
    )

    base = make_visual(p.GEOM_BOX, size=[.17, .035, .018], color=[.18, .30, .78, 1])
    peg = make_visual(
        p.GEOM_CYLINDER, radius=.02, length=.18, color=[.86, .25, .08, 1],
        orientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
    )
    workpiece = p.createMultiBody(
        0, -1, base, [0, .4, .28],
        linkMasses=[0, 0], linkCollisionShapeIndices=[-1, -1],
        linkVisualShapeIndices=[peg, peg],
        linkPositions=[[-.06, -.108, 0], [.06, -.108, 0]],
        linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
        linkParentIndices=[0, 0], linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
        linkJointAxis=[[0, 0, 0], [0, 0, 0]],
    )

    gray = [.35, .38, .46, 1]
    dark = [.12, .14, .18, 1]
    receiver_parts: list[tuple[int, list[float], list[float]]] = []
    for size, local in (
        ([.24, .012, .02], [0, .148, 0]),
        ([.24, .012, .02], [0, -.148, 0]),
        ([.012, .14, .02], [-.228, 0, 0]),
        ([.012, .14, .02], [.228, 0, 0]),
    ):
        visual = make_visual(p.GEOM_BOX, size=size, color=gray)
        receiver_parts.append((p.createMultiBody(0, -1, visual), local, [0, 0, 0, 1]))
    for cx in (-.06, .06):
        for index in range(16):
            angle = 2 * math.pi * index / 16
            local = [cx + .028 * math.cos(angle), .028 * math.sin(angle), .02]
            local_q = p.getQuaternionFromEuler([0, 0, angle])
            visual = make_visual(p.GEOM_BOX, size=[.003, .00448, .02], color=dark)
            receiver_parts.append((p.createMultiBody(0, -1, visual), local, local_q))
    return client, left, right, workpiece, receiver_parts


def set_receiver_pose(parts, position, quaternion):
    for body, local_position, local_quaternion in parts:
        world_position, world_quaternion = p.multiplyTransforms(
            position, quaternion, local_position, local_quaternion,
        )
        p.resetBasePositionAndOrientation(body, world_position, world_quaternion)


def render(args: argparse.Namespace) -> int:
    cfg = evaluation_config(load_base_config(), args.setting)
    trainer = A2POTrainer(
        DynamicVerticalDualPandaEnv(cfg, seed=args.seed),
        A2POTrainConfig(seed=args.train_seed, device="cpu"),
        args.output.parent,
    )
    if args.variant != "controller_only":
        load_weights(trainer, args.checkpoint)
        trainer.agent1.eval()
        trainer.agent2.eval()
    result, trace = rollout_episode(
        trainer, cfg, args.seed, args.variant, capture_trace=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix(".csv")
    with trace_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)

    client, left, right, workpiece, receiver_parts = build_scene()
    video = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H),
    )
    stride = max(1, args.frame_stride)
    for state in trace[::stride]:
        for body, values in ((left, state["left_q"]), (right, state["right_q"])):
            for joint, value in enumerate(values):
                p.resetJointState(body, joint, value)
        object_quat = xyzw(state["object_quat"])
        receiver_quat = xyzw(state["receiver_quat"])
        p.resetBasePositionAndOrientation(workpiece, state["object_pos"], object_quat)
        set_receiver_pose(receiver_parts, state["receiver_pos"], receiver_quat)
        view = p.computeViewMatrixFromYawPitchRoll([0, .15, .25], 1.25, 38, -18, 0, 2)
        projection = p.computeProjectionMatrixFOV(42, W / H, .03, 3)
        _, _, rgba, _, _ = p.getCameraImage(
            W, H, view, projection, renderer=p.ER_TINY_RENDERER,
        )
        image = np.asarray(rgba, dtype=np.uint8).reshape(H, W, 4)[:, :, :3][:, :, ::-1].copy()
        outcome = "SUCCESS" if result["success"] else result["failure_reason"].upper()
        cv2.rectangle(image, (14, 14), (580, 118), (15, 18, 24), -1)
        cv2.putText(image, f"FORMAL {args.setting.upper()} ROLLOUT | {args.variant}", (26, 42), cv2.FONT_HERSHEY_SIMPLEX, .60, (238, 242, 246), 1, cv2.LINE_AA)
        cv2.putText(image, f"seed {args.seed} | stage {state['stage']} | {outcome}", (26, 70), cv2.FONT_HERSHEY_SIMPLEX, .54, (105, 220, 170), 1, cv2.LINE_AA)
        cv2.putText(image, f"measured contact force {state['force_N']:.3f} N | step {state['step']}", (26, 98), cv2.FONT_HERSHEY_SIMPLEX, .51, (245, 190, 100), 1, cv2.LINE_AA)
        video.write(image)
    video.release()
    p.disconnect(client)
    print({"video": str(args.output), "trace": str(trace_path), "result": result})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--variant", choices=("controller_only", "a2po"), required=True)
    parser.add_argument("--setting", choices=("nominal", "hard", "ood"), default="hard")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return render(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
