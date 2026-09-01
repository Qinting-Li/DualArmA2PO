#!/usr/bin/env python3
"""Render current 7D Agent 2 internal-force success/slip MuJoCo rollouts."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from render_dynamic_vertical_formal_rollout import build_scene, set_receiver_pose, xyzw  # noqa: E402
from x_bimanual.panda_dual_assembly import (  # noqa: E402
    _quat_conj,
    _quat_mul,
    _rotvec,
    dual_config_from_mapping,
)
from x_bimanual.panda_dynamic_vertical_assembly import DynamicVerticalDualPandaEnv  # noqa: E402

WIDTH, HEIGHT, FPS = 1280, 720, 30
PANEL_WIDTH, PANEL_HEIGHT = 620, 520
MOTION_FRAMES, HOLD_FRAMES = 360, 60
OUTPUT_DIR = ROOT / "outputs/a2po_dual_panda_dynamic_vertical/internal_force"
VIDEO = OUTPUT_DIR / "agent2_internal_force_success_vs_slip.mp4"


def rollout(internal_scale: float, seed: int = 20260824) -> tuple[list[dict], dict]:
    with (ROOT / "configs/task.yaml").open() as handle:
        base = dual_config_from_mapping(yaml.safe_load(handle))
    cfg = replace(
        base,
        gravity=(0.0, 0.0, 0.0),
        lateral_threshold=0.001,
        required_depth=0.035,
        stable_steps=25,
        max_steps=500,
    )
    env = DynamicVerticalDualPandaEnv(cfg, seed=seed)
    env.reset(seed)
    impedance = np.array([0.50, 0.65, 0.65, 0.85, 0.90, 0.90, internal_scale])
    trace: list[dict] = []
    info: dict = {}
    for step in range(cfg.max_steps):
        target = env.control_target_pose()
        translation = (target[:3] - env.desired_pose[:3]) / cfg.action_translation_limit
        rotation = _rotvec(
            _quat_mul(_quat_conj(env.desired_pose[3:]), target[3:])
        ) / cfg.action_rotation_limit_rad
        action = np.clip(np.r_[translation, rotation], -1.0, 1.0)
        _, reward, done, info = env.step(action, impedance)
        object_pos, object_quat = env._workpiece_pose()
        lateral, depth = env.peg_errors()
        trace.append({
            "step": step,
            "phase": info.get("phase", env.phase),
            "stage": info.get("stage", env.stage.name),
            "success": int(env.success),
            "grasped": int(env.grasped),
            "grasp_failure": int(info.get("grasp_failure", False)),
            "weld_count": int(np.sum(env.data.eq_active)),
            "internal_force_N": float(info.get("internal_force_N", 0.0)),
            "grasp_capacity_N": float(info.get("grasp_capacity_N", 0.0)),
            "grasp_load_N": float(info.get("grasp_load_N", 0.0)),
            "grasp_margin_N": float(info.get("grasp_margin_N", 0.0)),
            "reward": float(reward),
            "peg1_lateral_m": float(lateral[0]),
            "peg2_lateral_m": float(lateral[1]),
            "peg1_depth_m": float(depth[0]),
            "peg2_depth_m": float(depth[1]),
            "object_pos": object_pos.tolist(),
            "object_quat": object_quat.tolist(),
            "receiver_pos": env.data.xpos[env.receiver_body].copy().tolist(),
            "receiver_quat": env.data.xquat[env.receiver_body].copy().tolist(),
            "left_q": env.data.qpos[env.arm_qpos[0]].copy().tolist(),
            "right_q": env.data.qpos[env.arm_qpos[1]].copy().tolist(),
            "impedance_action": impedance.tolist(),
        })
        if done:
            break
    result = {
        "success": env.success,
        "grasp_failure": env.grasp_failed,
        "phase": env.phase,
        "steps": len(trace),
        "internal_scale": internal_scale,
        "internal_force_N": cfg.internal_force_min
        + internal_scale * (cfg.internal_force_max - cfg.internal_force_min),
    }
    return trace, result


def write_trace(path: Path, trace: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)


def replay_state(state: dict, left: int, right: int, workpiece: int, receiver_parts) -> None:
    for body, values in ((left, state["left_q"]), (right, state["right_q"])):
        for joint, value in enumerate(values):
            p.resetJointState(body, joint, value)
    p.resetBasePositionAndOrientation(
        workpiece, state["object_pos"], xyzw(state["object_quat"]),
    )
    set_receiver_pose(
        receiver_parts, state["receiver_pos"], xyzw(state["receiver_quat"]),
    )


def camera_image() -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll([0, .15, .25], 1.25, 38, -18, 0, 2)
    projection = p.computeProjectionMatrixFOV(42, PANEL_WIDTH / PANEL_HEIGHT, .03, 3)
    _, _, rgba, _, _ = p.getCameraImage(
        PANEL_WIDTH,
        PANEL_HEIGHT,
        view,
        projection,
        renderer=p.ER_TINY_RENDERER,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(PANEL_HEIGHT, PANEL_WIDTH, 4)[:, :, :3][:, :, ::-1].copy()


def annotate(panel: np.ndarray, state: dict, title: str, force_color: tuple[int, int, int]) -> None:
    failure = bool(state["grasp_failure"])
    success = bool(state["success"])
    status = "SUCCESS" if success else "GRASP SLIP" if failure else state["phase"]
    status_color = (90, 225, 120) if success else (80, 90, 245) if failure else (100, 205, 245)
    cv2.rectangle(panel, (12, 12), (PANEL_WIDTH - 12, 142), (14, 17, 22), -1)
    cv2.putText(panel, title, (26, 40), cv2.FONT_HERSHEY_SIMPLEX, .62, (240, 243, 246), 1, cv2.LINE_AA)
    cv2.putText(panel, status, (26, 70), cv2.FONT_HERSHEY_SIMPLEX, .68, status_color, 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"Finternal {state['internal_force_N']:5.1f} N/side   welds {state['weld_count']}/2",
        (26, 99), cv2.FONT_HERSHEY_SIMPLEX, .51, force_color, 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"capacity {state['grasp_capacity_N']:5.1f} N   load {state['grasp_load_N']:5.1f} N   margin {state['grasp_margin_N']:+6.1f} N",
        (26, 126), cv2.FONT_HERSHEY_SIMPLEX, .43, (205, 212, 220), 1, cv2.LINE_AA,
    )
    center_x, arrow_y = PANEL_WIDTH // 2, PANEL_HEIGHT - 42
    length = int(45 + min(85, state["internal_force_N"] * 1.5))
    cv2.arrowedLine(panel, (center_x - length, arrow_y), (center_x - 10, arrow_y), force_color, 4, tipLength=.18)
    cv2.arrowedLine(panel, (center_x + length, arrow_y), (center_x + 10, arrow_y), force_color, 4, tipLength=.18)


def render() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success_trace, success_result = rollout(0.65)
    slip_trace, slip_result = rollout(0.0)
    if not success_result["success"]:
        raise RuntimeError(f"default internal-force rollout did not succeed: {success_result}")
    if not slip_result["grasp_failure"]:
        raise RuntimeError(f"low internal-force rollout did not slip: {slip_result}")
    write_trace(OUTPUT_DIR / "internal_force_success_trace.csv", success_trace)
    write_trace(OUTPUT_DIR / "internal_force_slip_trace.csv", slip_trace)

    client, left, right, workpiece, receiver_parts = build_scene()
    writer = cv2.VideoWriter(
        str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {VIDEO}")
    for frame in range(MOTION_FRAMES + HOLD_FRAMES):
        progress = min(frame, MOTION_FRAMES - 1) / (MOTION_FRAMES - 1)
        success_index = int(round(progress * (len(success_trace) - 1)))
        slip_index = min(success_index, len(slip_trace) - 1)
        success_state = success_trace[success_index]
        slip_state = slip_trace[slip_index]

        replay_state(success_state, left, right, workpiece, receiver_parts)
        success_panel = camera_image()
        annotate(success_panel, success_state, "SAFE ADAPTIVE INTERNAL FORCE", (90, 215, 155))
        replay_state(slip_state, left, right, workpiece, receiver_parts)
        slip_panel = camera_image()
        annotate(slip_panel, slip_state, "LOW INTERNAL FORCE", (90, 165, 245))

        canvas = np.full((HEIGHT, WIDTH, 3), (24, 27, 32), dtype=np.uint8)
        canvas[108:108 + PANEL_HEIGHT, 12:12 + PANEL_WIDTH] = success_panel
        canvas[108:108 + PANEL_HEIGHT, 648:648 + PANEL_WIDTH] = slip_panel
        cv2.putText(canvas, "AGENT 2 INTERNAL-FORCE CONTROL | CURRENT 7D MUJOCO ROLLOUT REPLAY", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, .77, (238, 242, 246), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"same seed 20260824 | aligned controller step {success_state['step']} | opposite arrows show zero-net squeeze", (28, 75), cv2.FONT_HERSHEY_SIMPLEX, .53, (184, 193, 205), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Safe clamp: 8-50 N/side | slip rule: negative margin for 5 consecutive control steps", (28, 675), cv2.FONT_HERSHEY_SIMPLEX, .55, (207, 214, 222), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    p.disconnect(client)
    print({"video": str(VIDEO), "success": success_result, "slip": slip_result})
    return 0


if __name__ == "__main__":
    raise SystemExit(render())
