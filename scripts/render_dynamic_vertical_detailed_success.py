#!/usr/bin/env python3
"""Create a detailed, time-resampled replay of one actual successful rollout."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

import run_dynamic_vertical_formal_study as formal_study
from render_dynamic_vertical_formal_rollout import build_scene, set_receiver_pose, xyzw
from run_dynamic_vertical_formal_study import (
    A2POTrainConfig,
    A2POTrainer,
    DynamicVerticalDualPandaEnv,
    evaluation_config,
    load_base_config,
    load_weights,
    rollout_episode,
)
from x_bimanual.panda_dual_assembly import _quat_conj, _quat_from_rotvec, _quat_mul, _rotvec

WIDTH, HEIGHT, FPS = 1280, 720, 30
PHASE_COLORS = {
    "APPROACH": (185, 185, 185),
    "GRASP": (90, 190, 245),
    "LIFT": (90, 210, 170),
    "TRANSPORT": (110, 190, 235),
    "COARSE ALIGNMENT": (80, 180, 245),
    "COMPLIANT ALIGNMENT": (115, 215, 130),
    "INSERTION": (80, 215, 150),
    "SUCCESS": (90, 225, 120),
}


class BalloonReceiverEnv(DynamicVerticalDualPandaEnv):
    """Dynamic receiver variant with slow buoyancy-like translation and sway."""

    def _receiver_disturbance(self) -> None:
        t = self.step_count * self.cfg.timestep * self.cfg.control_interval
        nominal_pos = np.array([0.0, 0.12, 0.28])
        nominal_quat = np.array([0.70710678, 0.70710678, 0.0, 0.0])
        pose, velocity = self._receiver_state()
        # Larger, low-frequency motion makes the target read as suspended in
        # air while remaining smooth enough for the compliant controller.
        sway = np.array([
            0.014 * np.sin(0.55 * t),
            0.011 * np.cos(0.70 * t),
            0.009 * np.sin(0.43 * t + 0.4),
        ])
        sway_rotation = np.array([
            0.040 * np.sin(0.42 * t),
            0.030 * np.cos(0.58 * t),
            0.026 * np.sin(0.34 * t),
        ])
        desired_quat = _quat_mul(nominal_quat, _quat_from_rotvec(sway_rotation))
        orientation_error = _rotvec(_quat_mul(_quat_conj(desired_quat), pose[3:]))
        self.data.xfrc_applied[self.receiver_body, :3] = (
            -5000.0 * (pose[:3] - nominal_pos - sway) - 260.0 * velocity[:3]
        )
        self.data.xfrc_applied[self.receiver_body, 3:] = (
            -5.0 * orientation_error - 2.0 * velocity[3:]
        )


def first_index(trace, stage: str, fallback: int) -> int:
    return next((index for index, row in enumerate(trace) if row["stage"] == stage), fallback)


def detailed_schedule(trace, speed: float = 2.2):
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    grasp = first_index(trace, "LIFT", max(1, len(trace) // 12))
    compliant = first_index(trace, "COMPLIANT_ALIGNMENT", max(grasp + 4, int(.82 * len(trace))))
    insertion = first_index(trace, "INSERTION", max(compliant + 1, int(.86 * len(trace))))
    # The verified controller advances LIFT/TRANSPORT/COARSE_ALIGNMENT in one
    # step each. Subdivide the subsequent real approach motion for readable
    # semantic playback while retaining the raw controller stage on screen.
    motion_start = min(grasp + 3, compliant - 3)
    span = max(6, compliant - motion_start)
    lift_end = motion_start + int(.12 * span)
    transport_end = motion_start + int(.48 * span)
    coarse_end = motion_start + int(.82 * span)
    segments = (
        ("APPROACH", 0, max(0, grasp - 10), 75),
        ("GRASP", max(0, grasp - 9), motion_start, 45),
        ("LIFT", motion_start, lift_end, 60),
        ("TRANSPORT", lift_end + 1, transport_end, 90),
        ("COARSE ALIGNMENT", transport_end + 1, coarse_end, 90),
        ("COMPLIANT ALIGNMENT", coarse_end + 1, max(coarse_end + 1, insertion - 1), 90),
        ("INSERTION", insertion, len(trace) - 1, 120),
        ("SUCCESS", len(trace) - 1, len(trace) - 1, 45),
    )
    schedule = []
    for phase, start, stop, base_frames in segments:
        # Keep every semantic phase visible while shortening the overall
        # playback.  The source trace remains untouched; this only changes
        # the time-resampled presentation.
        frames = max(2, int(round(base_frames / speed)))
        indices = np.rint(np.linspace(start, max(start, stop), frames)).astype(int)
        schedule.extend((phase, int(np.clip(index, 0, len(trace) - 1))) for index in indices)
    return schedule


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def scaled_receiver_pose(state: dict, float_scale: float) -> tuple[list[float], list[float], float]:
    """Exaggerate the receiver's measured free-body sway for visual clarity."""
    nominal_pos = np.array([0.0, 0.12, 0.28], dtype=float)
    nominal_quat = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=float)
    position = nominal_pos + float_scale * (np.asarray(state["receiver_pos"], dtype=float) - nominal_pos)

    quat = np.asarray(state["receiver_quat"], dtype=float)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    relative = quat_mul_wxyz(np.array([nominal_quat[0], -nominal_quat[1], -nominal_quat[2], -nominal_quat[3]]), quat)
    if relative[0] < 0.0:
        relative = -relative
    angle = 2.0 * math.acos(float(np.clip(relative[0], -1.0, 1.0)))
    sine = math.sin(angle / 2.0)
    if abs(sine) < 1e-8:
        rotation_vector = np.zeros(3)
    else:
        rotation_vector = relative[1:] / sine * angle
    half = 0.5 * float_scale * rotation_vector
    scaled_relative = np.r_[math.cos(np.linalg.norm(half)), np.zeros(3)]
    half_norm = float(np.linalg.norm(half))
    if half_norm > 1e-10:
        scaled_relative[1:] = math.sin(half_norm) * half / half_norm
    scaled_quat = quat_mul_wxyz(nominal_quat, scaled_relative)
    scaled_quat /= max(float(np.linalg.norm(scaled_quat)), 1e-12)
    sway_mm = float(np.linalg.norm(position - nominal_pos) * 1000.0)
    return position.tolist(), scaled_quat.tolist(), sway_mm


def camera_image(width, height, target, distance, yaw, pitch):
    view = p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0, 2)
    projection = p.computeProjectionMatrixFOV(42, width / height, .03, 3)
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, view, projection, renderer=p.ER_TINY_RENDERER,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3][:, :, ::-1].copy()


def draw_timeline(image, active_phase):
    labels = ("APPROACH", "GRASP", "LIFT", "TRANSPORT", "COARSE ALIGNMENT", "COMPLIANT ALIGNMENT", "INSERTION")
    left, right, y0, y1 = 25, WIDTH - 25, HEIGHT - 58, HEIGHT - 24
    widths = np.asarray([1.0, .8, .8, 1.0, 1.25, 1.45, 1.0])
    edges = left + np.r_[0, np.cumsum(widths / widths.sum() * (right - left))].astype(int)
    for index, label in enumerate(labels):
        active = label == active_phase or (active_phase == "SUCCESS" and label == "INSERTION")
        color = PHASE_COLORS[label] if active else (65, 68, 73)
        cv2.rectangle(image, (edges[index] + 2, y0), (edges[index + 1] - 2, y1), color, -1)
        text_size = .34 if len(label) > 12 else .39
        cv2.putText(image, label, (edges[index] + 8, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, text_size, (18, 20, 24) if active else (220, 222, 225), 1, cv2.LINE_AA)


def render(args: argparse.Namespace) -> int:
    if args.balloon:
        # rollout_episode resolves the environment class through its module
        # globals, so switch both references for this process-local render.
        formal_study.DynamicVerticalDualPandaEnv = BalloonReceiverEnv
        env_class = BalloonReceiverEnv
    else:
        env_class = DynamicVerticalDualPandaEnv
    cfg = evaluation_config(load_base_config(), "hard")
    trainer = A2POTrainer(
        env_class(cfg, seed=args.seed),
        A2POTrainConfig(seed=args.train_seed, device="cpu"),
        args.output.parent,
    )
    load_weights(trainer, args.checkpoint)
    trainer.agent1.eval(); trainer.agent2.eval()
    result, trace = rollout_episode(trainer, cfg, args.seed, "a2po", capture_trace=True)
    if not result["success"]:
        raise RuntimeError(f"seed {args.seed} is not a successful A2PO rollout: {result}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix(".csv")
    with trace_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader(); writer.writerows(trace)

    client, left, right, workpiece, receiver_parts = build_scene()
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    schedule = detailed_schedule(trace, args.speed)
    float_scale = args.float_scale if args.float_scale is not None else (1.0 if args.balloon else 3.0)
    for frame_number, (phase, trace_index) in enumerate(schedule):
        state = trace[trace_index]
        for body, values in ((left, state["left_q"]), (right, state["right_q"])):
            for joint, value in enumerate(values):
                p.resetJointState(body, joint, value)
        p.resetBasePositionAndOrientation(workpiece, state["object_pos"], xyzw(state["object_quat"]))
        receiver_pos, receiver_quat, sway_mm = scaled_receiver_pose(state, float_scale)
        set_receiver_pose(receiver_parts, receiver_pos, xyzw(receiver_quat))

        image = camera_image(WIDTH, HEIGHT, [0, .15, .25], 1.25, 38, -18)
        closeup = camera_image(420, 260, [0, .20, .28], .62, 8, -4)
        image[62:322, WIDTH-442:WIDTH-22] = closeup
        cv2.rectangle(image, (WIDTH-444, 60), (WIDTH-20, 324), (235, 238, 242), 2)
        cv2.rectangle(image, (WIDTH-442, 62), (WIDTH-22, 94), (14, 17, 22), -1)
        cv2.putText(image, "RECEIVER / DUAL-PEG CLOSE-UP", (WIDTH-426, 84), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 238, 242), 1, cv2.LINE_AA)

        cv2.rectangle(image, (18, 18), (760, 242), (14, 17, 22), -1)
        color = PHASE_COLORS[phase]
        title = "A2PO HARD SUCCESS | BALLOON-FLOAT FLEXIBLE REPLAY" if args.balloon else "ACTUAL A2PO HARD SUCCESS | FAST FLEXIBLE REPLAY"
        cv2.putText(image, title, (34, 48), cv2.FONT_HERSHEY_SIMPLEX, .62, (238, 242, 246), 1, cv2.LINE_AA)
        cv2.putText(image, f"VISUAL PHASE: {phase}", (34, 81), cv2.FONT_HERSHEY_SIMPLEX, .72, color, 2, cv2.LINE_AA)
        cv2.putText(image, f"raw controller stage: {state['stage']}  |  rollout step: {state['step']}", (34, 111), cv2.FONT_HERSHEY_SIMPLEX, .48, (198, 204, 212), 1, cv2.LINE_AA)
        lateral = max(state["peg1_lateral_error_m"], state["peg2_lateral_error_m"]) * 1000
        depth = min(state["peg1_depth_m"], state["peg2_depth_m"]) * 1000
        cv2.putText(image, f"max lateral {lateral:6.2f} mm   min depth {depth:6.2f} mm", (34, 140), cv2.FONT_HERSHEY_SIMPLEX, .52, (120, 215, 245), 1, cv2.LINE_AA)
        imp = state.get("impedance_action", [0.0] * 7)
        internal_force = 8.0 + 42.0 * float(imp[6])
        contact_stage = state["stage"] in {"FIRST_CONTACT", "COMPLIANT_ALIGNMENT", "INSERTION", "SUCCESS"}
        contact_label = "CONTACT DETECTED" if contact_stage else "NO CONTACT"
        contact_color = (110, 235, 150) if contact_stage else (170, 180, 190)
        cv2.putText(image, f"orientation {state['orientation_error_deg']:5.2f} deg   solver force {state['force_N']:6.3f} N", (34, 166), cv2.FONT_HERSHEY_SIMPLEX, .52, (120, 215, 245), 1, cv2.LINE_AA)
        cv2.putText(image, f"{contact_label}   RL internal-force command {internal_force:5.1f} N/side", (34, 192), cv2.FONT_HERSHEY_SIMPLEX, .50, contact_color, 1, cv2.LINE_AA)
        cv2.putText(image, f"receiver float {sway_mm:4.1f} mm (visual x{float_scale:g})   playback x{args.speed:g}", (34, 218), cv2.FONT_HERSHEY_SIMPLEX, .48, (210, 185, 120), 1, cv2.LINE_AA)
        draw_timeline(image, phase)
        writer.write(image)
    writer.release(); p.disconnect(client)
    print({"video": str(args.output), "trace": str(trace_path), "frames": len(schedule), "result": result})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=2.2, help="playback speed multiplier")
    parser.add_argument("--float-scale", type=float, default=None, help="visual receiver sway multiplier")
    parser.add_argument("--balloon", action="store_true", help="use the larger low-frequency balloon-like receiver dynamics")
    return render(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
