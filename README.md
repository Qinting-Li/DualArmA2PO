# Dual-Arm Compliant Peg-in-Hole Assembly

This project studies cooperative dual-arm assembly using two Franka Panda robots, a dual-peg/dual-hole task, a trajectory agent, and a variable-impedance agent. The main environment uses MuJoCo models converted from local Panda URDF meshes; the previous floating-target environment is retained only as a validated baseline.

## Key Features

* Two 7-DoF Panda arms jointly hold one rigid dual-peg workpiece.
* Agent 1 outputs a 6-DoF object pose increment.
* Agent 2 controls six grouped stiffness/damping parameters and one internal gripping-force parameter.
* Live Jacobian-based operational-space control maps Cartesian wrenches to joint torques.
* The task includes grasping, lifting, transport, coarse alignment, contact, compliant alignment, insertion, and success verification.
* Success requires both pegs to satisfy lateral-error, insertion-depth, orientation, and stability constraints simultaneously.
* Safety mechanisms include torque limits, force limits, synchronization monitoring, and slip detection.

The scripted baseline achieves lateral errors of **0.888/0.986 mm**, insertion depths of **36.74/36.59 mm**, and stable success for **25 consecutive steps**.

## Project Structure

```text
X/
├── assets/                         # Robot, workpiece, URDF/USD and meshes
├── configs/task.yaml               # Geometry, impedance and training settings
├── scripts/
│   ├── verify_dual_peg_assembly.py
│   ├── train_a2po.py
│   ├── train_a2po_dynamic_vertical.py
│   ├── evaluate_floating_6dof_insertion.py
│   ├── render_preview.py
│   └── smoke_controller.py
├── src/x_bimanual/
│   ├── controller.py               # Agent composition and safety layer
│   ├── osc.py                      # Cartesian wrench-to-torque mapping
│   ├── panda_dual_assembly.py      # Main MuJoCo environment
│   └── task.py                     # Task phases, rewards and termination
└── tests/
```

## Quick Start

```bash
cd /home/qinting/X

PYTHONPATH=src python scripts/smoke_controller.py
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/render_preview.py
PYTHONPATH=src python scripts/verify_dual_peg_assembly.py

PYTHONPATH=src python scripts/train_a2po_dynamic_vertical.py \
  --sanity-episodes 10 \
  --train-episodes 500 \
  --eval-episodes 50
```

## Control Interface

Agent 2 outputs:

```text
[K_parallel, K_lateral, K_rotation,
 D_parallel, D_lateral, D_rotation,
 F_internal]
```

`F_internal` is mapped to **8–50 N per arm**, producing equal and opposite gripping wrenches with zero net object force. The controller computes:

```text
wrench = Kp × (pose_error + correction) − Kd × twist
```

Joint torques are generated using Jacobian-transpose wrench mapping and damped null-space posture control.

Training proceeds in three stages:

1. Train the trajectory policy with fixed impedance.
2. Freeze it and train the impedance policy.
3. Jointly fine-tune both policies with small updates.

## Isaac Lab

The MuJoCo control and training pipeline runs without Isaac Lab. Isaac Lab is optional for large-scale parallel simulation and requires Ubuntu 22.04, Python 3.12, and a compatible NVIDIA driver/container environment.
