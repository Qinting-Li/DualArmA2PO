# X: 双臂轴孔对接柔顺控制

本项目用于完成“双 Franka/Panda + 双 peg/双 hole 协作装配 + 轨迹 agent + 变阻抗 agent”的仿真研究。当前主环境使用本地 Panda URDF 网格转换出的 MuJoCo 双 7-DoF 模型；旧的 floating-target 环境保留为已验证基线，但不再作为主机械臂装配环境。

## 当前状态

- 控制核心不依赖 Isaac，可在本机直接运行和测试；已包含双臂 Jacobian 转置力矩映射、阻尼零空间姿态控制和逐关节力矩限幅。
- Isaac 场景包含两台 Franka、一根直径 40 mm 的圆柱轴、四块碰撞孔壁和接触传感器。
- `src/x_bimanual/panda_dual_assembly.py` 使用本地 `franka_panda/panda.urdf` 的视觉/碰撞网格，构造两台真实 Panda 7-DoF 链、双夹爪、一个双 peg 刚性工件和一个双孔物理接收器。两台机械臂通过两个 weld 始终共同持有同一个工件。
- 动态竖直主环境的控制链为 Agent 1 对象级 6 维位姿增量 → 对象期望位姿 → Agent 2 的 6 维分组 K/D 加 1 维夹持内力 → 两臂 live Jacobian/OSC → MuJoCo torque dynamics；RL agent 不直接写入物体位姿。
- 新环境明确记录 initialization、grasp、lift、transport、coarse alignment、approach、first contact、compliant alignment、insertion、success，并要求两根 peg 同时满足 lateral、depth、orientation 和稳定保持条件。
- `scripts/verify_dual_peg_assembly.py` 是 RL 训练前的脚本基线，逐控制步输出 Agent 1/2 动作、K/D、接触 wrench、双 peg 误差、阶段、奖励和 success，并生成阻抗变化图。
- `scripts/evaluate_floating_6dof_insertion.py` 支持 `fixed`、`floating_zero_velocity` 和 `floating_random_velocity` 三个模式，按实时相对位姿/速度、接触力矩和插入深度控制，并输出每 episode/汇总 CSV、验证 JSON 和五张关键图。
- 动态竖直接口环境已修复运行时 weld `relpose` 布局、无碰撞抓取初始化、有界六维接收器悬置，以及双腕相容的 Cartesian impedance。严格配置的脚本 demo 已达到双 Peg 横向误差 0.888/0.986 mm、深度 36.74/36.59 mm，并稳定保持 25 步成功。
- `scripts/train_a2po_dynamic_vertical.py` 使用 7 维 Agent 2 动作：6 维分组阻抗加 1 维夹持内力，以及动态场景 curriculum。第 7 维映射为每侧 `8-50 N`，默认先验为 `35.3 N/侧`；旧 6 维 checkpoint 与当前策略头不兼容，需要重新训练。
- 动态竖直环境已经接入策略训练、双腕抓持约束、显式内力和低承载裕度滑脱判定。
- 本服务器有 4 张 A100 80GB，计算资源充足。
- 本服务器尚未安装 Isaac Lab；默认 Python 是 3.13，而 Isaac Sim 6.x 需要 Python 3.12。
- 宿主机是 Ubuntu 20.04、glibc 2.31，达不到 Isaac Sim 6.x 完整工作流要求的 Ubuntu 22.04/glibc 2.35。
- 当前 NVIDIA 驱动为 570.172.08；Isaac Lab 当前文档建议 Linux 使用 580.95.05 或更新的生产分支驱动。
- 当前也未安装 Docker 和 NVIDIA Container Toolkit。

## 目录

```text
X/
|-- assets/                  # 后续机器人/工件 URDF、USD 和网格
|-- configs/task.yaml        # 几何、阻抗、安全和训练参数
|-- scripts/
|   |-- check_environment.py
|   |-- dual_arm_scene.py    # Isaac Lab 场景冒烟测试
|   |-- evaluate_floating_6dof_insertion.py # 保留的旧 MuJoCo floating baseline
|   |-- verify_dual_peg_assembly.py # 真实双 Panda 双 peg/双 hole 脚本验收
|   |-- train_a2po.py         # sanity + 双 agent A2PO/PPO 训练
|   |-- render_preview.py    # 使用本地 Panda URDF 离屏渲染预览
|   `-- smoke_controller.py  # 无 Isaac 控制器冒烟测试
|-- src/x_bimanual/
|   |-- controller.py        # 双 agent 合成、变阻抗与固定安全层
|   |-- osc.py               # 双臂末端 wrench 到关节力矩的 OSC 映射
|   |-- panda_dual_assembly.py # 双 Panda 双 peg/双 hole MuJoCo 环境与 A2PO 接口
|   `-- task.py              # 阶段机、终止条件和奖励
`-- tests/                   # 无 Isaac 单元测试
```

## 立即可运行

```bash
cd /home/qinting/X
PYTHONPATH=src python scripts/smoke_controller.py
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/render_preview.py
PYTHONPATH=src python scripts/verify_dual_peg_assembly.py
PYTHONPATH=src python scripts/train_a2po.py --sanity-episodes 100 --formal-episodes 100
PYTHONPATH=src python scripts/train_a2po_dynamic_vertical.py --sanity-episodes 10 --train-episodes 500 --eval-episodes 50
python scripts/check_environment.py
PYTHONPATH=src python scripts/evaluate_floating_6dof_insertion.py --episodes 100
```

环境检查在 Isaac 未安装时返回非零状态，这是预期行为。

MuJoCo 评估结果写入 `results/floating_6dof_per_episode.csv` 和
`results/floating_6dof_summary.csv`，图写入 `figures/`，并在
`results/floating_6dof_verification.json` 中记录 freejoint、零重力、接触
力以及 target 6-DoF 位移/转动是否被观察到。三种模式共用同一个相对状态
控制器；fixed 模式只把 target body 生成成静态 baseline。

双 Panda 验收日志写入 `results/dual_peg_assembly_log.csv`，阻抗参数图和双
peg 深度/横向误差图写入 `figures/dual_peg_assembly/`。旧 floating 视频及其
简化 carrier 渲染脚本已移除，避免把简化模型误认为 Franka 动作。

旧水平场景训练输出写入 `results/a2po_dual_panda/`，包括 sanity/formal episode CSV、
learning curve、success/force/jamming 曲线、逐控制步 impedance trace 和
PyTorch checkpoint。训练器只调用现有环境的 `reset`/`step`，不修改物理模型。

动态竖直训练输出写入 `results/a2po_dual_panda_dynamic_vertical/`。该目录的
checkpoint 只适用于 `DynamicVerticalDualPandaEnv`，不可与旧水平场景互换。

## 准备 Isaac Lab 运行环境

不要在当前 Ubuntu 20.04 宿主机上直接执行 pip 安装。可行路线有两条：

1. 推荐：由管理员将宿主机升级为 Ubuntu 22.04，并把 NVIDIA 驱动升级到 580.95.05 或更新版本。
2. 集群路线：由管理员安装 Docker Engine、Docker Compose 和 NVIDIA Container Toolkit，然后使用 Isaac Lab 官方 GPU 容器。

宿主机升级后，可按官方 Isaac Lab 6.x 路线建立隔离环境。安装包很大，不要装进系统 Python：

```bash
cd /home/qinting/X
uv venv --python 3.12 --seed env_isaaclab
source env_isaaclab/bin/activate
uv pip install --upgrade pip
uv pip install "isaaclab[isaacsim,all]" \
  --overrides "https://raw.githubusercontent.com/isaac-sim/IsaacLab/develop/tools/wheel_builder/uv-overrides.txt" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow
uv pip install -U torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Isaac Lab 发布节奏较快，实际安装时以[官方安装页](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html)中的精确版本和 override 参数为准。容器路线也从该页面的 `Docker and HPC clusters` 章节进入。

## 启动场景

```bash
cd /home/qinting/X
source env_isaaclab/bin/activate
python scripts/dual_arm_scene.py --num_envs 1 --headless --device cuda:0
```

GUI 冒烟测试去掉 `--headless`。首次运行先使用一个环境，确认输出包含 `Scene ready` 和有限的接触力数值。

## 控制约定

动态竖直主环境使用对象级动作：

- 轨迹 agent 输出 6 维笛卡尔位姿增量。
- Agent 2 输出 `[Kparallel, Klateral, Krotation, Dparallel, Dlateral, Drotation, Finternal]`，范围为 `[0, 1]`。
- `Finternal` 映射到每侧 `8-50 N`，形成左右等大反向、净合力为零的夹持 wrench。
- 控制器计算 `wrench = Kp * (pose_error + correction) - Kd * twist`。
- OSC 接收 `2 x 6 x 7` 末端 Jacobian，通过 `J^T wrench` 和阻尼零空间投影生成 `2 x 7` 关节力矩；调用时必须显式传入安全停机状态。
- 固定安全层统一限制总力、总力矩、接触力和双臂同步误差，触发时两臂同时输出零 wrench。

训练顺序固定为：先用固定阻抗训练轨迹策略，再冻结轨迹策略训练阻抗策略，最后联合小步微调。

## 验收顺序

1. 双 Panda MuJoCo 模型和双 peg/双 hole 几何通过脚本验收。
2. 双臂共同 weld 持有一个刚性工件，脚本完成双 peg 同时插入。
3. 验证对象级 Cartesian impedance 和变量 K/D，再接入两个 RL policy。
4. 通过 `A2POCoordinator` 将 Agent 1 action 显式传给 Agent 2。
5. 最后进行 curriculum randomization、多环境训练和真实机器人迁移。
