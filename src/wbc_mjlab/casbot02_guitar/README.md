# CASBOT02-Guitar locomotion teacher

独立任务：`Casbot02-Guitar-Loco-Teacher-Flat`。

继承**当前** Casbot02 loco 教师的奖励、观测、动作、课程、随机化、
默认姿态和 PPO 配置，替换机器人模型并使用独立日志目录。
按照用户追加要求，取消手臂随膝关节摆动，训练和 sim2sim 均由位置电机
保持普通 Casbot02 的默认手臂目标角度（不是焊死关节，受力后仍可偏移）。
没有增加弹吉他动作；吉他为刚性固定载荷。
原 Casbot02 任务不变。

电机和默认位置在独立的 `motor_config.py` 中配置，初始数值与当前
Casbot02 完全一致，包括 Kp/Kd、力矩上限、armature、摩擦和通信延迟。
腿部 action scale 也从此文件计算。后续可独立修改，不会影响普通模型。
双臂默认位置（弧度）：左侧 `(0.1, 0.175, 0, -0.35, 0)`，
右侧 `(0.1, -0.175, 0, -0.35, 0)`。

## 模型来源与惯性

读取仓库根目录下的
`CASBOT02_ENCOS_7dof_shell_guitar/Serial/urdf/CASBOT02_ENCOS_7dof_shell_20251015.urdf`
和其引用网格。请保留此资源目录。随附 XML 没有更新吉他信息，因此不直接使用。

只导入 URDF 的 `waist_yaw_link` 上身/吉他组合几何与完整惯性矩阵；
其余部分复用现有 locomotion 模型，避免引入多余头部、腰部、腕部或手指自由度。
策略仍为 12 个腿部动作，模型仍有 22 个驱动关节。

- 上身/吉他组合质量：14.22139 kg（已含上身，不重复叠加）。
- 整机质量：50.653931 → 53.620321 kg，增加 2.96639 kg。
- URDF 组合局部质心：`(-0.011182, 0.00019844, 0.18439) m`。
  该局部坐标与普通模型相同，按提供文件保留，并未自行估算载荷质心。
- 默认姿态下整机世界质心约由 `(0.014193, 0.000127, 0.882736)`
  变为 `(0.012789, 0.000131, 0.903531) m`。

碰撞使用原上身碰撞体加 8 个独立吉他 CAD 实体（各自由 MuJoCo 凸包表示），
而非整个“上身＋吉他”单一凸包；后者会填充空隙并造成肩部虚假穿插。
退化的零体积 CAD 碎片不参与碰撞。该拆分针对当前提供的模型，
后续替换网格须重新验证。完整组合网格用于显示，默认站姿已检查无自碰撞。

当前左右镜像训练配置也原样继承。吉他载荷不完全对称，镜像假设并不严格；
此版遵照“不改训练参数”的要求保留，后续可单独评估。

## 训练

```bash
./train_casbot02_guitar_loco_teacher.sh 0
```

日志在 `logs/rsl_rl/casbot02_guitar_loco_teacher/`。默认从头训练。

## 导出与 sim2sim

将以下占位路径替换为实际训练目录和 checkpoint：

```bash
PYTHONPATH=. uv run python scripts/export_casbot02_onnx.py \
  Casbot02-Guitar-Loco-Teacher-Flat \
  logs/rsl_rl/casbot02_guitar_loco_teacher/<run>/model_20000.pt

PYTHONPATH=. uv run python scripts/sim2sim_casbot02_guitar_loco_onnx.py \
  logs/rsl_rl/casbot02_guitar_loco_teacher/<run>/<run>.onnx
```

sim2sim 不传 ONNX 路径时自动搜索吉他任务的独立日志目录，
并使用相同的吉他质量、惯性和碰撞模型。不要使用普通 Casbot02 的
sim2sim 入口验证吉他模型。
