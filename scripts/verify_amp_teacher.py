"""Step 0: 还原 Casbot02 AMP 教师,验证能输出 (mean, std) 且 mean 与 ONNX 一致。

在 amp_mjlab conda 环境运行(有 torch + mjlab + onnxruntime):
    conda activate amp_mjlab
    python scripts/verify_amp_teacher.py

这是 KL 蒸馏可行性的 gate:只有确认教师能还原出 (mean, std),蒸馏才有意义。
"""
from __future__ import annotations

import numpy as np
import torch

CKPT = (
    "/home/liuchenxu/Desktop/amp_mjlab/amp_mjlab/logs/rsl_rl/"
    "casbot02_leg_amp_locomotion/2026-08-25_11-28-46/model_8000.pt"
)
ONNX = (
    "/home/liuchenxu/Desktop/amp_mjlab/amp_mjlab/logs/rsl_rl/"
    "casbot02_leg_amp_locomotion/2026-08-25_11-28-46/export/"
    "0825-11Casbot02-Leg-AMP-Flat_model_8000.onnx"
)

ACTOR_HIDDEN = [512, 256, 128]
ACTOR_OUT = 12


def build_actor(mean, std):
    """按 amp_mjlab 的 actor 结构(MLP 512/256/128 -> 12, ELU)重建。"""
    layers = []
    in_d = 180
    for h in ACTOR_HIDDEN:
        layers.append(torch.nn.Linear(in_d, h))
        layers.append(torch.nn.ELU())
        in_d = h
    layers.append(torch.nn.Linear(in_d, ACTOR_OUT))
    mlp = torch.nn.Sequential(*layers)
    return mlp, mean, std


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]
    obs_norm = ckpt["obs_norm_state_dict"]

    # 1) 还原 actor MLP 权重
    mlp, norm_mean, norm_std = build_actor(
        obs_norm["_mean"], obs_norm["_std"]
    )
    state = {k.replace("actor.", ""): v for k, v in msd.items() if k.startswith("actor.")}
    # Sequential 的键是 0,2,4,6(权重), bias 同名;直接 rename actor.0 -> 0 即可
    mlp.load_state_dict(state, strict=True)
    mlp.eval()

    # 2) 教师 std
    std = msd["std"]  # (12,)
    print(f"actor MLP 加载成功, std 形状={tuple(std.shape)}, 值={std.tolist()}")
    print(f"obs 归一化 mean[0,:4]={norm_mean.flatten()[:4].tolist()}")
    print(f"obs 归一化 std[0,:4]={norm_std.flatten()[:4].tolist()}")

    # 3) 随机输入, 还原 mean, 对照 ONNX
    torch.manual_seed(0)
    x = torch.randn(1, 180)
    x_norm = (x - norm_mean) / norm_std
    with torch.no_grad():
        mean_torch = mlp(x_norm)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        mean_onnx = sess.run(None, {input_name: x.numpy().astype(np.float32)})[0]
        err = float(np.max(np.abs(mean_torch.numpy() - mean_onnx)))
        print(f"\nmean(torch) vs mean(onnx): max abs err = {err:.2e}")
        print("✅ 一致" if err < 1e-4 else "❌ 不一致, 检查 obs 归一化/结构")
        print(f"mean_onnx[:6]  = {mean_onnx[0,:6].tolist()}")
        print(f"mean_torch[:6] = {mean_torch[0,:6].tolist()}")
    except Exception as e:  # onnxruntime 可能没装
        print(f"[warn] onnxruntime 不可用, 跳过 ONNX 对照: {e}")
        print(f"mean_torch[:6] = {mean_torch[0,:6].tolist()}")

    print("\n结论: 教师可还原出 (mean, std), KL 蒸馏可行。")


if __name__ == "__main__":
    main()
