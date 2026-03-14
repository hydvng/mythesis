"""
Chapter 4 完整仿真评估脚本
===========================
功能：
1. 加载已训练的 V4SAC 模型（best_model.pt 或 final_model.pt）
2. 在 30s 正弦轨迹 + 海浪扰动场景下运行确定性仿真
3. 绘制并保存以下图表：
   - 轨迹跟踪对比图 (q vs q_des)
   - 跟踪误差曲线
   - 控制输入分解 (总输入 / 模型补偿 / RL输出)
   - STESO 扰动估计 vs 实际扰动
   - 漏斗安全边界 + 误差 + 硬切换区域标记
   - RMS 误差汇总条形图
4. 将仿真数据保存为 NPZ

用法：
    python simulate_chapter4.py                     # 默认 best_model.pt
    python simulate_chapter4.py --model final        # final_model.pt
    python simulate_chapter4.py --model /abs/path/to/model.pt
    python simulate_chapter4.py --scenario extreme   # 极端海况 Hs=4.0
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path
from typing import Dict

# ── 路径注入 ──────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent
CH4_DIR = THIS_DIR.parent
SIM_DIR = CH4_DIR.parent

for p in [
    str(SIM_DIR / "common"),           # wave_disturbance, platform_dynamics
    str(SIM_DIR / "disturbance"),      # load_mss_rao etc.
    str(CH4_DIR / "env"),              # rl_env_chapter4, steso_observer, eso_controller
    str(SIM_DIR / "chapter3" / "agents"),  # v4_sac
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from rl_env_chapter4 import PlatformRLEnvChapter4
from v4_sac import V4SAC

# ── 常量 ──────────────────────────────────────────────────────────────────────
MODEL_DIR = THIS_DIR / "chapter4_models_v2"   # 新课程训练输出目录
MODEL_DIR_LEGACY = THIS_DIR / "chapter4_models"  # 旧模型备用
FIGURE_DIR = CH4_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DOF_LABELS = ["z (m)", r"$\alpha$ (rad)", r"$\beta$ (rad)"]
DOF_TAGS   = ["z", "alpha", "beta"]
U_LABELS   = [r"$F_z$ (N)", r"$\tau_\alpha$ (N·m)", r"$\tau_\beta$ (N·m)"]
REGION_COLORS = {"center": "#2196F3", "boundary": "#FF9800", "unsafe": "#F44336"}

SCENARIOS = {
    "normal": dict(Hs=2.0, T1=8.0, label="Normal Sea (Hs=2.0 m)"),
    "moderate": dict(Hs=3.0, T1=9.0, label="Moderate Sea (Hs=3.0 m)"),
    "extreme": dict(Hs=4.0, T1=10.0, label="Extreme Sea (Hs=4.0 m)"),
}


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def build_agent(state_dim: int, action_dim: int, device: str) -> V4SAC:
    return V4SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.995,
        tau=0.002,
        device=device,
    )


def build_env(scenario: dict, dt: float = 0.01, episode_length: float = 30.0) -> PlatformRLEnvChapter4:
    return PlatformRLEnvChapter4(
        use_model_compensation=True,
        use_eso=True,
        use_steso=True,
        use_hard_switch=True,
        dt=dt,
        max_episode_steps=int(episode_length / dt),
        Hs=scenario["Hs"],
        T1=scenario["T1"],
        q_des_type="sinusoidal",
        diverge_threshold=0.9,
        # STESO 参数
        steso_lambda1=4.0,
        steso_beta1=12.0,
        steso_beta2=30.0,
        # HSC 参数
        switch_threshold=0.6,
        switch_beta=0.5,
    )


def resolve_model_path(arg: str) -> Path:
    if arg == "best":
        # 优先新目录，不存在则回退旧目录
        p = MODEL_DIR / "best_model.pt"
        if not p.exists():
            p = MODEL_DIR_LEGACY / "best_model.pt"
        return p
    if arg == "final":
        p = MODEL_DIR / "final_model.pt"
        if not p.exists():
            p = MODEL_DIR_LEGACY / "final_model.pt"
        return p
    p = Path(arg)
    if p.exists():
        return p
    raise FileNotFoundError(f"找不到模型文件: {arg}")


def region_to_int(region_list) -> np.ndarray:
    """把 region 字符串列表转成整数 0/1/2 便于着色"""
    mapping = {"center": 0, "boundary": 1, "unsafe": 2}
    return np.array([mapping.get(r, 0) for r in region_list], dtype=int)


# ── 主评估函数 ────────────────────────────────────────────────────────────────

def run_simulation(model_path: Path, scenario: dict, dt: float = 0.01, episode_length: float = 30.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    env = build_env(scenario, dt=dt, episode_length=episode_length)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = build_agent(state_dim, action_dim, device)
    assert model_path.exists(), f"模型文件不存在: {model_path}"
    agent.load(str(model_path))
    agent.actor.eval()
    print(f"  已加载模型: {model_path}")

    # ── 运行仿真 ──
    state = env.reset()
    total_reward = 0.0
    max_steps = int(episode_length / dt)

    for _ in range(max_steps):
        with torch.no_grad():
            action = agent.select_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    hist = env.get_history()
    n = len(hist["time"])
    print(f"  完成: steps={n}, total_reward={total_reward:.4f}")

    # ── 整理数组 ──
    time  = np.array(hist["time"])
    q     = np.array(hist["q"])
    qd    = np.array(hist["qd"])
    q_des = np.array(hist["q_des"])
    u     = np.array(hist["u"])
    v_rl  = np.array(hist["v_RL"])
    rewards = np.array(hist["reward"])
    d_steso  = np.array(hist["steso_disturbance"])     # 力域
    d_actual = np.array(hist["actual_disturbance"])    # 实际波浪扰动（力域）
    regions  = hist["region"]

    error = q - q_des
    rms_per_dof = np.sqrt(np.mean(error**2, axis=0))
    rms_total   = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))

    print(f"  RMS 误差: z={rms_per_dof[0]:.5f}, α={rms_per_dof[1]:.5f}, β={rms_per_dof[2]:.5f}")
    print(f"  总 RMS={rms_total:.6f}")

    return {
        "time": time, "q": q, "qd": qd, "q_des": q_des,
        "u": u, "v_rl": v_rl, "rewards": rewards,
        "d_steso": d_steso, "d_actual": d_actual,
        "regions": regions,
        "error": error,
        "rms_per_dof": rms_per_dof,
        "rms_total": rms_total,
        "total_reward": total_reward,
        "scenario": scenario,
    }


# ── 绘图函数 ──────────────────────────────────────────────────────────────────

def fig_tracking(data: dict, tag: str) -> None:
    """图1：轨迹跟踪"""
    t, q, q_des = data["time"], data["q"], data["q_des"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Chapter 4: Trajectory Tracking  |  {data['scenario']['label']}\n"
        f"Total RMS = {data['rms_total']:.5f}",
        fontsize=13, fontweight="bold",
    )
    for i, ax in enumerate(axes):
        ax.plot(t, q[:, i], color="#1565C0", linewidth=1.4, label="Actual")
        ax.plot(t, q_des[:, i], color="#C62828", linewidth=1.1,
                linestyle="--", label="Desired")
        ax.set_ylabel(DOF_LABELS[i], fontsize=10)
        rms = data["rms_per_dof"][i]
        ax.set_title(f"{DOF_TAGS[i]}  |  RMS={rms:.5f}", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    out = FIGURE_DIR / f"ch4_tracking_{tag}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  ✓ 保存: {out.name}")


def fig_error_with_funnel(data: dict, tag: str) -> None:
    """图2：误差 + 漏斗边界 + 区域着色"""
    t, error, regions = data["time"], data["error"], data["regions"]
    region_int = region_to_int(regions)

    # 重建漏斗曲线（与环境参数一致）
    rho_0, rho_inf, kappa = 2.0, 0.20, 0.25
    rho = (rho_0 - rho_inf) * np.exp(-kappa * t) + rho_inf

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Chapter 4: Tracking Error + Performance Funnel  |  {data['scenario']['label']}",
        fontsize=13, fontweight="bold",
    )

    for i, ax in enumerate(axes):
        # 区域背景色
        prev_r, seg_start = region_int[0], 0
        for k in range(1, len(t)):
            if region_int[k] != prev_r or k == len(t) - 1:
                color = ["#E3F2FD", "#FFF3E0", "#FFEBEE"][prev_r]
                ax.axvspan(t[seg_start], t[k], alpha=0.35, color=color, linewidth=0)
                seg_start = k
                prev_r = region_int[k]

        ax.plot(t, error[:, i], color="#1A237E", linewidth=1.2, label="Error")
        ax.plot(t, rho,  color="#B71C1C", linewidth=1.0, linestyle="--", label=r"$\rho(t)$")
        ax.plot(t, -rho, color="#B71C1C", linewidth=1.0, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel(f"$e_{{{DOF_TAGS[i]}}}$", fontsize=10)
        ax.grid(True, alpha=0.25)
        if i == 0:
            patch_c = mpatches.Patch(color="#E3F2FD", label="RL Center")
            patch_b = mpatches.Patch(color="#FFF3E0", label="HSC Boundary")
            ax.legend(handles=[ax.lines[0], ax.lines[1], patch_c, patch_b],
                      fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    out = FIGURE_DIR / f"ch4_error_funnel_{tag}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  ✓ 保存: {out.name}")


def fig_control_decompose(data: dict, tag: str) -> None:
    """图3：控制输入分解"""
    t, u, v_rl = data["time"], data["u"], data["v_rl"]
    tau_model = u - v_rl  # 模型补偿部分

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Chapter 4: Control Input Decomposition  |  {data['scenario']['label']}",
        fontsize=13, fontweight="bold",
    )
    for i, ax in enumerate(axes):
        ax.plot(t, u[:, i],          color="#1B5E20", linewidth=1.2, label="Total $u$")
        ax.plot(t, tau_model[:, i],  color="#F57F17", linewidth=1.0,
                linestyle="--", label="Model comp.")
        ax.plot(t, v_rl[:, i],       color="#880E4F", linewidth=0.9,
                linestyle=":",       label="RL $v_{RL}$")
        ax.set_ylabel(U_LABELS[i], fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    out = FIGURE_DIR / f"ch4_control_{tag}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  ✓ 保存: {out.name}")


def fig_disturbance_estimation(data: dict, tag: str) -> None:
    """图4：STESO 扰动估计 vs 实际扰动"""
    t, d_steso, d_actual = data["time"], data["d_steso"], data["d_actual"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Chapter 4: STESO Disturbance Estimation  |  {data['scenario']['label']}",
        fontsize=13, fontweight="bold",
    )
    for i, ax in enumerate(axes):
        ax.plot(t, d_actual[:, i], color="#B71C1C", linewidth=1.1,
                alpha=0.85, label="Actual $d_{ext}$")
        ax.plot(t, d_steso[:, i],  color="#0D47A1", linewidth=1.2,
                linestyle="--", label="STESO $\\hat{F}_{lumped}$")
        ax.set_ylabel(U_LABELS[i], fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        # 误差阴影
        ax.fill_between(t, d_actual[:, i], d_steso[:, i],
                        alpha=0.15, color="gray", label="Est. error")
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    out = FIGURE_DIR / f"ch4_steso_est_{tag}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  ✓ 保存: {out.name}")


def fig_rms_summary(results_dict: Dict[str, dict]) -> None:
    """图5：多场景 RMS 误差汇总条形图"""
    scenario_labels = list(results_dict.keys())
    n_sc = len(scenario_labels)
    dof_colors = ["#1565C0", "#2E7D32", "#6A1B9A"]

    fig, ax = plt.subplots(figsize=(max(8, 3 * n_sc), 5))
    x = np.arange(n_sc)
    width = 0.22

    for i, (dof, color) in enumerate(zip(DOF_TAGS, dof_colors)):
        vals = [results_dict[sc]["rms_per_dof"][i] for sc in scenario_labels]
        ax.bar(x + i * width, vals, width, label=dof, color=color, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([results_dict[sc]["scenario"]["label"] for sc in scenario_labels],
                       fontsize=9)
    ax.set_ylabel("RMS Tracking Error", fontsize=11)
    ax.set_title("Chapter 4: RMS Error Summary by DOF and Scenario", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURE_DIR / "ch4_rms_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  ✓ 保存: {out.name}")


def fig_region_pie(data: dict, tag: str) -> None:
    """图6：硬切换区域时间占比饼图"""
    region_int = region_to_int(data["regions"])
    counts = [np.sum(region_int == i) for i in range(3)]
    total = sum(counts)
    labels = [f"RL Center\n({counts[0]/total*100:.1f}%)",
              f"HSC Boundary\n({counts[1]/total*100:.1f}%)",
              f"Unsafe\n({counts[2]/total*100:.1f}%)"]
    colors = ["#2196F3", "#FF9800", "#F44336"]
    nonzero = [(l, c, v) for l, c, v in zip(labels, colors, counts) if v > 0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([v for _, _, v in nonzero],
           labels=[l for l, _, _ in nonzero],
           colors=[c for _, c, _ in nonzero],
           autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title(f"Hard-Switch Region Distribution\n{data['scenario']['label']}", fontsize=11)
    fig.tight_layout()
    out = FIGURE_DIR / f"ch4_region_pie_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ 保存: {out.name}")


def save_npz(data: dict, tag: str) -> None:
    out = FIGURE_DIR / f"ch4_sim_data_{tag}.npz"
    np.savez_compressed(
        str(out),
        time=data["time"],
        q=data["q"],
        qd=data["qd"],
        q_des=data["q_des"],
        u=data["u"],
        v_rl=data["v_rl"],
        d_steso=data["d_steso"],
        d_actual=data["d_actual"],
        region=np.array(data["regions"], dtype=object),
        rewards=data["rewards"],
        rms_per_dof=data["rms_per_dof"],
        rms_total=np.array(data["rms_total"]),
    )
    print(f"  ✓ NPZ 已保存: {out.name}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Chapter 4 仿真评估")
    p.add_argument("--model", default="best",
                   help="模型路径或 'best'/'final'，默认 best_model.pt")
    p.add_argument("--scenario", nargs="+", default=["normal"],
                   choices=list(SCENARIOS.keys()) + ["all"],
                   help="仿真场景，可多选，'all' 代表全部")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--episode_length", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()

    scenarios_to_run = list(SCENARIOS.keys()) if "all" in args.scenario else args.scenario
    model_path = resolve_model_path(args.model)

    print("=" * 65)
    print("  Chapter 4 仿真评估")
    print(f"  模型: {model_path}")
    print(f"  场景: {scenarios_to_run}")
    print(f"  dt={args.dt}s, 时长={args.episode_length}s")
    print("=" * 65)

    all_results: Dict[str, dict] = {}

    for sc_name in scenarios_to_run:
        scenario = SCENARIOS[sc_name]
        print(f"\n▶ 场景: {scenario['label']}")
        data = run_simulation(model_path, scenario,
                              dt=args.dt, episode_length=args.episode_length)
        all_results[sc_name] = data

        tag = sc_name
        print(f"  生成图表...")
        fig_tracking(data, tag)
        fig_error_with_funnel(data, tag)
        fig_control_decompose(data, tag)
        fig_disturbance_estimation(data, tag)
        fig_region_pie(data, tag)
        save_npz(data, tag)

    # 汇总对比图（多场景时才有意义）
    if len(all_results) > 1:
        print("\n▶ 生成多场景汇总图...")
        fig_rms_summary(all_results)

    # 打印汇总表格
    print("\n" + "=" * 55)
    print(f"{'场景':<20} {'RMS_z':>10} {'RMS_α':>10} {'RMS_β':>10} {'Total':>10}")
    print("-" * 55)
    for sc_name, data in all_results.items():
        r = data["rms_per_dof"]
        print(f"{sc_name:<20} {r[0]:>10.5f} {r[1]:>10.5f} {r[2]:>10.5f} {data['rms_total']:>10.6f}")
    print("=" * 55)
    print(f"\n所有图表保存在: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
