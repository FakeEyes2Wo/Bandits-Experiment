from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pybandits.smab_simulator import SmabSimulator

from bandit_policies import Policy


def simulate_manual(policies: List[Policy], action_probs: Dict[str, float], n_rounds: int, rng) -> pd.DataFrame:
    """手写策略的模拟器，按轮生成累计奖励与遗憾。"""
    best_prob = max(action_probs.values())
    rows = []
    for policy in policies:
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        for step in tqdm(range(1, n_rounds + 1), desc=f"manual-{policy.name}", leave=False):
            action = policy.select_action()
            reward_prob = action_probs[action]
            reward = int(rng.random() < reward_prob)
            policy.update(action, reward)
            cumulative_reward += reward
            cumulative_regret += best_prob - reward_prob
            rows.append(
                {
                    "round": step,
                    "policy": policy.name,
                    "cumulative_reward": cumulative_reward,
                    "cumulative_regret": cumulative_regret,
                }
            )
    return pd.DataFrame(rows)


def simulate_pybandits(name: str, mab, action_probs: Dict[str, float], n_rounds: int, seed: int) -> pd.DataFrame:
    """使用 pybandits 的 SmabSimulator 跑 TS 族策略。"""
    simulator = SmabSimulator(
        mab=deepcopy(mab),
        n_updates=n_rounds,
        batch_size=1,
        probs_reward={aid: float(p) for aid, p in action_probs.items()},
        random_seed=seed,
        save=False,
        verbose=False,
        visualize=False,
    )
    _run_smab_with_progress(simulator, desc=f"pybandits-{name}")
    res = simulator.results.copy()
    res["round"] = res["batch"] + 1
    res["policy"] = name
    res["expected_regret"] = res["max_prob_reward"] - res["selected_prob_reward"]
    res["cumulative_reward"] = res["reward"].cumsum()
    res["cumulative_regret"] = res["expected_regret"].cumsum()
    return res[["round", "policy", "cumulative_reward", "cumulative_regret"]]


def plot_regret_curves(df: pd.DataFrame, out_path: Path) -> None:
    """绘制遗憾曲线并保存图片。"""
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby("policy"):
        sorted_group = group.sort_values("round")
        plt.plot(sorted_group["round"], sorted_group["cumulative_regret"], label=name)
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret (expected)")
    plt.title("Regret curves on Open Bandit Dataset arms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """汇总各策略的最终指标。"""
    summary = {}
    for name, group in df.groupby("policy"):
        final = group.sort_values("round").iloc[-1]
        summary[name] = {
            "final_round": int(final["round"]),
            "cumulative_reward": float(final["cumulative_reward"]),
            "cumulative_regret": float(final["cumulative_regret"]),
            "average_reward": float(final["cumulative_reward"]) / float(final["round"]),
        }
    return summary


def save_outputs(regret_df: pd.DataFrame, results_dir: Path, file_prefix: str = "") -> None:
    """保存结果文件：曲线、汇总、图表。"""
    results_dir.mkdir(parents=True, exist_ok=True)
    regret_path = results_dir / f"{file_prefix}regret_curves.csv"
    regret_df.to_csv(regret_path, index=False)

    plot_path = results_dir / f"{file_prefix}regret_plot.png"
    plot_regret_curves(regret_df, plot_path)

    summary = summarize(regret_df)
    summary_path = results_dir / f"{file_prefix}advanced_ope_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved regret curves to {regret_path}")
    print(f"Saved regret plot to {plot_path}")
    print(f"Saved summary to {summary_path}")


def _run_smab_with_progress(simulator: SmabSimulator, desc: str) -> None:
    """仿照 SmabSimulator.run，加入 tqdm 进度条。"""
    for batch_index in tqdm(range(simulator.n_updates), desc=desc, leave=False):
        predict_kwargs, update_kwargs, metadata = simulator._get_batch_step_kwargs_and_metadata(batch_index)
        simulator._step(batch_index, metadata, predict_kwargs, update_kwargs)

    simulator._finalize_results()
    if simulator.verbose:
        simulator._print_results()
    if simulator.visualize:
        simulator._visualize_results()
    if simulator.save:
        simulator._save_results()
