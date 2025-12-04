from pathlib import Path
from typing import Optional,Union
from numpy import isin
import pandas as pd

from bandit_policies import build_manual_policies, build_pybandits_bandits
from config import load_config
from data_utils import load_action_probabilities, seed_everything
from simulation import save_outputs, simulate_manual, simulate_pybandits


def run_experiment(config_path:Union[str,Path]) -> pd.DataFrame:
    if isinstance(config_path,str):
        config_path = Path(config_path)
    base, full = load_config(config_path)

    def _build_jobs() -> list[tuple[str, str, int, int, int, Path | None]]:
        # run_full 时自动扫描 data 目录下的全部行为策略/子 campaign，确保“所有数据”都会跑到
        if not full.run_full:
            return [(base.behavior_policy, base.campaign, base.top_k, base.min_count, base.n_rounds, None)]

        data_root = full.data_path
        if not data_root.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_root}")

        # 仅遍历配置里声明的 behaviors × campaigns 组合，避免无意跑 men/women 等额外子集
        jobs = []
        for behavior in full.behaviors:
            behavior_dir = data_root / behavior
            if not behavior_dir.is_dir():
                continue
            for campaign in full.campaigns:
                campaign_dir = behavior_dir / campaign
                if not campaign_dir.is_dir():
                    continue
                jobs.append(
                    (
                        behavior,
                        campaign,
                        full.top_k,
                        full.min_count,
                        full.n_rounds,
                        data_root,
                    )
                )

        if not jobs:
            raise ValueError(f"在 {data_root} 下没有发现任何行为策略/数据子目录")
        return jobs

    jobs = _build_jobs()

    all_results = []
    for behavior_policy, campaign, top_k, min_count, n_rounds, data_path in jobs:
        rng = seed_everything(base.seed)
        action_probs = load_action_probabilities(
            behavior_policy=behavior_policy,
            campaign=campaign,
            top_k=top_k,
            min_count=min_count,
            data_path=data_path,
        )

        pybandits_runs = []
        for offset, pb_policy in enumerate(build_pybandits_bandits(action_ids=action_probs.keys(), config=base)):
            pybandits_runs.append(
                simulate_pybandits(
                    name=f"{pb_policy.name}_{behavior_policy}_{campaign}",
                    mab=pb_policy.mab,
                    action_probs=action_probs,
                    n_rounds=n_rounds,
                    seed=base.seed + offset,
                )
            )

        manual_policies = build_manual_policies(action_ids=action_probs.keys(), rng=rng, config=base)
        manual_runs = simulate_manual(
            policies=manual_policies, action_probs=action_probs, n_rounds=n_rounds, rng=rng
        )
        manual_runs["policy"] = manual_runs["policy"] + f"_{behavior_policy}_{campaign}"

        job_df = pd.concat(pybandits_runs + [manual_runs], ignore_index=True)
        all_results.append((job_df, f"{behavior_policy}_{campaign}_"))

    results_dir = Path("results")
    merged = []
    for job_df, prefix in all_results:
        save_outputs(job_df, results_dir=results_dir, file_prefix=prefix)
        merged.append(job_df)

    return pd.concat(merged, ignore_index=True)


if __name__ == "__main__":
    results = run_experiment(r"C:\Users\80163\Desktop\在线机器学习\+课程project的代码和描述\experiment\src\config.toml")
