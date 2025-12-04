from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from obp.dataset import OpenBanditDataset


def seed_everything(seed: int) -> np.random.Generator:
    """设置随机种子，返回 numpy 随机生成器。"""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def load_action_probabilities(
    behavior_policy: str,
    campaign: str,
    top_k: int,
    min_count: int,
    data_path: Optional[Path] = None,
) -> Dict[str, float]:
    """从 OBD 读取点击率，返回臂到点击率的映射。

    data_path 为空则使用内置小样本；设置为完整数据目录可跑全量。
    """
    dataset = OpenBanditDataset(behavior_policy=behavior_policy, campaign=campaign, data_path=data_path)
    df = dataset.data[["item_id", "click"]].copy()
    stats = (
        df.groupby("item_id")["click"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "reward_prob"})
        .sort_values("count", ascending=False)
    )
    stats = stats[stats["count"] >= min_count]
    if stats.empty:
        raise ValueError("样本数过滤后没有可用臂，请降低 min_count。")
    stats = stats.head(top_k)
    return {f"a{int(action)}": float(prob) for action, prob in stats["reward_prob"].items()}
