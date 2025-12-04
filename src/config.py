from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """单次实验的基础配置。"""

    behavior_policy: str
    campaign: str
    top_k: int
    min_count: int
    n_rounds: int
    epsilon: float
    exploit_p: float
    gamma: float
    ucb1_exploration_coef: float
    kl_ucb_c: float
    kl_ucb_max_iter: int
    kl_ucb_tol: float
    policy_klucb_prior_success: float
    policy_klucb_prior_failure: float
    policy_klucb_c: float
    policy_klucb_max_iter: int
    policy_klucb_tol: float
    thompson_alpha0: float
    thompson_beta0: float
    seed: int


@dataclass(frozen=True)
class FullConfig:
    """全量实验的全局配置。"""

    run_full: bool
    top_k: int
    min_count: int
    n_rounds: int
    data_path: Path
    behaviors: Tuple[str, ...]
    campaigns: Tuple[str, ...]


# 内置默认值（当配置文件缺失或字段缺省时使用）
_DEFAULT_BASE = ExperimentConfig(
    behavior_policy="random",
    campaign="all",
    top_k=8,
    min_count=50,
    n_rounds=1000,
    epsilon=0.1,
    exploit_p=0.7,
    gamma=0.07,
    ucb1_exploration_coef=2.0,
    kl_ucb_c=3.0,
    kl_ucb_max_iter=25,
    kl_ucb_tol=1e-6,
    policy_klucb_prior_success=1.0,
    policy_klucb_prior_failure=1.0,
    policy_klucb_c=3.0,
    policy_klucb_max_iter=25,
    policy_klucb_tol=1e-6,
    thompson_alpha0=1.0,
    thompson_beta0=1.0,
    seed=17,
)
_DEFAULT_FULL = FullConfig(
    run_full=False,
    top_k=_DEFAULT_BASE.top_k,
    min_count=_DEFAULT_BASE.min_count,
    n_rounds=_DEFAULT_BASE.n_rounds,
    data_path=Path("data"),
    behaviors=("random", "bts"),
    campaigns=("all", "men", "women"),
)


def load_config(path: Path = Path("./config.toml")) -> tuple[ExperimentConfig, FullConfig]:
    """从 TOML 加载配置；不存在则使用默认值。"""
    if not path.exists():
        return _DEFAULT_BASE, _DEFAULT_FULL

    with path.open("rb") as f:
        cfg = tomllib.load(f)

    base_cfg = cfg.get("base", {})
    full_cfg = cfg.get("full", {})

    base = ExperimentConfig(
        behavior_policy=base_cfg.get("behavior_policy", _DEFAULT_BASE.behavior_policy),
        campaign=base_cfg.get("campaign", _DEFAULT_BASE.campaign),
        top_k=int(base_cfg.get("top_k", _DEFAULT_BASE.top_k)),
        min_count=int(base_cfg.get("min_count", _DEFAULT_BASE.min_count)),
        n_rounds=int(base_cfg.get("n_rounds", _DEFAULT_BASE.n_rounds)),
        epsilon=float(base_cfg.get("epsilon", _DEFAULT_BASE.epsilon)),
        exploit_p=float(base_cfg.get("exploit_p", _DEFAULT_BASE.exploit_p)),
        gamma=float(base_cfg.get("gamma", _DEFAULT_BASE.gamma)),
        ucb1_exploration_coef=float(
            base_cfg.get("ucb1_exploration_coef", _DEFAULT_BASE.ucb1_exploration_coef)
        ),
        kl_ucb_c=float(base_cfg.get("kl_ucb_c", _DEFAULT_BASE.kl_ucb_c)),
        kl_ucb_max_iter=int(base_cfg.get("kl_ucb_max_iter", _DEFAULT_BASE.kl_ucb_max_iter)),
        kl_ucb_tol=float(base_cfg.get("kl_ucb_tol", _DEFAULT_BASE.kl_ucb_tol)),
        policy_klucb_prior_success=float(
            base_cfg.get("policy_klucb_prior_success", _DEFAULT_BASE.policy_klucb_prior_success)
        ),
        policy_klucb_prior_failure=float(
            base_cfg.get("policy_klucb_prior_failure", _DEFAULT_BASE.policy_klucb_prior_failure)
        ),
        policy_klucb_c=float(base_cfg.get("policy_klucb_c", _DEFAULT_BASE.policy_klucb_c)),
        policy_klucb_max_iter=int(base_cfg.get("policy_klucb_max_iter", _DEFAULT_BASE.policy_klucb_max_iter)),
        policy_klucb_tol=float(base_cfg.get("policy_klucb_tol", _DEFAULT_BASE.policy_klucb_tol)),
        thompson_alpha0=float(base_cfg.get("thompson_alpha0", _DEFAULT_BASE.thompson_alpha0)),
        thompson_beta0=float(base_cfg.get("thompson_beta0", _DEFAULT_BASE.thompson_beta0)),
        seed=int(base_cfg.get("seed", _DEFAULT_BASE.seed)),
    )

    # full 中的重复参数默认继承 base，除非显式覆盖
    data_path = Path(full_cfg.get("data_path", _DEFAULT_FULL.data_path))
    behaviors = tuple(full_cfg.get("behaviors", list(_DEFAULT_FULL.behaviors)))
    campaigns = tuple(full_cfg.get("campaigns", list(_DEFAULT_FULL.campaigns)))

    full = FullConfig(
        run_full=bool(full_cfg.get("run_full", _DEFAULT_FULL.run_full)),
        top_k=int(full_cfg.get("top_k", base.top_k)),
        min_count=int(full_cfg.get("min_count", base.min_count)),
        n_rounds=int(full_cfg.get("n_rounds", base.n_rounds)),
        data_path=data_path,
        behaviors=behaviors,
        campaigns=campaigns,
    )

    return base, full
