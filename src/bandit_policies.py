from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from config import ExperimentConfig
from pybandits.smab import SmabBernoulli, SmabBernoulliBAI


@dataclass
class Policy:
    
    name: str

    def select_action(self) -> str:
        raise NotImplementedError

    def update(self, action: str, reward: int) -> None:
        raise NotImplementedError


class PyBanditsPolicy(Policy):
    """直接使用 pybandits 的 MAB，预测与更新都委托给其内部方法。"""

    def __init__(self, name: str, mab):
        super().__init__(name=name)
        self.mab = mab

    def select_action(self) -> str:
        actions, _ = self.mab.predict(n_samples=1)
        return actions[0]

    def update(self, action: str, reward: int) -> None:
        self.mab.update(actions=[action], rewards=[int(reward)])


class RandomPolicy(Policy):
    """等概率随机选择臂。"""

    def __init__(self, name: str, action_ids: Sequence[str], rng: np.random.Generator):
        super().__init__(name=name)
        self.action_ids = list(action_ids)
        self.rng = rng

    def select_action(self) -> str:
        return self.rng.choice(self.action_ids)

    def update(self, action: str, reward: int) -> None:
        return


class Ucb1Policy(Policy):
    """UCB1 上置信界策略：利用均值+置信半径 sqrt(alpha * log t / n) 保证探索。

    参数：
    - action_ids: 可选臂列表。
    - exploration_coef alpha（可调）：越大则置信区间更宽、探索更积极。
    """

    def __init__(self, name: str, action_ids: Sequence[str], exploration_coef: float = 2.0):
        super().__init__(name=name)
        self.action_ids = list(action_ids)
        self.counts = {a: 0 for a in self.action_ids}
        self.values = {a: 0.0 for a in self.action_ids}
        self.total_pulls = 0
        self.exploration_coef = exploration_coef

    def select_action(self) -> str:
        # 先拉未探索过的臂
        for action, count in self.counts.items():
            if count == 0:
                return action

        total = sum(self.counts.values())
        scores = {
            action: self.values[action]
            + math.sqrt(self.exploration_coef * math.log(total) / self.counts[action])
            for action in self.action_ids
        }
        return max(scores, key=scores.get)

    def update(self, action: str, reward: int) -> None:
        self.total_pulls += 1
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = value + (reward - value) / n


class KLUcbPolicy(Policy):
    """KL-UCB：用 KL(p̂||q) <= (log t + c log log t)/n 求上置信界，更精细刻画伯努利臂。

    参数：
    - action_ids: 可选臂列表。
    - c: 探索系数，越大越保守。
    - max_iter: 二分求解上界的最大步数。
    - tol: 二分停止阈值。
    """

    def __init__(
        self,
        name: str,
        action_ids: Sequence[str],
        c: float = 3.0,
        max_iter: int = 25,
        tol: float = 1e-6,
    ):
        super().__init__(name=name)
        self.action_ids = list(action_ids)
        self.counts = {a: 0 for a in self.action_ids}
        self.successes = {a: 0.0 for a in self.action_ids}
        self.c = c
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _kl_divergence(p: float, q: float) -> float:
        eps = 1e-12
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def _solve_upper_bound(self, mean: float, exploration: float) -> float:
        # 在 [mean, 1] 内二分搜索满足 KL <= exploration 的最大 q
        low, high = mean, 1.0 - 1e-12
        for _ in range(self.max_iter):
            mid = (low + high) / 2
            if self._kl_divergence(mean, mid) > exploration:
                high = mid
            else:
                low = mid
            if high - low < self.tol:
                break
        return (low + high) / 2

    def select_action(self) -> str:
        for action, count in self.counts.items():
            if count == 0:
                return action

        total = sum(self.counts.values())
        scores = {}
        for action in self.action_ids:
            pulls = self.counts[action]
            mean = self.successes[action] / pulls
            exploration = (math.log(total) + self.c * math.log(max(total, 2))) / pulls
            scores[action] = self._solve_upper_bound(mean, exploration)
        return max(scores, key=scores.get)

    def update(self, action: str, reward: int) -> None:
        self.counts[action] += 1
        self.successes[action] += reward


class PolicyKLUcb(Policy):
    """policyKLUCB：在 KL-UCB 上叠加 Beta(α0, β0) 伪计数的先验，体现策略级的经验。

    参数：
    - action_ids: 可选臂列表。
    - prior_success α0 / prior_failure β0: 每个臂的先验成功/失败伪计数。
    - c、max_iter、tol: 与 KL-UCB 相同，控制置信区间搜索精度与探索强度。
    """

    def __init__(
        self,
        name: str,
        action_ids: Sequence[str],
        prior_success: float = 1.0,
        prior_failure: float = 1.0,
        c: float = 3.0,
        max_iter: int = 25,
        tol: float = 1e-6,
    ):
        super().__init__(name=name)
        self.action_ids = list(action_ids)
        self.successes = {a: float(prior_success) for a in self.action_ids}
        self.failures = {a: float(prior_failure) for a in self.action_ids}
        self.c = c
        self.max_iter = max_iter
        self.tol = tol

    def _counts(self, action: str) -> float:
        return self.successes[action] + self.failures[action]

    def _kl_divergence(self, p: float, q: float) -> float:
        eps = 1e-12
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def _solve_upper_bound(self, mean: float, exploration: float) -> float:
        low, high = mean, 1.0 - 1e-12
        for _ in range(self.max_iter):
            mid = (low + high) / 2
            if self._kl_divergence(mean, mid) > exploration:
                high = mid
            else:
                low = mid
            if high - low < self.tol:
                break
        return (low + high) / 2

    def select_action(self) -> str:
        total = sum(self._counts(a) for a in self.action_ids)
        scores = {}
        for action in self.action_ids:
            pulls = self._counts(action)
            mean = self.successes[action] / pulls
            exploration = (math.log(total) + self.c * math.log(max(total, 2))) / pulls
            scores[action] = self._solve_upper_bound(mean, exploration)
        return max(scores, key=scores.get)

    def update(self, action: str, reward: int) -> None:
        if reward:
            self.successes[action] += 1.0
        else:
            self.failures[action] += 1.0


class Exp3Policy(Policy):
    """EXP3 对抗性场景的权重策略：用指数权重维护概率，兼顾探索项 gamma/|A|。

    参数：
    - action_ids: 可选臂列表。
    - gamma: 探索率，越大越接近均匀随机；过小则更贪心。
    - rng: 随机数生成器。
    """

    def __init__(self, name: str, action_ids: Sequence[str], rng: np.random.Generator, gamma: float = 0.07):
        super().__init__(name=name)
        self.action_ids = list(action_ids)
        self.gamma = gamma
        self.weights = {a: 1.0 for a in self.action_ids}
        self.last_probs: Dict[str, float] = {a: 1.0 / len(self.action_ids) for a in self.action_ids}
        self.rng = rng

    def _probabilities(self) -> np.ndarray:
        weight_sum = sum(self.weights.values())
        base = np.array([self.weights[a] / weight_sum for a in self.action_ids])
        probs = (1 - self.gamma) * base + self.gamma / len(self.action_ids)
        self.last_probs = {a: float(p) for a, p in zip(self.action_ids, probs)}
        return probs

    def select_action(self) -> str:
        probs = self._probabilities()
        choice = self.rng.choice(self.action_ids, p=probs)
        return str(choice)

    def update(self, action: str, reward: int) -> None:
        prob = max(self.last_probs.get(action, 1.0 / len(self.action_ids)), 1e-6)
        est_reward = reward / prob
        self.weights[action] *= math.exp(self.gamma * est_reward / len(self.action_ids))


class ThompsonSamplingPolicy(Policy):
    """policyThompson：Beta-Bernoulli 的汤普森采样，用采样的成功率做贪心。

    参数：
    - action_ids: 可选臂列表。
    - alpha0 / beta0: Beta 先验超参数，相当于初始成功/失败伪计数。
    - rng: 随机数生成器。
    """

    def __init__(
        self,
        name: str,
        action_ids: Sequence[str],
        rng: np.random.Generator,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        super().__init__(name=name)
        self.action_ids = list(action_ids)
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alphas = {a: float(alpha0) for a in self.action_ids}
        self.betas = {a: float(beta0) for a in self.action_ids}
        self.rng = rng

    def select_action(self) -> str:
        samples = {a: float(self.rng.beta(self.alphas[a], self.betas[a])) for a in self.action_ids}
        return max(samples, key=samples.get)

    def update(self, action: str, reward: int) -> None:
        if reward:
            self.alphas[action] += 1.0
        else:
            self.betas[action] += 1.0


def build_pybandits_bandits(action_ids: Iterable[str], config: ExperimentConfig) -> List[PyBanditsPolicy]:
    """构造基于 pybandits 的不同 TS 变体。"""
    action_ids = set(action_ids)
    return [
        PyBanditsPolicy(name="thompson_ts", mab=SmabBernoulli(action_ids=action_ids)),
        PyBanditsPolicy(
            name=f"ts_eps{config.epsilon}", mab=SmabBernoulli(action_ids=action_ids, epsilon=config.epsilon)
        ),
        PyBanditsPolicy(
            name=f"bai_exploit_{config.exploit_p}",
            mab=SmabBernoulliBAI(action_ids=action_ids, strategy={"exploit_p": config.exploit_p}),
        ),
    ]


def build_manual_policies(action_ids: Iterable[str], rng: np.random.Generator, config: ExperimentConfig) -> List[Policy]:
    """构造手写基线策略（UCB/KL-UCB/EXP3/Thompson/随机）。

    参数说明集中在各策略的注释里，方便对照实验报告撰写。
    """
    action_ids = list(action_ids)
    return [
        Ucb1Policy(name="ucb1", action_ids=action_ids, exploration_coef=config.ucb1_exploration_coef),
        KLUcbPolicy(
            name="kl_ucb", action_ids=action_ids, c=config.kl_ucb_c, max_iter=config.kl_ucb_max_iter, tol=config.kl_ucb_tol
        ),
        PolicyKLUcb(
            name="policy_klucb",
            action_ids=action_ids,
            prior_success=config.policy_klucb_prior_success,
            prior_failure=config.policy_klucb_prior_failure,
            c=config.policy_klucb_c,
            max_iter=config.policy_klucb_max_iter,
            tol=config.policy_klucb_tol,
        ),
        Exp3Policy(name=f"exp3_gamma_{config.gamma}", action_ids=action_ids, rng=rng, gamma=config.gamma),
        ThompsonSamplingPolicy(
            name="policy_thompson",
            action_ids=action_ids,
            rng=rng,
            alpha0=config.thompson_alpha0,
            beta0=config.thompson_beta0,
        ),
        RandomPolicy(name="random", action_ids=action_ids, rng=rng),
    ]
