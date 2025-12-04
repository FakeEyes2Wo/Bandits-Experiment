# Bandit experiments on real data

This repo runs multiple multi-armed bandit strategies using `pybandits` on the real Open Bandit Dataset (OBD). The arms are the most frequent items in OBD; their reward probabilities are estimated directly from the logged clicks so the simulation is grounded in real-world feedback.

## Setup
```bash
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e .
```

## Run the experiment
```bash
.venv\Scripts\python pybandit_experiment.py
```
参数集中在 `run_experiment` 内部（行为策略、campaign、rounds、top_k、min_count、epsilon、exploit_p、gamma、seed）。`run_full` 开关设为 True 时会遍历完整 ZOZOTOWN 组合（random/bts × all/men/women），可将 `full_data_path` 指向本地完整 OBD 目录跑全量实验。pybandits 的 `SmabSimulator` 用于 Thompson 族，UCB1/EXP3/随机为手写基线但共用同一真实点击率环境。

## Outputs
- `results/regret_curves.csv`: cumulative reward and regret per round for each policy
- `results/regret_plot.png`: matplotlib plot of the regret curves
- `results/advanced_ope_summary.json`: final cumulative reward/regret and average reward per policy

All scripts download the bundled small OBD subset automatically if it is not cached locally.
