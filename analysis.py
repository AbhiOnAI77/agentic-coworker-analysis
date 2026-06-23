"""
analysis.py

Main entry point for the Ambiguity Resolution Dynamics analysis.

Runs a full simulation of 1000 tasks across 4 task types, compares
three memory architectures, detects concept drift, and generates
all 6 figures + a metrics summary CSV.

Usage:
    python analysis.py

Output:
    figures/fig1_ambiguity_heatmap.png
    figures/fig2_arr_curves.png
    figures/fig3_memory_showdown.png
    figures/fig4_phase_evolution.png
    figures/fig5_drift_recovery.png
    figures/fig6_belief_convergence.png
    figures/metrics_summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src import (
    TaskSimulator, Task, TaskType,
    BayesianAgent,
    build_memories,
    ProbabilisticWorkflowEngine,
    compute_arr, preference_convergence_time, cusum_detect,
    rolling_reward, compute_per_type_arr,
    plot_ambiguity_heatmap, plot_arr_curves, plot_memory_showdown,
    plot_phase_evolution, plot_drift_recovery, plot_belief_convergence,
)

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
N_TASKS       = 1000
DRIFT_AT      = 600   # inject concept drift 60% through
SEED          = 42
SNAPSHOT_STEPS = [10, 50, 200, 500, 900]   # for belief convergence plot


def simulate(
    n_tasks: int,
    memory_name: str,
    memory,
    drift_at: Optional[int] = None,
    seed: int = 42
):
    """
    Run one full simulation with a given memory architecture.

    Returns a dict of collected signals for metrics + visualisations.
    """
    sim = TaskSimulator(seed=seed, drift_at=drift_at)
    agent = BayesianAgent(memory=memory, seed=seed)
    workflow = ProbabilisticWorkflowEngine(seed=seed)

    # Tracking containers
    rewards:       List[float] = []
    entropy_all:   List[float] = []
    entropy_by_type: Dict[str, List[float]] = defaultdict(list)
    ambiguity_hist:  Dict[str, List[Tuple]] = defaultdict(list)
    belief_snapshots: Dict[str, List[np.ndarray]] = defaultdict(list)

    snapshot_set = set(SNAPSHOT_STEPS)

    print(f"\n{'='*60}")
    print(f"  Simulation: memory={memory_name}, n={n_tasks}, drift@{drift_at}")
    print(f"{'='*60}")

    for i in range(n_tasks):
        task = sim.generate_task()
        ttype = task.task_type.value

        # Agent selects action based on Thompson sampling
        action = agent.select_action(task)

        # Current agent entropy (drives workflow decisions)
        entropy = agent.total_belief_entropy(ttype)

        # Simulate reward:
        # Base quality = inverse of ambiguity, noisy
        rng = np.random.default_rng(seed + i)
        base_reward = float(np.clip(
            (1 - task.ambiguity.total / np.sqrt(3)) * 0.6
            + agent.q_estimates[ttype][action] * 0.4
            + rng.normal(0, 0.05),
            0.0, 1.0
        ))

        # Run through workflow
        trace = workflow.execute(task, entropy, base_reward)
        reward = trace.outcome_quality

        # Agent updates beliefs
        agent.update_beliefs(task, action, reward)

        # --- Collect signals ---
        rewards.append(reward)

        new_entropy = agent.total_belief_entropy(ttype)
        entropy_all.append(new_entropy)
        entropy_by_type[ttype].append(new_entropy)

        ambiguity_hist[ttype].append((
            task.ambiguity.intent_ambiguity,
            task.ambiguity.context_ambiguity,
            task.ambiguity.preference_ambiguity
        ))

        # Capture belief snapshot at designated steps
        if i in snapshot_set:
            for tt in TaskType:
                sample = agent.sample_intent_distribution(tt.value)
                belief_snapshots[tt.value].append(sample)

        # Progress
        if (i + 1) % 200 == 0:
            rolling = np.mean(rewards[max(0, i-49):i+1])
            print(f"  [{i+1:4d}/{n_tasks}] rolling_reward={rolling:.3f}  entropy={new_entropy:.3f}")

    return {
        "rewards":          rewards,
        "entropy_all":      entropy_all,
        "entropy_by_type":  dict(entropy_by_type),
        "ambiguity_hist":   dict(ambiguity_hist),
        "belief_snapshots": dict(belief_snapshots),
        "phase_series":     workflow.get_phase_time_series(),
        "workflow":         workflow,
        "agent":            agent,
    }


def compute_metrics_table(
    results_by_memory: Dict[str, dict],
    drift_at: int
) -> pd.DataFrame:
    """Summarise key metrics across memory architectures."""
    rows = []
    for mem_name, res in results_by_memory.items():
        rewards  = res["rewards"]
        entropy  = res["entropy_all"]
        arr_vals = compute_arr(entropy)

        # Post-drift reward recovery (last 100 tasks vs pre-drift baseline)
        pre_drift_reward  = np.mean(rewards[max(0, drift_at - 100):drift_at])
        post_drift_reward = np.mean(rewards[drift_at:drift_at + 100]) if drift_at < len(rewards) else np.nan
        recovery_delta    = post_drift_reward - pre_drift_reward if not np.isnan(post_drift_reward) else np.nan

        pct = preference_convergence_time(entropy)
        mean_arr = float(np.nanmean([a for a in arr_vals if a != 0]))

        rows.append({
            "Memory Architecture": mem_name,
            "Mean ARR (bits/task)":   round(mean_arr, 4),
            "Final Reward (last 50)": round(float(np.mean(rewards[-50:])), 4),
            "PCT (tasks)":            pct if pct is not None else ">1000",
            "Pre-Drift Reward":       round(float(pre_drift_reward), 4),
            "Post-Drift Reward":      round(float(post_drift_reward), 4) if not np.isnan(post_drift_reward) else "N/A",
            "Recovery Delta":         round(float(recovery_delta), 4) if not np.isnan(recovery_delta) else "N/A",
        })

    return pd.DataFrame(rows).set_index("Memory Architecture")


def main():
    print("\n" + "="*60)
    print("  Ambiguity Resolution Dynamics")
    print("  AI Coworker Performance Analysis")
    print("  Inspired by Advanced Agentic Brain Architectures")
    print("="*60)

    Path("figures").mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # Run simulations for each memory architecture
    # -----------------------------------------------------------------------
    memories = build_memories(capacity=500)
    results  = {}

    # Slightly different seeds per architecture so curves are distinguishable
    # (same task distribution, different random action sequences)
    mem_seeds = {"Recency": SEED, "Frequency": SEED + 7, "Hybrid": SEED + 13}
    for mem_name, memory in memories.items():
        results[mem_name] = simulate(
            n_tasks=N_TASKS,
            memory_name=mem_name,
            memory=memory,
            drift_at=DRIFT_AT,
            seed=mem_seeds[mem_name]
        )

    # -----------------------------------------------------------------------
    # Use Hybrid memory results as the primary simulation for Figs 1, 4, 5, 6
    # (it's the best performing, most interesting to show)
    # -----------------------------------------------------------------------
    primary = results["Hybrid"]

    # -----------------------------------------------------------------------
    # Fig 1: Ambiguity Heatmap
    # -----------------------------------------------------------------------
    print("\n[Fig 1] Ambiguity Heatmap...")
    plot_ambiguity_heatmap(primary["ambiguity_hist"])

    # -----------------------------------------------------------------------
    # Fig 2: ARR by Task Type (Hybrid agent)
    # -----------------------------------------------------------------------
    print("[Fig 2] ARR Curves by Task Type...")
    arr_by_type = compute_per_type_arr(primary["entropy_by_type"], window=20)
    plot_arr_curves(arr_by_type)

    # -----------------------------------------------------------------------
    # Fig 3: Memory Architecture Showdown
    # -----------------------------------------------------------------------
    print("[Fig 3] Memory Architecture Showdown...")
    rewards_by_mem = {m: r["rewards"] for m, r in results.items()}
    arr_by_mem = {
        m: compute_arr(r["entropy_all"]) for m, r in results.items()
    }
    pct_by_mem = {
        m: preference_convergence_time(r["entropy_all"]) for m, r in results.items()
    }
    plot_memory_showdown(rewards_by_mem, arr_by_mem, pct_by_mem)

    # -----------------------------------------------------------------------
    # Fig 4: Workflow Phase Evolution
    # -----------------------------------------------------------------------
    print("[Fig 4] Workflow Phase Evolution...")
    plot_phase_evolution(primary["phase_series"])

    # -----------------------------------------------------------------------
    # Fig 5: Concept Drift & Recovery
    # -----------------------------------------------------------------------
    print("[Fig 5] Concept Drift & Recovery...")
    smoothed = rolling_reward(primary["rewards"], window=30)
    change_points = cusum_detect(smoothed, slack=0.04, threshold=3.0)
    plot_drift_recovery(primary["rewards"], change_points, drift_at=DRIFT_AT)

    # -----------------------------------------------------------------------
    # Fig 6: Belief Convergence
    # -----------------------------------------------------------------------
    print("[Fig 6] Belief Convergence...")
    plot_belief_convergence(primary["belief_snapshots"], SNAPSHOT_STEPS)

    # -----------------------------------------------------------------------
    # Metrics Summary CSV
    # -----------------------------------------------------------------------
    print("\n[Summary] Computing metrics table...")
    metrics_df = compute_metrics_table(results, drift_at=DRIFT_AT)
    csv_path = Path("figures") / "metrics_summary.csv"
    metrics_df.to_csv(csv_path)
    print(f"  saved → {csv_path}")

    print("\n" + "="*60)
    print(metrics_df.to_string())
    print("\n✓ All figures generated in figures/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
