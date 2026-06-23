"""
visualizations.py

Six publication-quality figures for the Ambiguity Resolution Dynamics analysis.

Fig 1: Ambiguity heatmap over time (3 dimensions x N interactions)
Fig 2: ARR curves by task type
Fig 3: Memory architecture showdown (Recency vs Frequency vs Hybrid)
Fig 4: Workflow phase evolution (stacked area)
Fig 5: Concept drift detection + recovery
Fig 6: Posterior belief convergence (multi-panel)

Style choices:
  - Dark background with vibrant accent colours (looks great in GitHub READMEs)
  - Consistent colour palette across all figures
  - Tight layouts, no excessive whitespace
  - Every figure saved at 150dpi (good balance of quality vs file size)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# -----------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------
plt.style.use("dark_background")

PALETTE = {
    "Scheduling":    "#4FC3F7",   # sky blue
    "CRM Update":    "#81C784",   # green
    "Email Triage":  "#FFB74D",   # amber
    "Operational":   "#CE93D8",   # purple
    "Recency":       "#FF6B6B",   # coral
    "Frequency":     "#4ECDC4",   # teal
    "Hybrid":        "#FFE66D",   # yellow
    "accent":        "#405EF2",   # Brand blue
    "drift":         "#FF4081",   # hot pink for drift markers
}

PHASE_COLOURS = {
    "INTAKE":  "#37474F",
    "CLARIFY": "#FF8A65",
    "EXECUTE": "#4FC3F7",
    "VERIFY":  "#81C784",
    "DELIVER": "#CE93D8",
}

FIG_DIR = Path("figures")
DPI = 150


def _save(fig: plt.Figure, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  saved → {path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 1: Ambiguity Heatmap
# -----------------------------------------------------------------------
def plot_ambiguity_heatmap(
    ambiguity_history: Dict[str, List[Tuple[float, float, float]]],
    smooth_window: int = 20
):
    """
    3-row heatmap: one row per ambiguity dimension, columns = interactions.
    Separate panel per task type.
    """
    task_types = list(ambiguity_history.keys())
    n_types = len(task_types)
    dims = ["Intent", "Context", "Preference"]

    fig, axes = plt.subplots(
        3, n_types,
        figsize=(4 * n_types, 5),
        facecolor="#0D0D0D"
    )
    fig.suptitle(
        "Ambiguity Collapse Over Time\n"
        r"$\mathit{darker\ =\ lower\ ambiguity\ =\ agent\ has\ learned}$",
        fontsize=13, color="white", y=1.02
    )

    for col, ttype in enumerate(task_types):
        data = np.array(ambiguity_history[ttype])  # (N, 3)
        if len(data) < smooth_window:
            continue

        # Rolling mean smoothing
        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - smooth_window + 1)
            smoothed[i] = data[start:i + 1].mean(axis=0)

        for row, dim in enumerate(dims):
            ax = axes[row, col]
            vals = smoothed[:, row].reshape(1, -1)
            im = ax.imshow(
                vals, aspect="auto", cmap="YlOrRd_r",
                vmin=0, vmax=1, interpolation="gaussian"
            )
            ax.set_yticks([0])
            ax.set_yticklabels([dim], fontsize=8, color="white")
            ax.tick_params(colors="white", labelsize=7)

            if row == 0:
                colour = PALETTE.get(ttype, "white")
                ax.set_title(ttype, fontsize=9, color=colour, pad=4)
            if row == 2:
                n = len(data)
                ax.set_xlabel("Interactions", fontsize=7, color="grey")
                ticks = np.linspace(0, n - 1, 5, dtype=int)
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticks, fontsize=6, color="grey")
            else:
                ax.set_xticks([])

            ax.set_facecolor("#0D0D0D")
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Ambiguity")
    _save(fig, "fig1_ambiguity_heatmap.png")


# -----------------------------------------------------------------------
# Fig 2: ARR Curves by Task Type
# -----------------------------------------------------------------------
def plot_arr_curves(
    arr_by_type: Dict[str, List[float]],
    window: int = 30
):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    for ttype, arr in arr_by_type.items():
        colour = PALETTE.get(ttype, "white")
        # Smooth
        smoothed = []
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(arr[start:i + 1]))
        x = np.arange(len(smoothed))
        ax.plot(x, smoothed, color=colour, lw=2, label=ttype, alpha=0.9)
        ax.fill_between(x, 0, smoothed, color=colour, alpha=0.07)

    ax.axhline(0, color="white", lw=0.5, linestyle="--", alpha=0.3)
    ax.set_xlabel("Task Index", color="grey")
    ax.set_ylabel("ARR (bits / interaction)", color="grey")
    ax.set_title(
        "Ambiguity Resolution Rate by Task Type\n"
        r"$\mathit{higher\ =\ agent\ collapsing\ uncertainty\ faster}$",
        color="white", fontsize=12
    )
    ax.tick_params(colors="grey")
    ax.legend(facecolor="#1A1A1A", edgecolor="none", labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    plt.tight_layout()
    _save(fig, "fig2_arr_curves.png")


# -----------------------------------------------------------------------
# Fig 3: Memory Architecture Showdown
# -----------------------------------------------------------------------
def plot_memory_showdown(
    rewards_by_memory: Dict[str, List[float]],
    arr_by_memory: Dict[str, List[float]],
    pct_by_memory: Dict[str, Optional[int]],
    window: int = 30
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0D0D0D")
    ax1.set_facecolor("#0D0D0D")
    ax2.set_facecolor("#0D0D0D")

    for mem_name in ["Recency", "Frequency", "Hybrid"]:
        colour = PALETTE[mem_name]

        # --- Left: Smoothed rolling reward ---
        rewards = rewards_by_memory.get(mem_name, [])
        smoothed_r = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed_r.append(np.mean(rewards[start:i + 1]))
        ax1.plot(smoothed_r, color=colour, lw=2, label=mem_name)
        ax1.fill_between(range(len(smoothed_r)), 0.4, smoothed_r,
                         color=colour, alpha=0.08)

        # --- Right: ARR over time ---
        arr = arr_by_memory.get(mem_name, [])
        smoothed_a = []
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            smoothed_a.append(np.mean(arr[start:i + 1]))
        ax2.plot(smoothed_a, color=colour, lw=2, label=mem_name)

        # Mark PCT
        pct = pct_by_memory.get(mem_name)
        if pct is not None and pct < len(smoothed_r):
            ax1.axvline(pct, color=colour, lw=1, linestyle=":", alpha=0.6)
            ax1.annotate(
                f"PCT={pct}", xy=(pct, smoothed_r[pct]),
                xytext=(pct + 5, smoothed_r[pct] + 0.03),
                fontsize=7, color=colour,
                arrowprops=dict(arrowstyle="->", color=colour, lw=0.8)
            )

    for ax, title, ylabel in [
        (ax1, "Rolling Task Reward by Memory Architecture", "Reward"),
        (ax2, "ARR by Memory Architecture", "ARR (bits/interaction)")
    ]:
        ax.set_xlabel("Task Index", color="grey")
        ax.set_ylabel(ylabel, color="grey")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="grey")
        ax.legend(facecolor="#1A1A1A", edgecolor="none", labelcolor="white", fontsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    fig.suptitle(
        "Memory Architecture Showdown: Recency vs Frequency vs Hybrid (ACT-R)",
        color="white", fontsize=12, y=1.01
    )
    plt.tight_layout()
    _save(fig, "fig3_memory_showdown.png")


# -----------------------------------------------------------------------
# Fig 4: Workflow Phase Evolution
# -----------------------------------------------------------------------
def plot_phase_evolution(phase_time_series: Dict[str, List[float]]):
    phases = ["INTAKE", "CLARIFY", "EXECUTE", "VERIFY", "DELIVER"]
    available = [p for p in phases if p in phase_time_series and phase_time_series[p]]
    if not available:
        print("  [skip] no phase data")
        return

    n = len(phase_time_series[available[0]])
    x = np.arange(n)
    data = np.array([phase_time_series[p] for p in available])

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    colours = [PHASE_COLOURS.get(p, "white") for p in available]
    ax.stackplot(x, data, labels=available, colors=colours, alpha=0.85)

    ax.set_xlabel("Task Index", color="grey")
    ax.set_ylabel("Fraction of Workflow Time", color="grey")
    ax.set_title(
        "Workflow Phase Evolution Over Time\n"
        r"$\mathit{Clarify\ and\ Verify\ phases\ shrink\ as\ the\ agent\ learns}$",
        color="white", fontsize=12
    )
    ax.tick_params(colors="grey")
    ax.set_ylim(0, 1)
    ax.legend(
        loc="upper right", facecolor="#1A1A1A",
        edgecolor="none", labelcolor="white", fontsize=9
    )
    for spine in ax.spines.values():
        spine.set_color("#333333")

    plt.tight_layout()
    _save(fig, "fig4_phase_evolution.png")


# -----------------------------------------------------------------------
# Fig 5: Concept Drift & Recovery
# -----------------------------------------------------------------------
def plot_drift_recovery(
    rewards: List[float],
    change_points: List[int],
    drift_at: Optional[int],
    window: int = 30
):
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(float(np.mean(rewards[start:i + 1])))

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    x = np.arange(len(smoothed))
    ax.plot(x, smoothed, color=PALETTE["accent"], lw=2, label="Rolling Reward", zorder=3)
    ax.fill_between(x, 0.3, smoothed, color=PALETTE["accent"], alpha=0.1)

    # True drift injection
    if drift_at is not None and drift_at < len(smoothed):
        ax.axvline(drift_at, color=PALETTE["drift"], lw=1.5,
                   linestyle="--", label="Concept Drift Injected", zorder=4)
        ax.annotate(
            "Drift", xy=(drift_at, smoothed[drift_at]),
            xytext=(drift_at + 10, smoothed[drift_at] - 0.06),
            fontsize=9, color=PALETTE["drift"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["drift"], lw=1)
        )

    # CUSUM detections
    for i, cp in enumerate(change_points):
        if cp < len(smoothed):
            ax.axvline(cp, color="#FFE66D", lw=1, linestyle=":",
                       alpha=0.7, label="CUSUM Detection" if i == 0 else "")
            ax.scatter(cp, smoothed[cp], color="#FFE66D", s=40, zorder=5)

    ax.set_xlabel("Task Index", color="grey")
    ax.set_ylabel("Rolling Reward", color="grey")
    ax.set_title(
        "Concept Drift Detection & Recovery\n"
        r"$\mathit{CUSUM\ detects\ the\ shift,\ agent\ recovers\ over\ subsequent\ tasks}$",
        color="white", fontsize=12
    )
    ax.tick_params(colors="grey")
    ax.set_ylim(0.2, 1.05)
    ax.legend(facecolor="#1A1A1A", edgecolor="none", labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    plt.tight_layout()
    _save(fig, "fig5_drift_recovery.png")


# -----------------------------------------------------------------------
# Fig 6: Posterior Belief Convergence
# -----------------------------------------------------------------------
def plot_belief_convergence(
    belief_snapshots: Dict[str, List[np.ndarray]],
    snapshot_steps: List[int]
):
    """
    Multi-panel showing intent distribution at different learning stages.
    Each column = a snapshot in time. Each row = a task type.
    """
    task_types = list(belief_snapshots.keys())
    n_types = len(task_types)
    n_snaps = len(snapshot_steps)
    n_intents = 3
    intent_labels = ["Intent A", "Intent B", "Intent C"]

    fig, axes = plt.subplots(
        n_types, n_snaps,
        figsize=(3 * n_snaps, 2.5 * n_types),
        facecolor="#0D0D0D"
    )
    if n_types == 1:
        axes = axes.reshape(1, -1)
    if n_snaps == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        "Posterior Belief Convergence Over Time\n"
        r"$\mathit{Uniform\ prior\ →\ concentrated\ posterior\ as\ agent\ learns}$",
        color="white", fontsize=12, y=1.01
    )

    x = np.arange(n_intents)
    for row, ttype in enumerate(task_types):
        snapshots = belief_snapshots[ttype]
        colour = PALETTE.get(ttype, "white")

        for col in range(n_snaps):
            ax = axes[row, col]
            ax.set_facecolor("#0D0D0D")

            if col < len(snapshots):
                probs = snapshots[col]
                bars = ax.bar(x, probs, color=colour, alpha=0.8, width=0.6)
                # Shade uniform reference
                ax.axhline(1 / n_intents, color="white", lw=0.8,
                           linestyle="--", alpha=0.4, label="Uniform")
            else:
                ax.set_visible(False)
                continue

            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(intent_labels, fontsize=6, color="grey", rotation=15)
            ax.tick_params(colors="grey", labelsize=6)

            if col == 0:
                ax.set_ylabel(ttype, fontsize=7, color=colour)
            if row == 0:
                step = snapshot_steps[col] if col < len(snapshot_steps) else "?"
                ax.set_title(f"t={step}", fontsize=8, color="white")

            for spine in ax.spines.values():
                spine.set_color("#333333")

    plt.tight_layout()
    _save(fig, "fig6_belief_convergence.png")
