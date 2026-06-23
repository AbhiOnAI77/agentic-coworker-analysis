"""
metrics.py

Novel metrics for measuring AI coworker performance.

Key metrics:
  ARR  (Ambiguity Resolution Rate)     — bits of entropy collapsed per interaction
  PCT  (Preference Convergence Time)   — interactions to reach belief stability
  IGPI (Information Gain Per Interaction) — mutual info between action and outcome
  CUSUM (Cumulative Sum control chart) — change-point detection for concept drift

ARR is the central novel metric of this project.
It doesn't exist in the standard RL / ML evaluation literature.
The intuition: a coworker that learns fast should be collapsing ambiguity
(reducing entropy) efficiently with each new task.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.special import kl_div


def compute_arr(
    entropy_series: List[float],
    window: int = 20
) -> List[float]:
    """
    Ambiguity Resolution Rate (ARR) — bits per interaction.

    Measures the rolling rate of entropy decrease per step.
    Positive ARR = agent is learning (resolving ambiguity).
    Negative ARR = agent is becoming more uncertain (drift / new tasks).

    Args:
        entropy_series: List of entropy values over time (bits)
        window:         Rolling window size

    Returns:
        List of ARR values (same length as entropy_series)
    """
    arr = []
    for i in range(len(entropy_series)):
        if i < 2:
            arr.append(0.0)
            continue
        start = max(0, i - window)
        chunk = entropy_series[start:i + 1]
        if len(chunk) < 2:
            arr.append(0.0)
        else:
            # Negative slope of entropy = positive resolution rate
            # Use linear fit to reduce noise
            x = np.arange(len(chunk), dtype=float)
            slope = np.polyfit(x, chunk, 1)[0]
            arr.append(float(-slope))  # negate: falling entropy = positive ARR
    return arr


def compute_kl_from_uniform(alpha: np.ndarray) -> float:
    """
    KL divergence from the Dirichlet mean to a uniform distribution.
    Measures how far the agent's intent belief has moved from maximum uncertainty.

    Higher value = more confident / more resolved beliefs.
    """
    probs = alpha / alpha.sum()
    k = len(probs)
    uniform = np.ones(k) / k

    # KL(P || Q) = Σ P * log(P/Q)
    probs = np.clip(probs, 1e-10, 1.0)
    return float(np.sum(probs * np.log2(probs / uniform)))


def preference_convergence_time(
    entropy_series: List[float],
    threshold: float = 0.3
) -> Optional[int]:
    """
    Preference Convergence Time (PCT).

    Returns the first step at which entropy drops below `threshold` bits
    and stays below for at least 10 consecutive steps.

    Returns None if convergence is not achieved.
    """
    for i in range(len(entropy_series) - 10):
        if all(e < threshold for e in entropy_series[i:i + 10]):
            return i
    return None


def cusum_detect(
    values: List[float],
    target: Optional[float] = None,
    slack: float = 0.5,
    threshold: float = 4.0
) -> List[int]:
    """
    CUSUM (Cumulative Sum) change-point detection.

    Identifies when a time series undergoes a significant mean shift.
    Used here to detect concept drift: when user preferences suddenly change,
    the agent's reward signal shifts — CUSUM catches this.

    Args:
        values:    Time series of values (e.g., rolling reward)
        target:    Expected mean under null hypothesis (default: mean of first 20% of data)
        slack:     Allowable slack (k in CUSUM literature), typically σ/2
        threshold: Detection threshold (h), triggers alarm when CUSUM > h

    Returns:
        List of step indices where change-points are detected
    """
    if not values:
        return []

    if target is None:
        n_baseline = max(1, len(values) // 5)
        target = float(np.mean(values[:n_baseline]))

    s_pos = 0.0  # Upper CUSUM
    s_neg = 0.0  # Lower CUSUM
    change_points = []

    for i, v in enumerate(values):
        s_pos = max(0, s_pos + (v - target) - slack)
        s_neg = max(0, s_neg - (v - target) - slack)

        if s_pos > threshold or s_neg > threshold:
            change_points.append(i)
            s_pos = 0.0  # reset after detection
            s_neg = 0.0

    return change_points


def rolling_reward(rewards: List[float], window: int = 30) -> List[float]:
    """Smoothed rolling mean of reward signal."""
    result = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(rewards[start:i + 1])))
    return result


def information_gain_per_interaction(
    entropy_series: List[float]
) -> List[float]:
    """
    Information Gain Per Interaction (IGPI).

    Δ entropy between consecutive steps.
    Positive = entropy fell (agent learned something).
    Negative = entropy rose (new uncertainty introduced).
    """
    igpi = [0.0]
    for i in range(1, len(entropy_series)):
        igpi.append(entropy_series[i - 1] - entropy_series[i])
    return igpi


def compute_per_type_arr(
    entropy_by_type: Dict[str, List[float]],
    window: int = 20
) -> Dict[str, List[float]]:
    """Compute ARR separately for each task type."""
    return {
        ttype: compute_arr(series, window=window)
        for ttype, series in entropy_by_type.items()
    }
