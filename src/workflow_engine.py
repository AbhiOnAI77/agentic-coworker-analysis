"""
workflow_engine.py

POMDP-inspired probabilistic workflow engine.

Mirrors claims of a "probabilistic workflow engine that
determines the next best step at every moment."

In a Partially Observable Markov Decision Process (POMDP), the agent can't
directly observe the environment state — it maintains a *belief state* (a
probability distribution over states) and chooses actions based on that.

Here the "states" are task phases:
  Intake -> Clarify -> Execute -> Verify -> Deliver

The agent starts with a noisy observation of the task (ambiguity vector)
and decides whether to:
  - Ask a clarifying question (move to Clarify)
  - Execute directly (skip Clarify -> Execute)
  - Verify before delivering (add Verify step)
  - Deliver directly (skip Verify)

Over time the agent learns transition probabilities that minimize
wasted steps (unnecessary clarifications, over-verification).
The "Skip Rate" metric tracks how efficiently the agent learns to
bypass steps it no longer needs.
"""

import numpy as np
from enum import IntEnum
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from .task_simulator import Task, TaskType


class Phase(IntEnum):
    """Ordered workflow phases."""
    INTAKE   = 0
    CLARIFY  = 1
    EXECUTE  = 2
    VERIFY   = 3
    DELIVER  = 4


# Thresholds: if agent's uncertainty is below these, it can skip the phase
# These get adapted over time through learned transition matrices
DEFAULT_THRESHOLDS = {
    Phase.CLARIFY: 0.55,  # skip clarification if ambiguity < this
    Phase.VERIFY:  0.30,  # skip verification if ambiguity < this
}


@dataclass
class WorkflowTrace:
    """Records the phases actually executed for one task."""
    task_id: int
    phases_executed: List[Phase]
    total_steps: int
    skipped_clarify: bool
    skipped_verify: bool
    outcome_quality: float   # reward signal [0, 1]

    @property
    def skip_count(self) -> int:
        return int(self.skipped_clarify) + int(self.skipped_verify)


class ProbabilisticWorkflowEngine:
    """
    Decides which workflow phases to execute based on belief state.

    The belief state here = the agent's current ambiguity estimate for a task
    (informed by both the task's raw ambiguity vector and the Bayesian agent's
    current prior quality).

    Over time the engine learns per-task-type threshold adjustments:
    - If skipping Clarify leads to good outcomes -> lower the Clarify threshold
      (agent gets more confident, skips more often)
    - If skipping Verify leads to bad outcomes -> raise the Verify threshold
      (agent becomes more conservative)

    This implements a simple gradient-free policy update — adaptive thresholds
    rather than a full POMDP solver (the latter would be overkill for demo).
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

        # Per-task-type learned thresholds (start at defaults)
        self.thresholds: Dict[str, Dict[Phase, float]] = {
            t.value: dict(DEFAULT_THRESHOLDS) for t in TaskType
        }

        # Running stats for threshold adaptation
        self._skip_outcomes: Dict[str, Dict[Phase, List[float]]] = {
            t.value: {Phase.CLARIFY: [], Phase.VERIFY: []} for t in TaskType
        }

        # History of phase execution patterns (for the stacked area chart)
        self.phase_history: List[Dict[str, float]] = []

    def _belief_ambiguity(
        self,
        task: Task,
        agent_entropy: float,
        max_entropy: float = 3.0
    ) -> float:
        """
        Combine task's raw ambiguity with agent's current belief entropy.
        Higher entropy = agent is less confident = treat task as more ambiguous.
        """
        raw = task.ambiguity.total / np.sqrt(3)  # normalise [0,1]
        normalised_entropy = np.clip(agent_entropy / max_entropy, 0, 1)
        # Weighted combination — raw ambiguity dominates early,
        # agent entropy dominates once there's enough data
        return 0.6 * raw + 0.4 * normalised_entropy

    def execute(
        self,
        task: Task,
        agent_entropy: float,
        base_reward: float
    ) -> WorkflowTrace:
        """
        Run the workflow for a task.

        Args:
            task:          The incoming task
            agent_entropy: Current entropy of the Bayesian agent's beliefs
            base_reward:   Simulated reward before workflow quality multiplier

        Returns:
            WorkflowTrace with full execution record
        """
        ttype = task.task_type.value
        thresholds = self.thresholds[ttype]
        belief = self._belief_ambiguity(task, agent_entropy)

        phases_executed = [Phase.INTAKE]

        # --- Clarify phase ---
        skip_clarify = belief < thresholds[Phase.CLARIFY]
        if not skip_clarify:
            phases_executed.append(Phase.CLARIFY)

        # Execute is always done
        phases_executed.append(Phase.EXECUTE)

        # --- Verify phase ---
        # After execution, we have a better estimate of quality
        # use action outcome uncertainty as a proxy
        post_exec_ambiguity = belief * 0.7  # execution reduces uncertainty
        skip_verify = post_exec_ambiguity < thresholds[Phase.VERIFY]
        if not skip_verify:
            phases_executed.append(Phase.VERIFY)

        phases_executed.append(Phase.DELIVER)

        # Outcome quality: unnecessary steps slightly reduce efficiency
        # but skipping when uncertain reduces quality
        n_unnecessary = int(skip_clarify and belief > 0.4) + \
                         int(skip_verify and post_exec_ambiguity > 0.25)
        quality_penalty = 0.05 * n_unnecessary
        outcome_quality = max(0.0, min(1.0, base_reward - quality_penalty))

        trace = WorkflowTrace(
            task_id=task.task_id,
            phases_executed=phases_executed,
            total_steps=len(phases_executed),
            skipped_clarify=skip_clarify,
            skipped_verify=skip_verify,
            outcome_quality=outcome_quality
        )

        # Update thresholds based on outcome
        self._adapt_thresholds(ttype, trace, belief)

        # Record phase distribution for visualisation
        self._record_phase_distribution(phases_executed)

        return trace

    def _adapt_thresholds(
        self, ttype: str, trace: WorkflowTrace, belief: float
    ):
        """
        Adaptive threshold update — move thresholds in the direction of better outcomes.

        If skipping Clarify -> good outcome: lower threshold (skip more)
        If skipping Clarify -> bad outcome:  raise threshold (be more conservative)
        """
        lr = 0.02  # small step size for stability

        for phase in [Phase.CLARIFY, Phase.VERIFY]:
            was_skipped = (
                trace.skipped_clarify if phase == Phase.CLARIFY
                else trace.skipped_verify
            )
            if was_skipped:
                self._skip_outcomes[ttype][phase].append(trace.outcome_quality)
                if len(self._skip_outcomes[ttype][phase]) >= 5:
                    mean_q = np.mean(self._skip_outcomes[ttype][phase][-5:])
                    # Good outcomes when skipping -> can lower threshold
                    # Bad outcomes when skipping -> raise threshold
                    delta = lr * (mean_q - 0.6)
                    self.thresholds[ttype][phase] = np.clip(
                        self.thresholds[ttype][phase] - delta, 0.1, 0.9
                    )

    def _record_phase_distribution(self, phases: List[Phase]):
        """Record fraction of time in each phase for stacked area chart."""
        total = len(phases)
        dist = {p.name: 0.0 for p in Phase}
        for p in phases:
            dist[p.name] += 1.0 / total
        self.phase_history.append(dist)

    def rolling_skip_rate(self, window: int = 50) -> List[float]:
        """Fraction of tasks where at least one phase was skipped."""
        # Inferred from phase_history: if INTAKE+EXECUTE+DELIVER only = 3 steps, 2 skips
        skip_flags = []
        for dist in self.phase_history:
            has_clarify = dist.get("CLARIFY", 0) > 0
            has_verify = dist.get("VERIFY", 0) > 0
            skip_flags.append(int(not has_clarify or not has_verify))

        rates = []
        for i in range(len(skip_flags)):
            start = max(0, i - window + 1)
            rates.append(float(np.mean(skip_flags[start:i + 1])))
        return rates

    def get_phase_time_series(self) -> Dict[str, List[float]]:
        """Returns smoothed time series per phase for stacked area chart."""
        if not self.phase_history:
            return {}
        result = {p.name: [] for p in Phase}
        window = 30
        for i, dist in enumerate(self.phase_history):
            start = max(0, i - window + 1)
            window_dists = self.phase_history[start:i + 1]
            for p in Phase:
                result[p.name].append(
                    float(np.mean([d.get(p.name, 0.0) for d in window_dists]))
                )
        return result
