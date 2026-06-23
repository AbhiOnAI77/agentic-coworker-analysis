"""
bayesian_agent.py

The AI coworker modelled as a Bayesian learner.

Mirrors advanced "Agentic Brain" architectures which claim:
  - Mental Models: internal model of org structure and stakeholders
  - Patterns: learns user style preferences (formal/casual, CC rules, etc.)
  - Structures: reusable templates for recurring task types
  - Adaptive Learning: gets better at tasks over time

Implementation:
  - Dirichlet prior over intent distribution (conjugate update on each task)
  - Beta-Binomial for binary preference learning
  - Thompson Sampling for action selection (exploration/exploitation balance)
  - Memory integration: pulls weighted reward signal from BaseMemory

Notes:
  I originally tried a simple UCB agent here but Thompson sampling
  felt more natural — the "sample from beliefs and act" matches how
  a probabilistic workflow engine would actually work.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .task_simulator import Task, TaskType, AmbiguityVector
from .memory import BaseMemory, MemoryEntry


# Number of discrete actions available to the agent per task type.
# E.g. for scheduling: 0=Ask clarifying email, 1=Propose slots, 2=Confirm directly
N_ACTIONS = 3

# Number of intent classes the agent tries to infer
N_INTENTS = 3


@dataclass
class MentalModel:
    """
    Org-level context the agent builds over time.

    Maps to advanced "Mental Models" features:
    'Builds internal models of how your org works, who key stakeholders are,
     and how decisions get made.'

    Stored as a simple dict of context signals and their learned weights.
    In a real system this would be a knowledge graph — here we keep it
    lightweight to focus on the dynamics.
    """
    stakeholder_importance: Dict[str, float] = field(default_factory=dict)
    decision_cadence: float = 1.0      # mean days between decisions
    org_complexity: float = 0.5        # [0,1] how layered is approval process

    def update(self, feedback_signal: float):
        """Nudge org complexity estimate based on how hard tasks are."""
        self.org_complexity = 0.9 * self.org_complexity + 0.1 * feedback_signal


class BayesianAgent:
    """
    Bayesian AI coworker agent with Thompson Sampling action selection.

    Per task type, maintains:
      - Dirichlet prior over intent (α vectors, one per task type)
      - Beta priors over binary preferences (a, b per task type)
      - Memory: a BaseMemory instance (injected at init)

    The agent never sees true_intent or true_preference directly.
    It infers them from feedback signals (reward).
    """

    def __init__(self, memory: BaseMemory, seed: int = 0):
        self.memory = memory
        self.rng = np.random.default_rng(seed)
        self.mental_model = MentalModel()

        # Per task-type Dirichlet concentrations over N_INTENTS intent classes
        # Start uniform (maximum uncertainty)
        self.intent_priors: Dict[str, np.ndarray] = {
            t.value: np.ones(N_INTENTS) for t in TaskType
        }

        # Per task-type Beta prior over continuous preference [0,1]
        # (a, b) — starts at (1,1) i.e. uniform
        self.preference_priors: Dict[str, Tuple[float, float]] = {
            t.value: (1.0, 1.0) for t in TaskType
        }

        # Q-value estimates per (task_type, action) — initialised to 0.5
        self.q_estimates: Dict[str, np.ndarray] = {
            t.value: np.full(N_ACTIONS, 0.5) for t in TaskType
        }

        # Track how many times each action was taken (for diagnostics)
        self.action_counts: Dict[str, np.ndarray] = {
            t.value: np.zeros(N_ACTIONS, dtype=int) for t in TaskType
        }

        self.step = 0

    # ------------------------------------------------------------------
    # Core action selection — Thompson Sampling
    # ------------------------------------------------------------------

    def select_action(self, task: Task) -> int:
        """
        Thompson Sampling:
          1. Sample a Q-value from a Beta distribution for each action
          2. Pick the action with the highest sample

        The Beta parameters encode our uncertainty: high uncertainty -> wide
        distributions -> more exploration. As we observe more, distributions
        narrow -> exploitation dominates.
        """
        ttype = task.task_type.value
        n_taken = self.action_counts[ttype]

        # For each action, build a Beta(α, β) from its mean Q and counts
        actions_q = []
        for a in range(N_ACTIONS):
            # Memory-informed Q estimate
            mem_q = self.memory.get_weighted_reward(ttype, a)
            # Blend with running estimate
            n = max(1, n_taken[a])
            q = 0.7 * self.q_estimates[ttype][a] + 0.3 * mem_q

            # Beta parameters that encode our uncertainty
            # More data -> higher alpha+beta -> tighter distribution
            alpha = max(0.5, q * n)
            beta_p = max(0.5, (1 - q) * n)
            sample = self.rng.beta(alpha, beta_p)
            actions_q.append(sample)

        chosen = int(np.argmax(actions_q))
        self.action_counts[ttype][chosen] += 1
        return chosen

    # ------------------------------------------------------------------
    # Belief updates — after observing a task outcome
    # ------------------------------------------------------------------

    def update_beliefs(self, task: Task, action: int, reward: float):
        """
        Conjugate Bayesian updates after each task outcome.

        intent_prior  : Dirichlet update (add 1 to the inferred intent class)
        preference_prior: Beta update (reward signal as pseudo-observation)
        q_estimates   : Online weighted average
        mental_model  : Soft update from reward signal
        """
        ttype = task.task_type.value

        # --- Intent: infer most likely class from ambiguity + reward ---
        # Heuristic: lower ambiguity + high reward suggests a clear intent
        inferred_intent = int(round(
            (1 - task.ambiguity.intent_ambiguity) * (N_INTENTS - 1) * reward
        ))
        inferred_intent = np.clip(inferred_intent, 0, N_INTENTS - 1)
        self.intent_priors[ttype][inferred_intent] += reward  # soft increment

        # --- Preference: Beta update ---
        # Treat reward as a Bernoulli success (binarised at 0.6 threshold)
        a, b = self.preference_priors[ttype]
        if reward >= 0.6:
            self.preference_priors[ttype] = (a + 1.0, b)
        else:
            self.preference_priors[ttype] = (a, b + 1.0)

        # --- Q-value: online update with harmonic step size ---
        n = self.action_counts[ttype][action]
        lr = 1.0 / n  # decreasing learning rate -> convergence
        old_q = self.q_estimates[ttype][action]
        self.q_estimates[ttype][action] = old_q + lr * (reward - old_q)

        # --- Mental model update ---
        self.mental_model.update(task.ambiguity.context_ambiguity)

        # --- Store in memory ---
        entry = MemoryEntry(
            task_type=ttype,
            context_features=task.ambiguity.as_array(),
            action_taken=action,
            reward=reward,
            timestamp=self.step
        )
        self.memory.store(entry)
        self.step += 1

    # ------------------------------------------------------------------
    # Belief introspection — for metrics computation
    # ------------------------------------------------------------------

    def intent_entropy(self, task_type: str) -> float:
        """Shannon entropy of the current intent Dirichlet (in bits)."""
        alpha = self.intent_priors[task_type]
        alpha_0 = alpha.sum()
        probs = alpha / alpha_0
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log2(probs)))

    def preference_entropy(self, task_type: str) -> float:
        """Entropy of the Beta preference prior (in bits)."""
        a, b = self.preference_priors[task_type]
        # Beta entropy: ln(B(a,b)) - (a-1)*ψ(a) - (b-1)*ψ(b) + (a+b-2)*ψ(a+b)
        from scipy.special import betaln, digamma
        ent_nats = (
            betaln(a, b)
            - (a - 1) * digamma(a)
            - (b - 1) * digamma(b)
            + (a + b - 2) * digamma(a + b)
        )
        return float(ent_nats / np.log(2))  # convert to bits

    def total_belief_entropy(self, task_type: str) -> float:
        """Combined entropy across intent and preference beliefs."""
        return self.intent_entropy(task_type) + max(0.0, self.preference_entropy(task_type))

    def sample_intent_distribution(self, task_type: str) -> np.ndarray:
        """Sample from the Dirichlet prior — for the convergence plot."""
        alpha = self.intent_priors[task_type]
        return self.rng.dirichlet(alpha)
