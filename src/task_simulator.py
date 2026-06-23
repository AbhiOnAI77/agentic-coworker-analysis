"""
task_simulator.py

Models tasks the way an AI coworker platform actually handles them.
Each task has a 3D ambiguity vector: (intent, context, preference).

Notes:
  - Task arrival follows a bursty Poisson process (more realistic than uniform)
  - Each ambiguity dimension drawn from Beta distributions that can drift
  - Task types map directly to scheduling, CRM, email triage, operations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class TaskType(Enum):
    SCHEDULING = "Scheduling"
    CRM_UPDATE = "CRM Update"
    EMAIL_TRIAGE = "Email Triage"
    OPERATIONAL = "Operational"


@dataclass
class AmbiguityVector:
    """
    3D representation of task ambiguity.

    intent_ambiguity    : How clear is what the user actually wants?
                          e.g., "handle this" vs "schedule a 30min call with Sarah"
    context_ambiguity   : How much org context is needed to act?
                          e.g., who are the decision makers, what's the cadence?
    preference_ambiguity: How well-known are the user's style preferences?
                          e.g., formal/casual, CC rules, response urgency
    """
    intent_ambiguity: float      # [0, 1]
    context_ambiguity: float     # [0, 1]
    preference_ambiguity: float  # [0, 1]

    @property
    def total(self) -> float:
        """L2 norm of ambiguity vector — overall task difficulty."""
        return float(np.sqrt(
            self.intent_ambiguity**2 +
            self.context_ambiguity**2 +
            self.preference_ambiguity**2
        ))

    def as_array(self) -> np.ndarray:
        return np.array([
            self.intent_ambiguity,
            self.context_ambiguity,
            self.preference_ambiguity
        ])


@dataclass
class Task:
    task_id: int
    task_type: TaskType
    ambiguity: AmbiguityVector
    arrival_time: float
    true_intent: int          # ground truth intent class (hidden from agent)
    true_preference: float    # ground truth preference value (hidden from agent)
    feedback_given: bool = False
    resolution_time: Optional[float] = None
    resolved_correctly: Optional[bool] = None


class TaskSimulator:
    """
    Generates realistic task streams with configurable ambiguity dynamics.

    The Beta distribution parameters (alpha, beta) for each ambiguity dimension
    can shift mid-simulation to simulate concept drift — user preferences change,
    new org context emerges, or the user's communication style evolves.
    """

    # Base Beta distribution params per task type
    # Tuned so scheduling tasks have high context ambiguity,
    # email triage has high preference ambiguity, etc.
    TASK_PROFILES = {
        TaskType.SCHEDULING: {
            "intent":     (2.0, 5.0),   # fairly clear intent, low ambiguity
            "context":    (5.0, 2.0),   # high context needs (who? when? why?)
            "preference": (3.0, 3.0),   # moderate preference uncertainty
            "weight":     0.30
        },
        TaskType.CRM_UPDATE: {
            "intent":     (4.0, 3.0),   # moderately clear
            "context":    (4.0, 2.5),   # needs a lot of org context
            "preference": (2.0, 5.0),   # low preference ambiguity (CRM is structured)
            "weight":     0.25
        },
        TaskType.EMAIL_TRIAGE: {
            "intent":     (2.5, 4.0),   # intent can be ambiguous
            "context":    (2.0, 5.0),   # context often provided in email
            "preference": (5.0, 2.0),   # HIGH preference ambiguity (tone, routing)
            "weight":     0.30
        },
        TaskType.OPERATIONAL: {
            "intent":     (4.0, 2.5),   # usually clear what to do
            "context":    (3.5, 3.0),   # moderate context needs
            "preference": (4.0, 2.5),   # high preference ambiguity (approval chains)
            "weight":     0.15
        }
    }

    def __init__(self, seed: int = 42, drift_at: Optional[int] = None):
        """
        Args:
            seed:     Random seed for reproducibility
            drift_at: If set, inject a concept drift event at this task index.
                      Preference ambiguity params shift to simulate user style change.
        """
        self.rng = np.random.default_rng(seed)
        self.drift_at = drift_at
        self._task_counter = 0
        self._time = 0.0

        # Mutable copies so we can modify for drift
        self.profiles = {k: dict(v) for k, v in self.TASK_PROFILES.items()}

    def _apply_drift(self):
        """
        Simulates a sudden shift in user preferences mid-simulation.
        E.g., new user onboards, or existing user changes their working style.
        Specifically cranks up preference_ambiguity Beta params.
        """
        print(f"  [drift] Concept drift injected at task {self._task_counter}")
        for task_type in self.profiles:
            # Flip the preference Beta — high ambiguity again (like a new user)
            a, b = self.profiles[task_type]["preference"]
            self.profiles[task_type]["preference"] = (b * 1.2, a * 0.5)

    def _sample_ambiguity(self, task_type: TaskType) -> AmbiguityVector:
        profile = self.profiles[task_type]
        # Each dimension is an independent Beta draw
        return AmbiguityVector(
            intent_ambiguity=float(self.rng.beta(*profile["intent"])),
            context_ambiguity=float(self.rng.beta(*profile["context"])),
            preference_ambiguity=float(self.rng.beta(*profile["preference"]))
        )

    def _sample_task_type(self) -> TaskType:
        types = list(self.TASK_PROFILES.keys())
        weights = [self.TASK_PROFILES[t]["weight"] for t in types]
        idx = self.rng.choice(len(types), p=weights)
        return types[idx]

    def _next_arrival_time(self) -> float:
        """
        Bursty Poisson arrivals: inter-arrival times follow an Exp distribution
        but with occasional burst periods (modeled as a 2-component mixture).
        More realistic than a uniform task stream.
        """
        is_burst = self.rng.random() < 0.15  # 15% chance of burst
        rate = 5.0 if is_burst else 1.0       # tasks arrive 5x faster in bursts
        return float(self.rng.exponential(1.0 / rate))

    def generate_task(self) -> Task:
        """Generate a single task."""
        if self.drift_at is not None and self._task_counter == self.drift_at:
            self._apply_drift()

        task_type = self._sample_task_type()
        ambiguity = self._sample_ambiguity(task_type)

        # Ground truth hidden values — agent must infer these
        true_intent = int(self.rng.integers(0, 3))   # 3 intent classes
        true_preference = float(self.rng.beta(3, 2)) # continuous preference value

        self._time += self._next_arrival_time()

        task = Task(
            task_id=self._task_counter,
            task_type=task_type,
            ambiguity=ambiguity,
            arrival_time=self._time,
            true_intent=true_intent,
            true_preference=true_preference
        )
        self._task_counter += 1
        return task

    def generate_batch(self, n: int) -> List[Task]:
        """Generate a batch of n tasks."""
        return [self.generate_task() for _ in range(n)]
