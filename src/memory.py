"""
memory.py

Three competing memory architectures for the AI coworker.
These feed into the Bayesian agent's prior updates.

Architectures:
  1. RecencyMemory   — exponential decay, good for adapting to drift
  2. FrequencyMemory — frequency counts, good for common patterns
  3. HybridMemory    — ACT-R inspired power-law decay (recency + frequency)

The hybrid is the most interesting: it mirrors how human memory actually works,
which aligns with claims of "human-based memory". It also tends
to outperform the others in concept-drift scenarios.

Ref: Anderson, J.R. (1983). The Architecture of Cognition. ACT-R memory model.
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class MemoryEntry:
    """A single stored interaction."""
    task_type: str
    context_features: np.ndarray   # simplified feature vector
    action_taken: int              # which action the agent chose
    reward: float                  # outcome quality [0, 1]
    timestamp: int                 # task index when this was stored


class BaseMemory(ABC):
    """Abstract base — all memory architectures implement this interface."""

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.entries: List[MemoryEntry] = []
        self._current_time = 0

    def store(self, entry: MemoryEntry):
        self._current_time += 1
        if len(self.entries) >= self.capacity:
            self._evict()
        self.entries.append(entry)

    @abstractmethod
    def _evict(self):
        """Remove a memory entry when capacity is exceeded."""
        ...

    @abstractmethod
    def get_weighted_reward(self, task_type: str, action: int) -> float:
        """
        Return a weighted average reward for a given (task_type, action) pair.
        Weights differ by architecture.
        """
        ...

    def get_sample_count(self, task_type: str) -> int:
        return sum(1 for e in self.entries if e.task_type == task_type)


class RecencyMemory(BaseMemory):
    """
    Exponentially decays old interactions.
    Recent feedback is heavily weighted; old feedback fades quickly.

    Great for adapting when preferences change (concept drift).
    Struggles with rare-but-important patterns (low frequency, long ago).
    """

    def __init__(self, capacity: int = 500, decay_rate: float = 0.02):
        super().__init__(capacity)
        self.decay_rate = decay_rate  # λ in exp(-λ * age)

    def _evict(self):
        # Remove oldest entry
        self.entries.pop(0)

    def _recency_weight(self, entry: MemoryEntry) -> float:
        age = self._current_time - entry.timestamp
        return np.exp(-self.decay_rate * age)

    def get_weighted_reward(self, task_type: str, action: int) -> float:
        relevant = [e for e in self.entries
                    if e.task_type == task_type and e.action_taken == action]
        if not relevant:
            return 0.5  # neutral prior

        weights = np.array([self._recency_weight(e) for e in relevant])
        rewards = np.array([e.reward for e in relevant])

        total_w = weights.sum()
        if total_w < 1e-10:
            return 0.5
        return float(np.dot(weights, rewards) / total_w)


class FrequencyMemory(BaseMemory):
    """
    Weights interactions by frequency — common patterns dominate.
    Stable and resistant to noise, but slow to adapt when preferences shift.

    Essentially a counts-based estimate of E[reward | task_type, action].
    """

    def _evict(self):
        # Remove a random old entry (approximately)
        idx = np.random.randint(0, max(1, len(self.entries) // 4))
        self.entries.pop(idx)

    def get_weighted_reward(self, task_type: str, action: int) -> float:
        relevant = [e for e in self.entries
                    if e.task_type == task_type and e.action_taken == action]
        if not relevant:
            return 0.5
        # Simple mean — each entry has equal frequency weight
        return float(np.mean([e.reward for e in relevant]))


class HybridMemory(BaseMemory):
    """
    ACT-R inspired memory: combines recency AND frequency using a power-law.

    Activation of memory trace i:
        A_i = ln(Σ_j t_j^(-d)) where t_j = age at retrieval j, d = decay exponent

    In practice we approximate this as:
        weight_i = (1 / age_i^d) * frequency_bonus_i

    This is the most cognitively realistic model and the one that advanced
    "human-based memory" claims most closely resemble. It tends to win
    on the concept drift benchmark because it naturally down-weights
    stale patterns while still keeping trace of reliable long-term ones.
    """

    def __init__(self, capacity: int = 500, decay_exponent: float = 0.5):
        super().__init__(capacity)
        self.decay_exponent = decay_exponent  # d in ACT-R
        # Track retrieval counts per entry index (frequency component)
        self._retrieval_counts: Dict[int, int] = {}
        self._entry_ids: List[int] = []
        self._next_id = 0

    def store(self, entry: MemoryEntry):
        self._current_time += 1
        if len(self.entries) >= self.capacity:
            self._evict()
        self.entries.append(entry)
        entry_id = self._next_id
        self._next_id += 1
        self._entry_ids.append(entry_id)
        self._retrieval_counts[entry_id] = 0

    def _evict(self):
        # Evict lowest-activation entry
        if not self.entries:
            return
        activations = [
            self._activation(self.entries[i], self._entry_ids[i])
            for i in range(len(self.entries))
        ]
        min_idx = int(np.argmin(activations))
        eid = self._entry_ids.pop(min_idx)
        self._retrieval_counts.pop(eid, None)
        self.entries.pop(min_idx)

    def _activation(self, entry: MemoryEntry, entry_id: int) -> float:
        age = max(1, self._current_time - entry.timestamp)
        freq_bonus = 1.0 + 0.3 * self._retrieval_counts.get(entry_id, 0)
        return freq_bonus / (age ** self.decay_exponent)

    def get_weighted_reward(self, task_type: str, action: int) -> float:
        relevant_idx = [
            i for i, e in enumerate(self.entries)
            if e.task_type == task_type and e.action_taken == action
        ]
        if not relevant_idx:
            return 0.5

        activations = np.array([
            self._activation(self.entries[i], self._entry_ids[i])
            for i in relevant_idx
        ])
        rewards = np.array([self.entries[i].reward for i in relevant_idx])

        # Increment retrieval counts (memory trace strengthens on retrieval)
        for i in relevant_idx:
            eid = self._entry_ids[i]
            self._retrieval_counts[eid] = self._retrieval_counts.get(eid, 0) + 1

        total_w = activations.sum()
        if total_w < 1e-10:
            return 0.5
        return float(np.dot(activations, rewards) / total_w)


def build_memories(capacity: int = 500) -> Dict[str, BaseMemory]:
    """Convenience factory returning one instance of each architecture."""
    return {
        "Recency":   RecencyMemory(capacity=capacity),
        "Frequency": FrequencyMemory(capacity=capacity),
        "Hybrid":    HybridMemory(capacity=capacity),
    }
