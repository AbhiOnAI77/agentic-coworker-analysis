"""Ambiguity Resolution Dynamics — src package."""
from .task_simulator import TaskSimulator, Task, TaskType, AmbiguityVector
from .bayesian_agent import BayesianAgent
from .memory import build_memories, RecencyMemory, FrequencyMemory, HybridMemory
from .workflow_engine import ProbabilisticWorkflowEngine
from .metrics import (
    compute_arr, compute_kl_from_uniform, preference_convergence_time,
    cusum_detect, rolling_reward, information_gain_per_interaction,
    compute_per_type_arr,
)
from .visualizations import (
    plot_ambiguity_heatmap, plot_arr_curves, plot_memory_showdown,
    plot_phase_evolution, plot_drift_recovery, plot_belief_convergence,
)
