# Ambiguity Resolution Dynamics
### How AI Coworkers Learn to Navigate Uncertainty

*A data science analysis of AI agent learning dynamics and probabilistic workflow architectures.*

---

## Abstract

Modern AI coworker platforms claim their agents improve over time. However, the mathematical foundation of this improvement is rarely specified. This project models AI coworker performance using Bayesian inference, information theory, and a POMDP-inspired workflow engine to quantify the rate at which an agent collapses task ambiguity into confident action.

The central contribution is the **Ambiguity Resolution Rate (ARR)**: a novel metric measuring the bits of Shannon entropy reduced per task interaction. I also benchmark three memory architectures (Recency, Frequency, and an ACT-R inspired Hybrid) and demonstrate how CUSUM change-point detection can identify when user preferences shift mid-deployment.

---

## Motivation

Many next-generation AI agent products describe architectures that include the following components:
*   **Mental Models**: An internal representation of organizational structure and stakeholders.
*   **Patterns**: Learned user preferences, such as formal or casual tone and communication rules.
*   **Structures**: Reusable templates for common tasks.
*   **Adaptive Learning**: Mechanisms through which performance improves incrementally.

While these are compelling features, this project aims to measure and quantify these dynamics rigorously.

In this simulation, each coworker handles tasks across four categories: **Scheduling**, **CRM Updates**, **Email Triage**, and **Operational** workflows. Every task is characterized by a 3-dimensional **ambiguity vector** (comprising intent ambiguity, context ambiguity, and preference ambiguity). These dimensions are drawn from Beta distributions that can shift mid-simulation to model concept drift.

---

## Methodology

### Task Model
Each task carries a 3-dimensional ambiguity vector.

| Dimension | Meaning | High ambiguity example |
|---|---|---|
| `intent_ambiguity` | Clarity of user intent. | "Handle this email thread." |
| `context_ambiguity` | Required organizational context. | Unknown stakeholder hierarchy. |
| `preference_ambiguity` | Knowledge of user style preferences. | A new user with no interaction history. |

Tasks arrive via a **bursty Poisson process** with a 15% burst probability, providing a more realistic scenario than a uniform stream.

### Bayesian Agent
The coworker maintains several probabilistic models:
*   **Dirichlet prior** over intent classes, which is conjugate-updated after each task.
*   **Beta-Binomial prior** over binary preferences, updated via a binarized reward signal.
*   **Thompson Sampling** for action selection, which naturally balances exploration and exploitation as beliefs sharpen.

This approach mirrors advanced agentic architectures. The Dirichlet distribution represents the mental models, the Beta-Binomial distribution represents the learned patterns, and Thompson sampling represents the probabilistic selection of the next best step.

### POMDP Workflow Engine
The workflow consists of five phases: `Intake`, `Clarify`, `Execute`, `Verify`, and `Deliver`.

The agent learns **per-task-type thresholds**. If its belief entropy falls below a threshold, it skips the Clarify or Verify phase. These thresholds adapt online, allowing the agent to skip unnecessary steps as confidence increases. The **Skip Rate** metric tracks this efficiency gain.

### Memory Architectures
I evaluate three competing models of how the agent stores and retrieves past interactions:

| Architecture | Mechanism | Strength | Weakness |
|---|---|---|---|
| **Recency** | Exponential decay. | Fast drift adaptation. | Discards rare but important patterns. |
| **Frequency** | Equal-weight count mean. | Stable and noise-resistant. | Slow to adapt after preference shifts. |
| **Hybrid (ACT-R)** | Power-law decay. | Balances recency and frequency. | Requires more parameter tuning. |

---

## Key Findings

### Fig 1: Ambiguity Heatmap
*Darker regions indicate lower ambiguity, signifying that the agent has learned.*

Each task type has a distinct ambiguity signature. Scheduling tasks exhibit high **context** ambiguity, whereas Email Triage shows persistently high **preference** ambiguity because routing rules and tone are highly personal. The CRM Update's preference row shows a visible mid-simulation brightening, which corresponds to the concept drift event injected at task 600.

![Ambiguity Heatmap](figures/fig1_ambiguity_heatmap.png)

---

### Fig 2: ARR Curves by Task Type
*Higher ARR values indicate faster learning.*

ARR is highest during the first 100 interactions, reflecting rapid early learning, and stabilizes as beliefs converge. Email Triage maintains the lowest ARR. This aligns with its high preference ambiguity, which is the most challenging dimension to learn without explicit user feedback.

![ARR Curves](figures/fig2_arr_curves.png)

---

### Fig 3: Memory Architecture Showdown
*Comparison of three architectures on the same task stream.*

All three architectures perform similarly pre-drift. After concept drift is injected at task 600, **Recency** recovers fastest, as it discards stale memories quickly. Frequency-based memory is the slowest to adapt. The Hybrid model sits between them. The ACT-R power-law decay correctly down-weights old patterns but retains enough long-term structure to avoid the brittleness of the Recency model on rare tasks.

![Memory Showdown](figures/fig3_memory_showdown.png)

---

### Fig 4: Workflow Phase Evolution
*A stacked area chart of the time spent in each phase over 1000 tasks.*

The Clarify phase shrinks noticeably after approximately 200 tasks as the agent builds confidence in intent inference. The Verify phase also contracts, although at a slower rate. This validates the premise that the coworker becomes more efficient, spending less time on unnecessary verification as user preferences become well-established.

![Phase Evolution](figures/fig4_phase_evolution.png)

---

### Fig 5: Concept Drift Detection and Recovery
*CUSUM detects the preference shift shortly after it occurs.*

The dashed pink line marks the injection of a sudden user preference shift, simulating a change in working style or a new team member. CUSUM detects multiple change-points in the reward signal. The agent's reward decreases and then partially recovers as it re-learns. This recovery arc is clearly visible in the rolling reward curve.

![Drift Recovery](figures/fig5_drift_recovery.png)

---

### Fig 6: Posterior Belief Convergence
*Progression from a uniform prior to a concentrated posterior across five learning stages.*

Each panel displays the agent's Dirichlet distribution over intent classes at snapshots t=10, 50, 200, 500, and 900. Early in the simulation (t=10), the distribution is nearly flat. By t=900, the agent has collapsed toward a dominant intent class for each task type, visualizing the formation of a mental model in real time.

![Belief Convergence](figures/fig6_belief_convergence.png)

---

## Metrics Summary

| Memory Architecture | Mean ARR (bits/task) | Final Reward | Pre-Drift Reward | Post-Drift Reward | Recovery Δ |
|---|---|---|---|---|---|
| Recency | 0.0006 | 0.3362 | 0.3235 | 0.3300 | +0.0064 |
| Frequency | 0.0007 | 0.2518 | 0.3223 | 0.2966 | −0.0257 |
| Hybrid (ACT-R) | 0.0007 | 0.2762 | 0.3299 | 0.3144 | −0.0155 |

Key observation: Recency achieves the best post-drift recovery (positive delta), while Frequency is the least adaptable. The Hybrid architecture shows the highest pre-drift peak reward, as its power-law memory retrieval finds an optimal balance in a stable environment.

---

## Novel Contributions

1.  **Ambiguity Resolution Rate (ARR)**: A task-level metric that quantifies learning in bits per interaction, grounded in information theory.
2.  **3-Dimensional Ambiguity Vector**: Decomposing task difficulty into intent, context, and preference dimensions.
3.  **ACT-R Memory Benchmark**: Applying a cognitive science memory model to AI agent design and comparing it empirically against simpler baselines.
4.  **CUSUM-based Drift Detection**: Quantifying how quickly an AI coworker detects and recovers from preference shifts.

---

## Implications for Product Development

*   **Preference ambiguity** is the most difficult dimension to resolve. Agent products should incorporate explicit preference-capture mechanisms, such as asking users to rate responses or providing example emails.
*   **Recency-weighted memory** outperforms frequency-based memory in real-world deployments where teams and preferences evolve. This approach is highly recommended for production memory systems.
*   **Concept drift** is detectable within a short window of tasks using CUSUM. This capability could be used to power automatic "re-onboarding" triggers when a coworker's performance degrades.
*   **Skip Rate** serves as a practical proxy metric for coworker maturity. Teams could track this metric in dashboards to monitor how well the agent has adapted to its environment.

---

## Setup and Usage

```bash
# Clone the repository
git clone https://github.com/AbhiOnAI77/agentic-coworker-analysis.git
cd agentic-coworker-analysis

# Install dependencies
pip install -r requirements.txt

# Run the full analysis (generates all figures and CSV)
python analysis.py
```

All figures are saved to the `figures/` directory. The simulation takes approximately 10 seconds to execute on a modern laptop.

---

## Tech Stack

*   **Python 3.10+**
*   **NumPy and SciPy**: Bayesian inference, information theory, and CUSUM.
*   **Pandas**: Metrics aggregation.
*   **Matplotlib and Seaborn**: Visualizations.
*   **No ML Frameworks**: All models are implemented from scratch to ensure complete transparency.

---

*Built as part of a data science portfolio exploring AI agent learning dynamics.*
*Author: Abhishek S*
