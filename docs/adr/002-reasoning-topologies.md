# ADR 002: Selection of Reasoning Topologies (AIoT, GIoT, AGoT, GoT)

## Status
Accepted

## Context
Single-shot prompting is insufficient for complex, multi-step reasoning tasks. Different problem classes require different "cognitive architectures" to balance accuracy, latency, and token cost.

## Decision
AnyMind supports multiple reasoning topologies:
- **Autonomous Iteration of Thought (AIoT)**: Self-terminating inner dialogue for standard tasks.
- **Guided Iteration of Thought (GIoT)**: Fixed-step exploration for thoroughness.
- **Graph of Thoughts (GoT)**: Non-linear reasoning allowing for branching and merging of thoughts.
- **Adaptive Graph of Thoughts (AGoT)**: Dynamic DAG construction with LLM-driven complexity checks to optimize test-time compute.

## Consequences
- **Precision**: Heavy recursive reasoning (AGoT) is applied only when the LLM detects high complexity, saving tokens on trivial sub-tasks.
- **Flexibility**: The system can adapt its thought process based on the incoming query type (e.g., simple lookup vs. complex synthesis).

## Alternatives Considered
- **Chain of Thought (CoT) Only**: Rejected as it cannot handle non-linear problems or parallel hypothesis exploration.
- **Static Tree of Thoughts (ToT)**: Rejected because it lacks the ability to merge partial computations, which GoT handles via DAG structures.
