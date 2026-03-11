# ADR 003: Separate SOP and Research Orchestrations

## Status

Accepted

## Context

The repository supports both structured task execution and exploratory research. Those workflows need different coordination logic even when they share lower-level reasoning runtimes.

## Decision

AnyMind keeps two higher-level orchestration modes:

- **`sop_agent`** for executing JSON SOP graphs with node-level solving and optional optimization
- **`research_agent`** for decomposing a question into probes, routing those probes to different reasoning runtimes, and synthesizing the result

Both orchestrations sit on top of the shared agent catalog rather than embedding all behavior into one generalist runtime.

## Consequences

- SOP execution can stay explicit about graph structure, node dependencies, and refinement.
- Research execution can focus on decomposition, probe batching, and synthesis.
- The split keeps the code easier to reason about than a single runtime that tries to infer both modes from prompts alone.

## Alternatives Considered

- One generalist orchestration layer: rejected because it would hide important control flow differences and make both modes harder to tune.
