# ADR 002: Selection of Reasoning Topologies

## Status

Accepted

## Context

Single-shot prompting is not a good fit for every workload in this repo. Research synthesis, SOP execution, iterative refinement, and graph exploration place different demands on the runtime.

## Decision

AnyMind exposes multiple reasoning topologies as first-class agent runtimes:

- **AIoT** for iterative brain/worker loops
- **GIoT** for parallel multi-agent exploration with convergence
- **AGoT** for adaptive graph execution
- **GoT** for graph-of-thought search with branching and verification

Higher-level runtimes such as `research_agent` and `sop_agent` can delegate into those lower-level strategies as needed.

## Consequences

- The runtime can match the solving strategy to the task instead of forcing one shape onto every workload.
- Research and SOP flows can reuse the same lower-level reasoning building blocks.
- The tradeoff is added implementation complexity and a larger testing surface.

## Alternatives Considered

- Single general-purpose agent loop: rejected because it would blur important differences between research, iterative solving, and structured SOP execution.
- Chain-of-thought only: rejected because it does not cover the branching and multi-agent behaviors already implemented in the repo.
