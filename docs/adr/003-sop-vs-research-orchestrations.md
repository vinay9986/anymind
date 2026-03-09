# ADR 003: Dichotomy of SOP and Research Orchestrations

## Status
Accepted

## Context
Agentic workflows generally fall into two categories: deterministic/repeatable processes and exploratory/open-ended discovery. Using a single agent for both leads to either over-exploration on simple tasks or under-exploration on complex ones.

## Decision
I have bifurcated the system into two primary orchestration modes:
- **SOP Orchestration**: Optimized for linear, verified task completion (e.g., "follow this checklist"). Uses AIoT to iterate until a verification signal is met.
- **Research Orchestration**: Optimized for deep exploration. Uses AGoT to dynamically expand its search graph when it encounters complex sub-topics, merging findings from multiple tools (Kagi, Scrapfly, etc.) into a cohesive synthesis.

## Consequences
- **Efficiency**: SOPs are handled with minimal token overhead.
- **Depth**: Research tasks can drill deep into ambiguous problems without manual intervention.

## Alternatives Considered
- **Single Generalist Agent**: Rejected because "one-size-fits-all" agents often hallucinate completions when they should be exploring, or waste tokens when they should be finishing.
