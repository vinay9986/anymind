# ADR 001: Provider-Agnostic Model Orchestration

## Status

Accepted

## Context

The runtime needs to support different chat model providers without forcing every agent and tool path to know provider-specific details.

At the same time, the codebase already depends on LangChain and LangGraph primitives, so a realistic design should build on them instead of pretending they are absent.

## Decision

AnyMind centralizes model construction in `llm_factory.py` and treats provider choice as configuration. The runtime uses LangChain's `init_chat_model` as the portability layer, while keeping session creation, tool interception, evidence handling, and orchestration logic in project code.

## Consequences

- Switching between the provider examples in this repo is a configuration change, not an agent rewrite.
- Provider-specific concerns stay concentrated in the model factory, config, and a few middleware/interceptor paths.
- The project keeps the benefits of LangChain/LangGraph primitives without scattering provider branches through every runtime.

## Alternatives Considered

- Provider-specific model clients throughout the codebase: rejected because it would duplicate logic across agents and increase maintenance cost.
- A fully custom chat-model abstraction on top of raw SDKs: rejected because it would recreate functionality already provided by the LangChain layer used elsewhere in the project.
