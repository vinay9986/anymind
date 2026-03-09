# ADR 001: Provider-Agnostic Model Orchestration

## Status
Accepted

## Context
The AI ecosystem is highly volatile, with market leadership shifting frequently between OpenAI, Anthropic, and Google. For enterprise-grade agentic workflows, hard-coding to a single model provider creates significant execution risk and vendor lock-in. Different tasks (e.g., creative writing vs. complex reasoning) often perform better on different foundation models.

## Decision
I have implemented a strictly provider-agnostic runtime using a factory pattern (`llm_factory.py`) and a standardized tool-calling interface. The system abstracts away provider-specific message formats and tool-definition schemas.

## Consequences
- **Portability**: Agents can be migrated between Bedrock, OpenAI, and Ollama by simply updating a JSON configuration.
- **Resilience**: Outages or performance degradation in one provider can be mitigated by switching to another with zero code changes.
- **Tradeoff**: I chose to sacrifice some provider-specific "beta" features (like specific prompt caching mechanisms) in favor of a clean, maintainable abstraction layer.

## Alternatives Considered
- **Direct Integration**: Using provider-specific SDKs (e.g., `boto3` for Bedrock). Rejected due to high maintenance overhead and lack of portability.
- **LangChain/LlamaIndex**: While powerful, these frameworks introduce significant "framework bloat" and can make it difficult to debug the underlying data flow. I chose a custom factory for maximum control over token efficiency and latency.
