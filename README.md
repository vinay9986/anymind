# AnyMind

LangGraph-based multi-agent runtime with MCP tools, CLI, and API (Swagger).

## Quick start

```bash
poetry install
poetry run anymind chat --model gpt-4o-mini --provider openai

# API server (Swagger at /docs)
poetry run anymind serve --host 0.0.0.0 --port 8000
```

## Configuration

Config files are JSON. The runtime looks in this order (or uses explicit flags):

- `AM_MODEL_CONFIG` / `AM_PRICING_CONFIG` / `AM_MCP_CONFIG`
- `./model.json`, `./pricing.json`, `./mcp_servers.json`
- `./config/model.json`, `./config/pricing.json`, `./config/mcp_servers.json`
- `~/.config/anymind/model.json`, `~/.config/anymind/pricing.json`, `~/.config/anymind/mcp_servers.json`

### Tool policy

Set `tools_policy` in `model.json`:

- `auto` (default) - normal tool use
- `planner` - tool-selection planner gate (recommended if model lacks native tool calling)
- `confirm` - ask before each tool call
- `never` - disable tools even if tools are configured

Note: for Ollama, we default `auto` to `planner` to avoid tool-selection misfires.

### Budget enforcement

Set `budget_tokens` in `model.json` to enforce a strict token cap. When exceeded,
the session stops accepting new turns.

CLI overrides for fast testing:

```bash
anymind chat --model llama3.3:70b --provider ollama --tools-policy planner
anymind chat --model us.anthropic.claude-sonnet-4-20250514-v1:0 --provider bedrock
```

### Sample configs

See `config/` for examples. The default MCP config uses the built-in tool server:
`python -m anymind.tools.mcp_local_tools`.

### Checkpoints

Configure checkpoint storage in `model.json`:

```json
\"checkpoint\": {
  \"backend\": \"sqlite\",
  \"path\": \"~/.local/share/anymind/checkpoints.sqlite\"
}
```

Use `backend: \"memory\"` to disable persistence. Redis support can be added later.

## Built-in tools

The default MCP server exposes these tools:

- `current_time(format="iso"|"unix", timezone="UTC")`
- `google_search(search_term, result_num=10, country_region?, geolocation?, result_language?)`
- `internet_search(query, max_results=5, max_snippets=3, ...)` (Google CSE + Scrapfly)
- `pdf_extract_text(url|s3_key|pdf_base64, query/queries, search_mode="auto")`
- `retrieve_and_generate(query, number_of_results=5)` (Bedrock Knowledge Base)
- `get_current_time()` and `add(a, b)` (simple local tools)

### Tool configuration (env)

- Google CSE:
  - `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_ENGINE_ID`
  - or `GOOGLE_CSE_SECRET_ARN` (JSON with `api_key` + `engine_id`)
  - or `GOOGLE_CSE_API_KEY_SECRET_ARN` + `GOOGLE_CSE_ENGINE_ID_SECRET_ARN`
- Scrapfly (for `internet_search`):
  - `SCRAPFLY_API_KEY` (or `SCRAPFLY_API_KEY_SECRET_ARN`)
- PDF from S3:
  - `AGENT_DATA_BUCKET` (optional: `AGENT_DATA_BASE_PREFIX`)
- Bedrock KB:
  - `BEDROCK_KNOWLEDGE_BASE_ID`, `BEDROCK_KNOWLEDGE_BASE_MODEL_ARN`
  - plus AWS creds (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or `AWS_PROFILE`)

Semantic search for PDFs/pages is enabled by default. Set ONNX asset paths if you want
semantic search enabled:
`PDF_ONNX_MODEL_PATH`, `PDF_ONNX_TOKENIZER_PATH`, `PDF_ONNX_MAX_LENGTH`.

## Caching & pooling

LLM clients are reused via an internal pool. Tool results can be cached in Redis
when a Redis URL is provided in `model.json`:

```json
\"cache\": { \"redis_url\": \"redis://localhost:6379/0\", \"ttl_seconds\": 300 }
```

If Redis is unavailable, caching is skipped.

## Jobs (pause/resume)

The API supports async jobs for long-running agent calls:

- `POST /jobs` → returns `job_id`
- `GET /jobs/{job_id}` → status + result when complete
- `POST /jobs/{job_id}/pause`
- `POST /jobs/{job_id}/resume`
- `POST /jobs/{job_id}/cancel`

Pause is cooperative: it will pause before the next major step (planner/LLM call).

## Evidence ledger & citations

Tool calls are recorded in an evidence ledger. When `citations_enabled` is true,
the runtime rewrites the response to include citations like `[E1]` using the
tool outputs from the current turn.

You can disable this behavior in `model.json`:

```json
\"citations_enabled\": false
```

## Quality checks

Pre-commit hooks enforce:
- black
- pytest
- bandit
- pip-audit

Install hooks:

```bash
poetry run pre-commit install
```

## Structure

- `src/anymind/runtime/` - LLM factory, MCP tools, checkpoints, usage
- `src/anymind/policies/` - tool gating policies
- `src/anymind/agents/` - agent implementations
- `src/anymind/graphs/` - LangGraph graphs/workflows
- `src/anymind/cli/` - CLI entrypoints
- `src/anymind/api/` - FastAPI app (Swagger)

## Development

```bash
python -m pytest
```
