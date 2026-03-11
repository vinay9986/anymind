# AnyMind Reference

## Entry Points

### CLI

The `anymind` console script is backed by [../src/anymind/cli/main.py](../src/anymind/cli/main.py).

Important options:

- `--agent`
- `--config` / `-c`
- `--model`
- `--provider`
- `--temperature`
- `--tools-policy`
- `--tools-enabled` / `--no-tools`
- `--thread-id`
- `--query` / `-q`
- `--log-level`

Without `--query`, the CLI runs in interactive mode.

### API

The FastAPI app lives in [../src/anymind/api/app.py](../src/anymind/api/app.py).

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness check. |
| `POST /agents/run` | Synchronous agent execution. |
| `POST /jobs` | Async execution backed by `JobManager`. |
| `GET /jobs/{job_id}` | Job status and result retrieval. |
| `POST /jobs/{job_id}/pause` | Pause a running job. |
| `POST /jobs/{job_id}/resume` | Resume a paused job. |
| `POST /jobs/{job_id}/cancel` | Cancel a job. |

## Agents

| Agent | Implementation | Notes |
| --- | --- | --- |
| `research_agent` | `research_director_agent.py` | Delegates probe questions to other runtimes and synthesizes the final answer. |
| `sop_agent` | `sop_agent.py` | Expects a JSON SOP payload or `@path` to a JSON file. |
| `aiot_agent` | `aiot_agent.py` | Brain/worker iterative loop with validated JSON checkpoints. |
| `giot_agent` | `giot_agent.py` | Parallel agent loop with temperature spread and facilitator-style convergence. |
| `agot_agent` | `agot_agent.py` | Adaptive graph planning and worker-pool execution. |
| `got_agent` | `got_agent.py` | Graph-of-thought search with reflection and verification. |

## Configuration Discovery

### Model config

Search order:

1. `AM_MODEL_CONFIG`
2. `./model.json`
3. `./config/model.json`
4. `repo-root/config/model.json`
5. `~/.config/anymind/model.json`

Important fields in `ModelConfig`:

- `model`
- `model_provider`
- `model_parameters`
- `thread_id`
- `tools_enabled`
- `tools_policy`
- `budget_tokens`
- `search`
- `state_dir`
- `checkpoint`
- `cache`
- agent-specific sections such as `aiot`, `giot`, `agot`, `got`, `sop`, and `research_director`

### MCP config

Search order:

1. `AM_MCP_CONFIG`
2. `./mcp_servers.json`
3. `./config/mcp_servers.json`
4. `repo-root/config/mcp_servers.json`
5. `~/.config/anymind/mcp_servers.json`

The MCP registry resolves relative paths against the working directory used to build the session.

## Tool Policies

| Policy | Behavior |
| --- | --- |
| `auto` | Expose all configured tools. |
| `planner` | Ask the model to choose a subset of tools for the current request. |
| `confirm` | Expose all tools, but require a terminal confirmation before execution. |
| `never` | Disable tool use. |

When the model provider is `ollama` and the config says `auto`, the session factory upgrades the policy to `planner`.

## Built-in Local MCP Tools

The bundled local MCP server is [../src/anymind/tools/mcp_local_tools.py](../src/anymind/tools/mcp_local_tools.py). It exposes:

- `current_time`
- `pdf_extract_text`
- `internet_search` when Kagi and Scrapfly credentials are available

These can be used through the local MCP registry just like external servers.

Credential requirements:

- `internet_search` is only registered when Kagi plus Scrapfly credentials are available
- `current_time` does not require external API credentials
- `pdf_extract_text` does not require Kagi or Scrapfly credentials

## Persistence and Caching

- Checkpoints default to SQLite through `AsyncSqliteSaver` when available.
- If the SQLite checkpoint dependency is unavailable, the code falls back to in-memory checkpoints.
- `checkpoint.backend=redis` is not currently implemented and raises an error.
- Usage data can be stored in Redis via `ANYMIND_USAGE_REDIS_URL`.
- Tool-result caching can wrap MCP calls when `cache.redis_url` is set in the model config.

## Environment Variables

### Config and state

- `AM_MODEL_CONFIG`
- `AM_MCP_CONFIG`
- `AM_STATE_DIR`

### Logging and display

- `ANYMIND_LOG_PATH`
- `ANYMIND_LOG_DIR`
- `ANYMIND_LOG_LEVEL`
- `ANYMIND_LOG_RUN_ID`
- `ANYMIND_EVIDENCE_MAX_CHARS`
- `ANYMIND_EVIDENCE_ITEM_MAX_CHARS`

### Tool and embedding passthrough

These may be forwarded into MCP server environments by the registry:

- `KAGI_API_KEY`
- `KAGI_API_ENDPOINT`
- `SCRAPFLY_API_KEY`
- `SCRAPFLY_API_KEY_SECRET_ARN`
- `SCRAPFLY_COUNTRY`
- `SCRAPFLY_FORMAT`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN`
- `AWS_PROFILE`
- `AWS_REGION`
- `AWS_DEFAULT_REGION`
- `ONNX_MODEL_PATH`
- `ONNX_TOKENIZER_PATH`
- `ONNX_MAX_LENGTH`
- `ONNX_EMBED_BATCH_SIZE`
- `PDF_ONNX_MODEL_PATH`
- `PDF_ONNX_TOKENIZER_PATH`
- `PDF_ONNX_MAX_LENGTH`

For adapters of this repo, the minimum search-related setup is:

- `KAGI_API_KEY` or `search.kagi_api_key` in the model config
- `SCRAPFLY_API_KEY` or `SCRAPFLY_API_KEY_SECRET_ARN`

Without those values, `internet_search` is not registered by the bundled local MCP server.
