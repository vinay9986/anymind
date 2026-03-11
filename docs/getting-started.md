# AnyMind Getting Started

This guide focuses on running the code that is already in this repo.

## Prerequisites

- Python 3.10 to 3.12
- Poetry
- Credentials for the model provider you choose
- Optional: Redis if you want the configured cache backend from `config/model.json`
- Optional: ONNX build dependencies if you want local embedding assets

## Install

```bash
poetry install
```

Optional ONNX assets:

```bash
poetry install --with onnx
python onnx_assets/build.py
```

## Choose Configuration

By default, AnyMind looks for model config in these locations, in order:

1. `AM_MODEL_CONFIG`
2. `./model.json`
3. `./config/model.json`
4. `repo-root/config/model.json`
5. `~/.config/anymind/model.json`

MCP server config follows the same pattern, using `AM_MCP_CONFIG` and `mcp_servers.json`.

The repo includes example model configs:

- [../config/model.json](../config/model.json)
- [../config/model.openai.json](../config/model.openai.json)
- [../config/model.bedrock.json](../config/model.bedrock.json)
- [../config/model.ollama.json](../config/model.ollama.json)

Example environment setup:

```bash
export AM_MODEL_CONFIG=$PWD/config/model.openai.json
export AM_MCP_CONFIG=$PWD/config/mcp_servers.json
```

`config/mcp_servers.json` already includes the bundled `local_tools` MCP server, so no separate local-tools process is required.

### Search tool credentials

If you want `internet_search` to work in the bundled `local_tools` MCP server, configure both upstream providers:

- Kagi: set `KAGI_API_KEY`, or set `search.kagi_api_key` in the model config so the session factory exports it into the process environment
- Scrapfly: set `SCRAPFLY_API_KEY`, or provide `SCRAPFLY_API_KEY_SECRET_ARN`

Example:

```bash
export KAGI_API_KEY=your_kagi_key
export SCRAPFLY_API_KEY=your_scrapfly_key
```

Important: without these values, the bundled local MCP server does not register `internet_search`. `current_time` and `pdf_extract_text` do not depend on those search credentials.

## Run the CLI

Research-oriented workflow:

```bash
poetry run anymind --agent research_agent -q "Compare recent CPI trends across G7 nations"
```

Interactive CLI:

```bash
poetry run anymind --agent research_agent
```

SOP workflow from a file:

```bash
poetry run anymind --agent sop_agent -q "@/absolute/path/to/sop.json"
```

You can override model settings at runtime:

```bash
poetry run anymind \
  --agent research_agent \
  --provider openai \
  --model gpt-5.1 \
  --tools-policy planner \
  -q "Summarize the latest public SEC filing for Company X"
```

## Run the API

Start the FastAPI service:

```bash
poetry run anymind serve --host 0.0.0.0 --port 8000
```

Important endpoints:

- `GET /health`
- `POST /agents/run`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/pause`
- `POST /jobs/{job_id}/resume`
- `POST /jobs/{job_id}/cancel`

Example synchronous request:

```bash
curl -X POST http://127.0.0.1:8000/agents/run \
  -H 'content-type: application/json' \
  -d '{
    "message": "Compare recent CPI trends across G7 nations",
    "agent": "research_agent"
  }'
```

## Tests

Run the test suite:

```bash
poetry run pytest -v
```

The repository already includes targeted tests for config loading, tool policy, evidence handling, usage tracking, caching, and MCP registry behavior.
