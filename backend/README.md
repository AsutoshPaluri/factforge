# factforge — backend

FastAPI + LangGraph backend for the factforge fact-checking agent.

See the [repo-root README](../README.md) for the full project description.

## Local development

```bash
cd backend

# Install dependencies (uv creates .venv automatically)
uv sync

# Fill in your API keys
cp ../.env.example .env
# edit .env

# Run the dev server (hot-reload)
uv run uvicorn factforge.main:app --reload

# Visit http://localhost:8000/docs for OpenAPI UI
```

## Test

```bash
uv run pytest
```

## Layout

```
backend/
├── pyproject.toml
├── src/factforge/
│   ├── __init__.py
│   ├── main.py             # FastAPI app entry
│   ├── config.py           # Pydantic settings
│   ├── agent/              # LangGraph orchestration
│   │   ├── nodes/          # one file per stage
│   │   └── graph.py        # builds the graph
│   ├── clients/            # Gemini, Supabase, Redis, NLI, search
│   ├── middleware/         # cost tracker, kill switch, rate limit
│   ├── db/                 # repositories, models
│   ├── api/                # routes, schemas
│   └── utils/              # shared helpers
└── tests/
```
