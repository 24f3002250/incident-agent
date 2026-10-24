---
title: GA5 Incident Agent
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# GA5 incident agent (profile `ga5-incident-agent/v2`)

Routes: `POST /v2/incidents`, `POST /v2/incidents/{runId}/receipts`, `GET /v2/incidents/{runId}`.

Space secrets / variables:

| name | value |
|---|---|
| `AIPIPE_TOKEN` (secret) | your AI Pipe token |
| `LLM_MODEL` (optional) | default `gpt-4.1-mini` |
| `LLM_BASE_URL` (optional) | default `https://aipipe.org/openai/v1` |

- `planner.py` makes exactly one model call per new runId (the `chat incident-plan` span).
- `app.py` holds the persistent receipt-driven state machine (SQLite); receipts, GET and replays never call the model.
- `otlp.py` builds the trace purely from stored dispatches and receipts.
