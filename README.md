# Noto

Noto is a tiny Flask app that pairs a simple chat UI with a local LLM endpoint (LM Studio / OpenAI-compatible).  
It stores per-handle conversation history + light “memory” in a local SQLite DB under `instance/noto.db`.

> ⚠️ For personal use. Don’t put sensitive info in here. The `/debug/*` endpoints are locked to localhost or a token.

---

## Features

- Flask + SQLite via SQLAlchemy (stored in `instance/noto.db`)
- Frontend at `/` (plain HTML/JS) with a **handle selector**
- Chat endpoint `POST /send` (expects `{ "u": "<handle>", "text": "<message>" }`)
- Per-user memory, minimal crisis-language guardrails
- Debug endpoints (locked down)
