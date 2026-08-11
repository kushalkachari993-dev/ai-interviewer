# AI Interviewer Platform

FastAPI + React GenAI interview platform with adaptive LLM-generated questions, specialist interview agents, voice input, scorecards, strengths, weaknesses, and hiring recommendations.

## Features

- Server-owned interview sessions, rubrics, stage scores, and final evaluations
- 18 candidate-selected tracks spanning software, cloud, data/AI, security, quality, architecture, and management
- Shared Resume and HR stages plus two track-specific technical stages
- One required follow-up before each stage is scored
- Partial scores calculated only from completed stages
- OpenAI-backed evaluation through the Responses API
- Local deterministic fallback when no API key is configured
- Browser voice input and question playback
- React/Vite frontend and FastAPI backend

Available tracks: Backend, Frontend, Cloud, Terraform/IaC, DevOps/SRE, System Design, AI/ML, Data Engineering, Cybersecurity/AppSec, Kubernetes/Platform Engineering, Mobile, QA Automation/SDET, Full-Stack, Database Engineering, MLOps, DevSecOps/Security Operations, Solutions Architecture, and Engineering Management.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm install --prefix frontend
```

Copy `.env.example` to `.env` and set your key:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
FRONTEND_ORIGIN=http://127.0.0.1:5173
OPENAI_DISABLED=false
OPENAI_MAX_ATTEMPTS=3
OPENAI_TIMEOUT_SECONDS=20
OPENAI_RETRY_BUDGET_SECONDS=35
OPENAI_MAX_OUTPUT_TOKENS=4000
RATE_LIMIT_MODE=memory
```

`gpt-5-nano` automatically uses `minimal` reasoning effort to keep interview responses within the request timeout.
Set `OPENAI_REASONING_EFFORT` explicitly only when the selected model supports that value.

Do not commit `.env`. It is ignored by Git. The backend prefers values in `.env` over shell-level environment variables.
Set `OPENAI_DISABLED=true` when you want to force local fallback mode.

## Run In Development

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend:

```powershell
npm.cmd --prefix frontend run dev
```

Open `http://127.0.0.1:5173`.

## Production Build

```powershell
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

After `frontend/dist` is built, FastAPI serves the React app from `http://127.0.0.1:8000`.

## Deploy To Vercel

Import the GitHub repository into Vercel and add these environment variables in the Vercel dashboard:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_DISABLED=false
OPENAI_MAX_ATTEMPTS=3
OPENAI_TIMEOUT_SECONDS=20
OPENAI_RETRY_BUDGET_SECONDS=35
OPENAI_MAX_OUTPUT_TOKENS=4000
RATE_LIMIT_MODE=platform
```

The repo includes `api/index.py` as Vercel's Python ASGI entrypoint and `vercel.json` to build the React frontend into `frontend/dist`.

### Redis-free production rate limiting

Use Vercel WAF as the shared, globally enforced rate limiter. This avoids Redis and blocks abusive requests before they invoke FastAPI or the OpenAI API.

1. In the Vercel project, open **Firewall**, select **Configure**, and create a new rate-limit rule.
2. Match `POST` requests whose request path starts with `/api/`.
3. Use a fixed 60-second window with a limit of 15 requests, counted by IP and JA4 digest, and return HTTP 429.
4. Save and publish the rule.
5. Set `RATE_LIMIT_MODE=platform` only after that rule is active.

On plans that support multiple rate-limit rules, the routes can instead preserve the application-specific limits: session creation at 10/minute and answer submission at 30/minute. Keep `RATE_LIMIT_MODE=memory` for local development. The in-memory limiter is intentionally not presented as distributed protection.

If the app moves away from Vercel, use an API gateway or an atomic PostgreSQL rate-limit table as the shared enforcement point. Multiple server instances cannot enforce a global limit using process memory alone.

Vercel setup reference: https://vercel.com/docs/vercel-firewall/vercel-waf/rate-limiting

### API hardening

- Candidate answers are limited to 4,800 characters, combined interview context to 20,000 characters, and GenAI request bodies to 64 KiB when `Content-Length` is supplied.
- OpenAI responses use strict JSON Schema Structured Outputs and are validated again with Pydantic before being returned.
- Transient OpenAI 429 and 5xx failures use `Retry-After` or exponential backoff with jitter, capped by attempt and elapsed-time budgets.
- Candidate interview content is sent with `store: false`.

### Server-owned interview state

The browser may submit only a validated track ID when creating a session. The server selects the corresponding question banks and rubrics, then returns an opaque session ID, a server-issued version, public track metadata, and read-only stage views. The browser cannot submit rubrics, scores, prior results, or a final evaluation. Each answer must match the server's current version and phase, so stale or duplicate submissions are rejected, a primary answer always leads to a required follow-up, and only the follow-up submission can complete and score that stage.

The built-in session store is bounded to 500 sessions with a one-hour inactivity expiry. It is an in-memory store intended for local development or a single long-lived server. Before running this flow across multiple Vercel function instances, replace the store with shared durable persistence such as PostgreSQL; process memory is not shared between serverless instances.

## API

- `GET /api/health`
- `GET /api/tracks`
- `POST /api/sessions` with `{ "trackId": "backend" }`
- `GET /api/sessions/{session_id}`
- `POST /api/sessions/{session_id}/answer`
