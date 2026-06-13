# AI Interviewer Platform

FastAPI + React GenAI interview platform with adaptive LLM-generated questions, specialist interview agents, voice input, scorecards, strengths, weaknesses, and hiring recommendations.

## Features

- Adaptive question generation using previous candidate answers and evaluation results
- Resume, Coding, System Design, HR, and Evaluation agents
- OpenAI-backed evaluation through the Responses API
- Local deterministic fallback when no API key is configured
- Browser voice input and question playback
- React/Vite frontend and FastAPI backend

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
```

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
```

The repo includes `api/index.py` as Vercel's Python ASGI entrypoint and `vercel.json` to build the React frontend into `frontend/dist`.

## API

- `GET /api/health`
- `POST /api/question`
- `POST /api/agent`
- `POST /api/evaluation`
