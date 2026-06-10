from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"
ENV = {}


def load_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


ENV = load_env()
OPENAI_API_KEY = ENV.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = ENV.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
FRONTEND_ORIGIN = ENV.get("FRONTEND_ORIGIN") or os.getenv("FRONTEND_ORIGIN") or "http://127.0.0.1:5173"
OPENAI_KEY_SOURCE = "env_file" if ENV.get("OPENAI_API_KEY") else "process_env" if os.getenv("OPENAI_API_KEY") else "missing"
OPENAI_DISABLED = (os.getenv("OPENAI_DISABLED") or ENV.get("OPENAI_DISABLED") or "").lower() in {"1", "true", "yes"}
RATE_LIMITS = {
    "/api/question": (20, 60),
    "/api/agent": (15, 60),
    "/api/evaluation": (5, 60),
}
RATE_BUCKETS: dict[tuple[str, str], deque[float]] = defaultdict(deque)


class Agent(BaseModel):
    name: str
    short: str | None = None
    purpose: str
    metric: str
    question: str = ""
    keywords: list[str] = Field(default_factory=list)


class AgentRequest(BaseModel):
    agent: Agent
    answer: str = Field(min_length=1)


class AgentResult(BaseModel):
    score: float
    strengths: list[str]
    weaknesses: list[str]
    notes: str
    followUpQuestion: str
    mode: str


class QuestionRequest(BaseModel):
    agent: Agent
    agentIndex: int
    answers: list[str] = Field(default_factory=list)
    results: list[dict[str, Any] | None] = Field(default_factory=list)


class QuestionResult(BaseModel):
    question: str
    rationale: str
    mode: str


class EvaluationRequest(BaseModel):
    agents: list[Agent]
    answers: list[str]
    results: list[dict[str, Any] | None]


class ScorecardItem(BaseModel):
    metric: str
    score: float


class EvaluationResult(BaseModel):
    overallScore: float
    scorecard: list[ScorecardItem]
    strengths: list[str]
    weaknesses: list[str]
    hiringRecommendation: str
    extra: str
    mode: str


app = FastAPI(title="AI Interviewer Platform", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_genai_routes(request: Request, call_next):
    limit = RATE_LIMITS.get(request.url.path)
    if not limit:
        return await call_next(request)

    max_requests, window_seconds = limit
    client_ip = request.client.host if request.client else "unknown"
    bucket = RATE_BUCKETS[(client_ip, request.url.path)]
    now = time.monotonic()

    while bucket and now - bucket[0] > window_seconds:
        bucket.popleft()

    if len(bucket) >= max_requests:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please wait and try again."},
        )

    bucket.append(now)
    return await call_next(request)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mode": "fallback" if OPENAI_DISABLED or not OPENAI_API_KEY else "openai",
        "model": OPENAI_MODEL,
        "keySource": OPENAI_KEY_SOURCE,
    }


@app.post("/api/question", response_model=QuestionResult)
def generate_question(payload: QuestionRequest) -> dict[str, Any]:
    fallback = local_question(payload.agent, payload.agentIndex, payload.answers, payload.results)
    if OPENAI_DISABLED or not OPENAI_API_KEY:
        return {**fallback, "mode": "fallback"}

    system = (
        "You are an adaptive AI interviewer. Generate one interview question for the named agent. "
        "The question must be specific, concise, role-relevant, and different from earlier answered prompts. "
        "Use previous answers and results to adapt difficulty and avoid repetition. "
        "Return only strict JSON with these keys: question, rationale."
    )
    user = json.dumps(
        {
            "agent": payload.agent.model_dump(),
            "agentIndex": payload.agentIndex,
            "previousAnswers": payload.answers,
            "previousResults": payload.results,
        }
    )
    return with_openai_fallback(system, user, fallback)


@app.post("/api/agent", response_model=AgentResult)
def run_agent(payload: AgentRequest) -> dict[str, Any]:
    fallback = local_agent_result(payload.agent, payload.answer)
    if OPENAI_DISABLED or not OPENAI_API_KEY:
        return {**fallback, "mode": "fallback"}

    system = (
        "You are an expert AI interview agent. Assess the candidate answer for the named interview stage. "
        "Return only strict JSON with these keys: score, strengths, weaknesses, notes, followUpQuestion. "
        "score must be a number from 0 to 100. strengths and weaknesses must be arrays of short strings. "
        "followUpQuestion should be one concise next question."
    )
    user = json.dumps(
        {
            "agentName": payload.agent.name,
            "agentPurpose": payload.agent.purpose,
            "rubricMetric": payload.agent.metric,
            "question": payload.agent.question,
            "answer": payload.answer,
        }
    )
    return with_openai_fallback(system, user, fallback)


@app.post("/api/evaluation", response_model=EvaluationResult)
def run_evaluation(payload: EvaluationRequest) -> dict[str, Any]:
    fallback = local_final_evaluation(payload.agents, payload.answers, payload.results)
    if OPENAI_DISABLED or not OPENAI_API_KEY:
        return {**fallback, "mode": "fallback"}

    system = (
        "You are the final evaluation agent for a hiring interview. Use the provided agent results and candidate "
        "answers to produce a hiring summary. Return only strict JSON with these keys: overallScore, scorecard, "
        "strengths, weaknesses, hiringRecommendation, extra. scorecard must be an array of objects with metric and "
        "score. strengths and weaknesses must be arrays of short strings. overallScore must be a number from 0 to 100."
    )
    user = json.dumps(payload.model_dump())
    return with_openai_fallback(system, user, fallback)


def with_openai_fallback(system: str, user: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return {**fallback, **call_openai(system, user), "mode": "openai"}
    except Exception as exc:
        return {
            **fallback,
            "mode": "fallback",
            "extra": f"OpenAI call failed, using local fallback. {exc}",
        }


def call_openai(system: str, user: str) -> dict[str, Any]:
    request_body = json.dumps(
        {
            "model": OPENAI_MODEL,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=request_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")[:220]
        raise HTTPException(status_code=502, detail=f"OpenAI API returned {exc.code}: {detail}") from exc

    text = extract_output_text(data)
    return json.loads(text.removeprefix("```json").removesuffix("```").strip())


def extract_output_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"])

    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("text"):
                chunks.append(str(content["text"]))
    if not chunks:
        raise ValueError("No text returned from OpenAI")
    return "\n".join(chunks)


def local_agent_result(agent: Agent, answer: str) -> dict[str, Any]:
    normalized = answer.lower()
    keyword_hits = sum(1 for keyword in agent.keywords if keyword.lower() in normalized)
    length_score = min(len(answer.strip().split()) / 90, 1)
    keyword_score = keyword_hits / len(agent.keywords) if agent.keywords else 0.5
    score = round((length_score * 45 + keyword_score * 55) * 10) / 10

    return {
        "score": score,
        "strengths": [f"{agent.name} found relevant evidence for {agent.metric}."] if score >= 60 else [],
        "weaknesses": [f"{agent.name} needs more specific examples and measurable impact."] if score < 60 else [],
        "notes": "Local rubric used because no OpenAI API result was available.",
        "followUpQuestion": f"What is one concrete tradeoff you made during this {agent.purpose.lower()} example?",
    }


def local_question(
    agent: Agent, agent_index: int, answers: list[str], results: list[dict[str, Any] | None]
) -> dict[str, Any]:
    completed = len([answer for answer in answers if answer])
    prior_text = " ".join(answer.lower() for answer in answers if answer)
    weak_metrics = [
        str(result.get("notes") or result.get("weaknesses") or "")
        for result in results
        if result and float(result.get("score", 0)) < 60
    ]

    variants = {
        "Resume Agent": [
            "Walk me through a project from your resume where your personal contribution changed the outcome.",
            "Which resume project best represents the role you want next, and what measurable result did you drive?",
            "Pick one resume bullet and explain the context, your decisions, and the impact behind it.",
        ],
        "Coding Agent": [
            "How would you debug a production API that suddenly became slow?",
            "Describe how you would design tests for a bug that only appears under high traffic.",
            "What tradeoffs would you consider when refactoring a critical service with no downtime?",
        ],
        "System Design Agent": [
            "Design a scalable interview scheduling system for recruiters and candidates.",
            "Design a notification system that reliably handles reminders, cancellations, and retries.",
            "Design the data model and APIs for a collaborative candidate evaluation platform.",
        ],
        "HR Agent": [
            "Describe a time you handled disagreement with a teammate.",
            "Tell me about a time you received difficult feedback and changed your approach.",
            "Give an example of how you kept a project moving when priorities were unclear.",
        ],
        "Evaluation Agent": [
            "Summarize why this candidate should or should not advance.",
            "What is the strongest hiring signal so far, and what is the largest remaining risk?",
            "If you had one more live round, what would you probe and why?",
        ],
    }
    options = variants.get(agent.name, [agent.question or f"Tell me about your fit for {agent.purpose}."])
    selector = (completed + agent_index + len(prior_text) + len(weak_metrics)) % len(options)

    if weak_metrics and agent.name != "Evaluation Agent":
        question = f"{options[selector]} Please include one concrete example and one measurable signal."
    else:
        question = options[selector]

    return {
        "question": question,
        "rationale": "Local adaptive question selected from agent-specific variants using interview progress.",
    }


def local_final_evaluation(
    agents: list[Agent], answers: list[str], results: list[dict[str, Any] | None]
) -> dict[str, Any]:
    scorecard = [
        {"metric": agent.metric, "score": float((results[index] or {}).get("score", 0))}
        for index, agent in enumerate(agents)
    ]
    overall = round(sum(item["score"] for item in scorecard) / max(len(scorecard), 1))
    completed = len([answer for answer in answers if answer])
    strengths = [item for result in results if result for item in result.get("strengths", [])][:5]
    weaknesses = [item for result in results if result for item in result.get("weaknesses", [])][:5]

    if completed < len(agents):
        recommendation = "Complete the interview to generate a final recommendation."
    elif overall >= 72:
        recommendation = "Advance to the next round."
    elif overall >= 52:
        recommendation = "Hold for calibration and probe weaker areas."
    else:
        recommendation = "Do not advance based on current evidence."

    return {
        "overallScore": overall,
        "scorecard": scorecard,
        "strengths": strengths or ["Awaiting stronger evidence across the interview."],
        "weaknesses": weaknesses or ["No major weakness surfaced in this pass."],
        "hiringRecommendation": recommendation,
        "extra": "Local final evaluation generated without an OpenAI API result.",
    }


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
