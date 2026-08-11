from __future__ import annotations

import json
import logging
import os
import random
import secrets
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock, RLock
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"
ENV: dict[str, str] = {}

MAX_AGENTS = 4
MAX_ANSWER_CHARS = 4_800
MAX_CONTEXT_CHARS = 20_000
MAX_KEYWORDS = 20
MAX_REQUEST_BYTES = 64 * 1024
MAX_SESSIONS = 500
SESSION_TTL_SECONDS = 60 * 60
RETRYABLE_OPENAI_STATUS_CODES = {429, 500, 502, 503, 504}
LOGGER = logging.getLogger(__name__)

AnswerText = Annotated[str, Field(min_length=1, max_length=MAX_ANSWER_CHARS)]
FeedbackText = Annotated[str, Field(min_length=1, max_length=500)]
TrackId = Literal[
    "backend",
    "frontend",
    "cloud",
    "terraform",
    "devops",
    "system-design",
    "ai-ml",
    "data-engineering",
    "cybersecurity",
    "kubernetes-platform",
    "mobile",
    "qa-sdet",
    "full-stack",
    "database",
    "mlops",
    "devsecops",
    "solutions-architecture",
    "engineering-management",
]


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
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT") or ENV.get("OPENAI_REASONING_EFFORT")
if not OPENAI_REASONING_EFFORT and (
    OPENAI_MODEL == "gpt-5-nano" or OPENAI_MODEL.startswith("gpt-5-nano-")
):
    OPENAI_REASONING_EFFORT = "minimal"
OPENAI_DISABLED = (os.getenv("OPENAI_DISABLED") or ENV.get("OPENAI_DISABLED") or "").lower() in {
    "1",
    "true",
    "yes",
}
OPENAI_MAX_ATTEMPTS = min(
    max(int(os.getenv("OPENAI_MAX_ATTEMPTS") or ENV.get("OPENAI_MAX_ATTEMPTS") or "3"), 1), 5
)
OPENAI_TIMEOUT_SECONDS = min(
    max(float(os.getenv("OPENAI_TIMEOUT_SECONDS") or ENV.get("OPENAI_TIMEOUT_SECONDS") or "20"), 1), 45
)
OPENAI_RETRY_BUDGET_SECONDS = min(
    max(float(os.getenv("OPENAI_RETRY_BUDGET_SECONDS") or ENV.get("OPENAI_RETRY_BUDGET_SECONDS") or "35"), 1),
    90,
)
OPENAI_MAX_OUTPUT_TOKENS = min(
    max(int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS") or ENV.get("OPENAI_MAX_OUTPUT_TOKENS") or "4000"), 1_200),
    8_000,
)
RATE_LIMIT_MODE = (os.getenv("RATE_LIMIT_MODE") or ENV.get("RATE_LIMIT_MODE") or "memory").lower()
if RATE_LIMIT_MODE not in {"memory", "platform"}:
    raise RuntimeError("RATE_LIMIT_MODE must be either 'memory' or 'platform'")

RATE_BUCKETS: dict[tuple[str, str], deque[float]] = defaultdict(deque)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Agent(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    short: str = Field(min_length=1, max_length=8)
    purpose: str = Field(min_length=1, max_length=200)
    metric: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=1, max_length=1_500)
    keywords: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(max_length=MAX_KEYWORDS)


class AIScoredFeedback(StrictModel):
    score: float = Field(ge=0, le=100)
    strengths: list[FeedbackText] = Field(max_length=5)
    weaknesses: list[FeedbackText] = Field(max_length=5)
    notes: str = Field(min_length=1, max_length=2_000)


class AIAgentOutput(AIScoredFeedback):
    followUpQuestion: str = Field(min_length=1, max_length=1_500)


class PrimaryAgentResult(AIAgentOutput):
    mode: Literal["openai", "fallback"]


class AgentResult(AIScoredFeedback):
    mode: Literal["openai", "fallback"]


class AIQuestionOutput(StrictModel):
    question: str = Field(min_length=1, max_length=1_500)
    rationale: str = Field(min_length=1, max_length=1_000)


class AIFinalNarrative(StrictModel):
    strengths: list[FeedbackText] = Field(max_length=5)
    weaknesses: list[FeedbackText] = Field(max_length=5)
    hiringRecommendation: str = Field(min_length=1, max_length=1_000)
    extra: str = Field(min_length=1, max_length=2_000)


class ScorecardItem(StrictModel):
    metric: str = Field(min_length=1, max_length=120)
    score: float = Field(ge=0, le=100)


class EvaluationResult(AIFinalNarrative):
    overallScore: float = Field(ge=0, le=100)
    scorecard: list[ScorecardItem] = Field(min_length=1, max_length=MAX_AGENTS)
    mode: Literal["openai", "fallback"]


class AnswerSubmission(StrictModel):
    answer: AnswerText
    phase: Literal["primary", "follow_up"]
    version: int = Field(ge=0)

    @field_validator("answer")
    @classmethod
    def answer_must_contain_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer must contain text")
        return stripped


class SessionCreateRequest(StrictModel):
    trackId: TrackId


class TrackView(StrictModel):
    id: TrackId
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=240)
    skills: list[str] = Field(min_length=1, max_length=8)


class StageView(StrictModel):
    index: int = Field(ge=0, lt=MAX_AGENTS)
    name: str
    short: str
    purpose: str
    metric: str
    question: str
    sample: str
    answer: str
    followUpQuestion: str
    followUpAnswer: str
    result: AgentResult | None
    status: Literal["upcoming", "primary", "follow_up", "complete"]


class SessionView(StrictModel):
    sessionId: str
    track: TrackView
    version: int = Field(ge=0)
    phase: Literal["primary", "follow_up", "complete"]
    activeIndex: int = Field(ge=0, lt=MAX_AGENTS)
    totalStages: int = Field(ge=1, le=MAX_AGENTS)
    completedStages: int = Field(ge=0, le=MAX_AGENTS)
    partialScore: float = Field(ge=0, le=100)
    stages: list[StageView] = Field(min_length=1, max_length=MAX_AGENTS)
    finalEvaluation: EvaluationResult | None


@dataclass(frozen=True)
class InterviewRubric:
    agent: Agent
    sample: str
    question_variants: tuple[str, ...]


@dataclass(frozen=True)
class InterviewTrack:
    id: TrackId
    name: str
    description: str
    skills: tuple[str, ...]
    technical_rubrics: tuple[InterviewRubric, InterviewRubric]


@dataclass
class StageRecord:
    rubric: InterviewRubric
    question: str
    answer: str = ""
    primary_result: PrimaryAgentResult | None = None
    follow_up_question: str = ""
    follow_up_answer: str = ""
    result: AgentResult | None = None


@dataclass
class InterviewSession:
    session_id: str
    track: InterviewTrack
    stages: list[StageRecord]
    version: int = 0
    active_index: int = 0
    phase: Literal["primary", "follow_up", "complete"] = "primary"
    final_evaluation: EvaluationResult | None = None
    updated_at: float = field(default_factory=time.monotonic)
    lock: Lock = field(default_factory=Lock, repr=False)


RUBRICS = (
    InterviewRubric(
        agent=Agent(
            name="Resume Agent",
            short="RS",
            purpose="Experience fit",
            metric="Resume relevance",
            question="Walk me through a project from your resume where your personal contribution changed the outcome.",
            keywords=["led", "owned", "impact", "measured", "project", "reduced", "built"],
        ),
        sample=(
            "I led a customer support automation project that reduced average response time by 34%. I owned the "
            "architecture, coordinated with product and support, shipped the first version in six weeks, and measured "
            "success through ticket deflection and CSAT."
        ),
        question_variants=(
            "Walk me through a project from your resume where your personal contribution changed the outcome.",
            "Which resume project best represents the role you want next, and what measurable result did you drive?",
            "Pick one resume bullet and explain the context, your decisions, and the impact behind it.",
        ),
    ),
    InterviewRubric(
        agent=Agent(
            name="Coding Agent",
            short="CD",
            purpose="Problem solving",
            metric="Coding signal",
            question="How would you debug a production API that suddenly became slow?",
            keywords=["metrics", "traces", "database", "deploy", "reproduce", "rollback", "test"],
        ),
        sample=(
            "I would start with metrics and traces to isolate whether latency is from the app, database, network, or a "
            "dependency. Then I would compare recent deploys, inspect slow queries, reproduce with production-like "
            "inputs, mitigate with rollback or scaling, and write a regression test."
        ),
        question_variants=(
            "How would you debug a production API that suddenly became slow?",
            "Describe how you would design tests for a bug that only appears under high traffic.",
            "What tradeoffs would you consider when refactoring a critical service with no downtime?",
        ),
    ),
    InterviewRubric(
        agent=Agent(
            name="System Design Agent",
            short="SD",
            purpose="Architecture",
            metric="Design depth",
            question="Design a scalable interview scheduling system for recruiters and candidates.",
            keywords=["model", "api", "queue", "locking", "timezone", "scale", "audit"],
        ),
        sample=(
            "I would model availability, bookings, time zones, interviewers, and candidate preferences. The API would "
            "use optimistic locking to prevent double booking, a queue for notifications, calendar provider "
            "integrations, and audit logs for reschedules."
        ),
        question_variants=(
            "Design a scalable interview scheduling system for recruiters and candidates.",
            "Design a notification system that reliably handles reminders, cancellations, and retries.",
            "Design the data model and APIs for a collaborative candidate evaluation platform.",
        ),
    ),
    InterviewRubric(
        agent=Agent(
            name="HR Agent",
            short="HR",
            purpose="Behavioral fit",
            metric="Collaboration",
            question="Describe a time you handled disagreement with a teammate.",
            keywords=["disagreed", "risk", "shared", "agreed", "trust", "delivery", "teammate"],
        ),
        sample=(
            "A teammate and I disagreed about prioritizing refactor work before a launch. I asked them to define the "
            "risk, shared the launch constraint, and we agreed on a smaller cleanup plus a follow-up task. It kept "
            "trust intact and still protected delivery."
        ),
        question_variants=(
            "Describe a time you handled disagreement with a teammate.",
            "Tell me about a time you received difficult feedback and changed your approach.",
            "Give an example of how you kept a project moving when priorities were unclear.",
        ),
    ),
)

RESUME_RUBRIC = RUBRICS[0]
HR_RUBRIC = RUBRICS[3]


def technical_rubric(
    *,
    name: str,
    short: str,
    purpose: str,
    metric: str,
    question: str,
    keywords: list[str],
    sample: str,
    variants: tuple[str, ...],
) -> InterviewRubric:
    return InterviewRubric(
        agent=Agent(
            name=name,
            short=short,
            purpose=purpose,
            metric=metric,
            question=question,
            keywords=keywords,
        ),
        sample=sample,
        question_variants=variants,
    )


TRACKS: dict[TrackId, InterviewTrack] = {
    "backend": InterviewTrack(
        id="backend",
        name="Backend Engineering",
        description="APIs, data modeling, service reliability, performance, and scalable backend architecture.",
        skills=("APIs", "Databases", "Caching", "Queues", "Reliability"),
        technical_rubrics=(
            technical_rubric(
                name="Backend Agent",
                short="BE",
                purpose="Services and data",
                metric="Backend engineering",
                question="Design an idempotent order-creation API that remains correct during retries and partial failures.",
                keywords=["idempotency", "transaction", "database", "retry", "api", "validation", "locking"],
                sample="I would accept an idempotency key, persist it with the order in one transaction, enforce uniqueness, and return the stored result on retries. I would define validation, timeout, and failure semantics explicitly.",
                variants=(
                    "Design an idempotent order-creation API that remains correct during retries and partial failures.",
                    "How would you model and expose a multi-tenant permissions API without leaking data between tenants?",
                    "A read-heavy API is missing its latency SLO. How would you diagnose and improve it safely?",
                ),
            ),
            technical_rubric(
                name="Backend Scale Agent",
                short="BS",
                purpose="Scale and resilience",
                metric="Backend scalability",
                question="Design a reliable event-processing service that handles duplicates, retries, and poison messages.",
                keywords=["queue", "deduplication", "retry", "dead-letter", "ordering", "observability", "backpressure"],
                sample="I would use at-least-once delivery with idempotent consumers, bounded retries, a dead-letter queue, partition-aware ordering, backpressure, and metrics for lag, failures, and replay operations.",
                variants=(
                    "Design a reliable event-processing service that handles duplicates, retries, and poison messages.",
                    "How would you migrate a large database table with no downtime and a safe rollback path?",
                    "Design a caching strategy for a high-traffic catalog while controlling staleness and invalidation risk.",
                ),
            ),
        ),
    ),
    "frontend": InterviewTrack(
        id="frontend",
        name="Frontend Engineering",
        description="Accessible UI, state management, browser performance, testing, and frontend architecture.",
        skills=("React", "Accessibility", "State", "Performance", "Testing"),
        technical_rubrics=(
            technical_rubric(
                name="Frontend Agent",
                short="FE",
                purpose="UI engineering",
                metric="Frontend engineering",
                question="Build an accessible autocomplete that handles latency, stale responses, and keyboard navigation.",
                keywords=["accessibility", "keyboard", "debounce", "abort", "state", "loading", "testing"],
                sample="I would use the combobox ARIA pattern, support arrow and escape keys, debounce input, cancel stale requests, announce loading and errors, and test keyboard, screen-reader, and race-condition behavior.",
                variants=(
                    "Build an accessible autocomplete that handles latency, stale responses, and keyboard navigation.",
                    "How would you structure state for a complex form with validation, autosave, and server conflicts?",
                    "A React page rerenders excessively. How would you measure, isolate, and fix the cause?",
                ),
            ),
            technical_rubric(
                name="Frontend Architecture Agent",
                short="FA",
                purpose="Web architecture",
                metric="Frontend architecture",
                question="Design the frontend architecture for a multi-team analytics dashboard with shared components.",
                keywords=["boundaries", "design-system", "routing", "data", "performance", "testing", "deployment"],
                sample="I would define domain boundaries, a versioned design system, typed API contracts, route-level code splitting, a consistent data-fetching layer, visual and integration tests, and independent ownership rules.",
                variants=(
                    "Design the frontend architecture for a multi-team analytics dashboard with shared components.",
                    "How would you improve Core Web Vitals for a content-heavy application without harming functionality?",
                    "Design an offline-capable web workflow that resolves edits made across multiple devices.",
                ),
            ),
        ),
    ),
    "cloud": InterviewTrack(
        id="cloud",
        name="Cloud Engineering",
        description="Cloud architecture, networking, security, reliability, observability, and cost control.",
        skills=("Networking", "IAM", "Containers", "Observability", "Cost"),
        technical_rubrics=(
            technical_rubric(
                name="Cloud Agent",
                short="CL",
                purpose="Cloud architecture",
                metric="Cloud architecture",
                question="Design a secure multi-account cloud platform for several product teams.",
                keywords=["accounts", "iam", "network", "guardrails", "logging", "secrets", "cost"],
                sample="I would separate workloads by account and environment, centralize identity and audit logs, define network boundaries, enforce policy guardrails, manage secrets centrally, and allocate cost with tags and budgets.",
                variants=(
                    "Design a secure multi-account cloud platform for several product teams.",
                    "How would you connect private workloads across regions while limiting blast radius?",
                    "Design a cloud landing zone that balances team autonomy with security and cost controls.",
                ),
            ),
            technical_rubric(
                name="Cloud Reliability Agent",
                short="CR",
                purpose="Operations and resilience",
                metric="Cloud reliability",
                question="A regional cloud outage affects a critical service. Explain your failover and recovery design.",
                keywords=["rto", "rpo", "failover", "replication", "dns", "runbook", "testing"],
                sample="I would define RTO and RPO first, choose replication consistency accordingly, automate health-based failover, protect against split brain, maintain tested runbooks, and run recovery exercises regularly.",
                variants=(
                    "A regional cloud outage affects a critical service. Explain your failover and recovery design.",
                    "How would you build observability for a container platform used by dozens of services?",
                    "Cloud spend increased by 40 percent. How would you find savings without creating reliability risk?",
                ),
            ),
        ),
    ),
    "terraform": InterviewTrack(
        id="terraform",
        name="Terraform / IaC",
        description="Terraform modules, state, providers, delivery pipelines, policy, drift, and safe infrastructure change.",
        skills=("Terraform", "State", "Modules", "Policy", "Drift"),
        technical_rubrics=(
            technical_rubric(
                name="Terraform Agent",
                short="TF",
                purpose="Infrastructure as code",
                metric="Terraform engineering",
                question="Design reusable Terraform modules for networking across multiple environments and accounts.",
                keywords=["module", "state", "provider", "version", "validation", "output", "composition"],
                sample="I would keep modules small and composable, pin providers, validate inputs, expose stable outputs, separate state by environment and blast radius, and version modules with migration guidance.",
                variants=(
                    "Design reusable Terraform modules for networking across multiple environments and accounts.",
                    "How would you import existing cloud resources into Terraform without causing destructive changes?",
                    "Explain how you would structure Terraform state for many teams and environments.",
                ),
            ),
            technical_rubric(
                name="IaC Delivery Agent",
                short="IC",
                purpose="Safe infrastructure delivery",
                metric="Infrastructure delivery",
                question="Design a Terraform CI/CD workflow with review, policy checks, apply controls, and drift detection.",
                keywords=["plan", "apply", "approval", "policy", "drift", "locking", "rollback"],
                sample="Pull requests would run formatting, validation, security and policy checks, then publish a saved plan. Protected apply jobs would require approval, use state locking, record audit data, and schedule drift detection.",
                variants=(
                    "Design a Terraform CI/CD workflow with review, policy checks, apply controls, and drift detection.",
                    "A Terraform apply partially fails. How do you recover while preserving state integrity?",
                    "How would you roll out a breaking provider upgrade across hundreds of workspaces?",
                ),
            ),
        ),
    ),
    "devops": InterviewTrack(
        id="devops",
        name="DevOps / SRE",
        description="Delivery pipelines, Kubernetes, incident response, SLOs, automation, and operational reliability.",
        skills=("CI/CD", "Kubernetes", "SLOs", "Incidents", "Automation"),
        technical_rubrics=(
            technical_rubric(
                name="DevOps Agent",
                short="DO",
                purpose="Delivery automation",
                metric="DevOps engineering",
                question="Design a deployment pipeline for many services with fast feedback and safe production releases.",
                keywords=["pipeline", "test", "artifact", "canary", "rollback", "approval", "security"],
                sample="I would build immutable artifacts once, run layered tests and security scans, promote the same artifact, use canary releases with automated SLO checks, and support fast rollback with audited approvals.",
                variants=(
                    "Design a deployment pipeline for many services with fast feedback and safe production releases.",
                    "How would you standardize Kubernetes delivery without blocking teams that need customization?",
                    "A release causes elevated errors. Walk through automated detection, mitigation, and learning.",
                ),
            ),
            technical_rubric(
                name="SRE Agent",
                short="SR",
                purpose="Reliability engineering",
                metric="Site reliability",
                question="Define SLOs and an error-budget policy for a customer-facing API.",
                keywords=["sli", "slo", "error-budget", "latency", "availability", "alert", "tradeoff"],
                sample="I would select user-centered availability and latency SLIs, define targets from business needs, alert on burn rate, and use the error budget to balance feature delivery with reliability work.",
                variants=(
                    "Define SLOs and an error-budget policy for a customer-facing API.",
                    "How would you lead incident response for a cascading production failure?",
                    "Design capacity planning and autoscaling for a service with sharp seasonal traffic spikes.",
                ),
            ),
        ),
    ),
    "ai-ml": InterviewTrack(
        id="ai-ml",
        name="AI / ML Engineering",
        description="Production AI systems, RAG, model evaluation, safety, experimentation, and ML fundamentals.",
        skills=("RAG", "Evaluation", "Models", "Guardrails", "Experimentation"),
        technical_rubrics=(
            technical_rubric(
                name="AI Engineering Agent", short="AI", purpose="Production AI systems", metric="AI engineering",
                question="Design a production RAG assistant that provides grounded answers with citations and safe failure behavior.",
                keywords=["retrieval", "embedding", "chunking", "citation", "evaluation", "guardrail", "latency"],
                sample="I would build permission-aware retrieval, rerank relevant chunks, require citations, evaluate groundedness and recall, add refusal guardrails, and monitor latency, cost, and unsupported claims.",
                variants=(
                    "Design a production RAG assistant that provides grounded answers with citations and safe failure behavior.",
                    "How would you build a reliable agent that can call tools without causing unsafe side effects?",
                    "Design an AI support workflow that balances quality, latency, cost, and human escalation.",
                ),
            ),
            technical_rubric(
                name="Model Evaluation Agent", short="ME", purpose="Model quality and experiments", metric="Model evaluation",
                question="Design an evaluation framework for deciding whether a new model or prompt is safe to release.",
                keywords=["dataset", "baseline", "metric", "regression", "human", "bias", "experiment"],
                sample="I would version representative and adversarial datasets, compare against a baseline on task and safety metrics, review slices and failures, use blinded human judgment where needed, and gate releases on explicit thresholds.",
                variants=(
                    "Design an evaluation framework for deciding whether a new model or prompt is safe to release.",
                    "An ML model performs well offline but poorly in production. How would you investigate the gap?",
                    "How would you detect and reduce bias in a model used for a high-impact workflow?",
                ),
            ),
        ),
    ),
    "data-engineering": InterviewTrack(
        id="data-engineering",
        name="Data Engineering",
        description="Batch and streaming pipelines, data modeling, quality, orchestration, and analytics platforms.",
        skills=("Pipelines", "Streaming", "Warehouses", "Data Quality", "Orchestration"),
        technical_rubrics=(
            technical_rubric(
                name="Data Pipeline Agent", short="DP", purpose="Reliable data movement", metric="Data pipelines",
                question="Design a pipeline that combines batch and streaming events while remaining correct during retries.",
                keywords=["stream", "batch", "idempotency", "checkpoint", "schema", "late-data", "observability"],
                sample="I would define event contracts, use durable checkpoints and idempotent sinks, handle late data with watermarks, quarantine schema failures, reconcile outputs, and monitor freshness and completeness.",
                variants=(
                    "Design a pipeline that combines batch and streaming events while remaining correct during retries.",
                    "How would you backfill a year of data without disrupting current production pipelines?",
                    "Design data-quality controls for a platform consumed by finance and product teams.",
                ),
            ),
            technical_rubric(
                name="Analytics Modeling Agent", short="AM", purpose="Analytics data models", metric="Data modeling",
                question="Design a warehouse model for product metrics that supports history and changing dimensions.",
                keywords=["fact", "dimension", "grain", "history", "partition", "lineage", "semantic"],
                sample="I would declare fact-table grain, use conformed dimensions and explicit history rules, partition for common access, document lineage, test metric definitions, and expose a governed semantic layer.",
                variants=(
                    "Design a warehouse model for product metrics that supports history and changing dimensions.",
                    "How would you evolve a shared event schema without breaking downstream consumers?",
                    "Design a lakehouse layout for governed analytics and exploratory workloads.",
                ),
            ),
        ),
    ),
    "cybersecurity": InterviewTrack(
        id="cybersecurity",
        name="Cybersecurity / AppSec",
        description="Threat modeling, secure design, application security, vulnerability management, and incident response.",
        skills=("Threat Modeling", "AppSec", "OWASP", "Response", "Risk"),
        technical_rubrics=(
            technical_rubric(
                name="Application Security Agent", short="AS", purpose="Secure application design", metric="Application security",
                question="Threat-model a multi-tenant API and explain how you would prevent cross-tenant data access.",
                keywords=["asset", "trust-boundary", "authorization", "tenant", "validation", "logging", "test"],
                sample="I would map assets and trust boundaries, enforce tenant context server-side on every query, use deny-by-default authorization, validate inputs, log access, and add negative authorization tests.",
                variants=(
                    "Threat-model a multi-tenant API and explain how you would prevent cross-tenant data access.",
                    "How would you design secure authentication and session management for a public web application?",
                    "A critical dependency vulnerability is disclosed. How would you assess and remediate the risk?",
                ),
            ),
            technical_rubric(
                name="Security Response Agent", short="SC", purpose="Security incidents", metric="Incident response",
                question="A production credential has leaked publicly. Walk through containment, investigation, and recovery.",
                keywords=["revoke", "rotate", "contain", "audit", "scope", "evidence", "postmortem"],
                sample="I would revoke and rotate the credential immediately, preserve evidence, search audit logs for use, scope affected resources and data, contain related access, communicate by severity, and close detection gaps afterward.",
                variants=(
                    "A production credential has leaked publicly. Walk through containment, investigation, and recovery.",
                    "How would you prioritize vulnerabilities across thousands of assets?",
                    "Design a security review process that fits a fast software delivery lifecycle.",
                ),
            ),
        ),
    ),
    "kubernetes-platform": InterviewTrack(
        id="kubernetes-platform",
        name="Kubernetes / Platform Engineering",
        description="Kubernetes architecture, developer platforms, multi-tenancy, networking, operations, and golden paths.",
        skills=("Kubernetes", "Platforms", "Multi-tenancy", "Networking", "GitOps"),
        technical_rubrics=(
            technical_rubric(
                name="Kubernetes Engineering Agent", short="KE", purpose="Cluster architecture", metric="Kubernetes engineering",
                question="Design a secure multi-tenant Kubernetes platform for many product teams.",
                keywords=["namespace", "rbac", "network-policy", "quota", "admission", "upgrade", "observability"],
                sample="I would define tenancy boundaries, least-privilege RBAC, default-deny network policies, quotas, admission controls, workload identity, centralized observability, and tested upgrade and recovery procedures.",
                variants=(
                    "Design a secure multi-tenant Kubernetes platform for many product teams.",
                    "A Kubernetes cluster has intermittent networking failures. How would you diagnose them?",
                    "How would you upgrade clusters and workloads with minimal risk and downtime?",
                ),
            ),
            technical_rubric(
                name="Developer Platform Agent", short="PF", purpose="Internal developer platforms", metric="Platform engineering",
                question="Design a self-service golden path that speeds delivery without removing necessary team flexibility.",
                keywords=["self-service", "template", "catalog", "guardrail", "ownership", "feedback", "adoption"],
                sample="I would start from developer pain, provide versioned templates and paved workflows, embed security and observability defaults, allow documented escape hatches, measure adoption and lead time, and iterate with users.",
                variants=(
                    "Design a self-service golden path that speeds delivery without removing necessary team flexibility.",
                    "How would you measure whether an internal developer platform is succeeding?",
                    "Design a service catalog that keeps ownership and operational metadata trustworthy.",
                ),
            ),
        ),
    ),
    "mobile": InterviewTrack(
        id="mobile",
        name="Mobile Engineering",
        description="iOS and Android architecture, offline data, performance, releases, testing, and mobile reliability.",
        skills=("iOS / Android", "Offline", "Performance", "Releases", "Testing"),
        technical_rubrics=(
            technical_rubric(
                name="Mobile Architecture Agent", short="MA", purpose="Mobile application design", metric="Mobile engineering",
                question="Design an offline-first mobile workflow that synchronizes edits across multiple devices.",
                keywords=["cache", "sync", "conflict", "queue", "retry", "state", "encryption"],
                sample="I would persist an encrypted local model, queue idempotent mutations, track server versions, choose explicit conflict rules, retry with backoff, surface unresolved conflicts, and test reconnect scenarios.",
                variants=(
                    "Design an offline-first mobile workflow that synchronizes edits across multiple devices.",
                    "How would you structure a large mobile application for independent feature teams?",
                    "Design secure storage and authentication for a mobile banking application.",
                ),
            ),
            technical_rubric(
                name="Mobile Reliability Agent", short="MR", purpose="Mobile quality and delivery", metric="Mobile reliability",
                question="A mobile release increases crashes and startup time. How would you detect, contain, and fix it?",
                keywords=["crash", "startup", "telemetry", "rollout", "rollback", "profiling", "device"],
                sample="I would segment crash and startup telemetry by version and device, halt the staged rollout, use symbolicated traces and profiles, ship a focused fix, validate affected devices, and strengthen release gates.",
                variants=(
                    "A mobile release increases crashes and startup time. How would you detect, contain, and fix it?",
                    "How would you design a staged mobile release when app-store rollback is limited?",
                    "Build a testing strategy for a mobile app across devices, OS versions, and unreliable networks.",
                ),
            ),
        ),
    ),
    "qa-sdet": InterviewTrack(
        id="qa-sdet",
        name="QA Automation / SDET",
        description="Test strategy, automation frameworks, reliability, performance testing, and quality engineering.",
        skills=("Automation", "Test Design", "Performance", "CI", "Quality"),
        technical_rubrics=(
            technical_rubric(
                name="Test Strategy Agent", short="TS", purpose="Quality architecture", metric="Test strategy",
                question="Design a layered test strategy for a distributed web application with frequent releases.",
                keywords=["unit", "integration", "contract", "e2e", "risk", "environment", "feedback"],
                sample="I would map tests to product risk, keep most checks at unit and integration layers, use contract tests at service boundaries, reserve E2E for critical journeys, control test data, and track speed and defect escape.",
                variants=(
                    "Design a layered test strategy for a distributed web application with frequent releases.",
                    "How would you test a payment workflow for correctness under retries and partial failures?",
                    "Design a performance-testing program for an API with unpredictable traffic peaks.",
                ),
            ),
            technical_rubric(
                name="Automation Reliability Agent", short="AR", purpose="Reliable test automation", metric="Automation engineering",
                question="An end-to-end suite is slow and flaky. Explain how you would make it trustworthy.",
                keywords=["flaky", "isolation", "deterministic", "data", "parallel", "retry", "ownership"],
                sample="I would measure failures by cause, remove shared state and timing assumptions, create deterministic data, improve diagnostics, parallelize isolated tests, quarantine only with owners and deadlines, and avoid hiding failures with retries.",
                variants=(
                    "An end-to-end suite is slow and flaky. Explain how you would make it trustworthy.",
                    "How would you design a maintainable API automation framework for multiple teams?",
                    "A defect escaped despite passing tests. How would you improve the quality system?",
                ),
            ),
        ),
    ),
    "full-stack": InterviewTrack(
        id="full-stack",
        name="Full-Stack Engineering",
        description="End-to-end product delivery across user interfaces, APIs, data, security, and operations.",
        skills=("Frontend", "Backend", "Databases", "Security", "Delivery"),
        technical_rubrics=(
            technical_rubric(
                name="Full-Stack Delivery Agent", short="FS", purpose="End-to-end product delivery", metric="Full-stack engineering",
                question="Design and deliver a team-invitations feature across UI, API, database, and authorization.",
                keywords=["ui", "api", "schema", "authorization", "validation", "transaction", "test"],
                sample="I would define user states and the API contract, model expiring invitations with unique constraints, enforce team authorization server-side, make acceptance transactional and idempotent, and test the critical flow across layers.",
                variants=(
                    "Design and deliver a team-invitations feature across UI, API, database, and authorization.",
                    "How would you build a searchable activity feed from browser interaction through storage?",
                    "Design a file-upload feature with progress, validation, secure storage, and failure recovery.",
                ),
            ),
            technical_rubric(
                name="System Integration Agent", short="SI", purpose="Cross-layer integration", metric="System integration",
                question="Design error handling and observability across a browser, API, queue, and worker workflow.",
                keywords=["error", "trace", "correlation", "retry", "idempotency", "status", "monitoring"],
                sample="I would define stable error contracts, propagate correlation IDs, make asynchronous work idempotent, expose durable job status, bound retries, trace each boundary, and alert on user-impacting failures.",
                variants=(
                    "Design error handling and observability across a browser, API, queue, and worker workflow.",
                    "How would you migrate an end-to-end feature without breaking old clients?",
                    "A feature is fast locally but slow in production. How would you diagnose it across the stack?",
                ),
            ),
        ),
    ),
    "database": InterviewTrack(
        id="database",
        name="Database Engineering",
        description="Schema design, query performance, transactions, replication, migrations, backup, and recovery.",
        skills=("SQL", "Modeling", "Transactions", "Replication", "Recovery"),
        technical_rubrics=(
            technical_rubric(
                name="Database Design Agent", short="DB", purpose="Database architecture", metric="Database design",
                question="Design the database for a high-write financial ledger that must preserve an audit trail.",
                keywords=["ledger", "transaction", "constraint", "idempotency", "index", "partition", "audit"],
                sample="I would use immutable double-entry records, transactional balance invariants, idempotency keys and unique constraints, indexes for access paths, controlled partitioning, and reconciliation instead of mutating history.",
                variants=(
                    "Design the database for a high-write financial ledger that must preserve an audit trail.",
                    "How would you model inventory reservations while preventing overselling?",
                    "A critical SQL query becomes slow as data grows. Walk through diagnosis and tuning.",
                ),
            ),
            technical_rubric(
                name="Database Reliability Agent", short="DR", purpose="Database operations", metric="Database reliability",
                question="Plan a zero-downtime schema migration for a very large production table.",
                keywords=["migration", "expand-contract", "backfill", "lock", "replica", "rollback", "backup"],
                sample="I would use expand-contract changes, avoid long locks, backfill in throttled resumable batches, validate replicas and lag, provide rollback gates, and test recovery from backups.",
                variants=(
                    "Plan a zero-downtime schema migration for a very large production table.",
                    "Design replication and failover for a database with strict recovery objectives.",
                    "How would you prove that backups are complete and actually recoverable?",
                ),
            ),
        ),
    ),
    "mlops": InterviewTrack(
        id="mlops",
        name="MLOps",
        description="Reproducible ML pipelines, model registries, deployment, monitoring, drift, and governance.",
        skills=("ML Pipelines", "Registry", "Deployment", "Drift", "Governance"),
        technical_rubrics=(
            technical_rubric(
                name="MLOps Pipeline Agent", short="MP", purpose="ML delivery pipelines", metric="MLOps delivery",
                question="Design a reproducible pipeline from training data through model registration and deployment.",
                keywords=["version", "lineage", "feature", "experiment", "registry", "approval", "reproducible"],
                sample="I would version code, data, features, environments, and parameters; record lineage and evaluations; register only qualified artifacts; require approval; and promote the same immutable model across environments.",
                variants=(
                    "Design a reproducible pipeline from training data through model registration and deployment.",
                    "How would you prevent training-serving skew in an online prediction system?",
                    "Design a feature platform that supports reuse without leaking future information.",
                ),
            ),
            technical_rubric(
                name="Model Operations Agent", short="MO", purpose="Production model operations", metric="Model operations",
                question="Design monitoring and rollback for a model whose data distribution changes over time.",
                keywords=["drift", "quality", "shadow", "canary", "rollback", "baseline", "alert"],
                sample="I would monitor input and prediction drift alongside delayed outcome quality, compare against baselines, shadow and canary new versions, define business-aware alerts, preserve instant rollback, and trigger reviewed retraining.",
                variants=(
                    "Design monitoring and rollback for a model whose data distribution changes over time.",
                    "A newly deployed model degrades a business metric. How would you investigate and respond?",
                    "How would you govern model approvals and evidence for a regulated use case?",
                ),
            ),
        ),
    ),
    "devsecops": InterviewTrack(
        id="devsecops",
        name="DevSecOps / Security Operations",
        description="Secure delivery, software supply chain, cloud controls, detection, secrets, and operational response.",
        skills=("Supply Chain", "Cloud Security", "Secrets", "Detection", "Policy"),
        technical_rubrics=(
            technical_rubric(
                name="Secure Delivery Agent", short="SD", purpose="Software supply-chain security", metric="DevSecOps delivery",
                question="Design a secure software-delivery pipeline from source commit to production artifact.",
                keywords=["identity", "artifact", "signing", "sbom", "scan", "provenance", "policy"],
                sample="I would protect source and CI identities, build immutable artifacts in isolated runners, generate an SBOM and provenance, scan and sign artifacts, enforce risk-based policies, and verify signatures at deployment.",
                variants=(
                    "Design a secure software-delivery pipeline from source commit to production artifact.",
                    "How would you introduce security gates without making teams bypass the delivery process?",
                    "A build dependency is compromised. How would you identify exposure and recover?",
                ),
            ),
            technical_rubric(
                name="Cloud Security Operations Agent", short="SO", purpose="Cloud security operations", metric="Security operations",
                question="Design IAM, secrets, policy, and detection controls for a cloud-native production environment.",
                keywords=["least-privilege", "workload-identity", "secret", "policy", "audit", "detection", "response"],
                sample="I would use short-lived workload identity, least-privilege roles, centralized secret rotation, policy-as-code, immutable audit logs, detections for risky behavior, and rehearsed response playbooks.",
                variants=(
                    "Design IAM, secrets, policy, and detection controls for a cloud-native production environment.",
                    "How would you detect and contain suspicious activity in a production cloud account?",
                    "Design secrets rotation for many services without causing an outage.",
                ),
            ),
        ),
    ),
    "solutions-architecture": InterviewTrack(
        id="solutions-architecture",
        name="Solutions Architecture",
        description="Requirements discovery, system tradeoffs, integration design, migrations, governance, and communication.",
        skills=("Discovery", "Architecture", "Integration", "Migration", "Tradeoffs"),
        technical_rubrics=(
            technical_rubric(
                name="Solution Design Agent", short="SA", purpose="Constraint-driven architecture", metric="Solutions architecture",
                question="Design a phased modernization plan for a critical legacy application with strict uptime constraints.",
                keywords=["requirement", "constraint", "migration", "strangler", "risk", "cost", "stakeholder"],
                sample="I would clarify business outcomes and constraints, map dependencies, introduce a strangler boundary, migrate low-risk capabilities first, define data and rollback strategies, measure value, and communicate cost and risk.",
                variants=(
                    "Design a phased modernization plan for a critical legacy application with strict uptime constraints.",
                    "How would you choose between build, buy, and managed services for a strategic capability?",
                    "Design a multi-region customer platform while explaining cost and consistency tradeoffs.",
                ),
            ),
            technical_rubric(
                name="Integration Architecture Agent", short="IA", purpose="Enterprise integrations", metric="Integration architecture",
                question="Design an integration platform for internal services and external partners with different reliability levels.",
                keywords=["contract", "api", "event", "version", "identity", "retry", "governance"],
                sample="I would separate synchronous and event-driven contracts, version schemas, authenticate each partner, use idempotency and bounded retries, isolate failures, publish ownership and SLOs, and test compatibility.",
                variants=(
                    "Design an integration platform for internal services and external partners with different reliability levels.",
                    "How would you translate ambiguous stakeholder needs into an architecture decision?",
                    "A proposed architecture exceeds the customer's budget. How would you redesign and communicate tradeoffs?",
                ),
            ),
        ),
    ),
    "engineering-management": InterviewTrack(
        id="engineering-management",
        name="Engineering Management",
        description="People leadership, delivery, technical strategy, team health, stakeholder alignment, and execution.",
        skills=("Leadership", "Delivery", "Coaching", "Strategy", "Stakeholders"),
        technical_rubrics=(
            technical_rubric(
                name="Engineering Leadership Agent", short="EL", purpose="People and delivery leadership", metric="Engineering management",
                question="A capable team is repeatedly missing commitments and morale is falling. How would you respond?",
                keywords=["diagnose", "clarity", "capacity", "priority", "coach", "metric", "trust"],
                sample="I would gather evidence and listen to the team, clarify outcomes and ownership, reduce competing priorities, address capability or process gaps through coaching, reset commitments, and track delivery and team-health signals.",
                variants=(
                    "A capable team is repeatedly missing commitments and morale is falling. How would you respond?",
                    "How would you handle sustained underperformance while treating the engineer fairly?",
                    "Two senior engineers strongly disagree on a critical decision. How would you lead resolution?",
                ),
            ),
            technical_rubric(
                name="Engineering Strategy Agent", short="ES", purpose="Technical strategy and alignment", metric="Engineering strategy",
                question="Build an engineering investment plan that balances roadmap delivery, reliability, and technical debt.",
                keywords=["outcome", "roadmap", "risk", "debt", "capacity", "stakeholder", "measure"],
                sample="I would connect investments to business outcomes and operational risk, quantify debt, reserve capacity by explicit policy, sequence enabling work with roadmap value, agree tradeoffs, and measure results quarterly.",
                variants=(
                    "Build an engineering investment plan that balances roadmap delivery, reliability, and technical debt.",
                    "How would you reorganize team ownership as a product and organization grow?",
                    "Leadership asks for an unrealistic deadline. How would you create and communicate options?",
                ),
            ),
        ),
    ),
    "system-design": InterviewTrack(
        id="system-design",
        name="System Design",
        description="Scalable architecture, distributed systems, data consistency, messaging, and failure handling.",
        skills=("Architecture", "Scaling", "Consistency", "Messaging", "Resilience"),
        technical_rubrics=(RUBRICS[1], RUBRICS[2]),
    ),
}

SESSION_STORE: dict[str, InterviewSession] = {}
SESSION_STORE_LOCK = RLock()


app = FastAPI(title="AI Interviewer Platform", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def request_rate_limit(method: str, path: str) -> tuple[str, int, int] | None:
    if method != "POST":
        return None
    if path == "/api/sessions":
        return "session-create", 10, 60
    if path.startswith("/api/sessions/") and path.endswith("/answer"):
        return "session-answer", 30, 60
    return None


@app.middleware("http")
async def rate_limit_genai_routes(request: Request, call_next):
    limit = request_rate_limit(request.method, request.url.path)
    if limit:
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body must not exceed {MAX_REQUEST_BYTES} bytes."},
            )

    if not limit or RATE_LIMIT_MODE != "memory":
        return await call_next(request)

    route_key, max_requests, window_seconds = limit
    client_ip = request.client.host if request.client else "unknown"
    bucket = RATE_BUCKETS[(client_ip, route_key)]
    now = time.monotonic()

    while bucket and now - bucket[0] > window_seconds:
        bucket.popleft()

    if len(bucket) >= max_requests:
        retry_after = max(1, int(window_seconds - (now - bucket[0])) + 1)
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please wait and try again."},
            headers={"Retry-After": str(retry_after)},
        )

    bucket.append(now)
    return await call_next(request)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mode": "fallback" if OPENAI_DISABLED or not OPENAI_API_KEY else "openai",
        "model": OPENAI_MODEL,
        "reasoningEffort": OPENAI_REASONING_EFFORT or "model-default",
        "keySource": OPENAI_KEY_SOURCE,
        "rateLimitMode": RATE_LIMIT_MODE,
        "sessionStore": "server-memory",
    }


@app.get("/api/tracks", response_model=list[TrackView])
def list_interview_tracks() -> list[TrackView]:
    return [track_view(track) for track in TRACKS.values()]


@app.post("/api/sessions", response_model=SessionView, status_code=201)
def start_interview(payload: SessionCreateRequest) -> SessionView:
    session = create_interview_session(payload.trackId)
    return session_view(session)


@app.get("/api/sessions/{session_id}", response_model=SessionView)
def read_interview(session_id: str) -> SessionView:
    session = get_interview_session(session_id)
    with session.lock:
        session.updated_at = time.monotonic()
        return session_view(session)


@app.post("/api/sessions/{session_id}/answer", response_model=SessionView)
def submit_interview_answer(session_id: str, payload: AnswerSubmission) -> SessionView:
    session = get_interview_session(session_id)
    with session.lock:
        if session.phase == "complete":
            raise HTTPException(status_code=409, detail="This interview is already complete.")
        if payload.version != session.version or payload.phase != session.phase:
            raise HTTPException(status_code=409, detail="This interview step is stale. Refresh before continuing.")
        if session_context_size(session) + len(payload.answer) > MAX_CONTEXT_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"Combined interview answers must not exceed {MAX_CONTEXT_CHARS} characters.",
            )

        stage = session.stages[session.active_index]
        if session.phase == "primary":
            if stage.answer:
                raise HTTPException(status_code=409, detail="The primary answer was already submitted.")
            stage.answer = payload.answer
            primary_result = evaluate_primary_answer(session, stage)
            stage.primary_result = primary_result
            stage.follow_up_question = primary_result.followUpQuestion
            session.phase = "follow_up"
        else:
            if not stage.answer or not stage.follow_up_question:
                raise HTTPException(status_code=409, detail="The stage is not ready for a follow-up answer.")
            if stage.follow_up_answer:
                raise HTTPException(status_code=409, detail="The follow-up answer was already submitted.")
            stage.follow_up_answer = payload.answer
            stage.result = evaluate_completed_stage(session, stage)

            if session.active_index + 1 < len(session.stages):
                session.active_index += 1
                session.phase = "primary"
                next_stage = session.stages[session.active_index]
                next_stage.question = generate_next_question(session, session.active_index)
            else:
                session.phase = "complete"
                session.final_evaluation = build_final_evaluation(session)

        session.version += 1
        session.updated_at = time.monotonic()
        return session_view(session)


def create_interview_session(track_id: TrackId) -> InterviewSession:
    now = time.monotonic()
    track = TRACKS[track_id]
    with SESSION_STORE_LOCK:
        purge_expired_sessions(now)
        while len(SESSION_STORE) >= MAX_SESSIONS:
            oldest_id = min(SESSION_STORE, key=lambda key: SESSION_STORE[key].updated_at)
            del SESSION_STORE[oldest_id]

        session_id = secrets.token_urlsafe(24)
        rubrics = (RESUME_RUBRIC, *track.technical_rubrics, HR_RUBRIC)
        stages = [StageRecord(rubric=rubric, question=rubric.question_variants[0]) for rubric in rubrics]
        session = InterviewSession(session_id=session_id, track=track, stages=stages, updated_at=now)
        SESSION_STORE[session_id] = session
        return session


def get_interview_session(session_id: str) -> InterviewSession:
    now = time.monotonic()
    with SESSION_STORE_LOCK:
        purge_expired_sessions(now)
        session = SESSION_STORE.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found or expired.")
    return session


def purge_expired_sessions(now: float | None = None) -> None:
    cutoff = (now if now is not None else time.monotonic()) - SESSION_TTL_SECONDS
    expired = [session_id for session_id, session in SESSION_STORE.items() if session.updated_at < cutoff]
    for session_id in expired:
        del SESSION_STORE[session_id]


def session_context_size(session: InterviewSession) -> int:
    return sum(len(stage.answer) + len(stage.follow_up_answer) for stage in session.stages)


def track_view(track: InterviewTrack) -> TrackView:
    return TrackView(
        id=track.id,
        name=track.name,
        description=track.description,
        skills=list(track.skills),
    )


def session_view(session: InterviewSession) -> SessionView:
    completed_results = [stage.result for stage in session.stages if stage.result]
    partial_score = (
        round(sum(result.score for result in completed_results) / len(completed_results), 1)
        if completed_results
        else 0.0
    )
    stages: list[StageView] = []
    for index, stage in enumerate(session.stages):
        if stage.result:
            status: Literal["upcoming", "primary", "follow_up", "complete"] = "complete"
        elif index == session.active_index and session.phase != "complete":
            status = session.phase
        else:
            status = "upcoming"
        stages.append(
            StageView(
                index=index,
                name=stage.rubric.agent.name,
                short=stage.rubric.agent.short,
                purpose=stage.rubric.agent.purpose,
                metric=stage.rubric.agent.metric,
                question=stage.question,
                sample=stage.rubric.sample if status == "primary" else "",
                answer=stage.answer,
                followUpQuestion=stage.follow_up_question,
                followUpAnswer=stage.follow_up_answer,
                result=stage.result,
                status=status,
            )
        )

    return SessionView(
        sessionId=session.session_id,
        track=track_view(session.track),
        version=session.version,
        phase=session.phase,
        activeIndex=session.active_index,
        totalStages=len(session.stages),
        completedStages=len(completed_results),
        partialScore=partial_score,
        stages=stages,
        finalEvaluation=session.final_evaluation,
    )


def evaluate_primary_answer(session: InterviewSession, stage: StageRecord) -> PrimaryAgentResult:
    agent = stage.rubric.agent.model_copy(update={"question": stage.question})
    fallback = local_primary_result(agent, stage.answer)
    if OPENAI_DISABLED or not OPENAI_API_KEY:
        return PrimaryAgentResult.model_validate({**fallback, "mode": "fallback"})

    system = (
        "You are an expert interview agent. Assess the candidate's primary answer against the server-provided rubric "
        "and ask exactly one concise follow-up that probes missing evidence, a tradeoff, or measurable impact. Return "
        "only the required structured fields."
    )
    user = json.dumps(
        {
            "track": session.track.name,
            "agentName": agent.name,
            "agentPurpose": agent.purpose,
            "rubricMetric": agent.metric,
            "question": stage.question,
            "answer": stage.answer,
        }
    )
    result = with_openai_fallback(
        system,
        user,
        fallback,
        schema_name="primary_stage_evaluation",
        output_model=AIAgentOutput,
    )
    return PrimaryAgentResult.model_validate(result)


def evaluate_completed_stage(session: InterviewSession, stage: StageRecord) -> AgentResult:
    agent = stage.rubric.agent.model_copy(update={"question": stage.question})
    fallback = local_scored_result(agent, f"{stage.answer}\n{stage.follow_up_answer}", expected_words=140)
    if OPENAI_DISABLED or not OPENAI_API_KEY:
        return AgentResult.model_validate({**fallback, "mode": "fallback"})

    system = (
        "You are an expert interview agent. Produce the final score for one interview stage using both the primary "
        "answer and the required follow-up answer. Score only against the server-provided metric. Return only the "
        "required structured fields, with a score from 0 to 100."
    )
    user = json.dumps(
        {
            "track": session.track.name,
            "agentName": agent.name,
            "agentPurpose": agent.purpose,
            "rubricMetric": agent.metric,
            "primaryQuestion": stage.question,
            "primaryAnswer": stage.answer,
            "followUpQuestion": stage.follow_up_question,
            "followUpAnswer": stage.follow_up_answer,
        }
    )
    result = with_openai_fallback(
        system,
        user,
        fallback,
        schema_name="completed_stage_evaluation",
        output_model=AIScoredFeedback,
    )
    return AgentResult.model_validate(result)


def generate_next_question(session: InterviewSession, stage_index: int) -> str:
    stage = session.stages[stage_index]
    previous_stages = session.stages[:stage_index]
    answers = [f"{item.answer}\nFollow-up: {item.follow_up_answer}" for item in previous_stages]
    results = [item.result for item in previous_stages if item.result]
    fallback = local_question(stage.rubric, stage_index, answers, results)
    if OPENAI_DISABLED or not OPENAI_API_KEY:
        return fallback["question"]

    system = (
        "You are an adaptive AI interviewer. Generate one concise question for the current server-owned interview "
        "stage. Use prior completed answers and validated scores to adapt difficulty and avoid repetition. Return only "
        "the required structured fields."
    )
    user = json.dumps(
        {
            "track": session.track.name,
            "agent": {
                "name": stage.rubric.agent.name,
                "purpose": stage.rubric.agent.purpose,
                "metric": stage.rubric.agent.metric,
            },
            "stageIndex": stage_index,
            "previousAnswers": answers,
            "previousResults": [result.model_dump() for result in results],
        }
    )
    result = with_openai_fallback(
        system,
        user,
        fallback,
        schema_name="interview_question",
        output_model=AIQuestionOutput,
    )
    return AIQuestionOutput.model_validate({key: result[key] for key in ("question", "rationale")}).question


def build_final_evaluation(session: InterviewSession) -> EvaluationResult:
    completed = [(stage.rubric.agent, stage.result) for stage in session.stages if stage.result]
    if len(completed) != len(session.stages):
        raise RuntimeError("Cannot finalize an incomplete interview")

    scorecard = [ScorecardItem(metric=agent.metric, score=result.score) for agent, result in completed]
    overall_score = round(sum(item.score for item in scorecard) / len(scorecard), 1)
    fallback = local_final_narrative([result for _, result in completed], overall_score)

    if OPENAI_DISABLED or not OPENAI_API_KEY:
        narrative = {**fallback, "mode": "fallback"}
    else:
        system = (
            "You are the final hiring evaluation agent. Write the narrative synthesis from server-validated stage "
            "results. Do not invent or alter scores. Return only strengths, weaknesses, hiringRecommendation, and extra."
        )
        user = json.dumps(
            {
                "track": session.track.name,
                "overallScore": overall_score,
                "stages": [
                    {
                        "name": stage.rubric.agent.name,
                        "metric": stage.rubric.agent.metric,
                        "question": stage.question,
                        "answer": stage.answer,
                        "followUpQuestion": stage.follow_up_question,
                        "followUpAnswer": stage.follow_up_answer,
                        "validatedResult": stage.result.model_dump() if stage.result else None,
                    }
                    for stage in session.stages
                ],
            }
        )
        narrative = with_openai_fallback(
            system,
            user,
            fallback,
            schema_name="final_evaluation_narrative",
            output_model=AIFinalNarrative,
        )

    return EvaluationResult.model_validate(
        {
            **narrative,
            "overallScore": overall_score,
            "scorecard": [item.model_dump() for item in scorecard],
        }
    )


def with_openai_fallback(
    system: str,
    user: str,
    fallback: dict[str, Any],
    *,
    schema_name: str,
    output_model: type[BaseModel],
) -> dict[str, Any]:
    try:
        raw_output = call_openai(
            system,
            user,
            schema_name=schema_name,
            schema=output_model.model_json_schema(),
        )
        validated = output_model.model_validate(raw_output).model_dump()
        return {**validated, "mode": "openai"}
    except Exception as exc:
        LOGGER.warning(
            "OpenAI fallback used for schema=%s model=%s: %s",
            schema_name,
            OPENAI_MODEL,
            exc,
        )
        fallback_response = {**fallback, "mode": "fallback"}
        notice = "OpenAI request unavailable; validated local fallback used."
        for field_name in ("extra", "notes", "rationale"):
            if field_name in fallback_response:
                fallback_response[field_name] = f"{fallback_response[field_name]} {notice}"
                break
        return fallback_response


def call_openai(system: str, user: str, *, schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        "store": False,
    }
    if OPENAI_REASONING_EFFORT:
        payload["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
    request_body = json.dumps(payload).encode("utf-8")

    started_at = time.monotonic()
    for attempt in range(OPENAI_MAX_ATTEMPTS):
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
            with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read(2_000)
            if (
                exc.code not in RETRYABLE_OPENAI_STATUS_CODES
                or attempt + 1 >= OPENAI_MAX_ATTEMPTS
                or not wait_for_retry(exc.headers.get("Retry-After") if exc.headers else None, attempt, started_at)
            ):
                detail = openai_error_detail(error_body)
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(f"OpenAI API returned HTTP {exc.code}{suffix}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt + 1 >= OPENAI_MAX_ATTEMPTS or not wait_for_retry(None, attempt, started_at):
                raise RuntimeError("OpenAI API request failed") from exc
    else:
        raise RuntimeError("OpenAI API retry budget exhausted")

    if data.get("status") == "incomplete":
        details = data.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        suffix = f" ({reason})" if reason else ""
        raise RuntimeError(f"OpenAI response was incomplete{suffix}")

    text = extract_output_text(data)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI structured output was not an object")
    return parsed


def openai_error_detail(payload: bytes) -> str:
    try:
        data = json.loads(payload.decode("utf-8"))
        error = data.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        if not message:
            return ""
        return " ".join(str(message).split())[:500]
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return ""


def wait_for_retry(retry_after: str | None, attempt: int, started_at: float) -> bool:
    retry_after_seconds = parse_retry_after(retry_after)
    exponential_delay = min(0.5 * (2**attempt), 4.0)
    delay = max(retry_after_seconds or 0.0, exponential_delay) + random.uniform(0.05, 0.25)
    remaining_budget = OPENAI_RETRY_BUDGET_SECONDS - (time.monotonic() - started_at)
    if remaining_budget <= delay:
        return False
    time.sleep(delay)
    return True


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None

    try:
        return min(max(float(value), 0.0), OPENAI_RETRY_BUDGET_SECONDS)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return min(
                max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0),
                OPENAI_RETRY_BUDGET_SECONDS,
            )
        except (TypeError, ValueError, OverflowError):
            return None


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


def local_scored_result(agent: Agent, answer: str, *, expected_words: int) -> dict[str, Any]:
    normalized = answer.lower()
    keyword_hits = sum(1 for keyword in agent.keywords if keyword.lower() in normalized)
    length_score = min(len(answer.strip().split()) / expected_words, 1)
    keyword_score = keyword_hits / len(agent.keywords) if agent.keywords else 0.5
    score = round((length_score * 45 + keyword_score * 55) * 10) / 10

    return {
        "score": score,
        "strengths": [f"{agent.name} found relevant evidence for {agent.metric}."] if score >= 60 else [],
        "weaknesses": [f"{agent.name} needs more specific examples and measurable impact."] if score < 60 else [],
        "notes": "Local rubric used because no OpenAI API result was available.",
    }


def local_primary_result(agent: Agent, answer: str) -> dict[str, Any]:
    return {
        **local_scored_result(agent, answer, expected_words=90),
        "followUpQuestion": f"What is one concrete tradeoff you made during this {agent.purpose.lower()} example?",
    }


def local_question(
    rubric: InterviewRubric,
    stage_index: int,
    answers: list[str],
    results: list[AgentResult],
) -> dict[str, Any]:
    prior_text = " ".join(answer.lower() for answer in answers if answer)
    weak_count = len([result for result in results if result.score < 60])
    selector = (stage_index + len(prior_text) + weak_count) % len(rubric.question_variants)
    question = rubric.question_variants[selector]
    if weak_count:
        question = f"{question} Please include one concrete example and one measurable signal."
    return {
        "question": question,
        "rationale": "Local adaptive question selected from server-owned rubric and completed interview evidence.",
    }


def local_final_narrative(results: list[AgentResult], overall_score: float) -> dict[str, Any]:
    strengths = [item for result in results for item in result.strengths][:5]
    weaknesses = [item for result in results for item in result.weaknesses][:5]
    if overall_score >= 72:
        recommendation = "Advance to the next round."
    elif overall_score >= 52:
        recommendation = "Hold for calibration and probe weaker areas."
    else:
        recommendation = "Do not advance based on current evidence."
    return {
        "strengths": strengths or ["Consistent evidence was provided across the completed interview."],
        "weaknesses": weaknesses or ["No major weakness surfaced in this pass."],
        "hiringRecommendation": recommendation,
        "extra": "Final scorecard computed from server-validated stage results.",
    }


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
