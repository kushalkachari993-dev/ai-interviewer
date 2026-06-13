import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Mic, Square, Volume2 } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

const agents = [
  {
    name: "Resume Agent",
    short: "RS",
    purpose: "Experience fit",
    metric: "Resume relevance",
    question: "Walk me through a project from your resume where your personal contribution changed the outcome.",
    sample:
      "I led a customer support automation project that reduced average response time by 34%. I owned the architecture, coordinated with product and support, shipped the first version in six weeks, and measured success through ticket deflection and CSAT.",
    keywords: ["led", "owned", "impact", "measured", "project", "reduced", "built"],
  },
  {
    name: "Coding Agent",
    short: "CD",
    purpose: "Problem solving",
    metric: "Coding signal",
    question: "How would you debug a production API that suddenly became slow?",
    sample:
      "I would start with metrics and traces to isolate whether latency is from the app, database, network, or a dependency. Then I would compare recent deploys, inspect slow queries, reproduce with production-like inputs, mitigate with rollback or scaling, and write a regression test.",
    keywords: ["metrics", "traces", "database", "deploy", "reproduce", "rollback", "test"],
  },
  {
    name: "System Design Agent",
    short: "SD",
    purpose: "Architecture",
    metric: "Design depth",
    question: "Design a scalable interview scheduling system for recruiters and candidates.",
    sample:
      "I would model availability, bookings, time zones, interviewers, and candidate preferences. The API would use optimistic locking to prevent double booking, a queue for notifications, calendar provider integrations, and audit logs for reschedules.",
    keywords: ["model", "api", "queue", "locking", "timezone", "scale", "audit"],
  },
  {
    name: "HR Agent",
    short: "HR",
    purpose: "Behavioral fit",
    metric: "Collaboration",
    question: "Describe a time you handled disagreement with a teammate.",
    sample:
      "A teammate and I disagreed about prioritizing refactor work before a launch. I asked them to define the risk, shared the launch constraint, and we agreed on a smaller cleanup plus a follow-up task. It kept trust intact and still protected delivery.",
    keywords: ["disagreed", "risk", "shared", "agreed", "trust", "delivery", "teammate"],
  },
  {
    name: "Evaluation Agent",
    short: "EV",
    purpose: "Final synthesis",
    metric: "Evaluation quality",
    question: "What is the strongest hiring signal so far, and what is the largest remaining risk?",
    sample:
      "The candidate should advance because they showed clear ownership, structured debugging, practical system design choices, and mature collaboration. I would probe depth in distributed systems and code quality in a live round.",
    keywords: ["advance", "ownership", "structured", "design", "collaboration", "probe", "depth"],
  },
];

function App() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [answers, setAnswers] = useState(Array(agents.length).fill(""));
  const [results, setResults] = useState(Array(agents.length).fill(null));
  const [questions, setQuestions] = useState(agents.map((agent) => agent.question));
  const [finalEvaluation, setFinalEvaluation] = useState(null);
  const [draft, setDraft] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGeneratingQuestion, setIsGeneratingQuestion] = useState(false);
  const [muted, setMuted] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("Ready");
  const recognitionRef = useRef(null);

  const activeAgent = { ...agents[activeIndex], question: questions[activeIndex] };
  const scores = results.map((result) => Number(result?.score || 0));
  const completed = answers.filter(Boolean).length;
  const isEditingSavedAnswer = Boolean(answers[activeIndex]) && draft === answers[activeIndex];

  useEffect(() => {
    generateQuestion(0, answers, results, { silent: true });
  }, []);

  const feedback = useMemo(() => {
    if (finalEvaluation) {
      return {
        average: finalEvaluation.overallScore || 0,
        strengths: finalEvaluation.strengths?.length
          ? finalEvaluation.strengths
          : ["Awaiting stronger evidence across the interview."],
        weaknesses: finalEvaluation.weaknesses?.length
          ? finalEvaluation.weaknesses
          : ["No major weakness surfaced in this pass."],
        recommendation: finalEvaluation.hiringRecommendation,
        extra: finalEvaluation.extra,
      };
    }

    const average = scores.reduce((sum, score) => sum + score, 0) / agents.length;
    return {
      average,
      strengths: results.flatMap((result) => result?.strengths || []).slice(0, 5),
      weaknesses: results.flatMap((result) => result?.weaknesses || []).slice(0, 5),
      recommendation: "Complete the interview to generate a recommendation.",
      extra: completed
        ? `${completed} of ${agents.length} agents completed. Voice input and prompt playback are available where the browser allows them.`
        : "Voice support is available for candidate answers and reading prompts aloud.",
    };
  }, [completed, finalEvaluation, results, scores]);

  async function submitAnswer(event) {
    event.preventDefault();
    if (!draft.trim() || isSubmitting) return;

    const nextAnswers = [...answers];
    nextAnswers[activeIndex] = draft.trim();
    setAnswers(nextAnswers);
    setIsSubmitting(true);

    try {
      const agentResult = await postJson("/api/agent", { agent: activeAgent, answer: draft.trim() });
      const nextResults = [...results];
      nextResults[activeIndex] = agentResult;
      setResults(nextResults);

      if (nextAnswers.filter(Boolean).length === agents.length) {
        setFinalEvaluation(await postJson("/api/evaluation", { agents, answers: nextAnswers, results: nextResults }));
      } else {
        setFinalEvaluation(null);
      }

      if (activeIndex < agents.length - 1) {
        const nextIndex = activeIndex + 1;
        const nextQuestion = await generateQuestion(nextIndex, nextAnswers, nextResults);
        setActiveIndex(nextIndex);
        setDraft(nextAnswers[nextIndex] || "");
        if (!muted) speak(nextQuestion);
      } else {
        setDraft("");
      }
    } catch (error) {
      setFinalEvaluation({
        overallScore: 0,
        strengths: [],
        weaknesses: ["The backend request failed."],
        hiringRecommendation: "Retry after confirming the FastAPI server is running.",
        extra: error.message,
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  function selectAgent(index) {
    setActiveIndex(index);
    setDraft(answers[index] || "");
  }

  async function generateQuestion(index = activeIndex, nextAnswers = answers, nextResults = results, options = {}) {
    if (!options.silent) setIsGeneratingQuestion(true);
    try {
      const payloadAgent = { ...agents[index], question: questions[index] };
      const generated = await postJson("/api/question", {
        agent: payloadAgent,
        agentIndex: index,
        answers: nextAnswers,
        results: nextResults,
      });
      const nextQuestions = [...questions];
      nextQuestions[index] = generated.question || agents[index].question;
      setQuestions(nextQuestions);
      return nextQuestions[index];
    } catch {
      return questions[index] || agents[index].question;
    } finally {
      if (!options.silent) setIsGeneratingQuestion(false);
    }
  }

  function speak(text) {
    if (muted || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  }

  function startListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setVoiceStatus("Text only");
      return;
    }

    if (!recognitionRef.current) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = "en-US";
      recognition.onstart = () => setVoiceStatus("Listening");
      recognition.onend = () => setVoiceStatus("Ready");
      recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map((result) => result[0].transcript)
          .join(" ");
        setDraft(transcript);
      };
      recognitionRef.current = recognition;
    }

    recognitionRef.current.start();
  }

  return (
    <main className="shell">
      <aside className="sidebar" aria-label="Interview agents">
        <div className="brand">
          <span className="brand-mark">AI</span>
          <div>
            <h1>AI Interviewer Platform</h1>
            <p>FastAPI + React GenAI interview workflow.</p>
          </div>
        </div>

        <section className="panel">
          <div className="section-title">
            <span>Agents</span>
            <span className="count">{agents.length}</span>
          </div>
          <div className="agent-list">
            {agents.map((agent, index) => (
              <button
                className={`agent-button ${index === activeIndex ? "active" : ""}`}
                key={agent.name}
                onClick={() => selectAgent(index)}
                type="button"
              >
                <span className="agent-icon">{agent.short}</span>
                <span>
                  <span className="agent-name">{agent.name}</span>
                  <span className="agent-purpose">{agent.purpose}</span>
                </span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel voice-panel">
          <div className="section-title">
            <span>Voice</span>
            <span className="status-pill">{voiceStatus}</span>
          </div>
          <div className="voice-actions">
            <button className="icon-button" onClick={startListening} title="Start voice input" type="button">
              <Mic size={18} />
            </button>
            <button className="icon-button" onClick={() => speak(activeAgent.question)} title="Read current question" type="button">
              <Volume2 size={18} />
            </button>
            <button
              className={`icon-button ${muted ? "active" : ""}`}
              onClick={() => {
                setMuted(!muted);
                if (!muted && "speechSynthesis" in window) window.speechSynthesis.cancel();
              }}
              title="Toggle voice playback"
              type="button"
            >
              <Square size={18} />
            </button>
          </div>
        </section>
      </aside>

      <section className="workspace" aria-label="Interview workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">{activeAgent.name}</p>
            <h2>{activeAgent.question}</h2>
          </div>
          <div className="timer">{activeIndex + 1} of {agents.length}</div>
        </header>

        <section className="conversation" aria-live="polite">
          {agents.map((agent, index) => (
            <ConversationBlock
              agent={{ ...agent, question: questions[index] }}
              answer={answers[index]}
              key={agent.name}
              result={results[index]}
              show={index <= activeIndex || answers[index]}
            />
          ))}
        </section>

        <form className="answer-box" onSubmit={submitAnswer}>
          <label htmlFor="answerInput">
            {isEditingSavedAnswer ? "Saved candidate response" : "Candidate response"}
          </label>
          <textarea
            disabled={isSubmitting}
            id="answerInput"
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Type or dictate the candidate answer..."
            rows={6}
            value={draft}
          />
          <div className="answer-actions">
            <button
              className="secondary-button"
              disabled={isSubmitting || isGeneratingQuestion || Boolean(answers[activeIndex])}
              onClick={() => generateQuestion()}
              type="button"
            >
              {isGeneratingQuestion ? "Generating..." : "New question"}
            </button>
            <button className="secondary-button" onClick={() => setDraft(activeAgent.sample)} type="button">
              Use sample
            </button>
            <button className="primary-button" disabled={isSubmitting} type="submit">
              {isSubmitting ? "Asking AI..." : "Submit answer"}
            </button>
          </div>
        </form>
      </section>

      <aside className="output" aria-label="Interview output">
        <section className="scorecard">
          <div className="section-title">
            <span>Output</span>
            <span className="score">{Math.round(feedback.average)}</span>
          </div>
          <h3>Scorecard</h3>
          <div className="metric-list">
            {agents.map((agent, index) => {
              const score = finalEvaluation?.scorecard?.[index]?.score ?? scores[index];
              return <Metric agent={agent} key={agent.name} score={score} />;
            })}
          </div>
        </section>

        <section className="evaluation-grid">
          <EvaluationList title="Strengths" values={feedback.strengths} fallback="Awaiting stronger evidence across the interview." />
          <EvaluationList title="Weaknesses" values={feedback.weaknesses} fallback="Incomplete interview coverage." />
          <article className="recommendation">
            <h3>Hiring recommendation</h3>
            <p>{feedback.recommendation}</p>
          </article>
          <article>
            <h3>Extra</h3>
            <p>{feedback.extra}</p>
          </article>
        </section>
      </aside>
    </main>
  );
}

function ConversationBlock({ agent, answer, result, show }) {
  if (!show) return null;
  const mode = result?.mode === "openai" ? "GenAI" : "Local";
  return (
    <>
      <article className="message">
        <div className="message-meta">{agent.name}</div>
        <p>{agent.question}</p>
      </article>
      {answer && (
        <article className="message candidate">
          <div className="message-meta">
            Candidate answer - {Number(result?.score || 0)}/100 - {mode}
          </div>
          <p>{answer}</p>
          {result?.notes && <p className="agent-note">{result.notes}</p>}
        </article>
      )}
      {result?.followUpQuestion && (
        <article className="message">
          <div className="message-meta">{agent.name} follow-up</div>
          <p>{result.followUpQuestion}</p>
        </article>
      )}
    </>
  );
}

function Metric({ agent, score }) {
  return (
    <div className="metric">
      <div className="metric-row">
        <strong>{agent.metric}</strong>
        <span>{score}/100</span>
      </div>
      <div className="bar" aria-hidden="true">
        <div className="bar-fill" style={{ width: `${score}%` }} />
      </div>
    </div>
  );
}

function EvaluationList({ title, values, fallback }) {
  const items = values?.length ? values : [fallback];
  return (
    <article>
      <h3>{title}</h3>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </article>
  );
}

async function postJson(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return response.json();
}

createRoot(document.getElementById("root")).render(<App />);
