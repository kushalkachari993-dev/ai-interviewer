import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Mic, Square, Volume2 } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");
const MAX_ANSWER_CHARS = 4800;
const SESSION_STORAGE_KEY = "aiInterviewerSessionId";

function App() {
  const [session, setSession] = useState(null);
  const [tracks, setTracks] = useState([]);
  const [levels, setLevels] = useState([]);
  const [selectedTrackId, setSelectedTrackId] = useState("");
  const [selectedLevelId, setSelectedLevelId] = useState("mid");
  const [draft, setDraft] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [muted, setMuted] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("Ready");
  const recognitionRef = useRef(null);

  const stages = session?.stages || [];
  const activeStage = stages[session?.activeIndex || 0];
  const isFollowUp = session?.phase === "follow_up";
  const isComplete = session?.phase === "complete";
  const currentPrompt = isFollowUp ? activeStage?.followUpQuestion : activeStage?.question;
  const finalEvaluation = session?.finalEvaluation;
  const answerLimitState =
    draft.length >= MAX_ANSWER_CHARS ? "limit-reached" : draft.length >= MAX_ANSWER_CHARS * 0.9 ? "near-limit" : "";

  useEffect(() => {
    let cancelled = false;

    async function initializeSession() {
      try {
        const [availableTracks, availableLevels] = await Promise.all([
          getJson("/api/tracks"),
          getJson("/api/levels"),
        ]);
        if (!cancelled) {
          setTracks(availableTracks);
          setLevels(availableLevels);
          setSelectedTrackId(availableTracks[0]?.id || "");
          setSelectedLevelId(availableLevels.find((level) => level.id === "mid")?.id || availableLevels[0]?.id || "");
        }
        const savedId = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
        let nextSession = null;
        if (savedId) {
          try {
            nextSession = await getJson(`/api/sessions/${encodeURIComponent(savedId)}`);
          } catch {
            window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
          }
        }
        if (!cancelled && nextSession) applySession(nextSession);
      } catch (error) {
        if (!cancelled) setErrorMessage(error.message);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    initializeSession();
    return () => {
      cancelled = true;
    };
  }, []);

  const feedback = useMemo(() => {
    if (finalEvaluation) {
      return {
        average: finalEvaluation.overallScore,
        strengths: finalEvaluation.strengths,
        weaknesses: finalEvaluation.weaknesses,
        recommendation: finalEvaluation.hiringRecommendation,
        extra: finalEvaluation.extra,
      };
    }

    const completedResults = stages.map((stage) => stage.result).filter(Boolean);
    return {
      average: session?.partialScore || 0,
      strengths: completedResults.flatMap((result) => result.strengths || []).slice(0, 5),
      weaknesses: completedResults.flatMap((result) => result.weaknesses || []).slice(0, 5),
      recommendation: "Complete every stage and follow-up to generate a recommendation.",
      extra: session?.completedStages
        ? `${session.completedStages} of ${session.totalStages} stages completed. The score reflects completed stages only.`
        : "Each interview stage includes one required follow-up before it is scored.",
    };
  }, [finalEvaluation, session, stages]);

  function applySession(nextSession) {
    setSession(nextSession);
    setSelectedTrackId(nextSession.track.id);
    setSelectedLevelId(nextSession.level.id);
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, nextSession.sessionId);
  }

  async function startNewInterview(
    trackId = selectedTrackId || session?.track?.id,
    levelId = selectedLevelId || session?.level?.id,
  ) {
    if (!trackId || !levelId) return;
    setIsLoading(true);
    setErrorMessage("");
    try {
      const nextSession = await postJson("/api/sessions", { trackId, levelId });
      applySession(nextSession);
      setDraft("");
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setIsLoading(false);
    }
  }

  function chooseAnotherTrack() {
    window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
    setSelectedTrackId(session?.track?.id || tracks[0]?.id || "");
    setSelectedLevelId(session?.level?.id || levels.find((level) => level.id === "mid")?.id || levels[0]?.id || "");
    setSession(null);
    setDraft("");
    setErrorMessage("");
  }

  async function submitAnswer(event) {
    event.preventDefault();
    if (!draft.trim() || isSubmitting || !session || isComplete) return;

    setIsSubmitting(true);
    setErrorMessage("");
    try {
      const nextSession = await postJson(`/api/sessions/${encodeURIComponent(session.sessionId)}/answer`, {
        answer: draft.trim(),
        phase: session.phase,
        version: session.version,
      });
      applySession(nextSession);
      setDraft("");

      if (nextSession.phase !== "complete" && !muted) {
        const nextStage = nextSession.stages[nextSession.activeIndex];
        const nextPrompt = nextSession.phase === "follow_up" ? nextStage.followUpQuestion : nextStage.question;
        speak(nextPrompt);
      }
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  function speak(text) {
    if (muted || !text || !("speechSynthesis" in window)) return;
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
      recognition.onerror = () => setVoiceStatus("Try again");
      recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map((result) => result[0].transcript)
          .join(" ");
        setDraft(transcript.slice(0, MAX_ANSWER_CHARS));
      };
      recognitionRef.current = recognition;
    }

    recognitionRef.current.start();
  }

  if (isLoading && !session) {
    return (
      <main className="loading-state">
        <h1>Loading interview options…</h1>
        <p>The server is preparing the available tracks and experience levels.</p>
      </main>
    );
  }

  if (!session && tracks.length && levels.length) {
    return (
      <TrackSelector
        errorMessage={errorMessage}
        isLoading={isLoading}
        levels={levels}
        onLevelSelect={setSelectedLevelId}
        onSelect={setSelectedTrackId}
        onStart={startNewInterview}
        selectedLevelId={selectedLevelId}
        selectedTrackId={selectedTrackId}
        tracks={tracks}
      />
    );
  }

  if (!session) {
    return (
      <main className="loading-state">
        <h1>Interview unavailable</h1>
        <p>{errorMessage || "The interview session could not be created."}</p>
        <button className="primary-button" onClick={() => window.location.reload()} type="button">
          Try again
        </button>
      </main>
    );
  }

  return (
    <main className="shell">
      <aside className="sidebar" aria-label="Interview stages">
        <div className="brand">
          <span className="brand-mark">AI</span>
          <div>
            <h1>AI Interviewer Platform</h1>
            <p>Server-scored adaptive interview workflow.</p>
          </div>
        </div>

        <div className="track-badge">
          <span>Selected interview</span>
          <strong>{session.track.name}</strong>
          <small>{session.level.name} interview</small>
        </div>

        <section className="panel">
          <div className="section-title">
            <span>Stages</span>
            <span className="count">{session.totalStages}</span>
          </div>
          <div className="agent-list">
            {stages.map((stage, index) => (
              <div
                className={`agent-button ${index === session.activeIndex && !isComplete ? "active" : ""} ${stage.status}`}
                key={stage.name}
              >
                <span className="agent-icon">{stage.short}</span>
                <span>
                  <span className="agent-name">{stage.name}</span>
                  <span className="agent-purpose">
                    {stage.status === "complete" ? "Completed" : stage.status === "follow_up" ? "Follow-up" : stage.purpose}
                  </span>
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="panel voice-panel">
          <div className="section-title">
            <span>Voice</span>
            <span className="status-pill">{voiceStatus}</span>
          </div>
          <div className="voice-actions">
            <button className="icon-button" disabled={isComplete} onClick={startListening} title="Start voice input" type="button">
              <Mic size={18} />
            </button>
            <button className="icon-button" disabled={isComplete} onClick={() => speak(currentPrompt)} title="Read current question" type="button">
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
            <p className="eyebrow">
              {isComplete ? "Interview complete" : `${activeStage.name}${isFollowUp ? " · Required follow-up" : ""}`}
            </p>
            <h2>{isComplete ? "Your validated evaluation is ready." : currentPrompt}</h2>
          </div>
          <div className="timer">{session.completedStages} of {session.totalStages}</div>
        </header>

        <section className="conversation" aria-live="polite">
          {stages.map((stage, index) => (
            <ConversationBlock
              key={stage.name}
              show={index <= session.activeIndex || stage.status === "complete"}
              stage={stage}
            />
          ))}
        </section>

        {isComplete ? (
          <section className="answer-box completion-box">
            <h3>Interview complete</h3>
            <p>All four {session.track.name} stages and their follow-ups were scored by the server.</p>
            <div className="completion-actions">
              <button className="primary-button" onClick={() => startNewInterview(session.track.id, session.level.id)} type="button">
                Repeat this track
              </button>
              <button className="secondary-button" onClick={chooseAnotherTrack} type="button">
                Choose another track
              </button>
            </div>
          </section>
        ) : (
          <form className="answer-box" onSubmit={submitAnswer}>
            <div className="answer-label-row">
              <label htmlFor="answerInput">{isFollowUp ? "Follow-up response" : "Candidate response"}</label>
              <span className={`answer-limit ${answerLimitState}`} id="answerLimit">
                {draft.length.toLocaleString()} / {MAX_ANSWER_CHARS.toLocaleString()} characters
              </span>
            </div>
            <p className="answer-guidance" id="answerGuidance">
              {isFollowUp
                ? "This response completes the current stage and unlocks its score."
                : "Keep it focused—up to roughly 700–800 words. A follow-up comes next."}
            </p>
            <textarea
              aria-describedby="answerGuidance answerLimit"
              disabled={isSubmitting}
              id="answerInput"
              maxLength={MAX_ANSWER_CHARS}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={isFollowUp ? "Answer the required follow-up…" : "Type or dictate the candidate answer…"}
              rows={6}
              value={draft}
            />
            {errorMessage && <p className="form-error" role="alert">{errorMessage}</p>}
            <div className="answer-actions">
              {!isFollowUp && activeStage.sample && (
                <button className="secondary-button" disabled={isSubmitting} onClick={() => setDraft(activeStage.sample)} type="button">
                  Use sample
                </button>
              )}
              <button className="primary-button" disabled={isSubmitting || !draft.trim()} type="submit">
                {isSubmitting ? "Asking AI…" : isFollowUp ? "Submit follow-up" : "Submit answer"}
              </button>
            </div>
          </form>
        )}
      </section>

      <aside className="output" aria-label="Interview output">
        <section className="scorecard">
          <div className="section-title">
            <span>Validated output</span>
            <span className="score">{Math.round(feedback.average)}</span>
          </div>
          <h3>Scorecard</h3>
          <div className="metric-list">
            {stages.map((stage) => (
              <Metric key={stage.name} metric={stage.metric} score={stage.result?.score ?? 0} />
            ))}
          </div>
        </section>

        <section className="evaluation-grid">
          <EvaluationList title="Strengths" values={feedback.strengths} fallback="Awaiting completed stage evidence." />
          <EvaluationList title="Weaknesses" values={feedback.weaknesses} fallback="No completed stage has been scored yet." />
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

function TrackSelector({
  errorMessage,
  isLoading,
  levels,
  onLevelSelect,
  onSelect,
  onStart,
  selectedLevelId,
  selectedTrackId,
  tracks,
}) {
  const [query, setQuery] = useState("");
  const selectedTrack = tracks.find((track) => track.id === selectedTrackId) || tracks[0];
  const selectedLevel = levels.find((level) => level.id === selectedLevelId) || levels[0];
  const normalizedQuery = query.trim().toLowerCase();
  const visibleTracks = tracks.filter((track) => {
    if (!normalizedQuery) return true;
    return [track.name, track.description, ...track.skills]
      .join(" ")
      .toLowerCase()
      .includes(normalizedQuery);
  });

  return (
    <main className="track-page">
      <section className="track-hero">
        <div className="brand track-brand">
          <span className="brand-mark">AI</span>
          <div>
            <h1>AI Interviewer Platform</h1>
            <p>Choose the technical interview you want to practice.</p>
          </div>
        </div>
        <p className="eyebrow">Server-owned tracks and expectations</p>
        <h2>Choose your interview</h2>
        <p className="track-intro">
          Every interview includes Resume and HR stages plus two technical stages tailored to your selection.
          Each stage includes a required follow-up and a validated score.
        </p>
      </section>

      <section className="level-selector" aria-labelledby="level-selector-title">
        <div className="level-selector-heading">
          <div>
            <p className="eyebrow">Step 1</p>
            <h3 id="level-selector-title">Select your experience level</h3>
          </div>
          <span>Questions and scoring depth adapt to this choice.</span>
        </div>
        <div className="level-grid">
          {levels.map((level) => (
            <button
              aria-pressed={level.id === selectedLevel?.id}
              className={`level-card ${level.id === selectedLevel?.id ? "selected" : ""}`}
              key={level.id}
              onClick={() => onLevelSelect(level.id)}
              type="button"
            >
              <strong>{level.name}</strong>
              <span>{level.description}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="track-search" aria-label="Filter interview tracks">
        <label htmlFor="track-search-input"><span className="eyebrow">Step 2</span> Search and select a track</label>
        <div className="track-search-row">
          <input
            autoComplete="off"
            id="track-search-input"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Try data science, networks, cloud, product..."
            type="search"
            value={query}
          />
          <span aria-live="polite">{visibleTracks.length} of {tracks.length} tracks</span>
        </div>
      </section>

      <section className="track-grid" aria-label="Available interview tracks">
        {visibleTracks.map((track) => (
          <button
            aria-pressed={track.id === selectedTrack?.id}
            className={`track-card ${track.id === selectedTrack?.id ? "selected" : ""}`}
            key={track.id}
            onClick={() => onSelect(track.id)}
            type="button"
          >
            <span className="track-card-topline">
              <span className="track-monogram">
                {track.name.split(/[\s/]+/).filter(Boolean).map((word) => word[0]).join("").slice(0, 2)}
              </span>
              <span className="track-check">{track.id === selectedTrack?.id ? "Selected" : "Choose"}</span>
            </span>
            <strong>{track.name}</strong>
            <span className="track-description">{track.description}</span>
            <span className="skill-list">
              {track.skills.map((skill) => <span key={skill}>{skill}</span>)}
            </span>
          </button>
        ))}
        {visibleTracks.length === 0 && (
          <p className="track-empty" role="status">No interview tracks match “{query}”.</p>
        )}
      </section>

      <section className="track-start-panel">
        <div>
          <span className="eyebrow">Ready to begin</span>
          <h3>{selectedTrack?.name}</h3>
          <p>{selectedLevel?.name} interview · 4 stages · 4 required follow-ups · about 20–30 minutes</p>
          {errorMessage && <p className="form-error" role="alert">{errorMessage}</p>}
        </div>
        <button
          className="primary-button track-start-button"
          disabled={isLoading || !selectedTrack || !selectedLevel}
          onClick={() => onStart(selectedTrack.id, selectedLevel.id)}
          type="button"
        >
          {isLoading ? "Starting…" : `Start ${selectedTrack?.name || "interview"}`}
        </button>
      </section>
    </main>
  );
}

function ConversationBlock({ stage, show }) {
  if (!show) return null;
  const mode = stage.result?.mode === "openai" ? "GenAI" : "Local";
  return (
    <>
      <article className="message">
        <div className="message-meta">{stage.name}</div>
        <p>{stage.question}</p>
      </article>
      {stage.answer && (
        <article className="message candidate">
          <div className="message-meta">Candidate primary answer</div>
          <p>{stage.answer}</p>
        </article>
      )}
      {stage.followUpQuestion && (
        <article className="message follow-up-message">
          <div className="message-meta">{stage.name} · Required follow-up</div>
          <p>{stage.followUpQuestion}</p>
        </article>
      )}
      {stage.followUpAnswer && (
        <article className="message candidate">
          <div className="message-meta">Candidate follow-up answer</div>
          <p>{stage.followUpAnswer}</p>
        </article>
      )}
      {stage.result && (
        <article className="message stage-result">
          <div className="message-meta">Validated stage result · {stage.result.score}/100 · {mode}</div>
          <p>{stage.result.notes}</p>
        </article>
      )}
    </>
  );
}

function Metric({ metric, score }) {
  const safeScore = Math.max(0, Math.min(Number(score) || 0, 100));
  return (
    <div className="metric">
      <div className="metric-row">
        <strong>{metric}</strong>
        <span>{safeScore}/100</span>
      </div>
      <div className="bar" aria-hidden="true">
        <div className="bar-fill" style={{ width: `${safeScore}%` }} />
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

async function getJson(path) {
  return requestJson(path, { method: "GET" });
}

async function postJson(path, payload) {
  return requestJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    ...(payload === undefined ? {} : { body: JSON.stringify(payload) }),
  });
}

async function requestJson(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    if (response.status === 404) throw new Error("Your interview session expired. Start a new interview.");
    if (response.status === 409) throw new Error("This interview step was already submitted. Refresh to continue.");
    if (response.status === 429) {
      const retryAfter = response.headers.get("Retry-After");
      const waitTime = retryAfter ? `about ${retryAfter} seconds` : "a moment";
      throw new Error(`Several AI requests were made quickly. Please wait ${waitTime} and try again.`);
    }
    if (response.status === 413) {
      throw new Error(`This interview request is too large. Keep each answer under ${MAX_ANSWER_CHARS.toLocaleString()} characters.`);
    }
    if (response.status === 422) {
      throw new Error("This answer exceeds an allowed limit. Shorten it and try again.");
    }
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return response.json();
}

createRoot(document.getElementById("root")).render(<App />);
