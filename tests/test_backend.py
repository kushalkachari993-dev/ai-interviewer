import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from backend import main


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class BackendHardeningTests(unittest.TestCase):
    def setUp(self):
        self.previous_disabled = main.OPENAI_DISABLED
        self.previous_key = main.OPENAI_API_KEY
        main.OPENAI_DISABLED = True
        with main.SESSION_STORE_LOCK:
            main.SESSION_STORE.clear()

    def tearDown(self):
        main.OPENAI_DISABLED = self.previous_disabled
        main.OPENAI_API_KEY = self.previous_key
        with main.SESSION_STORE_LOCK:
            main.SESSION_STORE.clear()

    def submit(self, view, answer):
        return main.submit_interview_answer(
            view.sessionId,
            main.AnswerSubmission(answer=answer, phase=view.phase, version=view.version),
        )

    def test_answer_size_and_extra_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            main.AnswerSubmission(
                answer="x" * (main.MAX_ANSWER_CHARS + 1),
                phase="primary",
                version=0,
            )
        with self.assertRaises(ValidationError):
            main.AnswerSubmission.model_validate(
                {"answer": "valid", "phase": "primary", "version": 0, "score": 100}
            )

    def test_session_contract_replaces_client_scoring_routes(self):
        paths = {route.path for route in main.app.routes}
        self.assertIn("/api/sessions", paths)
        self.assertIn("/api/sessions/{session_id}/answer", paths)
        self.assertNotIn("/api/question", paths)
        self.assertNotIn("/api/agent", paths)
        self.assertNotIn("/api/evaluation", paths)

    def test_session_view_hides_rubric_keywords_and_provisional_score(self):
        session = main.start_interview()
        initial_payload = session.model_dump()
        self.assertNotIn("keywords", json.dumps(initial_payload))

        after_primary = self.submit(session, session.stages[0].sample)
        self.assertEqual(after_primary.phase, "follow_up")
        self.assertEqual(after_primary.completedStages, 0)
        self.assertEqual(after_primary.partialScore, 0)
        self.assertIsNone(after_primary.stages[0].result)
        self.assertTrue(after_primary.stages[0].followUpQuestion)

    def test_follow_up_is_required_before_stage_score_and_advance(self):
        session = main.start_interview()
        after_primary = self.submit(session, session.stages[0].sample)
        self.assertEqual(after_primary.activeIndex, 0)
        self.assertEqual(after_primary.phase, "follow_up")

        after_follow_up = self.submit(
            after_primary,
            "I chose a smaller launch scope and tracked response-time improvement.",
        )
        first_score = after_follow_up.stages[0].result.score
        self.assertEqual(after_follow_up.activeIndex, 1)
        self.assertEqual(after_follow_up.phase, "primary")
        self.assertEqual(after_follow_up.completedStages, 1)
        self.assertEqual(after_follow_up.partialScore, first_score)

    def test_full_interview_produces_server_computed_scorecard(self):
        view = main.start_interview()
        while view.phase != "complete":
            stage = view.stages[view.activeIndex]
            answer = (
                stage.sample
                if view.phase == "primary"
                else "I made a concrete tradeoff, measured the result, and adjusted based on the evidence."
            )
            view = self.submit(view, answer)

        self.assertEqual(view.completedStages, 4)
        self.assertEqual(len(view.finalEvaluation.scorecard), 4)
        self.assertNotIn("Evaluation Agent", [stage.name for stage in view.stages])
        expected = round(sum(stage.result.score for stage in view.stages) / len(view.stages), 1)
        self.assertEqual(view.partialScore, expected)
        self.assertEqual(view.finalEvaluation.overallScore, expected)

        with self.assertRaises(HTTPException) as context:
            main.submit_interview_answer(
                view.sessionId,
                main.AnswerSubmission(answer="submit again", phase="follow_up", version=view.version),
            )
        self.assertEqual(context.exception.status_code, 409)

    def test_stale_submission_cannot_be_replayed_as_follow_up(self):
        session = main.start_interview()
        stale_payload = main.AnswerSubmission(
            answer=session.stages[0].sample,
            phase=session.phase,
            version=session.version,
        )
        after_primary = main.submit_interview_answer(session.sessionId, stale_payload)
        self.assertEqual(after_primary.phase, "follow_up")

        with self.assertRaises(HTTPException) as context:
            main.submit_interview_answer(session.sessionId, stale_payload)
        self.assertEqual(context.exception.status_code, 409)
        current = main.read_interview(session.sessionId)
        self.assertEqual(current.completedStages, 0)

    def test_final_narrative_cannot_override_server_scores(self):
        session = main.create_interview_session()
        for index, stage in enumerate(session.stages):
            stage.answer = "Primary evidence"
            stage.follow_up_question = "What tradeoff?"
            stage.follow_up_answer = "Follow-up evidence"
            stage.result = main.AgentResult(
                score=20 * (index + 1),
                strengths=["Evidence"],
                weaknesses=[],
                notes="Validated",
                mode="fallback",
            )

        main.OPENAI_DISABLED = False
        main.OPENAI_API_KEY = "test-key"
        narrative = {
            "strengths": ["Synthesized strength"],
            "weaknesses": ["Synthesized weakness"],
            "hiringRecommendation": "Hold for calibration.",
            "extra": "Narrative only.",
        }
        with patch.object(main, "call_openai", return_value=narrative):
            evaluation = main.build_final_evaluation(session)

        self.assertEqual(evaluation.overallScore, 50)
        self.assertEqual([item.score for item in evaluation.scorecard], [20, 40, 60, 80])

    def test_structured_output_schema_is_strict(self):
        schema = main.AIAgentOutput.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            {"score", "strengths", "weaknesses", "notes", "followUpQuestion"},
        )

    @patch.object(main.urllib.request, "urlopen")
    def test_openai_request_uses_structured_outputs(self, urlopen):
        expected = {
            "question": "What tradeoff did you make?",
            "rationale": "Tests decision quality.",
        }
        urlopen.return_value = FakeResponse({"output_text": json.dumps(expected)})

        actual = main.call_openai(
            "system",
            "user",
            schema_name="interview_question",
            schema=main.AIQuestionOutput.model_json_schema(),
        )

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        output_format = body["text"]["format"]
        self.assertEqual(actual, expected)
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertFalse(body["store"])
        self.assertEqual(body["max_output_tokens"], main.OPENAI_MAX_OUTPUT_TOKENS)
        if main.OPENAI_REASONING_EFFORT:
            self.assertEqual(body["reasoning"], {"effort": main.OPENAI_REASONING_EFFORT})

    @patch.object(main, "wait_for_retry", return_value=True)
    @patch.object(main.urllib.request, "urlopen")
    def test_transient_openai_errors_are_retried(self, urlopen, wait_for_retry):
        rate_limit_error = urllib.error.HTTPError(
            "https://api.openai.com/v1/responses",
            429,
            "rate limited",
            {"Retry-After": "0"},
            io.BytesIO(b"{}"),
        )
        expected = {
            "question": "What tradeoff did you make?",
            "rationale": "Tests decision quality.",
        }
        urlopen.side_effect = [rate_limit_error, FakeResponse({"output_text": json.dumps(expected)})]

        actual = main.call_openai(
            "system",
            "user",
            schema_name="interview_question",
            schema=main.AIQuestionOutput.model_json_schema(),
        )

        self.assertEqual(actual, expected)
        self.assertEqual(urlopen.call_count, 2)
        wait_for_retry.assert_called_once()


if __name__ == "__main__":
    unittest.main()
