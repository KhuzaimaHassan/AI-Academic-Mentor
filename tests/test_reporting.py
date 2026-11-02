"""Offline regressions for provider failures and the full report workflow."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_Src_Code"))

import agentic_ai_pipeline as pipeline
from reporting import DEFAULT_GROQ_MODEL, MentorReportError, get_groq_model


class ReportWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.llm = Mock()
        self.rag = Mock()
        self.rag.get_intervention_advice.return_value = "Source 1 (guide.txt): Plan study time."
        self.provider = patch.object(pipeline, "ChatGroq", return_value=self.llm)
        self.provider_mock = self.provider.start()
        self.addCleanup(self.provider.stop)
        self.retrieval = patch.object(pipeline, "create_rag_system", return_value=self.rag)
        self.retrieval.start()
        self.addCleanup(self.retrieval.stop)
        self.data = patch.object(pipeline.pd, "read_csv", return_value=pd.DataFrame([
            {"id_student": 123, "studied_credits": 60, "num_of_prev_attempts": 0}
        ]))
        self.data.start()
        self.addCleanup(self.data.stop)
        self.prediction = patch.object(pipeline, "predict_student_result", return_value="Pass")
        self.prediction_mock = self.prediction.start()
        self.addCleanup(self.prediction.stop)

    def make_agent(self):
        return pipeline.AgenticAIPipeline(groq_api_key="test-only", groq_model=DEFAULT_GROQ_MODEL)

    def test_success_requires_all_graph_stages(self):
        self.llm.invoke.side_effect = [
            SimpleNamespace(content="Possible study-support needs."),
            SimpleNamespace(content="study planning"),
            SimpleNamespace(content="# Mentor Report\nReview a weekly study plan."),
        ]
        report = self.make_agent().run_student_analysis(123)
        self.assertIn("# Mentor Report", report)
        self.assertIn("**Model prediction:** Pass", report)
        self.assertIn("Source 1 (guide.txt): Plan study time.", report)
        self.assertIn("AI-generated draft for human review", report)
        self.assertEqual(self.llm.invoke.call_count, 3)
        self.prediction_mock.assert_called_once_with(123)
        self.rag.get_intervention_advice.assert_called_once_with("study planning")
        self.assertEqual(self.provider_mock.call_args.kwargs["model"], DEFAULT_GROQ_MODEL)

    def test_model_access_failure_stops_before_retrieval_or_report(self):
        error = RuntimeError("provider payload containing private diagnostics")
        error.status_code = 404
        self.llm.invoke.side_effect = error
        with self.assertRaisesRegex(MentorReportError, "GROQ_MODEL") as caught:
            self.make_agent().run_student_analysis(123)
        self.assertNotIn("private diagnostics", str(caught.exception))
        self.assertEqual(self.llm.invoke.call_count, 1)
        self.rag.get_intervention_advice.assert_not_called()

    def test_rate_limit_is_an_error_not_report_content(self):
        error = RuntimeError("rate limit")
        error.status_code = 429
        self.llm.invoke.side_effect = error
        with self.assertRaisesRegex(MentorReportError, "rate limit"):
            self.make_agent().run_student_analysis(123)

    def test_failed_prediction_prevents_llm_calls(self):
        self.prediction_mock.return_value = None
        with self.assertRaisesRegex(MentorReportError, "classifier"):
            self.make_agent().run_student_analysis(123)
        self.llm.invoke.assert_not_called()

    def test_failed_retrieval_prevents_report_compilation(self):
        self.llm.invoke.side_effect = [
            SimpleNamespace(content="Analysis."), SimpleNamespace(content="study planning")
        ]
        self.rag.get_intervention_advice.side_effect = RuntimeError("index unavailable")
        with self.assertRaisesRegex(MentorReportError, "guidance"):
            self.make_agent().run_student_analysis(123)
        self.assertEqual(self.llm.invoke.call_count, 2)

    def test_empty_provider_response_is_not_a_report(self):
        self.llm.invoke.return_value = SimpleNamespace(content="  ")
        with self.assertRaisesRegex(MentorReportError, "empty"):
            self.make_agent().run_student_analysis(123)

    def test_guidance_index_action_is_preserved_for_the_user(self):
        self.llm.invoke.side_effect = [
            SimpleNamespace(content="Analysis."), SimpleNamespace(content="study planning")
        ]
        self.rag.get_intervention_advice.side_effect = MentorReportError(
            "Restore 6_Models/vectorstore with the tested Chroma runtime."
        )
        with self.assertRaisesRegex(MentorReportError, "Restore 6_Models/vectorstore"):
            self.make_agent().run_student_analysis(123)
        self.assertEqual(self.llm.invoke.call_count, 2)

    def test_missing_key_does_not_generate_fallback_report(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {
            "streamlit": SimpleNamespace(secrets={}),
            "config": SimpleNamespace(GROQ_API_KEY="your_groq_api_key_here"),
        }):
            with self.assertRaisesRegex(MentorReportError, "GROQ_API_KEY"):
                pipeline.AgenticAIPipeline(groq_model=DEFAULT_GROQ_MODEL)


class ModelConfigurationTests(unittest.TestCase):
    def test_default_environment_secret_and_explicit_override(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {
            "streamlit": SimpleNamespace(secrets={})
        }):
            self.assertEqual(get_groq_model(), DEFAULT_GROQ_MODEL)
            with patch.dict(os.environ, {"GROQ_MODEL": "account-model"}):
                self.assertEqual(get_groq_model(), "account-model")
                with patch.dict(sys.modules, {"streamlit": SimpleNamespace(
                    secrets={"groq": {"GROQ_MODEL": "secret-model"}}
                )}):
                    self.assertEqual(get_groq_model(), "secret-model")
                    self.assertEqual(get_groq_model("explicit-model"), "explicit-model")


if __name__ == "__main__":
    unittest.main()
