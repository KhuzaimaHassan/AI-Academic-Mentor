"""Exercise the Streamlit error path without credentials or provider calls."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_Src_Code"))
from reporting import MentorReportError


class ReportInterfaceTests(unittest.TestCase):
    def test_provider_failure_has_no_report_download(self):
        agent = Mock()
        agent.run_student_analysis.side_effect = MentorReportError("Groq model unavailable. Set GROQ_MODEL.")
        with patch("agentic_ai_pipeline.AgenticAIPipeline", return_value=agent):
            app = AppTest.from_file(str(ROOT / "9_Deployment/app.py"))
            app.secrets["groq"] = {"GROQ_MODEL": "test-model"}
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("GROQ_MODEL" in error.value for error in app.error))
            self.assertEqual(len(app.get("download_button")), 0)

    def test_success_exposes_report_download(self):
        agent = Mock(model_name="test-model")
        agent.run_student_analysis.return_value = (
            "# Mentor Report\n\n| Step | Action |\n| --- | --- |\n| 1 | Review a study plan |"
            "\n\nSource excerpt:\n\n    1. Create a Study Schedule\n"
            "    - Include breaks\n\n    2. Set Clear Goals\n"
            "    - Use <script> as literal source text\n"
        )
        with patch("agentic_ai_pipeline.AgenticAIPipeline", return_value=agent):
            app = AppTest.from_file(str(ROOT / "9_Deployment/app.py"))
            app.secrets["groq"] = {"GROQ_MODEL": "test-model"}
            app.run(timeout=30)
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0, [error.value for error in app.error])
            self.assertEqual(len(app.get("download_button")), 1)
            self.assertTrue(any("<table>" in block.value for block in app.markdown))
            report_html = next(block.value for block in app.markdown if "<table>" in block.value)
            self.assertIn('<div class="report-source-block">', report_html)
            self.assertNotIn("<pre><code>", report_html)
            self.assertIn("2. Set Clear Goals", report_html)
            self.assertIn("&amp;lt;script&amp;gt;", report_html)
            self.assertNotIn("<script>", report_html)


if __name__ == "__main__":
    unittest.main()
