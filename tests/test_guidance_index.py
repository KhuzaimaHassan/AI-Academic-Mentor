"""Reject missing/empty guidance stores and resolve the default outside repo cwd."""
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_Src_Code"))
from rag_integration import RAGIntegration
from reporting import MentorReportError


class GuidanceIndexTests(unittest.TestCase):
    def setUp(self):
        self.embedding = patch("rag_integration.HuggingFaceEmbeddings")
        self.embedding.start()
        self.addCleanup(self.embedding.stop)

    def test_default_index_path_is_independent_of_working_directory(self):
        with TemporaryDirectory() as folder:
            previous = Path.cwd()
            try:
                os.chdir(folder)
                rag = RAGIntegration()
            finally:
                os.chdir(previous)
        self.assertEqual(Path(rag.persist_directory), ROOT / "6_Models/vectorstore")

    def test_missing_database_does_not_create_an_empty_collection(self):
        with TemporaryDirectory() as folder, patch("rag_integration.Chroma") as chroma:
            rag = RAGIntegration(folder)
            with self.assertRaisesRegex(MentorReportError, "missing, empty"):
                rag.get_intervention_advice("study planning")
            chroma.assert_not_called()

    def test_empty_collection_is_rejected_before_search(self):
        with TemporaryDirectory() as folder, patch("rag_integration.Chroma") as chroma:
            (Path(folder) / "chroma.sqlite3").touch()
            chroma.return_value.get.return_value = {"ids": []}
            rag = RAGIntegration(folder)
            self.assertFalse(rag.load_vectorstore())
            self.assertIsNone(rag.vectorstore)
            chroma.return_value.as_retriever.assert_not_called()

    def test_existing_collection_is_opened_without_implicit_creation(self):
        with TemporaryDirectory() as folder, patch("rag_integration.Chroma") as chroma:
            (Path(folder) / "chroma.sqlite3").touch()
            chroma.return_value.get.return_value = {"ids": ["guidance-1"]}
            rag = RAGIntegration(folder)
            self.assertTrue(rag.load_vectorstore())
            self.assertFalse(chroma.call_args.kwargs["create_collection_if_not_exists"])


if __name__ == "__main__":
    unittest.main()
