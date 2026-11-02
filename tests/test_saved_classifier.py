"""Run real inference with the repository's committed model and encoders."""

import os
from pathlib import Path
import sys
import unittest
import warnings

import sklearn
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_Src_Code"))
from utils.helpers import StudentPredictionHelper


class SavedClassifierTests(unittest.TestCase):
    def test_committed_artifacts_load_and_predict_with_their_training_version(self):
        self.assertEqual(sklearn.__version__, "1.6.1")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            helper = StudentPredictionHelper(
                models_dir=str(ROOT / "6_Models"), data_path=str(ROOT / "Data_csv's")
            )
            self.assertIsNotNone(helper.model)
            self.assertIsNotNone(helper.target_encoder)
            for student_id, expected in [(3733, "Withdrawn"), (6516, "Pass")]:
                with self.subTest(student_id=student_id):
                    self.assertEqual(helper.predict_student_result(student_id), expected)


if __name__ == "__main__":
    unittest.main()
