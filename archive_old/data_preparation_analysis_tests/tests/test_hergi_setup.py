"""Tests rapides de la configuration de travail de Hergi.

Ces tests ne lancent aucun entraînement ARIMA/SARIMA/SARIMAX.
"""

import ast
import csv
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERGI_SCRIPTS = PROJECT_ROOT / "hergi" / "scripts"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"


class TestHergiSetup(unittest.TestCase):
    def test_model_scripts_have_valid_python_syntax(self):
        scripts = sorted(HERGI_SCRIPTS.rglob("*.py"))
        names = {script.name for script in scripts}
        required = {
            "sarima_all_9892_fair_comparison.py",
            "sarimax_all_9892_compact.py",
            "generate_sarimax_9892_report.py",
        }
        self.assertTrue(required.issubset(names))
        for script in scripts:
            with self.subTest(script=script.name):
                ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

    def test_first_point_dataset(self):
        path = PROCESSED_DATA / "first_point_T_daily.csv"
        self.assertTrue(path.is_file(), f"Donnée absente : {path}")
        self.assertTrue({"DATE", "T"}.issubset(self._columns(path)))

    def test_safran_daily_dataset(self):
        path = PROCESSED_DATA / "safran_quot_clean.csv"
        self.assertTrue(path.is_file(), f"Donnée absente : {path}")
        required = {"LAMBX", "LAMBY", "DATE", "T", "PRELIQ", "PRENEI", "FF", "HU", "SSI"}
        self.assertTrue(required.issubset(self._columns(path)))

    @staticmethod
    def _columns(path):
        with path.open("r", encoding="utf-8", newline="") as stream:
            return set(next(csv.reader(stream)))


if __name__ == "__main__":
    unittest.main()
