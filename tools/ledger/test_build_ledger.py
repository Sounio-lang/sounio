"""Tests for tools/ledger/build_ledger.py producer verification.

Run: python3 -m unittest tools/ledger/test_build_ledger.py
"""
import argparse
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ledger as bl  # noqa: E402

HEAD = ["id", "linha", "afirmacao", "valor", "unidade", "repositorio", "requer",
        "comando", "espera", "oraculo", "status", "citado_em", "medido_em"]


def tsv(dirpath, rows):
    path = os.path.join(dirpath, "results.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(HEAD) + "\n")
        for i, (cmd, espera) in enumerate(rows, 1):
            f.write("\t".join([f"T{i:02d}", "L00", "teste", "", "", "sounio", "",
                               cmd, espera, "", "válido", "", ""]) + "\n")
    return path


def verify(rows):
    args = argparse.Namespace(verify=True, uhs_repo=None, gri_repo=None)
    with tempfile.TemporaryDirectory() as d:
        return bl.results_rows(args, tsv(d, rows))


class ProducerVerdict(unittest.TestCase):
    def test_exit0_and_string_is_verificado(self):
        r = verify([("printf 'LEDGER_TEST_OK\\n'; exit 0", "LEDGER_TEST_OK")])[0]
        self.assertEqual(r[6], "Verificado")

    def test_string_then_nonzero_exit_is_falhou(self):
        r = verify([("printf 'LEDGER_TEST_OK\\n'; exit 7", "LEDGER_TEST_OK")])[0]
        self.assertEqual(r[6], "Falhou")
        self.assertIn("código 7", r[7])

    def test_exit0_without_string_is_falhou(self):
        r = verify([("printf 'OTHER_OUTPUT\\n'; exit 0", "LEDGER_TEST_OK")])[0]
        self.assertEqual(r[6], "Falhou")
        self.assertIn("não encontrou", r[7])

    def test_shared_producer_runs_once_and_keeps_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            counter = os.path.join(d, "runs")
            cmd = f"echo x >> {counter}; printf 'A_OK\\nB_OK\\n'; exit 3"
            rows = verify([(cmd, "A_OK"), (cmd, "B_OK")])
            with open(counter) as f:
                self.assertEqual(len(f.readlines()), 1)  # cache reused
        self.assertEqual([r[6] for r in rows], ["Falhou", "Falhou"])
        self.assertTrue(all("código 3" in r[7] for r in rows))

    def test_shared_producer_success_verifies_both(self):
        cmd = "printf 'A_OK\\nB_OK\\n'"
        rows = verify([(cmd, "A_OK"), (cmd, "B_OK")])
        self.assertEqual([r[6] for r in rows], ["Verificado", "Verificado"])


if __name__ == "__main__":
    unittest.main()
