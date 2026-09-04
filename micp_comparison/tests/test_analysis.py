import unittest

from micp_comparison.analyze import summarize


class AnalysisTest(unittest.TestCase):
    def test_success_is_not_solver_feasibility(self):
        key = ("scenario", 1, "R_G1")
        method = {key: {"success": False, "feasible": True, "time": 1.0}}
        vp = {key: {"success": True, "time": 0.1}}
        rule = summarize(method, vp)[0]
        self.assertEqual(rule["micp_feasible"], 1)
        self.assertEqual(rule["micp_success"], 0)
        self.assertEqual(rule["paired_success_n"], 0)


if __name__ == "__main__":
    unittest.main()

