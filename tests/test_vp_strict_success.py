from types import SimpleNamespace

from z3 import sat, unsat

from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer


class FakeSatSolver:
    solver_mode = "dpll"

    def __init__(self, model_count):
        self.remaining = model_count
        self.update_count = 0

    def solve(self):
        return sat if self.remaining > 0 else unsat

    def model(self):
        self.remaining -= 1
        return [], []

    def update_formula(self):
        self.update_count += 1


def make_repairer(candidate_tvs):
    repairer = object.__new__(VPTrajectoryRepairer)
    repairer.rule_monitor = SimpleNamespace(tv_time_step=1, proposition_nodes=[])
    repairer.sat_solver = FakeSatSolver(len(candidate_tvs))
    repairer.nr_iter = 0
    repairer.sat_reasoning_time = 0.0
    repairer.domain_dict_time = 0.0
    repairer._domain_dict_initialized = False
    repairer._model = None
    repairer._tc = 0
    repairer._tv = -1
    repairer._assign_proposition = lambda *_args: None
    trajectories = [SimpleNamespace(state_list=[]) for _ in candidate_tvs]
    repairer._repair_with_velocity_planning = lambda: trajectories.pop(0)
    tvs = list(candidate_tvs)
    repairer.calc_tv_updated = lambda *_args: (tvs.pop(0), None)
    return repairer


def test_repair_rejects_finite_tv_and_accepts_later_compliant_candidate():
    repairer = make_repairer([0.8, float("inf")])

    result = repairer.repair()

    assert result is not None
    assert repairer.nr_iter == 2
    assert repairer.candidate_tvs == [0.8, float("inf")]
    assert repairer.sat_solver.update_count == 1


def test_repair_returns_none_when_all_candidates_have_finite_tv():
    repairer = make_repairer([0.4, 1.2])

    result = repairer.repair()

    assert result is None
    assert repairer.nr_iter == 2
    assert repairer.candidate_tvs == [0.4, 1.2]
    assert repairer.sat_solver.update_count == 2


def test_check_flag_false_preserves_unchecked_repair_mode():
    repairer = make_repairer([0.4])

    result = repairer.repair(check_flag=False)

    assert result is not None
    assert repairer.nr_iter == 1
    assert repairer.candidate_tvs == []
    assert repairer.sat_solver.update_count == 0
