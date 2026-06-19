import os
import math
import unittest
from sympy.logic.boolalg import is_cnf, is_dnf, is_nnf

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.smt.sat_solver.sat_solver import SATSolver
from crrepairer.smt.sat_solver.dpll import DPLL
from crrepairer.smt.t_solver.t_solver import TSolver
from crrepairer.smt.t_solver.qp_planner_repair import QPPlannerRepair

from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_crime.utility.simulation import Maneuver

from z3 import sat, unsat


class TestSMTSolver(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(
            self.scenario_root_path, "test_interstate/DEU_test_safe_distance.xml"
        )
        self.scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(
            lanelet_assignment=True
        )
        # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
        self.planning_problem = list(
            planning_problem_set.planning_problem_dict.values()
        )[0]
        ego_id = 1003
        self._ego_obs = self.scenario.obstacle_by_id(ego_id)
        rule = "R_G1"
        self.rule_monitor = STLRuleMonitor(self.scenario, ego_id, rule)

    def test_construction(self):
        self.assertEqual(len(self.rule_monitor.proposition_nodes), 4)
        rule_monitor = self.rule_monitor
        for i, node in enumerate(self.rule_monitor.proposition_nodes):
            self.assertEqual(rule_monitor.prop_robust_ttv[i], node.ttv_value)
        exp_compliance = False
        # rob_value = rule_monitor.prop_robust_all[rule_monitor.prop_robust_all>=0.0]
        rob_value = all([r >= 0.0 for r in rule_monitor.prop_robust_all.flatten()])
        self.assertEqual(
            exp_compliance,
            rob_value,
        )
        self.assertEqual(rule_monitor.other_id, 1004)

    def test_sat_solver(self):
        sat_solver = SATSolver(self.rule_monitor)
        # check whether the formula in the sat solver is CNF or not
        self.assertTrue(is_cnf(sat_solver.formula))
        sat_re = sat_solver.solve()
        self.assertEqual(sat_re, sat)
        _, m = sat_solver.model()
        self.assertEqual(list(m), ["a"])
        abstraction_nodes = self.rule_monitor.proposition_nodes
        # after negating all the possible solutions
        while len(m) != 0:
            sat_solver.update_formula()
            sat_re = sat_solver.solve()
            _, m = sat_solver.model()
            print(sat_solver.formula)
        self.assertEqual(sat_re, unsat)

    def test_t_solver(self):
        t_solver = TSolver(self._ego_obs, self.planning_problem, self.rule_monitor)
        proposition = next(
            (
                prop
                for prop in list(self.rule_monitor.proposition_nodes)
                if prop.name == "keeps_safe_distance_prec__0_1"
            ),
            None,
        )
        t_solver.assign_proposition([proposition], ["a"])
        # safe distance
        self.assertEqual(
            set(t_solver.compliant_maneuvers), {Maneuver.BRAKE, Maneuver.KICKDOWN}
        )
        tc = t_solver.search_tc()
        assert math.isclose(tc, 1.9, abs_tol=1e-2)
        proposition = next(
            (
                prop
                for prop in list(self.rule_monitor.proposition_nodes)
                if prop.name == "in_same_lane__0_1"
            ),
            None,
        )
        t_solver.assign_proposition([proposition], ["~c"])
        tc = t_solver.search_tc()
        assert math.isclose(tc, 0.4, abs_tol=1e-2)

    def test_dpll(self):
        dpll_solver = DPLL("~a | ~b | c | d", self.rule_monitor.proposition_nodes)
        self.assertEqual(dpll_solver.solve(), sat)
        self.assertEqual(list(dpll_solver.model), ["~a"])
        dpll_solver.update_cnf("~a & a")
        self.assertEqual(dpll_solver.solve(), unsat)
        self.assertEqual(dpll_solver.model, set())

    def test_cnf_dnf_nnf_converter(self):
        original_formula = "(a and b and !c) implies d"
        sat_solver = SATSolver(self.rule_monitor)
        cnf_formula = sat_solver.construct_cnf(original_formula)
        self.assertTrue(is_cnf(cnf_formula))
        dnf_formula = sat_solver.construct_dnf(original_formula)
        self.assertTrue(is_dnf(dnf_formula))
        nnf_formula = sat_solver.construct_nnf(original_formula)
        self.assertTrue(is_nnf(nnf_formula))

    def test_construct_qp_repair(self):
        t_solver = TSolver(self._ego_obs, self.planning_problem, self.rule_monitor)
        proposition = next(
            (
                prop
                for prop in list(self.rule_monitor.proposition_nodes)
                if prop.name == "keeps_safe_distance_prec__0_1"
            ),
            None,
        )
        assign_prop = [proposition]
        t_solver.assign_proposition(assign_prop, ["a"])
        t_solver.search_tc()
        tc_object = t_solver.tc_object
        qp_repairer = QPPlannerRepair(
            self.rule_monitor,
            tc_object,
            assign_prop,
            assign_prop,
            self.planning_problem,
        )
        self.assertIsInstance(qp_repairer, QPPlannerRepair)
        qp_repairer.rule_constraints.add()  # add constraints
        safe_distance_modes_t = [
            True for _ in range(tc_object.N_p - tc_object.tc_time_step + 1)
        ]  # tc + 1 ?
        self.assertEqual(
            qp_repairer.rule_constraints.safe_distance_modes, safe_distance_modes_t
        )
        self.assertEqual(
            len(qp_repairer.rule_constraints.safe_distance_modes),
            qp_repairer.total_time_steps + 1,
        )

    def test_rule_constraints(self):
        t_solver = TSolver(self._ego_obs, self.planning_problem, self.rule_monitor)
        proposition = next(
            (
                prop
                for prop in list(self.rule_monitor.proposition_nodes)
                if prop.name == "in_same_lane__0_1"
            ),
            None,
        )
        assign_prop = [proposition]
        t_solver.assign_proposition(assign_prop, ["~c"])
        t_solver.search_tc()
        tc_object = t_solver.tc_object
        qp_repairer = QPPlannerRepair(
            self.rule_monitor,
            tc_object,
            assign_prop,
            assign_prop,
            self.planning_problem,
        )
        qp_repairer.rule_constraints.add()  # add constraints
        for time_step, lanes in qp_repairer.rule_constraints.target_lanes.items():
            if time_step <= qp_repairer.rule_constraints.time_leave_lane:
                self.assertEqual(lanes, [self.rule_monitor.world.road_network.lanes[1]])
            elif time_step <= tc_object.tv_time_step:
                self.assertEqual(
                    set(lanes),
                    {
                        self.rule_monitor.world.road_network.lanes[1],
                        self.rule_monitor.world.road_network.lanes[2],
                    },
                )
            else:
                self.assertEqual(lanes, [self.rule_monitor.world.road_network.lanes[2]])
        # qp_repairer.rule_constraints.safe_distance_modes
