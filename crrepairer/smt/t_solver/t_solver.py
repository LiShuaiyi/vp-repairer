import math
import time
from typing import List

from crrepairer.cut_off.tc import TC
from crrepairer.smt.t_solver.qp_planner_repair import QPPlannerRepair
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode

from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.planning.planning_problem import PlanningProblem

from commonroad_crime.utility.simulation import Maneuver

from crmonitor.predicates.position import PositionPredicates


class TSolver:
    """
    T-solver for the SMT-based repairer.
    """
    def __init__(self,
                 ego_vehicle: DynamicObstacle,
                 planning_problem: PlanningProblem,
                 rule_monitor: STLRuleMonitor):
        self._sel_prop = None
        self._prop_full = None
        self._rule_monitor = rule_monitor
        self._tc_obj = TC(ego_vehicle, rule_monitor)
        self._compliant_maneuvers = list()
        self._repairability = False
        self._qp_planner = None
        self._planning_problem = planning_problem

        self.verbose = False

    @property
    def tc_object(self):
        return self._tc_obj

    @property
    def compliant_maneuvers(self):
        return self._compliant_maneuvers

    def assign_proposition(self, propositions: List[PropositionNode], model: list):
        """
        Assigns propositions to the T-solver.
        """
        self._prop_full = propositions
        self._sel_prop = list()
        for prop in propositions:
            # if not the same value
            if (prop.ttv_value < 0 and prop.alphabet in model) or\
                    (prop.ttv_value > 0 and '~' + prop.alphabet in model):
                self._sel_prop.append(prop)
        self._compliant_maneuvers = self.set_compliant_maneuver()

    def set_compliant_maneuver(self):
        """
        Set rul-compliant maneuvers based on the selected propositions.
        """
        assert self._sel_prop is not None, "<T-Solver>: the atomic proposition needs to be " \
                                           "assigned first for the T-solver"
        compliant_maneuver = list()
        for prop_node in self._sel_prop:
            for predicate in prop_node.children:
                if not hasattr(predicate, "evaluator"):
                    continue
                predicate_category = predicate.evaluator.predicate_name.__class__.__name__[:3]
                if predicate_category == "Pos" and \
                        predicate.evaluator.predicate_name in [PositionPredicates.KeepsSafeDistancePrec,
                                                               PositionPredicates.InFrontOf,
                                                               PositionPredicates.Precedes]:
                    compliant_maneuver += [Maneuver.BRAKE, Maneuver.KICKDOWN]
                elif predicate_category == "Pos" and \
                        predicate.evaluator.predicate_name in [PositionPredicates.StopLineInFront]:
                    compliant_maneuver += [Maneuver.BRAKE]
                elif predicate_category == "Pos":
                    compliant_maneuver += [Maneuver.STEERRIGHT,
                                           Maneuver.STEERLEFT]
                elif predicate_category == "Vel":
                    compliant_maneuver += [Maneuver.BRAKE,
                                           Maneuver.KICKDOWN]
                elif predicate_category == "Acc":
                    compliant_maneuver += [Maneuver.CONSTANT]
                else:
                    pass  # general predicate
                    # raise ValueError('<T-Solver>: the category {} is not specified'
                    #                  .format(predicate_category))
        compliant_maneuver = list(set(compliant_maneuver))
        if not compliant_maneuver:
            print("* \t<TSolver>: no compliant maneuver is selected")
        else:
            string = "* \t<TSolver>: compliant maneuver /"
            for m in compliant_maneuver:
                string += m.value + '/'
            string += " is selected"
            print(string)
        return compliant_maneuver

    def search_tc(self):
        """
        Searches the time-to-compliance.
        """
        if self._compliant_maneuvers is None:
            return -math.inf  # marked as not repairable
        tc = self.tc_object.generate(self._compliant_maneuvers)
        return tc

    def _optimization_based_repair(self):
        """
        Initializes the qp planner and uses it for trajectory repairing.
        """
        self._qp_planner = QPPlannerRepair(self._rule_monitor,
                                           self._tc_obj,
                                           self._sel_prop,
                                           self._prop_full,
                                           self._planning_problem,
                                           verbose=self.verbose)
        start_time = time.time()
        repaired_trajectory = self._qp_planner.plan()
        print(f"* \t<TSolver>: solving time {time.time()-start_time:.3f}s")
        return repaired_trajectory

    def check(self, proposition: List[PropositionNode], model: list) -> (bool, Trajectory):
        """
        Checks the T-consistency.
        """
        repaired_traj = None
        self.assign_proposition(proposition, model)
        if self.compliant_maneuvers is None:
            print("* \t<Tsolver>: tc = {}, tv = {}".format(-math.inf, -math.inf))
            return self._repairability, repaired_traj
        start_time = time.time()
        tc = self.search_tc()
        print("* \t<Tsolver>: tc = {}, tv = {}".format(self._tc_obj.tc, self._tc_obj.tv))
        print(f"* \t<Tsolver>: run time {time.time() - start_time:.3f}s")
        if tc != -math.inf:
            repaired_traj = self._optimization_based_repair()
            if repaired_traj is not None:
                self._repairability = True
        return self._repairability, repaired_traj




