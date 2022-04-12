import math
import time
from typing import List

from commonroad_repair.crrepairer.cut_off.tc import TC
from commonroad_repair.crrepairer.cut_off.simulation import CutOffAction
from commonroad_repair.crrepairer.smt.t_solver.qp_planner_repair import QPPlannerRepair
from commonroad_repair.crrepairer.smt.monitor_wrapper import STLRuleMonitor

from stl_crmonitor.crmonitor.predicates.predicate import Category
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode

from commonroad.scenario.trajectory import Trajectory


class TSolver:
    """
    T-solver for the SMT-based repairer.
    """
    def __init__(self,
                 rule_monitor: STLRuleMonitor):
        self._sel_prop = None
        self._rule_monitor = rule_monitor
        self._tc_obj = TC(rule_monitor)
        self._compliant_maneuvers = list()
        self._repairability = False
        self._qp_planner = None

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
                predicate_category = predicate.evaluator.predicate_category
                if predicate_category == Category.LON_POS:
                    compliant_maneuver += [CutOffAction.BRAKE, CutOffAction.KICKDOWN]
                elif predicate_category == Category.LAT_POS:
                    compliant_maneuver += [CutOffAction.LANECHANGELEFT,
                                           CutOffAction.LANECHANGERIGHT]
                elif predicate_category == Category.VEL:
                    compliant_maneuver += [CutOffAction.BRAKE,
                                           CutOffAction.KICKDOWN]
                elif predicate_category == Category.ACC:
                    compliant_maneuver += [CutOffAction.STEADYSPEED]
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
                                           self._sel_prop)
        start_time = time.time()
        repaired_trajectory = self._qp_planner.plan()
        print("* \t<TSolver>: solving time {}".format(time.time()-start_time))
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
        tc = self.search_tc()
        print("* \t<Tsolver>: tc = {}, tv = {}".format(self._tc_obj.tc, self._tc_obj.tv))

        if tc != -math.inf:
            repaired_traj = self._optimization_based_repair()
            if repaired_traj is not None:
                self._repairability = True
        return self._repairability, repaired_traj




