import math
from typing import List

from cut_off.tc import TC
from cut_off.simulation import CutOffAction
from abstraction.monitor import STLRuleMonitor
from commonroad_repair.crrepairer.t_solver.qp_planner import QPPlannerRepair
from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter

from stl_crmonitor.crmonitor.predicates.predicate import Category
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode


class TSolver:
    def __init__(self,
                 rule_abstracter: RuleAbstracter):
        self._rule_absracter = rule_abstracter
        self._sel_prop = None
        self._tc_obj = TC(rule_abstracter.rule_monitor)
        self._compliant_maneuvers = list()
        self._tc_dict = dict()
        self._repairability = False

    @property
    def tc_object(self):
        return self._tc_obj

    @property
    def compliant_maneuvers(self):
        return self._compliant_maneuvers

    def assign_proposition(self, proposition: PropositionNode):
        self._sel_prop = proposition
        self._compliant_maneuvers = self.set_compliant_maneuver()

    def set_compliant_maneuver(self):
        assert self._sel_prop is not None, "<T-Solver>: the atomic proposition needs to be " \
                                           "assigned first for the T-solver"
        compliant_maneuver = list()
        for prop_node in self._sel_prop:
            for predicate in prop_node.children:
                predicate_category = predicate.evaluator.predicate_category
                if predicate_category == Category.LON_POS:
                    compliant_maneuver += [CutOffAction.BRAKE, CutOffAction.KICKDOWN]
                elif predicate_category == Category.LAT_POS:
                    compliant_maneuver += [CutOffAction.LANECHANGELEFT,
                                           CutOffAction.LANECHANGERIGHT]
                    # todo: set the offset for steer
                elif predicate_category == Category.VEL:
                    compliant_maneuver += [CutOffAction.BRAKE,
                                           CutOffAction.KICKDOWN]
                elif predicate_category == Category.ACC:
                    compliant_maneuver += [CutOffAction.STEADYSPEED]
                else:
                    raise ValueError('<T-Solver>: the category {} is not specified'
                                     .format(predicate_category))
        return compliant_maneuver

    def search_tc(self):
        tc = self.tc_object.generate(self._compliant_maneuvers)
        return tc

    def _optimization_based_repair(self):
        qp_planner = QPPlannerRepair(self._rule_absracter,
                                     self._tc_obj,
                                     self._sel_prop)
        repaired_trajectory = qp_planner.plan()
        return repaired_trajectory

    def check(self, proposition: PropositionNode):
        repaired_traj = None
        self.assign_proposition(proposition)
        tc = self.search_tc()
        assert tc != math.inf, "<T-solver>: the trajectory is already rule-compliant," \
                               "i.e., doesn't need to be repaired"
        if tc != -math.inf:
            repaired_traj = self._optimization_based_repair()
            if repaired_traj is not None:
                self._repairability = True
        return self._repairability, repaired_traj




