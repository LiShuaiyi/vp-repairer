import math

from cut_off.tc import TC
from cut_off.simulation import CutOffAction

from stl_crmonitor.crmonitor.predicates.predicate import Category
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode


class TSolver:
    def __init__(self,
                 rule_monitor):
        self._sel_prop = PropositionNode()
        self._tc_obj = TC(rule_monitor)
        self._compliant_maneuvers = list()
        self._tc_dict = dict()
        self._repairability = False

    @property
    def compliant_maneuvers(self):
        return self._compliant_maneuvers

    def assign_proposition(self, proposition: PropositionNode):
        self._sel_prop = proposition
        self.set_compliant_maneuver()

    def set_compliant_maneuver(self):
        assert self._sel_prop is not None, "<T-Solver>: the atomic proposition needs to be " \
                                           "assigned first for the T-solver"
        for predicate in self._sel_prop.children:
            predicate_category = predicate.evaluator.predicate_category
            if predicate_category == Category.LON_POS:
                self._compliant_maneuvers = [CutOffAction.BRAKE,
                                             CutOffAction.KICKDOWN]
            elif predicate_category == Category.LAT_POS:
                self._compliant_maneuvers = [CutOffAction.LANECHANGELEFT,
                                             CutOffAction.LANECHANGERIGHT]
                # todo: set the offset for steer
            elif predicate_category == Category.VEL:
                self._compliant_maneuvers = [CutOffAction.BRAKE,
                                             CutOffAction.KICKDOWN]
            elif predicate_category == Category.ACC:
                self._compliant_maneuvers = [CutOffAction.STEADYSPEED]
            else:
                raise ValueError('<T-Solver>: the category {} is not specified'
                                 .format(predicate_category))

    def search_tc(self):
        tc_list = list()
        for maneuver in self._compliant_maneuvers:
            if maneuver not in self._tc_dict.keys():
                tc_maneuver = self._tc_obj.generate(maneuver)
                tc_list.append(tc_maneuver)
                self._tc_dict[maneuver] = tc_maneuver
            else:
                tc_list.append(self._tc_dict[maneuver])
        return max(tc_list)

    def _optimization_based_repair(self):
        repaired_trajectory = None
        return repaired_trajectory

    def check(self, proposition: PropositionNode):
        self.assign_proposition(proposition)
        tc = self.search_tc()
        assert tc != math.inf, "<T-solver>: the trajectory is already rule-compliant," \
                               "i.e., doesn't need to be repaired"
        if tc != -math.inf:
            repaired_traj = self._optimization_based_repair()
            if repaired_traj is not None:
                self._repairability = True
        return self._repairability, repaired_traj




