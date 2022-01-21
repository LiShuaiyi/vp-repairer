import math
from typing import List

from commonroad_repair.crrepairer.cut_off.tc import TC
from commonroad_repair.crrepairer.cut_off.simulation import CutOffAction
from commonroad_repair.crrepairer.abstraction.monitor import STLRuleMonitor
from commonroad_repair.crrepairer.t_solver.qp_planner import QPPlannerRepair
from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter

from stl_crmonitor.crmonitor.predicates.predicate import Category
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode


class TSolver:
    def __init__(self,
                 rule_abstracter: RuleAbstracter):
        self._sel_prop = None
        self._rule_abstracter = rule_abstracter
        self._tc_obj = TC(rule_abstracter.rule_monitor)
        self._compliant_maneuvers = list()
        self._tc_dict = dict()
        self._repairability = False
        self._qp_planner = None

    @property
    def tc_object(self):
        return self._tc_obj

    @property
    def compliant_maneuvers(self):
        return self._compliant_maneuvers

    def assign_proposition(self, propositions: List[PropositionNode], model: list):
        self._sel_prop = list()
        for prop in propositions:
            # if not the same value
            if (prop.ttv_value < 0 and prop.alphabet in model) or (prop.ttv_value > 0 and '~' + prop.alphabet in model):
                self._sel_prop.append(prop)
        self._compliant_maneuvers = self.set_compliant_maneuver()

    def set_compliant_maneuver(self):
        assert self._sel_prop is not None, "<T-Solver>: the atomic proposition needs to be " \
                                           "assigned first for the T-solver"
        compliant_maneuver = list()
        for prop_node in self._sel_prop:
            for predicate in prop_node.children:
                if not hasattr(predicate, "evaluator"):
                    continue
                predicate_category = predicate.evaluator.predicate_category
                print(predicate_category, predicate.name)
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
                    pass # general predicate
                    # raise ValueError('<T-Solver>: the category {} is not specified'
                    #                  .format(predicate_category))
        compliant_maneuver = list(set(compliant_maneuver))
        print("<TSolver>: compliant maneuver {} is selected".format(compliant_maneuver))
        return compliant_maneuver

    def search_tc(self):
        if self._compliant_maneuvers is None:
            print("<TSolver>: the compliant maneuver is not specified")
            return -math.inf
        tc = self.tc_object.generate(self._compliant_maneuvers)
        return tc

    def _optimization_based_repair(self):
        self._qp_planner = QPPlannerRepair(self._rule_abstracter,
                                           self._tc_obj,
                                           self._sel_prop)
        repaired_trajectory = self._qp_planner.plan()
        return repaired_trajectory

    def check(self, proposition: List[PropositionNode], model: list):
        repaired_traj = None
        self.assign_proposition(proposition, model)
        if self.compliant_maneuvers is None:
            return self._repairability, repaired_traj
        tc = self.search_tc()
        print("<T-solver>: tc = {}, tv = {}".format(self._tc_obj.tc, self._tc_obj.tv))

        assert tc != math.inf, "<T-solver>: the trajectory is already rule-compliant," \
                               "i.e., doesn't need to be repaired"
        if tc != -math.inf:
            repaired_traj = self._optimization_based_repair()
            if repaired_traj is not None:
                self._repairability = True

        return self._repairability, repaired_traj




