from cut_off.tc import TC
from cut_off.simulation import CutOffAction

from stl_crmonitor.crmonitor.predicates.predicate import Category
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode


class TSolver:
    def __init__(self,
                 rule_monitor):
        self._sel_prop: PropositionNode = None
        self._tc_obj = TC(rule_monitor)
        self._compliant_maneuvers = set()
        self._tc_dict = dict()

    @property
    def compliant_maneuvers(self):
        return self._compliant_maneuvers

    def assign_proposition(self, proposition: PropositionNode):
        self._sel_prop = proposition
        self.set_compliant_maneuver()

    def set_compliant_maneuver(self):
        assert self._sel_prop is not None, "<T-Solver>: the subformula needs to be assigned first for the T-solver"
        for predicate in self._sel_prop.children:
            predicate_category = predicate.evaluator.predicate_category
            if predicate_category is Category.POS:
                self._compliant_maneuvers.update([CutOffAction.BRAKE, CutOffAction.KICKDOWN,
                                                  CutOffAction.LANECHANGELEFT,
                                                  CutOffAction.LANECHANGERIGHT])
                # todo: set the offset for steer
            elif predicate_category == Category.VEL:
                self._compliant_maneuvers.update([CutOffAction.BRAKE, CutOffAction.KICKDOWN])
            elif predicate_category == Category.ACC:
                self._compliant_maneuvers.update([CutOffAction.STEADYSPEED])
            else:
                raise ValueError('the category {} is not specified'.format(predicate_category))