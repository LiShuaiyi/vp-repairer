from cut_off.tc import TC
from cut_off.simulation import CutOffAction


class TSolver:
    def __init__(self,
                 rule_monitor):
        self._sel_subform = None
        self._tc_obj = TC(rule_monitor)
        self._compliant_maneuvers = list()
        self._tc_dict = dict()

    def assign_subformula(self, subformula):
        self._sel_subform = subformula

    def set_compliant_maneuver(self):
        assert self._sel_subform is not None, "<T-Solver>: the subformula needs to be assigned first for the T-solver"
        pass