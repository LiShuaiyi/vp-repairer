from cut_off.tc import TC


class TSolver:
    def __init__(self,
                 sel_subform,
                 rule_monitor):
        self._sel_subform = sel_subform
        self._tc_obj = TC(rule_monitor)