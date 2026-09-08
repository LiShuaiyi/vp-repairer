import copy
import math
import time
import torch
from typing import List, Optional
try:
    from gurobipy import GurobiError
except Exception:  # pragma: no cover - optional dependency at runtime
    GurobiError = Exception

from crrepairer.cut_off.tc import TC
from crrepairer.smt.t_solver.qp_planner_repair import QPPlannerRepair
from crrepairer.smt.t_solver.miqp_planner_repair import MIQPPlannerRepair
# from crrepairer.smt.t_solver.clrrt_planner_repair import CLRRTPlannerRepair
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode
from crrepairer.utils.configuration import RepairerConfiguration

from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.planning.planning_problem import PlanningProblem

from commonroad_crime.utility.simulation import Maneuver

from crmonitor.predicates.position import PositionPredicates

tolerance = 1e-2 # tolerance for the gradient-based decision making

class TSolver:
    """
    T-solver for the SMT-based repairer.
    """

    def __init__(
        self,
        ego_vehicle: DynamicObstacle,
        rule_monitor: STLRuleMonitor,
        config: RepairerConfiguration
    ):
        self._sel_prop = None
        self._prop_full = None
        self._rule_monitor = rule_monitor
        self._tc_obj = TC(ego_vehicle, rule_monitor)
        self._compliant_maneuvers = list()
        self._repairability = False
        self._planner = None
        self._miqp_planner = None
        self._planning_problem = config.planning_problem

        self.verbose = True
        self.config = config
        self.requested_planner = config.repair.planner
        self.effective_planner = config.repair.planner
        self.planner_fallback_reason = ""

        # todo: same for QP
        if config.repair.planner == 1:
            # self._planner: Optional[MIQPPlannerRepair, QPPlannerRepair, CLRRTPlannerRepair] = QPPlannerRepair(
            self._planner: Optional[MIQPPlannerRepair, QPPlannerRepair] = QPPlannerRepair(
                self._rule_monitor,
                self._tc_obj,
                self.config,
                verbose=self.verbose,
            )
        elif config.repair.planner == 2:
            # self._planner: Optional[MIQPPlannerRepair, QPPlannerRepair, CLRRTPlannerRepair] = MIQPPlannerRepair(
            try:
                self._planner: Optional[MIQPPlannerRepair, QPPlannerRepair] = MIQPPlannerRepair(
                    self._rule_monitor,
                    self._tc_obj,
                    self.config
                )
            except GurobiError as err:
                if not getattr(config.repair, "allow_planner_fallback", True):
                    raise RuntimeError(
                        "requested MIQP planner is unavailable; refusing to "
                        f"substitute QP in a fixed-configuration run: {err}"
                    ) from err
                print(f"* \t<TSolver>: MIQP planner is unavailable ({err}); falling back to QP planner.")
                self.effective_planner = 1
                self.planner_fallback_reason = str(err)
                self._planner = QPPlannerRepair(
                    self._rule_monitor,
                    self._tc_obj,
                    self.config,
                    verbose=self.verbose,
                )
        # elif config.repair.planner == 3:
        #     self._planner: Optional[MIQPPlannerRepair, QPPlannerRepair, CLRRTPlannerRepair] = CLRRTPlannerRepair(
        #         self._rule_monitor,
        #         self._tc_obj,
        #         self.config
        #     )
        else:
            assert AssertionError("the given planner type is not supported")

        self.tc_search_time = 0
        self.reach_set_time = 0
        self.opti_plan_time = 0

    @property
    def total_runtime(self):
        return self.tc_search_time + self.reach_set_time + self.opti_plan_time

    @property
    def tc_object(self):
        return self._tc_obj

    @property
    def compliant_maneuvers(self):
        return self._compliant_maneuvers

    @property
    def selected_propositions(self):
        return tuple(self._sel_prop or ())

    def failed_semantic_core(self):
        """Return a small failed core for repeated equivalent theory models."""
        conflict_literals = [
            proposition.alphabet
            for proposition in self.selected_propositions
            if proposition.alphabet.startswith("~")
            and "in_intersection_conflict_area__0_1" in proposition.name
        ]
        if not conflict_literals:
            # If none of the violated literals has an implemented maneuver,
            # changing unrelated SAT facts cannot make this literal executable.
            # Block one such literal as a capability-level core so DPLL moves
            # on instead of enumerating all combinations around it.
            if self._compliant_maneuvers == [] and self.selected_propositions:
                return [
                    min(
                        self.selected_propositions,
                        key=lambda proposition: (
                            abs(float(proposition.ttv_h_min)), proposition.name
                        ),
                    ).alphabet
                ]
            return None
        # Prefer the direct conflict proposition over temporal duplicates.
        return [
            min(
                conflict_literals,
                key=lambda literal: next(
                    (
                        proposition.name.count("once")
                        for proposition in self.selected_propositions
                        if proposition.alphabet == literal
                    ),
                    999,
                ),
            )
        ]

    def assign_proposition(self, propositions: List[PropositionNode], model: list, use_mpr_derivative: bool):
        """
        Assigns propositions to the T-solver.
        """
        self._prop_full = propositions
        self._sel_prop = list()
        for prop in propositions:
            # if not the same value
            if prop is not None and prop.alphabet in model:
                if (prop.ttv_value < 0 and prop.alphabet[0] != "~") or (
                    prop.ttv_value > 0 and prop.alphabet[0] == "~"
                ):
                    # PropositionNode instances are reused for both polarities by
                    # the SAT layer.  Freeze the selected literal here; otherwise
                    # a later occurrence can mutate ``alphabet`` and silently
                    # invert the branch seen by TC/constraint extraction.
                    selected_prop = copy.copy(prop)
                    selected_prop.alphabet = prop.alphabet
                    self._sel_prop.append(selected_prop)
                    if self.verbose:
                        print(
                            f"* \t<TSolver>: selected propositions: {selected_prop.alphabet} "
                            f"{selected_prop.name} = {selected_prop.ttv_value}"
                        )
        self._tc_obj._selected_conflict_avoidance_predicates = [
            predicate
            for prop in self._sel_prop
            if prop.alphabet.startswith("~")
            and "in_intersection_conflict_area__0_1" in prop.name
            for predicate in prop.children
        ]
        self._compliant_maneuvers = self.set_compliant_maneuver(use_mpr_derivative)

    def set_compliant_maneuver(self, use_mpr_derivative: bool):
        """
        Set rule-compliant maneuvers based on the selected propositions.
        """
        assert self._sel_prop is not None, (
            "<T-Solver>: the atomic proposition needs to be "
            "assigned first for the T-solver"
        )
        compliant_maneuver = list()
        for prop_node in self._sel_prop:
            if prop_node.name.startswith("previous") or (
                "historically" in prop_node.name
                and getattr(prop_node, "vp_decomposition_group", None) is None
            ):
                continue
            for predicate in prop_node.children:
                if not hasattr(predicate, "evaluator"):
                    continue
                # category of the predicate
                predicate_category = (
                    predicate.evaluator.predicate_name.__class__.__name__[:3]
                )
                if use_mpr_derivative:
                    # if prop_node.name == predicate.name:
                        # value at TV
                    if predicate.mpr_gradient is None:
                        return [Maneuver.BRAKE]
                    if torch.cuda.is_available():
                        grad_tensor = predicate.mpr_gradient[0]
                    else:
                        grad_tensor = predicate.mpr_gradient
                    # Check dtype and convert if necessary
                    if grad_tensor.dtype != torch.float64:
                        grad_tensor = grad_tensor.float()
                    else:
                        grad_tensor = grad_tensor.double()

                    # Detach, move to CPU, and convert to numpy
                    grad_list = grad_tensor.detach().cpu().numpy()
                    print(f"* predicate: {predicate.evaluator.predicate_name}")
                    if (predicate_category == "Pos" and
                        predicate.evaluator.predicate_name in [PositionPredicates.KeepsSafeDistancePrec,
                                                               PositionPredicates.InFrontOf,
                                                               PositionPredicates.Precedes,
                                                               PositionPredicates.StopLineInFront,
                                                               PositionPredicates.InIntersectionConflictArea]) or \
                            (predicate_category == "Vel"):
                        grad_a = grad_list[4]
                        print(f"* gradient list: {grad_list}")
                        print(f"* gradient towards lon input: {grad_a}")
                        print(f"Multiplied value: {abs(predicate.latest_value * grad_a)}, Tolerance: {tolerance}")
                        if abs(predicate.latest_value * grad_a) <= tolerance:  # no decision can be made
                            compliant_maneuver += [Maneuver.BRAKE,
                                                   Maneuver.KICKDOWN]
                            print("* \t<TSolver>: no decision can be made, both maneuvers are selected")
                        # positive to negative, robustness needs to be decreased (Delta rob < 0)
                        # negative to positive, robustness needs to be increased (Delta rob > 0)
                        elif - predicate.latest_value * grad_a > 0:  # delta v > 0
                            compliant_maneuver += [Maneuver.KICKDOWN]
                        else:  # delta v < 0
                            compliant_maneuver += [Maneuver.BRAKE]
                    elif predicate_category == "Pos":
                        grad_theta = grad_list[9]
                        print(f"* gradient list: {grad_list}")
                        print(f"* gradient towards lat input: {grad_theta}")
                        print(f"Multiplied value: {abs(predicate.latest_value * grad_theta)}, Tolerance: {tolerance}")
                        if abs(predicate.latest_value * grad_theta) <= tolerance:  # no decision can be made
                            compliant_maneuver += [Maneuver.STEERLEFT,
                                                   Maneuver.STEERRIGHT]
                            print("* \t<TSolver>: no decision can be made, both maneuvers are selected")
                        elif - predicate.latest_value * grad_theta > 0:  # delta theta > 0
                            compliant_maneuver += [Maneuver.STEERLEFT]
                        else:  # delta theta < 0
                            compliant_maneuver += [Maneuver.STEERRIGHT]

                    elif predicate_category == "Acc":
                        compliant_maneuver += [Maneuver.CONSTANT]
                    else:
                        pass  # general predicate
                else:
                    print("* \t<TSolver>: Unfortunately, the model predictive robustness is"
                          " not really computed")
                    if (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [
                            PositionPredicates.KeepsSafeDistancePrec,
                            PositionPredicates.InFrontOf,
                            PositionPredicates.Precedes,
                        ]
                    ):
                        compliant_maneuver += [Maneuver.BRAKE] #, Maneuver.KICKDOWN]
                    elif (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [PositionPredicates.StopLineInFront]
                    ):
                        compliant_maneuver += [Maneuver.BRAKE]
                    elif (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [PositionPredicates.InIntersectionConflictArea]
                        and predicate.agent_placeholders == (0, 1)
                    ):
                        compliant_maneuver += [Maneuver.BRAKE]
                    elif (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [PositionPredicates.InIntersectionConflictArea]
                        and predicate.agent_placeholders == (1, 0)
                    ):
                        # TODO: FIXME add maneuvers
                        pass
                        # compliant_maneuver += [Maneuver.STEERRIGHT, Maneuver.STEERLEFT]
                    elif (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [PositionPredicates.OnLaneletWithTypeIntersection]
                    ):
                        compliant_maneuver += [Maneuver.BRAKE, Maneuver.KICKDOWN]
                    # elif predicate_category == "Pos":
                    #     # TODO: FIXME add maneuvers
                    #     pass
                    #     # compliant_maneuver += [Maneuver.STEERRIGHT, Maneuver.STEERLEFT]
                    elif predicate_category == "Vel":
                        compliant_maneuver += [Maneuver.BRAKE]
                    elif predicate_category == "Acc":
                        compliant_maneuver += [Maneuver.CONSTANT]
                    elif (predicate_category == "Pos"
                        and predicate.evaluator.predicate_name in [PositionPredicates.MainCarriagewayRightLane,
                                                                   PositionPredicates.InSameLane]):
                        compliant_maneuver += [Maneuver.STEERRIGHT,
                                               Maneuver.STEERLEFT]
                    else:
                        pass  # general predicate
                        # raise ValueError('<T-Solver>: the category {} is not specified'
                        #                  .format(predicate_category))
        compliant_maneuver = list(set(compliant_maneuver))
        # Sort by the first letter, giving preference to "Brake" over "Steer/Turn"
        compliant_maneuver.sort(key=lambda m: (m.value[0] != 'B', m.value))
        if not compliant_maneuver:
            print("* \t<TSolver>: no compliant maneuver is selected")
        else:
            string = "* \t<TSolver>: compliant maneuver /"
            for m in compliant_maneuver:
                string += m.value + "/"
            string += " is selected"
            print(string)
        return compliant_maneuver

    def search_tc(self, use_dummy_tc: bool):
        """
        Searches the time-to-compliance.
        """
        if self._compliant_maneuvers is None:
            return -math.inf  # marked as not repairable
        # Avoiding entry is a prefix property: once this branch is selected the
        # optimizer must be allowed to act from the first reparable state.  A
        # maneuver simulation cannot certify it reliably at tv=1 because that
        # state is part of the immutable monitor prefix.
        if self._tc_obj._selected_conflict_avoidance_predicates:
            self._tc_obj._tc = self._tc_obj.ego_vehicle.initial_state.time_step * self._tc_obj.dT
            return self._tc_obj._tc
        # ``not brakes_abruptly`` constrains the acceleration signal itself.
        # If TC is chosen after the first abrupt sample, that immutable prefix
        # still violates R_G2 even when the suffix QP is feasible.  Start the
        # direct avoidance branch at the first reparable state; the alternate
        # logical branch (a justified abrupt brake) remains untouched.
        if any(
            prop.source_rule == "R_G2"
            and prop.alphabet.startswith("~")
            and prop.name == "brakes_abruptly__0"
            for prop in self._sel_prop
        ):
            self._tc_obj._tc = (
                self._tc_obj.ego_vehicle.initial_state.time_step
                * self._tc_obj.dT
            )
            return self._tc_obj._tc
        # A positive safe-distance branch must be able to shape the complete
        # approach to the preceding vehicle.  Legacy maneuver-based TC search
        # often returned TV-1, leaving one QP interval and making otherwise
        # avoidable RG1 violations infeasible.  Start only this direct safety
        # branch at the first reparable state; alternative RG1 logic remains
        # governed by the original TC search.
        if any(
            prop.source_rule == "R_G1"
            and not prop.alphabet.startswith("~")
            and prop.name == "keeps_safe_distance_prec__0_1"
            for prop in self._sel_prop
        ):
            self._tc_obj._tc = (
                self._tc_obj.ego_vehicle.initial_state.time_step
                * self._tc_obj.dT
            )
            return self._tc_obj._tc
        # Keeping the stop line in front is likewise a prefix property.  If
        # this direct R_IN1 branch is selected, delaying TC until close to TV
        # can leave an immutable crossing (or an unnecessarily ill-conditioned
        # emergency stop) before the QP-controlled suffix.
        if any(
            prop.source_rule == "R_IN1"
            and not prop.alphabet.startswith("~")
            and prop.name == "stop_line_in_front__0"
            for prop in self._sel_prop
        ):
            self._tc_obj._tc = (
                self._tc_obj.ego_vehicle.initial_state.time_step
                * self._tc_obj.dT
            )
            return self._tc_obj._tc
        # IN1's positive historical standstill literal is stronger than the
        # instantaneous stop-line violation used by the legacy TC search.  A
        # TC close to TV can still avoid crossing the line, while leaving only
        # one sample before the required three-second zero-speed window.  The
        # resulting QP is then infeasible even though braking from the start of
        # the reparable trajectory is feasible.  Start this specific prefix
        # repair at the first state; all other Lin2022 TC decisions are kept.
        if any(
            prop.source_rule == "R_IN1"
            and not prop.alphabet.startswith("~")
            and "historically" in prop.name
            and "in_standstill" in prop.name
            for prop in self._sel_prop
        ):
            self._tc_obj._tc = (
                self._tc_obj.ego_vehicle.initial_state.time_step
                * self._tc_obj.dT
            )
            return self._tc_obj._tc
        tc = self.tc_object.generate(self._compliant_maneuvers, use_dummy_tc)
        return tc

    def _optimization_based_repair(self):
        """
        Initializes the qp planner and uses it for trajectory repairing.
        """
        start_time = time.time()
        if self.effective_planner == 1:
            self._planner.reset(scenario=self.tc_object.scenario,
                                tc_object=self.tc_object,
                                sel_proposition=self._sel_prop,
                                full_proposition=self._prop_full)
            self._planner.construct_constraints(self._sel_prop, self._prop_full)
            print("* \t<TSolver>: QP planner is invoked")
        elif self.effective_planner == 2:
            self._planner.reset(tc_object=self.tc_object,
                                rule_monitor=self._rule_monitor)
            suc = self._planner.construct_constraints(self._sel_prop, self._prop_full)
            if not suc:
                print("* \t<TSolver>: the constraints are not properly constructed")
                return
            print(f"* \t<TSolver>: MIQP planner is invoked")
        # elif self.config.repair.planner == 3:
        #     self._planner.reset(tc_object=self.tc_object,
        #                         rule_monitor=self._rule_monitor)
        #     print(f"* \t<TSolver>: CLRRT planner is invoked")

        else:
            raise Exception("Invalid option for the planner provided")
        print(f"* \t<TSolver>: initialization time {self.reach_set_time:.3f}s")
        start_time = time.time()

        try:
            repaired_trajectory = self._planner.plan()
        finally:
            # Preserve work performed on failed/exceptional planner paths too.
            # QPPlannerRepair used to expose zero here whenever longitudinal
            # optimization returned early, creating artificial fast failures.
            self.reach_set_time = self._planner.reach_set_time
            self.opti_plan_time = self._planner.opti_plan_time
        print(f"* \t<TSolver>: solving time {time.time() - start_time:.3f}s")
        return repaired_trajectory

    def check(
        self, proposition: List[PropositionNode], model: list, use_mpr_derivative=False, use_dummy_tc=False
    ) -> (bool, Trajectory):
        """
        Checks the T-consistency.
        """
        repaired_traj = None
        start_time = time.time()
        self.assign_proposition(proposition, model, use_mpr_derivative)
        if self.compliant_maneuvers is None:
            print("* \t<Tsolver>: tc = {}, tv = {}".format(-math.inf, -math.inf))
            return self._repairability, repaired_traj
        if self.tc_object.tv_time_step == math.inf:
            print("* \t<Tsolver>: tc = inf, tv = inf")
            return self._repairability, repaired_traj
        tc = self.search_tc(use_dummy_tc)
        print(
            "* \t<Tsolver>: tc = {}, tv = {}".format(self._tc_obj.tc, self._tc_obj.tv)
        )
        self.tc_search_time += time.time() - start_time
        print(f"* \t<Tsolver>: run time {self.tc_search_time:.3f}s")
        if tc != -math.inf:
            repaired_traj = self._optimization_based_repair()
            if repaired_traj is not None:
                self._repairability = True
        return self._repairability, repaired_traj
