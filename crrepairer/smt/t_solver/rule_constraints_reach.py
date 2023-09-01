import os

from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode

from crmonitor.common.vehicle import Vehicle

from typing import List
import copy

from commonroad_qp_planner.configuration import (
    PlanningConfigurationVehicle,
)
from commonroad_qp_planner.utility.compute_constraints import (
    longitudinal_position_constraints,
    lateral_position_constraints,
    longitudinal_velocity_constraints,
)

from crmonitor.predicates.position import (
    PredSafeDistPrec,
    PredInSameLane,
    PredInFrontOf,
    PredPreceding,
)

from crmonitor.predicates.velocity import (
    PredLaneSpeedLimit,
    PredFovSpeedLimit,
    PredBrSpeedLimit,
    PredTypeSpeedLimit,
)

from commonroad_qp_planner.constraints import LatConstraints, LonConstraints
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import Trajectory, CustomState
from commonroad.scenario.state import InitialState

from commonroad_reach.data_structure.reach.reach_set import ReachableSet
# specification-compliant reachable set
import commonroad_reach_semantic.data_structure.rule.priorities as priorities
from commonroad_reach_semantic.data_structure.config.semantic_configuration_builder import (
    SemanticConfigurationBuilder,
)
from commonroad_reach_semantic.data_structure.driving_corridor_extractor import (
    DrivingCorridorExtractor as DrivingCorridorExtractorSemantic,
)
from commonroad_reach.data_structure.reach.driving_corridor_extractor import DrivingCorridorExtractor
from commonroad_reach.data_structure.reach.reach_interface import ReachableSetInterface
from commonroad_reach_semantic.data_structure.environment_model.semantic_model import (
    SemanticModel,
)
from commonroad_reach_semantic.data_structure.model_checking.spot_interface import (
    SpotInterface,
)
from commonroad_reach_semantic.data_structure.reach.semantic_labeling_reach_set_py import (
    PySemanticLabelingReachableSet,
)
from commonroad_reach_semantic.data_structure.rule.proposition import (
    Proposition
)
from commonroad_reach_semantic.data_structure.rule.traffic_rule_interface import (
    TrafficRuleInterface,
)
from crmonitor.common.vehicle import Vehicle
from crmonitor.predicates.position import (
    PredSafeDistPrec,
    PredInSameLane,
    PredInFrontOf,
)
from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode


class RuleConstraintsReach:
    """
    Class for traffic rule constraints (manual definition)
    """

    def __init__(
        self,
        tc_object: TC,
        rule_monitor: STLRuleMonitor,
        sel_proposition_full: List[PropositionNode],
        proposition_full: List[PropositionNode],
        veh_config: PlanningConfigurationVehicle,
        initial_trajectory: Trajectory,
        planning_problem: PlanningProblem
    ):
        # initialize the needed components
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
        self._world_state = self._rule_monitor.world
        self._nr_ts = self._tc_obj.N - self._tc_obj.tc_time_step

        # ego vehicle
        self._ego_id = (
            self._rule_monitor.vehicle_id
        )  # if no target vehicle, the other_id stands for the ego
        self._ego_vehicle_cr = self._tc_obj.ego_vehicle
        self._ego_vehicle_world = self._world_state.vehicle_by_id(self._ego_id)
        self._ini_traj = initial_trajectory

        # other vehicle (rule-relevant)
        self._other_id = self._rule_monitor.other_id
        self._target_vehicle: Vehicle = self._world_state.vehicle_by_id(self._other_id)

        # configuration
        self._veh_config = veh_config
        self._compliant_maneuver = tc_object.compliant_maneuver
        self._sel_prop_full = sel_proposition_full
        self._prop_full = proposition_full

        # initialize the elements for rule constraints (from reach)
        # longitudinal components
        self._lon_dis_constraints = list()
        self._lon_vel_constraints = list()
        self._lon_acc_constraints = list()
        # lateral components
        self._lat_dis_constraints = list()
        self._lat_vel_constraints = list()
        self._lat_acc_constraints = list()

        # initialize the commonroad-reach
        # we use the default path of the reach folder
        # todo: use params for path

        path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "../../../../commonroad-reach-semantic")
        self.reach_config = SemanticConfigurationBuilder(
            path_root=path_root
        ).build_configuration(str(self._world_state.scenario.scenario_id))
        # update the time step and nr of computation
        self.reach_config.planning.dt = self._world_state.dt
        self.reach_config.planning.steps_computation = self._nr_ts
        # update the path
        self.reach_config.general.path_scenario = (
            "../../scenarios/" + str(self._world_state.scenario.scenario_id) + ".xml"
        )
        self.reach_config.update(scenario=tc_object.scenario, planning_problem=planning_problem)
        # use the original scenario

        self.corridor = None

    def update_reach_interface(
        self, vehicle_configuration: PlanningConfigurationVehicle
    ):
        # obtain the cut-off state
        cut_off_time_step = self._tc_obj.tc_time_step
        if cut_off_time_step == 0:
            cut_off_state = copy.deepcopy(self._ego_vehicle_cr.initial_state)
        else:
            cut_off_state = copy.deepcopy(
                self._ini_traj.state_at_time_step(cut_off_time_step)
            )

        assert cut_off_state.time_step == cut_off_time_step, (
            "the time step of the state_at_time_step "
            "doesn't match the corresponding state!"
        )

        # set the cut-off state as the initial state
        self.reach_config.planning_problem.initial_state.position = (
            cut_off_state.position
        )
        self.reach_config.planning_problem.initial_state.velocity = (
            cut_off_state.velocity
        )
        self.reach_config.planning_problem.initial_state.orientation = (
            cut_off_state.orientation
        )
        self.reach_config.planning_problem.initial_state.acceleration = (
            cut_off_state.acceleration
        )
        self.reach_config.planning_problem.initial_state.time_step = 0

        for obs in self.reach_config.scenario.dynamic_obstacles:
            new_state_list = []
            if obs.state_at_time(cut_off_time_step):
                obs.initial_state = InitialState(
                    time_step=0,
                    position=obs.state_at_time(cut_off_time_step).position,
                    velocity=obs.state_at_time(cut_off_time_step).velocity,
                    orientation=obs.state_at_time(cut_off_time_step).orientation,
                    acceleration=0.0,
                    yaw_rate=0.0,
                    slip_angle=0.0,
                )
            else:
                self.reach_config.scenario.remove_obstacle(obs)
                continue
            for i in range(cut_off_time_step, self._tc_obj.N + 1):
                if obs.state_at_time(i):
                    new_state_list.append(
                        CustomState(
                            time_step=i - cut_off_time_step,
                            position=obs.state_at_time(i).position,
                            velocity=obs.state_at_time(i).velocity,
                            orientation=obs.state_at_time(i).orientation,
                        )
                    )
                else:
                    break
            if len(new_state_list) > 0:
                new_traj = Trajectory(initial_time_step=0, state_list=new_state_list)
                obs.prediction.occupancy_set = obs.prediction.occupancy_set[
                        cut_off_time_step-1:
                    ]
                for occ in obs.prediction.occupancy_set:
                    occ.time_step -= cut_off_time_step - 1
                obs.prediction.trajectory = new_traj
            else:
                self.reach_config.scenario.remove_obstacle(obs)

        self.reach_config.update(
            planning_problem=self.reach_config.planning_problem,
            scenario=self.reach_config.scenario,
            CLCS=vehicle_configuration.CLCS,
        )

        if self.reach_config.traffic_rule.activated_rules:
            semantic_model = SemanticModel(self.reach_config)
            semantic_model.determine_traffic_priorities(
                priorities.dict_traffic_sign_to_priorities
            )

            # update the rule interface
            rule_interface = self.repair_rule_interface(semantic_model)

            # update the rule interface
            # initialize the reach interface
            self.reach_interface = ReachableSetInterface(self.reach_config)

            self.reach_interface._reach = PySemanticLabelingReachableSet(
                self.reach_config, semantic_model, rule_interface
            )
        else:
            self.reach_interface = ReachableSetInterface(self.reach_config)
            self.reach_interface._reach = ReachableSet.instantiate(self.reach_config)

        self.reach_interface.compute_reachable_sets(
            step_start=0, step_end=self._nr_ts, verbose=True
        )

        if self.reach_config.traffic_rule.activated_rules:
            self.spot_interface = SpotInterface(self.reach_interface, rule_interface)
            self.spot_interface.translate_ltl_formulas()
            self.spot_interface.translate_reachability_graph()
            self.spot_interface.check()

    def repair_rule_interface(self, semantic_model: SemanticModel) -> TrafficRuleInterface:
        """
        Based on the SAT result, deciding which propositions need to be added.
            either to LTL or TPL constraints.
        """
        repaired_rules = []
        # add the repairing propositions
        for prop in self._sel_prop_full:
            if PredSafeDistPrec.predicate_name in prop.name:
                semantic_prop = Proposition.safe_following_distance_to(self._other_id)
                if prop.ttv_value > 0:
                    # change the sign
                    semantic_prop = "!" + semantic_prop
                # fixme: safe following works only for TPL somehow
                repaired_rules.append('TPL ' + semantic_prop)
            else:
                if PredInSameLane.predicate_name in prop.name:
                    semantic_prop = Proposition.in_same_lane(self._other_id)
                elif PredInFrontOf.predicate_name in prop.name:
                    semantic_prop = Proposition.in_front_of(self._other_id)
                else:
                    # for instance unnecessary_braking
                    semantic_prop = None
                if semantic_prop:
                    if prop.ttv_value > 0:
                        # change the sign
                        semantic_prop = "!" + semantic_prop
                    repaired_rules.append('LTL G[' + str(self._tc_obj.tv_time_step) + '..' +
                                          str(self._tc_obj.N) + '](' + semantic_prop + ')')
        self.reach_config.traffic_rule.activated_rules = repaired_rules
        rule_interface = TrafficRuleInterface(self.reach_config, semantic_model)
        rule_interface.print_summary()
        return rule_interface

    def compute_semantic_reachable_set(self, vehicle_configuration, verbose=True):
        self.update_reach_interface(vehicle_configuration)
        if self.reach_config.traffic_rule.activated_rules:
            dc_extractor = DrivingCorridorExtractorSemantic(self.spot_interface)
            dc_extractor.extract_corridors(search=True)
            self.corridor = dc_extractor.determine_optimal_corridor()
        else:
            dc_extractor = DrivingCorridorExtractor(self.reach_interface.reachable_set, self.reach_config)
            driving_corridors = dc_extractor.extract()
            self.corridor = driving_corridors[0]


        # * for debugging the reach semantic
        #node_to_group = util_visual.groups_from_propositions(self.reach_interface._reach.labeler.reachable_set_to_propositions)
        #util_visual.show_interactive_reach_graph(self.reach_interface, use_images=True, node_to_group=node_to_group)
        #util_visual.plot_scenario_with_kripke_nodes(self.spot_interface, plot_accepting=True, save_gif=True)

    def longitudinal_constraints(self, vehicle_configuration):
        # compute the driving corridor
        self.compute_semantic_reachable_set(vehicle_configuration)

        if self.corridor is None:
            raise Exception("the driving corridor is either not computed or empty")
        else:
            s_min, s_max = longitudinal_position_constraints(self.corridor)
            v_min, v_max = longitudinal_velocity_constraints(self.corridor)
        c_tv_lon = LonConstraints.construct_constraints(
            s_min, s_max, s_min, s_max, v_min=v_min, v_max=v_max
        )
        return c_tv_lon

    def lateral_constraints(self, traj_lon, configuration_qp):
        traj_lon_positions = traj_lon.get_positions()[:, 0]
        lateral_driving_corridors = self.reach_interface.extract_driving_corridors()
        lat_dc = list(lateral_driving_corridors)[0]
        d_min, d_max = lateral_position_constraints(
            lat_dc, self.corridor, traj_lon_positions, configuration_qp
        )
        c_tv_lat = LatConstraints.construct_constraints(d_min, d_max, d_min, d_max)
        return c_tv_lat
