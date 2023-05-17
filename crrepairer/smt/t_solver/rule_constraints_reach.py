import numpy as np
import os
from collections import defaultdict

from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode

# class from STL monitor
from crmonitor.predicates.position import (PredSafeDistPrec, PredInSameLane,
                                           PredInFrontOf, PredPreceding)

from crmonitor.predicates.velocity import (PredLaneSpeedLimit, PredFovSpeedLimit,
                                           PredBrSpeedLimit, PredTypeSpeedLimit)
from crmonitor.predicates.general import PredCutIn
from crmonitor.predicates.acceleration import (PredAbruptBreaking, PredRelAbruptBreaking)

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle

from typing import List
import copy

from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.utility.compute_constraints import longitudinal_position_constraints, \
    lateral_position_constraints
from commonroad_qp_planner.constraints import LonConstraints, LatConstraints
from commonroad_qp_planner.initialization import convert_pos_curvilinear
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory
from commonroad_qp_planner.constraints import LatConstraints, LonConstraints

from commonroad.scenario.trajectory import Trajectory, CustomState
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.state import InitialState

# specification-compliant reachable set
try:
    from commonroad_reach_semantic.data_structure.configuration_builder import ConfigurationBuilder
    from commonroad_reach_semantic.data_structure.semantic_model import SemanticModel
    from commonroad_reach_semantic.data_structure.traffic_rule_interface import TrafficRuleInterface
    from commonroad_reach_semantic.data_structure.reach.reach_interface import ReachableSetInterface
    from commonroad_reach_semantic.data_structure.spot_interface import SpotInterface
    from commonroad_reach_semantic.data_structure.driving_corridor_extractor import DrivingCorridorExtractor
    from commonroad_reach_semantic.utility import visualization as util_visual
except ImportWarning:
    print("commonroad-reach-semantic is not installed")


class RuleConstraintsReach:
    """
    Class for traffic rule constraints (manual definition)
    """

    def __init__(self,
                 tc_object: TC,
                 rule_monitor: STLRuleMonitor,
                 sel_proposition_full: List[PropositionNode],
                 proposition_full: List[PropositionNode],
                 veh_config: PlanningConfigurationVehicle,
                 initial_trajectory: Trajectory):
        # initialize the needed components
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
        self._world_state = self._rule_monitor.world
        self._nr_ts = self._tc_obj.N - self._tc_obj.tc_time_step

        # ego vehicle
        self._ego_id = self._rule_monitor.vehicle_id  # if no target vehicle, the other_id stands for the ego
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
        self.reach_config = ConfigurationBuilder.build_configuration(
            str(self._world_state.scenario.scenario_id))
        # update the time step and nr of computation
        self.reach_config.planning.dt = self._world_state.dt
        self.reach_config.planning.steps_computation = self._nr_ts
        # update the path
        self.reach_config.general.path_scenario = "../../scenarios/" +\
                                                  str(self._world_state.scenario.scenario_id) + '.xml'
        self.reach_config.update()

        # semantic models


    def update_reach_interface(self):
        # obtain the cut-off state
        cut_off_time_step = self._tc_obj.tc_time_step
        if cut_off_time_step == 0:
            cut_off_state = copy.deepcopy(self._ego_vehicle_cr.initial_state)
        else:
            cut_off_state = copy.deepcopy(self._ini_traj.state_at_time_step(cut_off_time_step))

        assert cut_off_state.time_step == cut_off_time_step, "the time step of the state_at_time_step " \
                                                             "doesn't match the corresponding state!"

        # set the cut-off state as the initial state
        self.reach_config.planning_problem.initial_state.position = cut_off_state.position
        self.reach_config.planning_problem.initial_state.velocity = cut_off_state.velocity
        self.reach_config.planning_problem.initial_state.orientation = cut_off_state.orientation
        self.reach_config.planning_problem.initial_state.time_step = 0

        for obs in self.reach_config.scenario.dynamic_obstacles:
            new_state_list = []
            obs.initial_state = InitialState(time_step=0,
                                             position=obs.state_at_time(cut_off_time_step).position,
                                             velocity=obs.state_at_time(cut_off_time_step).velocity,
                                             orientation=obs.state_at_time(cut_off_time_step).orientation,
                                             acceleration=0.0,
                                             yaw_rate=0.0,
                                             slip_angle=0.0)
            for i in range(cut_off_time_step+1, self._tc_obj.N + 1):
                new_state_list.append(CustomState(
                    time_step=i - cut_off_time_step,
                    position=obs.state_at_time(i).position,
                    velocity=obs.state_at_time(i).velocity,
                    orientation=obs.state_at_time(i).orientation
                ))
            new_traj = Trajectory(initial_time_step=1,
                                  state_list=new_state_list)
            obs.prediction.trajectory = new_traj
        # remove the ego
        self.reach_config.scenario.remove_obstacle(
            self.reach_config.scenario.obstacle_by_id(self._ego_vehicle_cr.obstacle_id)
        )

        self.reach_config.update(
            planning_problem=self.reach_config.planning_problem,
            scenario=self.reach_config.scenario
        )
        semantic_model = SemanticModel(self.reach_config)
        rule_interface = TrafficRuleInterface(self.reach_config)
        semantic_model.determine_traffic_priorities(rule_interface.dict_traffic_sign_to_priorities)
        rule_interface.concretize_traffic_rules(semantic_model)
        rule_interface.print_summary()
        # todo: update the semantics model
        # initialize the reach interface
        self.reach_interface = ReachableSetInterface(self.reach_config,
                                                semantic_model,
                                                rule_interface)

        self.reach_interface.compute_reachable_sets(
            step_start=1, step_end=self._nr_ts, verbose=True
        )
        self.spot_interface = SpotInterface(self.reach_interface)
        self.spot_interface.translate_ltl_formulas()
        self.spot_interface.translate_reachability_graph()
        self.spot_interface.check()

    def compute_semantic_reachable_set(self, verbose=True):
        self.update_reach_interface()
        self.reach_interface.check()

        # ==== extract optimal driving corridor
        self.reach_interface.determine_optimal_corridor()
        corridor = self.reach_interface.corridor_optimal
        s_min, s_max = longitudinal_position_constraints(corridor)
        c_tv_lon = LonConstraints.construct_constraints(s_min, s_max, s_min, s_max)
        # * for debugging the reach semantic
        # import matplotlib.pyplot as plt
        # for time, node in self.reach_interface.reachable_set.items():
        #     print(time)
        #     for id, reach_node in node.items():
        #         print(reach_node[0].p_lon_min, reach_node[0].p_lon_max,
        #               reach_node[0].v_lon_min, reach_node[0].v_lon_max)
        #         if time >= 7:
        #             plt.plot(*reach_node[0].polygon_lon.exterior.xy)
        #             plt.title(str(time))
        #     plt.show()
        pass

    def longitudinal_constraints(self):
        self.compute_semantic_reachable_set()
        util_visual.plot_scenario_with_driving_corridor(
            step_start=1, step_end=self._nr_ts,
            reach_interface=self.reach_interface, save_gif=True)
        pass

