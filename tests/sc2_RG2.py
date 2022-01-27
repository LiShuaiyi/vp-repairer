from abstraction.abstracter import RuleAbstracter
from t_solver.t_solver import TSolver
from t_solver.qp_planner import QPPlannerRepair
from repairer.smt_repairer import SMTTrajectoryRepairer
from t_solver.utils import convert_traj_to_ego_vehicle
from commonroad_repair.crrepairer.repairer.visualization import visualize_repairing_result,\
    visualize_initial_result, visualize_a_profile, visualize_v_profile

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
import matplotlib.pyplot as plt
import math

scenario_id = "DEU_LocationAUpper-36_54_T-1"
scenario_id = 'DEU_LocationAUpper-26_30_T-1'
scenario_id = 'DEU_LocationALower-25_56_T-1'
scenario_id = "DEU_LocationAUpper-26_9_T-1"
scenario_id = "DEU_LocationAUpper-36_53_T-1"
scenario_id = "ZAM_Zip-1_56_T-1"

file_path = "../../commonroad-scenarios-master-scenarios/scenarios/hand-crafted/" + scenario_id + ".xml"
figure_path = "/home/yuanfei/commonroad/commonroad_repair/tests/figures/"

# file_path = "/home/yuanfei/commonroad/commonroad_repair/scenarios/test_interstate/DEU_test_unnecessary_braking.xml"

if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 2
    rule = "R_G2"

    ego_initial = scenario.obstacle_by_id(ego_id)
    # # change the time horizon
    initial_time_step = 50
    final_time_step = initial_time_step + 21
    # ego_states_20 = ego_initial.prediction.trajectory.state_list[nr:nr+21]
    # for state in ego_states_20:
    #     state.time_step -= nr + 1
    # ego_initial.initial_state = ego_states_20[0]
    # ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[nr:nr+21]
    # for occupancy in ego_initial.prediction.occupancy_set:
    #     occupancy.time_step -= nr + 1
    # ego_initial.prediction.trajectory.state_list = ego_states_20[1:]
    #
    # for obs in scenario.dynamic_obstacles:
    #     if obs.obstacle_id != ego_id:
    #
    #         states_20 = obs.prediction.trajectory.state_list[nr:nr+21]
    #         for state in states_20:
    #             state.time_step -= nr + 1
    #         obs.initial_state = states_20[0]
    #         obs.prediction.occupancy_set = obs.prediction.occupancy_set[nr:nr + 21]
    #         for occupancy in obs.prediction.occupancy_set:
    #             occupancy.time_step -= nr + 1
    #         obs.prediction.center_lanelet_assignment = obs.prediction.center_lanelet_assignment[nr:nr + 21]
    #         obs.prediction.final_time_step = 20


    for i in range(ego_initial.prediction.trajectory.final_state.time_step):
            ego_initial.state_at_time(i).acceleration = (ego_initial.state_at_time(i+1).velocity-
                                                         ego_initial.state_at_time(i).velocity)/scenario.dt

    # time_step = 0
    # rnd = MPRenderer(figsize=(40, 10))
    # scenario.draw(
    #     rnd,
    #     draw_params=ParamServer({"time_begin": time_step, "trajectory": {
    #              "draw_trajectory": False}, "occupancy": {
    #         "draw_occupancies": 0}, 'dynamic_obstacle': {'show_label': True}})
    # )
    # rnd.render()
    # plt.title(str(time_step))
    # plt.show()
    rule_abstracter = RuleAbstracter(scenario,
                                     planning_problem,
                                     ego_id, rule)

    repairer = SMTTrajectoryRepairer(rule_abstracter,
                                     ego_initial)
    repaired_traj = repairer.repair()

    plot_limits = [-50, 10, 3.5, 11.2]
    # visualize_profile(target_veh, ego_initial, ego_vehicle)
    if repaired_traj is not None:
        ego_vehicle = convert_traj_to_ego_vehicle(ego_initial.obstacle_shape,
                                                  ego_initial.initial_state,
                                                  repaired_traj)
        ego_initial.prediction.shape = ego_vehicle.prediction.shape
        # plot_limits = [-10, 100, -8, 8]
        visualize_a_profile(scenario.dt, ego_initial, ego_vehicle, initial_time_step, final_time_step,
                            int(repairer.tc), int(repairer.tv))
        for time_step in range(initial_time_step, final_time_step):
            # visualize_repairing_result(scenario,
            #                            ego_vehicle, time_step, None, plot_limits=plot_limits,
            #                            end_time=final_time_step, tc=int(repairer.tc))
            # visualize_initial_result(scenario, ego_initial,
            #                          time_step, None, plot_limits=plot_limits, end_time=final_time_step,
            #                          tv=int(repairer.tv), save_path=figure_path)
            visualize_repairing_result(scenario,
                                       ego_vehicle,
                                       time_step,
                                       tc=repairer.tc,
                                       end_time=final_time_step,
                                       plot_limits=plot_limits,
                                       ego_initial=ego_initial,
                                       save_path=figure_path)