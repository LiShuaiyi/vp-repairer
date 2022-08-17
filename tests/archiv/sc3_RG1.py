from abstraction.abstracter import RuleAbstracter
from t_solver.t_solver import TSolver
from t_solver.qp_planner import QPPlannerRepair
from repairer.smt_repairer import SMTTrajectoryRepairer
from t_solver.utils import convert_traj_to_ego_vehicle
from crrepairer.repairer.visualization import visualize_repairing_result, visualize_profile
from crrepairer.t_solver.utils import calculate_safe_distance

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
import matplotlib.pyplot as plt

scenario_id = "DEU_test_safe_distance"
# scenario_id = "ZAM_Tutorial-1_2_T-1"
# file_path = "/home/yuanfei/commonroad/commonroad-scenarios-master-scenarios/scenarios/hand-crafted/" \
#             + scenario_id + ".xml"
# file_path = "/home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/" \
#             + scenario_id + ".xml"
file_path = "/home/yuanfei/commonroad/commonroad_repairer/scenarios/test_interstate/" \
            + scenario_id + ".xml"

if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 1003
    rule = "R_G1"
    obs = scenario.obstacle_by_id(1000)
    obs.initial_state.position[0] += 10.0
    for state in obs.prediction.trajectory.state_list:
        state.position[0] += 10.0
    obs.prediction.occupancy_set = obs.prediction._create_occupancy_set()
    obs._initial_occupancy_shape.center[0] += 10
    ego_initial = scenario.obstacle_by_id(ego_id)

    ego_initial.prediction.trajectory.state_list = ego_initial.prediction.trajectory.state_list[:30]
    ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[:30]
    rule_abstracter = RuleAbstracter(scenario,
                                     planning_problem,
                                     ego_id, rule)
    repairer = SMTTrajectoryRepairer(rule_abstracter,
                                     ego_initial)
    repaired_traj = repairer.repair()

    ego_vehicle = convert_traj_to_ego_vehicle(ego_initial.obstacle_shape,
                                              ego_initial.initial_state,
                                              repaired_traj)
    ego_initial.prediction.shape = ego_vehicle.prediction.shape
    plot_limits = [0, 120, -2, 12]
    target_veh = scenario.obstacle_by_id(repairer.rule_abstracter.other_veh_id)
    # visualize_profile(target_veh, ego_initial, ego_vehicle)
    for time_step in range(ego_vehicle.prediction.final_time_step):
        time_step = 20
        visualize_repairing_result(scenario,
                                   ego_vehicle,
                                   time_step,
                                   plot_limits=plot_limits,
                                   target_veh=target_veh,
                                   ego_initial=ego_initial,
                                   tc=repairer.tc)