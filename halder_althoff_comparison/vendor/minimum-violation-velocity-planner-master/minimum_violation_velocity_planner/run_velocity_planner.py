import os
import time
from typing import List, Dict

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

from minimum_violation_velocity_planner.common.feasibility_checking import isFeasible
from minimum_violation_velocity_planner.common.utils import loadParameterFile, create_reference_path, \
    createOutputFolder, determineTimeHorizon, readRulebookFile
from minimum_violation_velocity_planner.common.visualization import visualizeScenario
from minimum_violation_velocity_planner.rulebook.rulebook import Rulebook
from minimum_violation_velocity_planner.velocity_planner.velocity_planner import VelocityPlanner


def runVelocityPlanner(scenario: Scenario, planning_problem: PlanningProblem, planner: str = 'A_STAR_MIN_RULE_EVAL',
                       total_order: List = None, scenario_params=None, check_feasibility=False, verbose=False,
                       show_available_rules=False, show_rule_evaluation_times=False, output_directory=None,
                       make_visualizations=False, visu_params: Dict=None, interpolate_initial_state_linear=False):

    # Create reference path
    ccosy, route_lanelets = create_reference_path(planning_problem, scenario)
    if not ccosy:
        raise Warning('Route planner could not determine a route for this scenario!')

    # If no scenario parameter file is provided, use default param file
    if scenario_params is None:
        default_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'defaults/default_scenario_params.yaml')
        scenario_params = loadParameterFile(default_params_path)

    # Grid specifications
    planner_start_time = time.time()
    vehicle_parameters = parameters_vehicle2()
    state_rounding_precision = scenario_params.get('general').get('state_rounding_precision')
    dt = round(scenario_params.get('general').get('delta_t'), state_rounding_precision)
    dv = round(scenario_params.get('general').get('delta_v'), state_rounding_precision)
    da = round(dv / dt * scenario_params.get('general').get('k_opt'), state_rounding_precision)
    ds = round(0.5 * dv * dt * scenario_params.get('general').get('k_opt'), state_rounding_precision)

    time_horizon = [0.0, determineTimeHorizon(scenario, scenario_params.get('general').get('max_time_horizon'), dt,
                                              state_rounding_precision)]
    station_horizon = [0.0, ccosy.length()]
    velocity_horizon = [0.0, vehicle_parameters.longitudinal.v_max]

    grid = (dt, dv, ds, da, time_horizon, station_horizon, velocity_horizon)

    # Instantiate Rulebook
    rulebook = Rulebook(scenario, planning_problem, ccosy, route_lanelets, vehicle_parameters, scenario_params, grid,
                        integration_strategy=2)  # 0 = linear_interpolation, 1 = max_interpolation, 2 = box_interpolation
    if show_available_rules:
        rulebook.printAvailableRules()

    # Set total order of rules
    if total_order is None:
        default_rulebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'defaults/default_rulebook.yaml')
        total_order = readRulebookFile(default_rulebook_path)
    rulebook.setOrder(total_order, verbose)

    # Instantiate Planner
    vp = VelocityPlanner(scenario, planning_problem, ccosy, rulebook, grid, vehicle_parameters, scenario_params)

    # Plan minimal violating trajectory
    planning_start_time = time.time()
    vp.plan(planner, verbose)
    planning_time = time.time() - planning_start_time

    # Check for physical feasibility
    if check_feasibility:
        print('Feasibility: ' + str(isFeasible(vp.getCREgoObject().prediction.trajectory, scenario.dt,
                                               show_plots=False)) + "\n")

    if verbose:
        print_info = [
            ['Used planner', planner, ''],
            ['Runtime', '{:.4f}'.format(time.time() - planner_start_time), '[s]'],
            ['Planning time', '{:.4f}'.format(planning_time), '[s]\n']
        ]
        for row in print_info:
            print("{: <20} {: >8} {: <2}".format(*row))
        #vp.printViolationOfOptimalTrajectory(show_all_time_steps=False)

    if show_rule_evaluation_times:
        print('Total rule evaluation time:', '{:.4f}'.format(rulebook.getTotalRuleEvaluationTime()), '[s]')
        rulebook.printEvaluationTimesPerRule()

    # Create Output Folder for Scenario
    if make_visualizations:
        if visu_params is None:
            visu_params = {'visualize_v_plot': True,
                           'save_v_plot': True,
                           'show_v_plot': False,
                           'visualize_graph_search_progress': False,
                           'save_optimal_violation': True,
                           'visualize_scenario': False,
                           'use_sampeled_trajectory': True,
                           'create_video': True,
                           'center_ego': True,
                           'draw_ref_path': True,
                           'draw_projection_domain': True,
                           'show_violation': True}

        if output_directory is None:
            output_directory = createOutputFolder(str(scenario.scenario_id))

        if visu_params['visualize_v_plot']:
            vp.visualize(output_directory, save_fig=visu_params['save_v_plot'], show=visu_params['show_v_plot'])

        if visu_params['visualize_graph_search_progress']:
            vp.visualizeProgress(output_directory)

        if visu_params['save_optimal_violation']:
            vp.saveOptimalViolation(output_directory)

        if visu_params['visualize_scenario']:
            visualizeScenario(planning_problem, scenario, ccosy, vp.getCREgoObject(), scenario_params, output_directory,
                              vp.getTagsAndViolationsOfOptimalTrajectory(use_sampled_trajectory=visu_params['use_sampeled_trajectory']),
                              create_video=visu_params['create_video'],
                              ego_centered=visu_params['center_ego'],
                              draw_ref_path=visu_params['draw_ref_path'],
                              draw_projection_domain=visu_params['draw_projection_domain'],
                              plot_violations=visu_params['show_violation'])

    return vp.optimal_trajectory, vp.optimal_violations, rulebook, \
           vp.getCREgoObject(interpolate_linear=interpolate_initial_state_linear), (planning_time, rulebook.getTotalRuleEvaluationTime())
