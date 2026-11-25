import os
from typing import Dict, List

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
import numpy as np
from matplotlib import pyplot as plt


def visualizeScenario(planning_problem, scenario: Scenario, ccosy, ego_object, params: Dict, output_dir: str,
                      tags_and_violations: List[Dict], create_video=False, ego_centered=False,
                      draw_ref_path=False, draw_projection_domain=False, plot_violations=False):

    # Make folder for images
    output_dir_images = os.path.abspath(output_dir + "/time_steps")
    os.makedirs(output_dir_images)

    if plot_violations:
        fig, axs = plt.subplots(1, 2, figsize=(35, 20), gridspec_kw={'width_ratios': [3, 1]})
    else:
        fig, axs = plt.subplots(figsize=(35, 20))

    # ----- s-t plot -----

    fig.tight_layout()
    for i in range(len(ego_object.prediction.trajectory.state_list)):
        ax = axs[0] if plot_violations else axs
        ax.clear()
        if ego_centered:
            ego_centered_view_size = 40
            rnd = MPRenderer(ax=ax,
                plot_limits=[ego_object.prediction.trajectory.state_list[i].position[0] - ego_centered_view_size,
                             ego_object.prediction.trajectory.state_list[i].position[0] + ego_centered_view_size,
                             ego_object.prediction.trajectory.state_list[i].position[1] - ego_centered_view_size,
                             ego_object.prediction.trajectory.state_list[i].position[1] + ego_centered_view_size])
        else:
            rnd = MPRenderer()

        draw_params_scn = {'time_begin': i, "dynamic_obstacle": {"draw_icon": True}, "planning_problem": {"initial_state": {"state": { "draw_arrow": True}}}}
        draw_params_ego = {'time_begin': i, "dynamic_obstacle": {"draw_icon": True, "vehicle_shape": {"occupancy": {"shape": {"polygon": {"opacity": 1.0,"facecolor": "green","edgecolor": "darkgreen"}}}}}}


        scenario.draw(rnd, draw_params=draw_params_scn)
        planning_problem.draw(rnd)
        ego_object.draw(rnd, draw_params=draw_params_ego)
        rnd.render(show=False)

        # Draw ref path
        if draw_ref_path:
            rp = np.asarray(ccosy.reference_path())
            ax.plot(rp[:, 0], rp[:, 1], zorder=1000, marker='.', color='green')

        # draw projection domain
        if draw_projection_domain:
            proj_domain_border = np.asarray(ccosy.projection_domain())
            ax.plot(proj_domain_border[:, 0], proj_domain_border[:, 1], zorder=100, color='orange')

        if plot_violations:
            ax = axs[1]
            ax.clear()

            tags = list(tags_and_violations[i].keys())
            y_pos = np.arange(len(tags))
            max_violations = list(tags_and_violations[-1].values())
            violations = list(tags_and_violations[i].values())
            normed_violations = [0.0 if max_violations[i] == 0 else (abs(violations[i])/abs(max_violations[i])) for i in range(len(violations))]
            ax.barh(tags, normed_violations, align='center', color=plt.get_cmap('YlOrRd')(normed_violations), alpha=0.8)

            for idx, tag in enumerate(tags):
                ax.text(0.1, idx, tag, va='center')

            ax.get_yaxis().set_ticklabels([])

            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Relative Violation')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_bounds((0, len(tags)-1))
            ax.set_xlim(0, 1)

        fig.savefig(output_dir_images + '/time_step_' + "{:04d}".format(i) + '.png')

    # Create Video
    if create_video:
        frame_rate = int(1.0 / scenario.dt)
        os.system("ffmpeg -y -r " + str(frame_rate) + " -f image2 -i " + output_dir_images + "/time_step_%04d.png " + output_dir + "/time_steps.mp4")


def showScenarioAndPlanningProblem(scenario: Scenario, planning_problem: PlanningProblem, time_step: int):
    rnd = MPRenderer()
    scenario.draw(rnd, draw_params={'time_begin': time_step})
    planning_problem.draw(rnd)
    rnd.render(show=True)
    plt.show()

