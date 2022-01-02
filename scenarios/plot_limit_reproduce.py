import matplotlib.pyplot as plt
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.common.file_reader import CommonRoadFileReader

if __name__ == '__main__':
    scenario_id_1 = "DEU_Gar-1_1_T-1"
    scenario_id_2 = "ZAM_Tutorial-1_2_T-1"
    plot_limits = [-5, 50, -4.5, 3]
    scenario_1, _ = CommonRoadFileReader(scenario_id_1+'.xml').open(lanelet_assignment=True)
    scenario_2, _ = CommonRoadFileReader(scenario_id_2+'.xml').open(lanelet_assignment=True)
    ego_veh = scenario_2.obstacle_by_id(43)
    rnd = MPRenderer(figsize=(20, 10), plot_limits=plot_limits)
    scenario_1.draw(rnd)
    ego_veh.draw(rnd,
                 draw_params=
                 {"time_begin": 0, "trajectory": {
                     "draw_trajectory": False},
                  "occupancy": {
                      "draw_occupancies": 1,
                      "shape": {"rectangle": {
                          "facecolor": "#0065bd",
                          "edgecolor": "#0065bd"}
                      }},
                  "dynamic_obstacle":
                      {"vehicle_shape": {
                          "occupancy": {
                              "shape": {"rectangle": {
                                  "facecolor": "#0065bd",
                                  "edgecolor": "#0065bd"}
                              }}}}})
    ego_veh.draw(rnd,
                 draw_params=
                 {"time_begin": 0, "trajectory": {
                     "draw_trajectory": False},
                  "occupancy": {
                      "draw_occupancies": 1,
                      "shape": {"rectangle": {
                          "facecolor": "#0065bd",
                          "edgecolor": "#0065bd"}
                      }},
                  "dynamic_obstacle":
                      {"vehicle_shape": {
                          "occupancy": {
                              "shape": {"rectangle": {
                                  "facecolor": "#0065bd",
                                  "edgecolor": "#0065bd"}
                              }}}}})
    rnd.render()
    plt.show()


