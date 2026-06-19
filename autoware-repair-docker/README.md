# Instruction to run the docker image of CRRepair Planner with AV2.0 docker images

## 1. Apply for and download a Gurobi WLS license

- Refer to the [Gurobi WLS license website](https://license.gurobi.com/manager/doc/overview/#optimizer) to apply for a free WLS license.
- Download the license file and save it to the `commonroad-repairer/autoware-repair-docker`(same directory as this README.md) directory.
- The WLS license will be automatically mounted to the docker container when you run the docker image.

## 2. Run the docker image

- Run the docker images in the simulation:

```bash
cd commonroad-repairer # make sure you are in the commonroad-repairer directory
cd autoware-repair-docker/compose_launch # go to the directory where the docker compose file is located
xhost + # allow the docker container to access the display
docker compose -f planning_sim_crrepair.yml --env-file .env pull # pull the docker images
./run_sim_docker_crrepair_log.sh # run the docker containers and save the log
```

- Run the docker images on EDGAR vehicle:

```bash
cd commonroad-repairer # make sure you are in the commonroad-repairer directory
cd autoware-repair-docker/compose_launch # go to the directory where the docker compose file is located
xhost + # allow the docker container to access the display
docker compose -f e2e.yml --env-file .env pull # pull the docker images
./run_sim_docker_e2e_log.sh # run the docker containers and save the log
```

After the docker containers are up, the RViz window will pop up displaying the FTM building simulation environment.

Select a start point and goal point that crosses the stop line.

## 3. Set the velocity limit to 5 km/h in the RViz window

## 4. Engage the vehicle and record rosbags

After selecting the start point and goal point, you can engage the vehicle and record rosbags by running the following command:

```bash
docker exec -it compose_launch-planning-1 bash # enter the planning docker container

# In the planning docker container
cd /autoware
source install/setup.bash
ros2 launch cr2autoware engage_and_rosbag.launch.py
```

The ego vehicle will drive from the start point to the goal point, and the rosbags will be saved to the `output/rosbag` directory in this repository.

After the ego vehicle reaches the goal point, you can terminate the rosbag recording by pressing `Ctrl+C` in the terminal.

## 5. Replay the rosbags

### 5.1 Replay the rosbags by docker containers

In the `autoware-repair-docker/compose_launch/planning_sim_crrepair_rosbag_replay.yml` file, the rosbag will be replayed by the planning docker container. You can then view the planning results in the RViz window.

Replace the rosbag file name `2024-12-01_18-13-01_46` at `line 17` in the `planning_sim_crrepair_rosbag_replay.yml` file with the name of the rosbag you want to replay, and run the following command:

```bash
cd commonroad-repairer # make sure you are in the commonroad-repairer directory
cd autoware-repair-docker/compose_launch # go to the directory where the docker compose file is located
docker compose -f planning_sim_crrepair_rosbag_replay.yml --env-file .env up # compose up the planning docker containers to replay the rosbags
```

If you want to pause the rosbag during replay, you need to start the docker compose this way:

```bash
cd commonroad-repairer # make sure you are in the commonroad-repairer directory
cd autoware-repair-docker/compose_launch # go to the directory where the docker compose file is located
docker compose -f planning_sim_crrepair_rosbag_replay.yml --env-file .env run --service-ports planning bash

# In planning docker container (same terminal as the one you used to run the docker compose)
ros2 bag play /crrepair/commonroad-repairer/output/rosbag/<rosbag_file_name> --clock
```

Press `Space` in the terminal to pause the rosbag replay.

### 5.2 Replay the rosbags by local Autoware installation

Here is another way to replay the rosbags if you have a local Autoware installation and don't want to use the docker containers:

- First, clone the repo https://gitlab.lrz.de/av2.0/edgar_vehicle_model.
- Build the edgar_vehicle packages by running the following command:

```bash
cd edgar_vehicle_model
source /opt/ros/humble/setup.bash
source <path_to_your_autoware_installation>/install/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

- Then, start rviz by running the following command:

```bash
source /opt/ros/humble/setup.bash
source <path_to_your_autoware_installation>/install/setup.bash
source <path_to_your_edgar_vehicle_model_installation>/install/setup.bash
ros2 run rviz2 rviz2 -d <path_to_this_commonroad-repairer_repo>/autoware-repair-docker/rviz/autoware.rviz
```

- Finally, replay the rosbags by running the following command:

```bash
source /opt/ros/humble/setup.bash
source <path_to_your_autoware_installation>/install/setup.bash
source <path_to_your_edgar_vehicle_model_installation>/install/setup.bash
ros2 bag play <path_to_this_commonroad-repairer_repo>/output/rosbag/<rosbag_file_name> --clock
```

Press `Space` in the terminal to pause the rosbag replay.

## 6. Collected planning results

The runtime results will be saved to the `output/crrepairer2autoware` directory in this repository.

After each simulation, post-processing will be performed to organize the runtime results, plot the planning figures, and save it under the `crrepairer2autoware_collected_copy` directory.

## [For Developer] Build the docker images

Refer to the [README_for_developer.md](./README_for_developer.md) for instructions on how the docker images are built.

## Notes

Scenario id of cr2repair is `DEU_GarchingCampus2D2D-2`
