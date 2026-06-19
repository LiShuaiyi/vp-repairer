# [For Developer] Instruction to build the docker image of CRRepair Planner with AV2.0 docker images

## 1. Build the docker image

```bash
cd commonroad-repairer # make sure you are in the commonroad-repairer directory
cd autoware-repair-docker/compose_launch # go to the directory where the docker compose file is located
docker compose -f planning_sim_cr2aw.yml --env-file .env pull # pull the docker images for AV2.0 autoware and other modules

cd commonroad-repairer # make sure you are in the commonroad-repairer directory
./autoware-repair-docker/build_docker_crrepair.sh # build the docker image for CRRepair Planner module
```

## 2. Modify the code in `control docker container`

1. In the running `control docker container`, edit the file `/autoware/src/universe/autoware.universe/control/pid_longitudinal_controller/src/pid_longitudinal_controller.cpp`

   The `keep_stopped_condition` is a parameter in the `control docker container` that checks if the vehicle can resume driving after stopping. Currently, we override this parameter to `false` to allow the vehicle to resume driving after stopping.

   - Find the following lines (around `line 600`):

      ```C++
      const bool keep_stopped_condition = std::fabs(current_vel) < vel_epsilon &&
                                          m_enable_keep_stopped_until_steer_convergence &&
                                          !lateral_sync_data_.is_steer_converged;
      ```

      Change the lines to (delete the `const` keyword):

      ```C++
      bool keep_stopped_condition = std::fabs(current_vel) < vel_epsilon &&
                                          m_enable_keep_stopped_until_steer_convergence &&
                                          !lateral_sync_data_.is_steer_converged;
      ```

   - Find the following lines (around `line 727`):

      ```C++
      if (has_nonzero_target_vel && keep_stopped_condition) {
        debug_msg_once("target speed > 0, but keep stop condition is met. Keep STOPPED.");
      }
      ```

      Change the lines to (set `keep_stopped_condition` to `false`):

      ```C++
      if (has_nonzero_target_vel && keep_stopped_condition) {
        debug_msg_once("target speed > 0, but keep stop condition is met. Keep STOPPED.");
        std::string info = "m_enable_keep_stopped_until_steer_convergence: " + std::to_string(
          m_enable_keep_stopped_until_steer_convergence) + ", !is_steer_converged: " +
                          std::to_string(!lateral_sync_data_.is_steer_converged);
        debug_msg_once(info.c_str());
        debug_msg_once("Ignoring steer convergence.");
        keep_stopped_condition = false;
      }
      ```

2. In the running `control docker container`, edit the file `/autoware/src/universe/autoware.universe/control/mpc_lateral_controller/src/mpc_utils.cpp`

   During repairing, the mpc_lateral_controller will interpolate the trajectory. The query_keys of the interpolation should not be empty and the base_key is required to be strictly ascending. We add a small offset to the base_key to make it strictly ascending when plateau occurs in the base_key.

   - To make sure the query_keys is not empty:
     - Find the following lines (around `line 118`):

        ```C++
        // To accurately sample the ego point, resample separately in the forward direction and the
        // backward direction from the current position.
        for (double s = std::clamp(
              input_arclength.at(nearest_seg_idx) + ego_offset_to_segment, 0.0,
              input_arclength.back() - 1e-6);
            0 <= s; s -= resample_interval_dist) {
          output_arclength.push_back(s);
        }
        std::reverse(output_arclength.begin(), output_arclength.end());
        for (double s = std::max(input_arclength.at(nearest_seg_idx) + ego_offset_to_segment, 0.0) +
                        resample_interval_dist;
            s < input_arclength.back(); s += resample_interval_dist) {
          output_arclength.push_back(s);
        }
        ```

     - Add the following lines `before` the above lines:

        ```C++
        // To make sure the query_keys is not empty, we add start_s to the query_keys.
        double start_s = std::clamp(
          input_arclength.at(nearest_seg_idx) + ego_offset_to_segment,
          0.0,
          input_arclength.back() - 1e-6);
        output_arclength.push_back(start_s);
        ```

   - To make sure the base_key is strictly ascending:
     - Find the following lines (around `line 149`):

        ```C++
        output.x = spline_arc_length(input.x);
        output.y = spline_arc_length(input.y);
        output.z = spline_arc_length(input.z);
        output.yaw = spline_arc_length(input.yaw);
        output.vx = lerp_arc_length(input.vx);  // must be linear
        output.k = spline_arc_length(input.k);
        output.smooth_k = spline_arc_length(input.smooth_k);
        output.relative_time = lerp_arc_length(input.relative_time);  // must be linear
        ```

     - Add the following lines `before` the above lines:

        ```C++
        // // Print input_arclength before offset
        // std::ostringstream oss_input_arclength_before_offset;
        // oss_input_arclength_before_offset.precision(15);
        // oss_input_arclength_before_offset << "Before offset input_arclength: ";
        // for (size_t i = 0; i < input_arclength.size(); ++i) {
        //     oss_input_arclength_before_offset << i << ": " << input_arclength[i];
        //     if (i != input_arclength.size() - 1) {
        //         oss_input_arclength_before_offset << "," << std::endl;
        //     }
        // }
        // RCLCPP_INFO(rclcpp::get_logger("mpc_utils"), oss_input_arclength_before_offset.str().c_str());

        // To make sure the base_key of the interpolation is strictly ascending, we add a small offset to the base_key to make it strictly ascending when plateau occurs in the base_key.
        // find left and right indices of same element in input_arclength [idx_l, idx_r)
        size_t idx_l = 0;
        size_t idx_r = 0;
        for (size_t i = 1; i < input_arclength.size(); ++i) {
          if (input_arclength.at(i - 1) != input_arclength.at(i)) {
            if (idx_r == 0) {
              idx_l = i;
            }
          } else {
            idx_r = i;
          }
        }

        // add offset to input_arclength to make it ascending
        for (size_t i = idx_l; i <= idx_r; ++i) {
          input_arclength.at(i) += 1e-13 * (i - idx_l);
        }

        // // Print input_arclength after offset
        // std::ostringstream oss_input_arclength_after_offset;
        // oss_input_arclength_after_offset.precision(15);
        // oss_input_arclength_after_offset << "After offset input_arclength: ";
        // for (size_t i = 0; i < input_arclength.size(); ++i) {
        //     oss_input_arclength_after_offset << i << ": " << input_arclength[i];
        //     if (i != input_arclength.size() - 1) {
        //         oss_input_arclength_after_offset << "," << std::endl;
        //     }
        // }
        // RCLCPP_INFO(rclcpp::get_logger("mpc_utils"), oss_input_arclength_after_offset.str().c_str());

        // // Print idx_l and idx_r
        // std::ostringstream oss_idx_l;
        // std::ostringstream oss_idx_r;
        // oss_idx_l << "idx_l: " << idx_l;
        // oss_idx_r << "idx_r: " << idx_r;
        // RCLCPP_INFO(rclcpp::get_logger("mpc_utils"), oss_idx_l.str().c_str());
        // RCLCPP_INFO(rclcpp::get_logger("mpc_utils"), oss_idx_r.str().c_str());

        // // check if input_arclength is ascending, if not, print the index
        // for (size_t i = 0; i < input_arclength.size() - 1; ++i) {
        //   if (input_arclength.at(i) >= input_arclength.at(i + 1)) {
        //     std::ostringstream oss;
        //     oss << i << ">=" << i + 1;
        //     RCLCPP_INFO(rclcpp::get_logger("mpc_utils"), oss.str().c_str());
        //   }
        // }
        ```

3. In the running `control docker container`, compile the code by running the following commands:

    ```bash
    cd /autoware
    source install/setup.bash
    colcon build --packages-select pid_longitudinal_controller mpc_lateral_controller
    ```

4. Restart the docker-compose by running the following commands on the `host machine`:

    ```bash
    cd commonroad-repairer
    cd autoware-repair-docker/compose_launch

    # Stop the docker containers. Do not use `down` command, because `down` will remove the containers
    docker compose -f planning_sim_crrepair.yml --env-file .env stop # Or Ctrl+C to terminate the previous docker compose terminal.
    docker compose -f planning_sim_crrepair.yml --env-file .env up # Start the docker containers
    ```

## Notes

Scenario id of cr2repair is `DEU_GarchingCampus2D2D-2`
