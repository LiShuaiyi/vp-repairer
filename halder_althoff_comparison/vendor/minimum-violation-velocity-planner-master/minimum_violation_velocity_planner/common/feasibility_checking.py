import numpy as np
from commonroad.common.solution import VehicleType
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.feasibility import feasibility_checker
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
from matplotlib import pyplot as plt


def isFeasible(trajectory: Trajectory, dt, show_plots=False):
    # Check trajectory for feasibility
    vehicle = VehicleDynamics.PM(VehicleType.BMW_320i)
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory, vehicle, dt)

    # reconstruct the trajectory using the reconstructed input vector inputs step wise
    reconstructed_states = [vehicle.convert_initial_state(trajectory.state_list[0])] + [
        vehicle.simulate_next_state(trajectory.state_list[idx], inp, dt)
        for idx, inp in enumerate(reconstructed_inputs.state_list)
    ]
    trajectory_reconstructed = Trajectory(initial_time_step=0, state_list=reconstructed_states)

    error_state = len(trajectory_reconstructed.state_list)
    if not feasible:
        print('Error found at state transition {} to {}'.format(error_state - 2, error_state - 1))

    if show_plots:
        ## Plot Figures
        fsize = (10, 5)

        # position figure
        orig_pos = np.array([x.position for x in trajectory.state_list])
        recon_pos = np.array([x.position for x in trajectory_reconstructed.state_list])

        plt.figure(figsize=fsize)
        plt.plot(orig_pos[:, 0], orig_pos[:, 1], '-or')
        plt.plot(recon_pos[:, 0], recon_pos[:, 1], '-xk')
        plt.legend(['Original', 'Reconstructed'], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('position x')
        plt.ylabel('position y')
        plt.axis('equal')
        plt.autoscale()
        plt.show()

        # velocity figure
        orig_velocity = np.array([[x.time_step, x.velocity] for x in trajectory.state_list])
        recon_velocity = np.array([[x.time_step, x.velocity] for x in trajectory_reconstructed.state_list])

        plt.figure(figsize=fsize)
        plt.plot(orig_velocity[:, 0], orig_velocity[:, 1], '-or')
        plt.plot(recon_velocity[:, 0], recon_velocity[:, 1], '-xk')
        plt.xlabel('time step k')
        plt.ylabel('velocity v')
        plt.legend(['Original', 'Reconstructed'], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.autoscale()
        plt.show()

        # orientation figure
        orig_orientation = np.array([[x.time_step, x.orientation] for x in trajectory.state_list])
        recon_orientation = np.array([[x.time_step, x.orientation] for x in trajectory_reconstructed.state_list])

        plt.figure(figsize=fsize)
        plt.plot(orig_orientation[:, 0], orig_orientation[:, 1], '-or')
        plt.plot(recon_orientation[:, 0], recon_orientation[:, 1], '-xk')
        plt.legend(['Original', 'Reconstructed'], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('time step k')
        plt.ylabel('orientation theta')
        plt.autoscale()
        plt.show()

        # acceleration figure
        orig_acc = np.array([[u.time_step, u.acceleration] for u in trajectory.state_list])
        recon_acc = np.array([[u.time_step, u.acceleration] for u in reconstructed_inputs.state_list])

        plt.figure(figsize=fsize)
        plt.plot(orig_acc[:, 0], orig_acc[:, 1], '-or')
        plt.plot(recon_acc[:, 0], recon_acc[:, 1], '-xk')
        plt.legend(['Original', 'Reconstructed'], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('time step k')
        plt.ylabel('acceleration a')
        plt.autoscale()
        plt.show()

        # trajectory error figure
        pos_error = np.array([
            [recon.time_step,
             np.sqrt(np.sum(np.square(np.subtract(trajectory.state_list[idx].position, recon.position))))]
            for idx, recon in enumerate(trajectory_reconstructed.state_list)
        ])

        vel_error = np.array([
            [recon.time_step, abs(recon.velocity - trajectory.state_list[idx].velocity)]
            for idx, recon in enumerate(trajectory_reconstructed.state_list)
        ])

        orient_error = np.array([
            [recon.time_step, abs(recon.orientation - trajectory.state_list[idx].orientation)]
            for idx, recon in enumerate(trajectory_reconstructed.state_list)
        ])

        plt.figure(figsize=fsize)
        plt.plot(pos_error[:, 0], pos_error[:, 1], 'r')
        plt.plot(orient_error[:, 0], orient_error[:, 1], 'b')
        plt.plot(pos_error[:, 0], np.repeat(2e-2, error_state), '-k')
        plt.legend(['Position Error (m)', 'Orientation Error (r)', 'Error 2e-2'],
                   loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('time step k')
        plt.ylabel('computed error')
        plt.autoscale()
        plt.show()

        # input vector error figure
        acc_error = np.array([
            [recon.time_step, abs(recon.acceleration - trajectory.state_list[idx].acceleration)]
            for idx, recon in enumerate(reconstructed_inputs.state_list)
        ])

        plt.figure(figsize=fsize)
        plt.plot(acc_error[:, 0], acc_error[:, 1], 'g')
        plt.legend(['Acceleration Error (m/s2)'],
                   loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('time step k')
        plt.ylabel('computed error')
        plt.autoscale()
        plt.show()

    return feasible
