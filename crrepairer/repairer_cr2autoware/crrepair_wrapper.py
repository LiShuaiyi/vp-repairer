import sys
import pandas as pd
import json
import ast
import math
from datetime import datetime
import pytz
from pathlib import Path
from typing import List, Set
import os, time
import matplotlib.pyplot as plt

def RepairerConfiguration_load(): # -> RepairerConfiguration
    from crrepairer.utils.configuration import RepairerConfiguration
    
    """Wrapper for loading the configuration file for the repairer."""
    path_config_crrepair2autoware = Path(__file__).resolve().parent.parent.parent / 'config' / 'config_repair_DEU_GarchingCampus2D2D-2.yaml'
    print(path_config_crrepair2autoware)
    return RepairerConfiguration.load(path_config_crrepair2autoware, "DEU_GarchingCampus2D2D-2")

class CRRepairLogger:
    def __init__(self, crrepair_log_file_timestamp_formatted: str):
        self.crrepair_log_file_timestamp_formatted = crrepair_log_file_timestamp_formatted
        self.scenario_dt = 0.1
        self.initial_trajectory_color = "blue"
        self.repaired_trajectory_color = "red"
        self.measured_trajectory_color = "black"
        self.line_style = "-"
        self.marker = ""
        self.linewidth = 1.0

        os.makedirs(os.path.join('output', 'crrepairer2autoware', 'data_logs'), exist_ok=True)
        self.csv_path_trajectory = os.path.join('output', 'crrepairer2autoware', 'data_logs', "".join([crrepair_log_file_timestamp_formatted, '_trajectory', '.csv']))
        self.csv_path_vehicle_state = os.path.join('output', 'crrepairer2autoware', 'data_logs', "".join([crrepair_log_file_timestamp_formatted, '_vehicle_state', '.csv']))

    @classmethod
    def build_from_trajectory_csv(cls, csv_path_trajectory: str):
        if not csv_path_trajectory:
            raise ValueError("The csv_path_trajectory cannot be empty.")
        crrepair_log_file_timestamp_formatted = os.path.basename(csv_path_trajectory)[:22]
        return cls(crrepair_log_file_timestamp_formatted)
    
    def log_repairer_trajectory(self, crrepair_log_current_timestamp: float, crrepair_log_data_dict: dict):
        """
        Save the initial and repaired CR states to a file.

        Args:
            crrepair_log_current_timestamp (float): Current timestamp from time.time().
            crrepair_log_data_dict (dict): Dictionary containing the log data.
                'initial_trajectory' (List[TraceState]): States of the initial trajectory, vehicle center coordinate.
                'repaired_trajectory' (List[TraceState]): States of the repaired trajectory, vehicle center coordinate.
        """
        germany_tz = pytz.timezone('Europe/Berlin')
        crrepair_log_current_timestamp_formatted = "".join([datetime.fromtimestamp(crrepair_log_current_timestamp, germany_tz).strftime('%Y-%m-%d_%H-%M-%S'), "_", f"{crrepair_log_current_timestamp % 1:.2f}"[2:]])

        initial_time_step_list: List[int] = []
        initial_x_list: List[float] = []
        initial_y_list: List[float] = []
        initial_velocity_list: List[float] = []
        initial_acceleration_list: List[float] = []

        repaired_time_step_list: List[int] = []
        repaired_x_list: List[float] = []
        repaired_y_list: List[float] = []
        repaired_velocity_list: List[float] = []
        repaired_acceleration_list: List[float] = []

        ego_history_time_step_list: List[int] = []
        ego_history_x_list: List[float] = []
        ego_history_y_list: List[float] = []
        ego_history_velocity_list: List[float] = []
        ego_history_acceleration_list: List[float] = []

        for initial_state in crrepair_log_data_dict['initial_trajectory']:
            initial_time_step_list.append(initial_state.time_step)
            initial_x_list.append(initial_state.position[0])
            initial_y_list.append(initial_state.position[1])
            initial_velocity_list.append(initial_state.velocity)
            initial_acceleration_list.append(initial_state.acceleration)

        for repaired_state in crrepair_log_data_dict['repaired_trajectory']:
            repaired_time_step_list.append(repaired_state.time_step)
            repaired_x_list.append(repaired_state.position[0])
            repaired_y_list.append(repaired_state.position[1])
            repaired_velocity_list.append(repaired_state.velocity)
            repaired_acceleration_list.append(repaired_state.acceleration)

        for ego_state in crrepair_log_data_dict['ego_state_history']:
            ego_history_time_step_list.append(ego_state.time_step)
            ego_history_x_list.append(ego_state.position[0])
            ego_history_y_list.append(ego_state.position[1])
            ego_history_velocity_list.append(ego_state.velocity)
            ego_history_acceleration_list.append(ego_state.acceleration)

        new_data_max_length = max(len(initial_time_step_list), len(crrepair_log_data_dict['ego_state_history_timestamp']))

        new_data = {
            'timestamp': crrepair_log_current_timestamp,
            'crrepair_log_current_timestamp_formatted': [crrepair_log_current_timestamp_formatted] + [None] * (new_data_max_length - 1),
            'initial_time_step': initial_time_step_list + [None] * (new_data_max_length - len(initial_time_step_list)),
            'initial_x': initial_x_list + [None] * (new_data_max_length - len(initial_x_list)),
            'initial_y': initial_y_list + [None] * (new_data_max_length - len(initial_y_list)),
            'initial_velocity': initial_velocity_list + [None] * (new_data_max_length - len(initial_velocity_list)),
            'initial_acceleration': initial_acceleration_list + [None] * (new_data_max_length - len(initial_acceleration_list)),
            'repaired_time_step': repaired_time_step_list + [None] * (new_data_max_length - len(repaired_time_step_list)),
            'repaired_x': repaired_x_list + [None] * (new_data_max_length - len(repaired_x_list)),
            'repaired_y': repaired_y_list + [None] * (new_data_max_length - len(repaired_y_list)),
            'repaired_velocity': repaired_velocity_list + [None] * (new_data_max_length - len(repaired_velocity_list)),
            'repaired_acceleration': repaired_acceleration_list + [None] * (new_data_max_length - len(repaired_acceleration_list)),
            'TV': [crrepair_log_data_dict['TV']] + [None] * (new_data_max_length - 1),
            'TC': [crrepair_log_data_dict['TC']] + [None] * (new_data_max_length - 1),
            'ego_state_history_timestamp': crrepair_log_data_dict['ego_state_history_timestamp'] + [None] * (new_data_max_length - len(crrepair_log_data_dict['ego_state_history_timestamp'])),
            'ego_history_time_step': ego_history_time_step_list + [None] * (new_data_max_length - len(ego_history_time_step_list)),
            'ego_history_x': ego_history_x_list + [None] * (new_data_max_length - len(ego_history_x_list)),
            'ego_history_y': ego_history_y_list + [None] * (new_data_max_length - len(ego_history_y_list)),
            'ego_history_velocity': ego_history_velocity_list + [None] * (new_data_max_length - len(ego_history_velocity_list)),
            'ego_history_acceleration': ego_history_acceleration_list + [None] * (new_data_max_length - len(ego_history_acceleration_list)),
            'ego_history_center_lanelet_ids': crrepair_log_data_dict['ego_center_lanelet_ids_history'] + [None] * (new_data_max_length - len(crrepair_log_data_dict['ego_center_lanelet_ids_history'])),
            'ego_history_shape_lanelet_ids': crrepair_log_data_dict['ego_shape_lanelet_ids_history'] + [None] * (new_data_max_length - len(crrepair_log_data_dict['ego_shape_lanelet_ids_history'])),
        }

        df = pd.DataFrame(new_data)

        # diff of ego_state_history_timestamp to the right of ego_state_history_timestamp
        df['ego_state_history_timestamp_diff'] = df['ego_state_history_timestamp'].diff()
        # move ego_state_history_timestamp_diff to the right of ego_state_history_timestamp
        df.insert(df.columns.get_loc('ego_state_history_timestamp') + 1, 'ego_state_history_timestamp_diff', df.pop('ego_state_history_timestamp_diff'))

        df.to_csv(self.csv_path_trajectory, mode='a', header=not os.path.exists(self.csv_path_trajectory), index=False)
        

    def plot_repairer_trajectory_at_timestamp(self, crrepair_log_current_timestamp: float):
        df_trajectory = pd.read_csv(self.csv_path_trajectory)

        # find the row with timestamp == crrepair_log_current_timestamp
        row = df_trajectory[df_trajectory['timestamp'] == crrepair_log_current_timestamp]

        initial_time_step_list: List[int] = row['initial_time_step'].tolist()
        initial_x_list: List[float] = row['initial_x'].tolist()
        initial_y_list: List[float] = row['initial_y'].tolist()
        initial_velocity_list: List[float] = row['initial_velocity'].tolist()
        initial_acceleration_list: List[float] = row['initial_acceleration'].tolist()

        repaired_time_step_list: List[int] = row['repaired_time_step'].tolist()
        repaired_x_list: List[float] = row['repaired_x'].tolist()
        repaired_y_list: List[float] = row['repaired_y'].tolist()
        repaired_velocity_list: List[float] = row['repaired_velocity'].tolist()
        repaired_acceleration_list: List[float] = row['repaired_acceleration'].tolist()

        if not row.empty:
            if row['TV'].iat[0] == math.inf:
                TV = math.inf
            else:
                TV = int(row['TV'].iat[0])
            if row['TC'].iat[0] == math.inf:
                TC = math.inf
            else:
                TC = int(row['TC'].iat[0])
        else:
            TV = math.inf
            TC = math.inf

        germany_tz = pytz.timezone('Europe/Berlin')
        crrepair_log_current_timestamp_formatted = "".join([datetime.fromtimestamp(crrepair_log_current_timestamp, germany_tz).strftime('%Y-%m-%d_%H-%M-%S'), "_", f"{crrepair_log_current_timestamp % 1:.2f}"[2:]])

        os.makedirs(os.path.join('output', 'crrepairer2autoware', 'figures', "".join([self.crrepair_log_file_timestamp_formatted, '_trajectory'])), exist_ok=True)

        fig, axs = plt.subplots(4, 1, figsize=(15, 15))
        fig.suptitle(f"Trajectory Repair | TV: {TV}, TC: {TC}")

        axs[0].plot(repaired_time_step_list, repaired_x_list, color=self.repaired_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='repaired_x')
        axs[0].plot(initial_time_step_list, initial_x_list, color=self.initial_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='initial_x')
        axs[0].set_xlabel("Time step [1/0.1s]")
        axs[0].set_ylabel("X [m]")
        axs[0].legend()

        axs[1].plot(repaired_time_step_list, repaired_y_list, color=self.repaired_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='repaired_y')
        axs[1].plot(initial_time_step_list, initial_y_list, color=self.initial_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='initial_y')
        axs[1].set_xlabel("Time step [1/0.1s]")
        axs[1].set_ylabel("Y [m]")
        axs[1].legend()

        axs[2].plot(repaired_time_step_list, repaired_velocity_list, color=self.repaired_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='repaired_velocity')
        axs[2].plot(initial_time_step_list, initial_velocity_list, color=self.initial_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='initial_velocity')
        axs[2].set_xlabel("Time step [1/0.1s]")
        axs[2].set_ylabel("Velocity [m/s]")
        axs[2].legend()

        axs[3].plot(repaired_time_step_list, repaired_acceleration_list, color=self.repaired_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='repaired_acceleration')
        axs[3].plot(initial_time_step_list, initial_acceleration_list, color=self.initial_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='initial_acceleration')
        axs[3].set_xlabel("Time step [1/0.1s]")
        axs[3].set_ylabel("Acceleration [m/s^2]")
        axs[3].legend()

        plt.tight_layout()
        plt.savefig(os.path.join('output', 'crrepairer2autoware', 'figures', "".join([self.crrepair_log_file_timestamp_formatted, '_trajectory']), f"{crrepair_log_current_timestamp_formatted}_trajectory_repair_TV_{TV}_TC_{TC}.svg"), format='svg')
        plt.close(fig)

    def plot_repairer_trajectory_all(self):
        print(f">>> CSV Trajectory: {Path(self.csv_path_trajectory).resolve()}")
        print(f">>> CSV Vehicle state: {Path(self.csv_path_vehicle_state).resolve()}")

        df_trajectory = pd.read_csv(self.csv_path_trajectory)
        df_vehicle_state = pd.read_csv(self.csv_path_vehicle_state)

        df_trajectory['time_planned_trajectory'] = df_trajectory['timestamp'] + df_trajectory['initial_time_step'] * self.scenario_dt
        df_trajectory.insert(df_trajectory.columns.get_loc('crrepair_log_current_timestamp_formatted') + 1, 'time_planned_trajectory', df_trajectory.pop('time_planned_trajectory'))

        start_time = min(df_trajectory['ego_state_history_timestamp'].min(), df_trajectory['time_planned_trajectory'].min())
        df_trajectory['time_planned_trajectory'] = df_trajectory['time_planned_trajectory'] - start_time
        df_trajectory['time_ego_state_history'] = df_trajectory['ego_state_history_timestamp'] - start_time
        df_trajectory.insert(df_trajectory.columns.get_loc('ego_state_history_timestamp') + 1, 'time_ego_state_history', df_trajectory.pop('time_ego_state_history'))
        df_vehicle_state['time_ego_state'] = df_vehicle_state['timestamp'] - start_time
        df_vehicle_state.insert(df_vehicle_state.columns.get_loc('timestamp') + 1, 'time_ego_state', df_vehicle_state.pop('time_ego_state'))
        df_vehicle_state = df_vehicle_state[df_vehicle_state['time_ego_state'] >= 0]

        fig, axs = plt.subplots(4, 1, figsize=(15, 15))
        fig.suptitle(f"Trajectory Repair")

        # not truncated
        # def plot_segmented(ax, x_data, y_data, color, linestyle, marker, linewidth, label):
        #     grouped = df_trajectory.groupby('timestamp')
        #     first = True
        #     for name, group in grouped:
        #         ax.plot(group[x_data], group[y_data], color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, label=label if first else "")
        #         first = False
        
        # truncated
        def plot_segmented(ax, x_data, y_data, color, linestyle, marker, linewidth, label):
            grouped = df_trajectory.groupby('timestamp')
            first = True
            timestamps = list(grouped.groups.keys())
            for i, (name, group) in enumerate(grouped):
                # Check if there is a next group
                if i < len(timestamps) - 1:
                    next_timestamp = timestamps[i + 1]
                    # Filter the group to only include x_data values less than the next timestamp
                    group = group[group[x_data] < next_timestamp - start_time] if group['TV'].iat[0] == float('inf') else group
                ax.plot(group[x_data], group[y_data], color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, label=label if first else "")
                first = False

        plot_segmented(axs[0], 'time_planned_trajectory', 'repaired_x', self.repaired_trajectory_color, self.line_style, self.marker, self.linewidth, 'repaired_x')
        plot_segmented(axs[0], 'time_planned_trajectory', 'initial_x', self.initial_trajectory_color, self.line_style, self.marker, self.linewidth, 'initial_x')
        axs[0].plot(df_vehicle_state['time_ego_state'], df_vehicle_state['x'], color=self.measured_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='measured_x')
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("X [m]")
        axs[0].legend()

        plot_segmented(axs[1], 'time_planned_trajectory', 'repaired_y', self.repaired_trajectory_color, self.line_style, self.marker, self.linewidth, 'repaired_y')
        plot_segmented(axs[1], 'time_planned_trajectory', 'initial_y', self.initial_trajectory_color, self.line_style, self.marker, self.linewidth, 'initial_y')
        axs[1].plot(df_vehicle_state['time_ego_state'], df_vehicle_state['y'], color=self.measured_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='measured_y')
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Y [m]")
        axs[1].legend()

        plot_segmented(axs[2], 'time_planned_trajectory', 'repaired_velocity', self.repaired_trajectory_color, self.line_style, self.marker, self.linewidth, 'repaired_velocity')
        plot_segmented(axs[2], 'time_planned_trajectory', 'initial_velocity', self.initial_trajectory_color, self.line_style, self.marker, self.linewidth, 'initial_velocity')
        axs[2].plot(df_vehicle_state['time_ego_state'], df_vehicle_state['v'], color=self.measured_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='measured_velocity')
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Velocity [m/s]")
        axs[2].legend()

        plot_segmented(axs[3], 'time_planned_trajectory', 'repaired_acceleration', self.repaired_trajectory_color, self.line_style, self.marker, self.linewidth, 'repaired_acceleration')
        plot_segmented(axs[3], 'time_planned_trajectory', 'initial_acceleration', self.initial_trajectory_color, self.line_style, self.marker, self.linewidth, 'initial_acceleration')
        axs[3].plot(df_vehicle_state['time_ego_state'], df_vehicle_state['a'], color=self.measured_trajectory_color, linestyle=self.line_style, marker=self.marker, linewidth=self.linewidth, label='measured_acceleration')
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel("Acceleration [m/s^2]")
        axs[3].legend()

        plt.tight_layout()
        output_folder = Path(self.csv_path_trajectory).parent.parent / "figures" / "".join([self.crrepair_log_file_timestamp_formatted, "_whole_trajectory"])
        os.makedirs(output_folder, exist_ok=True)

        df_trajectory.to_csv(output_folder / f"{self.crrepair_log_file_timestamp_formatted}_whole_trajectory_for_plotting.csv", index=False)
        df_vehicle_state.to_csv(output_folder / f"{self.crrepair_log_file_timestamp_formatted}_whole_trajectory_for_ego_state_plotting.csv", index=False)

        plt.savefig(output_folder / f"{self.crrepair_log_file_timestamp_formatted}_whole_trajectory.svg", format='svg', dpi=400)
        print("------------------------------------")
        print(f">>> Figure whole trajectory: {output_folder.resolve() / f'{self.crrepair_log_file_timestamp_formatted}_whole_trajectory.svg'}")
        
        # plt.show()
        plt.close(fig)
        
if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python script.py <crrepair_log_file_timestamp_formatted>, or set the crrepair_log_file_timestamp_formatted manually in the script")
        sys.exit(1)
    
    crrepair_log_file_timestamp_formatted_manual = "2024-10-18_15-24-14_04"
    
    crrepair_log_file_timestamp_formatted = sys.argv[1] if len(sys.argv) == 2 else crrepair_log_file_timestamp_formatted_manual

    crrepair_logger = CRRepairLogger(crrepair_log_file_timestamp_formatted)
    crrepair_logger.plot_repairer_trajectory_all()