import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum


class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMred = (227 / 255, 27 / 255, 35 / 255)
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMblack = (0, 0, 0)


def plot_segments(file_path, color, label, column_filter, ax, idx):
    """
    Plots segments from a CSV file on the provided axis.
    """
    data = pd.read_csv(file_path, header=None)
    time = data.iloc[:, 0]  # Time column
    values = data.iloc[:, idx]  # Data column (velocity, position, etc.)
    segments = []
    start_index = None

    for i, value in enumerate(data.iloc[:, 1]):  # Segmentation column
        if value == 0 or value == column_filter[0]:
            start_index = i
        elif value == column_filter[1] and start_index is not None:
            segments.append((start_index, i))
            start_index = None

    for start, end in segments:
        ax.plot(time[start:end + 1], values[start:end + 1], color=color, label=label if start == segments[0][0] else "")


def plot_general(file_path, measure_data, ax, idx, title, ylabel):
    """
    Generalized plotting function for plan, repaired, and measured data.
    """
    plot_segments(file_path="plan_data.csv", color=TUMcolor.TUMblue.value, label="Plan Data", column_filter=(0, 50), ax=ax, idx=idx)
    plot_segments(file_path="repaired_data.csv", color=TUMcolor.TUMred.value, label="Repaired Data", column_filter=('s', '50'), ax=ax, idx=idx)

    # Add Measure Data
    time = measure_data.iloc[:, 0]  # First column (time)
    values = measure_data.iloc[:, idx - 1]
    ax.plot(time, values, linestyle="--", color=TUMcolor.TUMgreen.value, label="Measure Data Curve")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)


def load_and_plot():
    measure_data = pd.read_csv("measure_data.csv")

    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    # Plot velocity
    plot_general(file_path="plan_data.csv", measure_data=measure_data, ax=axes[0], idx=4, title="Velocity Segments and Measure Curve", ylabel="Velocity [m/s]")

    # Plot position 1
    plot_general(file_path="plan_data.csv", measure_data=measure_data, ax=axes[1], idx=2, title="Position 1 Segments", ylabel="Position [m]")

    # Plot position 2
    plot_general(file_path="plan_data.csv", measure_data=measure_data, ax=axes[2], idx=3, title="Position 2 Segments", ylabel="Position [m]")

    # Plot another metric (e.g., acceleration)
    plot_general(file_path="plan_data.csv", measure_data=measure_data, ax=axes[3], idx=5, title="Acceleration Segments", ylabel="Acceleration [m/s²]")

    # Set shared x-label
    axes[-1].set_xlabel("Time [s]")

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


# Call the function to load and plot
load_and_plot()
