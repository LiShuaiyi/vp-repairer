import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum


class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMred = (227 / 255, 27 / 255, 35 / 255)
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMblack = (0, 0, 0)


def plot_segments(file_path, color, label, column_filter, ax, idx, linestyle, linewidth, marker=None, zorder=None, markersize=None):
    """
    Plots segments from a CSV file on the provided axis with specific styles.
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
        ax.plot(
            time[start:end + 1],
            values[start:end + 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            zorder=zorder,
            label=label if start == segments[0][0] else "",
        )


def plot_general(file_path, measure_data, ax, idx, title, ylabel):
    """
    Generalized plotting function for plan, repaired, and measured data.
    """
    # Plan trajectory
    plot_segments(
        file_path="plan_data.csv",
        color=TUMcolor.TUMblue.value,
        label="Plan Trajectory",
        column_filter=(0, 50),
        ax=ax,
        idx=idx,
        linestyle="-",
        linewidth=1.5,
    )

    # Repaired trajectory
    plot_segments(
        file_path="repaired_data.csv",
        color=TUMcolor.TUMgreen.value,
        label="Repaired Trajectory",
        column_filter=("s", "50"),
        ax=ax,
        idx=idx,
        linestyle=(0, (5, 5)),
        linewidth=1.5,
        zorder=5,
    )

    # Measured trajectory
    time = measure_data.iloc[:, 0]  # Time column
    values = measure_data.iloc[:, idx - 1]  # Corresponding data column
    ax.plot(
        time,
        values,
        color=TUMcolor.TUMblack.value,
        linewidth=2.5,
        marker="+",
        linestyle="",
        markersize=6,
        zorder=1,
        label="Measured Trajectory",
    )

    # Axis styling
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def load_and_plot():
    measure_data = pd.read_csv("measure_data.csv")

    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    # Plot velocity
    plot_general(
        file_path="plan_data.csv",
        measure_data=measure_data,
        ax=axes[0],
        idx=2,
        title="Velocity Segments and Measured Curve",
        ylabel="Velocity [m/s]",
    )
    axes[0].set_ylim(1494, 1508)
    axes[0].set_yticks([1495, 1505])


    # Plot position 1
    plot_general(
        file_path="plan_data.csv",
        measure_data=measure_data,
        ax=axes[1],
        idx=3,
        title="Position 1 Segments",
        ylabel="Position [m]",
    )
    axes[1].set_ylim(1869, 1881)
    axes[1].set_yticks([1870, 1880])
    # Plot position 2
    plot_general(
        file_path="plan_data.csv",
        measure_data=measure_data,
        ax=axes[2],
        idx=4,
        title="Position 2 Segments",
        ylabel="Position [m]",
    )
    axes[2].set_ylim(-0.5, 1.8)
    axes[2].set_yticks([0, 1])
    # Plot acceleration (or other metric)
    plot_general(
        file_path="plan_data.csv",
        measure_data=measure_data,
        ax=axes[3],
        idx=5,
        title="Acceleration Segments",
        ylabel="Acceleration [m/s²]",
    )
    axes[3].set_ylim(-1.6, 1.5)
    axes[3].set_yticks([-1, 0, 1])
    # Set shared x-label
    axes[-1].set_xlabel("Time [s]")
    # Set x-axis limit for all subplots
    for ax in axes:
        ax.set_xlim(60, 75)  # Limit x-axis between 60 and 75
        ax.set_xticks(range(60, 75, 5))  # Display x-ticks on the second subplot only
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


# Call the function to load and plot
load_and_plot()
