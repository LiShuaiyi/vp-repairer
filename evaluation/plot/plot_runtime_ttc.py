import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
csv_path = "highD_rg1_3_clean_sampling_updated.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)

# Calculate mean and standard deviation of total_time grouped by num_obstacles
stats = df.groupby("num_obstacles")["total_time"].agg(["mean", "std"]).reset_index()

# Create a figure with subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Subplot 1: Scatter plot with mean and std deviation
ax[0].scatter(df["num_obstacles"], df["total_time"], alpha=0.6, label="Data Points")
ax[0].plot(stats["num_obstacles"], stats["mean"], label="Mean", linewidth=2, color="orange")
ax[0].fill_between(
    stats["num_obstacles"],
    stats["mean"] - stats["std"],
    stats["mean"] + stats["std"],
    color="orange",
    alpha=0.3,
    label="Std Deviation",
)
ax[0].set_title("Total Time vs Number of Obstacles")
ax[0].set_ylabel("Total Time")
ax[0].legend()
ax[0].grid(True)

# Subplot 2: Boxplot
df.boxplot(column="total_time", by="num_obstacles", ax=ax[1], grid=False, patch_artist=True)
ax[1].set_title("Distribution of Total Time by Number of Obstacles")
ax[1].set_xlabel("Number of Obstacles")
ax[1].set_ylabel("Total Time")
fig.suptitle("")  # Remove the automatic Boxplot title

# Adjust layout and invert x-axis
plt.tight_layout()
plt.gca().invert_xaxis()

# Show the plot
plt.show()
