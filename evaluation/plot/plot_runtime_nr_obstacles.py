import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "highD_inD_total_sampling.csv"  # Replace with your CSV file path
# csv_path = "highD_rg1_3_clean_sampling_updated.csv"
df = pd.read_csv(csv_path)

# Create a boxplot for total_time grouped by num_obstacles
plt.figure(figsize=(10, 6))
boxplot = df.boxplot(
    column="total_time",
    by="num_obstacles",
    grid=False,
    showmeans=True,
    patch_artist=True,
    return_type="dict",
)

# Add a title and labels
plt.title("Boxplot of Total Time by Number of Obstacles", fontsize=14)
plt.suptitle("")  # Remove the automatic 'Boxplot grouped by' title
plt.xlabel("Number of Obstacles", fontsize=12)
plt.ylabel("Total Time", fontsize=12)

# Highlight the maximum total_time per num_obstacles
max_values = df.groupby("num_obstacles")["total_time"].max()
for i, num_obs in enumerate(max_values.index, start=1):
    plt.scatter(
        [i], [max_values[num_obs]],
        color="red",
        label="Max Value" if i == 1 else "",  # Add label only once for the legend
        zorder=3,
    )

# Add legend
plt.legend(loc="upper left")

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()