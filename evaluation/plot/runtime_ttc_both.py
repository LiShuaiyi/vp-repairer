import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample

# Define TUM colors
class TUMColor:
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMyellow = (203 / 255, 171 / 255, 1 / 255)
    TUMblack = (0, 0, 0)

# Function to load data and perform regression analysis
def plot_regression(ax, csv_path, title, y_range, scatter_color, degree=1, y_ticks=None, x_ticks=None):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Drop rows with NaN values in `ttc` or `total_time`
    df = df.dropna(subset=["ttc", "total_time"])

    # Prepare data for regression
    X = df["ttc"].values.reshape(-1, 1)
    y = df["total_time"].values

    # Fit a polynomial regression model
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict using the model
    X_range = np.linspace(0, 22, 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_pred = model.predict(X_range_poly)

    # Plot data and regression results
    ax.scatter(
        df["ttc"], df["total_time"], alpha=1, label="Data Points",
        edgecolors='none', color=scatter_color  # Filled points without borders
    )
    ax.plot(
        X_range, y_pred, label=f"Poly Regression (Degree {degree})",
        linestyle='--', color=TUMColor.TUMblack, linewidth=2
    )
    ax.set_title(title)
    ax.set_xlabel("TTC")
    ax.set_ylabel("Total Time")
    ax.legend()
    ax.set_ylim(y_range)  # Set y-axis range
    ax.set_xlim(22, 0)    # Reverse x-axis range
    ax.set_yticks(y_ticks)  # Custom y-ticks
    ax.set_xticks(x_ticks)  # Custom x-ticks
    ax.grid(True, which='both', axis='x')
    ax.grid(True, which='both', axis='y')

# File paths for the two CSV files
csv1 = "highD_inD_ttc_repair_np_der_common.csv"
csv2 = "highD_inD_ttc_sampling_common.csv"

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Plot each file with custom settings
plot_regression(
    axes[0], csv1, title="Repair Data", y_range=(0, 0.22),
    scatter_color=TUMColor.TUMgreen, degree=1,
    y_ticks=[0, 0.2], x_ticks=[20, 10, 0]
)
plot_regression(
    axes[1], csv2, title="Sampling Data", y_range=(0, 5),
    scatter_color=TUMColor.TUMyellow, degree=1,
    y_ticks=[0, 2, 4], x_ticks=[20, 10, 0]
)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
