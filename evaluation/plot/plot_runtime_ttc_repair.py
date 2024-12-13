import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample

# Load the CSV file
csv_path = "highD_inD_ttc_repair.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)

# Drop rows with NaN values in `num_obstacles` or `total_time`
df = df.dropna(subset=["num_obstacles", "total_time"])

# Prepare data for regression
X = df["num_obstacles"].values.reshape(-1, 1)
y = df["total_time"].values

# Fit a polynomial regression model
degree =2  # Degree of the polynomial
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Predict using the model
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred = model.predict(X_range_poly)

# Compute confidence intervals using bootstrap resampling
n_bootstrap = 1000
predictions = []
for _ in range(n_bootstrap):
    X_resampled, y_resampled = resample(X, y)
    X_resampled_poly = poly.transform(X_resampled)
    model.fit(X_resampled_poly, y_resampled)
    predictions.append(model.predict(X_range_poly))

predictions = np.array(predictions)
ci_lower = np.percentile(predictions, 2.5, axis=0)
ci_upper = np.percentile(predictions, 97.5, axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df["num_obstacles"], df["total_time"], alpha=0.6, label="Data Points")
plt.plot(X_range, y_pred, label=f"Polynomial Regression (Degree {degree})", color="orange", linewidth=2)
plt.fill_between(X_range.flatten(), ci_lower, ci_upper, color="orange", alpha=0.3, label="95% Confidence Region")
plt.title("Total Time vs Number of Obstacles with Regression Curve")
plt.xlabel("TTC")
plt.ylabel("Total Time")
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # Reverse x-axis
plt.tight_layout()

# Show the plot
plt.show()
