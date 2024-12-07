import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "plan_data.csv"
data = pd.read_csv(file_path, header=None)  # Assuming no header row

# Extract the required columns
time = data.iloc[:, 0]  # First column is time
velocity = data.iloc[:, 4]  # Fourth column is velocity
segment_column = data.iloc[:, 1]  # Second column for segmentation

# Find start and end indices where the second column goes from 0 to 50
segments = []
start_index = None
for idx, value in enumerate(segment_column):
    if value == 0:
        start_index = idx
    elif value == 50 and start_index is not None:
        segments.append((start_index, idx))
        start_index = None

# Plot each segment
plt.figure(figsize=(12, 6))
for i, (start, end) in enumerate(segments):
    plt.plot(time[start:end + 1], velocity[start:end + 1], color='black')


# Load the CSV file
file_path = "repaired_data.csv"
data = pd.read_csv(file_path, header=None)  # Assuming no header row

# Extract the required columns
time = data.iloc[:, 0]  # First column is time
velocity = data.iloc[:, 4]  # Fourth column is velocity
segment_column = data.iloc[:, 1]  # Second column for segmentation

# Find start and end indices where the second column goes from 0 to 50
segments = []
start_index = None
for idx, value in enumerate(segment_column):
    if value == 0 or value == 's':
        print("start", idx, value)
        start_index = idx
    print(value, value == '50')
    if value == '50':
        segments.append((start_index, idx))
        start_index = None

# Plot each segment
for i, (start, end) in enumerate(segments):
    plt.plot(time[start:end + 1], velocity[start:end + 1], color='red')

plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Velocity over Time (Segments where Column 2 ranges from 0 to 50)")
plt.grid(True)
plt.show()
