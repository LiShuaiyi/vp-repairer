import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files
df_repair_1 = pd.read_csv("inD_rin1_4_clean_repair_no_der.csv")
df_repair_2 = pd.read_csv("highD_rg1_3_clean_repair_no_der.csv")

# Combine the two dataframes
df_combined = pd.concat([df_repair_1, df_repair_2])

# Compute the 'other_time'
df_combined['other_time'] = df_combined['total_time'] - df_combined['reach_time'] - df_combined['SAT_time'] - df_combined['TC_time']

# Convert time values to milliseconds
df_combined[['SAT_time', 'TC_time', 'reach_time', 'other_time', 'total_time']] *= 1000

# Prepare the data for plotting
df_melted = pd.melt(df_combined, value_vars=['SAT_time', 'TC_time', 'reach_time', 'other_time', 'total_time'],
                    var_name='time_type', value_name='time_value')

# Plot the boxplot rotated 90 degrees
plt.figure(figsize=(8, 4.5))
sns.boxplot(y='time_type', x='time_value', data=df_melted)

# Customize the plot
plt.xlabel('Computation time [ms]')  # Set the x-axis label for time values
ax = plt.gca()


# Set limits for the x-axis
plt.xlim(left=-5, right=205)
plt.legend(title='', loc='upper right', frameon=True)

# Enable grid for better readability
ax.grid(True, which='both', axis='x')
ax.grid(True, which='both', axis='y')
plt.ylabel('')  # Remove y-axis label

# Print the mean computation times for each type in the console
mean_times = df_combined[['SAT_time', 'TC_time', 'reach_time', 'other_time', 'total_time']].mean()
print('mean', mean_times)

# Function to remove outliers based on 1.5*IQR rule
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for each time type
df_no_outliers = df_combined.copy()
for col in ['SAT_time', 'TC_time', 'reach_time', 'other_time', 'total_time']:
    df_no_outliers = remove_outliers(df_no_outliers, col)

# Compute the mean values without outliers
mean_no_outliers = df_no_outliers[['SAT_time', 'TC_time', 'reach_time', 'other_time', 'total_time']].mean()
print("mean", mean_no_outliers)

median_times = df_combined[['SAT_time', 'TC_time', 'reach_time', 'other_time', 'total_time']].median()
print("Median:", median_times)
# Final plot adjustments
plt.tight_layout()
# Calculate the percentage of rows where 'total_time' is less than 200 ms
total_time_less_than_200 = (df_combined['total_time'] < 200).mean() * 100

print(total_time_less_than_200)
# Show the plot
plt.show()
