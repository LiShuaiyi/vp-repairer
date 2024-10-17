import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files
df_repair_1 = pd.read_csv("inD_rin1_4_clean_repair.csv")
df_repair_2 = pd.read_csv("highD_rg1_3_clean_repair.csv")

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

# Final plot adjustments
plt.tight_layout()

# Show the plot
plt.show()
