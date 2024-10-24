import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator

# Function to load CSV and extract only the third and last columns
# Also converts the values from seconds to milliseconds
def load_csv(file_path):
    with open(file_path, "r") as f_r:
        reader = csv.reader(f_r)
        header = next(reader)
        data = [row for row in reader]

    # Get the third and last columns
    third_column = [row[2] for row in data]
    last_column = [row[-1] for row in data]

    # Convert the last column from seconds to milliseconds
    return pd.DataFrame({
        'R_IN': third_column,
        'Value': pd.to_numeric(last_column, errors='coerce') * 1000  # Convert to milliseconds
    })

# Load the three datasets for inD
df_repair = load_csv("inD_rin1_4_clean_repair.csv")
df_micp = load_csv("inD_rin1_4_clean_micp.csv")
df_sampling = load_csv("inD_rin1_4_clean_sampling.csv")

# Load the three datasets for highD
hd_repair = load_csv("highD_rg1_3_clean_repair.csv")
hd_micp = load_csv("highD_rg1_3_clean_micp.csv")
hd_sampling = load_csv("highD_rg1_3_clean_sampling.csv")

# Add a column to distinguish cases for inD
df_repair['case'] = 'Repair'
df_micp['case'] = 'MICP'
df_sampling['case'] = 'Sampling'

# Add a column to distinguish cases for highD
hd_repair['case'] = 'Repair'
hd_micp['case'] = 'MICP'
hd_sampling['case'] = 'Sampling'

# Add a dataset column to distinguish between inD and highD
df_repair['dataset'] = 'inD'
df_micp['dataset'] = 'inD'
df_sampling['dataset'] = 'inD'

hd_repair['dataset'] = 'highD'
hd_micp['dataset'] = 'highD'
hd_sampling['dataset'] = 'highD'

# Combine only the highD datasets into a single DataFrame
df_hd_combined = pd.concat([hd_repair, hd_micp, hd_sampling])

# Add an extra 'R_IN' value to distinguish 'hd_' rule with label 'R_G13'
df_hd_combined['R_IN'] = 'R_G13'

# Combine all inD datasets into a single DataFrame
df_inD_combined = pd.concat([df_repair, df_micp, df_sampling])

# Combine both inD and highD datasets
df_all = pd.concat([df_hd_combined, df_inD_combined])

# Set style for the plot
sns.set(style="whitegrid")

# Create the horizontal box plot with Seaborn for better aesthetics
# Adjust the figure size for a 16:9 aspect ratio
plt.figure(figsize=(8, 4.5))

# Create a boxplot with both color and hatch patterns for better differentiation
ax = sns.boxplot(x='Value', y='R_IN', hue='case', data=df_all,
                 palette={'Repair': 'lightgreen', 'MICP': 'lightblue', 'Sampling': 'lightpink'}, orient='h')

# Apply hatch patterns for each category
hatches = {'Repair': '////', 'MICP': '\\\\', 'Sampling': 'oo'}  # Different hatch patterns for categories

# Get the number of patches and loop through them
n_patches = len(ax.patches)
n_cases = len(df_all['case'].unique())
n_rins = len(df_all['R_IN'].unique())

# Loop through the patches and apply the hatch patterns based on their case
for i, patch in enumerate(ax.patches):
    # Determine the correct 'case' by dividing the patch index by the number of R_IN categories
    case_index = i % n_cases
    case = list(df_all['case'].unique())[case_index]
    patch.set_hatch(hatches[case])
    patch.set_edgecolor('black')  # Use black borders for better contrast

# Customize the plot
plt.xlabel('Computation time [ms]')  # Change label to reflect milliseconds
plt.xscale('log')

# Set ticks on the x-axis for better visibility
xticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]  # Custom ticks
ax = plt.gca()
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))

# Enable grid for minor ticks
ax.grid(True, which='both', axis='x')
ax.grid(True, which='both', axis='y')

# Set custom ticks and limits for the x-axis
plt.xticks(xticks, labels=[str(x) for x in xticks])
plt.xlim(left=50, right=10000)  # Set the x-axis lower limit to a small value close to zero (e.g., 10ms)

# Final plot adjustments
plt.ylabel('')  # Remove y-axis label
plt.legend(title='', loc='upper right', frameon=True)

# Show the plot
plt.tight_layout()
plt.show()
