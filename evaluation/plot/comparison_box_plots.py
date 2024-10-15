import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Function to load CSV and extract only the third and last columns
def load_csv(file_path):
    with open(file_path, "r") as f_r:
        reader = csv.reader(f_r)
        header = next(reader)
        data = [row for row in reader]

    # Get the third and last columns
    third_column = [row[2] for row in data]
    last_column = [row[-1] for row in data]

    return pd.DataFrame({
        'R_IN': third_column,
        'Value': pd.to_numeric(last_column, errors='coerce')
    })


# Load the three datasets
df_repair = load_csv("inD_rin1_4_clean_repair.csv")
df_micp = load_csv("inD_rin1_4_clean_micp.csv")
df_sampling = load_csv("inD_rin1_4_clean_sampling.csv")

# Add a column to distinguish cases
df_repair['case'] = 'Repair'
df_micp['case'] = 'MICP'
df_sampling['case'] = 'Sampling'

# Combine all data
df_all = pd.concat([df_repair, df_micp, df_sampling])

# Set style for the plot
sns.set(style="whitegrid")

# Create the horizontal box plot with Seaborn for better aesthetics
plt.figure(figsize=(10, 6))

# Plot grouped boxplot
sns.boxplot(x='Value', y='R_IN', hue='case', data=df_all,
            palette={ 'Repair': 'lightgreen','MICP': 'lightblue', 'Sampling': 'lightpink',}, orient='h')

# Customize the plot
plt.xlabel('Computation time [ms]')
plt.xscale('log')
plt.xlim(left=0.01)  # Set the x-axis lower limit to a small value close to zero (e.g., 1)
plt.ylabel('')
plt.legend(title='', loc='upper right', frameon=True)
plt.title('Comparison of MICP, Sampling, and Repair')

# Adjust x-axis to log scale (if needed, as in your reference image)
plt.xscale('log')

# Show the plot
plt.tight_layout()
plt.show()
