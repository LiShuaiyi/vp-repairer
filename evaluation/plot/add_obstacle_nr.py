import pandas as pd
import os
from commonroad.common.file_reader import CommonRoadFileReader

# Define the path to the CSV file and scenario folder
csv_path = "./highD_rg1_3_clean_sampling.csv"
# csv_path = "./inD_rin1_4_clean_sampling.csv"
scenario_folder = "/home/liny/Documents/commonroad/highD-repair/"
# scenario_folder = "/home/liny/Documents/commonroad/ind_scenarios_2024/"

# Function to count obstacles in a scenario file
def count_obstacles_in_scenario(scenario_id):
    scenario_path = os.path.join(scenario_folder, scenario_id + ".xml")  # Assuming XML file extension
    if not os.path.exists(scenario_path):
        print(f"Scenario file not found: {scenario_path}")
        return None  # Return None if the file does not exist
    try:
        # Open the scenario file using CommonRoadFileReader
        crscenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
        return len(crscenario.obstacles)  # Return the count of obstacles
    except Exception as e:
        print(f"Error processing {scenario_path}: {e}")
        return None

# Read the CSV file
df = pd.read_csv(csv_path)

# Apply the function to count obstacles for each scenario_id
df["num_obstacles"] = df["scenario_id"].apply(count_obstacles_in_scenario)

# Save the updated CSV with the new column
output_path = "./highD_rg1_3_clean_sampling_updated.csv"
# output_path = "./inD_rin1_4_clean_sampling_updated.csv"
df.to_csv(output_path, index=False)

print(f"Updated CSV saved to {output_path}")
