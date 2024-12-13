import pandas as pd
import os
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.measure import TTC
from commonroad_crime.data_structure.crime_interface import CriMeInterface

# Define the path to the CSV file and scenario folder
# csv_path = "./highD_rg1_3_clean_sampling.csv"
csv_path = "./inD_rin1_4_clean_sampling.csv"
csv_path = "./inD_rin1_4_clean_repair.csv"
csv_path = "./highD_rg1_3_clean_repair.csv"
csv_path = "./highD_rg1_3_clean_repair_no_der.csv"
csv_path = "./inD_rin1_4_clean_repair_no_der.csv"
# scenario_folder = "/home/liny/Documents/commonroad/highD-repair/"
scenario_folder = "/home/liny/Documents/commonroad/ind_scenarios_2024/"

# Function to evaluate the scenario and perform desired computations
def count_obstacles_in_scenario(row):
    scenario_id = row["scenario_id"]
    ego_id = row["ego_id"]
    scenario_path = os.path.join(scenario_folder, scenario_id + ".xml")  # Assuming XML file extension
    if not os.path.exists(scenario_path):
        print(f"Scenario file not found: {scenario_path}")
        return None  # Return None if the file does not exist
    try:
        # Open the scenario file using CommonRoadFileReader
        crscenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
        config = CriMeConfiguration()
        config.general.path_scenarios = scenario_folder
        config.general.set_scenario_name(scenario_id)
        config.vehicle.ego_id = ego_id
        config.update()

        crime_interface = CriMeInterface(config)
        # Perform evaluation
        crime_interface.evaluate_scenario([TTC], time_start=0, time_end=20)
        # Return the number of obstacles (or any other desired computation)
        print(min(item['time-to-collision'] for item in crime_interface.criticality_dict.values()))
        return min(item['time-to-collision'] for item in crime_interface.criticality_dict.values())
    except Exception as e:
        print(f"Error processing {scenario_path}: {e}")
        return None

# Read the CSV file
df = pd.read_csv(csv_path)

# Apply the function to evaluate each scenario and add the result as a new column
df["num_obstacles"] = df.apply(count_obstacles_in_scenario, axis=1)

# Save the updated CSV with the new column
# output_path = "./highD_rg1_3_clean_sampling_updated.csv"
# output_path = "./inD_rin1_4_clean_sampling_updated_ttc.csv"
# output_path = "./inD_rin1_4_clean_repair_updated_ttc.csv"
# output_path = "./highD_rg1_3_clean_repair_updated_ttc.csv"
# output_path = "./highD_rg1_3_clean_repair_no_der_updated_ttc.csv"
output_path = "./inD_rin1_4_clean_repair_no_der_updated_ttc.csv"
df.to_csv(output_path, index=False)

print(f"Updated CSV saved to {output_path}")
