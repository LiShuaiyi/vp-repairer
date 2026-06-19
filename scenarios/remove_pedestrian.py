import matplotlib.pyplot as plt

# import functions to read xml file and visualize commonroad objects
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.scenario.obstacle import ObstacleType

plt.rcParams['figure.max_open_warning'] = 50

# generate path of the file to be opened
file_path = ("/home/liny/repairverse/commonroad-repairer/"
             "scenarios/DEU_AAH1-2_81650_T-1799.xml")

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

for veh in scenario.obstacles:
    if veh.obstacle_type != ObstacleType.CAR:
        scenario.remove_obstacle(veh)

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.scenario.scenario import Tag

author = 'Yuanfei Lin'
affiliation = 'Technical University of Munich, Germany'
source = ''
tags = {Tag.CRITICAL, Tag.INTERSECTION}

# write new scenario
fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)

filename = "DEU_AAH1-2_81650_T-1799.xml"
fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)