


from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.world import World
from micp.constraints import InSameLaneConstraint
from crmonitor.evaluation.evaluation import RuleEvaluator

scenario_path = "../scenarios/DEU_Gar-1_1_T-1.xml"

# Open the scenario
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

world = World.create_from_scenario(scenario)

ego_id = 200

other_id = 202

in_same_lane_constr = InSameLaneConstraint()
in_same_lane_constr.compute(
    world.vehicle_by_id(other_id), 0, 20
)
