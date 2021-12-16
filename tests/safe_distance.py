# Import libraries
import matplotlib.pyplot as plt
import numpy as np

from commonroad_repair.crrepairer.repairer.rule_constraints import RuleConstraints
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

def safe_distance(v_follow, v_lead: float,
                  a_min_follow: float, a_min_lead: float,
                  t_react_follow: float) -> float:
    """
    Calculates safe distance analytically

    :param v_follow: velocity of following vehicle
    :param v_lead: velocity of leading vehicle
    :param a_min_follow: minimum acceleration of following vehicle
    :param a_min_lead: minimum acceleration of leading vehicle
    :param t_react_follow: reaction time of following vehicle
    :returns boolean indicating satisfaction
    """

    assert a_min_follow and 0 > a_min_lead, \
        '<BrakingPredicateCollection/safe_distance>: acceleration is not valid'
    d_safe = (
            (v_lead ** 2) / (-2. * np.abs(a_min_lead))
            - (v_follow ** 2) / (-2. * np.abs(a_min_follow))
            + v_follow * t_react_follow
    )
    return d_safe


def safe_dis_derivative(v_ego, a_ego, t_react):
    return v_ego/np.abs(a_ego) + t_react


v_follow = 16.
v_lead = 10.
a_min_follow = -10.
a_min_lead = -10.5
t_react = 0.4
veh_param = parameters_vehicle2()

print(safe_distance(v_follow, v_lead, a_min_follow, a_min_lead, t_react))

velocity = np.linspace(0, veh_param.longitudinal.v_max, 10)

target_s = 200
safe_distance_correct = target_s - safe_distance(velocity, v_lead, a_min_follow, a_min_lead, t_react)

fig = plt.figure(figsize=(10, 5))
# Create the plot
plt.plot(velocity, safe_distance_correct)

for i in range(len(velocity)):
    safe_dis_local = safe_distance(velocity[i], v_lead, a_min_follow, a_min_lead, t_react)
    safe_dis_deriv = safe_dis_derivative(velocity[i], a_min_follow, t_react)
    approxi = target_s - (safe_dis_local + safe_dis_deriv * (velocity-velocity[i]))
    plt.plot(velocity, approxi)

# Show the plot
plt.show()