import math
import os
from math import floor

from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common import utils
from minimum_violation_velocity_planner.common.node import Node
from minimum_violation_velocity_planner.common.priority_queue import PriorityQueue, PriorityQueueLex
from minimum_violation_velocity_planner.common.utils import getOrientationAtStation, isLexicographicallyBetter
from minimum_violation_velocity_planner.common.vehicle_model import VehicleModel
from minimum_violation_velocity_planner.rulebook.rulebook import Rulebook
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import State, Trajectory

from typing import List, Dict, Tuple, Union
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState

from matplotlib import pyplot as plt
import numpy as np

import logging


class VelocityPlanner:
    """
    """

    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, ccosy: CurvilinearCoordinateSystem,
                 rulebook: Rulebook, grid: tuple, vehicle_parameters: VehicleParameters, params: Dict):
        """
        """
        self._scenario = scenario
        self._planning_problem = planning_problem
        self._ccosy = ccosy
        self._rulebook = rulebook

        self.params = params

        self._vehicle_model = VehicleModel(vehicle_parameters, params)

        # Dictionary of nodes
        self._nodes = {}

        # Grid
        self._dt, self._dv, self._ds, self._da, self._time_horizon, self._station_horizon, self._velocity_horizon = grid
        self._actions = self.determineActions(self._da)
        self._n_t = int(self._time_horizon[1] / self._dt) + 1
        self._state_rounding_precision = params.get('general').get('state_rounding_precision')

        # Create initial node
        self.max_violation = 999999.0 # Todo unify with violation class
        self.nof_active_rules = self._rulebook.nofActiveRules()

        self._initial_state = self.getInitialState(planning_problem.initial_state, ccosy, params)
        self._initial_node = Node(self._initial_state.t, self._initial_state.s, self._initial_state.v, state=self._initial_state,
                                  g_violation=[0.0] * self.nof_active_rules,   # These violations will later on be overwritten when the children are evaluated!
                                  h_violation=[0.0] * self.nof_active_rules,
                                  i_violation=[0.0] * self.nof_active_rules,
                                  g_h_violation=[0.0] * self.nof_active_rules,
                                  transfer_values=[0.0] * self.nof_active_rules)
        self._nodes[self._initial_node.id] = self._initial_node


        self.optimal_end_node = None
        self.optimal_nodes = None
        self.optimal_trajectory = None
        self.optimal_violations = None
        self._parent_nodes = []

        self.rule_evaluation_count = 0
        self.nof_active_rules = self._rulebook.nofActiveRules()

    def getInitialState(self, initial_state: State, ccosy: CurvilinearCoordinateSystem, params: Dict) -> LongitudinalState:
        s = ccosy.convert_to_curvilinear_coords(initial_state.position[0], initial_state.position[1])[0]
        v = initial_state.velocity
        a = initial_state.acceleration if hasattr(initial_state, 'acceleration') else 0.0
        t = initial_state.time_step
        return LongitudinalState(s=round(s, self._state_rounding_precision),
                                 v=round(v, self._state_rounding_precision),
                                 a=a,
                                 t=round(t, self._state_rounding_precision))

    def determineActions(self, delta_a: float) -> List[float]:
        actions = []
        a_max = floor(self._vehicle_model.max_acceleration/delta_a) * delta_a
        a = a_max
        while a >= self._vehicle_model.min_acceleration:
            actions.append(a)
            a -= delta_a

        return actions

    def getLowerAndUpperVehicleModelBounds(self):
        lower_bound_s = [self._initial_state.s]
        lower_bound_v = [self._initial_state.v]

        upper_bound_s = [self._initial_state.s]
        upper_bound_v = [self._initial_state.v]

        for time_idx in range(1, self._n_t):
            lower_bound = self._vehicle_model.lowerBound(self._initial_state, time_idx, self._dt)
            lower_bound_s.append(lower_bound[0])
            lower_bound_v.append(lower_bound[1])

            upper_bound = self._vehicle_model.upperBound(self._initial_state, time_idx, self._dt)
            upper_bound_s.append(upper_bound[0])
            upper_bound_v.append(upper_bound[1])

        return lower_bound_s, lower_bound_v, upper_bound_s, upper_bound_v

    def plan(self, planner, verbose=False):
        if planner == 'BFS':
            self.optimal_end_node = self.executeBFS(verbose)
        elif planner == 'DIJKSTRA':
            self.optimal_end_node = self.executeDijkstra(verbose)
        elif planner == 'A_STAR':
            self.optimal_end_node = self.executeAStar(verbose)
        elif planner == 'A_STAR_MIN_RULE_EVAL':
            self.optimal_end_node = self.executeAStarMinRuleEval(verbose)
        else:
            raise ValueError('Planner ' + planner + ' not implemented!')

        self.optimal_nodes = self.backtrace(self.optimal_end_node)
        self.optimal_trajectory = self.getOptimalStates()
        self.optimal_violations = self.getOptimalViolations()

    def executeBFS(self, verbose=False) -> Union[None, Node]:

        if verbose:
            print('--- Starting planning until t=' + str(self._time_horizon[1]) + 's ---\n')

        # BFS
        frontier = PriorityQueue(lambda node1, node2: node1.state.t < node2.state.t)

        frontier.append(self._initial_node)

        explored = set()
        while frontier:
            parent = frontier.pop()
            self._parent_nodes.append(parent)

            parent_time_step = parent.state.t
            if math.isclose(parent_time_step, self._time_horizon[1]):
                explored.add(parent)
                end_nodes = [node for node in explored if node.state.t == self._time_horizon[1]]
                end_nodes = sorted(end_nodes, key=lambda node: node.g_violation)

                if verbose:
                    self.showGraphSearchInfos(len(explored), len(explored) + len(frontier) + 1, self.rule_evaluation_count)
                return end_nodes[0]

            explored.add(parent)

            children = self.getInitialChildren(parent) if math.isclose(parent_time_step, 0.0) else self.getChildren(parent)
            for child, updated_child_state in children:

                if child not in explored and child not in frontier:
                    new_g_violation, new_i_violation, new_h_violation, new_transfer_values = self._rulebook.evaluateRules(parent, child, updated_child_state)
                    self.rule_evaluation_count += self.nof_active_rules
                    child.update(state=updated_child_state, predecessor=parent, g_violation=new_g_violation, i_violation=new_i_violation, h_violation=new_h_violation, transfer_values=new_transfer_values)
                    frontier.append(child)

                elif child in frontier:
                    new_g_violation, new_i_violation, new_h_violation, new_transfer_values = self._rulebook.evaluateRules(parent, child, updated_child_state)
                    self.rule_evaluation_count += self.nof_active_rules

                    # Compare violations
                    if isLexicographicallyBetter(tuple(new_g_violation), tuple(child.g_violation)):
                        frontier.remove(child)
                        child.update(state=updated_child_state, predecessor=parent, g_violation=new_g_violation, i_violation=new_i_violation, h_violation=new_h_violation, transfer_values=new_transfer_values)
                        frontier.append(child)
        return None

    def executeDijkstra(self, verbose=False) -> Union[None, Node]:

        if verbose:
            print('--- Starting planning until t=' + str(self._time_horizon[1]) + 's ---\n')

        # Dijkstra
        frontier = PriorityQueueLex(self.nof_active_rules, [0.0] * self.nof_active_rules, lambda n1: n1.g_violation, self.compareCompleteViolation)

        frontier.append(self._initial_node)

        explored = set()
        while frontier:
            parent = frontier.pop()
            self._parent_nodes.append(parent)

            parent_time_step = parent.state.t
            if math.isclose(parent_time_step, self._time_horizon[1]):
                if verbose:
                    self.showGraphSearchInfos(len(explored), len(explored) + len(frontier) + 1, self.rule_evaluation_count)
                return parent

            explored.add(parent)

            children = self.getInitialChildren(parent) if math.isclose(parent_time_step, 0.0) else self.getChildren(parent)
            for child, updated_child_state in children:

                if child not in explored and child not in frontier:
                    new_g_violation, new_i_violation, new_h_violation, new_transfer_values = self._rulebook.evaluateRules(parent, child, updated_child_state)
                    self.rule_evaluation_count += self.nof_active_rules
                    child.update(state=updated_child_state, predecessor=parent, g_violation=new_g_violation, i_violation=new_i_violation, h_violation=new_h_violation, transfer_values=new_transfer_values)
                    frontier.append(child)

                elif child in frontier:
                    new_g_violation, new_i_violation, new_h_violation, new_transfer_values = self._rulebook.evaluateRules(parent, child, updated_child_state)
                    self.rule_evaluation_count += self.nof_active_rules

                    # Compare violations
                    if isLexicographicallyBetter(tuple(new_g_violation), tuple(child.g_violation)):
                        frontier.remove(child)
                        child.update(state=updated_child_state, predecessor=parent, g_violation=new_g_violation, i_violation=new_i_violation, h_violation=new_h_violation, transfer_values=new_transfer_values)
                        frontier.append(child)
        return None


    def executeAStar(self, verbose=False) -> Union[None, Node]:

        if verbose:
            print('--- Starting planning until t=' + str(self._time_horizon[1]) + 's ---\n')

        # A*
        frontier = PriorityQueueLex(self.nof_active_rules, [0.0] * self.nof_active_rules, lambda n: n.g_h_violation, self.compareCompleteViolation)

        frontier.append(self._initial_node)

        explored = set()
        while frontier:
            parent = frontier.pop()
            self._parent_nodes.append(parent)

            parent_time_step = parent.state.t
            if math.isclose(parent_time_step, self._time_horizon[1]):
                if verbose:
                    self.showGraphSearchInfos(len(explored), len(explored) + len(frontier) + 1, self.rule_evaluation_count)
                return parent

            explored.add(parent)

            children = self.getInitialChildren(parent) if math.isclose(parent_time_step, 0.0) else self.getChildren(parent)
            for child, updated_child_state in children:

                if child not in explored and child not in frontier:
                    new_g_violation, new_i_violation, new_h_violation, new_transfer_values = self._rulebook.evaluateRules(parent, child, updated_child_state)
                    self.rule_evaluation_count += self.nof_active_rules
                    child.update(state=updated_child_state, predecessor=parent, g_violation=new_g_violation, i_violation=new_i_violation, h_violation=new_h_violation, transfer_values=new_transfer_values)
                    frontier.append(child)

                elif child in frontier:
                    new_g_violation, new_i_violation, new_h_violation, new_transfer_values = self._rulebook.evaluateRules(parent, child, updated_child_state)
                    self.rule_evaluation_count += self.nof_active_rules

                    # Compare violations
                    if isLexicographicallyBetter(tuple(new_g_violation), tuple(child.g_violation)):
                        frontier.remove(child)
                        child.update(state=updated_child_state, predecessor=parent, g_violation=new_g_violation, i_violation=new_i_violation, h_violation=new_h_violation, transfer_values=new_transfer_values)
                        frontier.append(child)
        return None

    def executeAStarMinRuleEval(self, verbose=False) -> Union[None, Node]:

        if verbose:
            print('--- Starting planning until t=' + str(self._time_horizon[1]) + 's ---\n')

        # A* with minimum rule evaluation
        frontier = PriorityQueueLex(self.nof_active_rules, [0.0] * self.nof_active_rules, lambda n: n.g_h_violation, self.compareIncrementalViolation, self.evaluateRuleAtLevel)

        frontier.append(self._initial_node)

        explored = set()
        while frontier:
            parent = frontier.pop()
            self._parent_nodes.append(parent)

            parent_time_step = parent.state.t
            if math.isclose(parent_time_step, self._time_horizon[1]):
                if verbose:
                    self.showGraphSearchInfos(len(explored), len(explored) + len(frontier) + 1, self.rule_evaluation_count)
                return parent

            explored.add(parent)

            children = self.getInitialChildren(parent, use_defaults=True) if math.isclose(parent_time_step, 0.0) else self.getChildren(parent, use_defaults=True)
            for child, updated_child_state in children:
                if child not in explored and child not in frontier:
                    child.update(state=updated_child_state, predecessor=parent)
                    frontier.append(child)

                elif child in frontier:
                    is_lower, new_g_violation, new_i_violation, new_h_violation, new_transfer_values = self.compareViolations(parent, child, updated_child_state)

                    if is_lower:
                        frontier.remove(child)
                        child.update(state=updated_child_state, predecessor=parent, g_violation=new_g_violation, i_violation=new_i_violation, h_violation=new_h_violation, transfer_values=new_transfer_values)
                        frontier.append(child)
        return None

    def getInitialChildren(self, parent: Node, use_defaults=False) -> List[Tuple[Node, LongitudinalState]]:
        #TODO This can be done more efficiently with using reachability

        dt = self._dt
        ds = self._ds
        dv = self._dv

        children = []
        station = 0.0

        while station <= (self._station_horizon[1] - parent.state.s):
            velocity = 0.0
            while velocity <= self._velocity_horizon[1]:
                a0 = -(2*(2*dt*parent.state.v - 3*station + dt*velocity))/dt**2
                a1 = (2*(dt*parent.state.v - 3*station + 2*dt*velocity))/dt**2

                if self._vehicle_model.min_acceleration < a0 < self._vehicle_model.max_acceleration and self._vehicle_model.min_acceleration < a1 < self._vehicle_model.max_acceleration:
                    j = (6*(2*dt*parent.state.v - 2*station + dt*velocity))/dt**3

                    new_state = LongitudinalState(t=round(dt, self._state_rounding_precision),
                                                  s=round(parent.state.s + station, self._state_rounding_precision),
                                                  v=round(velocity, self._state_rounding_precision),
                                                  a=a1, j=j)

                    node_key = (new_state.t, new_state.s, new_state.v)
                    if use_defaults:
                        new_child_node = Node(new_state.t, new_state.s, new_state.v,
                                              g_violation=[self.max_violation]*self.nof_active_rules,
                                              h_violation=[self.max_violation]*self.nof_active_rules,
                                              i_violation=[self.max_violation]*self.nof_active_rules,
                                              g_h_violation=[self.max_violation]*self.nof_active_rules,
                                              transfer_values=[0.0]*self.nof_active_rules)
                    else:
                        new_child_node = Node(new_state.t, new_state.s, new_state.v)
                    self._nodes[node_key] = new_child_node
                    children.append((new_child_node, new_state))

                velocity += dv
            station += ds
        return children

    def getChildren(self, parent: Node, use_defaults=False) -> List[Tuple[Node, LongitudinalState]]:
        children = []
        for action in self._actions:
            child, updated_child_state = self.getChild(parent, action, use_defaults)
            if child:
                children.append((child, updated_child_state))
        return children

    def getChild(self, parent: Node, action: float, use_defaults=False) -> Tuple[Union[None, Node], Union[None, LongitudinalState]]:
        child_state = self._vehicle_model.propagate(parent.state, action, self._dt, self._ds, self._dv, self._initial_state.s)

        # If child is not in allowed station and velocity range, return None
        if not (self._initial_state.s <= child_state.s <= self._station_horizon[1]) \
                or not(self._velocity_horizon[0] <= child_state.v <= self._velocity_horizon[1]):
            return None, None

        # If node does not yet exist
        node_key = (child_state.t, child_state.s, child_state.v)
        node = self._nodes.get(node_key)
        if node is None:
            if use_defaults:
                new_child_node = Node(child_state.t, child_state.s, child_state.v,
                                      g_violation=[self.max_violation] * self.nof_active_rules,
                                      h_violation=[self.max_violation] * self.nof_active_rules,
                                      i_violation=[self.max_violation] * self.nof_active_rules,
                                      g_h_violation=[self.max_violation] * self.nof_active_rules,
                                      transfer_values=[0.0] * self.nof_active_rules)
            else:
                new_child_node = Node(child_state.t, child_state.s, child_state.v)
            self._nodes[node_key] = new_child_node
            return new_child_node, child_state
        else:
            return node, child_state

    def compareViolations(self, parent: Node, child: Node, updated_child_state: LongitudinalState) -> Tuple[bool, Union[None, List], Union[None, List], Union[None, List], Union[None, List]]:

        new_g_violation = [self.max_violation] * self.nof_active_rules
        new_i_violation = [self.max_violation] * self.nof_active_rules
        new_h_violation = [self.max_violation] * self.nof_active_rules
        new_transfer_values = [0.0] * self.nof_active_rules

        # Loop over all rule levels
        for rule_level in range(self.nof_active_rules):
            # Evaluate child violation at this level
            self.evaluateRuleAtLevel(child, rule_level)

            # Evaluate parent violation at this level and violation for updated state
            self.evaluateRuleAtLevel(parent, rule_level)
            new_g_violation_score, new_i_violation_score, new_h_violation_score, new_transfer_value = self._rulebook.evaluateRuleAtLevel(parent, child, updated_child_state, rule_level)

            new_g_violation[rule_level] = new_g_violation_score
            new_i_violation[rule_level] = new_i_violation_score
            new_h_violation[rule_level] = new_h_violation_score
            new_transfer_values[rule_level] = new_transfer_value

            self.rule_evaluation_count += 1

            child_g_violation_score = child.g_violation[rule_level]
            if math.isclose(new_g_violation_score, child_g_violation_score, abs_tol=1e-9):
                continue
            elif new_g_violation_score < child_g_violation_score:
                return True, new_g_violation, new_i_violation, new_h_violation, new_transfer_values

            else:
                return False, None, None, None, None

        # The violations are equal
        return False, None, None, None, None

    def compareCompleteViolation(self, violation: List, node: Node, violation_fct) -> int:
        len_scores = len(violation)
        for i in range(len_scores):
            violation_score = violation[i]
            node_score = violation_fct(node)[i]
            if math.isclose(violation_score, node_score, abs_tol=1e-9):
                continue
            elif violation_score < node_score:
                return i
            else:
                raise ValueError("Other is not equal or worse than Self")
        return len_scores - 1

    def compareIncrementalViolation(self, violation: List, node: Node, violation_fct) -> int:
        len_scores = len(violation)
        # Loop over all rule levels
        for i in range(len_scores):
            self.evaluateRuleAtLevel(node, i)

            violation_score = violation[i]
            node_score = violation_fct(node)[i]

            if math.isclose(violation_score, node_score, abs_tol=1e-9):
                continue
            elif violation_score < node_score:
                return i
            else:
                raise("Other is not equal or worse than Self")
        return len_scores - 1

    def evaluateRuleAtLevel(self, node: Node, rule_level: int):
        # If rule on this level was already evaluated
        if node.i_violation[rule_level] != self.max_violation:
            return
        else:
            nodes = self.backtraceUntilRuleLevelAvailable(node, rule_level)
            for node in nodes[1:]:
                new_g_violation_score, new_i_violation_score, new_h_violation_score, new_transfer_value = self._rulebook.evaluateRuleAtLevel(node.predecessor, node, node.state, rule_level)
                node.updateViolationAtLevel(rule_level, g_violation_score=new_g_violation_score, i_violation_score=new_i_violation_score, h_violation_score=new_h_violation_score, transfer_value=new_transfer_value)

                self.rule_evaluation_count += 1
            return

    def backtraceUntilRuleLevelAvailable(self, end_node: Node, rule_level: int) -> List[Node]:
        # Go backwards from end node
        node = end_node
        nodes = []
        while node is not None and node.i_violation[rule_level] == self.max_violation:
            nodes.insert(0, node)
            node = node.predecessor

        # First node shall be the node, where the rule_level shows up the first time
        if node is not None:
            nodes.insert(0, node)

        return nodes

    def backtrace(self, end_node: Node) -> List[Node]:
        # Go backwards from end node and find least violating nodes
        node = end_node
        nodes = []
        while node is not None:
            nodes.insert(0, node)
            node = node.predecessor
        return nodes

    def getOptimalStates(self) -> List[LongitudinalState]:
        states = []
        for node in self.optimal_nodes:
            states.append(node.state)

        # Calculate the acceleration of first state, since we use a different vehicle model for first time step
        states[0].a = -(2*(2*self._dt*states[0].v - 3*(states[1].s - states[0].s) + self._dt*states[1].v))/self._dt**2
        return states

    def getOptimalViolations(self) -> List[List]:
        states = []
        for node in self.optimal_nodes:
            states.append(node.g_violation)
        return states

    def saveOptimalViolation(self, output_dir):
        output_dir + '/planner_output.png'
        f = open(output_dir + '/optimal_violation.txt', 'w')

        dash = '-' * 51
        f.write('{:<40s}{:>4s}\n'.format('Key', 'Violation'))
        f.write(dash + '\n')

        active_rules = self._rulebook.getActiveRules()
        for i in range(len(active_rules)):
            f.write('{:<40s}{:>10.2f}\n'.format(active_rules[i].tag, self.optimal_end_node.g_violation[i]))
        f.write(dash + '\n')

        f.close()

    def printViolationOfOptimalTrajectory(self, show_all_time_steps=False):
        if show_all_time_steps:
            for node in self.optimal_nodes:
                print('t: ', node.state.t)
                self.printViolation(node.g_violation)
                print('')
        else:
            self.printViolation(self.optimal_end_node.g_violation)

    def printViolation(self, violation: List):
        dash = '-' * 51

        print(dash)
        print('{:<40s}{:>4s}'.format('Key', 'Violation'))
        print(dash)

        active_rules = self._rulebook.getActiveRules()
        for i in range(len(active_rules)):
            print('{:<40s}{:>10.2f}'.format(active_rules[i].tag, violation[i]))
        print(dash, '\n')

    def getTagsAndViolationsOfOptimalTrajectory(self, use_sampled_trajectory=False) -> List[Dict]:
        active_rules = self._rulebook.getActiveRules()
        nof_sampling_intervals = int(self._dt / self._scenario.dt) if use_sampled_trajectory else 1
        time_steps = []
        for node in self.optimal_nodes:
            for i in range(nof_sampling_intervals):
                tags_and_violations = {}
                for ruled_idx in range(len(active_rules)):
                    tags_and_violations[active_rules[ruled_idx].tag] = node.g_violation[ruled_idx]
                time_steps.append(tags_and_violations)
        if nof_sampling_intervals > 1:
            del time_steps[-nof_sampling_intervals+1:]
        return time_steps

    def resampleTrajectory(self, trajectory: List[LongitudinalState], interpolate_linear=False) -> List[LongitudinalState]:
        nof_sampling_intervals = int(self._dt / self._scenario.dt)

        vm = self._vehicle_model

        sampled_trajectory = [trajectory[0]]

        # For first timestep:
        for k in range(nof_sampling_intervals):
            if interpolate_linear:
                sampled_trajectory.append(vm.interpolate(trajectory[0], (trajectory[1].v - trajectory[0].v)/self._dt, (k + 1) * self._scenario.dt))
            else:
                sampled_trajectory.append(vm.interpolateJerkModel(trajectory[0], (trajectory[1].a - trajectory[0].a) / self._dt, (k + 1) * self._scenario.dt))
        # For all remaning time_steps
        for i in range(2, len(trajectory)):
            for k in range(nof_sampling_intervals):
                sampled_trajectory.append(vm.interpolate(trajectory[i - 1], trajectory[i].a, (k + 1) * self._scenario.dt))

        return sampled_trajectory

    def getCREgoObject(self, interpolate_linear=False) -> DynamicObstacle:
        # Resample trajectory if necessary
        planner_dt = self._dt
        scenario_dt = self._scenario.dt
        trajectory = self.optimal_trajectory
        if planner_dt != scenario_dt:
            trajectory = self.resampleTrajectory(trajectory, interpolate_linear=interpolate_linear)
            logging.warning(
                "Warning: Output trajectory time-step-size is sampled up from {}s to {}s to have same time-step-size as scenario!".format(
                    planner_dt, scenario_dt))

        wheel_base = self._vehicle_model.veh_params.a + self._vehicle_model.veh_params.b

        # Generate state list based on trajectory
        state_list = []
        for i, state in enumerate(trajectory):
            # add new state to state_list
            position = self._ccosy.convert_to_cartesian_coords(state.s, 0.0)
            orientation = getOrientationAtStation(self._ccosy, state.s)

            # TODO Make nicer
            if i < len(trajectory)-1:
                acceleration = trajectory[i+1].a
                steering_angle = math.atan2((getOrientationAtStation(self._ccosy, trajectory[i+1].s) - orientation) * wheel_base, state.v)
            else:
                steering_angle = 0
                acceleration = 0
            state_list.append(State(position=np.array([position[0], position[1]]),
                                    orientation=orientation,
                                    velocity=state.v,
                                    acceleration=acceleration,
                                    steering_angle=steering_angle,
                                    time_step=i))

        cr_trajectory = Trajectory(initial_time_step=state_list[0].time_step, state_list=state_list)

        # create the ego vehicle prediction using the trajectory and the shape of the obstacle
        dynamic_obstacle_initial_state = cr_trajectory.state_list[0]
        dynamic_obstacle_shape = Rectangle(length=self._vehicle_model.veh_params.l, width=self._vehicle_model.veh_params.w)
        dynamic_obstacle_prediction = TrajectoryPrediction(cr_trajectory, dynamic_obstacle_shape)

        # generate the dynamic obstacle according to the specification
        dynamic_obstacle_id = self._scenario.generate_object_id()
        dynamic_obstacle_type = ObstacleType.CAR
        dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                           dynamic_obstacle_type,
                                           dynamic_obstacle_shape,
                                           dynamic_obstacle_initial_state,
                                           dynamic_obstacle_prediction)

        return dynamic_obstacle

    def showGraphSearchInfos(self, nof_explored_nodes: int, nof_created_nodes: int, nof_evaluated_rules: int):
        print_info = [
            ['Nof explored nodes', nof_explored_nodes, '[-]'],
            ['Nof created nodes', nof_created_nodes, '[-]'],
            ['Total nof evaluated rules', nof_evaluated_rules, '[-]']
        ]
        for row in print_info:
            print("{: <25} {: >8} {: <2}".format(*row))
        print()

    def visualize(self, output_dir: str, show=False, save_fig=False):
        t, s, v, a, j = [], [], [], [], []
        for state in self.optimal_trajectory:
            t.append(state.t)
            s.append(state.s)
            v.append(state.v)
            a.append(state.a)
            j.append(state.j)

        trajectory_sampled = self.resampleTrajectory(self.optimal_trajectory)
        t_sampled, s_sampled, v_sampled, a_sampled, j_sampled = [], [], [], [], []
        for state in trajectory_sampled:
            t_sampled.append(state.t)
            s_sampled.append(state.s)
            v_sampled.append(state.v)
            a_sampled.append(state.a)
            j_sampled.append(state.j)

        lower_bound_s, lower_bound_v, upper_bound_s, upper_bound_v = self.getLowerAndUpperVehicleModelBounds()

        fig, axs = plt.subplots(3, 1, figsize=(10, 13))

        # ----- s-t plot -----
        ax = axs[0]

        ax.fill_between(t, lower_bound_s, upper_bound_s, color='orange', alpha=.1, edgecolor=None)

        ax.plot(t, s, 'o', color='orange', linewidth=2, zorder=500)
        ax.plot(t_sampled, s_sampled, color='orange',  zorder=500)
        ax.plot(t_sampled, s_sampled, 'x', color='dimgray',  zorder=501)

        ax.set_xlim([0, self._time_horizon[1]])
        ax.set_ylim([self._initial_state.s - 5.0, upper_bound_s[-1] + 5.0])

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        ax.set_xlabel('time in s')
        ax.set_ylabel('s in m')

        # ----- v-t plot -----
        ax = axs[1]

        ax.fill_between(t, lower_bound_v, upper_bound_v, color='orange', alpha=.1, edgecolor=None)

        ax.plot(t, v, 'o', color='orange', linewidth=2)
        ax.plot(t_sampled, v_sampled, color='orange')
        ax.plot(t_sampled, v_sampled, 'x', color='dimgray')

        ax.set_xlim([0, self._time_horizon[1]])
        ax.set_ylim([0, self._vehicle_model.max_velocity])

        ax.grid()

        ax.set_xlabel('time in s')
        ax.set_ylabel('v in m/s')

        # ----- a-t plot -----
        ax = axs[2]

        ax.plot(t[0:-1], a[0:-1], 'ro')
        ax.plot(t[1:-1], a[2:], 'ro')
        ax.hlines(a[2:], t[1:-1], t[2:], 'r', linewidth=2)
        ax.plot(t[:2], a[0:2], 'r-', linewidth=2)

        #ax.plot(t_sampled[0:-1], a_sampled[0:-1], 'x', color='dimgray')
        #ax.plot(t_sampled[1:], a_sampled[1:], 'x', color='dimgray')

        ax.set_xlim([0, self._time_horizon[1]])
        ax.set_ylim([self._vehicle_model.min_acceleration, self._vehicle_model.max_acceleration])

        ax.grid()

        ax.set_xlabel('time in s')
        ax.set_ylabel('a in m/s^2')

        # -----
        fig.tight_layout()

        if save_fig:
            plt.savefig(output_dir + '/planner_output.png')

        if show:
            plt.show()

    def visualizeProgress(self, output_dir: str, images_per_second: int = 10):

        print("Visualizing the progress of the planner, this may take some time.")

        # Make folder for progress images
        output_dir_progress_images = os.path.abspath(output_dir + "/progress")
        os.makedirs(output_dir_progress_images)


        t_values = []
        for state in self.optimal_trajectory:
            t_values.append(state.t)

        lower_bound_s, lower_bound_v, upper_bound_s, upper_bound_v = self.getLowerAndUpperVehicleModelBounds()

        fig, axs = plt.subplots(2, 1, figsize=(10, 13))
        #fig.tight_layout()

        # ----- s-t plot -----
        ax_st = axs[0]
        ax_st.fill_between(t_values, lower_bound_s, upper_bound_s, color='orange', alpha=.1, edgecolor=None)

        ax_st.set_xlim([0, self._time_horizon[1]])
        ax_st.set_ylim([self._initial_state.s -5.0, upper_bound_s[-1] + 5.0])

        ax_st.grid(which='minor', alpha=0.2)
        ax_st.grid(which='major', alpha=0.5)

        ax_st.set_xlabel('time in s')
        ax_st.set_ylabel('s in m')


        # ----- v-t plot -----
        ax_vt = axs[1]
        ax_vt.fill_between(t_values, lower_bound_v, upper_bound_v, color='orange', alpha=.1, edgecolor=None)
        ax_vt.set_xlim([0, self._time_horizon[1]])
        ax_vt.set_ylim([0, self._vehicle_model.max_velocity])

        ax_vt.grid()

        ax_vt.set_xlabel('time in s')
        ax_vt.set_ylabel('v in m/s')


        # ---------------
        t, s, v = [], [], []
        for node in self._parent_nodes:
            t.append(node.state.t)
            s.append(node.state.s)
            v.append(node.state.v)

        t_pred, s_pred, v_pred = [], [], []
        for i in range(1, len(self._parent_nodes)):
            predecessor_node = self._parent_nodes[i].predecessor
            t_pred.append(predecessor_node.state.t)
            s_pred.append(predecessor_node.state.s)
            v_pred.append(predecessor_node.state.v)

        # Loop over all parent states in order of appearance
        for i in range(len(self._parent_nodes)):
            utils.printProgressBar(i, len(self._parent_nodes), 'Progress')

            # Edges
            current_edge_s, current_edge_v = None, None
            if i > 0:
                ax_st.plot([t_pred[i-1], t[i]], [s_pred[i-1], s[i]], '-', color='tab:blue', alpha=0.3)
                ax_vt.plot([t_pred[i-1], t[i]], [v_pred[i-1], v[i]], '-', color='tab:blue', alpha=0.3)
                current_edge_s, = ax_st.plot([t_pred[i-1], t[i]], [s_pred[i-1], s[i]], '-r')
                current_edge_v, = ax_vt.plot([t_pred[i-1], t[i]], [v_pred[i-1], v[i]], '-r')

            # Nodes
            ax_st.plot(t[i], s[i], 'o', color='tab:blue', markeredgecolor='none', alpha=0.3)
            ax_vt.plot(t[i], v[i], 'o', color='tab:blue', markeredgecolor='none', alpha=0.3)

            #colors = plt.get_cmap('viridis_r')(np.linspace(0, 1, 20))
            #nof_not_evaluated_rules = sum(i == self.max_violation for i in self._parent_nodes[i].g_violation.scores)
            #ax_st.plot(t[i], s[i], 'o', color=colors[nof_not_evaluated_rules], markeredgecolor='none', alpha=1.0)
            #ax_vt.plot(t[i], v[i], 'o', color=colors[nof_not_evaluated_rules], markeredgecolor='none', alpha=1.0)


            current_state_s, = ax_st.plot(t[i], s[i], 'or')
            current_state_v, = ax_vt.plot(t[i], v[i], 'or')

            # Save figure
            plt.savefig(output_dir_progress_images + '/progress_' + "{:06d}".format(i) + '.png')

            #Remove current nodes and edges from plot
            current_state_s.remove()
            current_state_v.remove()
            if current_edge_s:
                current_edge_s.remove()
                current_edge_v.remove()

        # Plot optimal trajectory
        for node in self.optimal_nodes:
            ax_st.plot(node.state.t, node.state.s, 'or')
            ax_vt.plot(node.state.t, node.state.v, 'or')

            if node.predecessor:
                ax_st.plot([node.predecessor.state.t, node.state.t], [node.predecessor.state.s, node.state.s], '-r',linewidth=3)
                ax_vt.plot([node.predecessor.state.t, node.state.t], [node.predecessor.state.v, node.state.v], '-r',linewidth=3)

        plt.savefig(output_dir_progress_images + '/progress_' + "{:06d}".format(len(self._parent_nodes)) + '.png')


        # Create Video
        video_length = 10 # [s]
        frame_rate = int(len(self._parent_nodes)/video_length)
        os.system("ffmpeg -y -r " + str(frame_rate) + " -f image2 -i " + output_dir_progress_images + "/progress_%06d.png " + output_dir + "/progress.mp4")
