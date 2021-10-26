import numpy as np
from typing import Union

# commonroad-io
from commonroad.common.validity import is_real_number
from commonroad.scenario.lanelet import LaneletNetwork

# commonroad-collision-checker
import commonroad_dc.pycrcc as pycrcc

# commonroad-curvilinear-coordinate-system
import commonroad_dc.pycrccosy as pycrccosy

VEHICLE_ID = int
TIME_IDX = int

class RepairingConfigurationVehicle:
    """ Class which holds all necessary vehicle-specific parameters for the
    trajectory repairing."""

    def __init__(self):
        self.vehicle_id = -1
        self.min_speed_x = -np.inf
        self.max_speed_x = +np.inf
        self.min_speed_y = -np.inf
        self.max_speed_y = +np.inf
        self.a_max_x = 9.81
        self.a_min_x = -9.81
        self.a_max_y = 9.81
        self.a_min_y = -9.81
        self.a_max = 9.81
        self.j_min_x = -10.0 ** 4
        self.j_max_x = 10.0 ** 4
        self.j_min_y = -10.0 ** 4
        self.j_max_y = 10.0 ** 4
        self.desired_speed = 0.0
        self.max_radius = 0.9
        self.initial_time_idx = 0
        self._bounding_boxes = None
        self.radius = 1.3
        self.wheelbase = 3.0
        self.length = 4.5
        self.width = 2.0
        self._initial_state = None
        self._initial_discrete_state = -1
        self._curvilinear_coordinate_system = None
        self.collision_checker_curvilinear = pycrcc.CollisionChecker()
        self.collision_checker_world = pycrcc.CollisionChecker()
        self._reference_path = None
        self._lanelet_network = None
        self.coordinates = 'curvilinear'
        self.rasterized_obstacles = False
        self.collision_checks_in_curvilinear_cosy = False
        self._initial_lanelet_id = None

    # def convert_to_pycrreach_vehicle_parameters(self) -> pycrreach.VehicleParameters:
    #     params = pycrreach.VehicleParameters()
    #     params.id = self._vehicle_id
    #     params.v_min_x = self._min_speed_x
    #     params.v_max_x = self._max_speed_x
    #     params.v_min_y = self._min_speed_y
    #     params.v_max_y = self._max_speed_y
    #     params.a_min_x = self._a_min_x
    #     params.a_max_x = self._a_max_x
    #     params.a_min_y = self._a_min_y
    #     params.a_max_y = self._a_max_y
    #     params.width = self._width
    #     params.length = self._length
    #     params.radius = self._radius
    #     params.distance_left_right_circle_center = self._wheelbase
    #     params.position_uncertainty_x = self._position_uncertainty_x
    #     params.position_uncertainty_y = self._position_uncertainty_y
    #     params.initial_time_step = self._initial_time_idx
    #     params.grid_x = self._grid_x
    #     params.grid_y = self._grid_y
    #     params.min_radius = self._max_radius
    #     if self._coordinates == 'curvilinear':
    #         params.coordinate_system_type = pycrreach.CoordinateSystemType.CURVILINEAR
    #         params.curvilinear_coordinate_system = self._curvilinear_coordinate_system
    #         params.collision_checker_curvilinear = self._collision_checker_curvilinear
    #     elif self._coordinates == 'cartesian':
    #         params.coordinate_system_type = pycrreach.CoordinateSystemType.CARTESIAN
    #     else:
    #         raise ValueError("<VehicleConfiguration> Unknown coordinate type. Expected types: 'curvilinear',"
    #                          " 'cartesian'. Got type: {}".format(self._coordinates))
    #     params.collision_checker_cartesian = self._collision_checker_world
    #     params.collision_checks_in_curvilinear_cosy = self._collision_checks_in_curvilinear_cosy
    #     params.rasterized_obstacles = self._rasterized_obstacles
    #     params.lut_lon_enlargement = self._lut_lon_enlargement
    #     params.reference_point = self.reference_point
    #     return params

    @property
    def vehicle_id(self) -> VEHICLE_ID:
        """ Unique ID of the vehicle, e.g, the planning problem ID from CommonRoad."""
        return self._vehicle_id

    @vehicle_id.setter
    def vehicle_id(self, vehicle_id: VEHICLE_ID):
        assert (type(vehicle_id) is VEHICLE_ID), '<RepairConfiguration/vehicle_id> Expected type int; ' \
                                                 'Got type %s instead.' % (type(vehicle_id))
        self._vehicle_id = vehicle_id

    @property
    def min_speed_x(self) -> float:
        """ Minimum speed of the vehicle in longitudinal direction."""
        return self._min_speed_x

    @min_speed_x.setter
    def min_speed_x(self, min_speed_x: float):
        assert (type(min_speed_x) is float), '<RepairConfiguration/min_speed_x> Expected type float; ' \
                                             'Got type %s instead.' % (type(min_speed_x))
        self._min_speed_x = min_speed_x

    @property
    def max_speed_x(self) -> float:
        """ Maximum speed of the vehicle in longitudinal direction."""
        return self._max_speed_x

    @max_speed_x.setter
    def max_speed_x(self, max_speed_x: float):
        assert (type(max_speed_x) is float), '<RepairConfiguration/max_speed_x> Expected type float; ' \
                                             'Got type %s instead.' % (type(max_speed_x))
        self._max_speed_x = max_speed_x

    @property
    def min_speed_y(self) -> float:
        """ Minimum speed of the vehicle in lateral direction."""
        return self._min_speed_y

    @min_speed_y.setter
    def min_speed_y(self, min_speed_y: float):
        assert (type(min_speed_y) is float), '<RepairConfiguration/min_speed_y> Expected type float; ' \
                                             'Got type %s instead.' % (type(min_speed_y))
        self._min_speed_y = min_speed_y

    @property
    def max_speed_y(self) -> float:
        """ Maximum speed of the vehicle in lateral direction."""
        return self._max_speed_y

    @max_speed_y.setter
    def max_speed_y(self, max_speed_y: float):
        assert (type(max_speed_y) is float), '<RepairConfiguration/max_speed_y> Expected type float; ' \
                                             'Got type %s instead.' % (type(max_speed_y))
        self._max_speed_y = max_speed_y

    @property
    def a_max_x(self) -> float:
        """ Maximum acceleration of the vehicle in longitudinal direction."""
        return self._a_max_x

    @a_max_x.setter
    def a_max_x(self, a_max_x: float):
        assert (type(a_max_x) is float), '<RepairConfiguration/a_max_x> Expected type float; ' \
                                         'Got type %s instead.' % (type(a_max_x))
        self._a_max_x = a_max_x

    @property
    def a_min_x(self) -> float:
        """ Minimum acceleration of the vehicle in longitudinal direction."""
        return self._a_min_x

    @a_min_x.setter
    def a_min_x(self, a_min_x: float):
        assert (type(a_min_x) is float), '<RepairConfiguration/a_min_x> Expected type float; ' \
                                         'Got type %s instead.' % (type(a_min_x))
        self._a_min_x = a_min_x

    @property
    def a_max_y(self) -> float:
        """ Maximum acceleration of the vehicle in lateral direction."""
        return self._a_max_y

    @a_max_y.setter
    def a_max_y(self, a_max_y: float):
        assert (type(a_max_y) is float), '<RepairConfiguration/a_max_y> Expected type float; ' \
                                         'Got type %s instead.' % (type(a_max_y))
        self._a_max_y = a_max_y

    @property
    def a_min_y(self) -> float:
        """ Minimum acceleration of the vehicle in lateral direction."""
        return self._a_min_y

    @a_min_y.setter
    def a_min_y(self, a_min_y: float):
        assert (type(a_min_y) is float), '<RepairConfiguration/a_min_y> Expected type float; ' \
                                         'Got type %s instead.' % (type(a_min_y))
        self._a_min_y = a_min_y

    @property
    def a_max(self) -> float:
        """ Maximum overall acceleration of the vehicle."""
        return self._a_max

    @a_max.setter
    def a_max(self, a_max: float):
        assert (type(a_max) is float), '<RepairConfiguration/a_max> Expected type float; ' \
                                       'Got type %s instead.' % (type(a_max))
        self._a_max = a_max

    @property
    def j_min_x(self) -> float:
        """ Minimum jerk in longitudinal direction."""
        return self._j_min_x

    @j_min_x.setter
    def j_min_x(self, j_min_x: float):
        assert isinstance(j_min_x, float), '<RepairConfiguration/j_min_x> Expected type float; ' \
                                           'Got type %s instead.' % (type(j_min_x))
        self._j_min_x = j_min_x

    @property
    def j_max_x(self) -> float:
        """ Maximum jerk of the vehicle in longitudinal direction."""
        return self._j_max_x

    @j_max_x.setter
    def j_max_x(self, j_max_x: float):
        assert isinstance(j_max_x, float), '<RepairConfiguration/j_max_x> Expected type float; ' \
                                           'Got type %s instead.' % (type(j_max_x))
        self._j_max_x = j_max_x

    @property
    def j_min_y(self) -> float:
        """ Minimum jerk in lateral direction."""
        return self._j_min_y

    @j_min_y.setter
    def j_min_y(self, j_min_y: float):
        assert isinstance(j_min_y, float), '<RepairConfiguration/j_min_y> Expected type float; ' \
                                           'Got type %s instead.' % (type(j_min_y))
        self._j_min_y = j_min_y

    @property
    def j_max_y(self) -> float:
        """ Maximum jerk in lateral direction."""
        return self._j_max_y

    @j_max_y.setter
    def j_max_y(self, j_max_y: float):
        assert isinstance(j_max_y, float), '<RepairConfiguration/j_max_y> Expected type float; ' \
                                           'Got type %s instead.' % (type(j_max_y))
        self._j_max_y = j_max_y

    @property
    def desired_speed(self) -> float:
        """ Desired speed of the vehicle for trajectory planning."""
        return self._desired_speed

    @desired_speed.setter
    def desired_speed(self, desired_speed: float):
        assert isinstance(desired_speed, float), '<RepairConfiguration/desired_speed> Expected type float; ' \
                                                 'Got type %s instead.' % (type(desired_speed))
        self._desired_speed = desired_speed

    @property
    def grid_x(self) -> float:
        """
        The reachable sets are discretized in the merging and re-partitioning step using a grid with
        longitudinal segment length grid_x.
        """
        return self._grid_x

    @grid_x.setter
    def grid_x(self, grid_x: float):
        assert (type(grid_x) is float), '<RepairConfiguration/grid_x> Expected type float; ' \
                                        'Got type %s instead.' % (type(grid_x))
        self._grid_x = grid_x

    @property
    def grid_y(self) -> float:
        """
        The reachable sets are discretized in the merging and re-partitioning step using a grid with
        lateral segment length grid_y.
        """
        return self._grid_y

    @grid_y.setter
    def grid_y(self, grid_y):
        assert (type(grid_y) is float), '<RepairConfiguration/grid_y> Expected type float; ' \
                                        'Got type %s instead.' % (type(grid_y))
        self._grid_y = grid_y

    @property
    def max_radius(self) -> float:
        """
        Termination criterion for collision-detection during reachability analysis.
        :return: radius until rectangle is splitted during reachability analysis
        """
        return self._max_radius

    @max_radius.setter
    def max_radius(self, max_radius: float):
        assert (type(max_radius) is float), '<RepairConfiguration/max_radius> Expected type float; ' \
                                            'Got type %s instead.' % (type(max_radius))
        self._max_radius = max_radius

    @property
    def radius(self) -> float:
        """ The vehicle’s shape is approximated with three circles with equal radius."""
        return self._radius

    @radius.setter
    def radius(self, radius: float):
        assert isinstance(radius, (float, np.float64)), '<RepairConfiguration/radius> Expected type float; ' \
                                                        'Got type %s instead.' % (type(radius))
        self._radius = radius

    @property
    def wheelbase(self) -> float:
        """
        The vehicle’s shape is approximated with three circles with equal radius. The centers of the first and third
        circle coincides with the rear and front axle, respectively. The distance between the first and third center
        is the wheelbase."""
        return self._wheelbase

    @wheelbase.setter
    def wheelbase(self, wheelbase: float):
        assert (type(wheelbase) is float), '<RepairConfiguration/wheelbase> ' \
                                           'Expected type float; Got type %s instead.' % (type(wheelbase))
        self._wheelbase = wheelbase

    @property
    def length(self) -> float:
        """ Length of the vehicle."""
        return self._length

    @length.setter
    def length(self, length: float):
        assert is_real_number(length), '<RepairConfiguration/length>: argument "length" is not a real number. ' \
                                       'length = {}'.format(length)
        self._length = length

    @property
    def width(self) -> float:
        """ Width of the vehicle."""
        return self._width

    @width.setter
    def width(self, width: float):
        assert is_real_number(width), '<RepairConfiguration/width>: argument "width" is not a real number. ' \
                                      'width = {}'.format(width)
        self._width = width

    @property
    def initial_time_idx(self) -> int:
        """ Initial time step for reachability computation (should be 0)."""
        return self._initial_time_idx

    @initial_time_idx.setter
    def initial_time_idx(self, initial_time_idx: int):
        assert (type(initial_time_idx) is int), '<RepairConfiguration/initial_time_idx> Expected type int; ' \
                                                'Got type %s instead.' % (type(initial_time_idx))
        self._initial_time_idx = initial_time_idx

    # ToDo type for bounding boxes
    @property
    def bounding_boxes(self):
        return self._bounding_boxes

    @bounding_boxes.setter
    def bounding_boxes(self, bounding_boxes):
        self._bounding_boxes = bounding_boxes

    @property
    def initial_state(self) -> np.ndarray:
        """ Initial state for the reachable set computation:
            initial_state[0][0]: longitudinal position
            initial_state[0][1]: longitudinal velocity
            initial_state[1][0]: lateral position
            initial_state[1][1]: lateral velocity
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: tuple):
        assert (type(initial_state) is tuple), '<RepairConfiguration/initial_state> Expected type tuple; ' \
                                               'Got type %s instead.' % (type(initial_state))
        self._initial_state = initial_state

    @property
    def initial_position_x(self) -> float:
        """ Initial longitudinal position."""
        return self._initial_state[0][0]

    @property
    def initial_position_y(self) -> float:
        """ Initial lateral position."""
        return self._initial_state[1][0]

    @property
    def initial_speed_x(self) -> float:
        """ Initial longitudinal velocity."""
        return self._initial_state[0][1]

    @property
    def initial_speed_y(self) -> float:
        """ Initial lateral velocity."""
        return self._initial_state[1][1]

    @property
    def initial_discrete_state(self) -> int:
        """ Initial discrete state for hybrid reachability analysis. """
        return self._initial_discrete_state

    @initial_discrete_state.setter
    def initial_discrete_state(self, discrete_state):
        assert (type(discrete_state) is int), '<RepairConfiguration/initial_discrete_state> Expected type int; ' \
                                              'Got type %s instead.' % (type(discrete_state))
        self._initial_discrete_state = discrete_state

    @property
    def curvilinear_coordinate_system(self) -> pycrccosy.CurvilinearCoordinateSystem:
        """ Curvilinear coordinate system for the reachable set computation in curvilinear coordinates."""
        return self._curvilinear_coordinate_system

    @curvilinear_coordinate_system.setter
    def curvilinear_coordinate_system(
            self, curvilinear_coordinate_system: pycrccosy.CurvilinearCoordinateSystem):
        assert (isinstance(curvilinear_coordinate_system, pycrccosy.CurvilinearCoordinateSystem)), \
            '<RepairConfiguration/curvilinear_coordinate_system> Expected type ' \
            'pycrccosy.PolylineCoordinateSystem; Got type %s instead.' % (type(curvilinear_coordinate_system))
        self._curvilinear_coordinate_system = curvilinear_coordinate_system

    @property
    def collision_checker_curvilinear(self) \
            -> pycrcc.CollisionChecker:
        """ If the reachable set computation is performed in curvilinear coordinates, workspace obstacles can be
        represented with the collision_checker_curvilinear."""
        return self._collision_checker_curvilinear

    @collision_checker_curvilinear.setter
    def collision_checker_curvilinear(
            self,
            collision_checker_curvilinear: pycrcc.CollisionChecker):
        assert (isinstance(collision_checker_curvilinear, pycrcc.CollisionChecker)), \
            '<RepairConfiguration/curvilinear_coordinate_system> Expected type pycrcc.CollisionChecker; ' \
            'Got type %s instead.' % (type(collision_checker_curvilinear))
        self._collision_checker_curvilinear = collision_checker_curvilinear

    @property
    def collision_checker_world(self) -> pycrcc.CollisionChecker:
        """ Collision checker storing all obstacles in a global, Cartesian frame."""
        return self._collision_checker_world

    @collision_checker_world.setter
    def collision_checker_world(
            self, collision_checker_world: pycrcc.CollisionChecker):
        assert (isinstance(collision_checker_world, pycrcc.CollisionChecker)), \
            '<RepairConfiguration/curvilinear_coordinate_system> Expected type pyfvks.collision.CollisionChecker; ' \
            'Got type %s instead.' % (type(collision_checker_world))
        self._collision_checker_world = collision_checker_world

    @property
    def reference_path(self):
        """ Reference path of the vehicle for the generation of a curvilinear coordinate system or trajectory
        planning. The reference path must be given as polyline."""
        return self._reference_path

    @reference_path.setter
    def reference_path(self, reference_path: np.ndarray):
        assert isinstance(reference_path, np.ndarray) and reference_path.ndim == 2 \
               and len(reference_path) > 1 and len(reference_path[0, :]) == 2, \
            '<RepairConfiguration/reference_path>: Provided reference is not valid. reference = {}'. \
                format(reference_path)
        self._reference_path = reference_path

    @property
    def coordinates(self) -> str:
        """ The coordinates for the reachable set computation can be either cartesian or curvilinear."""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: str):
        assert isinstance(coordinates, str), '<RepairConfiguration/coordinates>: argument "coordinates" of wrong ' \
                                             'type. Expected type: %s. Got type: %s.' % (str, type(coordinates))
        self._coordinates = coordinates

    @property
    def lanelet_network(self) -> Union[None, LaneletNetwork]:
        """ The part of the lanelet network of the scenario, the vehicle is allowed or should drive on."""
        return self._lanelet_network

    @lanelet_network.setter
    def lanelet_network(self, lanelet_network: LaneletNetwork):
        assert isinstance(lanelet_network, LaneletNetwork), '<RepairConfiguration/lanelet_network>: argument ' \
                                                            'lanelet_network of wrong type. Expected type: %s. ' \
                                                            'Got type: %s.' % (LaneletNetwork, type(lanelet_network))
        self._lanelet_network = lanelet_network

    # @property
    # def position_uncertainty_x(self) -> float:
    #     return self._position_uncertainty_x
    #
    # @position_uncertainty_x.setter
    # def position_uncertainty_x(self, position_uncertainty_x: float):
    #     assert isinstance(position_uncertainty_x, float), '<RepairConfiguration/position_uncertainty_x>: argument ' \
    #                                                       'position_uncertainty_x of wrong type. Expected type: %s. ' \
    #                                                       'Got type: %s.' % (float, type(position_uncertainty_x))
    #     self._position_uncertainty_x = position_uncertainty_x
    #
    # @property
    # def position_uncertainty_y(self) -> float:
    #     return self._position_uncertainty_y
    #
    # @position_uncertainty_y.setter
    # def position_uncertainty_y(self, position_uncertainty_y: float):
    #     assert isinstance(position_uncertainty_y, float), '<RepairConfiguration/position_uncertainty_y>: argument ' \
    #                                                       'position_uncertainty_y of wrong type. Expected type: %s. ' \
    #                                                       'Got type: %s.' % (float, type(position_uncertainty_y))
    #     self._position_uncertainty_y = position_uncertainty_y
    #
    # @property
    # def rasterized_obstacles(self) -> bool:
    #     return self._rasterized_obstacles

    # @rasterized_obstacles.setter
    # def rasterized_obstacles(self, rasterized_obstacles: bool):
    #     assert isinstance(rasterized_obstacles, bool), '<RepairConfiguration/rasterized_obstacles>: argument ' \
    #                                                    'rasterized_obstacles of wrong type. Expected type: %s. ' \
    #                                                    'Got type: %s.' % (bool, type(rasterized_obstacles))
    #     self._rasterized_obstacles = rasterized_obstacles

    @property
    def collision_checks_in_curvilinear_cosy(self) -> bool:
        return self._collision_checks_in_curvilinear_cosy

    @collision_checks_in_curvilinear_cosy.setter
    def collision_checks_in_curvilinear_cosy(self, collision_checks_in_curvilinear_cosy: bool):
        assert isinstance(collision_checks_in_curvilinear_cosy, bool), \
            '<RepairConfiguration/collision_checks_in_curvilinear_cosy>: argument ' \
            'collision_checks_in_curvilinear_cosy of wrong type. Expected type: %s.' \
            ' Got type: %s.' % (bool, type(collision_checks_in_curvilinear_cosy))
        self._collision_checks_in_curvilinear_cosy = collision_checks_in_curvilinear_cosy

    # @property
    # def lut_lon_enlargement(self) -> pycrreach.LUTLongitudinalEnlargement:
    #     return self._lut_lon_enlargement
    #
    # @lut_lon_enlargement.setter
    # def lut_lon_enlargement(self, lut_lon_enlargement: dict):
    #     assert isinstance(lut_lon_enlargement, dict), \
    #         '<RepairConfiguration/lut_lon_enlargement>: argument ' \
    #         'lut_lon_enlargement of wrong type. Expected type: %s.' \
    #         ' Got type: %s.' % (dict, type(lut_lon_enlargement))
    #     self._lut_lon_enlargement = pycrreach.LUTLongitudinalEnlargement(lut_lon_enlargement)

    # @property
    # def reference_point(self) -> pycrreach.ReferencePoint:
    #     return self._reference_point
    #
    # @reference_point.setter
    # def reference_point(self, reference_point: pycrreach.ReferencePoint):
    #     assert isinstance(reference_point, pycrreach.ReferencePoint), \
    #         '<RepairConfiguration/reference_point>: argument reference_point of wrong type. Expected type: %s. ' \
    #         'Got type: %s' % (pycrreach.ReferencePoint, type(reference_point))
    #     self._reference_point = reference_point

    @property
    def initial_lanelet_id(self) -> int:
        return self._initial_lanelet_id

    @initial_lanelet_id.setter
    def initial_lanelet_id(self, initial_lanelet_id: int):
        assert isinstance(initial_lanelet_id, int), \
            '<RepairConfiguration/setter>: argument initial_lanelet_id of wrong type. Expected type: %s. ' \
            'Got type: %s' % (int, type(initial_lanelet_id))
        self._initial_lanelet_id = initial_lanelet_id

