import numpy as np
import utils
from typing import List


class BasicConstraint(object):
    def __init__(self):
        self.var_lat_x_ub = []
        self.var_lat_x_lb = []
        self.var_lat_u_lb = []
        self.var_lat_u_ub = []

        self.var_long_x_ub = []
        self.var_long_x_lb = []
        self.var_long_u_lb = []
        self.var_long_u_ub = []
        self.dynamic_matrix_list = []
        self.init_state = []


class LongitudinalConstraint(BasicConstraint):
    def __init__(self):
        super().__init__()
