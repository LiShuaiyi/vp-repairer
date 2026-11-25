class LongitudinalState:
    """
    LongitudinalState
    """

    def __init__(self, s: float = .0, v: float = .0, a: float = .0, j: float = .0, t: float = .0):
        """
        :param s: station
        :param v: velocity
        :param a: acceleration
        :param j: jerk
        :param t: time
        """

        self.s = s
        self.v = v
        self.a = a
        self.j = j
        self.t = t