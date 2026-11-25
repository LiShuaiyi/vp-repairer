from abc import ABC, abstractmethod
from typing import Tuple, Union, List

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class Rule(ABC):
    """
    Rule
    """

    def __init__(self, tag: str, statement: str, is_path_independent: bool = False):
        """
        :param tag: Tag of the rules
        :param statement: Statement of the rules
        """

        self._tag = tag
        self._statement = statement
        self._is_path_independent = is_path_independent

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def statement(self) -> str:
        return self._statement

    @tag.setter
    def tag(self, tag):
        self._tag = tag

    @statement.setter
    def statement(self, statement):
        self._statement = statement

    @abstractmethod
    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float, List]]:
        return 0.0, 0.0, 0.0
