"""
Agent
"""

from enum import Enum
from typing import Tuple

class Agent(Enum):
	"""
	An agent within the graph.
	"""

	# All agent kinds
	RED = 1
	BLUE = 2

	def threshold(self) -> float:
		"""
		Returns the satisfaction threshold for this agent
		"""

		match self:
			case Agent.RED:     return 0.5
			case Agent.BLUE:    return 0.35

	def color(self) -> Tuple[int, int, int]:
		"""
		Returns the color of this agent
		"""

		match self:
			case Agent.RED:    return [1.0, 0.0, 0.0]
			case Agent.BLUE:   return [0.0, 0.0, 1.0]
