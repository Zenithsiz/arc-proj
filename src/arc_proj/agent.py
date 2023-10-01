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

	def color(self) -> Tuple[int, int, int]:
		"""
		Returns the color of this agent
		"""
		if self == Agent.RED:
			return [1.0, 0.0, 0.0]
		elif self == Agent.BLUE:
			return [0.0, 0.0, 1.0]
