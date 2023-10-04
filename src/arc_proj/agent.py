"""
Agent
"""

from enum import Enum
from typing import Self, Tuple


class Agent(Enum):
	"""
	An agent within the graph.
	"""

	# All agent kinds
	RED = 1
	BLUE = 2

	def satisfaction(self, neighbors: list[Self]) -> float:
		"""
		Returns the satisfaction of this agent with relation to it's neighbors
		"""

		# If there are none, we are satisfied
		# Note: The simulation in `http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/` works like
		#       this: When setting `empty = 90%`, some agents are empty but still count as satisfied.
		if len(neighbors) == 0:
			return 1.0

		# Else our satisfaction is the number of similar agents within our neighbors
		return sum(self == neighbor for neighbor in neighbors) / len(neighbors)

	def threshold(self) -> float:
		"""
		Returns the satisfaction threshold for this agent
		"""

		match self:
			case Agent.RED:     return 0.5
			case Agent.BLUE:    return 0.5

	def color(self) -> Tuple[int, int, int]:
		"""
		Returns the color of this agent
		"""

		match self:
			case Agent.RED:    return [1.0, 0.0, 0.0]
			case Agent.BLUE:   return [0.0, 0.0, 1.0]
