"""
Agent
"""

from dataclasses import dataclass
from enum import Enum
from typing import Self, Tuple

class Agent:
	"""
	Generic agent
	"""

	def satisfaction(self, neighbors: list[Self]) -> float:
		"""
		Returns the satisfaction of this agent with relation to it's neighbors
		"""
		raise NotImplementedError

	def threshold(self) -> float:
		"""
		Returns the satisfaction threshold for this agent
		"""
		raise NotImplementedError

	def color(self) -> Tuple[int, int, int]:
		"""
		Returns the color of this agent
		"""
		raise NotImplementedError


class NAgentKind(Enum):
	"""
	Type of each normal agent
	"""

	RED = 1
	BLUE = 2

@dataclass(unsafe_hash=True)
class NAgent(Agent):
	"""
	A normal agent within the graph.
	"""

	# Agent kind
	kind: NAgentKind

	def satisfaction(self, neighbors: list[Self]) -> float:
		# If there are none, we are satisfied
		# Note: The simulation in `http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/` works like
		#       this: When setting `empty = 90%`, some agents are empty but still count as satisfied.
		if len(neighbors) == 0:
			return 1.0

		# Else our satisfaction is the number of similar agents within our neighbors
		return sum(self.kind == neighbor.kind for neighbor in neighbors) / len(neighbors)

	def threshold(self) -> float:
		return 0.5

	def color(self) -> Tuple[int, int, int]:
		match self.kind:
			case NAgentKind.RED:  return (1.0, 0.0, 0.0)
			case NAgentKind.BLUE: return (0.0, 0.0, 1.0)

@dataclass(unsafe_hash=True)
class GAgent(Agent):
	"""
	A gradient agent within the graph.
	"""

	# All agent kinds
	inner: float

	def satisfaction(self, neighbors: list[Self]) -> float:
		# If there are none, we are satisfied
		# Note: The simulation in `http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/` works like
		#       this: When setting `empty = 90%`, some agents are empty but still count as satisfied.
		if len(neighbors) == 0:
			return 1.0

		# Else our satisfaction is the number of similar agents within our neighbors
		return 1.0 - sum( abs(self.inner - neighbor.inner)**0.5 for neighbor in neighbors) / len(neighbors)

	def threshold(self) -> float:
		return 0.5

	def color(self) -> Tuple[int, int, int]:
		lhs = (1.0, 0.0, 0.0)
		rhs = (0.0, 1.0, 0.0)

		r = lhs[0] + (rhs[0] - lhs[0]) * self.inner
		g = lhs[1] + (rhs[1] - lhs[1]) * self.inner
		b = lhs[2] + (rhs[2] - lhs[2]) * self.inner

		return (r, g, b)
