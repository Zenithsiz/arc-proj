"""
Tile
"""

from dataclasses import dataclass
from typing import Tuple

from arc_proj.race import Race


@dataclass
class Tile():
	# Value, either a race or an empty tile
	inner: Race | None

	@staticmethod
	def empty():
		"""
		Returns an empty tile
		"""
		return Tile(None)

	@staticmethod
	def filled(race: Race):
		"""
		Returns a filled tile
		"""
		return Tile(race)


	def color(self) -> Tuple[int, int, int]:
		"""
		Returns the color of this tile
		"""
		if self.inner is None:
			return [0.5, 0.5, 0.5]
		elif self.inner == Race.RED:
			return [1.0, 0.0, 0.0]
		elif self.inner == Race.BLUE:
			return [0.0, 0.0, 1.0]
