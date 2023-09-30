"""
ARC Project
"""

# Imports
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import networkx as nx
import matplotlib.pyplot as plt

class Race(Enum):
	RED = 1
	BLUE = 2

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

# Graph (initialized to empty)
graph = nx.grid_2d_graph(50, 50)
nx.set_node_attributes(graph, Tile.empty(), 'tile')

# Set some initial tiles
graph.nodes[(0, 0)]['tile'] = Tile.filled(Race.BLUE)
graph.nodes[(1, 0)]['tile'] = Tile.filled(Race.RED)

# Visualize graph
with plt.ion():
	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(1, 1, 1)

	# Create the image to display
	img = [[0 for _ in range(50)] for _ in range(50)]
	for (tile_pos_x, tile_pos_y), tile in graph.nodes(data = True):
		img[tile_pos_y][tile_pos_x] = tile['tile'].color()

	# Then draw it without any axis
	ax.imshow(img)
	ax.axis("off")

	# And show it (until the user closes it)
	fig.tight_layout()
	plt.show(block = True)
