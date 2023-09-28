"""
ARC Project
"""

# Imports
from dataclasses import dataclass
from enum import Enum
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


	def color(self) -> str:
		"""
		Returns the color of this tile
		"""
		match self.inner:
			case None: return "gray"
			case Race.RED: return "red"
			case Race.BLUE: return "blue"

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

	pos = { tile_pos: tile_pos for tile_pos in graph.nodes()}
	node_colors = [tile['tile'].color() for tile_pos, tile in graph.nodes(data=True)]
	nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors, node_size=25, node_shape='s')

	fig.tight_layout()
	plt.show(block = True)
