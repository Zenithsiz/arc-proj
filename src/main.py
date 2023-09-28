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
			case None: return "white"
			case Race.RED: return "red"
			case Race.BLUE: return "blue"

# Graph
graph = nx.grid_2d_graph(10, 10)

nx.set_node_attributes(graph, Tile.empty(), 'tile')
nx.set_node_attributes(graph, { (0, 0): Tile.filled(Race.BLUE) }, 'tile')
nx.set_node_attributes(graph, { (0, 1): Tile.filled(Race.RED) }, 'tile')

# Visualize graph
pos = {(x,y):(y,-x) for x,y in graph.nodes()}
with plt.ion():
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	node_colors = [tile['tile'].color() for tile_pos, tile in graph.nodes(data=True)]
	nx.draw(graph, pos=pos, ax=ax, node_color=node_colors)

	fig.tight_layout()
	plt.show(block = True)
