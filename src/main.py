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

# Display method
class DisplayMethod(Enum):
	# Displays as a general graph, with edges
	GRAPH = 1

	# Displays as a compact grid, without any spacing between nodes
	GRID = 2

display_method = DisplayMethod.GRAPH

# Visualize graph
with plt.ion():
	# Create the figure
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	# Draw it using the display method
	if display_method == DisplayMethod.GRAPH:
		# Setup the positions and colors
		node_pos = { tile_pos: tile_pos for tile_pos in graph.nodes()}
		node_colors = [tile['tile'].color() for _, tile in graph.nodes(data=True)]

		# And finally draw the nodes and edges
		nx.draw_networkx_nodes(graph, pos=node_pos, ax=ax, node_color=node_colors, node_size=25, node_shape='s')
		nx.draw_networkx_edges(graph, pos=node_pos, ax=ax)

	elif display_method == DisplayMethod.GRID:
		# Disable axis
		ax.axis("off")

		# Then calculate the image
		img = [[0 for _ in range(50)] for _ in range(50)]
		for (tile_pos_x, tile_pos_y), tile in graph.nodes(data = True):
			img[tile_pos_y][tile_pos_x] = tile['tile'].color()

		# And finally show it
		ax.imshow(img)

	else:
		raise ValueError("Unknown display method")

	# Finally show it
	fig.tight_layout()
	plt.show(block = True)
