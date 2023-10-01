"""
ARC Project
"""

# Imports
from enum import Enum
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx

from arc_proj.race import Race
from arc_proj.tile import Tile
import random

def tile_neighbors_pos(tile_pos: Tuple[int, int], graph_size: Tuple[int, int]) -> list[Tuple[int, int]]:
	"""
	Returns all neighbors positions of `tile_pos`.
	"""

	x = tile_pos[0]
	y = tile_pos[1]
	w = graph_size[0]
	h = graph_size[1]

	neighbors = []

	# *--
	# *--
	# *--
	if x > 0:
		neighbors.append((x-1, y))
		if y > 0:
			neighbors.append((x-1, y-1))
		if y < w - 1:
			neighbors.append((x-1, y+1))

	# -*-
	# ---
	# -*-
	if y > 0:
		neighbors.append((x, y-1))
	if y < w - 1:
		neighbors.append((x, y+1))

	# --*
	# --*
	# --*
	if x < w - 1:
		neighbors.append((x+1, y))
		if y > 0:
			neighbors.append((x+1, y-1))
		if y < w - 1:
			neighbors.append((x+1, y+1))

	return neighbors

def tile_satisfaction(graph: nx.Graph, tile_pos: Tuple[int, int], graph_size: Tuple[int, int]) -> None | float:
	"""
	Returns the satisfaction of a tile, from 0.0 to 1.0.

	If the tile doesn't have a race, returns `None`
	"""

	# If the tile is empty, it doesn't have a satisfaction
	tile: Tile = graph.nodes[tile_pos]['tile']
	if tile.is_empty():
		return None

	# Else count all neighbors that aren't empty
	neighbors_pos = tile_neighbors_pos(tile_pos, graph_size)
	neighbors = (graph.nodes[neighbor_pos]['tile'] for neighbor_pos in neighbors_pos)
	neighbors = filter(lambda tile: not tile.is_empty(), neighbors)
	neighbors = list(neighbors)

	# If there are none, we are satisfied
	# Note: The simulation in `http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/` works like
	#       this: When setting `empty = 90%`, some agents are empty but still count as satisfied.
	if len(neighbors) == 0:
		return 1.0

	# Else their satisfaction is the number of similar agents within their non-empty neighbors
	return sum(tile == neighbor for neighbor in neighbors) / len(neighbors)

if __name__ == "__main__":
	# Graph (initialized to empty)
	graph_size = [50, 50]
	graph: nx.Graph = nx.grid_2d_graph(graph_size[0], graph_size[1])
	nx.set_node_attributes(graph, Tile.empty(), 'tile')

	# Add the diagonal edges
	for y in range(graph_size[1] - 1):
		for x in range(graph_size[0] - 1):
			graph.add_edge((x, y), (x + 1, y + 1))
			graph.add_edge((x + 1, y), (x, y + 1))

	# Set some initial tiles
	random.seed(773)
	for race in [Race.BLUE, Race.RED]:
		for _ in range(1000):
			x = random.randint(0, graph_size[0] - 1)
			y = random.randint(0, graph_size[1] - 1)
			graph.nodes[(x, y)]['tile'] = Tile.filled(race)

	# Display method
	class DisplayMethod(Enum):
		# Displays as a general graph, with edges
		GRAPH = 1

		# Displays as a compact grid, without any spacing between nodes
		GRID = 2

	display_method = DisplayMethod.GRID

	# Visualize graph
	with plt.ion():
		# Create the figure
		fig = plt.figure()

		# Draw it using the display method
		if display_method == DisplayMethod.GRAPH:
			# Setup the positions and colors
			node_pos = { tile_pos: tile_pos for tile_pos in graph.nodes()}
			node_colors = [tile['tile'].color() for _, tile in graph.nodes(data=True)]

			# And finally draw the nodes and edges
			ax = fig.add_subplot(1, 1, 1)
			nx.draw_networkx_nodes(graph, pos=node_pos, ax=ax, node_color=node_colors, node_size=25, node_shape='s')
			nx.draw_networkx_edges(graph, pos=node_pos, ax=ax)

		elif display_method == DisplayMethod.GRID:
			# Show all tiles
			img: list[list[Tuple[int, int, int]]] = [[[0, 0, 0] for _ in range(graph_size[0])] for _ in range(graph_size[1])]
			for tile_pos, tile in graph.nodes(data = True):
				tile: Tile = tile['tile']
				img[tile_pos[1]][tile_pos[0]] = tile.color()

			ax = fig.add_subplot(1, 2, 1)
			ax.axis("off")
			ax.imshow(img)

			# The the satisfaction of each one, in a separate plot
			for tile_pos in graph.nodes():
				satisfaction = tile_satisfaction(graph, tile_pos, graph_size)
				img[tile_pos[1]][tile_pos[0]] = [satisfaction, 0.0, 0.0] if satisfaction is not None else [0.0, 0.0, 0.0]

			ax = fig.add_subplot(1, 2, 2)
			ax.axis("off")
			ax.imshow(img)

		else:
			raise ValueError("Unknown display method")

		# Finally show it
		fig.tight_layout()
		plt.show(block = True)
