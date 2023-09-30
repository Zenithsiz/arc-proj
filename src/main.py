"""
ARC Project
"""

# Imports
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt

from arc_proj.race import Race
from arc_proj.tile import Tile

if __name__ == "__main__":
	# Graph (initialized to empty)
	graph_size = [50, 50]
	graph = nx.grid_2d_graph(graph_size[0], graph_size[1])
	nx.set_node_attributes(graph, Tile.empty(), 'tile')

	# Add the diagonal edges
	for y in range(graph_size[1] - 1):
		for x in range(graph_size[0] - 1):
			graph.add_edge((x, y), (x + 1, y + 1))
			graph.add_edge((x + 1, y), (x, y + 1))

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
			img = [[0 for _ in range(graph_size[0])] for _ in range(graph_size[1])]
			for (tile_pos_x, tile_pos_y), tile in graph.nodes(data = True):
				img[tile_pos_y][tile_pos_x] = tile['tile'].color()

			# And finally show it
			ax.imshow(img)

		else:
			raise ValueError("Unknown display method")

		# Finally show it
		fig.tight_layout()
		plt.show(block = True)
