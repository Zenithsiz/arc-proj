"""
ARC Project
"""

import random
from enum import Enum

import matplotlib.pyplot as plt
import networkx as nx

from arc_proj.agent import Agent
from arc_proj.graph import Graph

if __name__ == "__main__":
	# Create the graph
	graph = Graph([50, 50])
	random.seed(773)
	graph.spread_agents(Agent.RED, 1000)
	graph.spread_agents(Agent.BLUE, 1000)

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
			graph.draw(fig)

		elif display_method == DisplayMethod.GRID:
			# Show all nodes
			ax = fig.add_subplot(1, 2, 1)
			ax.axis("off")
			ax.imshow(graph.agent_img())

			# The the satisfaction of each one, in a separate plot
			ax = fig.add_subplot(1, 2, 2)
			ax.axis("off")
			ax.imshow(graph.satisfaction_img())

		else:
			raise ValueError("Unknown display method")

		# Finally show it
		fig.tight_layout()
		plt.show(block = True)
