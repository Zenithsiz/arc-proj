"""
ARC Project
"""

import time
from enum import Enum

import matplotlib.pyplot as plt
import numpy

from arc_proj.agent import Agent
from arc_proj.graph import Graph

if __name__ == "__main__":
	# Create the graph
	graph = Graph([50, 50], satisfaction_threshold = 0.5)
	numpy.random.seed(773)
	graph.fill_with_agents(0.05, { Agent.RED: 0.5, Agent.BLUE: 0.5 })

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

		# And draw
		while True:
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
			fig.canvas.draw()
			fig.canvas.flush_events()

			# Then sleep for a bit
			time.sleep(1.0 / 30.0)

			# And update the graph
			if graph.do_round():
				print(f"Reached equilibrium after {graph.cur_round} round(s)")
				break

		# Finally, once we're done, block until the user closes the plots
		plt.show(block=True)
