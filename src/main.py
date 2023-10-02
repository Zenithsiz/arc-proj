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
	graph = Graph([80, 80])
	numpy.random.seed(773)
	graph.fill_with_agents(0.1, { Agent.RED: 1, Agent.BLUE: 1 })

	# Display method
	class DisplayMethod(Enum):
		# Displays as a general graph, with edges
		GRAPH = 1

		# Displays as a compact grid, without any spacing between nodes
		GRID = 2

	display_method = DisplayMethod.GRID
	rounds_per_display = 1

	# Visualize graph
	with plt.ion():
		# Create the figure
		fig = plt.figure()

		def draw():
			"""
			Draws the graph
			"""

			# Clear any previous figures
			fig.clear()

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

		# And draw
		while True:
			# Draw
			draw()

			# And update the graph
			reached_equilibrium = False
			for _ in range(rounds_per_display):
				reached_equilibrium |= graph.do_round()
				if reached_equilibrium:
					break

			if reached_equilibrium:
				print(f"Reached equilibrium after {graph.cur_round} round(s)")
				draw()
				break

		# Finally, once we're done, block until the user closes the plots
		plt.show(block=True)
