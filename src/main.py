"""
ARC Project
"""

import os
import sys
import time
from enum import Enum
from io import StringIO

import matplotlib.pyplot as plt
import numpy

import arc_proj.util as util
from arc_proj.agent import Agent
from arc_proj.graph import Graph
from PIL import Image


def main():
	# Create the graph
	start_time = time.time()

	print("Creation:")
	graph_size = [80, 80]
	graph = Graph(graph_size=graph_size, seed=773)
	graph.fill_with_agents(0.1, { Agent.RED: 1, Agent.BLUE: 1 })

	creation_duration = time.time() - start_time
	print(f"\tTook {util.fmt_time(creation_duration)} ({util.fmt_time(creation_duration / (graph_size[0] * graph_size[1]))}/node)")

	# Display method
	class DisplayMethod(Enum):
		# Does not display
		NONE = 0

		# Displays as a general graph, with edges
		GRAPH = 1

		# Displays as a compact grid, without any spacing between nodes
		GRID = 2

		# Writes the compact grid image to file
		GRID_FILE = 3

		def needs_fig(self) -> bool:
			"""Returns if this display method needs a figure"""
			return self in [DisplayMethod.GRAPH, DisplayMethod.GRID]

		def needs_dir(self) -> bool:
			"""Returns if this display methods needs an output directory"""
			return self in [DisplayMethod.GRID_FILE]

	display_method = DisplayMethod.GRID
	rounds_per_display = 1

	# Setup display
	if display_method.needs_dir():
		os.makedirs("output/", exist_ok=True)

	# Visualize graph
	with plt.ion():
		# Create the figure
		fig = plt.figure() if display_method.needs_fig() else None

		def draw():
			"""
			Draws the graph
			"""

			# Check the display method
			if display_method == DisplayMethod.NONE:
				pass

			elif display_method == DisplayMethod.GRAPH:
				fig.clear()
				graph.draw(fig)

				fig.tight_layout()
				fig.canvas.draw()
				fig.canvas.flush_events()

			elif display_method == DisplayMethod.GRID:
				fig.clear()

				# Show all nodes
				ax = fig.add_subplot(1, 2, 1)
				ax.axis("off")
				ax.imshow(graph.agent_img())

				# The the satisfaction of each one, in a separate plot
				ax = fig.add_subplot(1, 2, 2)
				ax.axis("off")
				ax.imshow(graph.satisfaction_img())

				fig.tight_layout()
				fig.canvas.draw()
				fig.canvas.flush_events()

			elif display_method == DisplayMethod.GRID_FILE:
				buffer = graph.agent_img()
				buffer = [(int(255 * r), int(255 * g), int(255 * b)) for row in buffer for r, g, b in row]

				img = Image.new("RGB", (graph_size[0], graph_size[1]))
				img.putdata(buffer)
				img.save(f"output/{cur_round}.png")

			else:
				raise ValueError("Unknown display method")

		# And draw
		cur_round = 0
		while True:
			# Draw
			draw()

			# And update the graph
			reached_equilibrium = False
			for _ in range(rounds_per_display):
				print(f"Round #{cur_round+1}:")
				unsatisfied_nodes = len(graph.cache.unsatisfied_nodes)
				print(f"\tUnsatisfied nodes: {unsatisfied_nodes}")
				start_time = time.time()
				cur_round += 1

				reached_equilibrium |= graph.do_round()

				round_duration = time.time() - start_time
				print(f"\tTook {util.fmt_time(round_duration)} ({util.fmt_time(round_duration / unsatisfied_nodes if unsatisfied_nodes != 0 else 0.0)}/agent)")

				if reached_equilibrium:
					break

			if reached_equilibrium:
				print(f"Reached equilibrium after {cur_round} round(s)")
				draw()
				break

		# Finally, once we're done, block until the user closes the plots
		if display_method.needs_fig():
			plt.show(block=True)

if __name__ == "__main__":
	# Execution method
	class ExecMethod(Enum):
		# Normal execution
		NORMAL = 0

		# Benchmark execution
		BENCHMARK = 1

	exec_method = ExecMethod.NORMAL

	if exec_method == ExecMethod.NORMAL:
		main()

	elif exec_method == ExecMethod.BENCHMARK:
		# Limits
		max_samples = 100
		max_time_s = 30

		# Samples and totals
		samples = []
		total_time_s = 0
		for cur_sample in range(max_samples):
			# Measure
			# Note: We suppress stdout while measuring, to not clutter the output
			sys.stdout = StringIO()
			start_time_ns = time.time_ns()
			main()
			elapsed_time_ns = time.time_ns() - start_time_ns
			sys.stdout = sys.__stdout__

			print(f"Sample #{cur_sample+1}: {util.fmt_time(elapsed_time_ns / 1e9)}", end="\r")

			samples.append(elapsed_time_ns)

			# If we're past the total time, break
			total_time_s += elapsed_time_ns / 1e9
			if total_time_s > max_time_s:
				break

		# Then output some statistical information
		avg_ns = numpy.average(samples)
		std_ns = numpy.std(samples)
		min_ns = numpy.min(samples)
		max_ns = numpy.max(samples)

		print()
		print("Benchmark:")
		print(f"\tSamples: {len(samples)}")
		print(f"\tTime: {util.fmt_time(avg_ns / 1e9)} ± {util.fmt_time(std_ns / 1e9)}")
		print(f"\tRange: {util.fmt_time(min_ns / 1e9)} .. {util.fmt_time(max_ns / 1e9)}")

	else:
		raise ValueError("Unknown exec method")
