"""
ARC Project
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from typing import Tuple

import matplotlib.pyplot as plt
import numpy
from PIL import Image

import arc_proj.util as util
from arc_proj.agent import Agent, NAgent, NAgentKind
from arc_proj.graph import Graph


class DisplayMethod(Enum):
	"""
	Display method
	"""

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


@dataclass
class RunParams:
	"""
	Parameters for running the simulation
	"""

	# Graph size
	graph_size: Tuple[int, int]

	# Seed
	seed: int | None

	# Empty chance
	empty_chance: float

	# Agent weights
	agent_weights: dict[Agent, float]

	# Output json file path
	output_json_path: str | None

	# Output image dir path
	output_img_path: str | None

	# Display method
	display_method: DisplayMethod

	# Rounds per display
	rounds_per_display: int

def run(params: RunParams):
	"""
	Runs the simulation
	"""

	# Create the graph
	start_time = time.time()

	print("Creation:")
	graph = Graph(graph_size=params.graph_size, seed=params.seed)
	graph.fill_with_agents(params.empty_chance, params.agent_weights)

	average_satisfactions = []

	creation_duration = time.time() - start_time
	print(f"\tTook {util.fmt_time(creation_duration)} ({util.fmt_time(creation_duration / (params.graph_size[0] * params.graph_size[1]))}/node)")

	# Setup display
	if params.display_method.needs_dir():
		assert params.output_img_path is not None, "Display method needs output image path, but it was `None`"
		os.makedirs(params.output_img_path, exist_ok=True)

	# Visualize graph
	with plt.ion():
		# Create the figure
		fig = plt.figure() if params.display_method.needs_fig() else None

		def draw():
			"""
			Draws the graph
			"""

			# Check the display method
			if params.display_method == DisplayMethod.NONE:
				pass

			elif params.display_method == DisplayMethod.GRAPH:
				fig.clear()
				graph.draw(fig)

				fig.tight_layout()
				fig.canvas.draw()
				fig.canvas.flush_events()

			elif params.display_method == DisplayMethod.GRID:
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

			elif params.display_method == DisplayMethod.GRID_FILE:
				buffer = graph.agent_img()
				buffer = [(int(255 * r), int(255 * g), int(255 * b)) for row in buffer for r, g, b in row]

				img = Image.new("RGB", (params.graph_size[0], params.graph_size[1]))
				img.putdata(buffer)
				img.save(f"{params.output_img_path}/{cur_round}.png")

			else:
				raise ValueError("Unknown display method")

		# And draw
		cur_round = 0
		while True:
			# Draw
			draw()

			# And update the graph
			reached_equilibrium = False
			for _ in range(params.rounds_per_display):
				print(f"Round #{cur_round+1}:")
				unsatisfied_nodes = len(graph.cache.unsatisfied_nodes)
				print(f"\tUnsatisfied nodes: {unsatisfied_nodes}")
				average_satisfaction = graph.agent_average_satisfaction()
				print(f"\tAverage satisfaction: {average_satisfaction:.5f}")
				average_satisfactions.append(average_satisfaction)
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

		# Write the output
		if params.output_json_path is not None:
			output = {
				'average_satisfactions': average_satisfactions,
			}
			output_file = open(params.output_json_path, 'w', encoding="utf-8")
			json.dump(output, output_file)

		# Finally, once we're done, block until the user closes the plots
		if params.display_method.needs_fig():
			plt.show(block=True)

def main():
	"""
	Main function
	"""

	class ExecMethod(Enum):
		"""
		Execution method
		"""

		# Normal execution
		NORMAL = 0

		# Benchmark execution
		BENCHMARK = 1

	exec_method = ExecMethod.NORMAL

	if exec_method == ExecMethod.NORMAL:
		params = RunParams(
			graph_size=[80, 80],
			seed=773,
			empty_chance=0.1,
			agent_weights={ NAgent(NAgentKind.RED): 1, NAgent(NAgentKind.BLUE): 1 },
			output_json_path="output.json",
			output_img_path="output",
			display_method=DisplayMethod.GRID,
			rounds_per_display=1
		)

		run(params)

	elif exec_method == ExecMethod.BENCHMARK:
		# Limits
		max_samples = 100
		max_time_s = 30

		# Parameters
		params = RunParams(
			graph_size=[80, 80],
			seed=773,
			empty_chance=0.1,
			agent_weights={ NAgent(NAgentKind.RED): 1, NAgent(NAgentKind.BLUE): 1 },
			output_json_path=None,
			output_img_path=None,
			display_method=DisplayMethod.NONE,
			rounds_per_display=1
		)

		# Samples and totals
		samples = []
		total_time_s = 0
		for cur_sample in range(max_samples):
			# Measure
			# Note: We suppress stdout while measuring, to not clutter the output
			sys.stdout = StringIO()
			start_time_ns = time.time_ns()
			run(params)
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

if __name__ == "__main__":
	main()
