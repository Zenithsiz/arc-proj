"""
ARC Project
"""

# Imports
from enum import Enum
from typing import Any, Generator, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from arc_proj.agent import Agent
import random

def node_neighbors_pos(node_pos: Tuple[int, int], graph_size: Tuple[int, int]) -> list[Tuple[int, int]]:
	"""
	Returns all neighbors positions of `node_pos`.
	"""

	x = node_pos[0]
	y = node_pos[1]
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

def agent_satisfaction(graph: nx.Graph, node_pos: Tuple[int, int], graph_size: Tuple[int, int]) -> None | float:
	"""
	Returns the satisfaction of an agent, from 0.0 to 1.0.

	If the node doesn't have an agent, returns `None`
	"""

	# If the node is empty, it doesn't have a satisfaction
	node = graph.nodes[node_pos]
	agent: Agent | None = node['agent'] if 'agent' in node else None
	if agent is None:
		return None
	agent: Agent

	# Else count all neighbors that aren't empty
	neighbors_pos = node_neighbors_pos(node_pos, graph_size)
	neighbors: Generator[Any] = (graph.nodes[neighbor_pos] for neighbor_pos in neighbors_pos)
	neighbors: Generator[Agent | None] = map(lambda node: node['agent'] if 'agent' in node else None, neighbors)
	neighbors: Generator[Agent] = filter(lambda agent: agent is not None, neighbors)
	neighbors: list[Agent] = list(neighbors)

	# If there are none, we are satisfied
	# Note: The simulation in `http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/` works like
	#       this: When setting `empty = 90%`, some agents are empty but still count as satisfied.
	if len(neighbors) == 0:
		return 1.0

	# Else their satisfaction is the number of similar agents within their non-empty neighbors
	return sum(agent == neighbor for neighbor in neighbors) / len(neighbors)

if __name__ == "__main__":
	# Graph (initialized without any agents)
	graph_size = [50, 50]
	graph: nx.Graph = nx.grid_2d_graph(graph_size[0], graph_size[1])

	# Add the diagonal edges
	for y in range(graph_size[1] - 1):
		for x in range(graph_size[0] - 1):
			graph.add_edge((x, y), (x + 1, y + 1))
			graph.add_edge((x + 1, y), (x, y + 1))

	# Set some initial agents
	random.seed(773)
	for agent in [Agent.BLUE, Agent.RED]:
		for _ in range(1000):
			x = random.randint(0, graph_size[0] - 1)
			y = random.randint(0, graph_size[1] - 1)
			graph.nodes[(x, y)]['agent'] = agent

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
			node_pos = { node_pos: node_pos for node_pos in graph.nodes()}
			node_colors = [node['agent'].color() if 'agent' in node else [0.5, 0.5, 0.5] for _, node in graph.nodes(data=True)]

			# And finally draw the nodes and edges
			ax = fig.add_subplot(1, 1, 1)
			nx.draw_networkx_nodes(graph, pos=node_pos, ax=ax, node_color=node_colors, node_size=25, node_shape='s')
			nx.draw_networkx_edges(graph, pos=node_pos, ax=ax)

		elif display_method == DisplayMethod.GRID:
			# Show all nodes
			img: list[list[Tuple[int, int, int]]] = [[[0, 0, 0] for _ in range(graph_size[0])] for _ in range(graph_size[1])]
			for node_pos, node in graph.nodes(data = True):
				node: Agent | None = node['agent'] if 'agent' in node else None
				img[node_pos[1]][node_pos[0]] = node.color() if node is not None else [0.5, 0.5, 0.5]

			ax = fig.add_subplot(1, 2, 1)
			ax.axis("off")
			ax.imshow(img)

			# The the satisfaction of each one, in a separate plot
			for node_pos in graph.nodes():
				satisfaction = agent_satisfaction(graph, node_pos, graph_size)
				img[node_pos[1]][node_pos[0]] = [satisfaction, 0.0, 0.0] if satisfaction is not None else [0.0, 0.0, 0.0]

			ax = fig.add_subplot(1, 2, 2)
			ax.axis("off")
			ax.imshow(img)

		else:
			raise ValueError("Unknown display method")

		# Finally show it
		fig.tight_layout()
		plt.show(block = True)
