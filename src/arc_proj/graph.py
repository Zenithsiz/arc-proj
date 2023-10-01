"""
Graph
"""

from typing import Any, Generator, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy

import arc_proj.util as util
from arc_proj.agent import Agent


class Graph:
	"""
	Represents a graph of agents
	"""

	# Inner graph
	graph: nx.Graph

	# Graph size
	size: Tuple[int, int]

	# Current round
	cur_round: int

	# Satisfaction threshold
	satisfaction_threshold: float

	def __init__(self, graph_size: Tuple[int, int], satisfaction_threshold: float = 0.3) -> None:
		"""
		Initializes the graph with a size of `graph_size`
		"""

		# Create the graph and initialize it to empty
		self.size = graph_size
		self.graph: nx.Graph = nx.grid_2d_graph(graph_size[0], graph_size[1])
		self.cur_round = 0
		self.satisfaction_threshold = satisfaction_threshold

		# Then add the diagonal edges.
		for y in range(graph_size[1] - 1):
			for x in range(graph_size[0] - 1):
				self.graph.add_edge((x, y), (x + 1, y + 1))
				self.graph.add_edge((x + 1, y), (x, y + 1))

	def fill_with_agents(self, empty_chance: float, agent_weights: dict[Agent, float]) -> None:
		"""
		Fills the graph with agents.

		There is a `empty_change` chance of a node being empty.
		Else, the weights in `agent_weights` are used to generate an agent
		"""

		for node_pos in self.graph.nodes:
			# If we're meant to be empty, continue to the next node
			if numpy.random.random() < empty_chance:
				continue

			# Select a random agent
			agent = numpy.random.choice(list(agent_weights.keys()), p=list(agent_weights.values()))
			self.graph.nodes[node_pos]['agent'] = agent

	def agent_satisfaction(self, node_pos: Tuple[int, int]) -> float | None:
		"""
		Returns the satisfaction of an agent, from 0.0 to 1.0.

		If the node doesn't have an agent, returns `None`
		"""

		# If the node is empty, it doesn't have a satisfaction
		node = self.graph.nodes[node_pos]
		agent = util.try_index_dict(node, 'agent')
		if agent is None:
			return None
		agent: Agent

		# Else count all neighbors that aren't empty
		neighbors = nx.neighbors(self.graph, node_pos)
		neighbors: Generator[Any] = (self.graph.nodes[neighbor_pos] for neighbor_pos in neighbors)
		neighbors: Generator[Agent | None] = map(lambda node: util.try_index_dict(node, 'agent'), neighbors)
		neighbors: Generator[Agent] = filter(lambda agent: agent is not None, neighbors)
		neighbors: list[Agent] = list(neighbors)

		# If there are none, we are satisfied
		# Note: The simulation in `http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/` works like
		#       this: When setting `empty = 90%`, some agents are empty but still count as satisfied.
		if len(neighbors) == 0:
			return 1.0

		# Else their satisfaction is the number of similar agents within their non-empty neighbors
		return sum(agent == neighbor for neighbor in neighbors) / len(neighbors)

	def agent_img(self) -> list[list[Tuple[int, int, int]]]:
		"""
		Returns an image of all agents in the graph
		"""
		img = [[(0, 0, 0) for _ in range(self.size[0])] for _ in range(self.size[1])]
		for node_pos, node in self.graph.nodes(data = True):
			node = util.try_index_dict(node, 'agent')
			img[node_pos[1]][node_pos[0]] = node.color() if node is not None else [0.5, 0.5, 0.5]

		return img

	def satisfaction_img(self) -> list[list[Tuple[int, int, int]]]:
		"""
		Returns an image of the satisfaction of all agents in the graph
		"""
		img = [[(0, 0, 0) for _ in range(self.size[0])] for _ in range(self.size[1])]
		for node_pos in self.graph.nodes():
			satisfaction = self.agent_satisfaction(node_pos)
			img[node_pos[1]][node_pos[0]] = [satisfaction, 0.0, 0.0] if satisfaction is not None else [0.0, 0.0, 0.0]

		return img

	def draw(self, fig: plt.Figure):
		"""
		Draws the graph using standard `networkx` draw calls
		"""

		node_pos = { node_pos: node_pos for node_pos in self.graph.nodes()}
		agents = (util.try_index_dict(node, 'agent') for _, node in self.graph.nodes(data=True))
		node_colors = [agent.color() if agent is not None else [0.5, 0.5, 0.5] for agent in agents]

		# And finally draw the nodes and edges
		ax = fig.add_subplot(1, 1, 1)
		nx.draw_networkx_nodes(self.graph, pos=node_pos, ax=ax, node_color=node_colors, node_size=25, node_shape='s')
		nx.draw_networkx_edges(self.graph, pos=node_pos, ax=ax)

	def do_round(self) -> bool:
		"""
		Performs a single round.

		Returns whether we've reached equilibrium
		"""

		# Go up a round
		self.cur_round += 1

		# Go through all agents, and remove the unhappy ones
		removed_agents = []
		for node_pos, node in self.graph.nodes(data=True):
			agent = util.try_index_dict(node, 'agent')
			if agent is None:
				continue
			agent: Agent

			# Note: The agent exists, so this mustn't be `None`.
			satisfaction: float = self.agent_satisfaction(node_pos)

			if satisfaction < self.satisfaction_threshold:
				removed_agents.append(agent)
				del self.graph.nodes[node_pos]['agent']

		# If we removed None, we've reached equilibrium
		if len(removed_agents) == 0:
			return True

		# Else find all empty nodes and shuffle them
		empty_nodes = [node for _, node in self.graph.nodes(data=True) if 'agent' not in node]

		# And find a new spot for all removed agents
		for agent in removed_agents:
			empty_node_idx = numpy.random.choice(len(empty_nodes))
			empty_nodes[empty_node_idx]['agent'] = agent
			del empty_nodes[empty_node_idx]

		return False
