"""
Graph
"""

from dataclasses import dataclass
import dataclasses
import time
from typing import Any, Generator, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy

import arc_proj.util as util
from arc_proj.agent import Agent

# Node position type
NodePos = Tuple[int, int]

@dataclass
class GraphCache:
	"""
	Cache for `Graph`
	"""

	# Unsatisfied nodes
	unsatisfied_nodes: set[NodePos] = dataclasses.field(default_factory=set)

	# Empty nodes
	empty_nodes: set[NodePos] = dataclasses.field(default_factory=set)


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

	# Cache
	cache: GraphCache

	def __init__(self, graph_size: Tuple[int, int]) -> None:
		"""
		Initializes the graph with a size of `graph_size`
		"""

		# Create the graph and initialize it to empty
		self.size = graph_size
		self.graph: nx.Graph = nx.grid_2d_graph(graph_size[0], graph_size[1])
		self.cur_round = 0
		self.cache = GraphCache()

		# Then add the diagonal edges.
		for y in range(graph_size[1] - 1):
			for x in range(graph_size[0] - 1):
				self.graph.add_edge((x, y), (x + 1, y + 1))
				self.graph.add_edge((x + 1, y), (x, y + 1))

		# Initialize caches
		self.cache.empty_nodes = set(node_pos for node_pos in self.graph.nodes)

	def add_agent(self, node_pos: NodePos, agent: Agent):
		"""
		Adds an agent `agent` at node `node_pos`.

		The node at `node_pos` *must* have been empty
		"""

		# Set it on the graph
		assert 'agent' not in self.graph.nodes[node_pos], "Node position had agent"
		self.graph.nodes[node_pos]['agent'] = agent

		# Then update the caches
		self.cache.empty_nodes.remove(node_pos)
		self.update_unsatisfied_nodes_cache(node_pos)

	def remove_agent(self, node_pos: NodePos) -> Agent:
		"""
		Removes an agent at node `node_pos`.

		The node at `node_pos` *must* have an agent
		"""

		# Remove it from the graph
		agent = util.try_index_dict(self.graph.nodes[node_pos], 'agent')
		assert agent is not None, "Node position did not have agent"
		del self.graph.nodes[node_pos]['agent']

		# Then update the caches
		# Note: `unsatisfied_nodes.remove(node_pos)` might not succeed, but we're fine
		#       with that, we only want to remove it, if it exists anyway
		self.cache.empty_nodes.add(node_pos)
		self.cache.unsatisfied_nodes.remove(node_pos)
		self.update_unsatisfied_nodes_cache(node_pos, skip_node=True)

		return agent

	def remove_unsatisfied_agents(self) -> list[Agent]:
		"""
		Removes all unsatisfied agents
		"""

		# Remove them all from the graph
		agents = []
		for node_pos in self.cache.unsatisfied_nodes:
			agent = util.try_index_dict(self.graph.nodes[node_pos], 'agent')
			assert agent is not None, "Node position did not have agent"
			del self.graph.nodes[node_pos]['agent']
			agents.append(agent)

		# Then mass-update the unsatisfied nodes
		# Note: We need to update since some neighbor nodes might now be
		#       unsatisfied from a similar agent leaving their neighborhood.
		unsatisfied_nodes = list(self.cache.unsatisfied_nodes)
		self.cache.unsatisfied_nodes.clear()
		for node_pos in unsatisfied_nodes:
			self.update_unsatisfied_nodes_cache(node_pos)

		# Finally add them to the empty nodes
		self.cache.empty_nodes.update(unsatisfied_nodes)

		return agents

	def update_unsatisfied_nodes_cache(self, node_pos: NodePos):
		"""
		Updates the unsatisfied nodes cache for the node `node_pos` and neighbors.
		"""

		# Check the current agent
		if self.agent_satisfied(node_pos) == False:
			self.cache.unsatisfied_nodes.add(node_pos)

		# Then check all neighbors
		for neighbor in nx.neighbors(self.graph, node_pos):
			if self.agent_satisfied(neighbor) == False:
				self.cache.unsatisfied_nodes.add(neighbor)

	def fill_with_agents(self, empty_chance: float, agent_weights: dict[Agent, float]) -> None:
		"""
		Fills the graph with agents.

		There is a `empty_change` chance of a node being empty.
		Else, the weights in `agent_weights` are used to generate an agent
		"""

		# Normalize the weights to sum to 1
		weights_sum = sum(agent_weights.values())
		agent_weights = { agent: weight / weights_sum for agent, weight in agent_weights.items() }

		# Then for each node, select either empty or a random agent
		# TODO: Maybe optimize cases when `empty_cache` is high?
		for node_pos in self.graph.nodes:
			# If we're meant to be empty, continue to the next node
			if numpy.random.random() < empty_chance:
				continue

			# Select a random agent
			agent = numpy.random.choice(list(agent_weights.keys()), p=list(agent_weights.values()))
			self.add_agent(node_pos, agent)

	def agent_satisfaction(self, node_pos: NodePos) -> float | None:
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

	def agent_satisfied(self, node_pos: NodePos) -> bool | None:
		"""
		Returns if an agent is satisfied
		"""

		# Get the satisfaction
		satisfaction = self.agent_satisfaction(node_pos)
		if satisfaction is None:
			return None

		# Then check with the agent
		# Note: Since `satisfaction` isn't `None`, we know it must exist
		agent: Agent = self.graph.nodes[node_pos]['agent']

		return satisfaction >= agent.threshold()

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
		for node_pos, node in self.graph.nodes(data=True):
			satisfaction = self.agent_satisfaction(node_pos)
			satisfied = self.agent_satisfied(node_pos)
			img[node_pos[1]][node_pos[0]] = [satisfaction, 0.0, satisfied] if satisfaction is not None else [1.0, 0.0, 1.0]

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

		print(f"Round #{self.cur_round+1}:")
		print(f"\tUnsatisfied nodes: {len(self.cache.unsatisfied_nodes)}")
		start_time = time.time()
		self.cur_round += 1

		# Remove all unsatisfied agents
		removed_agents = self.remove_unsatisfied_agents()

		# Then sample some empty nodes
		# Note: Unfortunately this is faster than reservoir sampling with
		#       the set, as that's still `O(max(n, k))` due to not being able to
		#       efficiently advance the iterator by a delta.
		empty_nodes = list(self.cache.empty_nodes)
		numpy.random.shuffle(empty_nodes)

		# And find a new spot for all removed agents
		for agent, node_pos in zip(removed_agents, empty_nodes):
			self.add_agent(node_pos, agent)

		print(f"\tTook {util.fmt_time(time.time() - start_time)}")

		reached_equilibrium = len(self.cache.unsatisfied_nodes) == 0
		if reached_equilibrium:
			for node_pos in self.graph.nodes():
				satisfied = self.agent_satisfied(node_pos)
				assert satisfied is None or satisfied, "A"

		return reached_equilibrium
