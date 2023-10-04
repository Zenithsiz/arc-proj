"""
Graph
"""

from dataclasses import dataclass
import itertools
from typing import Any, Generator, Iterable, Tuple

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
	unsatisfied_nodes: set[NodePos]

	# Empty nodes
	empty_nodes: set[NodePos]

@dataclass
class DebugOptions:
	"""
	Debug options for `Graph`
	"""

	# Whether to sanity check the caches
	sanity_check_caches: bool

class Graph:
	"""
	Represents a graph of agents
	"""

	# Inner graph
	graph: nx.Graph

	# Graph size
	size: Tuple[int, int]

	# Random state
	random_state: numpy.random.RandomState

	# Cache
	cache: GraphCache

	# Debug
	debug: DebugOptions

	def __init__(self, graph_size: Tuple[int, int], seed: int) -> None:
		"""
		Initializes the graph with a size of `graph_size`
		"""

		# Create the graph and initialize it to empty
		self.size = graph_size
		self.graph: nx.Graph = nx.grid_2d_graph(graph_size[0], graph_size[1])
		self.random_state = numpy.random.RandomState(seed)
		self.debug = DebugOptions(
			sanity_check_caches=False
		)

		# Then add the diagonal edges.
		for y in range(graph_size[1] - 1):
			for x in range(graph_size[0] - 1):
				self.graph.add_edge((x, y), (x + 1, y + 1))
				self.graph.add_edge((x + 1, y), (x, y + 1))

		# And initialize caches
		self.cache = GraphCache(
			unsatisfied_nodes=set(),
			empty_nodes=set(node_pos for node_pos in self.graph.nodes),
		)

	def move_agents(self, nodes_pos: dict[NodePos, NodePos]):
		"""
		Moves all agents in `nodes_pos`.
		"""

		# Note: This pass both removes the agents from their previous locations
		#       and moves them to the next. In order to not override
		agents_cache = dict()
		for src_pos, dst_pos in nodes_pos.items():
			# Skip when the source and destination are the same
			if src_pos == dst_pos:
				continue

			# Get the agent in the node (but try our agent cache first, because we've might overridden it already)
			agent = agents_cache.pop(src_pos, None)
			if agent is None:
				src_node = self.graph.nodes[src_pos]
				assert 'agent' in src_node, f"Node position {src_node} did not have an agent"
				agent = src_node['agent']
				del src_node['agent']
				self.cache.empty_nodes.add(src_pos)

			# Then save the agent in the to slot (in case we need it later), then write our agent into it
			dst_node = self.graph.nodes[dst_pos]
			if 'agent' in dst_node:
				agents_cache[dst_pos] = dst_node['agent']
			dst_node['agent'] = agent
			self.cache.empty_nodes.discard(dst_pos)

		assert len(agents_cache) == 0, f"Destination nodes overlapped or destination node already had an agent: {agents_cache}"

		# Then mass-update the source and destination nodes
		self.update_unsatisfied_nodes_cache_multiple(itertools.chain(nodes_pos.keys(), nodes_pos.values()))

	def add_agent(self, node_pos: NodePos, agent: Agent):
		"""
		Adds an agent `agent` at node `node_pos`.

		The node at `node_pos` *must* have been empty
		"""

		# Set it on the graph
		node = self.graph.nodes[node_pos]
		assert 'agent' not in node, f"Node position {node_pos} already had an agent: {node['agent']}"
		node['agent'] = agent

		# Then update the caches
		self.cache.empty_nodes.remove(node_pos)
		self.update_unsatisfied_nodes_cache(node_pos)

	def remove_agent(self, node_pos: NodePos) -> Agent:
		"""
		Removes an agent at node `node_pos`.

		The node at `node_pos` *must* have an agent
		"""

		# Remove it from the graph
		node = self.graph.nodes[node_pos]
		assert 'agent' in node, f"Node position {node_pos} did not have agent"
		agent = node['agent']
		del node['agent']

		# Then update the caches
		self.cache.empty_nodes.add(node_pos)
		self.update_unsatisfied_nodes_cache(node_pos)

		return agent

	def update_unsatisfied_nodes_cache_multiple(self, nodes: Iterable[NodePos]):
		"""
		Updates the unsatisfied nodes cache for the nodes in `nodes` and neighbors.
		"""

		# Get all nodes and neighbors
		all_nodes = set()
		for node_pos in nodes:
			all_nodes.add(node_pos)
			all_nodes.update(self.graph.adj[node_pos])

		# Then check each one
		for node_pos in all_nodes:
			match self.agent_satisfied(node_pos):
				case True | None: self.cache.unsatisfied_nodes.discard(node_pos)
				case False:       self.cache.unsatisfied_nodes.add    (node_pos)

	def update_unsatisfied_nodes_cache(self, node_pos: NodePos):
		"""
		Updates the unsatisfied nodes cache for the node `node_pos` and neighbors.
		"""

		all_nodes = [node_pos, *self.graph.adj[node_pos]]
		for node_pos in all_nodes:
			match self.agent_satisfied(node_pos):
				case True | None: self.cache.unsatisfied_nodes.discard(node_pos)
				case False:       self.cache.unsatisfied_nodes.add    (node_pos)

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
			if self.random_state.random() < empty_chance:
				continue

			# Select a random agent
			agent = self.random_state.choice(list(agent_weights.keys()), p=list(agent_weights.values()))
			self.graph.nodes[node_pos]['agent'] = agent
			self.cache.empty_nodes.remove(node_pos)

		# At the end, update the remaining caches
		self.update_unsatisfied_nodes_cache_multiple(self.graph.nodes)

	def agent_satisfaction(self, node_pos: NodePos) -> float | None:
		"""
		Returns the satisfaction of an agent, from 0.0 to 1.0.

		If the node doesn't have an agent, returns `None`
		"""

		# If the node is empty, it doesn't have a satisfaction
		node = self.graph.nodes[node_pos]
		if 'agent' not in node:
			return None
		agent: Agent = node['agent']

		# Else count all neighbors that aren't empty
		neighbors: list[Agent] = []
		for neighbor_pos in self.graph.adj[node_pos]:
			neighbor_node = self.graph.nodes[neighbor_pos]
			if 'agent' not in neighbor_node:
				continue

			neighbors.append(neighbor_node['agent'])

		return agent.satisfaction(neighbors)

	def agent_satisfied(self, node_pos: NodePos) -> bool | None:
		"""
		Returns if an agent is satisfied
		"""

		# Get the satisfaction of the agent
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
			agent: Agent | None = util.try_index_dict(node, 'agent')
			img[node_pos[1]][node_pos[0]] = agent.color() if agent is not None else [0.5, 0.5, 0.5]

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

		# Sanity check: Ensure the caches are valid
		# Note: This is pretty slow normally, so we only enable it
		#       on debug and when explicitly requested
		if __debug__ and self.debug.sanity_check_caches:
			self.update_unsatisfied_nodes_cache_multiple(self.graph.nodes)
			for node_pos in self.graph.nodes:
				match self.agent_satisfied(node_pos):
					case True | None: assert node_pos not in self.cache.unsatisfied_nodes, f"Node {node_pos} was satisfied, but present in unsatisfied cache"
					case False      : assert node_pos     in self.cache.unsatisfied_nodes, f"Node {node_pos} wasn't satisfied, but not present in unsatisfied cache"

				if 'agent' not in self.graph.nodes[node_pos]:
					assert node_pos in self.cache.empty_nodes, f"Node {node_pos} was empty, but not present in empty cache"

		# Move all the current unsatisfied to another place
		# Note: We choose from both the empty nodes, as well as the unsatisfied, as they'll
		#       all be moving in a second, so we won't get duplicates.
		empty_nodes = list(self.cache.empty_nodes | self.cache.unsatisfied_nodes)
		self.random_state.shuffle(empty_nodes)
		self.move_agents(dict(zip(self.cache.unsatisfied_nodes, empty_nodes)))

		reached_equilibrium = len(self.cache.unsatisfied_nodes) == 0
		if reached_equilibrium:
			for node_pos in self.graph.nodes():
				satisfied = self.agent_satisfied(node_pos)
				assert satisfied is None or satisfied, f"Node {node_pos} wasn't satisfied after reaching equilibrium"

		return reached_equilibrium
