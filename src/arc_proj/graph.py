"""
Graph
"""

from dataclasses import dataclass
import itertools
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy

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

	# Whether to check equilibrium
	check_equilibrium: bool

class Graph:
	"""
	Represents a graph of agents
	"""

	# Inner graph
	# Note: Only used for edge-lookups, no values are stored
	#       within it.
	graph: nx.Graph

	# Graph size
	size: Tuple[int, int]

	# Agent type
	agent_ty: type[Agent] | None

	# Agents
	agents: dict[NodePos, Agent]

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
		self.agent_ty = None
		self.agents = dict()
		self.random_state = numpy.random.RandomState(seed)
		self.debug = DebugOptions(
			sanity_check_caches=False,
			check_equilibrium=True
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
			src_agent = agents_cache.pop(src_pos, None)
			if src_agent is None:
				src_agent = self.agents.pop(src_pos, None)
				assert src_agent is not None, f"Node position {src_pos} did not have an agent"
				self.cache.empty_nodes.add(src_pos)

			# Then save the agent in the to slot (in case we need it later), then write our agent into it
			dst_agent = self.agents.get(dst_pos, None)
			if dst_agent is not None:
				agents_cache[dst_pos] = dst_agent
			self.agents[dst_pos] = src_agent
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
		assert node_pos not in self.agents, f"Node position {node_pos} already had an agent: {self.agents[node_pos]}"
		self.agents[node_pos] = agent

		# Then update the caches
		self.cache.empty_nodes.remove(node_pos)
		self.update_unsatisfied_nodes_cache(node_pos)

	def remove_agent(self, node_pos: NodePos) -> Agent:
		"""
		Removes an agent at node `node_pos`.

		The node at `node_pos` *must* have an agent
		"""

		# Remove it from the graph
		agent = self.agents.pop(node_pos, None)
		assert agent is not None, f"Node position {node_pos} did not have agent"

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
		agents = list(agent_weights.keys())
		agent_weights = list(weight / weights_sum for weight in agent_weights.values())

		# Then for each node, select either empty or a random agent
		# TODO: Maybe optimize cases when `empty_cache` is high?
		for node_pos in self.graph.nodes:
			# If we're meant to be empty, continue to the next node
			if self.random_state.random() < empty_chance:
				continue

			# Select a random agent
			agent = self.random_state.choice(agents, p=agent_weights)

			# Then add it to the map
			self.agents[node_pos] = agent
			self.cache.empty_nodes.remove(node_pos)

			# Then update our agent type
			if self.agent_ty is None:
				self.agent_ty = type(agent)
			else:
				assert self.agent_ty == type(agent), "Cannot use agents of different types in the same graph!"

		# At the end, update the remaining caches
		self.update_unsatisfied_nodes_cache_multiple(self.graph.nodes)

	def agent_average_satisfaction(self) -> float | None:
		"""
		Returns the average satisfaction of all agents.

		If there are no agents, returns `None`
		"""

		if len(self.agents) == 0:
			return None

		return sum(self.agent_satisfaction(node_pos) for node_pos in self.agents.keys()) / len(self.agents)

	def agent_satisfaction(self, node_pos: NodePos) -> float | None:
		"""
		Returns the satisfaction of an agent, from 0.0 to 1.0.

		If the node doesn't have an agent, returns `None`
		"""

		# If the node is empty, it doesn't have a satisfaction
		agent = self.agents.get(node_pos, None)
		if agent is None:
			return None

		# Else count all neighbors that aren't empty
		neighbors: list[Agent] = []
		for neighbor_pos in self.graph.adj[node_pos]:
			neighbor_agent = self.agents.get(neighbor_pos, None)
			if neighbor_agent is None:
				continue

			neighbors.append(neighbor_agent)

		return agent.satisfaction(neighbors)

	def agent_satisfied(self, node_pos: NodePos) -> bool | None:
		"""
		Returns if an agent is satisfied
		"""

		# Get the satisfaction of the agent
		satisfaction = self.agent_satisfaction(node_pos)
		if satisfaction is None:
			return None

		return satisfaction >= self.agent_ty.threshold()

	def agent_img(self) -> list[list[Tuple[int, int, int]]]:
		"""
		Returns an image of all agents in the graph
		"""
		img = [[(0, 0, 0) for _ in range(self.size[0])] for _ in range(self.size[1])]
		for node_pos in self.graph.nodes:
			agent = self.agents.get(node_pos, None)
			img[node_pos[1]][node_pos[0]] = agent.color() if agent is not None else (0.5, 0.5, 0.5)

		return img

	def satisfaction_img(self) -> list[list[Tuple[int, int, int]]]:
		"""
		Returns an image of the satisfaction of all agents in the graph
		"""
		img = [[(0, 0, 0) for _ in range(self.size[0])] for _ in range(self.size[1])]
		for node_pos in self.graph.nodes:
			satisfaction = self.agent_satisfaction(node_pos)
			satisfied = self.agent_satisfied(node_pos)
			threshold = self.agent_ty.threshold()

			match satisfied:
				case None : color = (0.5, 0.5, 0.5)
				case True : color = (0.0, (satisfaction - threshold) / (1 - threshold), 0.0)
				case False: color = (1.0 - satisfaction / threshold, 0.0, 0.0)

			img[node_pos[1]][node_pos[0]] = color

		return img

	def draw(self, fig: plt.Figure):
		"""
		Draws the graph using standard `networkx` draw calls
		"""

		node_pos = { node_pos: node_pos for node_pos in self.graph.nodes()}
		agents = (self.agents.get(node_pos, None) for node_pos in self.graph.nodes)
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
		if self.debug.sanity_check_caches:
			self.update_unsatisfied_nodes_cache_multiple(self.graph.nodes)
			for node_pos in self.graph.nodes:
				match self.agent_satisfied(node_pos):
					case True | None: assert node_pos not in self.cache.unsatisfied_nodes, f"Node {node_pos} was satisfied, but present in unsatisfied cache"
					case False      : assert node_pos     in self.cache.unsatisfied_nodes, f"Node {node_pos} wasn't satisfied, but not present in unsatisfied cache"

				if node_pos not in self.agents:
					assert node_pos in self.cache.empty_nodes, f"Node {node_pos} was empty, but not present in empty cache"

		# Move all the current unsatisfied to another place
		# Note: We choose from both the empty nodes, as well as the unsatisfied, as they'll
		#       all be moving in a second, so we won't get duplicates.
		empty_nodes = list(self.cache.empty_nodes | self.cache.unsatisfied_nodes)
		self.random_state.shuffle(empty_nodes)
		self.move_agents(dict(zip(self.cache.unsatisfied_nodes, empty_nodes)))

		# We've reached equilibrium if there are no unsatisfied nodes
		# Note: If enabled, we also sanity check that all nodes are satisfied.
		reached_equilibrium = len(self.cache.unsatisfied_nodes) == 0
		if self.debug.check_equilibrium and reached_equilibrium:
			for node_pos in self.graph.nodes:
				satisfied = self.agent_satisfied(node_pos)
				assert satisfied is None or satisfied, f"Node {node_pos} wasn't satisfied after reaching equilibrium"

		return reached_equilibrium
