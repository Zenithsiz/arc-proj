"""
ARC Project
"""

# Imports
import networkx as nx
import matplotlib.pyplot as plt

# Graph
graph = nx.grid_2d_graph(10, 10)


# Visualize graph
pos = {(x,y):(y,-x) for x,y in graph.nodes()}
with plt.ion():
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	nx.draw(graph, pos=pos, ax=ax)

	fig.tight_layout()
	plt.show(block = True)
