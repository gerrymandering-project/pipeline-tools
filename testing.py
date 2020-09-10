import networkx as nx

G = nx.Graph()

G.add_edge('a', 'b')
G.add_edge('a', 'c')
G.add_edge('b', 'c')
G.add_edge('c', 'd')

print(list(G.neighbors('a')))
print('hi')