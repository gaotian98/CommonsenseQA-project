import pickle
import sys
import networkx as nx
from collections import OrderedDict

mode = sys.argv[1]
if mode == 'train':
	nodes = pickle.load(open('train.graph.adj.pk', 'rb'))
elif mode == 'dev':
	nodes = pickle.load(open('dev.graph.adj.pk', 'rb'))
elif mode == 'test':
	nodes = pickle.load(open('test.graph.adj.pk', 'rb'))
cpnet = nx.read_gpickle('../../cpnet/conceptnet.en.pruned.graph')

for g in nodes:
	G = nx.MultiDiGraph()
	s = {}
	for i in range(len(g['concepts'])):
		if g['amask'][i] == True or g['qmask'][i] == True:
			G.add_edge(-1, g['concepts'][i])
		for j in range(i + 1, len(g['concepts'])):
			n1 = g['concepts'][i]
			n2 = g['concepts'][j]
			if cpnet.has_edge(n1, n2):
				for e in cpnet[n1][n2].values():
					G.add_edge(n1, n2)
			if cpnet.has_edge(n2, n1):
				for e in cpnet[n2][n1].values():
					G.add_edge(n2, n1)

	pr = nx.pagerank(G, alpha=0.85, personalization={-1: 1})
	arr = []
	for k in g['cid2score'].keys():
		arr.append((k, g['cid2score'][k] + pr[k]))
	g['cid2score'] = OrderedDict(sorted(arr, key=lambda x: -x[1]))
if mode == 'train':
	pickle.dump(nodes, open('train.graph.pr.pk', 'wb'))
elif mode == 'dev':
	pickle.dump(nodes, open('dev.graph.pr.pk', 'wb'))
elif mode == 'test':
	pickle.dump(nodes, open('test.graph.pr.pk', 'wb'))
