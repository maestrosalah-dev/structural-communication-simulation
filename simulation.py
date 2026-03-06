import networkx as nx
import numpy as np
import random

# PARAMETERS
N = 3000
M = 4
TIMESTEPS = 50

symbolic_flow = 0.6
threshold = 0.5
rewire_alpha = 0.3

# INITIAL NETWORK
G = nx.barabasi_albert_graph(N, M)

# INITIAL OPINIONS
for node in G.nodes():
    G.nodes[node]['opinion'] = random.uniform(-1,1)

def compute_tension(G):
    tensions = []
    for i,j in G.edges():
        oi = G.nodes[i]['opinion']
        oj = G.nodes[j]['opinion']
        tensions.append(abs(oi - oj))
    return np.mean(tensions)

def polarization(G):
    opinions = [G.nodes[i]['opinion'] for i in G.nodes()]
    return np.var(opinions)

for t in range(TIMESTEPS):

    T = compute_tension(G)

    for node in list(G.nodes()):

        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue

        if random.random() < symbolic_flow * T:

            influence = np.mean([G.nodes[n]['opinion'] for n in neighbors])

            G.nodes[node]['opinion'] += 0.1*(influence - G.nodes[node]['opinion'])

    # EDGE REWIRING
    for i,j in list(G.edges()):

        oi = G.nodes[i]['opinion']
        oj = G.nodes[j]['opinion']

        tension = abs(oi - oj)

        if tension > threshold and random.random() < rewire_alpha:

            G.remove_edge(i,j)

            candidates = [n for n in G.nodes()
                          if abs(G.nodes[n]['opinion'] - oi) < 0.2 and n != i]

            if candidates:
                new = random.choice(candidates)
                G.add_edge(i,new)

    print("Step:",t,
          "Tension:",compute_tension(G),
          "Polarization:",polarization(G))
