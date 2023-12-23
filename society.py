import os
import networkx as nx
import matplotlib.pyplot as plt
import random
import math

# Parameters
percent_edge_removal = 0.1
percent_edge_addition = 0.01
local_max_app_adoption_rate = 0.05
global_max_app_adoption_rate = 0.02
base_distress_rate = 0.02
base_recovery_rate = 0.001
app_distress_recovery_rate = 0.05
base_talking_rate = 0.01
app_talking_rate = 0.1
trust_rate = 0.01
app_communication = True
willingness_rate = 0.8
app_recovery_stop_rate = 0.5

# Create the 'draw' directory
if not os.path.exists('draw'):
    os.makedirs('draw')

# Willingness to try the app while in distress (consistency)
def willingness(cycle):
    return willingness_rate**cycle


# Consistency of using the app
def consistency_sigmoid(count):
    return 1 - 1/(1 + math.exp(-((count/3))))


# Sigmoid to get percentage of local persuasion (liking)
def local_app_sigmoid(count):
    return local_max_app_adoption_rate / (1 + math.exp(-(count - 4)))


# Percentage of global persuasion (social proof)
def global_app_percentage(percentage):
    return global_max_app_adoption_rate * percentage


# Read Ego Files and return graph
def read_ego_file(files):
    G = nx.Graph()
    for f in files:
        print(f)
        for line in open(f):
            e1, es = line.split(':')
            es = es.split()
            for e in es:
                if e == e1:
                    continue
                G.add_edge(int(e1.strip()), int(e.strip()))
                # Initialize some nodes to have distress
                if random.random() < base_distress_rate:
                    G.nodes[int(e)]["distress"] = True
                    # Initialize some distressed people to have the app
                    if random.random() < local_app_sigmoid(0):
                        G.nodes[int(e)]["app"] = True
                    else:
                        G.nodes[int(e)]["app"] = False
                else:
                    G.nodes[int(e)]["distress"] = False
                    G.nodes[int(e)]["app"] = False
                # Initialize lists for talking/trusted people
                G.nodes[int(e)]["talking"] = []
                G.nodes[int(e)]["trust"] = []
                G.nodes[int(e)]["distress_count"] = 0
                G.nodes[int(e)]["app_count"] = 0
    return G


# Draw the graph and save to directory
def draw_graph(G, iteration):
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    C = (G.subgraph(c) for c in nx.connected_components(G))
    for g in C:
        c = []
        for e in g.nodes:
            if g.nodes[e]["distress"]:
                if g.nodes[e]["app"]:
                    c.append("purple")
                else:
                    c.append("red")
            else:
                if g.nodes[e]["app"]:
                    c.append("green")
                else:
                    c.append("blue")
        # c = ["red" if g.nodes[e]["app"] else "blue" for e in g.nodes]
        nx.draw(g, pos, node_size=40, node_color=c, with_labels=False)
        # nx.draw(g, node_size=40, node_color=c, with_labels=False)

    plt.savefig("draw/" + str(iteration) + ".png")
    plt.clf()


# # Files in egonet directory
# files = glob.glob("egonets/*.egonet")
# # Shuffle
# random.shuffle(files)
files = ["egonets/2895.egonet", "egonets/17951.egonet"]

# Get graph
G = read_ego_file(files[:2])
# Remove random edges
G.remove_edges_from(random.sample(list(G.edges), int(len(list(G.edges)) * percent_edge_removal)))
# Add random edges
for _ in range(int(len(G.nodes) * percent_edge_addition)):
    e = random.sample(list(G.nodes), 2)
    G.add_edge(e[0], e[1])

# Print clustering
clustering = nx.average_clustering(G)
print("Clustering:", clustering)

# Iterations
# PyGraphViz crashes after 253 epochs
iterations = 250
draw_graph(G, 0)

start_count = 0
for e in G.nodes:
    if G.nodes[e]["distress"]:
        start_count += 1
distress_count = [start_count]
user_count = []
for i in range(iterations):
    print("Iteration:", i)

    # Count the number of app users and get users as a list
    global_count = 0
    app_users = []
    for e in G.nodes:
        global_count += int(G.nodes[e]["app"])
        if G.nodes[e]["app"]:
            app_users.append(e)
    user_count.append(global_count)
    distress_c = 0
    # Loop through all nodes
    for e in G.nodes:
        count = 0
        # Change distress variable based on parameters
        if G.nodes[e]["distress"] and random.random() < base_recovery_rate * willingness(G.nodes[e]["distress_count"]):
            G.nodes[e]["distress"] = False
            G.nodes[e]["distress_count"] = 0
            if random.random() < app_recovery_stop_rate:
                G.nodes[e]["app"] = False
        elif app_communication and G.nodes[e]["distress"] and G.nodes[e]["app"] and random.random() < app_distress_recovery_rate * willingness(G.nodes[e]["distress_count"]):
            G.nodes[e]["distress"] = False
            G.nodes[e]["distress_count"] = 0
            if random.random() < app_recovery_stop_rate:
                G.nodes[e]["app"] = False
        elif not G.nodes[e]["distress"] and random.random() < base_distress_rate:
            G.nodes[e]["distress"] = True

        # Percentage that someone will stop using the app
        if random.random() < consistency_sigmoid(G.nodes[e]["app_count"]):
            G.nodes[e]["app"] = False

        # App adoption rate based on local and global persuasion (liking and social proof)
        if G.nodes[e]["distress"] and random.random() < (local_app_sigmoid(len(G.nodes[e]["trust"])) + global_app_percentage(global_count / len(G.nodes))) * willingness(G.nodes[e]["distress_count"]):
            G.nodes[e]["app"] = True

        # Update trusted people list
        for t in G.nodes[e]["talking"]:
            if random.random() < trust_rate * willingness(G.nodes[e]["distress_count"]):
                G.nodes[e]["trust"].append(t)

        # Update talking people list
        for n in G.neighbors(e):
            if random.random() < base_talking_rate * willingness(G.nodes[e]["distress_count"]):
                G.nodes[e]["talking"].append(n)

        # If there is an app communication feature, add app users to talking list
        if app_communication:
            for a in app_users:
                if random.random() < app_talking_rate * willingness(G.nodes[e]["distress_count"]):
                    G.nodes[e]["talking"].append(a)

        if G.nodes[e]["distress"]:
            G.nodes[e]["distress_count"] += 1

        if G.nodes[e]["app"] or len(G.nodes[e]["talking"]) > 0 or len(G.nodes[e]["trust"]) > 0:
            G.nodes[e]["distress_count"] = 0

        if G.nodes[e]["distress"]:
            G.nodes[e]["distress_count"] += 1
            distress_c += 1
        if G.nodes[e]["app"]:
            G.nodes[e]["app_count"] += 1
    distress_count.append(distress_c)

    draw_graph(G, i + 1)

user_counter = 0
for e in G.nodes:
    user_counter += int(G.nodes[e]["app"])
user_count.append(user_counter)
plt.plot(range(len(distress_count)), distress_count, label="Distress")
plt.plot(range(len(distress_count)), user_count, label="User")
plt.legend()

plt.show()
