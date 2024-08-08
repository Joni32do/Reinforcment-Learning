import networkx as nx
import plotly.graph_objects as go
import numpy as np

G = nx.grid_2d_graph(10, 8)

def heuristic(a, b):
    # Manhattan distance on a square grid
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


start = (0, 0)
goal = (9, 7)

path = nx.astar_path(G, start, goal, heuristic=heuristic)
pos = {node: (node[0], node[1]) for node in G.nodes()}
edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# Highlight the final path

path_x = []
path_y = []
for node in path:
    x, y = pos[node]
    path_x.append(x)
    path_y.append(y)

path_trace = go.Scatter(
    x=path_x, y=path_y,
    mode='lines+markers',
    line=dict(color='red', width=4),
    marker=dict(color='red', size=10),
    hoverinfo='text')

fig = go.Figure(data=[edge_trace, node_trace, path_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=20),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
)

fig.show()
