# This is a naive first implementation of the game Next Station: Tokyo

import networkx as nx
import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt

G = nx.Graph()

# Symmetric Station numbers

N_CORNER = 1
N_OUTER = 6
N_INNER = 4
N_CENTER = 8

# Directions 
DIRECTIONS = [N, E, S, W] = [0, 1, 2, 3]

# Stations
STATIONS = []
CORNERS = []
for d in DIRECTIONS:
    pass

OUTER = []
for d in DIRECTIONS:
    pass

INNER = []
for d in DIRECTIONS:
    pass

CENTER = []

SECTORS = []
n_stations = 0

for sector in SECTORS:
    for station in sector:
        STATIONS.append(station)

tracks = []





