import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import shapely
from shapely import speedups
print('speedup available:', speedups.available)
if speedups.available:
    speedups.enable()
import sec
from edge_assigment import assign_edges
import time

point_coords = (45.745591, 4.871167) # latitude and longitude of Montplaisir-Lumière, Lyon (France)
distance = 1000 # in meters
buffer = 0.01 # 1 cm
distance_threshold = 30

print("loading buildings...")
B = ox.buildings_from_point(point_coords, distance=distance)
print("buildings loaded.")

B = ox.project_gdf(B) # pass from angular (lat, long) coords to planar coords
B = B.geometry.buffer(buffer)

def merge_and_convex(df):
    go = True
    length = len(df)
    i = 0
    print(i, length)
    while go:
        i += 1
        df = gpd.GeoDataFrame(geometry=list(df.unary_union)).convex_hull
        print(i, len(df))
        if len(df) == length:
            go = False
        else:
            length = len(df)
    return df

print('merging...')
B = merge_and_convex(B)
print('finished merging')

nodes = B.centroid

print('assigning edges...')
t1 = time.time()
edges, weights = assign_edges(B, distance_threshold)
print('finished assigning edges in %s seconds' %(time.time() - t1))

