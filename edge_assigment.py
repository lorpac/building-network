import pickle
import os
import numpy as np
import shapely
from copy import deepcopy
import geopandas as gpd

def pairwise_weight(i, j, b_i, b_j, nodes, distance_threshold):
    """Defines pairwise weight between two buildings i and j.
    
    Parameters
    ----------
    i : int
        building i's index
    j : int
        building j's index
    b_i : Shapely Polygon
        building i
    b_j : Shapely Polygon
        building j
    nodes : GeoPandas GeoSeries
        GeoSeries containing the centroids of all buildings
    distance_threshold : int
        distance threshold for connection
    
    Returns
    -------
    float
        pairwise weight
    """

    n_i = nodes.iloc[i]
    n_j = nodes.iloc[j]
    b_i_buffered = b_i.buffer(distance_threshold)
    b_j_buffered = b_j.buffer(distance_threshold)
    area1 = b_i_buffered.intersection(b_j).area
    area2 = b_j_buffered.intersection(b_i).area
    wij = area1 + area2
    return wij



def assign_edges(B, distance_threshold, step=None):
    """Defines pairwise weight between two buildings i and j.
    
    Parameters
    ----------
    B : GeoPandas GeoSeries
        GeoSeries containing all buildings
    distance_threshold : int
        distance threshold for connection
    step : int or None, optional
        size of grid to use to check buildings adjency. If None, the distance_threshold value is used (default is None)
    
    Returns
    -------
    set
        set of edges
    dict
        dictionary of edge weights
    """

    if not step:
        step = distance_threshold

    k = int(
        np.ceil(distance_threshold / step)
    )  # number of cell lengths in a distance threshold


    nodes = B.centroid

    # initialize grid
    x0 = min([b.bounds[0] for b in B])
    y0 = min([b.bounds[1] for b in B])
    max_x = max([b.bounds[2] for b in B])
    max_y = max([b.bounds[3] for b in B])

    delta_x = max_x - x0
    delta_y = max_y - y0

    n_x = int(np.ceil(delta_x / step))
    n_y = int(np.ceil(delta_y / step))

    C = [[set() for j in range(n_y)] for i in range(n_x)]

    # initialize dictionary of cells touched by each building
    Cb = {i: set() for i in range(len(B))}


    def enclosing_rectangle(building):
        (minx, miny, maxx, maxy) = building.bounds
        return (minx, miny, maxx, maxy)


    def cell_of_point(x, y, step, x0, y0):
        x -= x0
        y -= y0
        i = int(np.floor(x / step))
        j = int(np.floor(y / step))
        return i, j


    def assign_building_cells(b, index, step, x0, y0, C, Cb):
        (minx, miny, maxx, maxy) = enclosing_rectangle(b)
        imin, jmin = cell_of_point(minx, miny, step, x0, y0)
        imax, jmax = cell_of_point(maxx, maxy, step, x0, y0)
        for i in range(imin, imax + 1):
            for j in range(jmin, jmax + 1):
                C[i][j].add(index)
                Cb[index].add((i, j))


    # assign cells to each buildings
    for index, b in enumerate(B):
        assign_building_cells(b, index, step, x0, y0, C, Cb)

    # edge assignment
    edges = set([])
    weights = {}
    for i, b_i in enumerate(B):
        node_i = nodes[i]
        neighbors = []
        # Cb_i = Cb[i]
        Cb_i = np.array(list(Cb[i]))
        max_i = max(Cb_i[:,0])
        max_j = max(Cb_i[:,1])
        min_i = min(Cb_i[:,0])
        min_j = min(Cb_i[:,1])

        potential_neighbors = set([])

        for ci, cj in Cb_i:
            potential_neighbors = potential_neighbors.union(C[ci][cj])
        
        for cl in range(min_i - k, max_i + k + 1):
            for cm in range(min_j - k, max_j + k + 1):
                if (
                    (not (cl == ci and cm == cj))
                    and cl < n_x
                    and cm < n_y
                    and cl >= 0
                    and cm >= 0
                ):
                    potential_neighbors = potential_neighbors.union(C[cl][cm])

        potential_neighbors_reduced = []
        for j in potential_neighbors:
            if i != j:
                b_j = B[j]
                dist = b_i.distance(b_j)
                if dist < distance_threshold:
                    potential_neighbors_reduced.append(j)

        for j in potential_neighbors_reduced:
            b_j = B[j]
            node_j = nodes[j]
            segment = shapely.geometry.LineString([list(node_i.coords)[0], list(node_j.coords)[0]])
            valid = True
            for q in potential_neighbors_reduced:
                if q == j: continue
                elif valid:
                    b_q = B[q]
                    if not b_q.intersection(segment).is_empty:
                        valid = False
                        break
            if valid and (j, i) not in edges:
                edges.add((i, j))
                wij = pairwise_weight(i, j, b_i, b_j, nodes, distance_threshold)
                weights[(i, j)] = wij

    return edges, weights