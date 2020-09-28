import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import shapely
from shapely import speedups
if speedups.available:
    speedups.enable()
import sec
from edge_assigment import assign_edges
import os
import json
from copy import deepcopy

plt.ioff()

class Building():
    
    def __init__(self, config_file=None, point_coords=None, place_name=None, distance=1000, distance_threshold=30):

        def load_config(filename):
            with open(filename) as f:
                config = json.load(f)
            return config

        if config_file:
            config = load_config(config_file)
            self.__dict__ = config
        else:
            self.point_coords = point_coords
            self.place_name = place_name
            if self.point_coords:
                self.distance = distance
            self.distance_threshold = distance_threshold
        self.config = deepcopy(self)
        self.is_downloaded = False
        self.is_merged = False
        self.nodes_assigned = False
        self.edges_assigned = False
        self.net_assigned = False

    def save_config(self, filename="config.json"):
        with open(filename, 'w') as file:
            json.dump(self.config.__dict__, file, indent=4)

    def download_buildings(self):
        if self.point_coords:
            self.buildings = ox.footprints.footprints_from_point(self.point_coords, distance=self.distance)
            self.buildings = ox.project_gdf(self.buildings)
        else:
            self.buildings = ox.footprints.footprints_from_place(self.place_name)
            self.buildings = ox.project_gdf(self.buildings)
        self.is_downloaded = True
    
    def plot_buildings(self, fc='black', ec='gray', figsize=(30, 30), save=True, imgs_folder = ".temp", filename="buildings", file_format='png', dpi=300):
        if self.is_merged:
            raise Exception("merge_and_convex() already performed on Building. Please use plot_merged_buildings()")
        fig, ax = ox.plot_shape(self.buildings, fc=fc, ec=ec, figsize=figsize)
        if save:
            ox.settings.imgs_folder = imgs_folder
            ox.save_and_show(fig, ax, save=True, show=False, close=True, filename=filename, file_format=file_format, dpi=dpi, axis_off=True)
        plt.close()
    
    def merge_and_convex(self, buffer=0.01):
        if self.is_merged:
            raise Exception("merge_and_convex() already performed on Building.")

        self.buildings = self.buildings.geometry.buffer(buffer)
        go = True
        length = len(self.buildings)
        i = 0
        print(i, length)
        while go:
            i += 1
            self.buildings = gpd.GeoDataFrame(geometry=list(self.buildings.unary_union)).convex_hull
            print(i, len(self.buildings))
            if len(self.buildings) == length:
                go = False
            else:
                length = len(self.buildings)
        self.is_merged = True
        self.buildings_df = gpd.GeoDataFrame(geometry=[build for build in self.buildings])

    def plot_merged_buildings(self, color='lightgray', edgecolor='black', figsize=(30, 30), save=True, imgs_folder = ".temp", filename="merged", file_format='png', show=True):
        if not self.is_merged:
            raise Exception("Please run merge_and_convex() before.")
        self.buildings.plot(figsize=figsize, color=color, edgecolor=edgecolor)
        plt.axis('off')
        plt.tight_layout()
        if save:
            os.makedirs(imgs_folder, exist_ok=True)
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close()

    def assign_nodes(self):
        if not self.is_merged:
            raise Exception("Please run merge_and_convex() before.")
        self.nodes = self.buildings.centroid
        col = [1 for build in self.buildings] + [2 for node in self.nodes]
        self.nodes_df = gpd.GeoDataFrame(col, geometry=[build for build in self.buildings] + [node for node in self.nodes], columns=['color'])
        self.nodes_assigned = True

    def assign_edges_weights(self):
        distance_threshold = self.distance_threshold 
        if not self.is_merged:
            raise Exception("Please run merge_and_convex() before.")
        self.edges, self.weights = assign_edges(self.buildings, distance_threshold=distance_threshold)

        nodes=self.nodes
        edges_segment = []
        for u, v in self.edges:
            node_u = nodes.iloc[u]
            node_v = nodes.iloc[v]
            edge_segment = shapely.geometry.LineString([list(node_u.coords)[0], list(node_v.coords)[0]])
            edges_segment.append(edge_segment)

        colors = [1 for build in self.buildings] + [2 for edge in edges_segment] + [3 for node in self.nodes]

        self.edges_df = gpd.GeoDataFrame(colors, geometry = [build for build in self.buildings] + edges_segment + 
                            [node for node in self.nodes], columns=['color'])
        self.edges_assigned = True
    
    def plot_nodes(self, figsize=(30, 30), colors=['lightgray', 'black'], markersize=0.1, save=True, imgs_folder = ".temp", filename="nodes", file_format='png', show=True):
        cm = ListedColormap(colors, N=len(colors))
        self.nodes_df.plot(figsize=figsize, column='color', markersize=markersize, cmap=cm)
        plt.axis('off')
        plt.tight_layout()
        if save:
            os.makedirs(imgs_folder, exist_ok=True)
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close()

    def plot_edges(self, figsize=(30, 30), colors=['lightgray', 'black'], markersize=1, linewidth=0.5, save=True, imgs_folder = ".temp", filename="edges", file_format='png', show=True):
        cm = ListedColormap(colors, N=len(colors))
        self.edges_df.plot(figsize=figsize, column='color', markersize=markersize, linewidth=linewidth, cmap=cm)
        plt.axis('off')
        plt.tight_layout()
        if save:
            os.makedirs(imgs_folder, exist_ok=True)
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close()

    def assign_network(self):
        G = nx.Graph()
        pos = {}
        for index, node in enumerate(self.nodes):
            G.add_node(index)
            pos[index] = list(node.coords)[0]
        for u, v in self.edges:
            G.add_edge(u, v)
            
        nx.set_edge_attributes(G, self.weights, name='weight')

        degrees_zero = []
        for node in G.nodes:
            k = nx.degree(G, node)
            if k == 0:
                degrees_zero.append(node)

        G.remove_nodes_from(degrees_zero)
        self.network = G        
        self.network_df = gpd.GeoDataFrame(geometry=[build for build in self.buildings])
        self.network_pos = pos
        self.net_assigned = True
        self.assign_node_color()
        self.assign_edge_color()

    def assign_node_color(self, colors = ['blue', 'cyan', 'greenyellow', 'yellow', 'orange', 'red']):
        G = self.network
        neigh_watch_sharp_dict = {}
        node_color = []
        for index, node in enumerate(G.nodes):
            k = nx.degree(G, node)
            w = nx.degree(G, node, weight='weight')
            nw = w / k
            
            if nw < 500:
                neigh_watch_sharp_dict[node] = 0
                node_color.append(colors[0])
            elif nw < 1000:
                neigh_watch_sharp_dict[node] = 1
                node_color.append(colors[1])
            elif nw < 1500:
                neigh_watch_sharp_dict[node] = 2
                node_color.append(colors[2])
            elif nw < 2000:
                neigh_watch_sharp_dict[node] = 3
                node_color.append(colors[3])
            elif nw < 2500:
                neigh_watch_sharp_dict[node] = 4
                node_color.append(colors[4])
            else:
                neigh_watch_sharp_dict[node] = 5
                node_color.append(colors[5])
        self.node_color = node_color
        self.neigh_watch_sharp_dict = neigh_watch_sharp_dict
        self.colors_nodes = colors

    def assign_edge_color(self, colors = ['blue', 'cyan', 'greenyellow', 'yellow', 'orange', 'red']):
        G = self.network
        edge_color = []

        for u, v in G.edges:
            wij = G.get_edge_data(u, v)['weight']
            if wij < 500: edge_color.append(colors[0])
            elif wij < 1000:  edge_color.append(colors[1])
            elif wij < 1500:  edge_color.append(colors[2])
            elif wij < 2000:  edge_color.append(colors[3])
            elif wij < 2500:  edge_color.append(colors[4])
            else: edge_color.append(colors[5])
        
        self.edge_color = edge_color
        self.colors_edges = colors


    def plot_net(self, figsize=(30, 30), save=True, imgs_folder = ".temp", filename="net", file_format='png', show=True, style="node_color"):
        
        G = self.network
        
        weights_values = [self.weights[(u, v)] for u, v in G.edges]
        
        fig, ax  = plt.subplots(figsize=figsize)
        base = self.buildings_df.plot(ax=ax, color='gray', alpha=0.2)
        pos = self.network_pos
        if style == "node_color":
            node_color = self.node_color
            nx.draw_networkx_nodes(G, pos=pos, with_labels=False, node_color=node_color, edgecolors='k', ax=ax)
            nx.draw_networkx_edges(G, pos=pos, width=[w * (2) ** (-8) for w in weights_values], ax=ax)
        elif style == "edge_color":
            edge_color = self.edge_color
            nx.draw_networkx_nodes(G, pos=pos, with_labels=False, edgecolors='k', node_color='w')
            nx.draw_networkx_edges(G, pos=pos, width=[w * (2) ** (-8) for w in weights_values], edge_color=edge_color)
        plt.axis('off')
        plt.tight_layout()
        if save:
            os.makedirs(imgs_folder, exist_ok=True)
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close() 

    def plot_buildings_color(self, figsize=(30, 30), save=True, imgs_folder = ".temp", filename="buildings_color",
        file_format='png', show=True):
            
            G = self.network
            node_color = self.node_color
            colors = self.colors_nodes
            neigh_watch_sharp_dict = self.neigh_watch_sharp_dict
            buildings_df = self.buildings_df

            cm = ListedColormap(colors, N=len(colors))
            buildings_df_colors = gpd.GeoDataFrame(columns=['geometry', 'nw_sharp'])
            for i, row in buildings_df.iterrows():
                if i in G.nodes:
                    nw_sharp = neigh_watch_sharp_dict[i]
                    geometry = row['geometry']
                    buildings_df_colors.loc[i] = [geometry, nw_sharp] 
            fig, ax = plt.subplots(figsize=figsize)
            base = buildings_df.plot(ax=ax, color='gray', alpha=0.2)
            buildings_df_colors.plot(ax=base, column='nw_sharp', cmap=cm, vmin=0,
                                vmax=5)
            buildings_df_colors.boundary.plot(ax=base, color='k')
            ax.axis('off')

            plt.tight_layout()
            if save:
                plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
            if show:
                plt.show()
            plt.close()

    def plot_nodes_legend(self, save=True, imgs_folder = ".temp", filename="legend_nodes" , file_format='png', show=True):
        colors = self.colors_nodes
        Glegend = nx.Graph()
        for n in range(1, 7):
            Glegend.add_node(str(n))

        poslegend = {}
        for i, node in enumerate(Glegend.nodes):
            poslegend[node] = np.array([0, i + 0.15])

        plt.figure() 
        nx.draw_networkx_nodes(Glegend, poslegend, node_color=colors,edgecolors='k')
        plt.axis('off')
        plt.xlim(-0.2, 3)
        plt.ylim(-0.2, 6)
        plt.rc('text', usetex=True)
        plt.rc('font', family='calibri')
        plt.text(0.2, 0, 'w/k $<$ 500 $m^2$',fontsize=16)
        plt.text(0.2, 1, '500 ${m}^2$ $\leq$ w/k $<$ 1000 ${m}^2$',fontsize=16)
        plt.text(0.2, 2, '1000 ${m}^2$ $\leq$ w/k $<$ 1500 ${m}^2$',fontsize=16)
        plt.text(0.2, 3, '1500 ${m}^2$ $\leq$ w/k $<$ 2000 ${m}^2$',fontsize=16)
        plt.text(0.2, 4, '2000 ${m}^2$ $\leq$ w/k $<$ 2500 ${m}^2$',fontsize=16)
        plt.text(0.2, 5, 'w/k $>$ 2500 ${m}^2$',fontsize=16)
        if save:
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close()

    def plot_edges_legend(self, save=True, imgs_folder = ".temp", filename="legend_edges" , file_format='png', show=True):
        colors = self.colors_edges
        texts = ['$w_{ij} \  <500\ m^2$',
         '$500\ m^2 \leq \ w_{ij} \  < 1000\ m^2$',
         '$1000\ m^2 \leq \ w_{ij} \  < 1500\ m^2$',
         '$1500\ m^2 \leq \ w_{ij} \  < 2000\ m^2$',
         '$2000\ m^2 \leq \ w_{ij} \  < 2500\ m^2$',
         '$w_{ij} \  \geq 2500\ m^2$']
        fig = plt.figure()
        plt.rc('text', usetex=True)
        plt.rc('font', family='calibri')
        for index, c in enumerate(colors):
            subf = int('61%s' %(index + 1))
            ax1 = fig.add_subplot(subf)
            ax1.add_patch(patches.Rectangle((0.1, 0), 0.15, .4, facecolor=c, edgecolor='k',
                                            linewidth=1.5))
            ax1.axis('off')
            ax1.text(0.3, 0, texts[index], fontsize=16)
        plt.axis('off')
        if save:
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close()
