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
import imageio
import pickle
import pandas as pd
mpl.rc('text', usetex=False)

plt.ioff()

class Building():

    @staticmethod
    def generate(load=False, filepath=".temp/B.p", point_coords=None, config_file=None, place_name=None, distance=1000, distance_threshold=30):
        if load:
            return pickle.load(open(filepath, "rb"))
        else:
            return Building(point_coords=point_coords, config_file=config_file, place_name=place_name, distance=distance, distance_threshold=distance_threshold)
    
    def __init__(self, point_coords=None, config_file=None, place_name=None, distance=1000, distance_threshold=30):

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

    def download_buildings(self, save=False, folder_path=".temp", filename="buildingsOSM.shp"):
        if self.point_coords:
            self.buildings = ox.footprints.footprints_from_point(self.point_coords, distance=self.distance)
            self.buildings = ox.project_gdf(self.buildings)
        else:
            self.buildings = ox.footprints.footprints_from_place(self.place_name)
            self.buildings = ox.project_gdf(self.buildings)
        self.is_downloaded = True
        self.downloaded_buildings = deepcopy(self.buildings)
        if save:
            os.makedirs(folder_path, exist_ok=True)
            gdf_save = self.buildings.applymap(lambda x: str(x) if isinstance(x, list) else x)
            gdf_save.drop(labels='nodes', axis=1).to_file(os.path.join(folder_path, filename))
    
    def plot_buildings(self, color='black', edgecolor='gray', figsize=(30, 30), save=True, imgs_folder = ".temp", filename="buildings", file_format='png', dpi=300, show=True):
        self.downloaded_buildings.plot(color=color, figsize=figsize, edgecolor=edgecolor)
        plt.axis("off")
        plt.tight_layout()
        if save:
            os.makedirs(imgs_folder, exist_ok=True)
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close()
    
    def plot_buildings_function(self, selected_functions=None, imgs_folder=".temp", filename="buildings_function", file_format='png', show=True, save=True, figsize=(30,30)):
        mpl.rc('text', usetex=False)
        downloaded_buildings =  gpd.GeoDataFrame(self.downloaded_buildings)

        building_functions = sorted(downloaded_buildings["building"].unique(), reverse=True)
        if not selected_functions:
            selected_functions = building_functions
        N = len(building_functions)
        bounds = np.arange(-.5, N)
        ticks = range(N)

        selected = downloaded_buildings[downloaded_buildings["building"].isin(selected_functions)]
        selected["function_number"] = [building_functions.index(f) for f in selected.building.values]

        fig = plt.figure(figsize=(35, 30))
        from matplotlib import gridspec
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                                width_ratios=[30, 1])
        
        ax0 = fig.add_subplot(spec[0])
        cmap = mpl.cm.tab20c
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ax1 = fig.add_subplot(spec[1])
        cbar = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                        norm=norm,
                                        orientation='vertical',
                                        ticks=ticks,
                                        drawedges=True)

        cbar.ax.set_yticklabels(building_functions)
        cbar.ax.tick_params(labelsize=25) 

        if selected_functions:
            downloaded_buildings.plot(color='lightgray', figsize=figsize, ax=ax0, edgecolor="k")
        selected.plot(column="function_number", figsize=figsize, ax=ax0, cmap=cmap, norm=norm, edgecolor="k")
        ax0.axis('off')

        
        plt.tight_layout()
        if save:
            os.makedirs(imgs_folder, exist_ok=True)
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close()

    def merge_and_convex(self, buffer=0.01, plot=False, imgs_folder=".temp", show=True, save=True, figsize=(30,30), status_to_file=False, status_to_console=True):
        if self.is_merged:
            raise Exception("merge_and_convex() already performed on Building.")

        self.buildings = self.buildings.geometry.buffer(buffer)
        go = True
        length = len(self.buildings)
        i = 0
        if plot:
            output_folder = os.path.join(imgs_folder, "merging_intermediates")
            self.plot_merged_buildings(imgs_folder=output_folder, filename=str(i), show=show, save=save, figsize=figsize)
        if status_to_console:
            print(i, length)
        if status_to_file:
            with open("merging_status", "w")  as f:
                f.write(str(i))
        while go:
            i += 1
            self.buildings = gpd.GeoDataFrame(geometry=list(self.buildings.unary_union)).convex_hull
            print(i, len(self.buildings))
            if len(self.buildings) == length:
                go = False
            else:
                if plot:
                    self.plot_merged_buildings(imgs_folder=output_folder, filename=str(i), show=show, save=save, figsize=figsize)
                if status_to_file:
                    with open("merging_status", "w")  as f:
                        f.write(str(i))
                length = len(self.buildings)
        self.is_merged = True
        self.buildings_df = gpd.GeoDataFrame(geometry=[build for build in self.buildings])

    def plot_merging_intermediates(self, imgs_folder=".temp", show=True, save=True, figsize=(30,30)):
        if not self.is_merged:
            print("merge_and_convex() had not be performed. merge_and_convex() is now being called.")
            self.merge_and_convex()
        output_folder = os.path.join(imgs_folder, "merging_intermediates")
        for i, b in enumerate(self.merging_intermediates):
            b.plot_merged_buildings(imgs_folder=output_folder, filename=str(i), show=show, save=save, figsize=figsize)

    def create_gif_merging(self, imgs_folder=".temp", fps=2, figsize=(30,30), filename='merging'):
        input_folder = os.path.join(imgs_folder, "merging_intermediates")
        try:
            os.listdir(input_folder)
        except FileNotFoundError:
            plot_merging_intermediates(imgs_folder=imgs_folder, figsize=figsize, show=False)
        
        filenames = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder), key= lambda x: int(x.split(".")[0]))]
        output_file = os.path.join(imgs_folder, filename + ".gif")

        with imageio.get_writer(output_file, mode='I', fps=fps) as writer:
            for f in filenames:
                image = imageio.imread(f)
                writer.append_data(image)
        
    
    def plot_merged_buildings(self, color='lightgray', edgecolor='black', figsize=(30, 30), save=True, imgs_folder = ".temp", filename="merged", file_format='png', show=True):
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
            self.buildings = gpd.GeoSeries(list(self.buildings.geometry), index=self.buildings.index)
            self.buildings_df = gpd.GeoDataFrame(geometry=[build for build in self.buildings], index=self.buildings.index)
        self.nodes = self.buildings.centroid
        col = [1 for build in self.buildings] + [2 for node in self.nodes]
        self.nodes_df = gpd.GeoDataFrame(col, geometry=[build for build in self.buildings] + [node for node in self.nodes], columns=['color'])
        self.nodes_assigned = True

    def assign_edges_weights(self, distance_threshold=None):
        if distance_threshold:
             self.distance_threshold = distance_threshold
        else:
            distance_threshold = self.distance_threshold 
        self.edges, self.weights = assign_edges(self.buildings, distance_threshold=distance_threshold)

        nodes=self.nodes
        edges_segment = []
        for u, v in self.edges:
            node_u = nodes[u]
            node_v = nodes[v]
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
        for index, node in self.nodes.items():
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

    def create_database(self, folder_path= ".temp", filename="database.csv"):

        G = self.network
        df = self.buildings_df
        neigh_watch_sharp_dict = {}
        area_sharp_dict = {}
        perimeter_sharp_dict = {}
        columns = ['id', 'k', 'w', 'w/k', 'area', 'perimeter', 'form_factor', 'neigh_watch_sharp', 'area_sharp', 'perimeter_sharp']

        area = df.area.values
        if len(area) > 50:
            max_area = sorted(area, reverse=True)[50]
        else:
            max_area = max(area)
        n_area_steps = 6
        step_area = max_area / n_area_steps ** 2 # quadratic steps
        area_steps = [step_area * (i ** 2) for i in range(1, n_area_steps + 1)] + [0] # EXTRA CLASS FOR BIGGER AREAS
        
        max_perimeter = 700
        step_perimeter = max_perimeter / 5
        perimeter_steps = [i * step_perimeter for i in range(1, 6)] + [0] # EXTRA CLASS FOR BIGGER PERIMETERS

        data = []
        for node in G.nodes:
            k = nx.degree(G, node)
            w = nx.degree(G, node, weight='weight')
            nw = w / k
            if self.neigh_watch_sharp_dict:
                neigh_watch_sharp = self.neigh_watch_sharp_dict[node]
            else:
                if nw < 500:
                    neigh_watch_sharp = 0
                elif nw < 1000:
                    neigh_watch_sharp = 1
                elif nw < 1500:
                    neigh_watch_sharp = 2
                elif nw < 2000:
                    neigh_watch_sharp = 3
                elif nw < 2500:
                    neigh_watch_sharp = 4
                else:
                    neigh_watch_sharp = 5
            
            b = df.loc[[node]]
            a = b.area.values[0]
            c = b.centroid.values[0]
            boundary = b.boundary.values[0]
            # the form factor is defined as the ratio between the area and the area of the circumscribed circle,
            # but not all polyogons have a circumscribed circle! I will use the smallest enclosing circle instead.
            
        #         r = b.hausdorff_distance(c) # this was my first guess of circumscribed circle

            # need to convert multilines to lines to get boundary coords
            if boundary.type == 'MultiLineString':
                multicoords = [list(line.coords) for line in boundary]
                boundary = shapely.geometry.LineString([item for sublist in multicoords  for item in sublist])
            
            # Welzl's algorithm to find the smallest enclosing circle:
            vertices = list(boundary.coords)
            x_c, y_c, r = sec.make_circle(vertices)
            a_circle = np.pi * (r ** 2)
            ff = a / a_circle
            per = boundary.length

            for j in (range(len(area_steps))):
                if a <= area_steps[j] or j == len(area_steps) - 1:
                    area_sharp = j + 1
                    break
                    
            for j in (range(len(perimeter_steps))):
                if per <= perimeter_steps[j] or j == len(perimeter_steps) - 1:
                    perimeter_sharp = j + 1
                    break

            data.append([node, k, w, w/k, a, per, ff, neigh_watch_sharp, area_sharp, perimeter_sharp])
        db = pd.DataFrame(data, columns=columns)
        db = db.sort_values('area')
        self.database = db
        db.to_csv(os.path.join(folder_path, filename), index=False)

    def report_buildings_stats(self, folder_path= ".temp", filename="building_stats.csv"):
        self.database.describe().to_csv(os.path.join(folder_path, filename))


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


    def plot_net(self, figsize=(30, 30), save=True, imgs_folder = ".temp", filename="net", file_format='png', show=True, style="node_color", draw_nodes=True):
        
        G = self.network
        weights = self.weights
        weights_values = [weights[(u, v)] if (u, v) in weights else weights[(v, u)] for u, v in G.edges]
        
        fig, ax  = plt.subplots(figsize=figsize)
        base = self.buildings_df.plot(ax=ax, color='gray', alpha=0.2)
        pos = self.network_pos
        if style == "node_color":
            node_color = self.node_color
            nx.draw_networkx_nodes(G, pos=pos, with_labels=False, node_color=node_color, edgecolors='k', ax=ax)
            nx.draw_networkx_edges(G, pos=pos, width=[w * (2) ** (-8) for w in weights_values], ax=ax)
        elif style == "edge_color":
            edge_color = self.edge_color
            if draw_nodes:
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

    def plot_neighborhood(self, building, imgs_folder=".temp", name="", grayscale=False, file_name = "", file_format="png", show=True, radius=1):
        G = self.network
        if not hasattr(self, 'database'):
            print("Creating self database")
            self.create_database()
        db = self.database
        df = self.buildings_df

        neighborhood = nx.ego_graph(G, building,radius=radius)
        buildings = df.loc[list(neighborhood)]
        pos = {n: (df.loc[[n]].centroid.x.values[0], df.loc[[n]].centroid.y.values[0])  for n in neighborhood}
        if grayscale:
            color = ["dimgray" if b == building else "lightgray" for b in buildings.index]
        else:
            colors = self.colors_nodes
            nw_dict = self.neigh_watch_sharp_dict
            color = [colors[nw_dict[b]] for b in buildings.index]
            buildings.plot(color=color, edgecolor="k")
        nx.draw(neighborhood, pos=pos, node_color='gray', weights='weights', node_size = 30, edgecolors="k")
        nx.draw_networkx_nodes(neighborhood, nodelist=[building], pos=pos, node_color='black', node_size=50)
        path = os.path.join(imgs_folder, "neighborhoods", name)
        os.makedirs(path, exist_ok=True)
        building_data = db[db.id == building]
        text =  "id: %s\n" %building + "w = %.0f \n" %building_data["w"].values[0] + "k = %.0f\n" %building_data["k"].values[0] + "w/k = %.0f\n" %building_data["w/k"].values[0] + "A = %.2f\n" %building_data["area"].values[0] + "2p = %.2f" %building_data["perimeter"].values[0]
        xtext = max(buildings.bounds.maxx) + 35
        plt.text(xtext, pos[building][1], text, fontsize=10, horizontalalignment='center', verticalalignment='center', fontweight="bold", fontfamily="serif", bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(path, str(building) + file_name + "." + file_format))
        if show:
            plt.show()
        plt.close()

    def dump(self, filepath=".temp/B.p"):
        pickle.dump(self, open(filepath, "wb"))

    @staticmethod
    def herfindhal_index(x):
        num = sum([v ** 2 for v in x])
        den = (sum(x)) ** 2
        H = num / den
        return H
   
    def plot_neighborhood_watch_distribution(self, imgs_folder = ".temp", filename="neighborhood_watch_distribution" ,file_format='png', show=True, save=True):
        plt.rcParams.update({'font.size': 22})
        mpl.rc('text', usetex=True)
        fig, ax1 = plt.subplots(figsize=(10,10))
        try:
            neigh_watch = self.database["w/k"]
        except AttributeError:
            self.create_database(filepath=imgs_folder)
        hist, bins, _ = plt.hist(neigh_watch, bins=np.arange(0, 6000, 100), rwidth=0.8)
        # plt.title("Node's average link weight distribution")
        plt.ylabel('N(w/k)')
        plt.xlabel('w/k [$m^2$]')
        N = len(self.database)
        H = self.herfindhal_index(neigh_watch)
        t = '$\\newline\\newline$'.join(['', 'Herfindhal index:', 'H = %.3f' %(H), '1/H = %.1f' %(1 / H),
                        'N = %s' %(N), 'N. outliers = %.3f * N' %(1 - 1 / (H * N))])
        x = max(bins) / 2
        y = max(hist) / 2
        plt.text(x, y, t, verticalalignment='center',
                horizontalalignment='left', fontsize=18)
        left, bottom, width, height = [0.35, 0.7, 0.5, 0.1]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.boxplot(neigh_watch, vert=False)
        plt.yticks([])
        plt.xlabel('w/k', fontsize=18)
        plt.xticks(fontsize=18)
        plt.xlim(-100, 6000)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close()

    def plot_link_weight_distribution(self, imgs_folder = ".temp", filename="link_weight_distribution" ,file_format='png', show=True, save=True, logscale=False):
        plt.rcParams.update({'font.size': 22})
        mpl.rc('text', usetex=True)
        fig, ax1 = plt.subplots(figsize=(10,10))
        weights_values = list(self.weights.values())
        hist, bins, _ = plt.hist(weights_values, bins=np.arange(0, 8000, 100), rwidth=0.8)
        if logscale:
            plt.yscale('log')
        plt.title('Pairwise weight distribution')
        plt.ylabel('N(w(ij))')
        plt.xlabel('w(ij) [$m^2$]')
        H = self.herfindhal_index(weights_values)
        M = len(weights_values)
        t = '$\\newline\\newline$'.join(['', 'Herfindhal index:', 'H = %.3f' %(H), '1/H = %.1f' %(1 / H),
                        'N = %s' %(M), 'N. outliers = %.3f M' %(1 - 1 / (H * M))])
        x = max(bins) / 2
        y = max(hist) / 2
        plt.text(x, y, t, verticalalignment='center',
                horizontalalignment='left', fontsize=22)
        left, bottom, width, height = [0.35, 0.7, 0.5, 0.1]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.boxplot(weights_values, vert=False)
        plt.yticks([])
        plt.xlabel('w(i,j)', fontsize=18)
        plt.xticks(fontsize=18)
        plt.xlim(-100, 6000)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(imgs_folder, filename + "." + file_format))
        if show:
            plt.show()
        plt.close()

    def report_stats_highwij(self, filepath="stats_highwij.txt"):
        net = self.network
        with open(filepath, "w") as f:
            f.write("Nw large if > 1500\n")
            f.write("wij large if > 2500\n")
            n = 0
            D = {}
            for u in net.nodes:
                k = net.degree(u)
                w = net.degree(u, weight='weight')
                nw = w/k
                count = 0
                if nw > 1500:
                    n += 1
                    for v in net.neighbors(u):
                        wij = net.get_edge_data(u, v)['weight']
                        if wij >  2500:
                            count += 1
                    try:
                        D[count] += 1
                    except KeyError:
                        D[count] = 1

            for c in sorted(D):
                D[c] = D[c] / n
                f.write("%.1f%% nodes with %s large-wij links among large-NW nodes\n" %(D[c]*100, c))

            wij_list = [net.get_edge_data(u, v)['weight'] for u, v in net.edges()]
            f.write("%.2f%% of links have wij>2250m^2\n" %(len([wij for wij in wij_list if wij > 2250]) / len(wij_list) * 100))

            n = 0
            D = {}
            for u, v in net.edges:
                wij = net.get_edge_data(u, v)['weight']
                if wij > 2500:
                    n += 1
                    count = 0
                    ku = net.degree(u)
                    wu = net.degree(u, weight='weight')
                    nwu = wu/ku
                    if nwu > 1500:
                        count += 1
                    kv = net.degree(v)
                    wv = net.degree(v, weight='weight')
                    nwv = wv/kv
                    if nwv > 1500:
                        count += 1
                        
                    try:
                        D[count] += 1
                    except KeyError:
                        D[count] = 1

            for c in sorted(D):
                D[c] = D[c] / n
                f.write("%.1f%% links with %s high NW endpoints among large-wij links" %(D[c]*100, c))
                    
            
