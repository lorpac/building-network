from Building import Building
import os
import sys
import matplotlib as mpl
mpl.use('Agg')

results_folder = os.path.join("www", ".temp")

try:
    _, lat, lng = sys.argv
except ValueError:
    # lat, lng = ("45.745591", "4.871167") # Montplaisir
    lat, lng = ("45.7394953462566", "4.93414878132277")

def update_status(status):
    status = str(status)
    with open("status", "w")  as f:
        f.write(status)

update_status(0)

point_coords = (float(lat), float(lng))
B = Building(point_coords=point_coords)
B.download_buildings()
B.plot_buildings(imgs_folder=results_folder)
update_status(1)

B.merge_and_convex()
B.plot_merged_buildings(imgs_folder=results_folder)
update_status(2)

B.assign_nodes()
B.plot_nodes(imgs_folder=results_folder)
update_status(3)

B.assign_edges_weights()
B.plot_edges(imgs_folder=results_folder)
update_status(4)

B.assign_network()
B.plot_net(imgs_folder=results_folder, style="edge_color")
update_status(5)

B.plot_buildings_color(imgs_folder= results_folder)
update_status(6)

B.save_config(os.path.join(results_folder, "config.json"))

