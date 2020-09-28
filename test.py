from Building import Building
import time
import os
import matplotlib as mpl
mpl.use('Agg') # avoid opening pf buildings figure

t0 = time.time()

point_coords = (45.7394953462566, 4.93414878132277)
distance = 1000 # in meters
buffer = 0.01 # 1 cm
distance_threshold = 30
output_folder = "test_output"
os.makedirs(output_folder, exist_ok=True)

B = Building(point_coords, distance=distance)

t1 = time.time()
print("loading buildings...")
B.download_buildings()
print("buildings loaded in %s seconds" %(time.time() - t1))
t1 = time.time()
B.plot_buildings(imgs_folder=output_folder, show=False)
print("plotted buildings %s seconds" %(time.time() - t1))

t1 = time.time()
print("merging...")
B.merge_and_convex(buffer=buffer)
print("finished merging in %s seconds" %(time.time() - t1))
t1 = time.time()
B.plot_merged_buildings(imgs_folder=output_folder, show=False)
print("plotted merged buildings %s seconds" %(time.time() - t1))

t1 = time.time()
print("assigning nodes...")
B.assign_nodes()
print("finished assigning nodes in %s seconds" %(time.time() - t1))
t1 = time.time()
B.plot_nodes(imgs_folder=output_folder, show=False)
print("plotted nodes %s seconds" %(time.time() - t1))

t1 = time.time()
print("assigning edges...")
B.assign_edges_weights(distance_threshold=distance_threshold)
print("finished assigning edges in %s seconds" %(time.time() - t1))
t1 = time.time()
B.plot_edges(imgs_folder=output_folder, show=False)
print("plotted edges %s seconds" %(time.time() - t1))

t1 = time.time()
print("assigning network...")
B.assign_network()
print("finished assigning network...in %s seconds" %(time.time() - t1))
t1 = time.time()
B.plot_net(imgs_folder=output_folder, show=False)
print("plotted network %s seconds" %(time.time() - t1))

print("total execution time: %s seconds" %(time.time() - t0))



