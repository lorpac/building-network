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
print("buildings loaded in %.2f seconds" %(time.time() - t1))
t1 = time.time()
B.plot_buildings(imgs_folder=output_folder, show=False)
print("plotted buildings in %.2f seconds" %(time.time() - t1))

t1 = time.time()
print("merging...")
B.merge_and_convex(buffer=buffer,plot=True, imgs_folder=output_folder, show=False)
print("finished merging in %.2f seconds" %(time.time() - t1))
t1 = time.time()
B.plot_merged_buildings(imgs_folder=output_folder, show=False)
print("plotted merged buildings in %.2f seconds" %(time.time() - t1))

t1 = time.time()
B.create_gif_merging(imgs_folder=output_folder)
print("gif of merged buildings intermediates created in  %.2f seconds" %(time.time() - t1))

t1 = time.time()
print("assigning nodes...")
B.assign_nodes()
print("finished assigning nodes in %.2f seconds" %(time.time() - t1))
t1 = time.time()
B.plot_nodes(imgs_folder=output_folder, show=False)
print("plotted nodes %.2f seconds" %(time.time() - t1))

t1 = time.time()
print("assigning edges...")
B.assign_edges_weights(distance_threshold=distance_threshold)
print("finished assigning edges in %.2f seconds" %(time.time() - t1))
t1 = time.time()
B.plot_edges(imgs_folder=output_folder, show=False)
print("plotted edges %.2f seconds" %(time.time() - t1))

t1 = time.time()
print("assigning network...")
B.assign_network()
print("finished assigning network in %.2f seconds" %(time.time() - t1))
t1 = time.time()
B.plot_net(imgs_folder=output_folder, show=False)
print("plotted network in %.2f seconds" %(time.time() - t1))

t1 = time.time()
B.plot_buildings_color(imgs_folder=output_folder, show=False)
print("plotted colored buildings in %.2f seconds" %(time.time() - t1))

print("total execution time: %.2f seconds" %(time.time() - t0))

B.dump(filepath=os.path.join(output_folder, "test.p"))
print("Dumped.")

B2 = Building.generate(load=True, filepath=os.path.join(output_folder, "test.p"))
print("Loaded")
print(B2.point_coords)



