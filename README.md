# Buildings network

A Jupyter Notebook to create a visualize the *Building network* of a geographical area using data from OpenStreetMap. The *Buildings network* is represented by a weighted graph, where nodes are complex hulls of blocks of buildings and weighted links represent proximity between them. The link weight is proportional to the inverse of the empty space between the two blocks.
## Getting Started


### Requirements

[Jupyter](https://jupyter.org/) is needed in order to run the `Buildings_network.ipynb` notebook and the following Python packages are needed:
- [OSMnx](https://github.com/gboeing/osmnx )
- [networkx](https://networkx.github.io/)
- [geopandas](http://geopandas.org/)
- [numpy](https://www.numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [shapely](https://github.com/Toblerity/Shapely)
  
See `requirements.txt` for details about the required versions of the packages and of their dependencies.

If working with [conda](https://docs.conda.io/en/latest/), you have probably Jupyter already installed on your machine. In that case, you can install OSMnx with:

```
conda config --prepend channels conda-forge
conda create -n ox --strict-channel-priority osmnx
```
and activate the `ox` conda environment with:
```
conda activate ox
```

As an alternative, you can install the required packages directly with `pip`. You need to first install the [OSMns requirements](https://github.com/gboeing/osmnx/blob/master/requirements.txt):
```
pip install descartes
pip install geopandas
pip install matplotlib
pip install networkx
pip install numpy
pip install pandas
pip install requests
pip install rtree
pip install shapely
```
please note that [rtree](https://pypi.org/project/Rtree/) requires the [libspatialindex](https://libspatialindex.org/) library. If you don't have it installed, please follow the instructions [here](https://github.com/libspatialindex/libspatialindex/wiki/1.-Getting-Started).

Then, you can install OSMnx
```
pip install osmnx
```
and jupyter
```
pip install jupyter
```

You also need to have LaTex installed on your system in order to produce the plots with Matplotlib. On Ubuntu, you can install LaTex and the necessary extensions by running

```
sudo apt-get install dvipng texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

For other operating systems, or if you encounter problems, please follow the instructions in Matplotlib's [tutorial](https://matplotlib.org/3.1.0/tutorials/text/usetex.html).


### Note for Windows users
Jupyter notebooks seems to raise problems when using Python 3.8 on Windows, due to the `tornado` server that it uses (see for example [here](https://stackoverflow.com/questions/58422817/jupyter-notebook-with-python-3-8-notimplementederror)). It is recommended to downgrade your Python and tornado versions:
```
conda install python=3.6.7
conda install tornado=4.5.3
```

## Authors

* **Lorenza Pacini** [lorpac](https://github.com/lorpac)

<!--- 
See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
-->
