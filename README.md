# Building network

A Jupyter Notebook and a R Shiny app to create a visualize the *Building network* of a geographical area using data from OpenStreetMap. The *Building network* is represented by a weighted graph, where nodes are complex hulls of blocks of buildings and weighted links represent proximity between them. The link weight is proportional to the inverse of the empty space between the two blocks.

- the [R Shiny app](https://github.com/lorpac/building-network/blob/master/app.R) allows you to create the Building Network of an area of 2kmx2km size around a geographical point.
- the [Jupyter notebook](https://github.com/lorpac/building-network/blob/master/Buildings_network.ipynb) allows you to create the Building Network of an area of the size of your choice around a geographical point, or the the Building Network of an entire city.

## Table of contents
<!-- vscode-markdown-toc -->
* 1. [Requirements](#Requirements)
	* 1.1. [Optional - Use a Python virtual environment](#Optional-UseaPythonvirtualenvironment)
		* 1.1.1. [Windows](#Windows)
		* 1.1.2. [MacOS or Linux](#MacOSorLinux)
	* 1.2. [Compulsory - Python requirements](#Compulsory-Pythonrequirements)
		* 1.2.1. [Windows](#Windows)
		* 1.2.2. [Linux](#Linux)
		* 1.2.3. [All operating systems](#Alloperatingsystems)
	* 1.3. [Requirements for the Jupyter notebook](#RequirementsfortheJupyternotebook)
		* 1.3.1. [Note for Windows users](#NoteforWindowsusers)
	* 1.4. [Requirements for the R Shiny app](#RequirementsfortheRShinyapp)
	* 1.5. [How to use the R Shiny app](#HowtousetheRShinyapp)
		* 1.5.1. [Launch the app](#Launchtheapp)
		* 1.5.2. [Create your Building Network](#CreateyourBuildingNetwork)
		* 1.5.3. [Retrieve the results](#Retrievetheresults)
* 2. [Authors](#Authors)
* 3. [Known issues (work in progress!)](#Knownissuesworkinprogress)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

##  1. <a name='Requirements'></a>Requirements
The calculations are done in Python. You need to install Python (version 3.x) and install the required packages (essentially, [OSMnx](https://github.com/gboeing/osmnx) with its dependencies, see below).

###  1.1. <a name='Optional-UseaPythonvirtualenvironment'></a>Optional - Use a Python virtual environment
####  1.1.1. <a name='Windows'></a>Windows
Due to geopandas installation requirement, installing with conda is required on Windows. We thus use a conda virtual environment.
```
conda env create -n cityenv
conda activate cityenv
```

####  1.1.2. <a name='MacOSorLinux'></a>MacOS or Linux
```
pip3 install virtualenv
python3 -m virtualenv .env
source .env/bin/activate
```
###  1.2. <a name='Compulsory-Pythonrequirements'></a>Compulsory - Python requirements
####  1.2.1. <a name='Windows'></a>Windows
Due to geopandas installation requirement, installing with conda is required on Windows.

The OSMnx dependency [rtree](https://pypi.org/project/Rtree/) requires the [libspatialindex](https://libspatialindex.org/) library. If you don't have it installed, please follow the instructions [here](https://github.com/libspatialindex/libspatialindex/wiki/1.-Getting-Started).

Install OSMnx dependencies:

```
conda install geopandas
conda install descartes
conda install matplotlib
conda install networkx
conda install numpy
conda install pandas
conda install requests
conda install rtree
conda install shapely
```

and install OSMnx with `pip`:

```
conda install pip
pip install osmnx
```

and finally install `imageio`, required to produce a GIF of the buildings merging intermediates:
```
conda install imageio
```

####  1.2.2. <a name='Linux:'></a>Linux

The OSMnx dependency [rtree](https://pypi.org/project/Rtree/) requires the [libspatialindex](https://libspatialindex.org/) library, that is not installed automatically. If you don't have it installed, please run:

```
sudo apt-get install libspatialindex-dev
```
You also need to install libgeos, that is required by Shapely. You can do so with:

```
sudo apt-get install libgeos-dev
```

Then, install OSMnx and it dependencies and  `imageio`, required to produce a GIF of the buildings merging intermediates, using pip:
```
pip install -r requirements.txt
```

####  1.2.3. <a name='Alloperatingsystems'></a>All operating systems
You also need to have LaTex installed on your system in order to produce the plots with Matplotlib. On Ubuntu, you can install LaTex and the necessary extensions by running

```
sudo apt-get install dvipng texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

For other operating systems, or if you encounter problems, please follow the instructions in Matplotlib's [tutorial](https://matplotlib.org/3.1.0/tutorials/text/usetex.html).


###  1.3. <a name='RequirementsfortheJupyternotebook'></a>Requirements for the Jupyter notebook
If working with [conda](https://docs.conda.io/en/latest/), you have probably Jupyter already installed on your machine. Otherwise, run
```
pip install jupyter
```

####  1.3.1. <a name='NoteforWindowsusers'></a>Note for Windows users
Jupyter seems to raise problems when using Python 3.8 on Windows, due to the `tornado` server that it uses (see for example [here](https://stackoverflow.com/questions/58422817/jupyter-notebook-with-python-3-8-notimplementederror)). It is recommended to downgrade your Python and tornado versions:
```
conda install python=3.6.7
conda install tornado=4.5.3
```

###  1.4. <a name='RequirementsfortheRShinyapp'></a>Requirements for the R Shiny app
First, you need to install [R](https://cran.r-project.org/) (and optionally [RStudio](https://rstudio.com/products/rstudio/download/)).

 The following R packages have to be installed:
- shiny
- leaflet
- leaflet.extras
- comprehenr
- markdown

You can istall them by typing 

```
install.packages("shiny", dependencies = TRUE)
install.packages("leaflet", dependencies = TRUE)
install.packages("leaflet.extras", dependencies = TRUE)
install.packages("comprehenr", dependencies = TRUE)
install.packages("markdown", dependencies = TRUE)
```

in the R console.


###  1.5. <a name='HowtousetheRShinyapp'></a>How to use the R Shiny app
####  1.5.1. <a name='Launchtheapp'></a>Launch the app

If you use [RStudio](https://rstudio.com/products/rstudio/download/):
- Run app.R, RStudio  will launch.
- Click **run App**. It is then advised to open the app in your browser (click **Open in browser**).

Alternatively, you can launch the app directly from your console with the following command:
```
R -e "shiny::runApp('~/building-network/app.R')"
```
(substitute `~/building-network_app/` with your path to the app, if you haven't cloned `building-network` to your `home/` folder).

####  1.5.2. <a name='CreateyourBuildingNetwork'></a>Create your Building Network
-  You can give your job a name using the **Job name** field. The default job name is *BuildingsNetwork*.
- Move the blue square in the map to select the area of interest and click the button **Run**. Alternatively, you can directly insert the geographical coordinates of the center of your are of interest in the (Latitude, Longitude) boxes and click the button **Run**. It is also possible to search for places by clicking on the magnifying glass icon in the map.
 
The area used for the creation of the Building Network is a 2km x 2km square.

####  1.5.3. <a name='Retrievetheresults'></a>Retrieve the results

- If **Save results** is checked before running the analysis (that's the default behavior), you will find a copy of the produced pictures (buildings footprint, merged buildings, Building Network, colored network), together with a text file containing the values of the input coordinates (center of the square area), in a subdirectory of the  `results/` folder, named from the job name and the day and time at which the analysis was run.
- You can also download the results by clicking on **Download** once the calculation has finished.




##  2. <a name='Authors'></a>Authors

* **Lorenza Pacini** [lorpac](https://github.com/lorpac)

##  3. <a name='Knownissuesworkinprogress'></a>Known issues (work in progress!)

- In the R Shiny app, the blue square in the map is deformed at latitudes far from the European latitude. However, this does not impact the shape of the area that is actually considered for the creation of the Building Network, it remains a 2km x 2km squared area centered around the center of the (deformed) square.
