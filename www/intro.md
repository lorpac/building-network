## Welcome to the **Buildings Network app**!

### How to run the calculation
 - You can give your job a name using the **Job name** field. The default job name is *BuildingsNetwork*.
 - Move the blue square in the map to select the area of interest and click the button <i class="fas fa-play"></i> **Run**. Alternatively, you can directly insert the geographical coordinates of the center of your are of interest in the (Latitude, Longitude) boxes and click the button <i class="fas fa-play"></i> **Run**. It is also possible to search for places by clicking on the magnifying glass icon in the map.

The area used for the creation of the Buildings Network is a 2km x 2km square.

### How to retrieve the results
- If **Save results** is checked before running the analysis (that's the default behavior), you will find a copy of the produced pictures (buildings footprint, merged buildings, Buildings Network, colored network), together with a text file containing the values of the input coordinates (center of the square area), in a subdirectory of the  `results/` folder, named from the job name and the day and time at which the analysis was run.
- You can also download the results by clicking on <i class="fas fa-download"></i> **Download** once the calculation has finished.

### How to cite
If you use the Buildings Network in your work, please cite 
- Pacini L.; Cazabet R.; Vuillon L.; and Lesieur C., **Diagnostics of sustainable building layouts in the city of Lyon: a biomimetic approach**, *Manuscript submitted to Biomimetics, Special Issue « Biomimicry and Sustainable Urban Design »*, 2020

### Going further

If you want to customize the size of the geographical area, produce the Buildings Network of a whole city, or retrieve the graph of the Buildings Network for analysis, please use our **BuildingsNetwork Jupyter notebook** available at <i class="fa fa-github" aria-hidden="true"></i> [lorpac/building-network](https://github.com/lorpac/building-network).

For any questions and suggestion, please write to  `lorenza.pacini AT univ-lyon1.fr`
