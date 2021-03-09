# Weather Regimes tool (WRtool)

Weather Regimes are recurrent and persistent dynamical configurations found at mid-latitudes, which are mostly used to test the performance of climate models and for impact studies under climate change.

WRtool calculates the Weather Regimes starting from an observed or modeled geopotential field focusing on a custom geographical region and season. The Weather Regimes are calculated in a low dimensional Empirical Orthogonal Functions space through a K-means clustering algorithm. A set of diagnostics is then applied to the regimes obtained in this way, comparing their pattern, frequency, mean persistence with a reference set of regimes (usually obtained from reanalysis).

The main results are synthetically listed in the *results.dat* file. Optionally also netcdf files and figures are produced.

![Weather Regimes in the North Atlantic](https://github.com/fedef17/WRtool/blob/master/eraclus.png)

To cite this repository: 
[![DOI](https://zenodo.org/badge/138048137.svg)](https://zenodo.org/badge/latestdoi/138048137)

## How to install WRtool

WRtool is written in Python3, with a couple of routines in Fortran, and needs the ClimTools library to work.

#### 0. Clone the WRtool repo
Open a terminal inside your code folder and do:
```
git clone https://github.com/fedef17/WRtool.git
```

This will create a folder WRtool inside your code folder.
To update your local version of the code, just go inside the WRtool folder and do:
```
git pull origin master
```

#### 1. Install Climtools

Follow the instructions in https://github.com/fedef17/ClimTools to **install ClimTools**. Perform all steps listed under Preliminaries, otherwise WRtool won't work.


## Before running: configuring the input file

WRtool needs many user inputs to work: these are all specified inside the **WRtool input file**.
Open the *input_WRtool_template.in* file, which you can find in the WRtool directory. It contains a list of keys put inside squared brackets: below each [key], the key value is set. Keep the formatting as it is, don't add any blank line between the key and its value.

The input file is structured in four main sections:
1) Directory and file paths:
    - set a name for your experiment;
    - specify input/output directories;
    - a list of input files and their names. To specify the input files, you may either list them inline under the key [filenames] or write them one per line in an external file indicated by [filelist];
    - the reference (observed) geopotential field to compare with.

**All the inputs in this section are mandatory: the tool won't work properly without.**

2) Options to set:
    - the period considered (year range);
    - the **season** (JJA, DJF, DJFM, ...);
    - the geographical **area** to be considered for the calculation (EAT, PNA, custom..);
    - the pressure level to be considered for the geopotential.

3) Options that set the details of the EOF and clustering technique:
    - the **number of EOFs** to be considered (i.e. the number of dimensions of the reduced space) or a threshold for the variance explained by the EOFs considered;
    - the **number of clusters**;
    - more advanced options, regarding the calculation of the anomalies and the detrending of the geopotential field.

4) Other secondary options regarding mainly the visualization of the results.

More details about all the options available can be found inside the template input file.

## Run!

To run the tool, open a terminal inside the WRtool folder and do:
```
conda activate ctl3
python newWRtool.py /path/to/myinputfile.in &
```

The path of the input file can be both the absolute or the relative path from the running directory.
The code will automatically create a log file and copy it to the output folder specified in the input file.
If the tool stops for some reason you will find the log file in the run folder instead.
