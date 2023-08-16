# coSIF: Spatial statistical prediction of solar-induced chlorophyll fluorescence (SIF) from multivariate OCO-2 data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8078592.svg)](https://doi.org/10.5281/zenodo.8078592)

This repository contains code to reproduce the results in the paper:

> Jacobson, J., Cressie, N., & Zammit-Mangion, A. (2023). Spatial statistical prediction of solar-induced chlorophyll fluorescence (SIF) from multivariate OCO-2 data. *Remote Sensing*, 15(16), 4038. https://doi.org/10.3390/rs15164038

Unless stated otherwise, all commands are to be run in the root directory of the repository.

The resulting coSIF data product for February, April, July, and October 2021 is available at: https://doi.org/10.5281/zenodo.8078592

A supplementary dataset of all fitted model parameters is available at: https://doi.org/10.5281/zenodo.8078560

<!-- <img align="center" src="https://github.com/joshhjacobson/coSIF/blob/main/cosif_202107.pdf" alt="drawing" width="50%"/> -->
![alt text](https://github.com/joshhjacobson/coSIF/blob/main/cosif_202107.png?raw=true)

## Installation and setup

Setup the `cosif` conda environment using the provided file `environment.yaml` (this may take a few minutes):
```
conda env create -f environment.yaml
```
Install the required R packages:
```
Rscript -e "install.packages(c(
  "tidyverse", "FRK", "sp", "sf", "rnaturalearth", "rnaturalearthdata"
))"
```
Create the required directories:
```
mkdir data figures
cd data
mkdir eda input intermediate output
cd intermediate
mkdir models validation
cd models
mkdir 202102 202104 202107 202110
cd ../validation
mkdir 202102 202104 202107 202110
```

## Getting the data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8078476.svg)](https://doi.org/10.5281/zenodo.8078476)

A compressed file containing the input datasets required for the analysis is archived [here](https://doi.org/10.5281/zenodo.8078476). This file can be downloaded and extracted into the directory `data/input`. These input datasets include both observational and auxiliary datasets, which we obtained as described below.

### Observational datasets: OCO-2 SIF and XCO2

Both the SIF and XCO2 datasets are publicly available through NASA's GES DISC (Goddard Earth Sciences Data and Information Services Center).

- The SIF Lite files (version 10r) are available [here](https://disc.gsfc.nasa.gov/datasets/OCO2_L2_Lite_SIF_10r/summary). A subset of these NetCDF files for February, April, July, and October 2021 should be located in the directory `data/input/OCO2_L2_Lite_SIF_10r`.
- The XCO2 Lite files (version 10r) are available [here](https://disc.gsfc.nasa.gov/datasets/OCO2_L2_Lite_FP_10r/summary). A subset of these NetCDF files for March, May, August, and November 2021 should be located in the directory `data/input/OCO2_L2_Lite_FP_10r`.

NOTE: To reproduce the exploratory time series in Figure 1 (see below), you will need to retrieve all of the SIF and XCO2 version 10r Lite files. Organize the SIF and XCO2 parent directories as `data/eda/OCO2_L2_Lite_SIF_10r` and `data/eda/OCO2_L2_Lite_FP_10r`, respectively.

### Auxiliary datasets: MODIS LCC

The Terra and Aqua combined Moderate Resolution Imaging Spectroradiometer (MODIS) Land Cover Climate Modeling Grid (CMG) (MCD12C1) Version 6.1 data product is publicly available on NASA's [Earthdata platform](https://lpdaac.usgs.gov/products/mcd12c1v061/). The product is available from 2001, but note that only the file for 2021 is needed. The HDF file should be located in the directory `data/input/MCD12C1v061`.

## Running the framework

In an initial exploratory data analysis (EDA) step, we create a bivariate time series (Figure 1) from monthly, gridded SIF and XCO2 data. This analysis is isolated in the directory `00_eda`. Note that all of the version 10r Lite files are needed for this step (see above).

There are four main steps in our multivariate spatial-statistical-prediction framework, corresponding to four numbered directories. These are: 

1. `01_data_preparation`: Numbered files are to be run in order. Notebooks create the land-cover binary mask; collect and format all daily OCO-2 Lite files into a single NetCDF file for daily, spatially irregular SIF and a single NetCDF file for daily, spatially irregular XCO2; group SIF and XCO2 datasets by month and compute an average for each 0.05-degree CMG grid cell; an R script evaluates bisquare basis functions for all CMG grid cells; a final notebook combines gridded SIF, XCO2, and basis-function datasets into a single NetCDF file. 
2. `02_modeling`: For each of February, April, July, and October 2021, notebooks compute empirical (cross-) semivariograms from the gridded SIF and XCO2 data (one month later), and fit modeled (cross-) semivariograms. Each notebook takes around 10 minutes to run, and can be run on a laptop.
3. `03_prediction`: For each of February, April, July, and October 2021, the predictions and prediction standard errors required for the coSIF data product are produced by running either `cokriging.py` or `kriging.py` from the command line and specifying the year-month string as an argument. For example, to use cokriging in July 2021, run
    ```
    conda activate cosif
    cd 03_prediction
    python cokriging.py 202107
    ```
    Or, to use kriging in October 2021, run
    ```
    conda activate cosif
    cd 03_prediction
    python kriging.py 202110
    ```
    Note that these are long-running processes that can take several hours to one day of compute time on a 64-core server. It is advised that they be executed in a [screen session](https://linuxize.com/post/how-to-use-linux-screen/) to avoid issues with interruption. Once the predictions and prediction standard errors have been produced for each month, the coSIF data product is collected and formatted in `collect_coSIF_datasets.ipynb`.
4. `04_validation`: For each of February, April, July, and October 2021, the validation predictions for the Corn Belt validation block (b1) and the Cropland validation block (b2) are produced by running `run_validation.py`. The script takes three arguments from the command line: 1) validation year-month; 2) block name; 3) number of cores for parallelization. For example, the script can be run for July 2021 in the Corn Belt validation block (b1) using 64 cores as follows:
    ```
    conda activate cosif
    cd 04_validation
    python run_validation.py 202107 b1 64
    ``` 
    The script can take around 30 minutes to run on a 64-core server. After running the script for both blocks in all four months, metrics used to summarize the validation predictions are collected in `collect_validation_results.ipynb`.

NOTE: ensure that all notebooks are run using the `cosif` conda environment.