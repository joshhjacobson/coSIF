# coSIF: Spatial statistical prediction of solar-induced chlorophyll ﬂuorescence (SIF) from multivariate OCO-2 data

This repository contains code to reproduce the results in the paper:

> Jacobson, J., Cressie, N., Zammit-Mangion, A. (n.d.) Spatial statistical prediction of solar-induced chlorophyll ﬂuorescence (SIF) from multivariate OCO-2 data. Under review in *Remote Sensing*.

Unless stated otherwise, all commands are to be run in the root directory of the repository.

The resulting *coSIF* data products for Februray, April, July, and October 2021 are archived here: TODO

Supplementary datasets are available here: TODO

## Installation and setup

Setup the `geostat` conda environment using the provide file `environment.yaml`:
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
mkdir production raw
cd production
mkdir models validation
cd models
mkdir 202102 202104 202107 202110
cd ../validation
mkdir 202102 202104 202107 202110
```

## Getting the data

All input datasets go into the directory `data/raw`. These includes both observational and auxiliary datasets.

### Observational datasets: OCO-2 SIF and XCO2

Both the SIF and XCO2 datasets are publicly available through NASA's Goddard Earth Sciences Data and Information Services Center (GES DISC).

- The SIF Lite files (version 10r) are available [here](https://disc.gsfc.nasa.gov/datasets/OCO2_L2_Lite_SIF_10r/summary). The NetCDF files should be placed in the directory `data/raw/OCO2_L2_Lite_SIF.10r/`.
- The XCO2 Lite files (version 10r) are available [here](https://disc.gsfc.nasa.gov/datasets/OCO2_L2_Lite_FP_10r/summary). The NetCDF files should be placed in the directory `data/raw/OCO2_L2_Lite_FP.10r/`.

### Auxiliary datasets: MODIS LCC

The Terra and Aqua combined Moderate Resolution Imaging Spectroradiometer (MODIS) Land Cover Climate Modeling Grid (CMG) (MCD12C1) Version 6.1 data product is publicly available on NASA's [Earthdata platform](https://lpdaac.usgs.gov/products/mcd12c1v061/). The product is available from 2001, but note that only 2021 is needed. The HDF file should be placed in the directory `data/raw/MCD12C1v061`.

## Running the framework

There are four main steps in our multivariate spatial-statistical-prediction framework, corresponding to the four numbered directories. These are: 

1. `01_data_preparation`:
2. `02_modeling`:
3. `03_prediction`:
4. `04_validation`: