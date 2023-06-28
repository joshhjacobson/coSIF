"""
Produce cokriging predictions at all land-based 0.05-degree grid cells in the North American
domain for a specified month.

This script should be run from the command line as in the following example:
```
conda activate cosif
cd 03_prediction
python cokriging.py 202107
```
where the string `202107` indicates that predictions and prediction standard errors 
will be produced for July 2021.

NOTE: This is a long-running processes that can take several hours to one day of compute 
time on a 64-core server.
"""

import sys

sys.path.insert(0, "../source")

import pickle
import xarray as xr
import prediction

YEARMONTH = sys.argv[1]
NUM_LOCAL_VALUES = 150

with open(f"../data/intermediate/models/{YEARMONTH}/fields.pickle", "rb") as f:
    mf = pickle.load(f)

with open(f"../data/intermediate/models/{YEARMONTH}/bivariate_model.pickle", "rb") as f:
    bivariate_matern = pickle.load(f)

with xr.open_dataset(
    "../data/intermediate/OCO2_005deg_months2021_north_america_with_basis.nc4"
) as ds:
    basis_vars = [x for x in list(ds.keys()) if x.startswith("B")]
    ds_covariates = ds[basis_vars].squeeze().drop_vars(["B1", "B10", "B20"])

pcoords = prediction.get_prediction_locations()
cokrige = prediction.Predictor(bivariate_matern, mf)

print(f"Running predictions for {YEARMONTH}...")
ds_cokrige = cokrige.predict(
    0, pcoords, ds_covariates=ds_covariates, num_local_values=NUM_LOCAL_VALUES
)

ds_cokrige.to_netcdf(
    f"../data/intermediate/cokriging_results_{YEARMONTH}.nc4", format="NETCDF4"
)
ds_cokrige.close()

print("Saved.")
