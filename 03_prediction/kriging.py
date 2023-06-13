# Produce kriging predictions at all land-based 0.05-degree grid cells in North American
# domain for the specified month

import sys

sys.path.insert(0, "../source")

from copy import deepcopy
import pickle
import numpy as np
import xarray as xr
import prediction

YEARMONTH = sys.argv[1]
NUM_LOCAL_VALUES = 150

with open(f"../data/intermediate/models/{YEARMONTH}/fields.pickle", "rb") as f:
    mf = pickle.load(f)

# keep only the SIF field
mf_univariate = deepcopy(mf)
mf_univariate.fields = np.array([mf_univariate.fields[0]])
mf_univariate.n_procs = 1

with open(f"../data/intermediate/models/{YEARMONTH}/univariate_model.pickle", "rb") as f:
    univariate_matern = pickle.load(f)

with xr.open_dataset(
    "../data/intermediate/OCO2_005deg_monthly_north_america_with_basis.nc4"
) as ds:
    basis_vars = [x for x in list(ds.keys()) if x.startswith("B")]
    ds_covariates = ds[basis_vars].squeeze().drop_vars(["B1", "B10", "B20"])

pcoords = prediction.get_prediction_locations()
krige = prediction.Predictor(univariate_matern, mf_univariate)

print(f"Running predictions for {YEARMONTH}...")
ds_krige = krige.predict(
    0, pcoords, ds_covariates=ds_covariates, num_local_values=NUM_LOCAL_VALUES
)

ds_krige.to_netcdf(f"../data/intermediate/kriging_results_{YEARMONTH}.nc4")
ds_krige.close()

print("Saved.")
