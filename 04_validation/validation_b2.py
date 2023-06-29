import sys

sys.path.insert(0, "../source")

YEARMONTH = str(sys.argv[1])
N_WORKERS = int(sys.argv[2])
NUM_LOCAL_VALUES = 150

from copy import deepcopy
import logging
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import dask
from dask.distributed import Client

import prediction
import validation


logging.basicConfig(
    filename=f"validation_b1_{YEARMONTH}.log",
    format="%(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
    level=logging.INFO,
)
dask.config.set({"logging.distributed": "error"})

if __name__ == "__main__":

    client = Client(n_workers=N_WORKERS)

    ## Global arguments
    NUM_LOCAL_VALUES = 150  # number of nearest observations used in point predictions
    YEAR = 2021
    MONTH = YEARMONTH[-2:]
    OFFSET = 1  # XCO2 months ahead of SIF

    ## Collect data
    month_xco2 = int(MONTH) + OFFSET
    date_sif = f"{YEAR}-{MONTH}-01"
    date_xco2 = f"{YEAR}-{month_xco2}-01"

    with xr.open_dataset(
        "../data/intermediate/OCO2_005deg_months2021_north_america_with_basis.nc4"
    ) as ds:
        basis_vars = [x for x in list(ds.keys()) if x.startswith("B")]
        ds_sif = ds[["sif", "sif_var"] + basis_vars].sel(time=f"{YEAR}-{MONTH}").squeeze()

    ## Setup validation block
    BLOCK_NAME = "b2"
    block_conditions = (
        (ds_sif["lon"] > -104)
        & (ds_sif["lon"] < -99)
        & (ds_sif["lat"] > 36.5)
        & (ds_sif["lat"] < 41.5)
    )
    df_test = (
        ds_sif.where(block_conditions)
        .to_dataframe()
        .dropna(subset="sif")
        .reset_index()[["lat", "lon", "sif", "sif_var"] + basis_vars]
        .rename(columns={"sif": "data", "sif_var": "me_variance"})
    )
    df_me_variance = df_test.loc[:, ["lat", "lon", "me_variance"]]
    df_test = df_test.drop(columns="me_variance")
    ds_covariates = (
        df_test.set_index(["lat", "lon"]).to_xarray().drop_vars(["data", "B1", "B10", "B20"])
    )
    df_test = df_test.loc[:, ["lat", "lon", "data"]]
    pcoords = df_test.loc[:, ["lat", "lon"]]
    df_test.to_csv(f"../data/intermediate/validation/{YEARMONTH}/df_test_{BLOCK_NAME}.csv")

    ## Run cokriging validation
    logging.info(f"Running cokriging validation, block {BLOCK_NAME}...")

    # Load multifield and model objects
    with open(f"../data/intermediate/models/{YEARMONTH}/fields.pickle", "rb") as f:
        mf = pickle.load(f)

    with open(f"../data/intermediate/models/{YEARMONTH}/bivariate_model.pickle", "rb") as f:
        bivariate_matern = pickle.load(f)

    # Make validation predictions
    cokrige = prediction.Predictor(bivariate_matern, mf)
    dict_cokrige = cokrige.validation_prediction(
        0,
        pcoords,
        df_me_variance,
        ds_covariates,
        num_local_values=NUM_LOCAL_VALUES,
    )
    ds_cokrige = dict_cokrige["ds"]

    # Save multivariate predictive distribution
    pd.DataFrame(dict_cokrige["predictive_mean"]).to_csv(
        f"../data/intermediate/validation/{YEARMONTH}/mean_cokriging_{BLOCK_NAME}.csv",
        header=None,
        index=None,
    )

    pd.DataFrame(dict_cokrige["predictive_covariance_matrix"]).to_csv(
        f"../data/intermediate/validation/{YEARMONTH}/covariance_cokriging_{BLOCK_NAME}.csv",
        header=None,
        index=None,
    )

    # Prepare validation results
    df_cokrige = (
        ds_cokrige.to_dataframe().reset_index()[["lat", "lon", "predictions", "rmspe"]].dropna()
    )

    df_test_cokrige = validation.prepare_validation_results(
        df_cokrige,
        df_test,
        method="cokriging",
        month=YEARMONTH,
        region=BLOCK_NAME,
    )

    # Compute QQ-plot quantiles
    quantiles_cokrige = sm.ProbPlot(df_test_cokrige["ratio"].values)
    df_quantiles_cokrige = pd.DataFrame(
        dict(
            sample=quantiles_cokrige.sample_quantiles,
            theoretical=quantiles_cokrige.theoretical_quantiles,
            Method="Cokriging",
        )
    )

    client.close()

    ## Run kriging validation
    logging.info(f"Running kriging validation, block {BLOCK_NAME}...")
    client = Client(n_workers=N_WORKERS, silence_logs=logging.ERROR)

    # Isolate SIF field only
    mf_univariate = deepcopy(mf)
    mf_univariate.fields = np.array([mf_univariate.fields[0]])
    mf_univariate.n_procs = 1

    # Load univariate model
    with open(f"../data/intermediate/models/{YEARMONTH}/univariate_model.pickle", "rb") as f:
        univariate_matern = pickle.load(f)

    # Make validation predictions
    krige = prediction.Predictor(univariate_matern, mf_univariate)
    dict_krige = krige.validation_prediction(
        0,
        pcoords,
        df_me_variance,
        ds_covariates,
        num_local_values=NUM_LOCAL_VALUES,
    )
    ds_krige = dict_krige["ds"]

    # Save multivariate predictive distribution
    pd.DataFrame(dict_krige["predictive_mean"]).to_csv(
        f"../data/intermediate/validation/{YEARMONTH}/mean_kriging_{BLOCK_NAME}.csv",
        header=None,
        index=None,
    )

    pd.DataFrame(dict_krige["predictive_covariance_matrix"]).to_csv(
        f"../data/intermediate/validation/{YEARMONTH}/covariance_kriging_{BLOCK_NAME}.csv",
        header=None,
        index=None,
    )

    # Prepare validation results
    df_krige = (
        ds_krige.to_dataframe().reset_index()[["lat", "lon", "predictions", "rmspe"]].dropna()
    )

    df_test_krige = validation.prepare_validation_results(
        df_krige,
        df_test,
        method="kriging",
        month=YEARMONTH,
        region=BLOCK_NAME,
    )

    # Compute QQ-plot quantiles
    quantiles_krige = sm.ProbPlot(df_test_krige["ratio"].values)
    df_quantiles_krige = pd.DataFrame(
        dict(
            sample=quantiles_krige.sample_quantiles,
            theoretical=quantiles_krige.theoretical_quantiles,
            Method="Kriging",
        )
    )

    client.close()

    ## Run trend surface validation
    logging.info(f"Running trend surface validation, block {BLOCK_NAME}...")

    # Prepare validation results
    df_trend = (
        ds_krige.to_dataframe()
        .reset_index()[["lat", "lon", "trend_surface", "trend_surface_rmspe"]]
        .dropna()
        .rename(columns={"trend_surface": "predictions", "trend_surface_rmspe": "rmspe"})
    )

    df_test_trend = validation.prepare_validation_results(
        df_trend,
        df_test,
        method="Trend surface",
        month=YEARMONTH,
        region=BLOCK_NAME,
    )

    # Compute QQ-plot quantiles
    quantiles_trend = sm.ProbPlot(df_test_trend["ratio"].values)
    df_quantiles_trend = pd.DataFrame(
        dict(
            sample=quantiles_trend.sample_quantiles,
            theoretical=quantiles_trend.theoretical_quantiles,
            Method="Trend surface",
        )
    )

    # Save multivariate predictive distribution
    df_test_trend["predictions"].to_csv(
        f"../data/intermediate/validation/{YEARMONTH}/mean_trend_{BLOCK_NAME}.csv",
        header=None,
        index=None,
    )
    pd.DataFrame(np.diag(df_test_trend["rmspe"].values ** 2)).to_csv(
        f"../data/intermediate/validation/{YEARMONTH}/covariance_trend_{BLOCK_NAME}.csv",
        header=None,
        index=None,
    )

    ## Collect all validation results
    logging.info("Saving validation results.")

    df_validation_results = pd.concat(
        [df_test_trend, df_test_krige, df_test_cokrige], axis=0
    ).reset_index(drop=True)
    df_validation_results.to_csv(
        f"../data/intermediate/validation/{YEARMONTH}/validation_results_{BLOCK_NAME}.csv"
    )

    df_quantiles = pd.concat(
        [df_quantiles_trend, df_quantiles_krige, df_quantiles_cokrige], axis=0
    ).reset_index(drop=True)
    df_quantiles.to_csv(
        f"../data/intermediate/validation/{YEARMONTH}/validation_ratio_quantiles_{BLOCK_NAME}.csv"
    )

    logging.info("Done.")
