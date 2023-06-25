# Utilities for reading, writing, and formatting data
from collections import Iterable
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe


def get_iterable(ax):
    """Cast matplotlib axes object as Iterable type if needed."""
    if isinstance(ax, Iterable):
        return ax
    else:
        return (ax,)


def prep_sif(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess an OCO-2 SIF Lite file.

    NOTE:
    SIF_Uncertainty_740nm is defined as "estimated 1-sigma uncertainty of Solar Induced
    Fluorescence at 740 nm. Uncertainty computed from continuum level radiance at 740 nm."
    Squaring this value yields the measurement error variance.
    """

    # isolate required variables
    variable_list = [
        "SIF_740nm",
        "SIF_Uncertainty_740nm",
        "Quality_Flag",
        "Longitude",
        "Latitude",
        "Delta_Time",
    ]
    ds = ds[variable_list]

    # apply quality filters
    ds["SIF_plus_3sig"] = ds.SIF_740nm + 3 * ds.SIF_Uncertainty_740nm
    ds = ds.where(ds.Quality_Flag != 2, drop=True)
    ds = ds.where(ds.SIF_plus_3sig >= 0, drop=True)

    # format dataset
    return xr.Dataset(
        {
            "sif": (["time"], ds.SIF_740nm.data),
            "sif_var": (["time"], ds.SIF_Uncertainty_740nm.data),
        },
        coords={
            "lon": (["time"], ds.Longitude.data),
            "lat": (["time"], ds.Latitude.data),
            "time": ds.Delta_Time.data,
        },
    )


def prep_xco2(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess an OCO-2 FP Lite file.

    NOTE:
    xco2_uncertainty is defined as "the posterior uncertainty in XCO2 calculated by the L2
    algorithm, in ppm." Squaring this value yields the measurement error variance.
    """

    # drop unused variables
    variable_list = [
        "xco2",
        "xco2_uncertainty",
        "xco2_quality_flag",
        "longitude",
        "latitude",
        "time",
    ]
    ds = ds[variable_list]

    # apply quality filters
    ds = ds.where(ds.xco2_quality_flag == 0, drop=True)

    # format dataset
    return xr.Dataset(
        {
            "xco2": (["time"], ds.xco2.data),
            "xco2_var": (["time"], ds.xco2_uncertainty.data),
        },
        coords={
            "lon": (["time"], ds.longitude.data),
            "lat": (["time"], ds.latitude.data),
            "time": ds.time.data,
        },
    )


def regrid(
    df: pd.DataFrame,
    lon0_b: float = None,
    lon1_b: float = None,
    lat0_b: float = None,
    lat1_b: float = None,
    d_lon: float = None,
    d_lat: float = None,
) -> pd.DataFrame:
    """Reassign data coordinates using a regular grid."""
    kwargs = locals()
    df = kwargs.pop("df", None)

    ds_grid = xe.util.cf_grid_2d(**kwargs)
    lon_bounds = np.append(ds_grid.lon_bounds.values[:, 0], ds_grid.lon_bounds.values[-1, 1])
    lat_bounds = np.append(ds_grid.lat_bounds.values[:, 0], ds_grid.lat_bounds.values[-1, 1])

    # overwrite lon-lat values with grid values
    df["lon"] = pd.cut(df["lon"], lon_bounds, labels=ds_grid["lon"].values).astype(np.float32)
    df["lat"] = pd.cut(df["lat"], lat_bounds, labels=ds_grid["lat"].values).astype(np.float32)

    return df


def monthly_average(df_grid: pd.DataFrame) -> pd.DataFrame:
    """Group dataframe by relabeled lat-lon coordinates and compute monthly average."""
    return (
        df_grid.groupby(["lat", "lon"])
        .resample("1MS", on="time")
        .mean()
        .drop(columns=["lat", "lon"])
        .reset_index()
    )


def apply_gridded_average(
    ds: xr.Dataset,
    lon0_b: float = None,
    lon1_b: float = None,
    lat0_b: float = None,
    lat1_b: float = None,
    d_lon: float = None,
    d_lat: float = None,
) -> xr.Dataset:
    """
    Aggregate irregular data to a regular grid of monthly averages within the specified extents.

    NOTE: measurement error standard deviations are squared to obtain measurement error
    variances before averaging.
    """
    kwargs = locals()
    ds = kwargs.pop("ds", None)

    # NOTE: selection with pandas is much faster than xr.where()
    df = ds.to_dataframe().reset_index()
    bounds = (
        (df["lon"] >= lon0_b)
        & (df["lon"] <= lon1_b)
        & (df["lat"] >= lat0_b)
        & (df["lat"] <= lat1_b)
    )
    # drop data outside domain extents to exclude from bin averages at edges
    df = df.loc[bounds].reset_index()

    # square individual measurement error standard deviations to obtain measurement
    # error variances
    find_var_name = df.columns.str.endswith("_var")
    df.loc[:, find_var_name] = df.loc[:, find_var_name] ** 2

    df_grid = regrid(df, **kwargs)
    if "index" in df_grid.columns:
        df_grid = df_grid.drop(columns="index")

    return monthly_average(df_grid).set_index(["lat", "lon", "time"]).to_xarray()
