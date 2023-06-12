import warnings
from typing import Optional, Union, Tuple, List, Dict
from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from xarray import Dataset
from sklearn.metrics.pairwise import haversine_distances
from sklearn.linear_model import LinearRegression

# Earth's volumetric mean radius in kilometers
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
EARTH_RADIUS = 6371.0


class VarioConfig:
    """Structure to store configuration parameters for an empirical variogram."""

    def __init__(
        self,
        max_dist: Union[float, Dict[str, float]],
        n_bins: int,
        n_procs: int = 2,
        kind: str = "Semivariogram",
    ) -> None:
        self.max_dist = max_dist
        self.n_bins = n_bins
        self.n_procs = n_procs
        self.kind = kind
        if self.kind == "Covariogram":
            self.covariogram = True
        else:
            self.covariogram = False


@dataclass
class EmpiricalVariogram:
    """Empirical variogram"""

    df: pd.DataFrame
    config: VarioConfig
    timestamp: str
    timedeltas: list[int]


class Field:
    """
    Stores residual data values, uncertainties, coordinates, and other spatial components
    for a single process at a fixed time.
    """

    def __init__(
        self,
        ds: Dataset,
        timestamp: Union[str, np.datetime64],
    ) -> None:
        if isinstance(timestamp, np.datetime64):
            timestamp = np.datetime_as_string(timestamp, unit="D")
        self.timestamp = timestamp
        self.data_name, self.var_name = _get_field_names(ds)

        self.ds = _preprocess_ds(ds)
        df = self.to_dataframe()

        self.coords = df[["lat", "lon"]].values.astype(np.float32)
        self.values = df[self.data_name].values
        self.size = len(self.values)
        self.spatial_trend = df["spatial_trend"].values
        self.spatial_mean = self.ds.attrs["spatial_mean"]
        self.scale_factor = self.ds.attrs["scale_factor"]
        self.variance_estimates = df[self.var_name].values / (self.scale_factor**2)
        self.measurement_error_component = np.median(df[self.var_name].values) / (
            self.scale_factor**2
        )

        # compute trend nugget as average of squared, standardized residuals
        self.trend_nugget = np.nanmean(self.values**2)
        if self.trend_nugget > self.measurement_error_component:
            self.trend_ms_variation = self.trend_nugget - self.measurement_error_component
        else:
            self.trend_ms_variation = 0

    def to_dataframe(self):
        """Converts the field dataset to a data frame."""
        return self.ds.to_dataframe().reset_index().dropna(subset=[self.data_name])

    def to_xarray(self) -> Dataset:
        """Converts the field data frame to an xarray dataset."""
        return (
            pd.DataFrame(
                {
                    "lat": self.coords[:, 0],
                    "lon": self.coords[:, 1],
                    self.data_name: self.values,
                    self.var_name: self.variance_estimates,
                }
            )
            .set_index(["lat", "lon"])
            .to_xarray()
            .assign_coords({"time": np.array(self.timestamp, dtype=np.datetime64)})
        )


class MultiField:
    """
    Stores a multivariate process, each of class Field, along with modelling attributes.

    Parameters:
        datasets: list of xarray datasets
        timestamp: the main timestamp for the MultiField, timedeltas are with reference to
            this value
        timedeltas: list of offsets (by month) with elements corresponding to each dataset
            (negative = back, positive = forward)
    """

    def __init__(
        self,
        datasets: list[Dataset],
        timestamp: Union[str, np.datetime64],
        timedeltas: list[int],
    ) -> None:
        self.datasets = datasets
        _check_length_match(datasets, timedeltas)
        self.timestamp = timestamp
        self.timedeltas = timedeltas
        self.fields = np.array(
            [
                Field(datasets[i], self._apply_timedelta(timedeltas[i]))
                for i in range(len(datasets))
            ]
        )
        self.n_procs = len(self.fields)
        self.n_data = self._count_data()

    def _apply_timedelta(self, timedelta: int) -> str:
        """Returns timestamp with month offset by timedelta as string."""
        t0 = datetime.strptime(self.timestamp, "%Y-%m-%d")
        return (t0 + relativedelta(months=timedelta)).strftime("%Y-%m-%d")

    def _count_data(self) -> int:
        """Returns the total number of data values across all fields."""
        return np.sum([f.size for f in self.fields])

    def calc_dist_matrix(
        self,
        ids: Tuple,
        indices: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Wrapper to calculate a (cross-) distance matrix from components of the MultiField
        specified by 'ids', optionally at a specified subset of indices.
        """
        assert len(ids) == 2
        if indices is not None:
            coord_list = [self.fields[i].coords[indices[i], :] for i in ids]
        else:
            coord_list = [self.fields[i].coords for i in ids]
        return distance_matrix(*coord_list)

    def _variogram_cloud(self, i: int, j: int, config: VarioConfig) -> pd.DataFrame:
        """Calculate the (cross-) variogram cloud for corresponding field id's (i, j)."""
        dist = self.calc_dist_matrix((i, j))
        if i == j:
            # marginal-variogram
            idx = np.triu_indices(dist.shape[0], k=1, m=dist.shape[1])
            dist = dist[idx]
            cloud = _cloud_calc(self.fields[[i, i]], config.covariogram)[idx]
        else:
            # cross-variogram
            dist = dist.flatten()
            cloud = _cloud_calc(self.fields[[i, j]], config.covariogram).flatten()

        assert cloud.shape == dist.shape
        return pd.DataFrame({"distance": dist, "variogram": cloud})

    def get_variogram(self, i: int, j: int, config: VarioConfig) -> pd.DataFrame:
        """
        Compute the (cross-) variogram of the specified kind for the pair of fields (i, j).
        Return as a dataframe with bin averages and bin counts.
        """
        df_cloud = self._variogram_cloud(i, j, config)

        if isinstance(config.max_dist, dict):
            max_dist = config.max_dist[f"{i+1}{j+1}"]
        else:
            max_dist = config.max_dist
        df_cloud = df_cloud[df_cloud["distance"] <= max_dist]

        bin_centers, bin_edges = _construct_variogram_bins(df_cloud, config.n_bins)
        df_cloud["bin_center"] = pd.cut(
            df_cloud["distance"], bin_edges, labels=bin_centers, include_lowest=True
        )
        df = (
            df_cloud.groupby("bin_center")["variogram"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "bin_mean", "count": "bin_count"})
            .reset_index()
        )
        # convert bins from categories to numeric
        df["bin_center"] = df["bin_center"].astype("string").astype("float")
        if (df["bin_count"] < 30).any():
            warnings.warn(
                "WARNING: Fewer than 30 pairs used for at least one bin in variogram"
                " calculation."
            )
        # establish multi-index using field id's
        df["i"], df["j"] = i, j
        return df.set_index(["i", "j", df.index])

    def empirical_variograms(self, config: VarioConfig) -> EmpiricalVariogram:
        """
        Compute empirical variogram of the specified kind for each field in the MultiField and
        the cross-variogram of the specified kind for each pair of fields.

        Parameters:
            max_dist: maximum distance (in units corresponding to the MultiField) across which
                variograms will be computed
            n_bins: number of bins to use when averaging variogram cloud values
            kind: either `semivariogram` (default) or `covariogram`
        Returns:
            df: multi-index dataframe with first two indices corresponding to the field ids used
                in the calculation
        """
        variograms = [
            self.get_variogram(i, j, config)
            for i in range(self.n_procs)
            for j in range(self.n_procs)
            if i <= j
        ]
        return EmpiricalVariogram(
            pd.concat(variograms), config, self.timestamp, self.timedeltas
        )


def _check_length_match(*args):
    """Check that each input list has the same length."""
    if len({len(i) for i in args}) != 1:
        raise ValueError("Not all lists have the same length")


def _get_field_names(ds: Dataset):
    """Returns data and estimated variance names from dataset."""
    var_name = [name for name in list(ds.keys()) if "_var" in name][0]
    data_name = var_name.replace("_var", "")
    return data_name, var_name


def get_group_ids(group: pd.DataFrame):
    """Returns the group ids as a tuple (i, j)."""
    i = group.index.get_level_values("i")[0]
    j = group.index.get_level_values("j")[0]
    return i, j


def distance_matrix(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Computes the chordal distance among all pairs of points given two sets of coordinates.

    NOTE:
    - points should be formatted in rows as [lat, lon]
    """
    # enforce 2d array in case of single point
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    # chordal distances in kilometers
    X1_r = np.deg2rad(X1)
    X2_r = np.deg2rad(X2)
    return 2 * EARTH_RADIUS * np.sin(haversine_distances(X1_r, X2_r) / 2.0)


def fit_ols_basis(ds: Dataset, data_name: str) -> Dataset:
    """
    Fit and predict the mean surface using ordinary least squares with bisquare
    basis covariates.
    """
    df = ds.to_dataframe().drop(columns=[f"{data_name}_var"]).dropna().reset_index()
    if df.shape[0] == 0:  # no data
        return ds[data_name] * np.nan

    # fit and predict
    basis_columns = [col for col in df if col.startswith("B")]
    X = df.loc[:, basis_columns].values
    y = df[data_name].values
    model = LinearRegression().fit(X, y)
    df_pred = df.loc[:, ["lat", "lon"]]
    df_pred["ols_mean"] = model.predict(X)
    ds_pred = df_pred.set_index(["lat", "lon"]).to_xarray()

    return ds_pred["ols_mean"], model


def _preprocess_ds(ds: Dataset) -> Dataset:
    """Apply data transformations and compute surface mean and standard deviation."""
    if not set(ds.dims).issubset(set(["lat", "lon", "x", "y"])):
        raise ValueError(f"Dataset must have spatial dimensions only; got {ds.sizes}")

    data_name, _ = _get_field_names(ds)
    ds_field = ds.copy().transpose("lat", "lon")

    # Remove the spatial trend by OLS
    ds_field["spatial_trend"], ds_field.attrs["spatial_model"] = fit_ols_basis(
        ds_field, data_name
    )
    ds_field[data_name] = ds_field[data_name] - ds_field["spatial_trend"]

    # Standardize the residuals
    ds_field.attrs["spatial_mean"] = np.nanmean(ds_field[data_name].values)
    ds_field.attrs["scale_factor"] = np.nanstd(ds_field[data_name].values, ddof=1)
    ds_field[data_name] = (
        ds_field[data_name] - ds_field.attrs["spatial_mean"]
    ) / ds_field.attrs["scale_factor"]

    return ds_field


def _cloud_calc(fields: list[Field], covariogram: bool) -> np.ndarray:
    """Calculate the semivariogram or covariogram for all point pairs."""
    center = lambda f: f.values - f.values.mean()
    residuals = [center(f) for f in fields]
    if covariogram:
        cloud = np.multiply.outer(*residuals)
    else:
        cloud = 0.5 * (np.subtract.outer(*residuals)) ** 2
    return cloud


def _construct_variogram_bins(
    df_cloud: pd.DataFrame, n_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partitions the domain of a variogram cloud into `n_bins` bins; first bin extended to zero.
    """
    # use min non-zero dist for consistency between variograms and cross-variograms
    min_dist = df_cloud[df_cloud["distance"] > 0]["distance"].min()
    max_dist = df_cloud["distance"].max()
    bin_centers = np.linspace(min_dist, max_dist, n_bins)
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = np.arange(min_dist - 0.5 * bin_width, max_dist + bin_width, bin_width)
    # check that bin centers are actually centered
    if not np.allclose((bin_edges[1:] + bin_edges[:-1]) / 2, bin_centers):
        warnings.warn("WARNING: variogram bins are not centered.")
    # extend first bin to origin
    bin_edges[0] = 0
    return bin_centers, bin_edges
