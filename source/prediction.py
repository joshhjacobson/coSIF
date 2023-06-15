from typing import Optional, Union, Tuple, List, Dict
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from scipy.linalg import cho_factor, cho_solve, LinAlgError


from fields import MultiField, distance_matrix
from model import MultivariateMatern

import logging


logging.basicConfig(
    format="%(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s", level=logging.INFO
)


def get_prediction_locations() -> pd.DataFrame:
    """Produce prediction coordinates (land only) using MODIS LCC-derived binary land mask."""
    with xr.open_dataset("../data/intermediate/land_cover_north_america.nc4") as ds:
        df = ds["land_cover"].to_dataframe().reset_index()
        df = df.where(df["land_cover"] == 1).dropna()[["lat", "lon"]].reset_index(drop=True)
        logging.info(f"Number of prediction locations: {df.shape[0]}")
    return df


class Predictor:
    """Multivariate spatial prediction (cokriging) framework."""

    def __init__(self, model: MultivariateMatern, mf: MultiField) -> None:
        if model.n_procs != mf.n_procs:
            raise ValueError(
                f"Number of theoretical processes ({model.n_procs}) different from number of "
                f"empirical processes ({mf.n_procs})."
            )
        self.n_procs = model.n_procs
        self.model = deepcopy(model)
        self.mf = deepcopy(mf)

        # initialize internal attributes
        self._i = None  # index of process being predicted
        self._validation = False  # flag indicating whether predictions are for validation

    def predict(
        self,
        i: int,
        pcoords: pd.DataFrame,
        ds_covariates: Optional[xr.Dataset] = None,
        num_local_values: Optional[int] = 150,
    ) -> xr.Dataset:
        """
        Apply (co)kriging prediction at each location in the specified prediction coordinates.

        Parameters:
            i: index of the process in the MultiField to be predicted
            pcoords: prediction coordinates with columns [lat, lon]
            ds_covariates: dataset of prediction covariates; coordinates in this dataset must
                align  with those given in pcoords
            num_local_values: the number of data values to use in prediction at each location;
                values closest to the location of interest will be collected

        Returns:
            dataset of predicted values, prediction standard errors, and other components
        """
        self._i = i
        self._remove_measurement_error_from_nugget()
        c0 = self.model.covariance(self._i, 0, use_nugget=True)[0]

        # format prediction locations as dask data array
        da_pcoords = xr.DataArray(
            data=da.from_array(pcoords[["lat", "lon"]].values, chunks=(1000, 2)),
            coords={"s": pcoords.index, "coordinates": ["lat", "lon"]},
        )

        # compute predictions and prediction standard errors (on the standardized scale)
        # at each location in parallel
        ds_predictions = xr.apply_ufunc(
            self._point_prediction,
            da_pcoords,
            input_core_dims=[["coordinates"]],
            kwargs={"c0": c0, "num_local_values": num_local_values},
            output_core_dims=[["prediction_values"]],
            output_dtypes=[float],
            dask_gufunc_kwargs=dict(output_sizes={"prediction_values": 3}),
            vectorize=True,
            dask="parallelized",
        )

        # align the predictions with prediction locations
        ds_predictions = (
            ds_predictions.to_dataframe()
            .unstack("prediction_values")
            .rename_axis(None, axis=0)
            .set_axis(["residual_predictions", "residual_uncertainty", "validity_flag"], axis=1)
            .join(pcoords)
            .set_index(["lat", "lon"])
            .to_xarray()
        )

        # check validity flags
        if not np.all(ds_predictions["validity_flag"].values):
            count = np.sum(1 - ds_predictions["validity_flag"].values)
            logging.warning(f"{count} predictions are not valid.")

        # reverse the standardization
        ds_predictions = self._postprocess_predictions(ds_predictions, ds_covariates)

        # add the timestamp and timedelta to the dataset
        ds_predictions = ds_predictions.assign_coords(
            coords={"time": self.mf.timestamp}
        ).assign_attrs(timedeltas=self.mf.timedeltas)

        return ds_predictions.assign_attrs(num_local_values=num_local_values)

    def validation_prediction(
        self,
        i: int,
        pcoords: pd.DataFrame,
        df_me_variance: pd.DataFrame,
        ds_covariates: Optional[xr.Dataset] = None,
        num_local_values: Optional[int] = 150,
    ) -> Dict[str, Union[xr.Dataset, np.ndarray]]:
        """
        Apply validation prediction at each validation location.

        Parameters:
            i: index of the process in the MultiField to be predicted
            pcoords: validation prediction coordinates with columns [lat, lon]
            df_me_variance: dataframe of measurement error variances at
                prediction locations
            ds_covariates: dataset of prediction covariates; coordinates in this dataset must
                align  with those given in pcoords
            num_local_values: the number of data values to use in prediction at each location;
                values closest to the location of interest will be collected

        Returns:
            dataset of predicted values, prediction standard errors, and other components
        """
        self._i = i
        self._validation = True
        self._pcoords = pcoords.copy()
        self._df_me_variance = df_me_variance.copy()
        self._withhold_data_validation(pcoords)
        self._remove_measurement_error_from_nugget()

        c0 = self.model.covariance(self._i, 0, use_nugget=True)[0]

        # format prediction locations as dask data array
        da_pcoords = xr.DataArray(
            data=da.from_array(pcoords[["lat", "lon"]].values),
            coords={"s": pcoords.index, "coordinates": ["lat", "lon"]},
        )

        # compute predictions and prediction standard errors (on the standardized scale)
        # at each location in parallel
        da_prediction_components = xr.apply_ufunc(
            self._point_prediction,
            da_pcoords,
            input_core_dims=[["coordinates"]],
            kwargs={"c0": c0, "num_local_values": num_local_values},
            output_core_dims=[["components", "items"]],
            output_dtypes=[float],
            dask_gufunc_kwargs=dict(
                output_sizes={"components": num_local_values * self.n_procs, "items": 3}
            ),
            vectorize=True,
            dask="parallelized",
        ).compute()

        # align the predictions with prediction locations
        da_predictions = da_prediction_components[:, :3, 0]
        ds_predictions = (
            da_predictions.to_dataframe()
            .unstack("components")
            .rename_axis(None, axis=0)
            .set_axis(["residual_predictions", "residual_uncertainty", "validity_flag"], axis=1)
            .join(pcoords)
            .set_index(["lat", "lon"])
            .to_xarray()
        )

        # check validity flags
        if not np.all(ds_predictions["validity_flag"].values):
            count = np.sum(1 - ds_predictions["validity_flag"].values)
            logging.warning(f"{count} predictions are not valid.")

        # reverse the standardization
        ds_predictions = self._postprocess_predictions(ds_predictions, ds_covariates)

        # add the timestamp and timedelta to the dataset
        ds_predictions = ds_predictions.assign_coords(
            coords={"time": self.mf.timestamp}
        ).assign_attrs(timedeltas=self.mf.timedeltas)

        # collect multivariate predictive distribution components
        predictive_mean = (
            ds_predictions[["lat", "lon", "predictions"]]
            .to_dataframe()
            .dropna()
            .reset_index()
            .loc[:, "predictions"]
            .values
        )
        predictive_covariance_matrix = self._compute_prediction_covariance_matrix(
            da_prediction_components, num_local_values
        )

        return {
            "ds": ds_predictions.assign_attrs(num_local_values=num_local_values),
            "predictive_mean": predictive_mean,
            "predictive_covariance_matrix": predictive_covariance_matrix,
        }

    def _withhold_data_validation(self, pcoords: pd.DataFrame) -> None:
        """
        For validation prediction, remove data at prediction locations from the field being
        predicted.
        """
        field = self.mf.fields[self._i]
        df = field.to_dataframe().loc[:, ["lat", "lon", field.data_name]]
        # identify coordinates to keep by performing a pseudo set difference on coordinates
        df_keep = pd.concat([pcoords, df.loc[:, ["lat", "lon"]]]).drop_duplicates(keep=False)
        # recollect data at retained locations
        df_keep = df_keep.merge(df, how="left")
        # update values on self
        self.mf.fields[self._i].coords = df_keep.loc[:, ["lat", "lon"]].values
        self.mf.fields[self._i].values = df_keep.loc[:, field.data_name].values

    def _remove_measurement_error_from_nugget(self) -> None:
        """
        During fitting, the nugget is treated as a combination of micro-scale variance and
        measurement error variance. In prediction, we know the latter and it needs to be
        incorporated. This method removes the measurement error variance from the nugget so
        it can be handled separately in prediction.
        """
        nuggets = np.copy(self.model.params.nugget.get_values())

        for i, nugget in enumerate(nuggets):
            if nugget > self.mf.fields[i].measurement_error_component:
                # nugget contains micro-scale and measurement error variance; remove the latter
                nuggets[i] = nugget - self.mf.fields[i].measurement_error_component
            else:
                # nugget is all measurement error variance; but should only represent
                # micro-scale variance (and is thus null)
                nuggets[i] = 0

        self.model.params.nugget.set_values(nuggets)

    def _compute_local_block_covariance(
        self, indices: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute the upper-triangular blocks of the data block-covariance matrix, with each
        block describing the dependence within a process or between processes; lower
        off-diagonal blocks are taken as the transpose of the corresponding upper-triangular
        block.
        """
        blocks = {}
        for i in range(self.n_procs):
            for j in range(self.n_procs):
                if i <= j:
                    # compute the distances between the local data locations
                    h = self.mf.calc_dist_matrix((i, j), indices=indices)
                    if i == j:
                        blocks[f"{i}{j}"] = self.model.covariance(i, h, use_nugget=True)
                        # add spatially varying measurement-error variance to diagonal
                        blocks[f"{i}{j}"] += np.diag(
                            self.mf.fields[i].variance_estimates[indices[i]]
                        )
                    else:
                        blocks[f"{i}{j}"] = self.model.cross_covariance(i, j, h)
                else:
                    # blocks in lower-triangle are the transpose of the upper-triangle
                    blocks[f"{i}{j}"] = blocks[f"{j}{i}"].T
        return blocks

    def _prediction_covariance(self, distances: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute the covariance and cross-covariance vectors for a set of local distances about
        a prediction location.
        """
        covariance_vectors = []
        for j in range(self.n_procs):
            if self._i == j:
                # measurement error has been removed from the nugget and anything remaining
                # is micro-scale variation
                covariance_vectors.append(
                    self.model.covariance(self._i, distances[self._i], use_nugget=True)
                )
            else:
                covariance_vectors.append(self.model.cross_covariance(self._i, j, distances[j]))
        return covariance_vectors

    def _get_local_distance_idx(
        self, s0: np.ndarray, num_local_values: int = 150
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Determine the indices and corresponding distances of the num_local_values data locations
        with the shortest distances to the specified prediction location `s0` for each dataset.
        """
        distance_vectors = [
            distance_matrix(s0, field.coords).squeeze() for field in self.mf.fields
        ]

        # get the indices of the num_local_values smallest distances from each vector
        # NOTE: distance may not be sorted in ascending order, but algorithm is O(n)
        indices = [
            np.argpartition(d, num_local_values)[:num_local_values] for d in distance_vectors
        ]

        # collect the corresponding distances
        distances = [d[idx] for d, idx in zip(distance_vectors, indices)]

        return indices, distances

    def _get_local_values(
        self, s0: np.ndarray, num_local_values: int = 150
    ) -> Dict[str, np.ndarray]:
        """
        Collect the data vectors, prediction covariance vectors, and block covariance matrix
        for the num_local_values locations with the shortest distances to the specified
        prediction location `s0`.
        """
        # get the indices and corresponding distances of the locations closest to s0
        local_indices, local_distances = self._get_local_distance_idx(s0, num_local_values)
        indices_vector = np.hstack(local_indices)

        # collect prediction components at the identified locations
        covariance_blocks = self._compute_local_block_covariance(local_indices)
        block_covariance = np.block(
            [
                [covariance_blocks[f"{i}{j}"] for j in range(self.n_procs)]
                for i in range(self.n_procs)
            ]
        )
        covariance_vector = np.hstack(self._prediction_covariance(local_distances))
        data_vector = np.hstack(
            [self.mf.fields[i].values[local_indices[i]] for i in range(self.n_procs)]
        )

        return {
            "data_vector": data_vector,
            "covariance_vector": covariance_vector,
            "block_covariance": block_covariance,
            "indices_vector": indices_vector,
        }

    def _prediction_with_uncertainty(
        self,
        c0: float = None,
        covariance_vector: np.ndarray = None,
        block_covariance: np.ndarray = None,
        data_vector: np.ndarray = None,
        indices_vector: np.ndarray = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Local prediction and uncertainty calculation."""
        try:
            covariance_weights = cho_solve(
                cho_factor(
                    block_covariance.copy(), lower=True, overwrite_a=True, check_finite=False
                ),
                covariance_vector.copy(),
                overwrite_b=True,
                check_finite=False,
            ).T
        except LinAlgError:
            logging.warning("Local covariance matrix not positive definite.")
            return np.array([0.0, 0.0, 0])

        prediction = np.matmul(covariance_weights, data_vector)
        prediction_uncertainty = c0 - np.matmul(covariance_weights, covariance_vector)

        validity_flag = self._verify_model(c0, covariance_vector, block_covariance)

        prediction_values = np.array([prediction, prediction_uncertainty, validity_flag])

        if self._validation:
            prediction_info = np.zeros_like(covariance_weights)
            prediction_info[:3] = prediction_values
            return np.column_stack([prediction_info, covariance_weights, indices_vector])
        else:
            return prediction_values

    @staticmethod
    def _verify_model(
        c0: float = None,
        covariance_vector: np.ndarray = None,
        block_covariance: np.ndarray = None,
    ) -> float:
        """
        Check that the joint covariance matrix for a given prediction location is positive
        definite by attempting a Cholesky decomposition. Return 1 if valid, 0 else.
        """
        matrix = np.block(
            [
                [np.atleast_2d(c0), np.atleast_2d(covariance_vector)],
                [np.atleast_2d(covariance_vector).T, block_covariance],
            ]
        )
        try:
            cho_factor(matrix, overwrite_a=True)
            flag = 1
        except LinAlgError:
            flag = 0
        return flag

    def _point_prediction(
        self, s0: np.ndarray, c0: float = None, num_local_values: int = 150
    ) -> Tuple[float, float]:
        """
        Wrapper to compute the predicted value with uncertainty at the specified
        location.
        """
        local_values = self._get_local_values(s0, num_local_values)
        return self._prediction_with_uncertainty(c0=c0, **local_values)

    def _postprocess_predictions(
        self, ds_predictions: xr.Dataset, ds_covariates: xr.Dataset = None
    ) -> xr.Dataset:
        """
        Compute the large-scale trend at prediction locations and reverse standardization in
        residual predictions to obtain process level predictions.
        """
        field = self.mf.fields[self._i]

        # transform predictions and errors to original data scale
        da_scaled_predictions = (
            ds_predictions["residual_predictions"] * field.scale_factor + field.spatial_mean
        )
        da_scaled_uncertainty = ds_predictions["residual_uncertainty"] * field.scale_factor**2

        # format tabular data
        df = (
            ds_predictions.merge(ds_covariates, join="left")
            .to_dataframe()
            .reset_index()
            .dropna()
        )

        # get the trend surface
        basis_columns = [col for col in df if col.startswith("B")]
        covariates = df.loc[:, basis_columns].values
        ols_model = field.ds.attrs["spatial_model"]
        df["trend_surface"] = ols_model.predict(covariates)

        if self._validation:
            # include measurement error as a data variable
            df = df.merge(self._df_me_variance, how="left")

        # combine the large-scale and small-scale terms to produce predictions
        ds_postprocessed = df.set_index(["lat", "lon"]).to_xarray()
        ds_postprocessed["predictions"] = (
            ds_postprocessed["trend_surface"] + da_scaled_predictions
        )
        ds_postprocessed["scaled_residuals"] = da_scaled_predictions

        if self._validation:
            trend_ms_variation = field.trend_ms_variation * field.scale_factor**2
            # add measurement error variance to process-scale prediction errors
            ds_postprocessed["trend_surface_rmspe"] = np.sqrt(
                trend_ms_variation + ds_postprocessed["me_variance"].fillna(0)
            )
            ds_postprocessed["rmspe"] = np.sqrt(
                da_scaled_uncertainty + ds_postprocessed["me_variance"].fillna(0)
            )
        else:
            ds_postprocessed["rmspe"] = np.sqrt(da_scaled_uncertainty)

        return ds_postprocessed

    def _compute_prediction_covariance_matrix(
        self,
        da_prediction_components: xr.DataArray,
        num_local_values: int,
    ) -> np.ndarray:
        """
        Compute the conditional covariance matrix described in Appendix 3 by applying
        Equation (A.18) in parallel across all pairs of validation locations (a, b).
        """
        logging.info("Running predictive covariance")
        da_covariance_weights = da_prediction_components[:, :, 1]
        da_data_indices = da_prediction_components[:, :, 2].astype(int)

        # initialize empty covariance matrix
        num_predictions = self._pcoords.shape[0]
        covariance_matrix = np.zeros((num_predictions, num_predictions))

        # iterate over the upper-triangular indices and compute elements

        # format pairs of indices as data array
        pairs_list = np.triu_indices_from(covariance_matrix)
        da_ids = xr.DataArray(
            data=da.from_array(np.column_stack(pairs_list), chunks=(1000, 2)),
            coords={"ids": np.arange(pairs_list[0].size), "location": ["a", "b"]},
        )

        da_covariance_elements = xr.apply_ufunc(
            self._compute_predictive_covariance_element,
            da_ids,
            input_core_dims=[["location"]],
            kwargs={
                "da_covariance_weights": da_covariance_weights,
                "da_data_indices": da_data_indices,
                "num_local_values": num_local_values,
            },
            output_dtypes=[float],
            vectorize=True,
            dask="parallelized",
        )
        # assign upper-triangular elements
        covariance_matrix[pairs_list] = da_covariance_elements.values

        # fill in the lower-triangular elements to complete the symmetric matrix
        lower_indices = np.tril_indices_from(covariance_matrix, k=-1)
        covariance_matrix[lower_indices] = covariance_matrix.T[lower_indices]

        # rescale the covariance matrix to match the scale of the data;
        # Equation (A.15) of Appendix 3
        # NOTE: standardized measurement-error variance added in element-wise calculation
        covariance_matrix *= self.mf.fields[self._i].scale_factor ** 2

        # test that covariance matrix is positive definite
        try:
            cho_factor(covariance_matrix.copy(), overwrite_a=True, check_finite=False)
        except LinAlgError:
            logging.warning("Predictive covariance matrix is not positive definite.")

        return covariance_matrix

    def _compute_predictive_covariance_element(
        self,
        ids: xr.DataArray,
        da_covariance_weights: xr.DataArray = None,
        da_data_indices: xr.DataArray = None,
        num_local_values: int = None,
    ) -> float:
        """
        Compute the result in Equation (A.18) of Appendix 3, the conditional covariance
        between locations s_a and s_b (on the standardized scale).
        """
        a, b = ids
        if a == b and a % 100 == 0:
            # status update every hundredth diagonal element
            logging.info(f"Status: a = {a} ...")

        # need the indices for a and b, for each process
        indices_list = [
            [
                da_data_indices[a, num_local_values * k : num_local_values * (k + 1)].values,
                da_data_indices[b, num_local_values * k : num_local_values * (k + 1)].values,
            ]
            for k in range(self.n_procs)
        ]

        # build the covariance matrix for the combination of data locations used for a and b;
        # see Equation (A.20) in Appendix 3
        blocks = {}
        for i in range(self.n_procs):
            for j in range(self.n_procs):
                if i <= j:
                    # collect array of coordinates used for each of a and b
                    coord_list = [
                        self.mf.fields[i].coords[indices_list[i][0], :],
                        self.mf.fields[j].coords[indices_list[j][1], :],
                    ]
                    # compute the distances between the local data locations
                    h = distance_matrix(*coord_list)
                    if i == j:
                        blocks[f"{i}{j}"] = self.model.covariance(i, h, use_nugget=True)
                        # NOTE: since i = j, the process is the same; along the diagonal of
                        # h, the location is also the same (i.e., a = b)

                        # add spatially varying measurement error variance to diagonal
                        blocks[f"{i}{j}"] += np.diag(
                            self.mf.fields[i].variance_estimates[indices_list[i][0]]
                        )
                    else:
                        blocks[f"{i}{j}"] = self.model.cross_covariance(i, j, h)
                else:
                    # blocks in lower-triangle are the transpose of the upper-triangle
                    blocks[f"{i}{j}"] = blocks[f"{j}{i}"].T

        block_covariance = np.block(
            [[blocks[f"{i}{j}"] for j in range(self.n_procs)] for i in range(self.n_procs)]
        )

        # compute the local covariance vectors; Equation (A.19) in Appendix 3
        covariance_vectors = []
        for l, k in zip((a, b), (1, 0)):
            local_distances = [
                distance_matrix(
                    self._pcoords.iloc[l, :].values, field.coords[indices_list[i][k]]
                ).squeeze()
                for i, field in enumerate(self.mf.fields)
            ]
            covariance_vectors.append(np.hstack(self._prediction_covariance(local_distances)))

        # compute the covariance between locations a and b
        if a == b:
            # NOTE: measurement-error variance added here
            c_me = (
                self._df_me_variance["me_variance"].iloc[a]
                / (self.mf.fields[self._i].scale_factor) ** 2
            )
            c0 = self.model.covariance(self._i, 0, use_nugget=True)[0] + c_me
        else:
            distance = distance_matrix(
                self._pcoords.iloc[a, :].values, self._pcoords.iloc[b, :].values
            )
            c0 = self.model.covariance(self._i, distance, use_nugget=False)[0]

        # calculation in last line of Equation (A.18)
        return (
            c0
            - np.dot(da_covariance_weights[a, :].values, covariance_vectors[1])
            - np.dot(covariance_vectors[0], da_covariance_weights[b, :].values)
            + np.dot(
                np.dot(da_covariance_weights[a, :].values, block_covariance),
                da_covariance_weights[b, :].values,
            )
        )
