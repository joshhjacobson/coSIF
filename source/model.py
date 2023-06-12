import warnings
from typing import Optional, Union, List, Tuple

from numba import njit
import numpy as np
import pandas as pd
import scipy.special as sps
from scipy.optimize import minimize

from fields import EmpiricalVariogram, get_group_ids


class MarginalParam:
    """Multivariate marginal Matern covariance parameter."""

    def __init__(
        self, name: str, default: float, bounds: Union[Tuple, List[Tuple]], n_procs: int = 2
    ) -> None:
        self.name = name
        self.n_procs = n_procs
        self.default = default
        self.bounds = bounds
        self.values = np.nan * np.zeros((n_procs, n_procs))
        np.fill_diagonal(self.values, default)

    def get_names(self):
        return [f"{self.name}_{i+1}{i+1}" for i in range(self.n_procs)]

    def get_values(self):
        return self.values.diagonal()

    def set_values(self, x: np.ndarray):
        np.fill_diagonal(self.values, x)
        return self

    def reset_values(self):
        np.fill_diagonal(self.values, self.default)
        return self

    def count_params(self):
        return len(self.get_values())

    def to_dataframe(self):
        df = (
            pd.DataFrame.from_dict(
                dict(zip(self.get_names(), self.get_values())),
                orient="index",
                columns=["value"],
            )
            .reset_index()
            .rename(columns={"index": "name"})
        )
        if isinstance(self.bounds, list):
            assert len(self.bounds) == self.count_params()
            df["bounds"] = self.bounds
        else:
            df["bounds"] = [self.bounds] * len(df)
        return df


class CrossParam(MarginalParam):
    """Multivariate Matern covariance parameter with cross dependence."""

    def __init__(self, name: str, default: float, bounds: tuple, n_procs: int = 2) -> None:
        super().__init__(name, default, bounds, n_procs=n_procs)
        self._triu_index = np.triu_indices(n_procs)
        self.values[self._triu_index] = default

    def get_names(self):
        return [
            f"{self.name}_{i+1}{j+1}"
            for i in range(self.n_procs)
            for j in range(self.n_procs)
            if i <= j
        ]

    def get_values(self):
        return self.values[self._triu_index]

    def set_values(self, x: np.ndarray):
        self.values[self._triu_index] = x
        return self

    def reset_values(self):
        self.values[self._triu_index] = self.default
        return self


class RhoParam(CrossParam):
    """Multivariate Matern covariance parameter with cross dependence only."""

    def __init__(self, name: str, default: float, bounds: tuple, n_procs: int = 2) -> None:
        super().__init__(name, default, bounds, n_procs=n_procs)
        self._triu_index = np.triu_indices(n_procs, k=1)
        self.values[self._triu_index] = default

    def get_names(self):
        return [
            f"{self.name}_{i+1}{j+1}"
            for i in range(self.n_procs)
            for j in range(self.n_procs)
            if i < j
        ]


class MaternParams:
    """
    Multivariate Matern covariance parameters.

    Formulation:
        sigma: process specific standard deviation
        nu: process specific smoothness
        len_scale: process specific length scale
        nugget: process specific squared nugget (i.e., tau^2)
        rho: cross-correlation coefficient
    """

    def __init__(self, n_procs: int = 2) -> None:
        self.n_procs = n_procs
        self.sigma = MarginalParam("sigma", 1.0, (0.4, 3.5), n_procs=n_procs)
        self.nu = CrossParam("nu", 1.5, (0.2, 2.5), n_procs=n_procs)
        self.len_scale = CrossParam("len_scale", 100, (10.0, 500), n_procs=n_procs)
        self.nugget = MarginalParam("nugget", 0.1, (0.0, 0.9), n_procs=n_procs)
        if n_procs == 1:
            self.rho = RhoParam("rho", np.nan, (-1.0, 1.0), n_procs=n_procs)
        else:
            self.rho = RhoParam("rho", -0.05, (-1.0, -1e-8), n_procs=n_procs)
        self._params = [self.sigma, self.nu, self.len_scale, self.nugget, self.rho]
        self.n_params = 0
        for p in self._params:
            self.n_params += p.count_params()

    def to_dataframe(self):
        df_list = [p.to_dataframe() for p in self._params]
        return pd.concat(df_list, ignore_index=True)

    def get_names(self):
        return self.to_dataframe()["name"].values

    def get_values(self):
        return self.to_dataframe()["value"].values

    def set_values(self, x: np.ndarray):
        if len(x) != self.n_params:
            raise ValueError("Incorrect number of parameters in input array.")
        for p in self._params:
            n_params = p.count_params()
            vals, x = x[:n_params], x[n_params:]
            p.set_values(vals)
        return self

    def reset_values(self):
        for p in self._params:
            p.reset_values()
        return self

    def get_bounds(self):
        return self.to_dataframe()["bounds"].values

    def set_bounds(self, **kwargs):
        for name, bounds in kwargs.items():
            try:
                param = getattr(self, name)
                param.bounds = bounds
            except AttributeError:
                raise AttributeError(f"`{name}` is not a valid parameter.")
        return self


class MultivariateMatern:
    """
    Multivariate Matern covariance model (Gneiting et al., 2010).

    NOTE:
    - Parameterization follows Rassmussen and Williams (2006; see MaternParams)
    """

    def __init__(
        self, n_procs: int = 2, params: Optional[Union[np.ndarray, MaternParams]] = None
    ) -> None:
        self.n_procs = n_procs
        if params is None:
            self.params = MaternParams(n_procs=n_procs)
        elif isinstance(params, MaternParams):
            self.params = params
        elif isinstance(params, np.ndarray):
            self.params = MaternParams(n_procs=n_procs).set_values(params)
        else:
            raise ValueError("Invalid parameters type.")

        self.fit_result = None

    def correlation(self, i: int, j: int, h: np.ndarray) -> np.ndarray:
        nu = self.params.nu.values[i, j]
        len_scale = self.params.len_scale.values[i, j]
        return _matern_correlation(nu, len_scale, h)

    def covariance(self, i: int, h: np.ndarray, use_nugget: bool = True) -> np.ndarray:
        cov = self.params.sigma.values[i, i] ** 2 * self.correlation(i, i, h)
        if use_nugget:
            cov[h == 0] += self.params.nugget.values[i, i]
        return cov

    def cross_covariance(self, i: int, j: int, h: np.ndarray) -> np.ndarray:
        if i > j:
            # swap indices (cross-covariance is symmetric)
            i, j = j, i
        return (
            self.params.rho.values[i, j]
            * np.nanprod(self.params.sigma.values)
            * self.correlation(i, j, h)
        )

    def semivariance(self, i: int, h: np.ndarray) -> np.ndarray:
        return (
            self.params.sigma.values[i, i] ** 2 * (1.0 - self.correlation(i, i, h))
            + self.params.nugget.values[i, i]
        )

    def cross_semivariance(self, i: int, j: int, h: np.ndarray) -> np.ndarray:
        if i > j:
            # swap indices (cross-covariance is symmetric)
            i, j = j, i
        sill = 0.5 * np.nansum(self.params.sigma.values**2 + self.params.nugget.values)
        return sill - self.cross_covariance(i, j, h)

    def get_variogram(self, i: int, j: int, h: np.ndarray, kind: str) -> pd.DataFrame:
        if kind == "covariogram":
            if i == j:
                v = self.covariance(i, h)
            else:
                v = self.cross_covariance(i, j, h)
        else:
            if i == j:
                v = self.semivariance(i, h)
            else:
                v = self.cross_semivariance(i, j, h)
        df = pd.DataFrame({"distance": h, "variogram": v, "i": i, "j": j})
        return df.set_index(["i", "j", df.index])

    def variograms(self, h: np.ndarray, kind: str = "semivariogram") -> pd.DataFrame:
        """
        Produce modeled variograms and cross-variogram(s) of the specified kind for the
        given separation distances.
        """
        variograms = [
            self.get_variogram(i, j, h, kind)
            for i in range(self.n_procs)
            for j in range(self.n_procs)
            if i <= j
        ]
        return pd.concat(variograms)

    def _map_fit(self, df_group: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a new `fit` column with the semivariogram model evaluated at `bin_centers`.
        """
        i, j = get_group_ids(df_group)
        if i == j:
            df_group["fit"] = self.semivariance(i, df_group["bin_center"].values)
        else:
            df_group["fit"] = self.cross_semivariance(i, j, df_group["bin_center"].values)
        return df_group

    def _composite_wls(self, p, df_vario: pd.DataFrame) -> float:
        """Composite WLS cost function."""
        self.params.set_values(p)
        df_vario = df_vario.groupby(level=[0, 1], group_keys=False).apply(self._map_fit)
        ydata, yfit, counts = df_vario[["bin_mean", "fit", "bin_count"]].T.values
        non_zero = np.argwhere(yfit != 0.0)
        return _wls(ydata[non_zero], yfit[non_zero], counts[non_zero])

    def fit(self, estimate: EmpiricalVariogram, guess: MaternParams = None):
        """
        Fit the model parameters to empirical (cross-) semivariograms *simultaneously* using
        composite weighted least squares.

        Parameters:
            estimate: EmpiricalVariogram to which the model will be fit
            guess: initial guess for the parameters, can also set the bounds here
        Reference: Extension of Cressie (1985)
        """
        if estimate.config.n_procs != self.n_procs:
            raise ValueError(
                "Number of theoretical processes different from empirical processes."
            )
        if guess is None:
            init_params = self.params.reset_values().get_values()
        else:
            init_params = self.params.get_values()
            bound_settings = dict()
            for p in guess._params:
                bound_settings[p.name] = p.bounds
            self.params.set_bounds(**bound_settings)
        bounds = self.params.get_bounds()
        optim_result = minimize(
            self._composite_wls,
            init_params,
            args=(estimate.df),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if optim_result.success == False:
            warnings.warn("ERROR: optimization did not converge.")
        self.params.set_values(optim_result.x)
        self.fit_result = FittedVariogram(self, estimate, optim_result.fun)
        return self


class FittedVariogram:
    """Model parameters and theoretical variogram for the corresponding empirical variogram."""

    def __init__(
        self, model: MultivariateMatern, estimate: EmpiricalVariogram, cost: float
    ) -> None:
        self.config = estimate.config
        self.timestamp = estimate.timestamp
        self.timedeltas = estimate.timedeltas
        self.df_empirical = estimate.df
        h = np.linspace(0, self.df_empirical["bin_center"].max(), 100)
        self.df_theoretical = model.variograms(h)
        self.params = model.params
        self.cost = cost


def _matern_correlation(nu: float, len_scale: float, h: np.ndarray) -> np.ndarray:
    r"""
    Matern correlation function.

    Parameters:
        nu: smoothness parameter
        len_scale: length scale parameter
        h: array of spatial separation distances (lags), e.g., distance matrix or vector

    Returns:
        array of correlations with dimension = shape(h)

    .. math::
        \rho(h) =
        \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
        \left(\sqrt{2\nu}\cdot\frac{h}{\ell}\right)^{\nu} \cdot
        \mathrm{K}_{\nu}\left(\sqrt{2\nu}\cdot\frac{h}{\ell}\right)
    """
    h = np.atleast_1d(np.abs(h))
    # calculate by log-transformation to prevent numerical errors
    h_positive_scaled = h[h > 0.0] / len_scale
    corr = np.ones_like(h)
    corr[h > 0.0] = np.exp(
        (1.0 - nu) * np.log(2)
        - sps.gammaln(nu)
        + nu * np.log(np.sqrt(2.0 * nu) * h_positive_scaled)
    ) * sps.kv(nu, np.sqrt(2.0 * nu) * h_positive_scaled)

    # correction for non-finite correlations
    corr[np.logical_not(np.isfinite(corr))] = 0.0
    # Matern correlation is positive
    corr = np.maximum(corr, 0.0)

    return corr


@njit
def _wls(ydata: np.ndarray, yfit: np.ndarray, bin_counts: np.ndarray) -> float:
    """Computes the weighted least squares cost specified by Cressie (1985)."""
    return np.sum(bin_counts * ((ydata - yfit) / yfit) ** 2)
