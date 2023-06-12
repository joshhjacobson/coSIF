import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


def interval_score(df: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """Interval score according to Gneiting and Raftery (2007), JASA"""
    return (df["interval_upper"] - df["interval_lower"]) + (2 / alpha) * (
        (df["data"] < df["interval_lower"]).astype(int) * (df["interval_lower"] - df["data"])
        + (df["data"] > df["interval_upper"]).astype(int) * (df["data"] - df["interval_upper"])
    )


def dss(df: pd.DataFrame) -> pd.Series:
    """Dawid–Sebastiani score according to Gneiting and Katzfuss (2014)"""
    sigma2 = df["mspe"]
    sigma = np.sqrt(sigma2)
    return (df["data"] - df["predictions"]) ** 2 / sigma2 + 2 * np.log(sigma)


def multivariate_dss(data: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    """Multivariate Dawid–Sebastiani score according to Gneiting and Raftery (2007), JASA"""
    difference = data - mean
    sign, logdet = np.linalg.slogdet(covariance)
    return (
        np.matmul(difference.T, cho_solve(cho_factor(covariance), difference.copy()))[0, 0]
        + sign * logdet
    )


def prepare_validation_results(
    df_validation: pd.DataFrame,
    df_test: pd.DataFrame,
    method: str,
    month: str,
    region: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    # collect and organize data
    df = df_test.merge(df_validation, on=["lat", "lon"], how="outer")
    df["difference"] = df["predictions"] - df["data"]
    df["ratio"] = df["difference"].values / df["rmspe"].values
    df["Method"] = method.capitalize()
    df["Month"] = month
    df["Region"] = region

    # compute individual prediction metrics
    df["INT"] = interval_score(df, alpha=alpha)
    df["DSS"] = dss(df)

    return df.loc[
        :,
        [
            "Method",
            "Month",
            "Region",
            "lat",
            "lon",
            "data",
            "predictions",
            "rmspe",
            "difference",
            "ratio",
            "INT",
            "DSS",
        ],
    ]


def collect_metrics(df: pd.DataFrame, num_decimal: int = 2) -> pd.DataFrame:
    groups = ["Method", "Month", "Region"]
    df_metrics = df.groupby(groups).agg(
        N=("difference", "count"),
        BIAS=("difference", lambda x: np.round_(np.mean(x), num_decimal)),
        RASPE=("difference", lambda x: np.round_(np.sqrt(np.mean(x**2)), num_decimal)),
        DSS_MEAN=("DSS", lambda x: np.round_(np.mean(x), num_decimal)),
        INT_MEAN=("INT", lambda x: np.round_(np.mean(x), num_decimal)),
    )
    return df_metrics
