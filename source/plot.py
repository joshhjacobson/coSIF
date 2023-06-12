from copy import deepcopy
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from cmcrameri import cm

from data_utils import get_iterable
from fields import MultiField
from model import FittedVariogram

# global settings
plt.style.use("seaborn-deep")
XCO2_COLOR = "#4C72B0"
SIF_COLOR = "#55A868"
LINEWIDTH = 4
ALPHA = 0.6
PROJECTION = ccrs.PlateCarree()


def prep_axes(ax, extents=[-125, -65, 22, 58]):
    ax.set_extent(extents, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="gray", zorder=1)
    ax.spines[:].set_color("gray")
    gl = ax.gridlines(
        crs=PROJECTION,
        linewidth=0.8,
        color="gray",
        alpha=0.5,
        linestyle="--",
        draw_labels=True,
        zorder=0,
    )
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True
    gl.ylocator = mticker.FixedLocator([30, 40, 50])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax


def plot_da(
    da,
    vmin=None,
    vmax=None,
    robust=False,
    cmap=cm.bamako_r,
    title=None,
    label=None,
    fontsize=12,
):
    extents = [-130, -60, 18, 60]  # extended to verify spatial domain
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": PROJECTION})
    xr.plot.imshow(
        darray=da,
        transform=ccrs.PlateCarree(),
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        cbar_kwargs={"label": label},
    )
    prep_axes(ax, extents)
    ax.set_title(title, fontsize=fontsize)


def plot_df(
    df,
    data_name,
    vmin=None,
    vmax=None,
    cmap=cm.bamako_r,
    s=2,
    title=None,
    label=None,
    fontsize=12,
):
    extents = [-130, -60, 18, 60]  # extended to verify spatial domain
    cmap = deepcopy(cmap)
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": PROJECTION})
    prep_axes(ax, extents)
    plt.scatter(
        x=df.lon,
        y=df.lat,
        c=df[data_name],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        s=s,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
    )
    fig.colorbar(label=label)
    ax.set_title(title, fontsize=fontsize)


def get_data(field):
    da = field.ds[field.data_name].rename({"lon": "Longitude", "lat": "Latitude"})
    return da


def get_month_from_string(timestamp: str) -> str:
    m = np.datetime64(timestamp)
    return np.datetime_as_string(m, unit="M")


def plot_fields(
    mf: MultiField,
    data_names: list[str],
    title: str = None,
    fontsize: int = 12,
):
    CMAP = cm.roma_r
    if title is None:
        title = "0.05-degree monthly average residuals"

    fig, axes = plt.subplots(
        1, mf.n_procs, figsize=(8 * mf.n_procs, 4), subplot_kw={"projection": PROJECTION}
    )
    fig.suptitle(title, size=fontsize)

    for i, ax in enumerate(get_iterable(axes)):
        prep_axes(ax)
        xr.plot.imshow(
            darray=get_data(mf.fields[i]),
            transform=ccrs.PlateCarree(),
            ax=ax,
            cmap=CMAP,
            vmin=-2,
            center=0,
            cbar_kwargs={"label": "Standardized residuals"},
        )
        ax.set_title(
            f"{data_names[i]}: {get_month_from_string(mf.fields[i].timestamp)}",
            fontsize=fontsize,
        )


def plot_empirical_group(ids, group, fit_result, data_names, ax):
    idx = np.sum(ids)
    group.plot(
        x="bin_center",
        y="bin_mean",
        kind="scatter",
        color="black",
        alpha=0.8,
        ax=ax[idx],
        label=f"Empirical {fit_result.config.kind.lower()}",
    )

    if idx == 1:
        ax[idx].set_ylabel(
            f"Cross-{fit_result.scale_lab.lower()}", fontsize=fit_result.fontsize
        )
        ax[idx].set_title(
            f"Cross-{fit_result.config.kind.lower()}: {data_names[ids[0]]} vs"
            f" {data_names[ids[1]]} at"
            f" {np.abs(fit_result.timedeltas[idx])} month(s) lag",
            fontsize=fit_result.fontsize,
        )
    else:
        ax[idx].set_ylabel(fit_result.scale_lab, fontsize=fit_result.fontsize)
        ax[idx].set_title(
            f"{fit_result.config.kind}: {data_names[ids[1]]}",
            fontsize=fit_result.fontsize,
        )
        if fit_result.scale_lab.lower() != "covariance":
            ax[idx].set_ylim(bottom=0.0)
    ax[idx].set_xlabel(
        "Separation distance (km)",
        fontsize=fit_result.fontsize,
    )
    ax[idx].legend(loc="lower right")


def plot_model_group(ids, group, ax):
    idx = np.sum(ids)
    ax[idx].plot(
        group["distance"],
        group["variogram"],
        linestyle="--",
        color="black",
        label="Fitted model",
    )


def triangular_number(n):
    return n * (n + 1) // 2


def plot_variograms(
    fit_result: FittedVariogram, data_names: list[str], title: str = None, fontsize: int = 13
):
    n_procs = fit_result.config.n_procs
    n_plots = triangular_number(n_procs)
    fit_result.fontsize = fontsize
    if fit_result.config.kind == "Covariogram":
        fit_result.scale_lab = "Covariance"
    else:
        fit_result.scale_lab = "Semivariance"

    fig, ax = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), constrained_layout=True)

    groups1 = fit_result.df_empirical.groupby(level=[0, 1])
    for ids, df_group in groups1:
        plot_empirical_group(ids, df_group, fit_result, data_names, get_iterable(ax))

    groups2 = fit_result.df_theoretical.groupby(level=[0, 1])
    for ids, df_group in groups2:
        plot_model_group(ids, df_group, get_iterable(ax))

    if title is None:
        fig.suptitle(
            f"{fit_result.config.kind}s and cross-{fit_result.config.kind.lower()} for"
            f" {' and '.join(data_names)} residuals\n {fit_result.timestamp},"
            f" 0.05-degree North America, {fit_result.config.n_bins} bins, CompWLS:"
            f" {np.int(fit_result.cost)}",
            fontsize=fontsize,
            y=1.1,
        )
    else:
        fig.suptitle(
            title,
            fontsize=fontsize,
        )
