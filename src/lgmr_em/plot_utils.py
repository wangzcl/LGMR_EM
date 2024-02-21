"""
Utilities for plotting.
"""

import numpy as np
from numpy.typing import ArrayLike
from h5netcdf import File
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from eigen_microstate import EMG
from . import data_utils


def plot_evolution_series(
    x: ArrayLike,
    y: ArrayLike,
    ax=None,
    invert_x: bool = True,
    invert_y: bool = False,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    **kwargs,
):
    # TODO: edit docstring
    """
    Plot the time series. Add some features for convenience.

    Parameters:
        x (ArrayLike): The x-axis values.
        y (ArrayLike): The y-axis values.
        ax (Optional): The matplotlib axes object to plot on. If None, the current axes will be used.
        invert_x (bool): Whether to invert the x-axis. Default is True.
        xlabel (str): The label for the x-axis. Default is None.
        ylabel (str): The label for the y-axis. Default is None.
        title (str): The title of the plot. Default is None.
        **kwargs: Additional keyword arguments to be passed to the plot function.

    Returns:
        ax: The matplotlib axes object.
    """
    if ax is None:
        ax = plt.gca()
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.plot(x, y, **kwargs)
    return ax


def plot_2d_raster(
    U: ArrayLike,
    lon: ArrayLike,
    lat: ArrayLike,
    ax=None,
    title: str = None,
    set_global=True,
    coastlines=True,
    colorbar=True,
    **kwargs,
):
    """
    Plot 2d raster data, e.g. a single eigen microstate.
    """
    if ax is None:
        ax = plt.gca()

    if set_global:
        ax.set_global()
    if coastlines:
        ax.coastlines()
    if "vmin" not in kwargs or "vmax" not in kwargs:
        ulim = np.nanmax(np.abs(U))
        if "vmin" not in kwargs:
            kwargs.setdefault("vmin", -ulim)
        if "vmax" not in kwargs:
            kwargs.setdefault("vmax", ulim)

    if "cmap" not in kwargs:
        kwargs.setdefault("cmap", "RdBu_r")

    cf = ax.pcolormesh(
        lon,
        lat,
        U,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )
    if title is not None:
        ax.set_title(title, fontsize="large")
    if colorbar:
        plt.colorbar(cf, ax=ax)
    return ax


class SingleEMPlot:
    """
    Plot a single eigen microstate (V and U in left and right panel).
    """

    def __init__(self, fig=None, width_ratios=[0.9, 1], projection=None):
        if fig is None:
            fig = plt.gcf()

        gs = fig.add_gridspec(1, 2, width_ratios=width_ratios)
        ax_v = fig.add_subplot(gs[0])
        if projection is None:
            projection = ccrs.Robinson(central_longitude=180)
        ax_u = fig.add_subplot(gs[1], projection=projection)

        self.fig = fig
        self.ax_v = ax_v
        self.ax_u = ax_u
        self.projection = projection

    def plot(
        self,
        V: ArrayLike,
        U: ArrayLike,
        weight: float,
        lon: ArrayLike,
        lat: ArrayLike,
        time: ArrayLike,
        em_idx: int,
        ylabel_fmt=r"$V_{}$",
        vtitle_fmt=r"Evolution $V_{}$",
        utitle_fmt=r"Microstate $U_{}$",
    ):
        self.fig.suptitle(f"EM{em_idx} ({weight:.2%})")
        ax_v = self.ax_v
        ax_u = self.ax_u

        ax_v.plot(time, V)
        ax_v.axhline(0, color="black", linestyle="--")
        ax_v.invert_xaxis()
        ax_v.set_xlabel("Age (yr BP)")
        ax_v.set_ylabel(ylabel_fmt.format(em_idx))
        ax_v.set_title(vtitle_fmt.format(em_idx), fontsize="large")

        plot_2d_raster(U, lon, lat, ax=ax_u, title=utitle_fmt.format(em_idx))


class MultiEMPlot:
    """
    Plot multiple eigen microstates (top n EMs of a single ensemble).
    """

    def __init__(self, n_top: int, fig=None):
        if fig is None:
            fig = plt.gcf()
        subfigures = fig.subfigures(n_top, 1, hspace=0.005)

        self.fig = fig
        self.subfigures = subfigures
        self.n_top = n_top

    def plot(
        self,
        em: EMG,
        title: str = None,
        ylabel_fmt=r"$V_{}$",
        vtitle_fmt=r"Evolution $V_{}$",
        utitle_fmt=r"Microstate $U_{}$",
    ):
        """
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to ``SingleEMPlot.plot``,
            Including ``ylabel_fmt``, ``vtitle_fmt``, ``utitle_fmt``.
        """
        self.fig.suptitle(title, fontsize="x-large", fontweight="bold")
        for i in range(self.n_top):
            fig = self.subfigures[i]
            em_plot = SingleEMPlot(fig)
            em_plot.plot(
                em.evolution_[i],
                em.microstates_[i],
                em.weights_[i],
                em.lon,
                em.lat,
                em.time,
                i + 1,
                ylabel_fmt,
                vtitle_fmt,
                utitle_fmt,
            )


class MultiVarEMPlot:
    """
    Plot multiple ensembles (variables).
    """

    def __init__(self, n_var: int, fig=None):
        if fig is None:
            fig = plt.gcf()
        subfigures = fig.subfigures(1, n_var, wspace=0.04)
        self.fig = fig
        self.subfigures = subfigures
        self.n_var = n_var

    def plot(
        self,
        data_dict: dict[str, EMG],
        n_top: int,
        title_fmt="EM of {}",
        ylabel_fmt=r"$V_{}$",
        vtitle_fmt=r"Evolution $V_{}$",
        utitle_fmt=r"Microstate $U_{}$",
    ):
        """
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to ``MultiEMPlot.plot``,
            Including ``ylabel_fmt``, ``vtitle_fmt``, ``utitle_fmt``.
        """
        for i, (varname, em) in enumerate(data_dict.items()):
            fig = self.subfigures[i]
            em_plot = MultiEMPlot(n_top, fig)
            em_plot.plot(
                em, title_fmt.format(varname), ylabel_fmt, vtitle_fmt, utitle_fmt
            )


class ProxyTrendsMap:
    def __init__(self, fig=None, mesh_ax=None, ax_crs=None, data_crs=None):
        if fig is None:
            fig = plt.gcf()
        self.fig = fig
        if mesh_ax is None:
            mesh_ax = plt.gca()
        self.mesh_ax = mesh_ax

        if ax_crs is None:
            ax_crs = ccrs.Robinson(central_longitude=180)
        if data_crs is None:
            data_crs = ccrs.PlateCarree()

        position = mesh_ax.get_position()
        ax = fig.add_axes(
            [position.x0, position.y0, position.width, position.height],
            projection=ax_crs,
            frameon=False,
        )
        ax.patch.set_alpha(0)

        self.ax = ax
        self.ax_crs = ax_crs
        self.data_crs = data_crs

    def plot(
        self,
        proxy_trends: list[data_utils.ProxyTrends],
        trend_color={1: "violet", -1: "lightgreen", 0: "yellow"},
        explode=4000,
        radius=60000,
        **kwargs,
    ):
        ax = self.ax

        for item in proxy_trends:
            lat, lon = item.lat, item.lon
            trends = item.trends
            ax.pie(
                [1 for _ in trends],
                colors=[trend_color[t] for t in trends],
                startangle=90,
                explode=[explode for _ in trends],
                radius=radius,
                center=self.ax_crs.transform_point(lon, lat, self.data_crs),
                **kwargs,
            )

        ax.set_global()
        return


def plot_proxy_series_and_location(
    proxydb,
    site_name,
    proxy_names,
    upper_boundary=data_utils.DEGLACIAL_INTERGLACIAL_BOUNDARY,
    lower_boundary=0,
):

    n = len(proxy_names)
    site = proxydb[site_name]
    lat, lon = site.attrs["latitude"], site.attrs["longitude"]
    site_data = site["data"]
    age = site_data["age_median"][...]

    fig = plt.figure(figsize=(4 * 2, 4 * n), constrained_layout=True)
    gs = fig.add_gridspec(n, 2, width_ratios=[1, 1])

    geoaxe = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    plot_site_location(site_name, lon, lat, geoaxe)

    for i, proxy_name in enumerate(proxy_names):
        v = site_data[proxy_name][...]
        v = v[(age >= lower_boundary) & (age <= upper_boundary)]

        ax = fig.add_subplot(gs[i, 0])
        ax.set_xlim(lower_boundary, upper_boundary)
        scatter_proxy_series(
            age[(age >= lower_boundary) & (age <= upper_boundary)],
            v,
            proxy_name,
            ax,
        )

    return fig


def scatter_proxy_series(
    x: ArrayLike,
    y: ArrayLike,
    proxy_name: str,
    ax=None,
    invert_x: bool = True,
    xlabel: str = "Age (yr BP)",
    **kwargs,
):
    ylabel_dict = {
        "d18o": r"$\delta^{18}$O (‰)",
        "mgca": "Mg/Ca (mmol/mol)",
        "uk37": r"U$^k_{37}$",
        "tex86": r"TEX$_{86}$",
    }
    ylabel = ylabel_dict[proxy_name.split("_")[0]]
    return plot_evolution_series(
        x,
        y,
        ax,
        invert_x,
        data_utils.proxy_temp_correlation(proxy_name) == -1,
        xlabel,
        ylabel,
        proxy_name,
        "o",
        **kwargs,
    )


def plot_site_location(site_name, lon, lat, ax=None):
    a, b = 10, 3
    ax.set_extent(
        [
            (lon // a - b) * a,
            (lon // a + b) * a,
            (lat // a - b) * a,
            (lat // a + b) * a,
        ],
        ccrs.PlateCarree(),
    )
    ax.stock_img()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.plot(lon, lat, markersize=3, marker="o", color="r")
    ax.annotate(
        site_name,
        xy=(lon, lat),
        xytext=(0, 10),
        textcoords="offset points",
        horizontalalignment="center",
    )
    return ax
