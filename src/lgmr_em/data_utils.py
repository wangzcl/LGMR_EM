"""
Ultilities for reading and processing LGMR data
"""
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr
import h5netcdf
from dataclasses import dataclass
from scipy.stats import kendalltau
from collections import namedtuple

LGMR_VAR_KEYS = {"SST": "sst", "SAT": "sat", "d18Op": "d18Op"}
LGMR_PHASES = ("glacial", "deglacial", "interglacial")

GLACIAL_DEGLACIAL_BOUNDARY = 16900
DEGLACIAL_INTERGLACIAL_BOUNDARY = 9500
HOLOCENE_BOUNDARY = 11700



@dataclass
class GeospatialRaster:
    """
    A class for storing geospatial raster data

    (When this project was started,
    xarray was not able to handle curvilinear grids well,
    so we used this class instead of xarray.DataArray.)

    Attributes
    ----------
    values : np.ndarray
        The data values.
    lat : np.ndarray
        Latitude values (1D for orthogonal grids, 2D for curvilinear grids).
    lon : np.ndarray
        Longitude values (1D for orthogonal grids, 2D for curvilinear grids).
    time : np.ndarray
        Time values.
    tarea : np.ndarray, optional, default: None
        Area of each grid cell.
        Propotional to `np.cos(np.deg2rad(lat))` if the grid is orthogonal.
    """

    values: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray
    tarea: np.ndarray = None

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index) -> "GeospatialRaster":
        return GeospatialRaster(
            self.values[index], self.lat, self.lon, self.time[index]
        )


def convert_longitude(x: ArrayLike) -> np.ndarray:
    """
    Convert longitude values to [-180, 180] range.
    """
    return ((x + 180) % 360) - 180


def load_lgmr_data(
    filename: str, varname: str, convert_lon: bool = True
) -> GeospatialRaster:
    """
    Load LGMR data from a NetCDF file, in chronological order.

    Note: In LGMR data, time index is yr BP, which means the original data are
    in reverse chronological order.

    Parameters
    ----------
    filename : str
        Path to the NetCDF file.
    varname : str
        Variable name to load. Choose from "sst", "sat" and "d18Op".
    convert_lon : bool, optional, default: True
        Whether to convert longitude values to [-180, 180] range.

    Returns
    -------
    data : GeospatialRaster
        A GeospatialRaster object containing the loaded data.
    """
    with h5netcdf.File(filename, "r", decode_vlen_strings=True) as f:
        values = f[varname][...][::-1]
        lat = f["lat"][...]
        lon = f["lon"][...]
        if convert_lon:
            lon = convert_longitude(lon)
        age = f["age"][...][::-1]
        tarea = f["tarea"][...] if "tarea" in f else None

    data = GeospatialRaster(values, lat, lon, age, tarea)
    return data


def lgmr_dataslice(data: GeospatialRaster, timeslice: str | slice) -> GeospatialRaster:
    """
    Get a slice of LGMR data. You can specify the slice as either "glacial",
    "deglacial", "interglacial", "holocene" or a slice object.

    Parameters
    ----------
    data : GeospatialRaster
        The data to slice.
    timeslice : slice or str
        The slice to get.

    Returns
    -------
    sliced_data : GeospatialRaster
        The sliced data.
    """
    if timeslice == "glacial":
        return data[data.time >= GLACIAL_DEGLACIAL_BOUNDARY]
    elif timeslice == "deglacial":
        return data[
            (data.time >= DEGLACIAL_INTERGLACIAL_BOUNDARY)
            & (data.time <= GLACIAL_DEGLACIAL_BOUNDARY)
        ]
    elif timeslice == "interglacial":
        return data[data.time <= DEGLACIAL_INTERGLACIAL_BOUNDARY]
    elif timeslice == "holocene":
        return data[data.time <= HOLOCENE_BOUNDARY]
    else:
        return data[timeslice]


def load_trace_data(
    filename: str, convert_lon=True, resample=True, 
) -> GeospatialRaster:
    data = xr.open_dataarray(filename).squeeze()
    # reindex
    data.coords["time"] = 1000 * data.time
    # ignore data after 1950 (negative year BP)
    data = data[data.time<=0]
    if resample:
        bins = np.arange(-22000, 100, 200)
        labels = bins[:-1] + 100.0
        data = (
            data.groupby_bins("time", bins, labels=labels).mean().rename({"time_bins": "time"})
        )
    
    time = -data.time.values

    if "TLAT" in data.coords:
        lat = data.TLAT.values
    elif "lat" in data.coords:
        lat = data.lat.values
    
    if "TLONG" in data.coords:
        lon = data.TLONG.values
    elif "lon" in data.coords:
        lon = data.lon.values
    
    if convert_lon:
        lon = convert_longitude(lon)
    
    values = data.values - 273.15

    return GeospatialRaster(values, lat, lon, time)


def ascend_or_descend(
    age, data, p_thres: float = None, age_bc=True
) -> Literal[-1, 0, 1]:
    """
    Ascending or descending trend (Kendall Tau) of data with age.

    Parameters:
    -----------
    age : array_like
        Age values.
    data : array_like
        Data series.
    p_thres : float, optional, default: None
        p-value threshold for significance test, a one-side hypothesis test is used.
        The function returns 0 if the test is not significant,
        regardless of the real trend. If None, no significance test is performed.
        We usually use a value of 0.1.
    age_bc : bool, optional, default: True
        Whether the ages are in BC. If True,
        the function returns -1 if the trend is ascending.

    Returns:
    --------
    sign : -1, 0 or 1
        1 if ascending, -1 if descending, 0 if unknown.
    """
    # p_thres usuall set to 0.1
    x = kendalltau(age, data).statistic
    if x > 0 and (
        (p_thres is None)
        or (kendalltau(age, data, alternative="greater").pvalue < p_thres)
    ):
        sign = 1
    elif x < 0 and (
        (p_thres is None)
        or (kendalltau(age, data, alternative="less").pvalue < p_thres)
    ):
        sign = -1
    else:
        sign = 0

    if age_bc:
        sign *= -1

    return sign


def proxy_temp_correlation(proxy_name: str) -> Literal[-1, 0, 1]:
    """
    Positive of negative correlation between proxy values and temperature.

    Parameters:
    -----------
    proxy_name : str
        Name of the proxy.

    Returns:
    --------
    sign : -1, 0, or 1
        1 if positive correlation, -1 if negative correlation, 0 if unknown.
    """
    d = {
        "d18o": -1,
        "mgca": 1,
        "uk37": 1,
        "tex86": 1,
    }
    for k, v in d.items():
        if k in proxy_name.lower():
            return v
    return 0


def clean_proxy_series(data, age, timeslice="interglacial"):
    """
    Select a slice of proxy series and remove NaN values.

    Parameters:
    -----------
    data : array_like
        Proxy values.
    age : array_like
        Age values.
    timeslice : str, optional, default: "interglacial"
        Slice to select. Choose from "interglacial", "glacial" and "deglacial".
        Only "interglacial" is supported now.

    Returns:
    --------
    cleaned_data : pd.Series
        Cleaned proxy values.
    """
    if timeslice == "interglacial":
        age_mask = age <= DEGLACIAL_INTERGLACIAL_BOUNDARY
    else:
        raise ValueError("Only interglacial is supported now")

    data_mask = ~np.isnan(data)
    mask = age_mask & data_mask

    return data[mask], age[mask]


def is_high_quality_proxy(
    data: ArrayLike,
    age: ArrayLike,
    n_values: int,
    boundary_distance: int,
    timeslice="interglaical",
) -> bool:
    """
    Whether a proxy series is good enough to represent time variability in a given period.

    Parameters:
    -----------
    data : array_like
        Proxy values.
    age : array_like
        Age of proxy values.
    n_values : int
        The required number of values for a high quaility series.
    boundary_distance : int
        Maximum distance to the boundary the period.
        If the distance between the earliest or latest age of the series and the
        boundary of the period is longer than this value, the series is not good
        enough.
    timeslice : str, optional, default: "interglacial"
        Time period of the series. Choose from "interglacial", "glacial" and "deglacial".
        Only "interglacial" is supported now.

    Returns:
    --------
    is_high_quality : bool
        Whether the proxy series is high quality.
    """
    if timeslice == "interglacial":
        upper_boundary = DEGLACIAL_INTERGLACIAL_BOUNDARY
        lower_boundary = 0
    else:
        raise ValueError("Only interglacial is supported now")

    if len(data) < n_values:
        return False
    if age.max() < upper_boundary - boundary_distance:
        return False
    if age.min() > lower_boundary + boundary_distance:
        return False
    return True


ProxyTrends = namedtuple("ProxyTrends", ["site", "lat", "lon", "trends"])


def proxy_trends(
    filepath: str, timeslice: str = "interglacial", p_thres: float = None
) -> list[ProxyTrends]:
    """
    Load proxy database from a NetCDF file, and calculate increasing or decreasing
    temperature trends of proxy series.

    Parameters:
    -----------
    filepath : str
        Path to the proxy database NetCDF file.
    timeslice : str, optional, default: "interglacial"
        Time period of the series. Choose from "interglacial", "glacial" and "deglacial".
        Only "interglacial" is supported now.

    Returns:
    --------
    all_trends : list[ProxyTrends]
        A list of ProxyTrends objects, each of which contains the site name,
        latitude, longitude and trends (sorted) of a proxy record.
        If a site "a" is locate at (30°N, 120°E), and has 2 series representing
        increasing temperature, 1 series representing decreasing temperature,
        the corresponding ProxyTrends object is:
        ProxyTrends(site="a", lat=30, lon=120, trends=(1, 1, -1))

    """
    if timeslice == "interglacial":
        proxy_length_min_required = 30
        proxy_boundary_distance_required = 500
    else:
        raise ValueError("Only interglacial is supported now")

    proxydb = h5netcdf.File(filepath, "r", decode_vlen_strings=True)
    all_trends = []
    for site_name, site in proxydb.items():
        lat, lon = site.attrs["latitude"], site.attrs["longitude"]
        site_data = site["data"]
        age_raw = site_data["age_median"][...]

        trends = []
        for var_name, value in site_data.items():
            if "age" in var_name or "depth" in var_name:
                continue
            data, age = clean_proxy_series(value[...], age_raw, timeslice)
            if not is_high_quality_proxy(
                data,
                age,
                proxy_length_min_required,
                proxy_boundary_distance_required,
                timeslice,
            ):
                continue
            trend = ascend_or_descend(age, data, p_thres)
            trend *= proxy_temp_correlation(var_name)
            trends.append(trend)

        if trends:
            trends.sort(reverse=True)
            all_trends.append(ProxyTrends(site_name, lat, lon, tuple(trends)))
    proxydb.close()

    return all_trends
