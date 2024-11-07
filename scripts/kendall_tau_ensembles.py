"""
Evaluate the Kendall tau statistic for the LGMR ensemble members.
"""

import xarray as xr
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from eigen_microstate import *
from lgmr_em import data_utils, plot_utils

plt.rcParams["font.family"] = "Arial"
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.bbox"] = "tight"


for i in range(10):
    begin_idx = i * 50 + 1
    end_idx = (i + 1) * 50
    da = xr.open_dataset(f"data/LGMR/LGMR_SST_ens_{begin_idx}-{end_idx}.nc")["sst"]
    for j in range(50):
        idx =i*50+j
        print(f"Processing ensemble member {idx}")
        if i == 0 and j == 0:
            trends = np.empty((500, *da[0,0].shape), dtype=float)
            lat = da.lat.values
            lon = data_utils.convert_longitude(da.lon.values)

        sst = GeospatialRaster(
            da[j].values,
            da.lat.values,
            data_utils.convert_longitude(da.lon.values),
            da.age.values,
        )
        sst = data_utils.lgmr_dataslice(sst, "interglacial")

        trends[idx] = np.apply_along_axis(lambda y: kendalltau(-sst.time, y).statistic, 0, sst.values)


projection = ccrs.Robinson(central_longitude=180)
fig, axes = plt.subplots(
    3,
    figsize=(9, 12),
    subplot_kw={"projection": projection},
    layout="compressed",
)

np.save("LGMR_SST_ens_kendall_tau.npy", trends)


quantiles = np.quantile(trends, [0.25, 0.5, 0.75], axis=0)
titles = ["lower quartile", "median", "upper quartile"]


for i in range(3):
    plot_utils.plot_2d_raster(
        quantiles[i],
        lon,
        lat,
        ax=axes[i],
    )
    axes[i].set_title(titles[i])

plt.savefig(f"images/sst_ensemble_kendall_stats.pdf")
