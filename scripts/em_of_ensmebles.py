import os
import xarray as xr
import matplotlib.pyplot as plt
from eigen_microstate import *
from lgmr_em import plot_utils


da = xr.open_dataarray("data/LGMR/LGMR_SAT_ens.nc")
sat = GeospatialRaster(None, da.lat.values, da.lon.values, da.age.values)
fig = plt.figure(figsize=(12,8), constrained_layout=True)
image_dir = "images/ensembles/sat"
os.makedirs(image_dir, exist_ok=True)
plotter = plot_utils.MultiEMPlot(2, fig)

for i in range(500):
    sat.values = da[i].values
    em = eigen_microstate_geospatial(sat, rescale=True)
    plotter.plot(em,title=f"LGMR SST EM of ensemble member {i}")
    plt.savefig(f"{image_dir}/em_{i}.png")
    fig.clear()

del sat
image_dir = "images/ensembles/sst"
os.makedirs(image_dir, exist_ok=True)

for i in range(10):
    begin_idx = i*50+1
    end_idx = (i+1)*50
    da = xr.open_dataarray(f"data/LGMR/LGMR_SST_ens_{begin_idx}-{end_idx}.nc")
    sst = GeospatialRaster(None, da.lat.values, da.lon.values, da.age.values)
    for j in range(50):
        sst.values = da[j].values
        em = eigen_microstate_geospatial(sst, rescale=True)
        plotter.plot(em,title=f"LGMR SST EM of ensemble member {begin_idx+j}")
        plt.savefig(f"{image_dir}/em_{begin_idx+j}.png")
        fig.clear()