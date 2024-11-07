"""
Do eigen microstate analysis for 500 LGMR ensemble members.
"""

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from eigen_microstate import *
from lgmr_em import data_utils, plot_utils

image_dir = Path("images/ensembles")
image_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.bbox"] = "tight"

for i in range(10):
    begin_idx = i * 50 + 1
    end_idx = (i + 1) * 50
    da = xr.open_dataset(f"data/LGMR/LGMR_SST_ens_{begin_idx}-{end_idx}.nc")["sst"]
    for j in range(50):
        sst = GeospatialRaster(
            da[j].values, da.lat.values, data_utils.convert_longitude(da.lon.values), da.age.values
        )
        sst = data_utils.lgmr_dataslice(sst, "interglacial")

        em = eigen_microstate_geospatial(
            sst, rescale=True
        )
        if data_utils.ascend_or_descend(sst.time, em.evolution_[0]) == -1:
            em.sign_reverse((0,))

        fig = plt.figure(figsize=(6,4.5), layout="compressed")
        plotter = plot_utils.MultiEMPlot(2, fig)
        plotter.plot(em, title=f"LGMR Interglacial SST EM of ensemble member {begin_idx+j}", uvspacing=0.05)
        plt.savefig(image_dir/f"em_{begin_idx+j}.pdf")
        plt.close(fig)
