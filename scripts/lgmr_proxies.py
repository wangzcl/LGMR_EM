"""
Visualize the high-res proxy data in the LGMR database.
"""

from pathlib import Path
from h5netcdf import File
import matplotlib.pyplot as plt
from lgmr_em import data_utils, plot_utils

plt.rcParams["font.family"] = "Arial"
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.bbox"] = "tight"

save_dir = Path("images/proxies")
save_dir.mkdir(parents=True, exist_ok=True)

proxydb = File("data/LGMR/proxyDatabase.nc", "r", decode_vlen_strings=True)

proxy_length_min_required = 30
proxy_boundary_distance_required = 500

for site_name, site in proxydb.items():
    site_data = site["data"]
    age_raw = site_data["age_median"][...]

    var_names = []
    for var_name, value in site_data.items():
        if "age" in var_name or "depth" in var_name:
            continue
        data, age = data_utils.clean_proxy_series(value[...], age_raw)
        if data_utils.is_high_quality_proxy(
            data,
            age,
            proxy_length_min_required,
            proxy_boundary_distance_required,
        ):
            var_names.append(var_name)
    if len(var_names) > 0:
        plot_utils.plot_proxy_series_and_location(proxydb, site_name, var_names)
        plt.savefig(save_dir/f"{site_name}.pdf")
    plt.close()

# legacy code
sites_and_proxies = {
    "so90-63ka": ("d18o_ruber",),
    "niop-c2_905_pc": ("tex86", "uk37"),
    "geob12605-3": ("d18o_ruber", "mgca_ruber", "d18o_dutertrei"),
    "geob12610-2": ("d18o_ruber", "mgca_dutertrei", "d18o_dutertrei", "mgca_ruber"),
    "geob12615-4": ("mgca_ruber",),
    "gik16160-3": ("d18o_ruber", "mgca_ruber"),
    "geob9307-3": ("d18o_ruber", "mgca_ruber"),
    "geob9310-4": ("d18o_ruber", "mgca_ruber"),
    "sk237-gc04": ("d18o_ruber", "mgca_ruber"),
    "so189-039kl": ("d18o_ruber", "mgca_ruber"),
    "so189-144kl": ("d18o_ruber",),
    "bj8-03_70ggc": ("d18o_ruber", "mgca_ruber"),
    "fan17": ("d18o_ruber_pink", "mgca_ruber_pink"),
    "geob1023-5": ("d18o_inflata", "uk37"),
    "geob4905-4": ("d18o_ruber_pink", "mgca_ruber_pink"),
    "geob6518-1": ("d18o_ruber",),
    "md03-2707": ("d18o_ruber_pink", "mgca_ruber_pink"),
    "odp_175-1084b": ("mgca_bulloides",),
    "md02-2575": ("d18o_peregrina", "mgca_ruber"),
    "so164-03-4": ("d18o_ruber", "mgca_ruber"),
    "geob3129-1": ("d18o_ruber", "d18o_ruber_pink", "d18o_sacculifer", "mgca_ruber"),
    "geob3910-2": ("d18o_sacculifer", "d18o_tumida"),
    "m35003-4": ("d18o_ruber_pink", "uk37"),
    "pl07-39pc": (
        "d18o_ruber",
        "mgca_ruber",
        "d18o_ruber_pink",
        "d18o_bulloides",
        "d18o_dutertrei",
    ),
    "vm12-107": ("mgca_ruber",),
    "odp_167-1019c": ("uk37",),
    "mv99-pc14": ("mgca_bulloides",),
    "md02-2529": ("d18o_ruber", "uk37"),
    "me0005a-43jc": ("d18o_dutertrei", "d18o_ruber", "mgca_ruber"),
    "odp202-1240": ("d18o_ruber", "mgca_ruber"),
    "tr163-22": ("d18o_ruber", "mgca_ruber"),
    "vm21-30": ("d18o_ruber", "d18o_sacculifer", "mgca_sacculifer"),
    "m77-2-056-5": ("d18o_dutertrei", "mgca_dutertrei"),
    "m77-2-059-1": ("d18o_dutertrei", "mgca_dutertrei", "uk37"),
    "m77_2_003-2": ("uk37",),
    "m135-005-3": ("uk37",),
    "pc01": ("uk37",),
    "gik17051-3": ("d18o_bulloides", "d18o_pachyderma"),
    "knr166-14_11jpc": ("d18o_bulloides", "d18o_pachyderma"),
    "md95-2024": ("d18o_bulloides", "mgca_bulloides"),
    "md99-2251": ("mgca_bulloides",),
    "md03-2699": ("uk37",),
    "geob33131": ("uk37",),
}
