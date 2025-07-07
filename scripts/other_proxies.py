import math
import string
from pathlib import Path
from h5netcdf import File
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from lgmr_em import data_utils

plt.rcParams["font.family"] = "Arial"
plt.rcParams["savefig.format"] = "pdf"
# plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.bbox"] = "tight"

proxydb = File("data/LGMR/proxyDatabase.nc", "r", decode_vlen_strings=True)
other_proxies_dir = Path("data/Proxies")

Record = namedtuple("Record", ["ocean", "site", "lat", "lon", "proxy", "age", "value"])


def record_from_proxydb(proxydb, ocean, site, proxy):
    site = proxydb[site]
    site_data = site["data"]
    lat, lon = site.attrs["latitude"], site.attrs["longitude"]
    age_raw = site_data["age_median"][...]
    value, age = data_utils.clean_proxy_series(site_data[proxy][...], age_raw)
    return Record(
        ocean=ocean,
        site=site,
        lat=lat,
        lon=lon,
        proxy=proxy,
        age=age,
        value=value,
    )


records = []

# 10.1016/j.quascirev.2018.06.023 haddam2018
md07_3088 = np.genfromtxt(
    "data/Proxies/MD07-3088_SST.tab", dtype=float, skip_header=16, delimiter="\t"
)
age = md07_3088[:, 1] * 1000  # Convert to years
value = md07_3088[:, 3]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Southern Ocean",
        site="MD07-3088",
        lat=-46.071667,
        lon=-75.687167,
        proxy="uk37",
        age=age,
        value=value,
    )
)


# 10.1016/j.quascirev.2021.106821 anderson2021
# 10.1126/science.1084451 pahnke2003
md97_2120 = np.genfromtxt(
    "data/Proxies/MD97-2120_age_model_sst_alk.tab",
    dtype=float,
    skip_header=23,
    delimiter="\t",
)
age = md97_2120[:, -3]
age = age * 1000
value = md97_2120[:, -1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Southern Ocean",
        site="MD97-2120",
        lat=-45.534333,
        lon=174.930833,
        proxy="uk37",
        age=age,
        value=value,
    )
)

# 10.5194/cp-10-293-2014 romahn2014
geob12615_4 = np.genfromtxt(
    "data/Proxies/Romahn_2014a/datasets/GeoB12615-4_isotope.tab",
    dtype=float,
    skip_header=20,
    delimiter="\t",
    usecols=(1, 7),
)
age = geob12615_4[:, 0] * 1000
value = geob12615_4[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Western Indian Ocean",
        site="geob12615-4",
        lat=-7.138333,
        lon=39.840833,
        proxy="mgca_ruber",
        age=age,
        value=value,
    )
)


# 10.1016/j.epsl.2010.01.024 mohtadi2010
geob10038_4 = np.genfromtxt(
    "data/Proxies/GeoB10038-4_d18O_Mg-Ca_SST.tab",
    dtype=float,
    skip_header=19,
    delimiter="\t",
)
age = geob10038_4[:, 1] * 1000
value = geob10038_4[:, 5]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Eastern Indian Ocean",
        site="geob10038-4",
        lat=-5.937500,
        lon=103.246000,
        proxy="mgca_ruber",
        age=age,
        value=value,
    )
)

# 10.1038/nature02903 stott2004
wtp = pd.read_excel(
    "data/Proxies/41586_2004_BFnature02903_MOESM2_ESM.xls",
    sheet_name="MD81",
    header=0,
    skiprows=[
        1,
    ],
    usecols="B,D",
)
value, age = data_utils.clean_proxy_series(wtp.values[:, 1], wtp.values[:, 0])

records.append(
    Record(
        ocean="Western Tropical Pacific",
        site="MD81",
        lat=None,
        lon=None,
        proxy="mgca_ruber",
        age=age,
        value=value,
    )
)
# 10.1016/j.quascirev.2010.01.004 leduc2010
# 10.1126/science.1072376 koutavas2002
v21_30 = np.loadtxt(
    "data/Proxies/V21-30_SST-age_MgCa.tab", dtype=float, delimiter="\t", skiprows=13
)
age = v21_30[:, 0] * 1000
value = v21_30[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Eastern Tropical Pacific",
        site="V21-30",
        lat=-1.217000,
        lon=-89.680000,
        proxy="mgca_ruber_and_sacculifer",
        age=age,
        value=value,
    )
)

# 10.1029/2002PA000768 barron2003
odp167_1019c = np.genfromtxt(
    "data/Proxies/167-1019C_UK37_SST.tab",
    dtype=float,
    skip_header=20,
    delimiter="\t",
    usecols=(2, 3),
)

age = odp167_1019c[:, 0] * 1000
value = odp167_1019c[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Northeast Pacific",
        site="ODP167-1019C",
        lat=41.682900,
        lon=-124.932000,
        proxy="uk37",
        age=age,
        value=value,
    )
)

# 10.1130/G25667A.1 isono2009
md01_2421_composite = np.loadtxt(
    "data/Proxies/MD01-2421_composite_sst.tab", dtype=float, delimiter="\t", skiprows=12
)
age = md01_2421_composite[:, 0] * 1000
value = md01_2421_composite[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Northwest Pacific",
        site="MD01-2421",
        lat=36.033000,
        lon=141.783000,
        proxy="uk37",
        age=age,
        value=value,
    )
)

# 10.1016/j.quascirev.2010.01.004 leduc2010
# 10.1016/S0277-3791(01)00105-6 marchal2002
md95_2015 = np.loadtxt(
    "data/Proxies/MD95-2015_SST-age_alk.tab", dtype=float, delimiter="\t", skiprows=13
)
age = md95_2015[:, 0] * 1000
value = md95_2015[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="North Atlantic",
        site="MD95-2015",
        lat=58.762333,
        lon=-25.959000,
        proxy="uk37",
        age=age,
        value=value,
    )
)

# 10.1016/j.quascirev.2010.01.004 leduc2010
# 10.1016/j.quascirev.2010.04.004 rodrigues2010
d13882 = np.loadtxt(
    "data/Proxies/D13882_SST-age_alk.tab", dtype=float, delimiter="\t", skiprows=13
)
age = d13882[:, 0] * 1000
value = d13882[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Northeast Atlantic",
        site="D13882",
        lat=38.634500,
        lon=-9.454200,
        proxy="uk37",
        age=age,
        value=value,
    )
)

# 10.1016/j.quascirev.2010.01.004 leduc2010
# 10.1029/2006GL028495 sachs2007
oce326_ggc30 = np.loadtxt(
    "data/Proxies/OCE326-GGC30_SST-age_alk.tab",
    dtype=float,
    delimiter="\t",
    skiprows=12,
)
age = oce326_ggc30[:, 0] * 1000
value = oce326_ggc30[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Northwest Atlantic",
        site="OCE326-GGC30",
        lat=43.882000,
        lon=-62.800000,
        proxy="uk37",
        age=age,
        value=value,
    )
)

# 10.1016/j.quascirev.2010.01.004 leduc2010
# 10.1029/2000PA000502 cacho2001
bs79_38 = np.loadtxt(
    "data/Proxies/BS79-38_SST-age_alk.tab", dtype=float, delimiter="\t", skiprows=12
)
age = bs79_38[:, 0] * 1000
value = bs79_38[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Mediterranean",
        site="BS79-38",
        lat=38.412000,
        lon=13.577000,
        proxy="uk37",
        age=age,
        value=value,
    )
)

# 10.1038/990069 ruhlemann1999
m35003_4 = np.genfromtxt(
    "data/Proxies/Ruehlemann_1999/datasets/M35003-4_Alkenones_SST.tab",
    dtype=float,
    skip_header=20,
    delimiter="\t",
    usecols=(1, 7),
)
age = m35003_4[:, 0] * 1000
value = m35003_4[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Western Tropical Atlantic",
        site="M35003-4",
        lat=12.090000,
        lon=-61.243333,
        proxy="uk37",
        age=age,
        value=value,
    )
)

# 10.1016/j.quascirev.2010.01.004 leduc2010
# 10.1126/science.1140461 weldeab2007b

md03_2707 = np.loadtxt(
    "data/Proxies/MD03-2707_SST-age_MgCa.tab",
    dtype=float,
    delimiter="\t",
    skiprows=12,
)
age = md03_2707[:, 0] * 1000
value = md03_2707[:, 1]
value, age = data_utils.clean_proxy_series(value, age)

records.append(
    Record(
        ocean="Eastern Tropical Atlantic",
        site="MD03-2707",
        lat=2.502,
        lon=9.395,
        proxy="mgca_ruber",
        age=age,
        value=value,
    )
)
n_records = len(records)
print(n_records, "records loaded from other proxies")

fig = plt.figure(figsize=(3.5, 16))

# set grids

default_len = 7
len_offsets = [
    0,
    0,
    0,
    0,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]
pos_offsets = [
    0,
    -4,
    -1,
    -2,
    -1,
    -3,
    -1,
    -1,
    -3,
    -1,
    -3,
    -2,
    -2,
    -0,
]


slices = []
for i in range(n_records):
    if i == 0:
        upper = 0
        lower = 0
    else:
        upper = lower + pos_offsets[i]
    lower = upper + default_len + len_offsets[i]
    print((upper, lower))
    slices.append(slice(upper, lower))

gs = GridSpec(lower, 1, figure=fig)

# set grids finished

palette = sns.color_palette("colorblind", n_records)

labels = list(string.ascii_lowercase)

annotate_locs = [
    (0.7, 0.85),
    (0.4, 0.5),
    (0.35, 0.7),
    (0.4, 0.6),
    (0.28, 0.25),
    (0.28, 0.6),
    (0.7, 0.28),
    (0.7, 0.78),
    (0.45, 0.6),
    (0.4, 0.5),
    (0.7, 0.6),
    (0.6, 0.2),
    (0.7, 0.33),
    (0.7, 0.95),
]

axes = []
for i in range(n_records):
    isfirst = i == 0
    islast = i == n_records - 1
    isodd = i % 2 != 0

    record = records[i]
    ax = fig.add_subplot(gs[slices[i], 0], sharex=axes[0] if not isfirst else None)

    ax.patch.set_alpha(0.0)

    ax.plot(record.age, record.value, c=palette[i], lw=1.2)
    """
    data_range = record.value.max() - record.value.min()
    step = 0.5 * (10 ** math.floor(math.log10(data_range)))
    ymin = math.floor(record.value.min() / step) * step
    ymax = math.ceil(record.value.max() / step) * step
    """
    ymin = math.floor(record.value.min())
    ymax = math.ceil(record.value.max())
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + 1, 1))
    # ax.set_yticks([ymin, ymax])

    ax.tick_params(
        bottom=islast,
        top=False,
        left=isodd,
        right=not isodd,
        labelleft=isodd,
        labelright=not isodd,
        labelbottom=islast,
        labeltop=False,
    )

    if isodd:
        ax.yaxis.set_label_position("left")
    else:
        ax.yaxis.set_label_position("right")
    ax.set_ylabel("SST (°C)")

    ax.annotate(
        f"{labels[i]}",
        xy=(0.0, 1.0) if isodd else (1.0, 1.0),
        xycoords="axes fraction",
        xytext=(-0.8, 1.3) if isodd else (0.8, 1.3),
        textcoords="offset fontsize",
        fontsize="large",
        ha="left" if isodd else "right",
        va="top",
        fontweight="bold",
    )

    ax.annotate(
        record.ocean,
        xy=annotate_locs[i],
        xycoords="axes fraction",
        # xytext=(0, 0),
        # textcoords="offset fontsize",
        fontsize="medium",
        ha="center",
        va="center",
        color=palette[i],
    )

    ax.spines["left"].set_visible(isodd)
    ax.spines["right"].set_visible(not isodd)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(islast)

    axes.append(ax)

# fig.subplots_adjust(hspace=-0.1)

axes[0].set_xlim(9500, 0)
axes[-1].set_xlabel("Age (years BP)")

fig.savefig("images/other_proxies.pdf", dpi=300)
