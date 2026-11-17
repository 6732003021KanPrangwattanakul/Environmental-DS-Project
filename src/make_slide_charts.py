"""Slide-ready charts for the presentation (no in-image titles; the slide headline carries the message).

Reads the processed tables written by notebooks/DS_analysis.ipynb and writes PNGs to figures/slides/.
Palette: a two-hue purple and gold pair chosen for maximum separation (validated for colour-vision deficiency,
worst-case CIEDE-like OKLab difference 40 under protanopia), slate ink for text, and grey for de-emphasis.
Usage: python src/make_slide_charts.py
"""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import statsmodels.formula.api as smf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC, OUT = os.path.join(ROOT, "data/processed"), os.path.join(ROOT, "figures/slides")
os.makedirs(OUT, exist_ok=True)

FONT_DIR = os.environ.get("POPPINS_DIR", "/tmp/fonts")
if os.path.isdir(FONT_DIR):
    for f in os.listdir(FONT_DIR):
        if f.endswith(".ttf"): fm.fontManager.addfont(os.path.join(FONT_DIR, f))
    plt.rcParams["font.family"] = "Poppins"

PURPLE, GOLD, GOLD_DK, INK, GREY, PALE, LIGHT, LAV = "#5B3A9E", "#E8A000", "#B87F00", "#2B2F3A", "#9AA0A6", "#D5D7DD", "#ECECEF", "#ECE6F6"
NAVY = INK
plt.rcParams.update({"font.size": 15, "axes.edgecolor": "#C9C9CF", "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.labelsize": 15, "axes.labelweight": "medium"})
MON = ["J","F","M","A","M","J","J","A","S","O","N","D"]

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

pm = pd.read_csv(os.path.join(PROC, "pm25_station_monthly_clean.csv"), parse_dates=["date"])
city = pd.read_csv(os.path.join(PROC, "analysis_city_month.csv"), parse_dates=["date"], index_col="date")
R = json.load(open(os.path.join(ROOT, "results.json")))

# 1 availability heat map -----------------------------------------------------------------
idx = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
order = sorted(pm.station_code.unique(), key=lambda s: (s.endswith("R"), s))
av = pm.pivot_table(index="station_code", columns="date", values="pm25_mean").reindex(index=order, columns=idx)
cmap = LinearSegmentedColormap.from_list("haze", ["#FFF6D5", "#F7CF57", GOLD, "#B0507A", PURPLE, "#24124D"])
cmap.set_bad("#E4E5EA")
fig, ax = plt.subplots(figsize=(12, 4.6))
im = ax.imshow(np.ma.masked_invalid(av.values), aspect="auto", cmap=cmap, vmin=5, vmax=60, interpolation="nearest")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=13)
yp = [i for i, d in enumerate(idx) if d.month == 1]; ax.set_xticks(yp); ax.set_xticklabels([idx[i].year for i in yp], fontsize=13)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03); cb.set_label("PM2.5 (µg/m³)", fontsize=13); cb.outline.set_visible(False)
ax.text(1.0, -0.13, "grey = no PM2.5 measurement", transform=ax.transAxes, ha="right", fontsize=12, color=GREY)
save(fig, "s01_availability.png")

# 2 seasonal cycle bars --------------------------------------------------------------------
clim = pm.groupby("month").pm25_mean.mean()
dry = [11, 12, 1, 2, 3, 4]
fig, ax = plt.subplots(figsize=(10, 5.2))
cols = [PURPLE if m in dry else GOLD for m in clim.index]
ax.bar(clim.index, clim.values, color=cols, width=0.72)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PURPLE, label="Dry season (Nov to Apr)"), Patch(color=GOLD, label="Wet season (May to Oct)")], loc="upper center", ncol=2, fontsize=13)
for m in (1, 8):
    ax.text(m, clim[m] + 0.8, f"{clim[m]:.1f}", ha="center", fontsize=17, fontweight="bold", color=NAVY)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(MON, fontsize=15); ax.set_ylabel("Monthly mean PM2.5 (µg/m³)")
ax.grid(axis="x", visible=False); ax.set_ylim(0, clim.max() * 1.25)
save(fig, "s02_seasonal.png")

# 3 drivers: three mini panels ---------------------------------------------------------------
cm = city.groupby("month")[["precip", "vc", "fires_600km"]].mean()
fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.6), sharex=True)
specs = [("precip", PURPLE, "Rain (mm)"), ("vc", INK, "Ventilation (m²/s)"), ("fires_600km", GOLD, "Fires ≤600 km")]
for ax, (k, c, lab) in zip(axes, specs):
    if k == "vc":
        ax.plot(cm.index, cm[k], color=c, lw=3, marker="o", ms=6)
    else:
        ax.bar(cm.index, cm[k], color=c, width=0.7)
    ax.set_ylabel(lab, fontsize=12.5); ax.grid(axis="x", visible=False)
    for m in (1, 10): ax.axvspan(m - 0.48, m + 0.48, color=LAV, alpha=1, lw=0, zorder=0)
axes[-1].set_xticks(range(1, 13)); axes[-1].set_xticklabels(MON)
fig.tight_layout(); save(fig, "s03_drivers.png")

# 4 roadside vs general ---------------------------------------------------------------------
pair = pm.pivot_table(index="date", columns="type", values="pm25_mean").dropna()
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.fill_between(pair.index, pair.General, pair.Roadside, where=pair.Roadside >= pair.General, color=PURPLE, alpha=0.14, lw=0)
ax.plot(pair.index, pair.General, color=GOLD, lw=2.6, label="General area")
ax.plot(pair.index, pair.Roadside, color=PURPLE, lw=2.6, label="Roadside")
ax.set_ylabel("PM2.5 (µg/m³)"); ax.legend(loc="upper right", fontsize=14, ncol=2); ax.set_ylim(top=pair.values.max() * 1.15)
save(fig, "s04_roadside.png")

# 5 trend: year effects ---------------------------------------------------------------------
pm["ym"] = pm.date.dt.strftime("%Y-%m")
mY = smf.ols("pm25_mean ~ C(station_code) + C(month) + C(year)", data=pm).fit(cov_type="cluster", cov_kwds={"groups": pm["ym"]})
yrs = list(range(2016, 2026)); ci = mY.conf_int()
eff = [0] + [mY.params[f"C(year)[T.{y}]"] for y in yrs[1:]]
lo = [0] + [ci.loc[f"C(year)[T.{y}]", 0] for y in yrs[1:]]; hi = [0] + [ci.loc[f"C(year)[T.{y}]", 1] for y in yrs[1:]]
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(yrs, lo, hi, color=GOLD, alpha=0.28, lw=0, label="95% confidence interval")
ax.plot(yrs, eff, color=PURPLE, lw=3, marker="o", ms=9, markeredgecolor="white", markeredgewidth=2, label="Year effect")
ax.legend(loc="lower left", fontsize=13)
ax.axhline(0, color=GREY, lw=1.2, ls="--")
ax.set_xticks(yrs); ax.set_xticklabels([str(y) for y in yrs], fontsize=13); ax.set_ylabel("PM2.5 vs 2016 (µg/m³)")
ax.grid(axis="x", visible=False)
save(fig, "s05_trend.png")

# 6 compliance ------------------------------------------------------------------------------
rec = pm[pm.year.isin([2024, 2025])].groupby("station_code").pm25_mean.mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(rec.index, rec.values, color=[PURPLE if s.endswith("R") else GOLD for s in rec.index], height=0.65)
ax.axvline(15, color=INK, lw=2.5, ls="--"); ax.axvline(5, color=GREY, lw=2.5, ls=":")
ax.text(15.4, len(rec) - 0.35, "Thai annual standard 15", color=INK, fontsize=13, fontweight="bold")
ax.text(5.4, len(rec) - 0.35, "WHO guideline 5", color=GREY, fontsize=13, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PURPLE, label="Roadside"), Patch(color=GOLD, label="General area")], loc="lower right", fontsize=13)
for i, v in enumerate(rec.values): ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=12.5, color=NAVY)
ax.set_xlabel("Mean PM2.5, 2024 to 2025 (µg/m³)"); ax.grid(axis="y", visible=False); ax.set_xlim(0, rec.max() * 1.15); ax.set_ylim(-0.6, len(rec) + 0.3)
save(fig, "s06_compliance.png")

# 7 fire anomaly scatter --------------------------------------------------------------------
c = city.copy(); c["lf"] = np.log1p(c.fires_600km)
an = c[["pm25", "lf"]].groupby(c.month).transform(lambda s: s - s.mean())
an["dry"] = c.month.isin(dry)
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.scatter(an.lf[~an.dry], an.pm25[~an.dry], s=60, color=GOLD, edgecolor="white", linewidth=1.2, label="Wet season")
ax.scatter(an.lf[an.dry], an.pm25[an.dry], s=70, color=PURPLE, edgecolor="white", linewidth=1.2, label="Dry season")
sl, ic, r, p, _ = stats.linregress(an.lf, an.pm25); xx = np.linspace(an.lf.min(), an.lf.max(), 30)
ax.plot(xx, ic + sl * xx, color=NAVY, lw=2.5)
ax.axhline(0, color=GREY, lw=1); ax.axvline(0, color=GREY, lw=1)
ax.set_xlabel("Fire anomaly (more fires than usual to the right)"); ax.set_ylabel("PM2.5 anomaly (µg/m³)"); ax.legend(loc="upper left", fontsize=13)
save(fig, "s07_fires.png")

# 8 confounding: fire coefficient across models ----------------------------------------------
cc = city.copy(); cc["month"] = cc.index.month; cc["year"] = cc.index.year
cc["log_fires_600km"] = np.log1p(cc.fires_600km); cc["t"] = (cc.year - 2016) + (cc.month - 1) / 12
cc["vc_k"] = cc.vc / 1000; cc["rain_100"] = cc.precip / 100; cc = cc.dropna(subset=["vc_k"])
forms = ["pm25 ~ C(month) + log_fires_600km + t", "pm25 ~ C(month) + log_fires_600km + vc_k + t",
         "pm25 ~ C(month) + log_fires_600km + vc_k + rain_100 + rh + t"]
fits = [smf.ols(f, data=cc).fit(cov_type="HC3") for f in forms]
labs = ["Fires only", "+ ventilation", "+ rain & humidity"]
coef = [m.params["log_fires_600km"] for m in fits]; se = [m.bse["log_fires_600km"] for m in fits]; pv = [m.pvalues["log_fires_600km"] for m in fits]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
y = np.arange(3)[::-1]
for yi, cf, s, p_, col in zip(y, coef, se, pv, [PURPLE, PURPLE, GREY]):
    ax.errorbar(cf, yi, xerr=1.96 * s, fmt="o", color=col, ms=14, elinewidth=4, capsize=0, markeredgecolor="white", markeredgewidth=2)
    ax.text(cf, yi + 0.28, f"+{cf:.2f}  (p = {p_:.3f})", ha="center", fontsize=13, color=NAVY)
ax.axvline(0, color=NAVY, lw=1.2, ls="--")
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=14); ax.set_ylim(-0.6, 2.7)
ax.set_xlabel("Fire effect on PM2.5 (µg/m³ per log-unit, 95% CI)"); ax.grid(axis="y", visible=False)
save(fig, "s08_confounding.png")

# 9 ML model comparison ---------------------------------------------------------------------
h = R["ml_holdout"]; ro = R["ml_rolling_mean"]
names = ["Mean baseline (DummyRegressor)", "Gradient boosting", "Climatology baseline", "Ridge regression", "Random forest"]
short = ["Mean (Dummy)", "Gradient boosting", "Climatology", "Ridge", "Random forest"]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
vals = [h[n]["RMSE"] for n in names]
ax.barh(short, vals, color=[PALE, PALE, GOLD, PALE, PURPLE], height=0.62)
for i, v in enumerate(vals): ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=14, color=NAVY, fontweight="bold" if i == 4 else "normal")
ax.set_xlabel("Error on 2024 to 2025 (RMSE, µg/m³), lower is better"); ax.grid(axis="y", visible=False); ax.set_xlim(0, max(vals) * 1.15)
save(fig, "s09_ml_models.png")

# 10 permutation importance ------------------------------------------------------------------
imp = pd.Series(R["rf_permutation_importance_rmse_increase"]).sort_values(ascending=False).head(8)[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp.index, imp.values, color=[GOLD if "Fire" in n else PURPLE for n in imp.index], height=0.62)
ax.set_xlabel("Importance (RMSE increase, µg/m³)"); ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GOLD, label="Fire variables"), Patch(color=PURPLE, label="Other variables")], loc="lower right", fontsize=13)
save(fig, "s10_ml_importance.png")
print("charts written to", OUT, sorted(os.listdir(OUT)))
