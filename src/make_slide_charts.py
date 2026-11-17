"""Slide-ready charts for the presentation.

Design: one dominant hue (deep purple) plus one accent (saffron gold), grey for context.
Each chart carries a short in-plot annotation so the key number can be read without the narration.
The slide headline carries the message, so there are no in-image titles.
Reads the processed tables written by notebooks/DS_analysis.ipynb and writes PNGs to figures/slides/.
Usage: python src/make_slide_charts.py   (set POPPINS_DIR to a folder with Poppins .ttf files)
"""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
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

DEEP, PURPLE, LILAC, LAV = "#2A1A5E", "#5B3A9E", "#B7A6DD", "#EEE9F7"
GOLD, GOLD_DK, INK, GREY, PALE, LIGHT = "#E8A000", "#9C6B00", "#1F2230", "#6B7080", "#D3D5DC", "#ECEDF1"
plt.rcParams.update({"font.size": 16, "axes.edgecolor": PALE, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 1,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.unicode_minus": False})
MON = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def note(ax, x, y, bold, rest=None, color=INK, coords="data", size=15):
    """Two-level annotation: a bold lead line and an optional regular grey second line."""
    tr = ax.transData if coords == "data" else ax.transAxes
    t = ax.text(x, y, bold, transform=tr, ha="left", va="top", fontsize=size, fontweight="bold", color=color)
    if rest:
        ax.annotate(rest, xycoords=t, xy=(0, 0), xytext=(0, -4), textcoords="offset points",
                    ha="left", va="top", fontsize=size - 1, color=GREY)
    return t


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


pm = pd.read_csv(os.path.join(PROC, "pm25_station_monthly_clean.csv"), parse_dates=["date"])
city = pd.read_csv(os.path.join(PROC, "analysis_city_month.csv"), parse_dates=["date"], index_col="date")
R = json.load(open(os.path.join(ROOT, "results.json")))
dry = [11, 12, 1, 2, 3, 4]

# 1 availability heat map: single-hue sequential, hatched = not measured --------------------
idx = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
order = sorted(pm.station_code.unique(), key=lambda s: (s.endswith("R"), s))
av = pm.pivot_table(index="station_code", columns="date", values="pm25_mean").reindex(index=order, columns=idx)
cmap = LinearSegmentedColormap.from_list("purple", ["#F6F2FC", LILAC, "#8C6CC8", PURPLE, DEEP])
n_st = pm.groupby("date").station_code.nunique()
full = n_st[n_st >= len(order)].index.min()
fig, ax = plt.subplots(figsize=(11.5, 5.0))
ax.add_patch(plt.Rectangle((-0.5, -0.5), len(idx), len(order), facecolor="white", edgecolor=PALE, hatch="////", lw=0, zorder=0))
im = ax.imshow(np.ma.masked_invalid(av.values), aspect="auto", cmap=cmap, vmin=5, vmax=55, interpolation="nearest", zorder=1)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=13)
yp = [i for i, d in enumerate(idx) if d.month == 1]; ax.set_xticks(yp); ax.set_xticklabels([idx[i].year for i in yp], fontsize=14)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
xf = list(idx).index(full)
ax.axvline(xf - 0.5, color=GOLD, lw=2.5, zorder=2)
ax.text(xf + 0.5, -0.9, f"All {len(order)} stations reporting from {full:%b %Y}", ha="left", va="bottom", fontsize=14, fontweight="bold", color=GOLD_DK)
cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03); cb.set_label("PM2.5 (µg/m³)", fontsize=13); cb.outline.set_visible(False)
ax.text(0, -0.13, "Hatched = PM2.5 not yet measured", transform=ax.transAxes, fontsize=13, color=GREY)
save(fig, "s01_availability.png")

# 2 seasonal cycle bars ----------------------------------------------------------------------
clim = pm.groupby("month").pm25_mean.mean()
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.bar(clim.index, clim.values, color=[PURPLE if m in dry else GOLD for m in clim.index], width=0.72)
for m in (1, 8):
    ax.text(m, clim[m] + 0.8, f"{clim[m]:.1f}", ha="center", fontsize=17, fontweight="bold", color=INK)
ratio = clim[1] / clim[8]
note(ax, 4.6, clim.max() * 0.98, f"January is {ratio:.1f}× August", "all dry months exceed all wet months", color=DEEP)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(MON); ax.set_ylabel("Mean PM2.5 (µg/m³)")
ax.grid(axis="x", visible=False); ax.set_ylim(0, clim.max() * 1.12)
ax.legend(handles=[Patch(color=PURPLE, label="Dry season (Nov to Apr)"), Patch(color=GOLD, label="Wet season (May to Oct)")],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=13)
save(fig, "s02_seasonal.png")

# 3 drivers: three aligned panels ------------------------------------------------------------
cm = city.groupby("month")[["precip", "vc", "fires_600km"]].mean()
fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.9), sharex=True)
specs = [("precip", PURPLE, "Rain (mm per month)"), ("vc", INK, "Ventilation (m²/s)"), ("fires_600km", GOLD, "Fires within 600 km")]
for ax, (k, c, lab) in zip(axes, specs):
    for m in (1, 10): ax.axvspan(m - 0.48, m + 0.48, color=LAV, lw=0, zorder=0)
    if k == "vc": ax.plot(cm.index, cm[k], color=c, lw=3, marker="o", ms=7, zorder=3)
    else: ax.bar(cm.index, cm[k], color=c, width=0.68, zorder=3)
    ax.text(0.01, 1.02, lab, transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK, va="bottom")
    ax.grid(axis="x", visible=False); ax.tick_params(axis="y", labelsize=13)
for m in (1, 10):
    axes[0].text(m, cm.precip[m] + 12, f"{cm.precip[m]:.0f} mm", ha="center", fontsize=13, fontweight="bold", color=PURPLE)
    axes[2].text(m, cm.fires_600km[m] + 500, f"{cm.fires_600km[m]:,.0f}", ha="center", fontsize=13, fontweight="bold", color=GOLD_DK)
axes[0].set_ylim(0, cm.precip.max() * 1.25); axes[2].set_ylim(0, cm.fires_600km.max() * 1.25)
axes[1].annotate("lowest ventilation", xy=(10, cm.vc[10]), xytext=(5.4, 980), fontsize=13, color=GREY,
                 arrowprops=dict(arrowstyle="-", color=GREY, lw=1))
axes[-1].set_xticks(range(1, 13)); axes[-1].set_xticklabels(MON)
fig.tight_layout(h_pad=1.6); save(fig, "s03_drivers.png")

# 4 roadside vs general ----------------------------------------------------------------------
pair = pm.pivot_table(index="date", columns="type", values="pm25_mean").dropna()
rg = R["road_minus_general"]
fig, ax = plt.subplots(figsize=(11.5, 5.0))
ax.fill_between(pair.index, pair.General, pair.Roadside, where=pair.Roadside >= pair.General, color=LILAC, alpha=0.55, lw=0)
ax.plot(pair.index, pair.General, color=GOLD, lw=2.6, label="General area (7 stations)")
ax.plot(pair.index, pair.Roadside, color=PURPLE, lw=2.6, label="Roadside (5 stations)")
ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_ylim(5, pair.values.max() * 1.32)
ax.legend(loc="upper right", fontsize=13, ncol=1)
n_hi = int(round(rg["share_months_road_higher"] * rg["n_months"]))
note(ax, 0.01, 0.98, f"Roadside higher in {n_hi} of {rg['n_months']} months", "shaded band = the traffic premium", color=DEEP, coords="axes")
save(fig, "s04_roadside.png")

# 5 trend: year effects ----------------------------------------------------------------------
pm["ym"] = pm.date.dt.strftime("%Y-%m")
mY = smf.ols("pm25_mean ~ C(station_code) + C(month) + C(year)", data=pm).fit(cov_type="cluster", cov_kwds={"groups": pm["ym"]})
yrs = list(range(2016, 2026)); ci = mY.conf_int()
eff = [0] + [mY.params[f"C(year)[T.{y}]"] for y in yrs[1:]]
lo = [0] + [ci.loc[f"C(year)[T.{y}]", 0] for y in yrs[1:]]; hi = [0] + [ci.loc[f"C(year)[T.{y}]", 1] for y in yrs[1:]]
fy = city.groupby(city.index.year).fires_600km.sum()
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.fill_between(yrs, lo, hi, color=LAV, lw=0, label="95% confidence interval")
ax.plot(yrs, eff, color=PURPLE, lw=3, marker="o", ms=9, markeredgecolor="white", markeredgewidth=2, label="Year effect vs 2016")
ax.axhline(0, color=GREY, lw=1.2, ls="--")
e23 = eff[yrs.index(2023)]
ax.scatter([2023], [e23], s=200, facecolor="none", edgecolor=GOLD, lw=2.5, zorder=4)
ax.annotate(f"2023: drier year,\nfires {fy[2023] / fy[2022]:.1f}× of 2022", xy=(2023, e23), xytext=(2020.4, 1.0), fontsize=14,
            color=GOLD_DK, fontweight="bold", arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.5))
ax.text(2025.18, eff[-1], f"{eff[-1]:.1f}", va="center", fontsize=15, fontweight="bold", color=PURPLE)
ax.set_xticks(yrs); ax.set_xticklabels([str(y) for y in yrs], fontsize=13); ax.set_ylabel("PM2.5 vs 2016 (µg/m³)")
ax.set_xlim(2015.6, 2025.8); ax.set_ylim(min(lo) - 1, 3.0); ax.grid(axis="x", visible=False)
ax.legend(loc="lower left", fontsize=13)
save(fig, "s05_trend.png")
print("2025 year effect", round(eff[-1], 2), "fires 2023/2022", round(fy[2023] / fy[2022], 2))

# 6 compliance -------------------------------------------------------------------------------
rec = pm[pm.year.isin([2024, 2025])].groupby("station_code").pm25_mean.mean().sort_values()
fig, ax = plt.subplots(figsize=(9.6, 6.4))
ax.barh(rec.index, rec.values, color=[PURPLE if s.endswith("R") else GOLD for s in rec.index], height=0.66)
ax.axvline(15, color=INK, lw=2.5, ls="--"); ax.axvline(5, color=GREY, lw=2, ls=":")
ax.text(15.4, len(rec) - 0.25, "Thai annual standard 15", color=INK, fontsize=13, fontweight="bold")
ax.text(5.4, len(rec) - 0.25, "WHO guideline 5", color=GREY, fontsize=13, fontweight="bold")
for i, v in enumerate(rec.values): ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=13, color=INK)
ax.set_xlabel("Mean PM2.5, 2024 to 2025 (µg/m³)"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, rec.max() * 1.18); ax.set_ylim(-0.6, len(rec) + 0.5)
ax.legend(handles=[Patch(color=PURPLE, label="Roadside"), Patch(color=GOLD, label="General area")], loc="lower right", fontsize=13)
save(fig, "s06_compliance.png")

# 7 fire anomaly scatter ---------------------------------------------------------------------
c = city.copy(); c["lf"] = np.log1p(c.fires_600km)
an = c[["pm25", "lf"]].groupby(c.month).transform(lambda s: s - s.mean())
an["dry"] = c.month.isin(dry)
ac = R["anomaly_corr"]
fig, ax = plt.subplots(figsize=(9.2, 6.2))
ax.scatter(an.lf[~an.dry], an.pm25[~an.dry], s=62, color=GOLD, edgecolor="white", linewidth=1.2, label="Wet season", zorder=3)
ax.scatter(an.lf[an.dry], an.pm25[an.dry], s=72, color=PURPLE, edgecolor="white", linewidth=1.2, label="Dry season", zorder=3)
sl, ic, r, p, _ = stats.linregress(an.lf, an.pm25); xx = np.linspace(an.lf.min(), an.lf.max(), 30)
ax.plot(xx, ic + sl * xx, color=INK, lw=2.5, zorder=4)
ax.axhline(0, color=PALE, lw=1.2); ax.axvline(0, color=PALE, lw=1.2)
ax.set_xlabel("Fire anomaly (more fires than usual to the right)"); ax.set_ylabel("PM2.5 anomaly (µg/m³)")
ax.legend(loc="upper left", fontsize=13)
ax.text(0.98, 0.04, f"r = {ac['Dry season (Nov to Apr)']['log_fires_600km']:.2f} dry season\nr = {ac['All months']['log_fires_600km']:.2f} all months",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=15, fontweight="bold", color=DEEP, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", fc=LAV, ec="none"))
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
fig, ax = plt.subplots(figsize=(9.6, 5.0))
y = np.arange(3)[::-1]
for yi, cf, s, p_, col in zip(y, coef, se, pv, [PURPLE, PURPLE, GREY]):
    ax.errorbar(cf, yi, xerr=1.96 * s, fmt="o", color=col, ms=15, elinewidth=4, capsize=0, markeredgecolor="white", markeredgewidth=2)
    ax.text(cf, yi + 0.3, f"{cf:.2f}  (p = {p_:.3f})", ha="center", fontsize=14, fontweight="bold", color=PURPLE if col == PURPLE else INK)
ax.axvline(0, color=INK, lw=1.2, ls="--")
ax.annotate("halves and is no\nlonger significant", xy=(coef[2] + 1.96 * se[2], 0), xytext=(3.5, 0.45), fontsize=14, color=GOLD_DK,
            fontweight="bold", arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.5))
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=15); ax.set_ylim(-0.6, 2.75); ax.set_xlim(-1.2, 5.8)
ax.set_xlabel("Fire effect on PM2.5 (µg/m³ per log-unit, 95% CI)"); ax.grid(axis="y", visible=False)
save(fig, "s08_confounding.png")
print("n months in confounding models", int(fits[0].nobs))

# 9 ML model comparison ----------------------------------------------------------------------
h = R["ml_holdout"]
names = ["Mean baseline (DummyRegressor)", "Gradient boosting", "Climatology baseline", "Ridge regression", "Random forest"]
short = ["Mean (Dummy)", "Gradient boosting", "Climatology", "Ridge", "Random forest"]
vals = [h[n]["RMSE"] for n in names]
fig, ax = plt.subplots(figsize=(9.2, 5.2))
ax.barh(short, vals, color=[PALE, PALE, GOLD, PALE, PURPLE], height=0.62)
for i, v in enumerate(vals): ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=14, color=INK, fontweight="bold" if i in (2, 4) else "normal")
gain = 1 - h["Random forest"]["RMSE"] / h["Climatology baseline"]["RMSE"]
ax.text(h["Random forest"]["RMSE"] + 1.0, 4, f"{gain:.0%} lower than climatology", va="center", fontsize=14, fontweight="bold", color=DEEP)
ax.set_xlabel("Test error, 2024 to 2025 (RMSE, µg/m³), lower is better"); ax.grid(axis="y", visible=False); ax.set_xlim(0, max(vals) * 1.18)
save(fig, "s09_ml_models.png")

# 10 permutation importance ------------------------------------------------------------------
imp = pd.Series(R["rf_permutation_importance_rmse_increase"]).sort_values(ascending=False).head(8)[::-1]
fig, ax = plt.subplots(figsize=(9.2, 5.6))
ax.barh(imp.index, imp.values, color=[GOLD if "Fire" in n else PURPLE for n in imp.index], height=0.62)
ax.set_xlabel("Importance (increase in test RMSE, µg/m³)"); ax.grid(axis="y", visible=False); ax.set_xlim(0, imp.max() * 1.3)
ax.legend(handles=[Patch(color=GOLD, label="Fire variables"), Patch(color=PURPLE, label="Other variables")], loc="lower right", fontsize=13)
top5 = sum("Fire" in n for n in imp[::-1].index[:5])
ax.text(0.98, 0.25, f"{top5} of the top 5 are fire variables", transform=ax.transAxes, ha="right", fontsize=14, fontweight="bold", color=GOLD_DK)
save(fig, "s10_ml_importance.png")
print("charts written to", OUT, sorted(os.listdir(OUT)))
