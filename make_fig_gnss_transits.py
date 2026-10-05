#!/usr/bin/env python3
import datetime as dtm
import glob
import os
from datetime import timezone

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import analyze_vis002 as A

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(HERE), "paper", "SPIE_mod", "figures")
CACHE = os.path.join(HERE, "gnss_fig_cache.npz")

ZOOM_EVENTS = (1, 83)
ZOOM_HALF_S = 900.0
FIT_HALF_S = 120.0
FRINGE_THRESH = 0.02
FRAC_THRESH = 0.5

INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
TRACE = "#31688e"
RE_C = "#31688e"
IM_C = "#35b779"
L2_C = "#35b779"
MISS_C = "#35b779"
SHORT = {
    "GPS L5 / Galileo E5a": "GPS L5 / Gal E5a",
    "GLONASS L3": "GLONASS L3",
    "Galileo E5b / BeiDou B2b": "Gal E5b / BDS B2b",
    "GPS L2": "GPS L2",
    "BeiDou B3": "BDS B3",
    "Galileo E6": "Gal E6",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "legend.fontsize": 7,
    "legend.frameon": False,
})


def utc(t):
    return np.array([dtm.datetime.fromtimestamp(float(x), tz=timezone.utc)
                     for x in np.atleast_1d(t)])


def load_zoom(t_center):
    files = sorted(glob.glob(os.path.join(A.DATA_DIR, "vis_002*.h5")))
    keep = {k: [] for k in ("timestamp", "auto0", "auto1",
                            "cross_real", "cross_imag")}
    for fn in files:
        with h5py.File(fn, "r") as f:
            ts = f["timestamp"][:]
            m = np.abs(ts - t_center) <= ZOOM_HALF_S
            if not m.any():
                continue
            i0, i1 = np.flatnonzero(m)[[0, -1]]
            for k in keep:
                keep[k].append(f[k][i0:i1 + 1])
    d = {k: np.concatenate(v) for k, v in keep.items()}
    o = np.argsort(d["timestamp"])
    return {k: v[o] for k, v in d.items()}


def main():
    c = np.load(CACHE)
    ts, freq = c["ts"], c["freq"]
    names = list(c["names"])
    ev_line, ev_idx = c["ev_line"], c["ev_idx"]
    ev_fringe = c["ev_fringe"]
    tdt = utc(ts)
    active = [li for li in range(len(names)) if np.any(ev_line == li)]
    nlines = len(active)

    fig = plt.figure(figsize=(7.0, 6.4))
    gs = GridSpec(2, 1, height_ratios=[1.55, 1.0], hspace=0.28, figure=fig)
    gtop = gs[0].subgridspec(nlines, 1, hspace=0.08)
    gbot = gs[1].subgridspec(1, 3, wspace=0.42)

    top_axes = []
    for row, li in enumerate(active):
        ax = fig.add_subplot(gtop[row], sharex=top_axes[0] if top_axes else None)
        top_axes.append(ax)
        ratio = c[f"bp_{li}"] / c[f"base_{li}"]
        ax.plot(tdt, ratio, lw=0.45, color=TRACE)
        ax.axhline(1 + FRAC_THRESH, color=MUTED, lw=0.5, ls=(0, (3, 2)))
        ax.set_yscale("log")
        ax.set_ylim(0.8, max(3.0, ratio.max() * 3.5))
        sel = np.flatnonzero(ev_line == li)
        ok = sel[ev_fringe[sel] > FRINGE_THRESH]
        bad = sel[ev_fringe[sel] <= FRINGE_THRESH]
        yk = ratio[ev_idx[ok]] * 1.7
        ax.plot(tdt[ev_idx[ok]], yk, "v", ms=3.2, mfc=INK, mec="none")
        if bad.size:
            ax.plot(tdt[ev_idx[bad]], ratio[ev_idx[bad]] * 1.7, "v", ms=3.6,
                    mfc="white", mec=MISS_C, mew=0.8)
        ax.text(0.004, 0.86, f"{SHORT[names[li]]}  {c['fc'][li]:.2f} MHz",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
                color=INK, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
        ax.text(0.996, 0.86, f"{sel.size} ev.", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.8, color=MUTED)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.tick_params(axis="x", labelbottom=(row == nlines - 1), length=2)
        ax.tick_params(axis="y", length=2, pad=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    tz = timezone.utc
    top_axes[-1].xaxis.set_major_locator(mdates.HourLocator(tz=tz))
    top_axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    top_axes[-1].set_xlim(tdt[0], tdt[-1])
    top_axes[-1].set_xlabel("UTC, 2026-07-25/26 (run#1)")
    fig.text(0.035, 0.705, r"band power / local floor", rotation=90,
             ha="center", va="center", fontsize=8, color=INK)
    t_z = ts[ev_idx[ZOOM_EVENTS[0]]]
    for ax in (top_axes[0], top_axes[3]):
        ax.axvspan(utc(t_z - ZOOM_HALF_S)[0], utc(t_z + ZOOM_HALF_S)[0],
                   color="#cfd8e3", alpha=0.55, lw=0, zorder=0)
    top_axes[0].set_title("(a)", loc="left", fontweight="bold", pad=2)
    h1, = top_axes[0].plot([], [], "v", ms=3.2, mfc=INK, mec="none")
    h2, = top_axes[0].plot([], [], "v", ms=3.6, mfc="white", mec=MISS_C, mew=0.8)
    h3, = top_axes[0].plot([], [], color=MUTED, lw=0.5, ls=(0, (3, 2)))
    fig.legend([h1, h2, h3],
               ["fringe-confirmed event", "not fringe-confirmed",
                "50% excess threshold"],
               loc="upper right", ncol=3, bbox_to_anchor=(0.985, 0.995),
               handlelength=1.6, columnspacing=1.2)

    z = load_zoom(t_z)
    zt = z["timestamp"] - t_z
    auto = 0.5 * (z["auto0"] + z["auto1"])

    axb = fig.add_subplot(gbot[0])
    styles = {ZOOM_EVENTS[0]: ("-", TRACE), ZOOM_EVENTS[1]: ("-", L2_C)}
    for k in ZOOM_EVENTS:
        li = ev_line[k]
        fc, hw = c["fc"][li], c["hw"][li]
        sel = (freq >= fc - hw) & (freq <= fc + hw)
        bp = auto[:, sel].mean(1)
        floor = c[f"base_{li}"][ev_idx[k]]
        ls, col = styles[k]
        ipk = int(np.argmin(np.abs(zt)))
        hwin = int(FIT_HALF_S / np.median(np.diff(zt)))
        fit = A.fit_local_gaussian(zt, bp, ipk, hwin)
        if fit:
            tt = np.linspace(-FIT_HALF_S, FIT_HALF_S, 200)
            yy = A.gaussian(tt, fit["A"], fit["t0"], fit["sigma_s"], fit["C"])
            axb.plot(tt / 60, yy / floor, color=INK, lw=0.8,
                     ls=(0, (2, 1.2)), zorder=4)
        lab = names[li].split(" /")[0].replace("GPS ", "")
        if fit:
            lab += f", FWHM {fit['fwhm_s']:.0f} s"
        axb.plot(zt / 60, bp / floor, ls, lw=0.9, color=col, label=lab)
    axb.plot([], [], color=INK, lw=0.8, ls=(0, (2, 1.2)),
             label=r"$\pm$2 min Gaussian fit")
    axb.axhline(1.0, color=MUTED, lw=0.5)
    axb.set_xlabel("time from 20:24:59 UTC (min)")
    axb.set_ylabel("band power / local floor")
    axb.set_xlim(-ZOOM_HALF_S / 60, ZOOM_HALF_S / 60)
    axb.set_ylim(0.5, 11.5)
    axb.legend(loc="upper left", handlelength=1.4, borderaxespad=0.1,
               fontsize=6.3)
    axb.set_title("(b) GPS BIIF-11 (PRN 10) transit", loc="left", pad=2)

    for panel, k, lab in zip(("(c)", "(d)"), ZOOM_EVENTS, ("L5", "L2")):
        ax = fig.add_subplot(gbot[1 if panel == "(c)" else 2], sharey=None)
        li = ev_line[k]
        fc, hw = c["fc"][li], c["hw"][li]
        sel = np.flatnonzero((freq >= fc - hw) & (freq <= fc + hw))
        V = z["cross_real"][:, sel] + 1j * z["cross_imag"][:, sel]
        a = auto[:, sel]
        ch = sel[np.argmax(np.abs(V).max(0) / a.mean(0))]
        g = z["auto0"][:, ch] * z["auto1"][:, ch]
        norm = np.sqrt(np.where(g > 0, g, np.nan))
        re = z["cross_real"][:, ch] / norm
        im = z["cross_imag"][:, ch] / norm
        env = auto[:, ch] / auto[:, ch].max()
        ax.fill_between(zt / 60, 0, env, color="#e3e7ec", lw=0, zorder=0,
                        label=r"auto (norm.)")
        ax.plot(zt / 60, re, color=RE_C, lw=0.9, label=r"Re $\rho$")
        ax.plot(zt / 60, im, color=IM_C, lw=0.9, label=r"Im $\rho$")
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.set_ylim(-1.32, 1.05)
        ax.set_xlim(-ZOOM_HALF_S / 60, ZOOM_HALF_S / 60)
        ax.set_xlabel("time from 20:24:59 UTC (min)")
        if panel == "(c)":
            ax.set_ylabel(r"$\rho = V_{EW}/\sqrt{A_E A_W}$")
            ax.legend(loc="lower left", handlelength=1.0, ncol=3,
                      borderaxespad=0.1, columnspacing=0.8, fontsize=6.3,
                      handletextpad=0.4)
        ax.set_title(f"{panel} GPS {lab} fringe, {freq[ch]:.2f} MHz",
                     loc="left", pad=2)

    for ax in fig.axes[nlines:]:
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.075)
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"fig2_gnss_transits.{ext}")
        fig.savefig(p, dpi=300)
        print("wrote", p)


if __name__ == "__main__":
    main()
