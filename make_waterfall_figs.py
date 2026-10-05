#!/usr/bin/env python3
import datetime as dtm
import glob
import os
import sys
from datetime import timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(HERE, "..", "figures")
sys.path.insert(0, os.path.join(ROOT, "run2_vis002_analysis"))
import analyze_vis002 as A

TZ = timezone.utc
CLOCKWISE = {(0, 0): "a", (0, 1): "b", (1, 1): "c", (1, 0): "d"}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def utc_num(t):
    return mdates.date2num(dtm.datetime.fromtimestamp(float(t), tz=timezone.utc))


def panel(ax, pos, text):
    ax.set_title(f"({CLOCKWISE[pos]}) {text}", loc="left")


def fig3_waterfalls():
    files = sorted(glob.glob(os.path.join(A.DATA_DIR, "vis_002*.h5")))
    d, nchan, _ = A.load(files)
    freq = A.rf_axis(nchan, A.FS_HZ, A.ZONE)
    ts = np.asarray(d["timestamp"], dtype=float)
    y0, y1 = utc_num(ts[0]), utc_num(ts[-1])
    extent = [float(freq[0]), float(freq[-1]), y0, y1]

    spec = {
        (0, 0): ("auto0", "East autocorrelation"),
        (0, 1): ("auto1", "West autocorrelation"),
        (1, 1): ("cross_imag", "Cross-correlation, imaginary (E$\\times$W)"),
        (1, 0): ("cross_real", "Cross-correlation, real (E$\\times$W)"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for pos, (key, title) in spec.items():
        ax = axes[pos]
        img = d[key]
        if key.startswith("cross"):
            v = float(np.nanpercentile(np.abs(img), 99.5))
            kw = dict(cmap="viridis", vmin=-v, vmax=v)
            cl = "amplitude (arb.)"
        else:
            img = A.db(img)
            fin = img[np.isfinite(img)]
            lo, hi = np.percentile(fin, [5, 99.5])
            kw = dict(cmap="viridis", vmin=lo, vmax=hi)
            cl = "power (dB, arb.)"
        im = ax.imshow(img, aspect="auto", origin="lower", extent=extent,
                       interpolation="nearest", **kw)
        fig.colorbar(im, ax=ax, label=cl, pad=0.015)
        panel(ax, pos, title)
    ax0 = axes[0, 0]
    ax0.yaxis_date(tz=TZ)
    ax0.yaxis.set_major_locator(mdates.HourLocator(tz=TZ))
    ax0.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    ax0.set_ylim(y1, y0)
    for ax in axes[1, :]:
        ax.set_xlabel("RF frequency (MHz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("time (UTC), 2026-07-25/26")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_waterfalls.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def fig5_cygnus(half_win_min=90.0):
    c = np.load(os.path.join(ROOT, "cygnus_a_calibration", "cygnus_cache.npz"))
    ts, freq, mask = c["ts"], c["freq"], c["mask"]
    t_pred = float(c["t_pred_unix"])
    sel = np.abs(ts - t_pred) < half_win_min * 60.0
    t = ts[sel]
    ic = np.flatnonzero(mask)
    span = slice(ic[0], ic[-1] + 1)
    f = freq[span]
    keep = mask[span]

    def smooth(a, k=5):
        kern = np.ones(k) / k
        return np.apply_along_axis(lambda x: np.convolve(x, kern, mode="same"), 0, a)

    cr = smooth(c["cross_real"][sel][:, span])
    ci = smooth(c["cross_imag"][sel][:, span])
    amp = np.hypot(cr, ci)
    phase = np.arctan2(ci, cr)

    y0, y1 = utc_num(t[0]), utc_num(t[-1])
    yp = utc_num(t_pred)
    extent = [f[0], f[-1], y0, y1]
    spec = {
        (0, 0): (cr, "Cross-correlation, real", "signed", "amplitude (arb.)"),
        (0, 1): (ci, "Cross-correlation, imaginary", "signed", "amplitude (arb.)"),
        (1, 1): (phase, "Cross-correlation phase", "phase", "phase (rad)"),
        (1, 0): (A.db(amp), "Cross-correlation amplitude", "db", "amplitude (dB, arb.)"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for pos, (img, title, cmap, cl) in spec.items():
        ax = axes[pos]
        if cmap == "signed":
            v = np.nanpercentile(np.abs(img[:, keep]), 99)
            kw = dict(vmin=-v, vmax=v)
        elif cmap == "db":
            lo, hi = np.nanpercentile(img[:, keep], [1, 99])
            kw = dict(vmin=lo, vmax=hi + 3.0)
        else:
            kw = dict(vmin=-np.pi, vmax=np.pi)
        cm = plt.get_cmap("viridis")
        im = ax.imshow(img, aspect="auto", origin="lower", extent=extent,
                       cmap=cm, interpolation="nearest", **kw)
        ax.axhline(yp, color="k", lw=0.8, ls=":", alpha=0.8)
        cb = fig.colorbar(im, ax=ax, pad=0.02, label=cl)
        if cmap == "phase":
            cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cb.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        panel(ax, pos, title)
    ax0 = axes[0, 0]
    ax0.yaxis_date(tz=TZ)
    ax0.yaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=TZ))
    ax0.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    ax0.set_ylim(y0, y1)
    for ax in axes[1, :]:
        ax.set_xlabel("RF frequency (MHz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("time (UTC), 2026-07-26")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_cygnus_fringes.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out, f"(predicted transit {dtm.datetime.fromtimestamp(t_pred, tz=TZ):%H:%M:%S} UTC)")


if __name__ == "__main__":
    fig5_cygnus()
    if "--cyg-only" not in sys.argv:
        fig3_waterfalls()
