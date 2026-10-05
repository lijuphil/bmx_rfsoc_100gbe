#!/usr/bin/env python3
import datetime as dtm
import os
from datetime import timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CACHE = os.path.join(ROOT, "cygnus_a_calibration", "cygnus_cache.npz")
OUT = os.path.join(HERE, "..", "figures", "fig_cygnus_channel.png")

F_MHZ = 1135.6
HALF_WIN_MIN = 90.0
SMOOTH_SAMPLES = 5
TZ = timezone.utc
RE_C, IM_C = "#31688e", "#35b779"


def smooth(x, k=SMOOTH_SAMPLES):
    return np.convolve(x, np.ones(k) / k, mode="same")


def main():
    c = np.load(CACHE)
    ts, freq = c["ts"], c["freq"]
    t_pred = float(c["t_pred_unix"])
    sel = np.abs(ts - t_pred) < HALF_WIN_MIN * 60.0
    ch = int(np.argmin(np.abs(freq - F_MHZ)))
    re = smooth(c["cross_real"][sel, ch])
    im = smooth(c["cross_imag"][sel, ch])
    t = [dtm.datetime.fromtimestamp(float(x), tz=TZ) for x in ts[sel]]
    tp = dtm.datetime.fromtimestamp(t_pred, tz=TZ)
    print(f"channel {ch}: {freq[ch]:.2f} MHz, {sel.sum()} spectra")

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, re, lw=0.6, color=RE_C, label="real")
    ax.plot(t, im, lw=0.6, color=IM_C, label="imaginary")
    ax.axvline(tp, color="k", ls=":", lw=1)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylabel("cross-correlation (arb.)")
    ax.set_xlabel("time (UTC), 2026-07-26")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.text(0.01, 0.95, f"{freq[ch]:.1f} MHz", transform=ax.transAxes,
            ha="left", va="top")
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
