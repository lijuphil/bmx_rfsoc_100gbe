#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timezone
import datetime as dtm

HI_REST_MHZ = 1420.405751786
C_KMS = 299792.458
TZ = timezone.utc

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "hi_cache.npz")
REPORT_PATH = os.path.join(HERE, "report_hi.txt")

V_LSR = 14.43

_report_lines = []


def rlog(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    _report_lines.append(s)


def continuum_subtract(auto, freq, line, base):
    fb = freq[base]
    n = auto.shape[0]
    resid = np.empty((n, line.sum()))
    li = np.empty(n)
    for i in range(n):
        spec = auto[i]
        cf = np.polyfit(fb, spec[base], 2)
        b_fit = np.polyval(cf, freq)
        r = (spec - b_fit)[line]
        resid[i] = r
        li[i] = r.sum()
    return resid, li


def main():
    c = np.load(CACHE)
    ts, freq, l_deg, b_deg = c["ts"], c["freq"], c["l_deg"], c["b_deg"]
    auto0, auto1 = c["auto0"], c["auto1"]
    auto = 0.5 * (auto0 + auto1)

    line = (freq > 1417.0) & (freq < 1424.5)
    base = ((freq > 1409) & (freq < 1417)) | ((freq > 1426) & (freq < 1436))
    ff = freq[line]
    v_topo = (HI_REST_MHZ - ff) / HI_REST_MHZ * C_KMS
    v_lsr_axis = v_topo + V_LSR

    t_local = np.array([dtm.datetime.fromtimestamp(tt, tz=timezone.utc)
                         for tt in ts])

    resid, li = continuum_subtract(auto, freq, line, base)
    resid0, li0 = continuum_subtract(auto0, freq, line, base)
    resid1, li1 = continuum_subtract(auto1, freq, line, base)

    win = 61
    sm = np.convolve(li, np.ones(win) / win, mode="same")

    corr = np.corrcoef(li, -np.abs(b_deg))[0, 1]
    rlog(f"Correlation of HI line integral with -|b| (plane-crossing proxy): "
         f"{corr:+.3f}  (positive = signal rises as beam nears the plane, as "
         f"expected for real Galactic HI)")
    corr0 = np.corrcoef(li0, -np.abs(b_deg))[0, 1]
    corr1 = np.corrcoef(li1, -np.abs(b_deg))[0, 1]
    rlog(f"  same correlation computed independently on each dish: "
         f"auto0={corr0:+.3f}  auto1={corr1:+.3f} "
         f"(both dishes agreeing is evidence against single-feed RFI)")

    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    ax1.plot(t_local, li, lw=0.4, color="0.7", label="raw line integral")
    ax1.plot(t_local, sm, lw=1.6, color="C3", label=f"smoothed ({win}-pt)")
    ax1.set_ylabel("HI line integral (arb, continuum-subtracted)")
    ax1.set_xlabel(f"time (UTC)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t_local, b_deg, color="0.2", lw=1.2, ls="--", label="Galactic b")
    ax2.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax2.set_ylabel("Galactic latitude b (deg)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.suptitle("BMX HI line integral vs time, with Galactic latitude of the "
                 "zenith beam (vis_0014-0018)")
    fig.tight_layout()
    out = os.path.join(HERE, "hi_lightcurve_vs_b.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    rlog(f"wrote {out}")

    order = np.argsort(b_deg)
    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = np.nanpercentile(np.abs(resid), 98)
    im = ax.imshow(resid[order], aspect="auto", origin="lower",
                    extent=[v_topo[0], v_topo[-1], b_deg[order][0], b_deg[order][-1]],
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("topocentric velocity (km/s)")
    ax.set_ylabel("Galactic latitude b (deg, spectra resorted by b)")
    ax.set_title("Continuum-subtracted residual vs Galactic latitude")
    fig.colorbar(im, ax=ax, label="residual power (arb)")
    fig.tight_layout()
    out = os.path.join(HERE, "hi_waterfall_vs_b.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    rlog(f"wrote {out}")

    for b_thresh in (3.0, 5.0, 8.0):
        sel = np.abs(b_deg) < b_thresh
        rlog(f"  |b|<{b_thresh:.0f} deg: {sel.sum()} spectra")

    b_thresh = 5.0
    sel_on = np.abs(b_deg) < b_thresh
    sel_off = np.abs(b_deg) > 15.0
    prof_on = np.median(resid[sel_on], axis=0)
    prof_off = np.median(resid[sel_off], axis=0)
    noise_off = resid[sel_off].std(0).mean()
    n_bad_on = int((np.abs(resid[sel_on]).max(axis=1) > 10 * noise_off).sum())
    rlog(f"  {n_bad_on}/{sel_on.sum()} on-plane spectra have a >10-sigma outlier "
         f"sample somewhere in the line window (likely RFI); median stacking "
         f"is robust to these, mean stacking would not be.")

    diff = prof_on - prof_off

    core = np.abs(v_topo) < 200.0
    w = np.clip(diff, 0, None) * core
    if w.sum() > 0:
        cen_topo = (v_topo * w).sum() / w.sum()
    else:
        cen_topo = np.nan
    cen_lsr = cen_topo + V_LSR
    peak_snr = diff.max() / max(noise_off, 1e-30)

    if w.sum() > 0:
        var = (w * (v_topo - cen_topo) ** 2).sum() / w.sum()
        width_kms = 2.3548 * np.sqrt(max(var, 0))
    else:
        width_kms = np.nan

    rlog(f"\n-- Stacked profile, |b| < {b_thresh:.0f} deg ({sel_on.sum()} spectra), "
         f"differenced against |b|>15 deg off-plane reference ({sel_off.sum()} spectra) --")
    rlog(f"  peak (on-off) excess = {diff.max():.4g}, off-plane rms noise = "
         f"{noise_off:.4g}, peak S/N ~ {peak_snr:.1f}")
    rlog(f"  intensity-weighted centroid (within +/-200 km/s of line center): "
         f"{cen_topo:+.1f} km/s topocentric -> {cen_lsr:+.1f} km/s LSR")
    rlog(f"  second-moment width: {width_kms:.0f} km/s (resolution-limited floor "
         f"is the {54.1:.0f} km/s channel width; treat this as an upper bound, "
         f"not a resolved physical line width)")

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax = axes[0]
    ax.step(v_lsr_axis, prof_on, where="mid", color="C0", lw=1.4,
            label=f"|b|<{b_thresh:.0f} deg (on-plane, N={sel_on.sum()})")
    ax.step(v_lsr_axis, prof_off, where="mid", color="0.5", lw=1.0,
            label=f"|b|>15 deg (off-plane, N={sel_off.sum()})")
    ax.axhspan(-noise_off, noise_off, color="0.85", alpha=0.6,
               label="off-plane rms noise")
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_ylabel("continuum-subtracted power (arb)")
    ax.set_title(f"BMX Galactic HI, toward l={l_deg[np.argmin(np.abs(b_deg))]:.0f} deg "
                 f"(zenith drift scan, vis_0014-0018)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.step(v_lsr_axis, diff, where="mid", color="C3", lw=1.4,
             label="on-plane minus off-plane (position-switched)")
    ax2.axhspan(-noise_off, noise_off, color="0.85", alpha=0.6,
                label="off-plane rms noise")
    ax2.axvline(0, color="k", ls=":", lw=0.8)
    ax2.axvline(cen_lsr, color="r", ls="--", lw=1.2,
               label=f"centroid {cen_lsr:+.0f} km/s LSR")
    ax2.set_xlabel("velocity, LSR (km/s)")
    ax2.set_ylabel("on-plane minus off-plane (arb)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "hi_profile.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    rlog(f"wrote {out}")

    prof_on0 = np.median(resid0[sel_on], axis=0)
    prof_on1 = np.median(resid1[sel_on], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(v_lsr_axis, prof_on0, where="mid", lw=1.2, label="auto0 (East-X)")
    ax.step(v_lsr_axis, prof_on1, where="mid", lw=1.2, label="auto1 (West-X)", alpha=0.8)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("velocity, LSR (km/s)")
    ax.set_ylabel("continuum-subtracted power (arb)")
    ax.set_title("Independent-dish consistency check (on-plane stack)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "hi_dish_consistency.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    rlog(f"wrote {out}")

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(_report_lines) + "\n")


if __name__ == "__main__":
    main()
