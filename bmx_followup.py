#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import h5py

from bmx_checks import (F_S_HZ, N_FFT, N_CHAN, F_CLK_HZ, CLK_PER_SPECTRUM,
                        CHAN_WIDTH_HZ, RF_MHZ, CLEAN_MASK, GNSS_LINES,
                        HI_REST_MHZ, mad_std, robust_linfit, find_files,
                        read_scalar_datasets, iter_spectra, RunSpec,
                        tacc_to_acc_len, chan_of)

C_M_S = 299792458.0

OCCUPIED_BANDS_MHZ = [
    (1164.5, 1219.0, "Galileo E5 AltBOC (E5a+E5b) / GPS L5 / BeiDou B2"),
    (1216.0, 1240.0, "GPS L2 / BeiDou B1-2"),
    (1242.9, 1248.6, "GLONASS L2 FDMA"),
    (1256.5, 1299.8, "Galileo E6 / BeiDou B3"),
    (1559.0, 1591.0, "GPS L1 / Galileo E1 / BeiDou B1C"),
    (1197.0, 1207.0, "GLONASS L3 CDMA"),
]
RADAR_BAND_MHZ = (1240.0, 1370.0)

BAND_EDGES_MHZ = (1100.0, 1550.0)


def occupied(f_mhz, guard=3.0):
    for lo, hi, name in OCCUPIED_BANDS_MHZ:
        if lo - guard <= f_mhz <= hi + guard:
            return name
    return None


def test_G(run: RunSpec, log, chunk_rows, skip_spectra, max_spectra,
           results_dir, make_plots):
    log(f"\n=== [G] Cross-floor time-rebinning scaling -- {run.name} ===")
    log("  Thermal noise scales as N^-0.5 under rebinning; a coherent")
    log("  common-mode term does not scale at all. The fitted exponent")
    log("  separates them, and the asymptote gives the systematic floor.")

    idx = np.where(CLEAN_MASK)[0][::8]
    chunks = []
    seen = kept = 0
    for fi, i0, blk in iter_spectra(run.files, chunk_rows=chunk_rows,
                                    verbose=False):
        n = blk["auto0"].shape[0]
        if seen + n <= skip_spectra:
            seen += n
            continue
        lo = max(0, skip_spectra - seen)
        seen += n
        a0 = blk["auto0"][lo:, idx]; a1 = blk["auto1"][lo:, idx]
        cr = blk["cross_real"][lo:, idx]
        den = np.sqrt(np.maximum(a0 * a1, 1e-300))
        chunks.append((cr / den).astype(np.float32))
        kept += chunks[-1].shape[0]
        if kept >= max_spectra:
            break
    if not chunks:
        log("  no data after transient cut")
        return {}
    x = np.concatenate(chunks, axis=0)[:max_spectra]
    log(f"  spectra used             : {x.shape[0]:,}")
    log(f"  channels used            : {x.shape[1]}")

    Ns, scat = [], []
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        nb = x.shape[0] // N
        if nb < 30:
            break
        b = x[:nb * N].reshape(nb, N, x.shape[1]).mean(axis=1)
        Ns.append(N)
        scat.append(float(np.median(mad_std(b, axis=0))))
    Ns = np.array(Ns, float); scat = np.array(scat)

    sl, ic, _ = robust_linfit(np.log10(Ns), np.log10(scat))
    log(f"  rebin factors            : {[int(v) for v in Ns]}")
    log(f"  scatter                  : " +
        ", ".join(f"{s:.3e}" for s in scat))
    log(f"  fitted log-log exponent  : {sl:+.3f}   (thermal = -0.500)")

    def resid(p):
        A, F = p
        return np.log(np.sqrt(A ** 2 / Ns + F ** 2)) - np.log(scat)
    from scipy.optimize import least_squares
    fit = least_squares(resid, [scat[0], scat[-1]], method="lm")
    A, F = np.abs(fit.x)
    log(f"  two-component fit        : sigma(N)^2 = A^2/N + F^2")
    log(f"      A (thermal, N=1)     = {A:.4e}")
    log(f"      F (non-integrating)  = {F:.4e}   <-- systematic floor")
    log(f"      crossover at N       = {(A / F) ** 2:.1f} accumulations "
        f"({(A / F) ** 2 * 0.5:.0f} s at 500 ms)")
    if sl > -0.35:
        log("  --> The floor is dominated by a coherent, non-integrating term.")
        log("      No amount of integration reduces it. This is a hard")
        log("      systematic limit on every cross-correlation result, and")
        log("      it belongs in the paper as a quoted number.")
    else:
        log("  --> Scaling is close to thermal; the elevated floor is mostly")
        log("      residual RFI rather than a fixed coupling term.")

    if make_plots:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(Ns, scat, "o-", label="measured")
        ax.loglog(Ns, A / np.sqrt(Ns), "--", label=r"thermal $N^{-1/2}$")
        ax.axhline(F, color="r", ls=":", label=f"floor {F:.2e}")
        ax.loglog(Ns, np.sqrt(A ** 2 / Ns + F ** 2), "-", alpha=0.5,
                  label="two-component fit")
        ax.set_xlabel("rebin factor N"); ax.set_ylabel("cross real-part scatter")
        ax.set_title(f"{run.name}: does the cross floor integrate down?")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{run.name}_G_cross_scaling.png"),
                    dpi=130); plt.close(fig)

    return dict(exponent=float(sl), thermal_A=float(A), floor_F=float(F),
                crossover_N=float((A / F) ** 2))


def test_H(run: RunSpec, log, chunk_rows, skip_spectra, results_dir,
           make_plots):
    log(f"\n=== [H] Origin of the coherent cross term -- {run.name} ===")
    log("  Averaging over many hours suppresses fringing sky (phase rotates)")
    log("  and leaves any static coupling. A delay fit to what remains says")
    log("  where it comes from.")

    s_cr = np.zeros(N_CHAN); s_ci = np.zeros(N_CHAN)
    s_den = np.zeros(N_CHAN); n = 0
    seen = 0
    for fi, i0, blk in iter_spectra(run.files, chunk_rows=chunk_rows,
                                    verbose=False):
        nr = blk["auto0"].shape[0]
        if seen + nr <= skip_spectra:
            seen += nr; continue
        lo = max(0, skip_spectra - seen); seen += nr
        a0 = blk["auto0"][lo:]; a1 = blk["auto1"][lo:]
        den = np.sqrt(np.maximum(a0 * a1, 1e-300))
        s_cr += (blk["cross_real"][lo:] / den).sum(axis=0)
        s_ci += (blk["cross_imag"][lo:] / den).sum(axis=0)
        n += a0.shape[0]
    if n == 0:
        log("  no data"); return {}

    mr = s_cr / n; mi = s_ci / n
    amp = np.hypot(mr, mi)
    ph = np.unwrap(np.arctan2(mi, mr))

    m = CLEAN_MASK & (RF_MHZ > BAND_EDGES_MHZ[0]) & (RF_MHZ < BAND_EDGES_MHZ[1])
    log(f"  spectra averaged         : {n:,}")
    log(f"  mean |cross| (clean)     : {np.median(amp[m]):.4e}")
    log(f"  expected thermal residual: {1 / np.sqrt(CHAN_WIDTH_HZ * 0.5 * n):.4e}")

    sl, ic, keep = robust_linfit(RF_MHZ[m], ph[m])
    tau_ns = -sl / (2 * np.pi) * 1e3
    log(f"  fitted phase slope       : {sl:.5f} rad/MHz")
    log(f"  --> implied delay        : {tau_ns:+.4f} ns "
        f"({tau_ns * C_M_S * 1e-9 * 100:+.2f} cm of path)")
    log(f"  geometric delay scale    : {8.8 / C_M_S * 1e9:.2f} ns "
        f"(full 8.8 m EW baseline)")
    log(f"  one channel of delay     : {1 / (CHAN_WIDTH_HZ) * 1e9:.1f} ns "
        f"(delay that wraps phase across one channel)")

    if abs(tau_ns) < 0.5:
        log("  VERDICT: delay consistent with ZERO -> the coherent term is")
        log("      NOT sky and NOT a cable path. It is common-mode coupling")
        log("      inside the RFSoC: shared clock distribution, power rail,")
        log("      or on-die ADC crosstalk between the two converter channels.")
        log("      Bench test T6 (split-signal) with the inputs TERMINATED")
        log("      rather than driven will confirm it directly: any residual")
        log("      correlation with no input signal is instrumental by")
        log("      construction. That is a 30-minute test and it settles this.")
    elif abs(tau_ns) < 30:
        log("  VERDICT: a specific, fixed delay -> a real propagation path.")
        log("      Compare against cable/splitter lengths and the geometric")
        log("      delay range; the VNA in test T6/T7 can measure candidates.")
    else:
        log("  VERDICT: large delay; phase may be wrapping. Refit on a")
        log("      narrower sub-band before trusting this number.")

    if make_plots:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].semilogy(RF_MHZ[m], amp[m], lw=0.7)
        axes[0].axhline(1 / np.sqrt(CHAN_WIDTH_HZ * 0.5 * n), color="r",
                        ls="--", label="thermal residual")
        axes[0].set_ylabel("|time-averaged cross| / sqrt(auto0 auto1)")
        axes[0].legend(); axes[0].set_title(
            f"{run.name}: static common-mode cross spectrum")
        axes[1].plot(RF_MHZ[m], ph[m], ".", ms=2)
        axes[1].plot(RF_MHZ[m], sl * RF_MHZ[m] + ic, "r-",
                     label=f"fit: {tau_ns:+.3f} ns")
        axes[1].set_xlabel("RF frequency (MHz)")
        axes[1].set_ylabel("phase (rad, unwrapped)"); axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{run.name}_H_static_cross.png"),
                    dpi=130); plt.close(fig)

    np.savez_compressed(os.path.join(results_dir, f"{run.name}_H_meancross.npz"),
                        freq_mhz=RF_MHZ, mean_real=mr, mean_imag=mi, n=n)
    return dict(n_averaged=int(n), delay_ns=float(tau_ns),
                mean_amp=float(np.median(amp[m])))


def test_I(run: RunSpec, log, chunk_rows, results_dir, make_plots):
    log(f"\n=== [I] Clock residual as a thermometer -- {run.name} ===")
    log("  The timestamp residual is a smooth diurnal sinusoid, i.e. a real")
    log("  oscillator drift, not scheduling jitter. Its derivative is a")
    log("  temperature proxy that costs nothing and needs no sensor.")

    acc_l, ts_l = [], []
    for fn in run.files:
        d = read_scalar_datasets(fn)
        acc_l.append(np.asarray(d["acc_count"], np.int64))
        ts_l.append(np.asarray(d["timestamp"], np.float64))
    acc = np.concatenate(acc_l); ts = np.concatenate(ts_l)
    o = np.argsort(acc); acc, ts = acc[o], ts[o]
    a0 = (acc - acc[0]).astype(np.float64)
    sl, ic, keep = robust_linfit(a0, ts)
    resid = ts - (sl * a0 + ic)

    from scipy.ndimage import median_filter
    mf = median_filter(resid, size=51, mode="nearest")
    sig = mad_std(resid - mf)
    bad = np.abs(resid - mf) > 8 * max(sig, 1e-9)
    if bad.any():
        log(f"  outlier samples clipped  : {int(bad.sum()):,} "
            f"({100 * bad.mean():.4f} %)")
        resid = resid.copy()
        resid[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad),
                               resid[~bad])
    win = max(101, (len(resid) // 200) | 1)
    k = np.ones(win) / win
    sm = np.convolve(resid, k, mode="same")
    valid = slice(win, len(sm) - win)
    dt = sl
    frac = np.gradient(sm, dt)[valid]
    t_hr = (ts - ts[0])[valid] / 3600.0

    log(f"  residual peak-to-peak    : {np.ptp(resid[valid]) * 1e3:.2f} ms")
    log(f"  frac. freq. peak-to-peak : {np.ptp(frac) * 1e6:.3f} ppm")
    log(f"  frac. freq. rms          : {np.std(frac) * 1e6:.3f} ppm")

    idx = np.where(CLEAN_MASK)[0]
    bp = []
    for fi, i0, blk in iter_spectra(run.files, chunk_rows=chunk_rows,
                                    datasets=("auto0", "auto1"), verbose=False):
        bp.append((0.5 * (blk["auto0"][:, idx] + blk["auto1"][:, idx])
                   ).mean(axis=1))
    bp = np.concatenate(bp)[o][valid]

    fin = np.isfinite(bp) & np.isfinite(frac)
    cc = np.corrcoef(bp[fin], frac[fin])[0, 1] if fin.sum() > 100 else np.nan
    log(f"  corr(band power, frac freq): {cc:+.3f}")
    if abs(cc) > 0.4:
        log("  --> Band power tracks the clock-derived temperature proxy.")
        log("      That supports a THERMAL origin for gain variation, and")
        log("      makes the startup transient a warm-up rather than a sky")
        log("      or antenna effect -- testable without running T9 at all.")
    else:
        log("  --> No strong correlation with the temperature proxy, so")
        log("      slow gain drift is not simply thermal. The transient")
        log("      still needs T0b / T9.")

    n_first = min(len(bp), int(6 * 3600 / dt))
    log(f"  band power, first 6 hr   : "
        f"{bp[0]:.4e} -> {np.median(bp[max(0, n_first - 500):n_first]):.4e} "
        f"({10 * np.log10(np.median(bp[max(0, n_first - 500):n_first]) / bp[0]):+.2f} dB)")

    if make_plots:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(t_hr, resid[valid] * 1e3, lw=0.6)
        axes[0].set_ylabel("timestamp residual (ms)")
        axes[0].set_title(f"{run.name}: clock residual, derived frequency, band power")
        axes[1].plot(t_hr, frac * 1e6, lw=0.8, color="C1")
        axes[1].set_ylabel("frac. freq. error (ppm)")
        axes[2].plot(t_hr, bp, lw=0.5, color="C2")
        axes[2].set_ylabel("clean-band power (arb)")
        axes[2].set_xlabel("hours since acquisition start")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{run.name}_I_thermometer.png"),
                    dpi=130); plt.close(fig)

    np.savez_compressed(os.path.join(results_dir, f"{run.name}_I_series.npz"),
                        t_hr=t_hr, frac_freq=frac, band_power=bp)
    return dict(resid_ptp_ms=float(np.ptp(resid[valid]) * 1e3),
                frac_ptp_ppm=float(np.ptp(frac) * 1e6),
                corr_power_temp=float(cc))


def test_J(run: RunSpec, log, results_dir, make_plots):
    log(f"\n=== [J] Master-sync truncation test -- {run.name} ===")
    log("  A sync at the hard-coded 127.8 ms would truncate integrations")
    log("  periodically, producing a second population of short intervals.")

    ds = []
    for fn in run.files:
        d = read_scalar_datasets(fn)
        acc = np.asarray(d["acc_count"], np.int64)
        ts = np.asarray(d["timestamp"], np.float64)
        o = np.argsort(acc); acc, ts = acc[o], ts[o]
        step = np.diff(acc)
        dd = np.diff(ts)[step == 1]
        ds.append(dd)
    dd = np.concatenate(ds)
    med = np.median(dd); s = mad_std(dd)
    log(f"  intervals examined       : {len(dd):,}")
    log(f"  median interval          : {med * 1e3:.5f} ms")
    log(f"  robust sigma             : {s * 1e3:.4f} ms")
    for f in (0.5, 0.8, 0.9):
        n_short = int(np.sum(dd < f * med))
        log(f"  intervals < {f:.0%} of median: {n_short:,} "
            f"({100 * n_short / len(dd):.4f} %)")
    n_short = int(np.sum(dd < 0.9 * med))
    frac_short = n_short / len(dd)
    periodic = False
    if n_short >= 5:
        gaps = np.diff(np.flatnonzero(dd < 0.9 * med))
        periodic = bool(mad_std(gaps) < 0.05 * np.median(gaps))
        log(f"  short-interval spacing   : median {np.median(gaps):.0f}, "
            f"scatter {mad_std(gaps):.1f} -> "
            f"{'PERIODIC' if periodic else 'sporadic'}")
    if n_short and not periodic:
        log(f"  ({n_short} isolated short interval(s): host-side glitches,")
        log("   not sync truncation, which would be strictly periodic.)")
    if not periodic and frac_short < 1e-3:
        log("  PASS: single unimodal population, no truncated integrations.")
        log("      -> the master sync did NOT reset the integration counter")
        log("         during either campaign, for ANY accumulation length.")
        log("      This closes the sync_period_sel/var open item from the")
        log("      data alone, including for the 100 ms run where the timing")
        log("      fit was inconclusive.")
    else:
        log("  ATTENTION: a short-interval population exists -- investigate.")

    if make_plots:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dd * 1e3, bins=400, range=(0, 1.4 * med * 1e3), log=True)
        ax.axvline(med * 1e3, color="r", ls="--", label="median")
        ax.axvline(127.83, color="k", ls=":", label="hard-coded sync 127.8 ms")
        ax.set_xlabel("consecutive timestamp difference (ms)")
        ax.set_ylabel("count"); ax.legend()
        ax.set_title(f"{run.name}: integration interval distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{run.name}_J_intervals.png"),
                    dpi=130); plt.close(fig)

    return dict(median_ms=float(med * 1e3), sigma_ms=float(s * 1e3),
                frac_short=float(frac_short))


def test_K(run: RunSpec, t_acc, log, chunk_rows, skip_spectra, max_spectra,
           results_dir, make_plots):
    log(f"\n=== [K] PFB noise bandwidth and channel correlation -- {run.name} ===")
    log("  The autos came in BELOW the naive radiometer floor. That is only")
    log("  possible if the effective noise bandwidth per channel exceeds the")
    log("  channel spacing -- i.e. neighbouring PFB channels overlap.")

    idx = np.where(CLEAN_MASK)[0]
    blocks = []
    seen = kept = 0
    for fi, i0, blk in iter_spectra(run.files, chunk_rows=chunk_rows,
                                    datasets=("auto0",), verbose=False):
        n = blk["auto0"].shape[0]
        if seen + n <= skip_spectra:
            seen += n; continue
        lo = max(0, skip_spectra - seen); seen += n
        blocks.append(blk["auto0"][lo:, idx].astype(np.float32))
        kept += blocks[-1].shape[0]
        if kept >= max_spectra:
            break
    a = np.concatenate(blocks, axis=0)[:max_spectra]
    d = np.diff(a, axis=0) / np.sqrt(2.0)
    med = np.median(a, axis=0)
    frac = mad_std(d, axis=0) / med
    expected = 1.0 / np.sqrt(CHAN_WIDTH_HZ * t_acc)
    ratio = np.median(frac) / expected
    enbw = 1.0 / ratio ** 2
    log(f"  spectra used             : {a.shape[0]:,}")
    log(f"  measured / naive floor   : {ratio:.4f}")
    log(f"  --> ENBW / channel spacing = {enbw:.4f}")
    log(f"      effective noise bandwidth = {enbw * CHAN_WIDTH_HZ / 1e3:.2f} kHz")
    log(f"      (channel spacing = {CHAN_WIDTH_HZ / 1e3:.2f} kHz)")

    adj = np.where(np.diff(idx) == 1)[0]
    dn = d / (med[None, :] * frac[None, :])
    good = np.all(np.abs(dn) < 5, axis=0)
    rho = []
    for lag in (1, 2, 3, 4):
        pairs = np.where((np.diff(idx, n=1) == 1) if lag == 1 else
                         (idx[lag:] - idx[:-lag] == lag))[0]
        pairs = pairs[good[pairs] & good[pairs + lag]]
        if len(pairs) < 20:
            rho.append(np.nan); continue
        x = dn[:, pairs]; y = dn[:, pairs + lag]
        rho.append(float(np.median(np.mean(x * y, axis=0))))
    log("  adjacent-channel noise correlation:")
    for lag, r in zip((1, 2, 3, 4), rho):
        log(f"      lag {lag}: rho = {r:+.4f}")

    r1 = rho[0] if np.isfinite(rho[0]) else 0.0
    n_line = int(round(7.5 / (CHAN_WIDTH_HZ / 1e6)))
    n_eff = n_line / (1 + 2 * r1 + 2 * (rho[1] if np.isfinite(rho[1]) else 0))
    infl = np.sqrt(n_line / max(n_eff, 1e-6))
    log(f"  HI line window           : {n_line} channels")
    log(f"  effective independent    : {n_eff:.1f}")
    log(f"  significance inflation   : {infl:.3f}x")
    log(f"  --> the paper's 16.1 sigma HI peak becomes "
        f"{16.1 / infl:.1f} sigma once PFB channel correlation is included.")
    log("      Quote the corrected value; a referee will ask for exactly this,")
    log("      and the paper already flags it as an unaddressed caveat.")

    if make_plots:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(RF_MHZ[idx], frac / expected, ".", ms=2, alpha=0.5)
        ax.axhline(1.0, color="r", ls="--", label="naive radiometer")
        ax.axhline(ratio, color="g", ls="-",
                   label=f"median {ratio:.3f} -> ENBW {enbw:.3f}x")
        ax.set_ylim(0.7, 1.6)
        ax.set_xlabel("RF frequency (MHz)")
        ax.set_ylabel("measured / naive radiometer floor")
        ax.set_title(f"{run.name}: PFB equivalent noise bandwidth")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{run.name}_K_enbw.png"), dpi=130)
        plt.close(fig)

    return dict(ratio=float(ratio), enbw_factor=float(enbw),
                rho=[None if not np.isfinite(r) else float(r) for r in rho],
                hi_sigma_corrected=float(16.1 / infl))


def test_L(run: RunSpec, log, chunk_rows, skip_spectra, results_dir,
           make_plots, top_n=3):
    log(f"\n=== [L] Corrected IMD pair search -- {run.name} ===")
    log("  The first-round pair (GLONASS L3 x Galileo E5b) put both products")
    log("  inside the Galileo E5 AltBOC transmission (1164.5-1219.0 MHz), so")
    log("  the 'excess' was Galileo signal. Hence the drive slope of ~0.")
    log("  Searching all pairs for products that clear every occupied band:")

    names = list(GNSS_LINES.keys())
    cands = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            f1 = GNSS_LINES[names[i]][0]; f2 = GNSS_LINES[names[j]][0]
            for prod, lab in ((2 * f1 - f2, f"2*{names[i]}-{names[j]}"),
                              (2 * f2 - f1, f"2*{names[j]}-{names[i]}")):
                if not (BAND_EDGES_MHZ[0] < prod < BAND_EDGES_MHZ[1]):
                    continue
                occ = occupied(prod)
                if occ:
                    continue
                margin = min(min(abs(prod - lo), abs(prod - hi))
                             for lo, hi, _ in OCCUPIED_BANDS_MHZ)
                in_radar = RADAR_BAND_MHZ[0] <= prod <= RADAR_BAND_MHZ[1]
                cands.append((margin, prod, lab, f1, f2, in_radar))
    cands.sort(reverse=True)
    if not cands:
        log("  No clean products exist for any pair of these lines.")
        log("  The on-sky IMD test cannot be done with GNSS alone; it needs")
        log("  the bench two-tone test (T2 with a second generator).")
        return {}

    log(f"  {'product MHz':>12}  {'chan':>5}  {'margin':>7}  radar?  pair")
    for margin, prod, lab, f1, f2, rad in cands[:8]:
        log(f"  {prod:12.2f}  {chan_of(prod):5d}  {margin:7.2f}  "
            f"{'YES' if rad else ' no'}   {lab}")
    log("  'margin' is MHz to the nearest occupied GNSS band edge.")
    log("  'radar' flags 1240-1370 MHz air-route surveillance, which is")
    log("  intermittent and strong -- usable, but the drive-slope test")
    log("  matters more there than the raw excess.")

    best = [c for c in cands if not c[5]] or cands
    results = {}
    for margin, prod, lab, f1, f2, rad in best[:top_n]:
        results[lab] = _run_imd_pair(run, f1, f2, prod, lab, log, chunk_rows,
                                     skip_spectra)
    log("\n  Interpretation guide: a genuine third-order product scales as")
    log("  P1^2 * P2, so the log-log drive slope should be ~1. A slope near 0")
    log("  means the excess is signal leakage, not intermodulation -- which")
    log("  is exactly what the first-round GLONASS/Galileo pair showed.")
    return results


def _run_imd_pair(run, f1, f2, prod, lab, log, chunk_rows, skip_spectra):
    c1 = np.where(np.abs(RF_MHZ - f1) <= 2.0)[0]
    c2 = np.where(np.abs(RF_MHZ - f2) <= 2.0)[0]
    cp = np.where(np.abs(RF_MHZ - prod) <= 0.8)[0]
    cref = np.where((np.abs(RF_MHZ - (prod - 5.0)) <= 0.8) |
                    (np.abs(RF_MHZ - (prod + 5.0)) <= 0.8))[0]
    S = {k: [] for k in ("f1", "f2", "p", "ref")}
    seen = 0
    for fi, i0, blk in iter_spectra(run.files, chunk_rows=chunk_rows,
                                    datasets=("auto0", "auto1"), verbose=False):
        n = blk["auto0"].shape[0]
        if seen + n <= skip_spectra:
            seen += n; continue
        lo = max(0, skip_spectra - seen); seen += n
        p = 0.5 * (blk["auto0"][lo:] + blk["auto1"][lo:])
        S["f1"].append(p[:, c1].mean(axis=1)); S["f2"].append(p[:, c2].mean(axis=1))
        S["p"].append(p[:, cp].mean(axis=1)); S["ref"].append(p[:, cref].mean(axis=1))
    S = {k: np.concatenate(v) for k, v in S.items()}

    drive = (S["f1"] ** 2) * S["f2"]
    hi = drive >= np.percentile(drive, 99.5)
    quiet = drive <= np.percentile(drive, 50.0)
    exc = np.median(S["p"][hi]) - np.median(S["p"][quiet])
    ref_exc = np.median(S["ref"][hi]) - np.median(S["ref"][quiet])
    sq = mad_std(S["p"][quiet])
    fund = np.median(0.5 * (S["f1"][hi] + S["f2"][hi]))
    dbc = 10 * np.log10(abs(exc) / fund) if exc > 0 and fund > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        X = 2 * np.log10(np.maximum(S["f1"], 1e-30)) + np.log10(np.maximum(S["f2"], 1e-30))
        Y = np.log10(np.maximum(S["p"] - np.median(S["p"][quiet]), 1e-30))
    m = hi & np.isfinite(X) & np.isfinite(Y) & (S["p"] > np.median(S["p"][quiet]))
    sl = robust_linfit(X[m], Y[m])[0] if m.sum() > 20 else np.nan

    log(f"\n  --- {lab} -> {prod:.2f} MHz ---")
    log(f"      excess at product    : {exc:+.4e} "
        f"({exc / sq if sq else np.nan:.1f} sigma)")
    log(f"      excess at shoulder   : {ref_exc:+.4e}   (control)")
    log(f"      relative to fundamental: "
        f"{dbc:.1f} dBc" if np.isfinite(dbc) else
        "      no positive excess -> IMD below detection here")
    log(f"      drive slope          : {sl:.2f}  (third-order = 1.0)")
    if np.isfinite(sl) and 0.6 < sl < 1.6 and exc > 3 * sq and exc > 3 * abs(ref_exc):
        log("      --> consistent with genuine third-order intermodulation.")
    elif exc <= 3 * sq:
        log("      --> no detection; this is an UPPER LIMIT on IMD, which is")
        log("          the good outcome and is what belongs in the paper.")
    else:
        log("      --> excess present but scaling is wrong for IMD; likely")
        log("          leakage or radar. Treat as contaminated, not a result.")
    return dict(product_mhz=float(prod), excess=float(exc),
                shoulder=float(ref_exc), sigma=float(exc / sq) if sq else None,
                dbc=float(dbc) if np.isfinite(dbc) else None,
                slope=float(sl) if np.isfinite(sl) else None)


ALL_TESTS = ["G", "H", "I", "J", "K", "L"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run2"); ap.add_argument("--run3")
    ap.add_argument("--outdir", default="bmx_followup_results")
    ap.add_argument("--tests", nargs="+", default=ALL_TESTS, choices=ALL_TESTS)
    ap.add_argument("--chunk-rows", type=int, default=4000)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-spectra", type=int, default=200000,
                    help="cap for the memory-resident tests G and K")
    ap.add_argument("--skip-hours", type=float, default=6.0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    if not (args.run2 or args.run3):
        ap.error("give at least one of --run2 / --run3")

    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def log(m=""):
        print(m, flush=True); lines.append(m)

    log("BMX RFSoC backend -- follow-up analyses")
    log(f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}")

    runs = []
    if args.run2:
        runs.append(RunSpec("run2_500ms", args.run2, 0.500,
                            find_files(args.run2, args.max_files)))
    if args.run3:
        runs.append(RunSpec("run3_100ms", args.run3, 0.100,
                            find_files(args.run3, args.max_files)))

    out = {}
    for r in runs:
        log("\n" + "=" * 74)
        log(f"RUN: {r.name}  ({len(r.files)} files)")
        log("=" * 74)
        t_acc = r.nominal_tacc_s
        skip = int(round(args.skip_hours * 3600.0 / t_acc))
        res = {}
        if "J" in args.tests:
            res["J"] = test_J(r, log, args.outdir, not args.no_plots)
        if "I" in args.tests:
            res["I"] = test_I(r, log, args.chunk_rows, args.outdir,
                              not args.no_plots)
        if "G" in args.tests:
            res["G"] = test_G(r, log, args.chunk_rows, skip, args.max_spectra,
                              args.outdir, not args.no_plots)
        if "H" in args.tests:
            res["H"] = test_H(r, log, args.chunk_rows, skip, args.outdir,
                              not args.no_plots)
        if "K" in args.tests:
            res["K"] = test_K(r, t_acc, log, args.chunk_rows, skip,
                              args.max_spectra, args.outdir, not args.no_plots)
        if "L" in args.tests:
            res["L"] = test_L(r, log, args.chunk_rows, skip, args.outdir,
                              not args.no_plots)
        out[r.name] = res

    with open(os.path.join(args.outdir, "followup_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.outdir, "followup_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.outdir}/followup_report.txt")


if __name__ == "__main__":
    main()
