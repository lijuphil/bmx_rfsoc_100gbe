#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import h5py


F_S_HZ = 1050e6
N_FFT = 4096
N_CHAN = 2048
SAMPLES_PER_CLK = 4
F_CLK_HZ = F_S_HZ / SAMPLES_PER_CLK
CLK_PER_SPECTRUM = N_FFT // SAMPLES_PER_CLK
CHAN_WIDTH_HZ = F_S_HZ / N_FFT
NYQUIST_ZONE = 3
BYTES_PER_PACKET_WIRE = 8322
PACKETS_PER_SPECTRUM = 8
SYNC_PERIOD_DEFAULT_CLK = 2**25 - 2


def acc_len_to_tacc(acc_len: float) -> float:
    return acc_len * CLK_PER_SPECTRUM / F_CLK_HZ


def tacc_to_acc_len(t_acc: float) -> float:
    return t_acc * F_CLK_HZ / CLK_PER_SPECTRUM


def rf_axis_mhz() -> np.ndarray:
    k = np.arange(N_CHAN)
    return (F_S_HZ + k * CHAN_WIDTH_HZ) / 1e6


RF_MHZ = rf_axis_mhz()


def chan_of(freq_mhz: float) -> int:
    return int(np.argmin(np.abs(RF_MHZ - freq_mhz)))


GNSS_LINES = {
    "GPS L5 / Gal E5a":   (1176.45, 12.0),
    "GLONASS L3":         (1202.03, 5.0),
    "Gal E5b / BDS B2b":  (1207.14, 6.0),
    "GPS L2":             (1227.60, 12.0),
    "BeiDou B3":          (1268.52, 10.0),
    "Galileo E6":         (1278.75, 10.0),
}
HI_REST_MHZ = 1420.405752
HI_MASK_HALFWIDTH_MHZ = 12.0
CLEAN_BAND_MHZ = (1120.0, 1520.0)

IMD_F1 = 1202.03
IMD_F2 = 1207.14
IMD_LO = 2 * IMD_F1 - IMD_F2
IMD_HI = 2 * IMD_F2 - IMD_F1


def clean_channel_mask() -> np.ndarray:
    m = (RF_MHZ >= CLEAN_BAND_MHZ[0]) & (RF_MHZ <= CLEAN_BAND_MHZ[1])
    for _, (f0, hw) in GNSS_LINES.items():
        m &= np.abs(RF_MHZ - f0) > (hw + 2.0)
    m &= np.abs(RF_MHZ - HI_REST_MHZ) > HI_MASK_HALFWIDTH_MHZ
    for f in (IMD_LO, IMD_HI):
        m &= np.abs(RF_MHZ - f) > 2.0
    return m


CLEAN_MASK = clean_channel_mask()


def mad_std(x, axis=None):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def robust_linfit(x, y, n_iter=5, n_sigma=4.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    slope = intercept = np.nan
    for _ in range(n_iter):
        if keep.sum() < 3:
            break
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        resid = y - (slope * x + intercept)
        s = mad_std(resid[keep])
        if not np.isfinite(s) or s == 0:
            break
        new_keep = keep & (np.abs(resid) < n_sigma * s)
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    return slope, intercept, keep


@dataclass
class RunSpec:
    name: str
    path: str
    nominal_tacc_s: float
    files: list = field(default_factory=list)


def find_files(path, max_files=None):
    files = sorted(glob.glob(os.path.join(path, "vis_*.h5")))
    if not files:
        raise SystemExit(f"No vis_*.h5 files found in {path}")
    if max_files:
        files = files[:max_files]
    return files


def read_scalar_datasets(fname):
    out = {}
    with h5py.File(fname, "r") as f:
        for key in ("acc_count", "timestamp", "timestamp_utc", "freq_mhz"):
            if key not in f:
                continue
            d = f[key]
            if d.dtype.kind in ("S", "O", "U"):
                out[key] = d[()]
            else:
                out[key] = np.asarray(d[()])
        out["_nspec"] = int(f["auto0"].shape[0])
        out["_nchan"] = int(f["auto0"].shape[1])
        out["_dtype"] = str(f["auto0"].dtype)
    return out


def iter_spectra(files, chunk_rows=4000, datasets=("auto0", "auto1",
                                                   "cross_real", "cross_imag"),
                 verbose=True):
    for fi, fname in enumerate(files):
        with h5py.File(fname, "r") as f:
            n = f[datasets[0]].shape[0]
            for i0 in range(0, n, chunk_rows):
                i1 = min(i0 + chunk_rows, n)
                block = {k: np.asarray(f[k][i0:i1], dtype=np.float64)
                         for k in datasets}
                yield fi, i0, block
        if verbose:
            print(f"      ... {os.path.basename(fname)} done "
                  f"({fi + 1}/{len(files)})", flush=True)


def check_continuity(run: RunSpec, log):
    log(f"\n=== [A] Accumulation continuity -- {run.name} ===")
    all_acc, all_ts = [], []
    per_file = []
    for fname in run.files:
        d = read_scalar_datasets(fname)
        acc = np.asarray(d["acc_count"], dtype=np.int64)
        ts = np.asarray(d.get("timestamp", np.full(acc.shape, np.nan)),
                        dtype=np.float64)
        nspec = d["_nspec"]
        span = int(acc[-1] - acc[0]) + 1
        per_file.append(dict(file=os.path.basename(fname), nspec=nspec,
                             acc_first=int(acc[0]), acc_last=int(acc[-1]),
                             span=span, missing=span - nspec,
                             nchan=d["_nchan"], dtype=d["_dtype"]))
        all_acc.append(acc)
        all_ts.append(ts)

    acc = np.concatenate(all_acc)
    ts = np.concatenate(all_ts)
    order = np.argsort(acc)
    acc, ts = acc[order], ts[order]

    diffs = np.diff(acc)
    n_dup = int(np.sum(diffs == 0))
    gap_idx = np.where(diffs > 1)[0]
    n_missing = int(np.sum(diffs[diffs > 1] - 1))
    n_total = len(acc)
    span_total = int(acc[-1] - acc[0]) + 1

    log(f"  files                    : {len(run.files)}")
    log(f"  spectra archived         : {n_total:,}")
    log(f"  acc_count span           : {span_total:,} "
        f"({acc[0]:,} .. {acc[-1]:,})")
    log(f"  duplicate acc_count      : {n_dup}")
    log(f"  missing accumulations    : {n_missing}")
    log(f"  gaps (discontinuities)   : {len(gap_idx)}")
    if len(gap_idx):
        log("    first few gaps:")
        for i in gap_idx[:10]:
            log(f"      after acc={acc[i]:,}: {diffs[i] - 1} missing")

    packets = n_total * PACKETS_PER_SPECTRUM
    wire_bytes = packets * BYTES_PER_PACKET_WIRE
    log(f"  --> packets delivered    : {packets:,}")
    log(f"  --> wire bytes           : {wire_bytes:,} "
        f"({wire_bytes / 2**30:.2f} GiB)")
    if n_missing == 0 and n_dup == 0:
        log(f"  PASS: zero loss over {packets:,} packets "
            f"({wire_bytes / 2**30:.2f} GiB on the wire).")
    else:
        log("  ATTENTION: continuity is not perfect -- see gap list above.")

    files_with_missing = [p for p in per_file if p["missing"] != 0]
    if files_with_missing:
        log(f"  files with internal gaps : {len(files_with_missing)}")

    return dict(n_spectra=n_total, acc_span=span_total, n_missing=n_missing,
                n_duplicate=n_dup, n_gaps=int(len(gap_idx)),
                packets=int(packets), wire_bytes=int(wire_bytes),
                per_file=per_file), acc, ts


def check_timing(run: RunSpec, acc, ts, log, results_dir, make_plots):
    log(f"\n=== [B] Timing fit: acc_len and sync period -- {run.name} ===")
    good = np.isfinite(ts) & (ts > 0)
    if good.sum() < 100:
        log("  Insufficient finite timestamps; skipping.")
        return {}

    a = acc[good].astype(np.float64)
    t = ts[good].astype(np.float64)
    a0 = a - a[0]

    slope, icept, keep = robust_linfit(a0, t)
    resid = t - (slope * a0 + icept)
    resid_rms = mad_std(resid[keep])

    n_edge = max(50, int(0.001 * len(a)))
    t_lo = np.median(t[:n_edge] - slope * a0[:n_edge]) + slope * np.median(a0[:n_edge])
    t_hi = np.median(t[-n_edge:] - slope * a0[-n_edge:]) + slope * np.median(a0[-n_edge:])
    da = np.median(a0[-n_edge:]) - np.median(a0[:n_edge])
    slope_endpoint = (t_hi - t_lo) / da if da else np.nan

    acc_len_fit = tacc_to_acc_len(slope)
    acc_len_end = tacc_to_acc_len(slope_endpoint)
    acc_len_round = int(round(acc_len_end if np.isfinite(acc_len_end) else acc_len_fit))

    total_span_s = t[-1] - t[0]
    sigma_slope = resid_rms / (np.std(a0) * np.sqrt(keep.sum())) if keep.sum() else np.nan
    sigma_acc_len = tacc_to_acc_len(sigma_slope)

    log(f"  usable timestamps        : {good.sum():,}")
    log(f"  baseline                 : {total_span_s / 3600:.3f} hr")
    log(f"  T_acc (robust fit)       : {slope * 1e3:.6f} ms")
    log(f"  T_acc (endpoint)         : {slope_endpoint * 1e3:.6f} ms")
    log(f"  host timestamp jitter    : {resid_rms * 1e3:.3f} ms rms (robust)")
    log(f"  --> acc_len (fit)        : {acc_len_fit:,.2f}")
    log(f"  --> acc_len (endpoint)   : {acc_len_end:,.2f}  "
        f"+/- {sigma_acc_len:,.2f} (stat)")
    log(f"  --> acc_len (integer)    : {acc_len_round:,}")
    log(f"      implies T_acc        = {acc_len_to_tacc(acc_len_round) * 1e3:.6f} ms")

    nominal_acc_len = int(round(tacc_to_acc_len(run.nominal_tacc_s)))
    cands = sorted({acc_len_round, nominal_acc_len,
                    int(round(tacc_to_acc_len(run.nominal_tacc_s))),
                    128160, 128176, 25632, 256352})
    log("  candidate discrimination (divergence over the full run):")
    for c in cands:
        tc = acc_len_to_tacc(c)
        if abs(tc - slope) > 0.25 * slope:
            continue
        drift = (tc - slope_endpoint) * len(a)
        log(f"      acc_len={c:>7,}  T_acc={tc * 1e3:11.6f} ms  "
            f"drift over run = {drift:+9.2f} s")

    sync_default_s = SYNC_PERIOD_DEFAULT_CLK / F_CLK_HZ
    log(f"  hard-coded sync default  : {sync_default_s * 1e3:.2f} ms")
    if slope > sync_default_s * 1.01:
        log(f"  PASS: measured T_acc ({slope * 1e3:.2f} ms) exceeds the "
            f"hard-coded sync period.")
        log(f"        -> sync_period_sel must have selected sync_period_var,")
        log(f"           set to >= {tacc_to_acc_len(slope):,.0f} clk-equivalents.")
        log(f"        -> integrations were NOT truncated by the master sync.")
        sync_verdict = "sync_period_var selected; no truncation"
    else:
        log(f"  NOTE: measured T_acc is at or below the hard-coded sync period;")
        log(f"        truncation cannot be excluded from timing alone.")
        sync_verdict = "inconclusive from timing alone"

    frac = (slope - acc_len_to_tacc(acc_len_round)) / slope
    log(f"  frac. offset vs integer  : {frac:+.3e} "
        f"({frac * 1e6:+.3f} ppm host-vs-FPGA)")

    if make_plots:
        _plot_timing(run, a0, resid, keep, results_dir)

    return dict(t_acc_fit_s=float(slope), t_acc_endpoint_s=float(slope_endpoint),
                acc_len_fit=float(acc_len_fit),
                acc_len_endpoint=float(acc_len_end),
                acc_len_integer=int(acc_len_round),
                acc_len_sigma=float(sigma_acc_len),
                resid_rms_s=float(resid_rms),
                baseline_hr=float(total_span_s / 3600),
                frac_offset=float(frac),
                sync_verdict=sync_verdict)


def _plot_timing(run, a0, resid, keep, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    step = max(1, len(a0) // 20000)
    ax.plot(a0[::step] , resid[::step] * 1e3, ".", ms=1, alpha=0.4,
            label="all")
    ax.plot(a0[keep][::step], resid[keep][::step] * 1e3, ".", ms=1,
            alpha=0.6, label="kept")
    ax.set_xlabel("acc_count - acc_count[0]")
    ax.set_ylabel("timestamp residual (ms)")
    ax.set_title(f"{run.name}: host timestamp vs acc_count, linear fit residual")
    ax.legend(markerscale=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{run.name}_timing_residual.png"),
                dpi=130)
    plt.close(fig)


def check_freq_axis(run: RunSpec, log):
    log(f"\n=== [C] Stored vs recomputed frequency axis -- {run.name} ===")
    log(f"  recomputed: f[k] = {F_S_HZ / 1e6:.3f} + k x "
        f"{CHAN_WIDTH_HZ / 1e3:.5f} kHz, "
        f"k=0..{N_CHAN - 1}  ({RF_MHZ[0]:.3f}..{RF_MHZ[-1]:.3f} MHz)")
    bad, good, missing = [], [], []
    for fname in run.files:
        d = read_scalar_datasets(fname)
        base = os.path.basename(fname)
        if "freq_mhz" not in d:
            missing.append(base)
            continue
        stored = np.asarray(d["freq_mhz"], dtype=np.float64)
        if stored.shape != RF_MHZ.shape:
            bad.append((base, f"shape {stored.shape} != {RF_MHZ.shape}"))
            continue
        dev = np.max(np.abs(stored - RF_MHZ))
        if dev > 1e-3:
            bad.append((base, f"max deviation {dev:.4f} MHz "
                              f"({dev / (CHAN_WIDTH_HZ / 1e6):.2f} channels); "
                              f"stored span {stored[0]:.3f}..{stored[-1]:.3f}"))
        else:
            good.append(base)
    log(f"  files with correct axis  : {len(good)}")
    log(f"  files with INCORRECT axis: {len(bad)}")
    for b, why in bad[:20]:
        log(f"      {b}: {why}")
    if missing:
        log(f"  files with no freq_mhz   : {len(missing)}")
    log("  (analysis always uses the recomputed axis regardless.)")
    return dict(n_good=len(good), n_bad=len(bad), n_missing=len(missing),
                bad_files=[b for b, _ in bad])


def check_cauchy_schwarz(run: RunSpec, log, chunk_rows, skip_spectra,
                         results_dir, make_plots):
    log(f"\n=== [D] Cauchy-Schwarz bound on |r| -- {run.name} ===")
    log("  r[k,t] = sqrt(cross_real^2+cross_imag^2)/sqrt(auto0*auto1)")
    log("  Must satisfy 0 <= r <= 1 for every channel and every spectrum")
    log("  if and only if the UFix_64_34 / Fix_64_34 word map is correct.")

    max_r = 0.0
    argmax = (None, None)
    n_viol = 0
    n_tested = 0
    n_nonpos = 0
    per_chan_max = np.zeros(N_CHAN)
    r_hist = np.zeros(220)
    hist_edges = np.linspace(0, 1.1, 221)
    seen = 0

    for fi, i0, blk in iter_spectra(run.files, chunk_rows=chunk_rows):
        a0, a1 = blk["auto0"], blk["auto1"]
        cr, ci = blk["cross_real"], blk["cross_imag"]
        nrow = a0.shape[0]
        if seen + nrow <= skip_spectra:
            seen += nrow
            continue
        lo = max(0, skip_spectra - seen)
        a0, a1, cr, ci = a0[lo:], a1[lo:], cr[lo:], ci[lo:]
        seen += nrow

        denom = a0 * a1
        ok = denom > 0
        n_nonpos += int(np.sum(~ok))
        r = np.zeros_like(denom)
        r[ok] = np.sqrt(cr[ok] ** 2 + ci[ok] ** 2) / np.sqrt(denom[ok])

        n_tested += int(ok.sum())
        viol = ok & (r > 1.0)
        n_viol += int(viol.sum())

        cm = r.max(axis=0)
        np.maximum(per_chan_max, cm, out=per_chan_max)

        m = r[ok].max() if ok.any() else 0.0
        if m > max_r:
            max_r = float(m)
            idx = np.unravel_index(np.argmax(np.where(ok, r, -1)), r.shape)
            argmax = (int(idx[0]), int(idx[1]))

        h, _ = np.histogram(r[ok], bins=hist_edges)
        r_hist += h

    log(f"  samples tested           : {n_tested:,}")
    log(f"  auto0*auto1 <= 0 samples : {n_nonpos:,}")
    log(f"  max |r| observed         : {max_r:.6f}")
    if argmax[1] is not None:
        log(f"      at channel {argmax[1]} ({RF_MHZ[argmax[1]]:.3f} MHz)")
    log(f"  violations (r > 1)       : {n_viol:,}")
    if n_viol == 0 and n_tested > 0:
        log("  PASS: Cauchy-Schwarz satisfied everywhere. The 64-bit field map,")
        log("        the unsigned/signed reinterpretation, and the common")
        log("        2^-34 scale are mutually consistent end to end.")
    elif n_tested:
        log("  FAIL: r > 1 occurs. Suspect the word map (Table 1), the")
        log("        byte-order detection, or the sign restoration.")

    if make_plots and n_tested:
        _plot_cs(run, per_chan_max, r_hist, hist_edges, results_dir)

    return dict(n_tested=int(n_tested), max_r=float(max_r),
                n_violations=int(n_viol), n_nonpositive=int(n_nonpos),
                argmax_channel=argmax[1],
                argmax_freq_mhz=(float(RF_MHZ[argmax[1]])
                                 if argmax[1] is not None else None))


def _plot_cs(run, per_chan_max, r_hist, edges, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].plot(RF_MHZ, per_chan_max, lw=0.7)
    axes[0].axhline(1.0, color="r", ls="--", lw=1, label="Cauchy-Schwarz bound")
    axes[0].set_xlabel("RF frequency (MHz)")
    axes[0].set_ylabel("max |r| over run")
    axes[0].set_title(f"{run.name}: per-channel maximum normalized correlation")
    axes[0].legend()
    centers = 0.5 * (edges[1:] + edges[:-1])
    axes[1].step(centers, np.maximum(r_hist, 0.1), where="mid", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].axvline(1.0, color="r", ls="--", lw=1)
    axes[1].set_xlabel("|r|")
    axes[1].set_ylabel("samples")
    axes[1].set_title("distribution of |r| (all channels, all spectra)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{run.name}_cauchy_schwarz.png"),
                dpi=130)
    plt.close(fig)


def check_radiometer(run: RunSpec, t_acc_s, log, chunk_rows, skip_spectra,
                     results_dir, make_plots):
    log(f"\n=== [E] Radiometer floor -- {run.name} ===")
    expected = 1.0 / np.sqrt(CHAN_WIDTH_HZ * t_acc_s)
    log(f"  channel width            : {CHAN_WIDTH_HZ / 1e3:.4f} kHz")
    log(f"  integration time         : {t_acc_s * 1e3:.4f} ms")
    log(f"  expected 1/sqrt(dnu*T)   : {expected:.6e} "
        f"({expected * 100:.4f} % per channel per accumulation)")

    idx = np.where(CLEAN_MASK)[0]
    sum_a0 = np.zeros(len(idx)); sum_a1 = np.zeros(len(idx))
    n_sum = 0
    diffs_a0, diffs_a1 = [], []
    r_meds = []
    last = None
    seen = 0

    for fi, i0, blk in iter_spectra(run.files, chunk_rows=chunk_rows):
        nrow = blk["auto0"].shape[0]
        if seen + nrow <= skip_spectra:
            seen += nrow
            last = None
            continue
        lo = max(0, skip_spectra - seen)
        seen += nrow
        a0 = blk["auto0"][lo:, idx]
        a1 = blk["auto1"][lo:, idx]
        cr = blk["cross_real"][lo:, idx]
        ci = blk["cross_imag"][lo:, idx]
        if a0.shape[0] < 2:
            continue

        sum_a0 += a0.sum(axis=0); sum_a1 += a1.sum(axis=0); n_sum += a0.shape[0]

        den = a0 * a1
        okd = den > 0
        norm = np.zeros_like(den)
        norm[okd] = cr[okd] / np.sqrt(den[okd])
        rr = np.zeros_like(den)
        rr[okd] = np.sqrt(cr[okd] ** 2 + ci[okd] ** 2) / np.sqrt(den[okd])
        r_meds.append((mad_std(norm[okd]),
                       np.median(rr[okd]) / np.sqrt(2 * np.log(2)))
                      if okd.any() else (np.nan, np.nan))

        if last is not None:
            a0d = np.vstack([last[0], a0])
            a1d = np.vstack([last[1], a1])
        else:
            a0d, a1d = a0, a1
        last = (a0[-1:], a1[-1:])
        d0 = np.diff(a0d, axis=0) / np.sqrt(2.0)
        d1 = np.diff(a1d, axis=0) / np.sqrt(2.0)
        diffs_a0.append(mad_std(d0, axis=0) / np.median(a0d, axis=0))
        diffs_a1.append(mad_std(d1, axis=0) / np.median(a1d, axis=0))

    frac0 = np.nanmedian(np.concatenate(diffs_a0)) if diffs_a0 else np.nan
    frac1 = np.nanmedian(np.concatenate(diffs_a1)) if diffs_a1 else np.nan
    rm = np.array(r_meds, dtype=np.float64) if r_meds else np.zeros((0, 2))
    r_floor = np.nanmedian(rm[:, 0]) if len(rm) else np.nan
    r_floor_ray = np.nanmedian(rm[:, 1]) if len(rm) else np.nan

    log(f"  auto0 fractional rms     : {frac0:.6e}  "
        f"(ratio to expected {frac0 / expected:.3f})")
    log(f"  auto1 fractional rms     : {frac1:.6e}  "
        f"(ratio to expected {frac1 / expected:.3f})")
    log(f"  cross real-part scatter  : {r_floor:.6e}  "
        f"(ratio to expected {r_floor / expected:.3f})   <- quantitative")
    log(f"  median |r| / sqrt(2ln2)  : {r_floor_ray:.6e}  "
        f"(ratio to expected {r_floor_ray / expected:.3f})")
    if np.isfinite(r_floor) and r_floor > 1.5 * expected:
        log("  NOTE: the cross floor sits well above the thermal expectation.")
        log("        Both estimators elevated together means genuine residual")
        log("        coherence between the two inputs -- common-mode pickup,")
        log("        crosstalk, or correlated sky in the 'clean' channels --")
        log("        rather than a noise-statistics artefact. This directly")
        log("        bounds the systematic floor of any cross-correlation")
        log("        result, so it is worth quoting in the paper either way.")
    elif np.isfinite(r_floor):
        log("  PASS: cross floor consistent with thermal noise; no measurable")
        log("        common-mode coherence between the two receiver chains.")

    t_eff0 = 1.0 / (CHAN_WIDTH_HZ * frac0 ** 2) if np.isfinite(frac0) else np.nan
    t_eff_r = (1.0 / (CHAN_WIDTH_HZ * r_floor ** 2)
               if np.isfinite(r_floor) and r_floor > 0 else np.nan)
    log(f"  --> T_eff from auto0     : {t_eff0 * 1e3:.4f} ms "
        f"(-> acc_len {tacc_to_acc_len(t_eff0):,.0f})")
    log(f"  --> T_eff from cross |r| : {t_eff_r * 1e3:.4f} ms "
        f"(-> acc_len {tacc_to_acc_len(t_eff_r):,.0f})")
    log("  Agreement with check [B] confirms the accumulator is summing the")
    log("  number of spectra implied by acc_len, independently of the host clock.")

    if make_plots and diffs_a0:
        _plot_radiometer(run, idx, np.nanmedian(np.vstack(diffs_a0), axis=0),
                         expected, results_dir)

    return dict(expected=float(expected), auto0_frac_rms=float(frac0),
                auto1_frac_rms=float(frac1), cross_r_floor=float(r_floor),
                cross_r_floor_rayleigh=float(r_floor_ray),
                t_eff_auto0_s=float(t_eff0), t_eff_cross_s=float(t_eff_r))


def _plot_radiometer(run, idx, per_chan, expected, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(RF_MHZ[idx], per_chan, ".", ms=2, alpha=0.5, label="measured")
    ax.axhline(expected, color="r", ls="--",
               label=r"$1/\sqrt{\Delta\nu\,T}$")
    ax.set_xlabel("RF frequency (MHz)")
    ax.set_ylabel("fractional rms (successive differences)")
    ax.set_title(f"{run.name}: radiometer floor, clean channels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{run.name}_radiometer.png"), dpi=130)
    plt.close(fig)


def _band_channels(f0, hw):
    return np.where(np.abs(RF_MHZ - f0) <= hw)[0]


def check_imd(run: RunSpec, log, chunk_rows, skip_spectra, results_dir,
              make_plots):
    log(f"\n=== [F] GNSS third-order intermodulation search -- {run.name} ===")
    log(f"  f1 = {IMD_F1} MHz (GLONASS L3), f2 = {IMD_F2} MHz (E5b/B2b), "
        f"separation {IMD_F2 - IMD_F1:.2f} MHz")
    log(f"  2*f1-f2 = {IMD_LO:.2f} MHz -> channel {chan_of(IMD_LO)}")
    log(f"  2*f2-f1 = {IMD_HI:.2f} MHz -> channel {chan_of(IMD_HI)}")
    log("  Both products land in-band in a clean region. This is a two-tone")
    log("  IMD test at the real operating point, with no second generator.")

    c_f1 = _band_channels(IMD_F1, 2.0)
    c_f2 = _band_channels(IMD_F2, 2.0)
    c_lo = _band_channels(IMD_LO, 0.8)
    c_hi = _band_channels(IMD_HI, 0.8)
    c_ref = np.concatenate([_band_channels(IMD_LO - 4.0, 0.8),
                            _band_channels(IMD_HI + 4.0, 0.8)])

    series = {k: [] for k in ("f1", "f2", "lo", "hi", "ref")}
    seen = 0
    for fi, i0, blk in iter_spectra(run.files, chunk_rows=chunk_rows,
                                    datasets=("auto0", "auto1")):
        nrow = blk["auto0"].shape[0]
        if seen + nrow <= skip_spectra:
            seen += nrow
            continue
        lo_i = max(0, skip_spectra - seen)
        seen += nrow
        p = 0.5 * (blk["auto0"][lo_i:] + blk["auto1"][lo_i:])
        series["f1"].append(p[:, c_f1].mean(axis=1))
        series["f2"].append(p[:, c_f2].mean(axis=1))
        series["lo"].append(p[:, c_lo].mean(axis=1))
        series["hi"].append(p[:, c_hi].mean(axis=1))
        series["ref"].append(p[:, c_ref].mean(axis=1))

    S = {k: np.concatenate(v) for k, v in series.items() if v}
    if not S:
        log("  No data after transient cut; skipping.")
        return {}

    floor = np.median(S["ref"])
    e_f1 = S["f1"] - np.median(S["f1"])
    e_f2 = S["f2"] - np.median(S["f2"])
    e_lo = S["lo"] - np.median(S["lo"])
    e_hi = S["hi"] - np.median(S["hi"])

    drive_lin = (S["f1"] ** 2) * S["f2"]
    thr = np.percentile(drive_lin, 99.5)
    both = drive_lin >= thr
    quiet = drive_lin <= np.percentile(drive_lin, 50.0)
    log(f"  spectra analysed         : {len(S['f1']):,}")
    log(f"  top-0.5% drive epochs    : {int(both.sum()):,}")
    log(f"  drive dynamic range      : "
        f"{10 * np.log10(np.median(drive_lin[both]) / max(np.median(drive_lin[quiet]), 1e-300)):.1f} dB")

    out = dict(n_spectra=int(len(S["f1"])), n_both=int(both.sum()))
    if both.sum() < 10:
        log("  Too few high-drive epochs for a meaningful limit; use more files.")
        return out

    sig_q = mad_std(S["lo"][quiet]) if quiet.sum() > 10 else np.nan
    for tag, arr, exc in (("2f1-f2", S["lo"], e_lo), ("2f2-f1", S["hi"], e_hi)):
        strong = np.median(arr[both])
        base = np.median(arr[quiet]) if quiet.sum() > 10 else floor
        delta = strong - base
        fund = np.median(0.5 * (S["f1"][both] + S["f2"][both]))
        dbc = 10 * np.log10(abs(delta) / fund) if delta > 0 and fund > 0 else np.nan
        sig = delta / sig_q if np.isfinite(sig_q) and sig_q > 0 else np.nan
        log(f"  {tag}: excess during strong transits = {delta:.4e} "
            f"({sig:.2f} sigma vs quiet scatter)")
        if np.isfinite(dbc):
            log(f"          relative to fundamental : {dbc:.1f} dBc")
        else:
            log(f"          no positive excess -> IMD below the noise here")
        out[tag] = dict(excess=float(delta), sigma=float(sig),
                        dbc=float(dbc) if np.isfinite(dbc) else None)

    with np.errstate(divide="ignore", invalid="ignore"):
        drive = 2 * np.log10(np.maximum(S["f1"], 1e-30)) + \
                np.log10(np.maximum(S["f2"], 1e-30))
        resp = np.log10(np.maximum(e_lo, 1e-30))
    m = both & np.isfinite(drive) & np.isfinite(resp) & (e_lo > 0)
    if m.sum() > 20:
        sl, ic, _ = robust_linfit(drive[m], resp[m])
        log(f"  log-log slope of 2f1-f2 excess vs P1^2*P2 : {sl:.2f}")
        log("     (slope ~1 supports a genuine third-order product;")
        log("      slope ~0 suggests the 'excess' is beam-correlated leakage)")
        out["scaling_slope"] = float(sl)

    np.savez_compressed(os.path.join(results_dir, f"{run.name}_imd_series.npz"),
                        **S)
    if make_plots:
        _plot_imd(run, S, both, results_dir)
    return out


def _plot_imd(run, S, both, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    n = len(S["f1"])
    x = np.arange(n)
    step = max(1, n // 40000)
    axes[0].semilogy(x[::step], S["f1"][::step], lw=0.5, label="GLONASS L3")
    axes[0].semilogy(x[::step], S["f2"][::step], lw=0.5, label="E5b/B2b")
    axes[0].legend(); axes[0].set_ylabel("band power (arb)")
    axes[0].set_title(f"{run.name}: fundamentals and IMD products")
    axes[1].semilogy(x[::step], np.maximum(S["lo"][::step], 1e-30), lw=0.5,
                     label="2f1-f2")
    axes[1].semilogy(x[::step], np.maximum(S["hi"][::step], 1e-30), lw=0.5,
                     label="2f2-f1")
    axes[1].semilogy(x[::step], np.maximum(S["ref"][::step], 1e-30), lw=0.5,
                     color="k", alpha=0.5, label="reference shoulder")
    axes[1].legend(); axes[1].set_ylabel("band power (arb)")
    axes[1].set_xlabel("spectrum index (post-transient)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{run.name}_imd.png"), dpi=130)
    plt.close(fig)


def run_all(run: RunSpec, args, log, results_dir):
    log("\n" + "=" * 74)
    log(f"RUN: {run.name}   ({run.path})")
    log(f"     nominal accumulation {run.nominal_tacc_s * 1e3:.1f} ms, "
        f"{len(run.files)} files")
    log("=" * 74)

    res = {}
    res["continuity"], acc, ts = check_continuity(run, log)
    res["timing"] = check_timing(run, acc, ts, log, results_dir,
                                 not args.no_plots)

    t_acc = res["timing"].get("t_acc_endpoint_s") or run.nominal_tacc_s
    if not np.isfinite(t_acc) or t_acc <= 0:
        t_acc = run.nominal_tacc_s

    res["freq_axis"] = check_freq_axis(run, log)

    skip_spectra = int(round(args.skip_hours * 3600.0 / t_acc))
    log(f"\n  [transient cut] skipping first {args.skip_hours} hr "
        f"= {skip_spectra:,} spectra for checks [E] and [F]")

    if not args.skip_heavy:
        res["cauchy_schwarz"] = check_cauchy_schwarz(
            run, log, args.chunk_rows, 0, results_dir, not args.no_plots)
        res["radiometer"] = check_radiometer(
            run, t_acc, log, args.chunk_rows, skip_spectra, results_dir,
            not args.no_plots)
        res["imd"] = check_imd(
            run, log, args.chunk_rows, skip_spectra, results_dir,
            not args.no_plots)
    else:
        log("\n  [--skip-heavy] checks D, E, F not run.")
    return res


def main():
    ap = argparse.ArgumentParser(
        description="Tier-0 archived-data checks for the BMX RFSoC backend.")
    ap.add_argument("--run2", help="path to bmx_run2_500ms_1050mhz")
    ap.add_argument("--run3", help="path to bmx_run3_100ms_18hrs")
    ap.add_argument("--outdir", default="bmx_check_results")
    ap.add_argument("--chunk-rows", type=int, default=4000,
                    help="spectra per HDF5 read (memory knob)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="limit files per run, for a fast smoke test")
    ap.add_argument("--skip-hours", type=float, default=6.0,
                    help="hours to drop at run start (startup transient)")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="run only the cheap 1-D checks A/B/C")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not (args.run2 or args.run3):
        ap.error("give at least one of --run2 / --run3")

    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("BMX RFSoC backend -- archived-data checks")
    log(f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"numpy {np.__version__}, h5py {h5py.__version__}")

    runs = []
    if args.run2:
        runs.append(RunSpec("run2_500ms", args.run2, 0.500,
                            find_files(args.run2, args.max_files)))
    if args.run3:
        runs.append(RunSpec("run3_100ms", args.run3, 0.100,
                            find_files(args.run3, args.max_files)))

    results = {}
    for r in runs:
        results[r.name] = run_all(r, args, log, args.outdir)

    if len(runs) == 2:
        log("\n" + "=" * 74)
        log("CROSS-RUN CONSISTENCY")
        log("=" * 74)
        try:
            a2 = results["run2_500ms"]["timing"]["acc_len_integer"]
            a3 = results["run3_100ms"]["timing"]["acc_len_integer"]
            log(f"  run2 acc_len = {a2:,}, run3 acc_len = {a3:,}, "
                f"ratio = {a2 / a3:.5f}")
            if abs(a2 / a3 - 5.0) < 0.01:
                log("  run2 is exactly 5x run3 -- consistent with 128160 = 5 x 25632.")
            tp = results["run2_500ms"]["continuity"]["packets"] + \
                 results["run3_100ms"]["continuity"]["packets"]
            tb = results["run2_500ms"]["continuity"]["wire_bytes"] + \
                 results["run3_100ms"]["continuity"]["wire_bytes"]
            ta = results["run2_500ms"]["continuity"]["n_spectra"] + \
                 results["run3_100ms"]["continuity"]["n_spectra"]
            tm = results["run2_500ms"]["continuity"]["n_missing"] + \
                 results["run3_100ms"]["continuity"]["n_missing"]
            log(f"  COMBINED: {ta:,} accumulations, {tp:,} packets, "
                f"{tb / 2**30:.2f} GiB, {tm} lost.")
            log("  ^ this is the sentence to put in the abstract.")
        except (KeyError, TypeError):
            log("  (timing results incomplete; skipping)")

    with open(os.path.join(args.outdir, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {args.outdir}/report.txt and {args.outdir}/results.json")


if __name__ == "__main__":
    main()
