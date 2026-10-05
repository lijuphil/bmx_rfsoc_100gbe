#!/usr/bin/env python3
import glob
import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import timezone
import datetime as dtm
from zoneinfo import ZoneInfo

NFFT = 4096
FS_HZ = 1050e6
ZONE = 3

BMX_LAT = 40.869944
BMX_LON = -72.865750
BMX_ELEV = 20.0

TZ = timezone.utc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "bmx_run2_500ms_1050mhz")
OUT_DIR = HERE
REPORT_PATH = os.path.join(OUT_DIR, "report_vis002.txt")

GNSS_LINES = [
    ("GPS L5 / Galileo E5a",     1176.45, 12.0),
    ("GLONASS L3",               1202.03,  5.0),
    ("Galileo E5b / BeiDou B2b", 1207.14,  6.0),
    ("GPS L2",                   1227.60, 12.0),
    ("BeiDou B3",                1268.52, 10.0),
    ("Galileo E6",               1278.75, 10.0),
    ("GPS L1 / Galileo E1",      1575.42, 12.0),
    ("Inmarsat/MSS downlink",    1542.0,   8.0),
]

RADIO_SOURCES = {
    "Cygnus A":        ("19h59m28.36s", "+40d44m02.1s"),
    "Cassiopeia A":    ("23h23m24s",    "+58d48m54s"),
    "Taurus A (Crab)": ("05h34m31.94s", "+22d00m52.2s"),
    "Virgo A (M87)":   ("12h30m49.42s", "+12d23m28s"),
}

_report_lines = []


def rlog(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    _report_lines.append(s)


def rf_axis(nchan, fs_hz, zone, nfft=NFFT):
    fbb = np.arange(nchan) * (fs_hz / nfft) / 1e6
    half = (fs_hz / 2) / 1e6
    z = int(zone)
    return (z - 1) * half + fbb if z % 2 else z * half - fbb


def load(files):
    keys = ("auto0", "auto1", "cross_real", "cross_imag")
    data = {k: [] for k in keys}
    acc, ts = [], []
    nchan = None
    ts_dataset_names = None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if ts_dataset_names is None:
                ts_dataset_names = [k for k in f.keys() if "timestamp" in k]
            if f["auto0"].shape[0] == 0:
                continue
            for k in keys:
                data[k].append(f[k][:])
            acc.append(f["acc_count"][:])
            ts.append(f["timestamp"][:])
            if nchan is None:
                nchan = f["auto0"].shape[1]
    out = {k: np.concatenate(v, axis=0) for k, v in data.items()}
    out["acc_count"] = np.concatenate(acc)
    out["timestamp"] = np.concatenate(ts)
    order = np.argsort(out["acc_count"], kind="stable")
    for k in list(out):
        out[k] = out[k][order]
    return out, nchan, ts_dataset_names


def db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.asarray(x, dtype=float)
        return 10 * np.log10(np.where(x > 0, x, np.nan))


def waterfalls_local(d, freq, out):
    SPECTRA = ("auto0", "auto1", "cross_real", "cross_imag")
    TITLES = {"auto0": "East-X auto (auto0)", "auto1": "West-X auto (auto1)",
              "cross_real": "cross real (E x W)", "cross_imag": "cross imag (E x W)"}

    ts = np.asarray(d["timestamp"], dtype=float)
    dt0 = dtm.datetime.fromtimestamp(ts[0], tz=timezone.utc).astimezone(TZ)
    dt1 = dtm.datetime.fromtimestamp(ts[-1], tz=timezone.utc).astimezone(TZ)
    y0, y1 = mdates.date2num(dt0), mdates.date2num(dt1)
    extent = [float(freq[0]), float(freq[-1]), y0, y1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), SPECTRA):
        img = d[name]
        signed = name.startswith("cross")
        if signed:
            v = np.nanpercentile(np.abs(img), 99.5)
            v = float(v) if np.isfinite(v) and v > 0 else 1.0
            kw = dict(cmap="RdBu_r", vmin=-v, vmax=v)
            cbar_label = "amplitude (arb)"
            img_show = img
        else:
            w = db(img)
            fin = w[np.isfinite(w)]
            lo, hi = (np.percentile(fin, [5, 99.5]) if fin.size else (0.0, 1.0))
            kw = dict(cmap="viridis", vmin=lo, vmax=hi)
            cbar_label = "dB (arb)"
            img_show = w

        im = ax.imshow(img_show, aspect="auto", origin="lower", extent=extent,
                        interpolation="nearest", **kw)
        ax.set_title(TITLES[name])
        fig.colorbar(im, ax=ax, label=cbar_label, pad=0.015)

    span_s = (dt1 - dt0).total_seconds()
    tfmt = "%H:%M" if span_s < 86400 else "%m-%d %H:%M"
    ticks = np.linspace(y0, y1, 8)
    fmt = mdates.DateFormatter(tfmt, tz=TZ)
    for ax in axes.ravel():
        ax.yaxis_date(tz=TZ)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(fmt)
        ax.tick_params(axis="y", labelsize=9)
    axes[0, 0].set_ylim(y1, y0)

    for ax in axes[1, :]:
        ax.set_xlabel("RF frequency (MHz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"time (UTC)")

    n = d["auto0"].shape[0]
    fig.suptitle(f"BMX run2 vis_002* ({n} spectra)   "
                 f"{dt0:%Y-%m-%d %H:%M} - {dt1:%H:%M} {dt1.tzname()}   |   "
                 f"zone {ZONE}, {freq[0]:.1f}-{freq[-1]:.1f} MHz", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    rlog(f"wrote {out}")


def gaussian(t, A, t0, sig, C):
    return A * np.exp(-0.5 * ((t - t0) / sig) ** 2) + C


def fit_local_gaussian(tsec, power, i0, half_win):
    lo = max(0, i0 - half_win)
    hi = min(len(tsec), i0 + half_win)
    tt, pp = tsec[lo:hi], power[lo:hi]
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return None
    C0 = np.median(pp)
    A0 = pp.max() - C0
    t00 = tt[np.argmax(pp)]
    s0 = max((tt[-1] - tt[0]) / 6.0, 1.0)
    try:
        p, _ = curve_fit(gaussian, tt, pp, p0=[A0, t00, s0, C0], maxfev=20000)
        A, t0, sig, C = p
        if A <= 0 or abs(sig) > (tt[-1] - tt[0]):
            return None
        return dict(A=A, t0=t0, sigma_s=abs(sig), fwhm_s=2.3548 * abs(sig), C=C)
    except Exception:
        return None


def gnss_transits(d, freq, out_plot):
    from scipy.signal import find_peaks
    from scipy.ndimage import percentile_filter

    auto = 0.5 * (d["auto0"] + d["auto1"])
    cr = d["cross_real"]
    ts = d["timestamp"]
    t = ts - ts[0]
    dt = np.median(np.diff(t))
    cadence_s = float(dt)

    rows = []
    total_events = 0
    for name, fc, hw in GNSS_LINES:
        if fc < freq[0] or fc > freq[-1]:
            continue
        sel = (freq >= fc - hw) & (freq <= fc + hw)
        bp = auto[:, sel].mean(1)

        win = max(41, int(1800 / cadence_s) | 1)
        baseline = percentile_filter(bp, percentile=15, size=win, mode="nearest")
        baseline_safe = np.where(baseline > 0, baseline, np.nan)
        frac = bp / baseline_safe - 1.0
        resid = bp - baseline
        noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-30

        min_sep_s = 60.0
        distance = max(1, int(min_sep_s / cadence_s))
        peaks, props = find_peaks(frac, height=0.5, distance=distance,
                                   prominence=0.3)
        peaks = peaks[resid[peaks] >= 6 * noise]

        events = []
        half_win = max(5, int(120 / cadence_s))
        for pk in peaks:
            fit = fit_local_gaussian(t, bp, pk, half_win)
            selc = sel
            fr = cr[:, selc].mean(1)
            lo, hi = max(0, pk - half_win), min(len(t), pk + half_win)
            fr_amp = np.std(fr[lo:hi]) / max(np.mean(bp[lo:hi]), 1e-30)
            events.append(dict(idx=int(pk), t0=float(t[pk]),
                                snr=float(resid[pk] / noise), fit=fit,
                                fringe_amp=float(fr_amp)))
        rows.append((name, fc, hw, bp, baseline, resid, events))
        total_events += len(events)

    rlog("\n-- GNSS / L-band satellite transit inventory (vis_002*) --")
    rlog(f"{'line':26} {'MHz':>8} {'events':>7} {'avg FWHM(s)':>12} {'fringing':>9}")
    for name, fc, hw, bp, baseline, resid, events in rows:
        fw = [e["fit"]["fwhm_s"] for e in events if e["fit"]]
        avg_fwhm = np.mean(fw) if fw else float("nan")
        n_fringe = sum(1 for e in events if e["fringe_amp"] > 0.02)
        rlog(f"{name:26} {fc:8.2f} {len(events):7d} {avg_fwhm:12.1f} "
             f"{n_fringe:4d}/{len(events)}")
    rlog(f"\nTotal individual satellite transit EVENTS detected across all "
         f"lines: {total_events}")
    rlog("(Each event = one satellite crossing the fixed zenith beam on one "
         "GNSS frequency; the same physical satellite is not deduplicated "
         "across different signal bands it may transmit on.)")

    active = [r for r in rows if len(r[6]) > 0]
    if active:
        k = len(active)
        fig, axes = plt.subplots(k, 1, figsize=(12, 2.3 * k + 1), sharex=False,
                                  squeeze=False)
        for ax, (name, fc, hw, bp, baseline, resid, events) in zip(axes[:, 0], active):
            tl_dt = [dtm.datetime.fromtimestamp(ts[0] + tt, tz=timezone.utc)
                     .astimezone(TZ) for tt in [0, t[-1]]]
            tmin = np.array([dtm.datetime.fromtimestamp(ts[0], tz=timezone.utc)
                             .astimezone(TZ) + dtm.timedelta(seconds=float(tt))
                             for tt in t])
            ax.plot(tmin, bp, lw=0.5, color="C0")
            for e in events:
                ax.axvline(tmin[e["idx"]], color="r", ls=":", lw=0.8, alpha=0.7)
            ax.set_title(f"{name} ~{fc:.1f} MHz  ({len(events)} events)", fontsize=9)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
        axes[-1, 0].set_xlabel(f"time (UTC)")
        fig.suptitle("BMX run2 vis_002*: GNSS L-band transits")
        fig.tight_layout()
        fig.savefig(out_plot, dpi=130)
        plt.close(fig)
        rlog(f"wrote {out_plot}")

    return total_events, rows


LINE_CONSTELLATIONS = {
    "GPS L5 / Galileo E5a":       ["GPS", "Galileo"],
    "GLONASS L3":                 ["GLONASS"],
    "Galileo E5b / BeiDou B2b":   ["Galileo", "BeiDou"],
    "GPS L2":                     ["GPS"],
    "BeiDou B3":                  ["BeiDou"],
    "Galileo E6":                 ["Galileo"],
    "Inmarsat/MSS downlink":      [],
}

TLE_FILES = {
    "GPS": "gps-ops.tle",
    "GLONASS": "glo-ops.tle",
    "Galileo": "galileo.tle",
    "BeiDou": "beidou.tle",
}


def beam_map(ts, gnss_rows, out_dir):
    try:
        from skyfield.api import load, wgs84
    except Exception:
        rlog("\nskyfield not available: skipping satellite beam mapping")
        return

    tle_dir = os.path.join(out_dir, "tle")
    constellations = {}
    for cname, fn in TLE_FILES.items():
        path = os.path.join(tle_dir, fn)
        if os.path.exists(path):
            constellations[cname] = load.tle_file(path)
    if not constellations:
        rlog("\nno TLE files found: skipping satellite beam mapping")
        return

    sf_ts = load.timescale()
    site = wgs84.latlon(BMX_LAT, BMX_LON, elevation_m=BMX_ELEV)
    cadence_s = float(np.median(np.diff(ts)))
    t0u = ts[0]

    rlog("\n-- Beam mapping from matched GNSS transits --")
    n_cons = {c: len(s) for c, s in constellations.items()}
    rlog(f"TLE catalog sizes: {n_cons}")

    CORE_THRESH_DEG = 5.0
    WIDE_WIN_S = 900.0

    samples = []
    grazing_samples = []
    match_log = []
    seen_passes = set()
    for name, fc, hw, bp, baseline, resid, events in gnss_rows:
        cands = LINE_CONSTELLATIONS.get(name, [])
        sats = [s for c in cands for s in constellations.get(c, [])]
        if not sats or not events:
            continue

        for e in events:
            t_peak_unix = t0u + e["t0"]
            t_peak = sf_ts.utc(dtm.datetime.fromtimestamp(t_peak_unix, tz=timezone.utc))
            best_sat, best_zd = None, 1e9
            for sat in sats:
                alt, az, _ = (sat - site).at(t_peak).altaz()
                zd = 90.0 - alt.degrees
                if zd < best_zd:
                    best_sat, best_zd = sat, zd
            if best_sat is None or best_zd > 75.0:
                continue

            wide_lo = max(0, np.searchsorted(ts, t_peak_unix - WIDE_WIN_S))
            wide_hi = min(len(ts), np.searchsorted(ts, t_peak_unix + WIDE_WIN_S))
            twide = ts[wide_lo:wide_hi]
            t_sf_wide = sf_ts.utc([dtm.datetime.fromtimestamp(tt, tz=timezone.utc) for tt in twide])
            alt_w, az_w, _ = (best_sat - site).at(t_sf_wide).altaz()
            offset_w = 90.0 - alt_w.degrees
            i_min = int(np.argmin(offset_w))
            min_offset = float(offset_w[i_min])
            t_min_unix = float(twide[i_min])

            pass_key = (best_sat.name, round(t_min_unix / 1800.0))
            if pass_key in seen_passes:
                continue
            seen_passes.add(pass_key)

            fwhm = e["fit"]["fwhm_s"] if e["fit"] else 60.0
            half_win = int(max(3 * fwhm, 120.0) / cadence_s)
            idx_min = wide_lo + i_min
            lo, hi = max(0, idx_min - half_win), min(len(ts), idx_min + half_win)

            base_win = baseline[lo:hi]
            bp_win = bp[lo:hi]
            frac = (bp_win - base_win) / np.maximum(base_win, 1e-30)
            i_ref = np.argmin(np.abs(ts[lo:hi] - t_min_unix))
            ref_val = frac[i_ref]
            if ref_val <= 0:
                continue
            norm = frac / ref_val

            t_sf = sf_ts.utc([dtm.datetime.fromtimestamp(tt, tz=timezone.utc) for tt in ts[lo:hi]])
            alt, az, _ = (best_sat - site).at(t_sf).altaz()
            offset = 90.0 - alt.degrees

            keep = (offset < 90.0) & np.isfinite(norm)
            pts = np.column_stack([offset[keep], norm[keep]])
            match_log.append(dict(line=name, sat=best_sat.name, zd_peak=float(best_zd),
                                    min_offset=min_offset, t_peak=t_peak_unix))
            if min_offset <= CORE_THRESH_DEG:
                samples.append(pts)
            else:
                grazing_samples.append(pts)

    if not samples:
        rlog("no transits passed close enough to zenith (<=%.0f deg) to "
             "calibrate a beam profile: skipping" % CORE_THRESH_DEG)
        return

    n_core = sum(1 for m in match_log if m["min_offset"] <= CORE_THRESH_DEG)
    rlog(f"matched {len(match_log)}/{sum(len(r[6]) for r in gnss_rows)} events to a "
         f"specific satellite (zenith-dist <= 75 deg at detected peak)")
    rlog(f"  of these, {n_core} tracks pass within {CORE_THRESH_DEG:.0f} deg of true "
         f"zenith (boresight) -- only these are self-normalization-safe and "
         f"used for the calibrated radial profile below.")
    rlog(f"  the remaining {len(match_log)-n_core} are grazing passes (never "
         f"closer than {CORE_THRESH_DEG:.0f} deg); shown faded, NOT used for "
         f"the fit, since self-normalizing a grazing pass to its own local "
         f"peak falsely implies unit gain at a large offset.")
    rlog("\nboresight-crossing tracks used for calibration:")
    for m in sorted(match_log, key=lambda x: x["t_peak"]):
        if m["min_offset"] <= CORE_THRESH_DEG:
            lt = dtm.datetime.fromtimestamp(m["t_peak"], tz=timezone.utc).astimezone(TZ)
            rlog(f"  {lt:%H:%M:%S}  {m['line']:26} -> {m['sat']:22} "
                 f"min offset={m['min_offset']:5.2f} deg")

    samples = np.concatenate(samples, axis=0)
    offsets, norms = samples[:, 0], samples[:, 1]
    graz_offsets, graz_norms = (np.concatenate(grazing_samples, axis=0)[:, 0],
                                 np.concatenate(grazing_samples, axis=0)[:, 1]) \
        if grazing_samples else (np.array([]), np.array([]))
    rlog(f"\n{len(offsets)} (offset, power) samples from {n_core} boresight-crossing "
         f"tracks used for the calibrated profile; {len(graz_offsets)} samples from "
         f"{len(match_log)-n_core} grazing tracks shown for context only.")

    edges = np.arange(0, min(45, offsets.max() + 2), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(len(centers), np.nan)
    p25 = np.full(len(centers), np.nan)
    p75 = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        sel = (offsets >= edges[i]) & (offsets < edges[i + 1])
        counts[i] = sel.sum()
        if sel.sum() >= 5:
            med[i] = np.median(norms[sel])
            p25[i], p75[i] = np.percentile(norms[sel], [25, 75])

    fwhm_deg = None
    try:
        from scipy.optimize import curve_fit
        good = np.isfinite(med) & (counts >= 5)
        if good.sum() >= 4:
            def g(x, sig):
                return np.exp(-0.5 * (x / sig) ** 2)
            p, _ = curve_fit(g, centers[good], med[good], p0=[10.0], maxfev=10000)
            fwhm_deg = 2.3548 * abs(p[0])
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if len(graz_offsets):
        axes[0].scatter(graz_offsets, graz_norms, s=2, alpha=0.04, color="0.6",
                         label="grazing tracks (not calibrated, context only)")
    axes[0].scatter(offsets, norms, s=3, alpha=0.25, color="C0",
                     label="boresight-crossing tracks")
    axes[0].plot(centers, med, "r-", lw=1.8, label="median (1 deg bins)")
    axes[0].fill_between(centers, p25, p75, color="r", alpha=0.2, label="IQR")
    if fwhm_deg:
        axes[0].axvline(fwhm_deg / 2, color="k", ls="--", lw=1,
                         label=f"HWHM ~ {fwhm_deg/2:.1f} deg")
    axes[0].set_xlabel("angular offset from zenith boresight (deg)")
    axes[0].set_ylabel("normalized power (closest-approach = 1)")
    axes[0].set_title("Beam radial profile (linear)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, edges[-1])
    axes[0].set_ylim(-0.05, 1.15)

    if len(graz_offsets):
        axes[1].scatter(graz_offsets, db(graz_norms), s=2, alpha=0.04, color="0.6")
    axes[1].scatter(offsets, db(norms), s=3, alpha=0.25, color="C0")
    axes[1].plot(centers, db(med), "r-", lw=1.8)
    axes[1].axhline(-3, color="0.5", ls=":", lw=0.8, label="-3 dB")
    if fwhm_deg:
        axes[1].axvline(fwhm_deg / 2, color="k", ls="--", lw=1,
                         label=f"FWHM ~ {fwhm_deg:.1f} deg")
    axes[1].set_xlabel("angular offset from zenith boresight (deg)")
    axes[1].set_ylabel("normalized power (dB)")
    axes[1].set_title("Beam radial profile (dB)")
    axes[1].set_xlim(0, edges[-1])
    axes[1].set_ylim(-25, 2)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"BMX zenith beam, radial profile from {n_core} boresight-crossing "
                 f"GNSS transits (<= {CORE_THRESH_DEG:.0f} deg, vis_002*)")
    fig.tight_layout()
    out = os.path.join(out_dir, "beam_profile_vis002.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    rlog(f"wrote {out}")

    rlog(f"\nBinned profile (deg, median norm power, N samples):")
    for c, m_, n_ in zip(centers, med, counts):
        if n_ > 0:
            rlog(f"  {c:5.1f}  {m_:6.3f}  N={n_}")
    if fwhm_deg:
        rlog(f"\nFitted Gaussian beam FWHM (from {n_core} boresight-crossing, "
             f"self-normalized transits): {fwhm_deg:.1f} deg")
    else:
        rlog("\nGaussian fit to the radial profile did not converge "
             "(insufficient offset coverage close to boresight).")
    rlog(f"Caveat: each track is normalized to its power at its own true "
         f"closest-approach point (found from a wide +/-{WIDE_WIN_S/60:.0f} min "
         f"search, not just the auto-detected event peak), so this measures "
         f"the RELATIVE beam shape, not absolute gain. Only tracks confirmed "
         f"to pass within {CORE_THRESH_DEG:.0f} deg of true zenith are used for "
         f"the fit -- grazing tracks are excluded because normalizing them to "
         f"their own local peak would falsely claim unit gain far off "
         f"boresight. TLEs are propagated ~2 days from the TLE epoch to the "
         f"observation date, adequate for sub-0.1 deg accuracy on MEO orbits.")


def radio_source_check(d, freq, out_plot):
    try:
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun
    except Exception:
        rlog("\nastropy not available: skipping celestial radio-source check")
        return

    import astropy.units as u

    loc = EarthLocation(lat=BMX_LAT * u.deg, lon=BMX_LON * u.deg, height=BMX_ELEV * u.m)
    ts = d["timestamp"]
    t0u, t1u = ts[0], ts[-1]

    tgrid = Time(t0u - 4 * 3600, format="unix") + \
        (Time(t1u + 4 * 3600, format="unix") - Time(t0u - 4 * 3600, format="unix")) * \
        np.linspace(0, 1, 4000)
    aa = AltAz(obstime=tgrid, location=loc)

    in_win = (tgrid.unix >= t0u) & (tgrid.unix <= t1u)

    rlog("\n-- Celestial radio-source zenith-transit geometry --")
    rlog(f"BMX site: {BMX_LAT:.6f} N, {-BMX_LON:.6f} W (zenith-pointing)")
    rlog(f"vis_002* window: {t0u:.0f} .. {t1u:.0f} unix "
         f"({dtm.datetime.fromtimestamp(t0u, tz=timezone.utc).astimezone(TZ):%Y-%m-%d %H:%M} - "
         f"{dtm.datetime.fromtimestamp(t1u, tz=timezone.utc).astimezone(TZ):%H:%M} UTC)")

    findings = {}
    for name, (ra, dec) in RADIO_SOURCES.items():
        sc = SkyCoord(ra=ra, dec=dec)
        altaz = sc.transform_to(aa)
        i = int(np.argmax(altaz.alt.deg))
        t_transit = tgrid[i]
        zd = 90 - altaz.alt.deg[i]
        inside = bool(in_win[i])
        local_t = t_transit.to_datetime(timezone.utc).astimezone(TZ)
        rlog(f"  {name:18} dec={sc.dec.deg:+6.2f}  true meridian transit "
             f"zenith-dist {zd:6.2f} deg at {local_t:%Y-%m-%d %H:%M:%S} UTC "
             f"-- {'INSIDE' if inside else 'OUTSIDE'} vis_002* window")
        findings[name] = dict(zd=zd, t_transit=t_transit.unix, inside=inside)

    sun_aa = get_sun(tgrid).transform_to(aa)
    i = int(np.argmax(sun_aa.alt.deg))
    zd_sun = 90 - sun_aa.alt.deg[i]
    local_t = tgrid[i].to_datetime(timezone.utc).astimezone(TZ)
    inside = bool(in_win[i])
    rlog(f"  {'Sun':18}        true meridian transit "
         f"zenith-dist {zd_sun:6.2f} deg at {local_t:%Y-%m-%d %H:%M:%S} UTC "
         f"-- {'INSIDE' if inside else 'OUTSIDE'} vis_002* window")
    below = tgrid[sun_aa.alt.deg < 0]
    if below.size and (below.unix.min() >= t0u) and (below.unix.min() <= t1u):
        st = below[0].to_datetime(timezone.utc).astimezone(TZ)
        rlog(f"    sunset within window at {st:%H:%M:%S} UTC")

    auto = 0.5 * (d["auto0"] + d["auto1"])
    clean = np.ones_like(freq, dtype=bool)
    clean &= (freq > freq[0] + 5) & (freq < freq[-1] - 5)
    for _, fc, hw in GNSS_LINES:
        clean &= ~((freq >= fc - hw - 3) & (freq <= fc + hw + 3))
    clean &= ~((freq > 1417) & (freq < 1424))
    rlog(f"\nBroadband (source-search) band: {clean.sum()} clean channels "
         f"of {len(freq)} (GNSS lines + band edges + HI masked out)")

    bandpass = np.median(auto[:, clean], axis=0)
    bpsafe = np.where(bandpass > 0, bandpass, np.nan)
    resid = auto[:, clean] / bpsafe - 1.0
    broadband = np.nanmean(resid, axis=1)

    t_local = np.array([dtm.datetime.fromtimestamp(tt, tz=timezone.utc).astimezone(TZ)
                         for tt in ts])

    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    ax1.plot(t_local, broadband, lw=0.6, color="C0", label="broadband fractional excess (auto)")
    ax1.set_ylabel("fractional excess (clean band)")
    ax1.set_xlabel(f"time (UTC)")
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))

    ax2 = ax1.twinx()
    tgrid_local = np.array([tt.to_datetime(timezone.utc).astimezone(TZ) for tt in tgrid])
    ax2.plot(tgrid_local, sun_aa.alt.deg, lw=1.2, color="orange", ls="--", label="Sun altitude")
    cyg = SkyCoord(ra=RADIO_SOURCES["Cygnus A"][0], dec=RADIO_SOURCES["Cygnus A"][1])
    cyg_aa = cyg.transform_to(aa)
    ax2.plot(tgrid_local, cyg_aa.alt.deg, lw=1.2, color="green", ls="--", label="Cygnus A altitude")
    vir = SkyCoord(ra=RADIO_SOURCES["Virgo A (M87)"][0], dec=RADIO_SOURCES["Virgo A (M87)"][1])
    vir_aa = vir.transform_to(aa)
    ax2.plot(tgrid_local, vir_aa.alt.deg, lw=1.2, color="purple", ls="--", label="Virgo A altitude")
    ax2.axhline(90, color="0.5", lw=0.5)
    ax2.set_ylabel("altitude (deg)")
    ax2.set_ylim(0, 95)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_xlim(t_local[0], t_local[-1])
    fig.suptitle("BMX run2 vis_002*: broadband light curve vs predicted source altitude")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=130)
    plt.close(fig)
    rlog(f"wrote {out_plot}")

    sun_alt_at_ts = np.interp(ts, tgrid.unix, sun_aa.alt.deg)
    corr = np.corrcoef(broadband, sun_alt_at_ts)[0, 1]
    rlog(f"\nCorrelation of broadband auto excess with Sun altitude: {corr:+.3f}")
    if corr > 0.3:
        rlog("  -> broadband power tracks the Sun: solar pickup detected "
             "(dish is not blocking the daytime Sun even off-boresight).")
    else:
        rlog("  -> no strong broadband tracking of the Sun's altitude.")


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "vis_002*.h5")))
    rlog(f"Found {len(files)} vis_002* files in {DATA_DIR}")
    for fn in files:
        rlog("  ", os.path.basename(fn))

    d, nchan, ts_dataset_names = load(files)
    freq = rf_axis(nchan, FS_HZ, ZONE)
    n = d["auto0"].shape[0]
    ts = d["timestamp"]

    rlog(f"\nTimestamp datasets present in each h5 file: {len(ts_dataset_names)}"
         f" -> {ts_dataset_names}")
    rlog(f"Total spectra loaded (10 files): {n}")
    rlog(f"RF axis (recomputed, fs={FS_HZ/1e6:g} MHz zone {ZONE}): "
         f"{freq[0]:.3f}-{freq[-1]:.3f} MHz, {nchan} channels")

    t0_local = dtm.datetime.fromtimestamp(ts[0], tz=timezone.utc).astimezone(TZ)
    t1_local = dtm.datetime.fromtimestamp(ts[-1], tz=timezone.utc).astimezone(TZ)
    rlog(f"Span: {t0_local:%Y-%m-%d %H:%M:%S %Z} .. {t1_local:%H:%M:%S %Z} "
         f"({(ts[-1]-ts[0])/3600:.2f} h)")

    waterfalls_local(d, freq, os.path.join(OUT_DIR, "waterfalls_local_vis002.png"))
    total_events, gnss_rows = gnss_transits(d, freq, os.path.join(OUT_DIR, "gnss_transits_vis002.png"))
    beam_map(ts, gnss_rows, OUT_DIR)
    radio_source_check(d, freq, os.path.join(OUT_DIR, "broadband_vs_sources_vis002.png"))

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(_report_lines) + "\n")
    rlog(f"\nreport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
