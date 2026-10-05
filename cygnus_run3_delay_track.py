#!/usr/bin/env python3
import glob
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timezone
import datetime as dtm

NFFT = 4096
FS_HZ = 1050e6
ZONE = 3
TZ = timezone.utc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "bmx_run3_100ms_18hrs")
OUT_DIR = HERE
REPORT_PATH = os.path.join(OUT_DIR, "report_cygnus_run3.txt")

BMX_LAT, BMX_LON, BMX_ELEV = 40.869944, -72.865750, 20.0
C_LIGHT = 299792458.0
B_EW = 8.8
BEAM_FWHM_DEG = 4.0
T_SYS_K = 70.0
D_EFF_M = 3.95
CYG_RA = "19h59m28.36s"
CYG_DEC = "+40d44m02.1s"

GNSS_LINES = [
    (1176.45, 12.0), (1202.03, 5.0), (1207.14, 6.0), (1227.60, 12.0),
    (1268.52, 10.0), (1278.75, 10.0), (1575.42, 12.0), (1542.0, 8.0),
]

FILES = ["vis_0006", "vis_0007", "vis_0008", "vis_0009", "vis_0010", "vis_0011"]

_report_lines = []


def rlog(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _report_lines.append(s)


def rf_axis(nchan, fs_hz, zone, nfft=NFFT):
    fbb = np.arange(nchan) * (fs_hz / nfft) / 1e6
    half = (fs_hz / 2) / 1e6
    z = int(zone)
    return (z - 1) * half + fbb if z % 2 else z * half - fbb


def clean_mask(freq):
    m = (freq > 1120.0) & (freq < 1520.0)
    for fc, hw in GNSS_LINES:
        m &= ~((freq >= fc - hw - 3) & (freq <= fc + hw + 3))
    m &= ~((freq > 1415) & (freq < 1426))
    return m


def cyg_a_flux_jy(freq_mhz):
    x = np.log10(np.asarray(freq_mhz, dtype=float) / 1000.0)
    logS = 3.3498 - 1.0022 * x - 0.2246 * x**2 + 0.0227 * x**3 + 0.0425 * x**4
    return 10 ** logS


def load(files):
    keys = ("auto0", "auto1", "cross_real", "cross_imag")
    data = {k: [] for k in keys}
    ts = []
    nchan = None
    for fn in files:
        with h5py.File(fn, "r") as f:
            for k in keys:
                data[k].append(f[k][:])
            ts.append(f["timestamp"][:])
            if nchan is None:
                nchan = f["auto0"].shape[1]
    out = {k: np.concatenate(v, axis=0) for k, v in data.items()}
    out["timestamp"] = np.concatenate(ts)
    order = np.argsort(out["timestamp"], kind="stable")
    for k in list(out):
        out[k] = out[k][order]
    return out, nchan


def gaussian(t, A, t0, sig, C):
    return C + A * np.exp(-0.5 * ((t - t0) / sig) ** 2)


def main():
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord
    from scipy.optimize import least_squares
    import astropy.units as u

    files = []
    for stem in FILES:
        files += glob.glob(os.path.join(DATA_DIR, stem + "_*.h5"))
    files = sorted(set(files))
    rlog(f"Files: {[os.path.basename(f) for f in files]}")

    d, nchan = load(files)
    freq = rf_axis(nchan, FS_HZ, ZONE)
    mask = clean_mask(freq)
    ts = d["timestamp"]
    rlog(f"{len(ts)} spectra, {ts[0]:.0f}..{ts[-1]:.0f} unix "
         f"({(ts[-1]-ts[0])/3600:.2f} h), {mask.sum()} clean channels")

    loc = EarthLocation(lat=BMX_LAT * u.deg, lon=BMX_LON * u.deg, height=BMX_ELEV * u.m)
    t_ap = Time(ts, format="unix", location=loc)
    lst = t_ap.sidereal_time("apparent").hour
    cyg = SkyCoord(ra=CYG_RA, dec=CYG_DEC)
    H_deg = (lst - cyg.ra.hour) * 15.0
    tau_pred = (B_EW / C_LIGHT) * np.cos(cyg.dec.radian) * np.sin(np.radians(H_deg))

    i_pred = int(np.argmin(np.abs(tau_pred)))
    t_pred = float(ts[i_pred])
    t_pred_local = dtm.datetime.fromtimestamp(t_pred, tz=timezone.utc)
    rlog(f"Predicted transit (zero geometric delay): {t_pred_local}, "
         f"tau={tau_pred[i_pred]*1e9:.3f} ns")

    freq_hz = freq[mask] * 1e6
    V = (d["cross_real"][:, mask] + 1j * d["cross_imag"][:, mask]).astype(np.complex128)
    phase_pred = 2.0 * np.pi * freq_hz[None, :] * tau_pred[:, None]

    best_sign, best_amp = None, -1
    for sign in (+1, -1):
        Vt = V * np.exp(sign * 1j * phase_pred)
        avg = Vt.mean(axis=1)
        near = np.abs(ts - t_pred) < 30 * 60
        score = np.abs(avg[near]).max() if near.any() else np.abs(avg).max()
        rlog(f"  sign={sign:+d}: peak |V_tracked| near transit = {score:.4g}")
        if score > best_amp:
            best_amp, best_sign = score, sign
    rlog(f"Using sign={best_sign:+d}")

    Vt = V * np.exp(best_sign * 1j * phase_pred)
    v_tracked = Vt.mean(axis=1)
    v_untracked = V.mean(axis=1)
    amp_tracked = np.abs(v_tracked)

    from scipy.ndimage import median_filter
    amp_tracked = median_filter(amp_tracked, size=5)

    excl = np.abs(ts - t_pred) < 75 * 60.0
    tb, ab = ts[~excl], amp_tracked[~excl]
    tc = ts.mean()
    coeffs = np.polyfit(tb - tc, ab, 5)
    baseline_full = np.polyval(coeffs, ts - tc)
    resid_amp = amp_tracked - baseline_full
    baseline_at_pred = np.polyval(coeffs, t_pred - tc)

    beam_fwhm_time_min = BEAM_FWHM_DEG / 11.37 * 60
    sig_fix = beam_fwhm_time_min * 60 / 2.3548

    A0 = resid_amp[np.abs(ts - t_pred) < 15 * 60].max() if (np.abs(ts - t_pred) < 15 * 60).any() else resid_amp.max()
    p0 = np.array([max(A0, 1e-8), t_pred, sig_fix, 0.0])
    lo = [0, t_pred - 1800, 300, -np.inf]
    hi = [np.inf, t_pred + 1800, 3600, np.inf]
    tt, rr = ts, resid_amp
    for _ in range(3):
        scale = 1.4826 * np.median(np.abs(rr - np.median(rr))) or 1e-30
        res = least_squares(lambda p: (gaussian(tt, *p) - rr) / scale,
                             p0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, max_nfev=20000)
        p0 = res.x
        model = gaussian(ts, *p0)
        s2 = 1.4826 * np.median(np.abs((resid_amp - model) - np.median(resid_amp - model))) or 1e-30
        keep = np.abs((resid_amp - model) / s2) < 6
        tt, rr = ts[keep], resid_amp[keep]
    rlog(f"\n-- single free Gaussian (diagnostic) -- "
         f"t0 offset {p0[1]-t_pred:+.0f}s, FWHM {2.3548*p0[2]/60:.1f} min "
         f"(predicted {beam_fwhm_time_min:.1f} min): still diffuse-pedestal-shaped, "
         f"as expected from the run2 experience with an 8.8m baseline")

    def model2(t, A1, A2, t2, sig2, C):
        return (C + A1 * np.exp(-0.5 * ((t - t_pred) / sig_fix) ** 2)
                  + A2 * np.exp(-0.5 * ((t - t2) / sig2) ** 2))

    p0b = np.array([A0, A0, t_pred + 20 * 60, 40 * 60, 0.0])
    lob = [0, 0, t_pred - 3600, 600, -np.inf]
    hib = [np.inf, np.inf, t_pred + 3 * 3600, 10800, np.inf]
    tt, rr = ts, resid_amp
    for _ in range(3):
        scale = 1.4826 * np.median(np.abs(rr - np.median(rr))) or 1e-30
        res = least_squares(lambda p: (model2(tt, *p) - rr) / scale,
                             p0b, bounds=(lob, hib), loss="soft_l1", f_scale=3.0, max_nfev=30000)
        p0b = res.x
        m = model2(ts, *p0b)
        s2 = 1.4826 * np.median(np.abs((resid_amp - m) - np.median(resid_amp - m))) or 1e-30
        keep = np.abs((resid_amp - m) / s2) < 6
        tt, rr = ts[keep], resid_amp[keep]

    A, A2, t2, sig2, C0 = p0b
    t0 = t_pred
    fwhm_min = beam_fwhm_time_min
    t0_local = t_pred_local
    rlog(f"\n-- two-component fit (Cygnus A shape FIXED to prediction) --")
    rlog(f"  Cygnus A component: t0 FIXED={t_pred_local:%H:%M:%S}, "
         f"FWHM FIXED={beam_fwhm_time_min:.1f} min, amplitude A1={A:.4g}")
    rlog(f"  diffuse component (free): peaks at "
         f"{dtm.datetime.fromtimestamp(t2, tz=timezone.utc):%H:%M:%S}, "
         f"FWHM={2.3548*sig2/60:.1f} min, amplitude A2={A2:.4g}")

    p0_off = np.median(d["auto0"][:, mask], axis=1)
    p1_off = np.median(d["auto1"][:, mask], axis=1)
    near = np.abs(ts - t_pred) < 3600
    P0_off = np.median(p0_off[near])
    P1_off = np.median(p1_off[near])
    deflection = A / np.sqrt(P0_off * P1_off)
    S_mean = cyg_a_flux_jy(freq[mask]).mean()
    SEFD_dualpol = S_mean / deflection
    SEFD_singlepol = (S_mean / 2) / deflection

    rlog(f"  off-source autocorrelation levels near transit: P0={P0_off:.4g}, P1={P1_off:.4g}")
    rlog(f"  implied deflection d = {deflection:.5f} ({deflection*100:.3f}%)")
    rlog(f"  SEFD (dual-pol flux {S_mean:.0f} Jy): {SEFD_dualpol:.0f} Jy")
    rlog(f"  SEFD (single-pol, S/2): {SEFD_singlepol:.0f} Jy")

    k_B = 1.380649e-23
    A_eff = np.pi * (D_EFF_M / 2) ** 2
    SEFD_theory = 2 * k_B * T_SYS_K / A_eff / 1e-26
    rlog(f"  theoretical single-pol SEFD (T_sys={T_SYS_K}K, D_eff={D_EFF_M}m): {SEFD_theory:.0f} Jy")
    rlog(f"  ratio measured/theoretical: {SEFD_singlepol/SEFD_theory:.2f}")

    far = np.abs(ts - t_pred) > 2 * 3600
    noise_far = 1.4826 * np.median(np.abs(resid_amp[far] - np.median(resid_amp[far])))
    near15 = np.abs(ts - t_pred) < 15 * 60
    peak_val = resid_amp[near15].max() if near15.any() else np.nan
    rlog(f"\n  noise (robust, far from transit): {noise_far:.4g}")
    rlog(f"  peak residual (+/-15 min): {peak_val:.4g}")
    rlog(f"  naive SNR (peak/robust noise): {peak_val/noise_far:.2f}")

    cadence_s = float(np.median(np.diff(ts)))
    win_s = fwhm_min * 60
    n_avg = max(1, int(win_s / cadence_s))
    smoothed = np.convolve(resid_amp, np.ones(n_avg) / n_avg, mode="same")
    noise_smoothed_far = 1.4826 * np.median(np.abs(smoothed[far] - np.median(smoothed[far])))
    peak_smoothed = smoothed[near15].max() if near15.any() else np.nan
    rlog(f"  SNR after {win_s:.0f}s smoothing: {peak_smoothed/noise_smoothed_far:.2f}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    zoom = np.abs(ts - t_pred) < 3 * 3600
    t_local = np.array([dtm.datetime.fromtimestamp(tt, tz=timezone.utc)
                         for tt in ts[zoom]])
    cyg_only = C0 + A * np.exp(-0.5 * ((ts[zoom] - t_pred) / sig_fix) ** 2)
    ax.plot(t_local, resid_amp[zoom], lw=0.5, color="0.5", label="delay-tracked, background-subtracted")
    ax.plot(t_local, model2(ts[zoom], *p0b), color="C3", lw=1.8, label="Cygnus A + diffuse fit")
    ax.plot(t_local, cyg_only, color="C0", lw=1.4, ls=":", label="Cygnus A component only (shape fixed)")
    ax.axvline(t_pred_local, color="k", ls=":", lw=1, label="predicted transit")
    ax.set_xlabel(f"time (UTC)")
    ax.set_ylabel("|V_tracked| residual (arb)")
    ax.set_title(f"Cygnus A (run3, 100ms): SEFD (single-pol) = {SEFD_singlepol:.0f} Jy, "
                 f"SNR(smoothed)={peak_smoothed/noise_smoothed_far:.1f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "cygnus_run3_delay_tracked_residual.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    rlog(f"wrote {out}")

    np.savez(os.path.join(OUT_DIR, "cygnus_run3_result.npz"),
              ts=ts, resid_amp=resid_amp, popt=p0b, t_pred=t_pred,
              SEFD_singlepol=SEFD_singlepol, deflection=deflection,
              noise_far=noise_far, cadence_s=cadence_s)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(_report_lines) + "\n")


if __name__ == "__main__":
    main()
