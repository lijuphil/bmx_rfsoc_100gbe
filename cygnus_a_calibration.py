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
BMX_LAT, BMX_LON, BMX_ELEV = 40.869944, -72.865750, 20.0
TZ = timezone.utc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "bmx_run2_500ms_1050mhz")
OUT_DIR = HERE
REPORT_PATH = os.path.join(OUT_DIR, "report_cygnus.txt")

FILES = ["vis_0029", "vis_0030", "vis_0031", "vis_0032", "vis_0033"]

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
HI_MHZ = (1417.0, 1424.5)

PREDICTED_T0_UNIX = None

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


def load(files):
    keys = ("auto0", "auto1", "cross_real", "cross_imag")
    data = {k: [] for k in keys}
    acc, ts = [], []
    nchan = None
    for fn in files:
        with h5py.File(fn, "r") as f:
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
    return out, nchan


def cyg_a_flux_jy(freq_mhz):
    x = np.log10(np.asarray(freq_mhz, dtype=float) / 1000.0)
    logS = 3.3498 - 1.0022 * x - 0.2246 * x**2 + 0.0227 * x**3 + 0.0425 * x**4
    return 10 ** logS


def clean_mask(freq):
    m = (freq > 1120.0) & (freq < 1520.0)
    for _, fc, hw in GNSS_LINES:
        m &= ~((freq >= fc - hw - 3) & (freq <= fc + hw + 3))
    m &= ~((freq > HI_MHZ[0] - 2) & (freq < HI_MHZ[1] + 2))
    return m


def gaussian_plus_quad(t, A, t0, sig, c0, c1, c2):
    tt = t - t.mean()
    baseline = c0 + c1 * tt + c2 * tt ** 2
    return baseline + A * np.exp(-0.5 * ((t - t0) / sig) ** 2)


def main():
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz
    import astropy.units as u
    from scipy.optimize import curve_fit

    files = []
    for stem in FILES:
        files += glob.glob(os.path.join(DATA_DIR, stem + "_*.h5"))
    files = sorted(set(files))
    rlog(f"Files: {[os.path.basename(f) for f in files]}")

    d, nchan = load(files)
    freq = rf_axis(nchan, FS_HZ, ZONE)
    ts = d["timestamp"]
    n = len(ts)
    t0l = dtm.datetime.fromtimestamp(ts[0], tz=timezone.utc)
    t1l = dtm.datetime.fromtimestamp(ts[-1], tz=timezone.utc)
    rlog(f"{n} spectra, {t0l:%Y-%m-%d %H:%M} - {t1l:%H:%M %Z} ({(ts[-1]-ts[0])/3600:.2f} h)")
    rlog(f"RF axis: {freq[0]:.3f}-{freq[-1]:.3f} MHz, {nchan} ch")

    loc = EarthLocation(lat=BMX_LAT * u.deg, lon=BMX_LON * u.deg, height=BMX_ELEV * u.m)
    tgrid = Time(ts[0], format="unix") + (Time(ts[-1], format="unix") - Time(ts[0], format="unix")) * np.linspace(0, 1, 20000)
    cyg = SkyCoord(ra="19h59m28.36s", dec="+40d44m02.1s")
    aa = cyg.transform_to(AltAz(obstime=tgrid, location=loc))
    i_pred = int(np.argmax(aa.alt.deg))
    t_pred_unix = tgrid[i_pred].unix
    zd_pred = 90 - aa.alt.deg[i_pred]
    rlog(f"\nPredicted Cygnus A transit: {tgrid[i_pred].to_datetime(timezone.utc):%Y-%m-%d %H:%M:%S %Z} "
         f"(zenith dist {zd_pred:.3f} deg)")

    mask = clean_mask(freq)
    rlog(f"Clean broadband channels: {mask.sum()}/{nchan} "
         f"({freq[mask].min():.1f}-{freq[mask].max():.1f} MHz, GNSS lines/HI/edges excluded)")
    S_band = cyg_a_flux_jy(freq[mask])
    rlog(f"Perley & Butler (2017) Cygnus A flux across clean band: "
         f"{S_band.min():.0f}-{S_band.max():.0f} Jy (mean {S_band.mean():.0f} Jy)")

    np.savez(os.path.join(OUT_DIR, "cygnus_cache.npz"),
              ts=ts, freq=freq, mask=mask, auto0=d["auto0"], auto1=d["auto1"],
              cross_real=d["cross_real"], cross_imag=d["cross_imag"],
              t_pred_unix=t_pred_unix, zd_pred=zd_pred)
    rlog("\n(cached to cygnus_cache.npz for stage 2)")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(_report_lines) + "\n")


if __name__ == "__main__":
    main()
