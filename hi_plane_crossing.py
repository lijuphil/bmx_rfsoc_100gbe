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

BMX_LAT = 40.869944
BMX_LON = -72.865750
BMX_ELEV = 20.0

HI_REST_MHZ = 1420.405751786
C_KMS = 299792.458

TZ = timezone.utc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "bmx_run2_500ms_1050mhz")
OUT_DIR = HERE
REPORT_PATH = os.path.join(OUT_DIR, "report_hi.txt")

FILES = ["vis_0014", "vis_0015", "vis_0016", "vis_0017", "vis_0018"]

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


def galactic_coords(ts):
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz
    import astropy.units as u

    loc = EarthLocation(lat=BMX_LAT * u.deg, lon=BMX_LON * u.deg, height=BMX_ELEV * u.m)
    t = Time(ts, format="unix")
    aa = AltAz(obstime=t, location=loc, alt=90 * u.deg, az=0 * u.deg)
    gal = SkyCoord(aa).icrs.galactic
    return gal.l.deg, gal.b.deg


def compute_v_lsr(ts, loc_kwargs):
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz
    import astropy.units as u

    loc = EarthLocation(**loc_kwargs)
    mid = Time(ts[len(ts) // 2], format="unix")
    zen_full = SkyCoord(AltAz(alt=90 * u.deg, az=0 * u.deg, obstime=mid,
                               location=loc)).icrs
    zen = SkyCoord(ra=zen_full.ra, dec=zen_full.dec, frame="icrs")
    vcorr = zen.radial_velocity_correction(
        kind="barycentric", obstime=mid, location=loc).to(u.km / u.s).value
    lsr_apex = SkyCoord(ra=18 * u.hourangle, dec=30 * u.deg)
    ang = zen.separation(lsr_apex).rad
    v_sun_lsr = 20.0 * np.cos(ang)
    return vcorr + v_sun_lsr


def db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.asarray(x, dtype=float)
        return 10 * np.log10(np.where(x > 0, x, np.nan))


def main():
    files = [os.path.join(DATA_DIR, f) for stem in FILES
              for f in glob.glob(os.path.join(DATA_DIR, stem + "_*.h5"))]
    files = sorted(set(files))
    rlog(f"Files: {[os.path.basename(f) for f in files]}")

    d, nchan = load(files)
    freq = rf_axis(nchan, FS_HZ, ZONE)
    ts = d["timestamp"]
    n = len(ts)
    chan_hz = FS_HZ / NFFT
    v_res = C_KMS * chan_hz / (HI_REST_MHZ * 1e6)

    t0_local = dtm.datetime.fromtimestamp(ts[0], tz=timezone.utc)
    t1_local = dtm.datetime.fromtimestamp(ts[-1], tz=timezone.utc)
    rlog(f"\n{n} spectra, {t0_local:%Y-%m-%d %H:%M} - {t1_local:%H:%M %Z} "
         f"({(ts[-1]-ts[0])/3600:.2f} h)")
    rlog(f"RF axis: {freq[0]:.3f}-{freq[-1]:.3f} MHz, {nchan} ch, "
         f"{chan_hz/1e3:.3f} kHz/ch -> {v_res:.2f} km/s/channel velocity resolution")

    l_deg, b_deg = galactic_coords(ts)
    rlog(f"Galactic l: {l_deg.min():.1f}-{l_deg.max():.1f} deg, "
         f"b: {b_deg.min():+.2f} to {b_deg.max():+.2f} deg over this file range")
    i_plane = int(np.argmin(np.abs(b_deg)))
    rlog(f"Closest approach to plane: b={b_deg[i_plane]:+.3f} deg at "
         f"{dtm.datetime.fromtimestamp(ts[i_plane], tz=timezone.utc):%H:%M:%S} UTC "
         f"(l={l_deg[i_plane]:.1f} deg)")

    v_lsr = compute_v_lsr(ts, dict(lat=BMX_LAT, lon=BMX_LON, height=BMX_ELEV))
    rlog(f"v_LSR correction (topo->LSR, toward zenith, mid-run): {v_lsr:+.2f} km/s")

    line = (freq > 1417.0) & (freq < 1424.5)
    base = ((freq > 1409) & (freq < 1417)) | ((freq > 1426) & (freq < 1436))
    ff = freq[line]
    fb = freq[base]
    v_line_topo = (HI_REST_MHZ - ff) / HI_REST_MHZ * C_KMS
    rlog(f"\nLine-search window: {ff[0]:.3f}-{ff[-1]:.3f} MHz -> topocentric "
         f"velocity range {v_line_topo.max():+.0f} to {v_line_topo.min():+.0f} km/s "
         f"({line.sum()} channels)")
    rlog(f"(LSR frame: add {v_lsr:+.1f} km/s -> "
         f"{v_line_topo.max()+v_lsr:+.0f} to {v_line_topo.min()+v_lsr:+.0f} km/s)")
    rlog(f"Baseline/continuum-fit window: {fb.min():.1f}-{fb.max():.1f} MHz "
         f"({base.sum()} channels, excluded from the line itself)")

    np.savez(os.path.join(OUT_DIR, "hi_cache.npz"),
              ts=ts, freq=freq, l_deg=l_deg, b_deg=b_deg,
              auto0=d["auto0"], auto1=d["auto1"])
    rlog("\n(cached auto0/auto1/freq/ts/l/b to hi_cache.npz for the next stage)")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(_report_lines) + "\n")


if __name__ == "__main__":
    main()
