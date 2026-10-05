#!/usr/bin/env python3
import glob
import os

import numpy as np

import analyze_vis002 as A

WIN_S = 300.0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gnss_fig_cache.npz")


def main():
    files = sorted(glob.glob(os.path.join(A.DATA_DIR, "vis_002*.h5")))
    d, nchan, _ = A.load(files)
    freq = A.rf_axis(nchan, A.FS_HZ, A.ZONE)
    ts = d["timestamp"]
    cadence = float(np.median(np.diff(ts)))
    _, rows = A.gnss_transits(d, freq, os.path.join(HERE, "_scratch_gnss.png"))
    os.remove(os.path.join(HERE, "_scratch_gnss.png"))

    out = dict(ts=ts, freq=freq)
    names, fcs, hws = [], [], []
    ev = {k: [] for k in ("line", "idx", "snr", "fwhm", "fringe")}
    win = int(WIN_S / cadence)
    for li, (name, fc, hw, bp, base, resid, events) in enumerate(rows):
        names.append(name); fcs.append(fc); hws.append(hw)
        noise = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        out[f"bp_{li}"] = bp
        out[f"base_{li}"] = base
        out[f"noise_{li}"] = noise
        sel = (freq >= fc - hw) & (freq <= fc + hw)
        for e in events:
            k = len(ev["line"])
            ev["line"].append(li)
            ev["idx"].append(e["idx"])
            ev["snr"].append(e["snr"])
            ev["fwhm"].append(e["fit"]["fwhm_s"] if e["fit"] else np.nan)
            ev["fringe"].append(e["fringe_amp"])
            lo, hi = max(0, e["idx"] - win), min(len(ts), e["idx"] + win)
            out[f"win_{k}_lohi"] = np.array([lo, hi])
            for key in ("auto0", "auto1", "cross_real", "cross_imag"):
                out[f"win_{k}_{key}"] = d[key][lo:hi][:, sel].astype(np.float32)
            out[f"win_{k}_freq"] = freq[sel]
    for k, v in ev.items():
        out[f"ev_{k}"] = np.array(v)
    out["names"] = np.array(names)
    out["fc"] = np.array(fcs)
    out["hw"] = np.array(hws)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}: {len(ev['line'])} events, "
          f"{int(np.sum(np.array(ev['fringe']) > 0.02))} fringe-confirmed")


if __name__ == "__main__":
    main()
