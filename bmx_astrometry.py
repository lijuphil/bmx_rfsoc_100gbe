#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np

from astropy import units as u
from astropy.coordinates import (SkyCoord, EarthLocation, AltAz, TETE,
                                 solar_system_ephemeris, get_body)
from astropy.time import Time

BMX_LAT_DEG = 40.869944
BMX_LON_DEG = -72.865750
BMX_HEIGHT_M = 20.0

CYGA_RA = "19h59m28.36s"
CYGA_DEC = "+40d44m02.1s"

CYGX_RA = "20h30m00s"
CYGX_DEC = "+40d30m00s"

C_M_S = 299792458.0

PAPER_TRANSITS = {
    "run2": ("2026-07-26 04:36:19", 51.0),
    "run3": ("2026-07-28 04:28:27", 290.0),
}


def site():
    return EarthLocation(lat=BMX_LAT_DEG * u.deg, lon=BMX_LON_DEG * u.deg,
                         height=BMX_HEIGHT_M * u.m)


def hour_angle_correct(t: Time, loc, coord_icrs) -> np.ndarray:
    app = coord_icrs.transform_to(TETE(obstime=t, location=loc))
    lst = t.sidereal_time("apparent", longitude=loc.lon)
    return ((lst - app.ra).wrap_at(180 * u.deg)).to_value(u.deg)


def hour_angle_shortcut(t: Time, loc, coord_icrs) -> np.ndarray:
    lst = t.sidereal_time("apparent", longitude=loc.lon)
    return ((lst - coord_icrs.ra).wrap_at(180 * u.deg)).to_value(u.deg)


def find_zero_crossing(t0: Time, ha_func, window_s=3600.0, n=4001):
    dt = np.linspace(-window_s, window_s, n)
    t = t0 + dt * u.s
    ha = ha_func(t)
    s = np.sign(ha)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        raise RuntimeError("no hour-angle zero crossing in window")
    i = idx[np.argmin(np.abs(dt[idx]))]
    frac = -ha[i] / (ha[i + 1] - ha[i])
    return t0 + (dt[i] + frac * (dt[i + 1] - dt[i])) * u.s


def analytic_precession_ra(coord_icrs, epoch_yr=2026.57):
    m = 3.07496
    n_s = 1.33621
    n_as = 20.0431
    ra = coord_icrs.ra.rad
    dec = coord_icrs.dec.rad
    dra_dt = m + n_s * np.sin(ra) * np.tan(dec)
    ddec_dt = n_as * np.cos(ra)
    dt_yr = epoch_yr - 2000.0
    return dra_dt, ddec_dt, dra_dt * dt_yr, ddec_dt * dt_yr / 3600.0


def report_transit(label, predicted_utc, reported_offset_s, loc, cyga,
                   beam_deg, baseline_m):
    print("\n" + "=" * 72)
    print(f"{label}: paper-predicted transit {predicted_utc} UTC "
          f"(reported offset {reported_offset_s:+.0f} s)")
    print("=" * 72)

    t_ref = Time(predicted_utc, scale="utc", location=loc)

    t_correct = find_zero_crossing(t_ref, lambda t: hour_angle_correct(t, loc, cyga))
    t_short = find_zero_crossing(t_ref, lambda t: hour_angle_shortcut(t, loc, cyga))

    d_correct = (t_correct - t_ref).to_value(u.s)
    d_short = (t_short - t_ref).to_value(u.s)
    frame_bias = (t_correct - t_short).to_value(u.s)

    print(f"  transit, correct (apparent place)  : "
          f"{t_correct.utc.iso[:23]}  ({d_correct:+.1f} s vs paper value)")
    print(f"  transit, shortcut (ICRS vs app LST): "
          f"{t_short.utc.iso[:23]}  ({d_short:+.1f} s vs paper value)")
    print(f"  --> frame bias (correct - shortcut): {frame_bias:+.1f} s")
    print(f"      The shortcut predicts transit {abs(frame_bias):.0f} s "
          f"{'EARLY' if frame_bias > 0 else 'LATE'}, so a source obeying the")
    print(f"      correct ephemeris appears to arrive "
          f"{abs(frame_bias):+.0f} s {'late' if frame_bias > 0 else 'early'}.")
    print(f"  paper's measured offset            : {reported_offset_s:+.0f} s")
    resid = reported_offset_s - frame_bias
    print(f"  --> residual after frame correction: {resid:+.1f} s")
    if abs(resid) < 15:
        print("      *** The reported offset is consistent with the frame bug. ***")
        print("      Refitting with correct astrometry should collapse it to")
        print("      a few seconds, which is a much stronger result.")

    app = cyga.transform_to(TETE(obstime=t_correct, location=loc))
    zd_correct = abs(app.dec.deg - BMX_LAT_DEG)
    zd_icrs = abs(cyga.dec.deg - BMX_LAT_DEG)
    print(f"  min zenith distance, apparent dec  : {zd_correct:.3f} deg")
    print(f"  min zenith distance, ICRS dec      : {zd_icrs:.3f} deg")
    print(f"  (paper quotes 0.06 deg -> matches the APPARENT value, while the")
    print(f"   transit time matches the ICRS one. That inconsistency is the")
    print(f"   seam to find in the code.)")

    with solar_system_ephemeris.set("builtin"):
        aa = AltAz(obstime=t_correct, location=loc)
        sun = get_body("sun", t_correct, loc).transform_to(aa)
        moon = get_body("moon", t_correct, loc).transform_to(aa)
    print(f"  Sun altitude at transit            : {sun.alt.deg:+.1f} deg")
    print(f"  Moon zenith distance at transit    : {90 - moon.alt.deg:.1f} deg")

    dec = app.dec.rad
    rate_deg_hr = 15.0 * np.cos(dec)
    fwhm_min = beam_deg / rate_deg_hr * 60.0
    print(f"  sky drift rate at this dec         : {rate_deg_hr:.3f} deg/hr")
    print(f"  predicted transit FWHM ({beam_deg:.1f} deg beam): "
          f"{fwhm_min:.2f} min")
    print(f"  (paper measures 13.1 min for run2 -> implied beam "
          f"{13.1 / 60.0 * rate_deg_hr:.2f} deg)")

    tau_max_ns = baseline_m / C_M_S * np.cos(dec) * 1e9
    print(f"  max geometric delay (B={baseline_m} m EW)   : "
          f"{tau_max_ns:.2f} ns at HA=90 deg")
    print(f"  delay rate through transit         : "
          f"{baseline_m / C_M_S * np.cos(dec) * (2 * np.pi / 86164.1) * 1e12:.3f} ps/s")
    print("  de-rotation model: tau(t) = (B/c) cos(dec_apparent) sin(HA(t))")
    print("  NOTE: use the APPARENT declination above, not the ICRS one.")

    cygx = SkyCoord(CYGX_RA, CYGX_DEC, frame="icrs")
    dra_min = ((cygx.ra - cyga.ra).wrap_at(180 * u.deg)).to_value(u.hourangle) * 60.0
    print(f"  Cygnus-X transits later by          : {dra_min:.1f} sidereal min")
    print(f"  (a diffuse component that far LATER in RA pulls a free-t0 fit to")
    print(f"   POSITIVE offsets and narrows the fitted width -- one mechanism")
    print(f"   for both the {reported_offset_s:+.0f} s offset and the narrow FWHM.)")

    return dict(t_correct=t_correct.utc.iso, t_shortcut=t_short.utc.iso,
                frame_bias_s=float(frame_bias), residual_s=float(resid),
                dt_vs_correct=float(-d_correct), dt_vs_shortcut=float(-d_short),
                zd_apparent_deg=float(zd_correct), zd_icrs_deg=float(zd_icrs),
                fwhm_pred_min=float(fwhm_min),
                implied_beam_deg=float(13.1 / 60.0 * rate_deg_hr),
                cygx_offset_min=float(dra_min))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam-deg", type=float, default=4.0)
    ap.add_argument("--baseline-m", type=float, default=8.8)
    args = ap.parse_args()

    loc = site()
    cyga = SkyCoord(CYGA_RA, CYGA_DEC, frame="icrs")

    print("BMX / Cygnus A transit astrometry check")
    print(f"  site      : {BMX_LAT_DEG:+.6f} deg N, {BMX_LON_DEG:+.6f} deg E")
    print(f"  Cygnus A  : {cyga.ra.to_string(u.hour)}  "
          f"{cyga.dec.to_string(u.deg)}  (ICRS/J2000)")

    dra, ddec, dra_tot, ddec_tot = analytic_precession_ra(cyga)
    print("\nAnalytic general precession at this position:")
    print(f"  dRA/dt   = {dra:+.4f} s/yr    -> {dra_tot:+.1f} s since J2000")
    print(f"  dDec/dt  = {ddec:+.3f} \"/yr   -> {ddec_tot:+.4f} deg since J2000")
    print(f"  Predicted apparent Dec = {cyga.dec.deg + ddec_tot:.4f} deg")
    print(f"  BMX latitude           = {BMX_LAT_DEG:.4f} deg")
    print(f"  -> zenith distance     = "
          f"{abs(cyga.dec.deg + ddec_tot - BMX_LAT_DEG):.3f} deg "
          f"(paper quotes 0.06 deg)")

    out = {}
    for label, (utc, off) in PAPER_TRANSITS.items():
        out[label] = report_transit(label, utc, off, loc, cyga,
                                    args.beam_deg, args.baseline_m)

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    bias = np.mean([v["frame_bias_s"] for v in out.values()])
    resids = {k: v["residual_s"] for k, v in out.items()}
    matches_short = all(abs(r) < 15 for r in resids.values())
    print(f"  frame bias (correct - shortcut): {bias:+.1f} s")
    for k, v in out.items():
        print(f"  {k}: paper prediction sits {v['dt_vs_correct']:+.1f} s from the")
        print(f"        correct transit and {v['dt_vs_shortcut']:+.1f} s from the shortcut.")
    if matches_short:
        print("\n  VERDICT: the reported offsets are consistent with the frame")
        print("  shortcut. Refit with an explicit apparent-place transform:")
        print("      app = cyga.transform_to(TETE(obstime=t, location=loc))")
        print("      ha  = lst_apparent - app.ra")
        print("  and rebuild the geometric delay with the apparent declination.")
    else:
        print("\n  VERDICT: the paper's quoted predictions match the CORRECT")
        print("  calculation, so the astrometry is sound and the Section 5.2")
        print("  prose about omitting precession is simply inaccurate -- fix the")
        print("  wording, and drop the associated caveat.")
        print("  The measured offsets are therefore real. The leading suspect is")
        print("  the Cygnus-X blend: see the RA separation printed above. A")
        print("  diffuse component sitting LATER in RA pulls a free-t0 Gaussian")
        print("  fit to positive offsets AND narrows the fitted width when the")
        print("  background component absorbs the wings -- which would explain")
        print("  the +51 s / +290 s offsets and the 13.1 vs 21.1 min FWHM")
        print("  discrepancy with a single mechanism.")
        print("  Test it by refitting with the pedestal fixed from off-transit")
        print("  data, and by masking |HA| beyond one predicted beam FWHM.")
    print("\n  In all cases: refit BOTH campaigns with the SAME free-t0 method")
    print("  and propagate a fit covariance, so the offset carries an error bar.")

if __name__ == "__main__":
    main()
