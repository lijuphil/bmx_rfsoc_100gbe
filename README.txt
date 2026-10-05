
Data-integrity and noise checks
--------------------------------
bmx_checks.py
    Checks on the archived HDF5 visibility files for both campaigns (Sec. 4).
    - Accumulation-counter continuity: accumulations, packets, and bytes
      delivered, with gaps and duplicates (958,918 accumulations, 0 lost).
    - Timing fit of timestamp vs. acc_count: T_acc and accumulation-interval
      jitter.
    - Stored vs. recomputed frequency axis, file by file.
    - Cauchy-Schwarz bound r = |V_EW| / sqrt(P_E P_W) <= 1 on every channel of
      every accumulation (Sec. 4.1).
    - Radiometer noise from successive differences of the autocorrelations
      (Sec. 4.2).
    - GNSS third-order intermodulation search.
    Writes report.txt, results.json and diagnostic plots.

bmx_followup.py
    Second-round analyses that build on bmx_checks.py (imports it).
    Includes: cross-correlation floor scaling and origin, clock-drift
    residual, accumulation-interval statistics, and PFB equivalent noise
    bandwidth with adjacent-channel noise correlation (rho_1) (Sec. 4.2,
    Table 6). Writes followup_report.txt, followup_results.json and plots.

bmx_astrometry.py
    Cygnus A transit prediction with astropy: transforms the ICRS position to
    topocentric alt/az and takes the time of maximum altitude. Also reports
    Sun and Moon positions in the transit windows and the minimum zenith
    distance (Sec. 6.2).


GNSS transits, Sec. 6.1
-----------------------
analyze_vis002.py
    Analysis of the 10-hour run#1 subset (vis_002*.h5). GNSS event search:
    band-averaged power per line, 15th-percentile rolling floor, MAD sigma,
    peak detection (>=50% fractional excess, >=6 sigma, >=60 s apart),
    Gaussian FWHM fits over +/-120 s, and the fringe-confirmation criterion.
    Produces the GNSS inventory (Table 8: 141 events, 140 fringe-confirmed),
    along with local waterfalls and a text report. The figure scripts below
    import it as a module.

cache_gnss_fig.py
    Runs analyze_vis002's loader and event search unchanged and saves the
    inputs for the GNSS figure to gnss_fig_cache.npz: timestamps, per-line
    band power, floors, event list, and per-event data windows.

make_fig_gnss_transits.py
    Makes Fig. 4 (fig_gnss_transits): (a) per-line band power over the floor
    with the detected events; (b) the GPS BIIF-11 (PRN 10) transit in L5 and
    L2 with the Gaussian fits; (c,d) single-channel normalized
    cross-correlation Re/Im fringes. Reads gnss_fig_cache.npz plus raw data
    around the transit.


Cygnus A, run#1, Sec. 6.2
-------------------------
cygnus_a_calibration.py
    Reads run#1 files vis_0029-vis_0033 (2026-07-26 01:57-06:57 UTC), which
    bracket the predicted transit, and caches the visibilities to
    cygnus_cache.npz. The paper figure scripts use this cache. The script
    also does a broadband light-curve fit and a flux-scale estimate, but
    the paper does not use those results.


Cygnus A, run#2, Sec. 6.2
-------------------------
cygnus_run3_delay_track.py
    The same Cygnus A analysis on run#2 (100 ms accumulation, files
    vis_0006-vis_0011, transit 2026-07-28 04:28:27 UTC). It confirms that the
    transit signature appears again in the second campaign.


Galactic HI, Sec. 6.3
---------------------
hi_plane_crossing.py
    Reads run#1 files vis_0014-vis_0018 (2026-07-25 10:57-15:57 UTC), which
    bracket the l = 166 deg Galactic-plane crossing. Computes the zenith
    beam's Galactic coordinates and the topocentric-to-LSR velocity
    correction, then caches the spectra to hi_cache.npz.

hi_stage2.py
    HI extraction from hi_cache.npz. For each spectrum it fits and subtracts
    a 2nd-order polynomial continuum from the flanking baseline windows,
    then splits the spectra into on-plane (|b|<5 deg) and off-plane
    (|b|>15 deg) sets. It median-stacks each set and takes the on-minus-off
    difference to get the peak significance (16.1 sigma), the LSR centroid
    (-12.0 km/s) and the second-moment width. Writes report_hi.txt and plots.


Final paper figures
-------------------
make_waterfall_figs.py
    Makes Fig. 3 (fig_waterfalls): East/West autocorrelation and
    cross-correlation Re/Im waterfalls over the 10-hour run#1 GNSS subset.
    Also makes Fig. 5 (fig_cygnus_fringes): cross-correlation Re, Im, phase
    and amplitude waterfalls +/-90 min around the run#1 Cygnus A transit.
    Imports analyze_vis002.py and reads cygnus_cache.npz.

make_cygnus_channel_fig.py
    Makes Fig. 6 (fig_cygnus_channel): raw Re/Im of the cross-correlation at
    1135.6 MHz through the run#1 Cygnus A transit, with a 2.5 s boxcar.
    Reads cygnus_cache.npz.

make_hi_figs.py
    Makes Fig. 7 (fig_hi_plane_crossing): on/off-plane stacked profiles and
    their difference. Also makes Fig. 8 (fig_hi_dish_consistency): the
    on-plane profile for the East and West dishes separately. Uses the same
    computation as hi_stage2.py and reads hi_cache.npz.


Run order
---------
1. bmx_checks.py, then bmx_followup.py
2. bmx_astrometry.py
3. analyze_vis002.py, then cache_gnss_fig.py, then make_fig_gnss_transits.py
4. cygnus_a_calibration.py, then make_waterfall_figs.py and
   make_cygnus_channel_fig.py
5. cygnus_run3_delay_track.py
6. hi_plane_crossing.py, then hi_stage2.py and make_hi_figs.py

Figures 1 (system overview) and 2 (packetizer timing) are diagrams. No
script in this folder makes them.

PLEASE NOTE: The data is stored locally and not on a server. It can be made available upon request.
