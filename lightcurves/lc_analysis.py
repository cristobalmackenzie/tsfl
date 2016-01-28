from lightcurveutil import *

files = get_lightcurve_paths()

for fp in files:
    if len(fp) == 0:
        continue
    lc = open_lightcurve(fp)
    lc_c = get_lc_class(fp)
    lc_b = get_lc_band(fp)
    ot = get_total_observation_time(lc)
    ts = get_avg_timestep(lc)
    print "{0} {1} {2} {3}".format(ot, ts, lc_b, lc_c)
