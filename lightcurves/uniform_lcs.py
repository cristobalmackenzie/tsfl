from lightcurveutil import *

files = get_lightcurve_paths()
min_observation_time = 2000

for fp in files:
    if len(fp) == 0:
        continue
    lc = open_lightcurve(fp)
    lc_c = get_lc_class(fp)
    lc_b = get_lc_band(fp)
    ot = get_total_observation_time(lc)
    if ot >= min_observation_time:
        lc_s, t_s = uniform_samples(lc.index.values, lc['mag'], timestep=2,
                num_samples=1000, smooth=0.000000000000025)
        print ",".join(str(x) for x in lc_s) + ",{0},{1}".format(lc_c, lc_b)
