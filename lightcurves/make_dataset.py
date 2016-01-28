from lightcurveutil import *

files = get_lightcurve_paths()
min_length = 800
band = 'B'

for fp in files:
    if len(fp) == 0:
        continue
    lc = open_lightcurve(fp)
    lc_c = get_lc_class(fp)
    lc_b = get_lc_band(fp)
    lc_id = get_macho_id(fp)
    lc_mag = lc['mag'].tolist()
    ot = get_total_observation_time(lc)
    if len(lc_mag) >= min_length and lc_b == band:
        if lc_c > 0:
            print ",".join(str(x) for x in lc_mag[0:min_length]) + ",{0},{1}".format(lc_id, lc_c)
        else:
            print ",".join(str(x) for x in lc_mag[0:min_length]) + ",{0}".format(lc_id)
