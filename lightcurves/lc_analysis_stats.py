import pandas as pd
import numpy as np

df = pd.read_csv('lc_analysis_out.csv', names=['ot','ts','band','c'])

"""
Estadisticas para los datos completos
"""
ot = df['ot']
ts = df['ts']
ot_mean, ot_max, ot_min = np.mean(ot), np.max(ot), np.min(ot)
ts_mean, ts_max, ts_min = np.mean(ts), np.max(ts), np.min(ts)

print "total {0} {1} {2} {3} {4} {5}".format(ot_mean, ot_max, ot_min,
        ts_mean, ts_max, ts_min)
"""
Estadisticas para los datos por banda
"""
df_b = df[df.band == 'B']
ot = df_b['ot']
ts = df_b['ts']
ot_mean, ot_max, ot_min = np.mean(ot), np.max(ot), np.min(ot)
ts_mean, ts_max, ts_min = np.mean(ts), np.max(ts), np.min(ts)

print "band_B {0} {1} {2} {3} {4} {5}".format(ot_mean, ot_max, ot_min,
        ts_mean, ts_max, ts_min)

df_b = df[df.band == 'R']
ot = df_b['ot']
ts = df_b['ts']
ot_mean, ot_max, ot_min = np.mean(ot), np.max(ot), np.min(ot)
ts_mean, ts_max, ts_min = np.mean(ts), np.max(ts), np.min(ts)

print "band_R {0} {1} {2} {3} {4} {5}".format(ot_mean, ot_max, ot_min,
        ts_mean, ts_max, ts_min)
"""
Estadisticas para los datos por clase
"""
for i in range(1,9):
    df_b = df[df.c == i]
    ot = df_b['ot']
    ts = df_b['ts']
    ot_mean, ot_max, ot_min = np.mean(ot), np.max(ot), np.min(ot)
    ts_mean, ts_max, ts_min = np.mean(ts), np.max(ts), np.min(ts)

    print "class_{6} {0} {1} {2} {3} {4} {5}".format(ot_mean, ot_max, ot_min,
        ts_mean, ts_max, ts_min, i)

"""
Estadisticas para los datos por clase, banda
"""
