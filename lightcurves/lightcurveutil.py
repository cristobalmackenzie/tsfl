from preprocess import eliminate_lc_noise
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

MACHO_LC_FILE_PATH_B = "/Users/cristobal/Documents/msc-data/kmeans-ufl/util/lightcurve_files_B.txt"
MACHO_LC_FILE_PATH_R = "/Users/cristobal/Documents/msc-data/kmeans-ufl/util/lightcurve_files_R.txt"
MACHO_LC_VAR_FILE_PATH_B = "/Users/cristobal/Documents/msc-data/kmeans-ufl/util/lightcurve_files_B_variable.txt"
MACHO_LC_VAR_FILE_PATH_R = "/Users/cristobal/Documents/msc-data/kmeans-ufl/util/lightcurve_files_R_variable.txt"
MACHO_VAR_LC_FILE_PATH_B = "/Users/cristobal/Documents/msc-data/kmeans-ufl/util/var_lcs_B.txt"

OGLE_LCS_PATH = "/Users/cristobal/Documents/msc-data/kmeans-ufl/util/ogle_lightcurve_paths.txt"
OGLE_TRAINING_LCS_PATH = "/Users/cristobal/Documents/msc-data/kmeans-ufl/util/ogle_training_lightcurve_paths.txt"

EROS_TRAINING_LCS_PATH = "/Users/cristobal/Documents/msc-data/kmeans-ufl/util/eros_training_paths.txt"


def get_dataset_classes(dataset="macho"):
    if dataset == "macho":
        classes = (["Be_lc", "CEPH", "EB", "longperiod_lc", "microlensing_lc",
                    "quasar_lc", "RRL"])
    elif dataset == "ogle":
        classes = ["-LPV-", "-CEP-", "-T2CEP-", "-RRLYR-", "-ECL-", "-DSCT-"]
    elif dataset == "eros":
        classes = ["/0/", "/1/", "/2/", "/3/", "/4/", "/5/", "/6/", "/7/"]
    return classes


def get_avg_timestep(df):
    """
    df: Pandas Dataframe with the lightcurve data

    return val: average timestep between oreturnbservations of the lightcurve
    """
    times = df.index.values
    diff_sums = 0
    for i in range(len(times) - 1):
        diff_sums += times[i+1] - times[i]
    return diff_sums / float(len(times)-1)


def get_color_curve(fp, given_band="B"):
    if given_band == "B":
        b_fp = fp
        r_fp = b_fp.replace(".B.mjd", ".R.mjd")
    else:
        r_fp = fp
        b_fp = r_fp.replace(".R.mjd", ".B.mjd")

    lc_b = eliminate_lc_noise(open_lightcurve(b_fp))
    lc_r = eliminate_lc_noise(open_lightcurve(r_fp))

    lc_join = pd.merge(lc_b, lc_r, left_index=True, right_index=True,
                       suffixes=('_b', '_r'))
    return pd.DataFrame(lc_join['mag_b'] - lc_join['mag_r'], columns=['mag'])


def open_lightcurve(fp, dataset='macho'):
    """
    fp: Absolute file path of the lightcurve file
    """
    cols = ['mjd', 'mag', 'err']
    if dataset == 'macho':
        data = pd.read_table(fp, skiprows=[0, 1, 2], names=cols,
                             index_col='mjd', sep='\s+')
        data = eliminate_lc_noise(data)
    elif dataset == 'ogle':
        data = pd.read_table(fp, names=cols, index_col='mjd', sep='\s+')
    elif dataset == 'eros':
        cols = ['mjd', 'mag', 'err', 'mag2', 'err2']
        data = pd.read_table(fp, skiprows=[0, 1, 2, 3], names=cols,
                             index_col='mjd', sep='\s+')
        """
        EROS lcs come with some measurements that say 99.999 in both bands,
        which need to be removed before calculating any features
        """
        data = data[data['mag'] != 99.999]   # filter out the first band
        data = data[data['mag2'] != 99.999]  # filter out the second band
    return data


def open_lightcurves(lc_list, dataset='macho'):
    """
    lc_list: iterable with lightcurve file paths
    returns: list with Pandas DataFrames for each lightcurve
    """
    return map(lambda x: open_lightcurve(x, dataset=dataset), lc_list)


def open_ogle_lightcurve(fp):
    cols = ['mjd', 'mag', 'err']
    data = pd.read_table(fp, names=cols, index_col='mjd', sep='\s+')
    return data


def open_ogle_lightcurves(lc_list):
    return map(open_ogle_lightcurve, lc_list)


def get_lc_class(fp, dataset="macho"):
    """
    fp: lightcurve file path
    return val: class as an integer
    """
    if dataset == "macho":
        if "Be_lc" in fp:
            return 1
        elif "CEPH" in fp:
            return 2
        elif "EB" in fp:
            return 3
        elif "longperiod_lc" in fp:
            return 4
        elif "microlensing_lc" in fp:
            return 5
        elif "non_variables" in fp:
            return 6
        elif "quasar_lc" in fp:
            return 7
        elif "RRL" in fp:
            return 8
    if dataset == "ogle":
        if "-LPV-" in fp:
            return 1
        elif "-CEP-" in fp:
            return 2
        elif "-T2CEP-" in fp:
            return 3
        elif "-RRLYR-" in fp:
            return 4
        elif "-ECL-" in fp:
            return 5
        elif "-DSCT-" in fp:
            return 6
    if dataset == "eros":
        if "/0/" in fp:
            return 0
        if "/1/" in fp:
            return 1
        if "/2/" in fp:
            return 2
        if "/3/" in fp:
            return 3
        if "/4/" in fp:
            return 4
        if "/5/" in fp:
            return 5
        if "/6/" in fp:
            return 6
        if "/7/" in fp:
            return 7
    return 0


def get_lc_class_string(num):
    """
    num: lightcurve class number
    return val: class as a string
    """
    if num == 4:
        return "Be_lc"
    elif num == 5:
        return "CEPH"
    elif num == 7:
        return "EB"
    elif num == 9:
        return "longperiod_lc"
    elif num == 8:
        return "microlensing_lc"
    elif num == 2:
        return "non_variables"
    elif num == 3:
        return "quasar_lc"
    elif num == 6:
        return "RRL"
    return ""


def get_lc_band(fp):
    """
    fp: lightcurve file path
    return val: band as a string
    """
    if ".R.mjd" in fp:
        return "R"
    elif ".B.mjd" in fp:
        return "B"


def get_macho_id(fp):
    """
    fp: lightcurve file path
    return val: ID of the light curve in its corresponding dataset
    """
    return fp[fp.find('/lc_')+4:-6]


def get_lightcurve_id(fp, dataset="macho"):
    """
    fp: lightcurve file path
    return val: ID of the light curve in its corresponding dataset
    """
    if dataset == "macho":
        return fp[fp.find('/lc_')+4:-6]
    if dataset == "ogle":
        return fp.split("/")[-1][:-4]
    if dataset == "eros":
        return fp.split("/")[-1][:-5]
    return ""


def get_ra_dec(fp):
    """
    fp: lightcurve file path
    return val: tuple (RA, DEC)
    """
    lc_file = open(fp, 'r')
    lc_file.readline()
    vals = lc_file.readline().split(' ')
    return (vals[3], vals[4])


def get_total_observation_time(df):
    """
    df: Pandas Dataframe with the lightcurve data
    returns: Total observation time
    """
    times = df.index.values
    return max(times) - min(times)


def get_ogle_lightcurve_paths():
    f = open(OGLE_LCS_PATH, 'r')
    return [l[:-1] for l in f]


def get_ogle_training_lightcurve_paths():
    f = open(OGLE_TRAINING_LCS_PATH, 'r')
    return [l[:-1] for l in f]


def get_lightcurve_paths(band="B"):
    """
    return val: file object with lightcurve paths in each line
    """
    if band == "B":
        f = open(LC_FILE_PATH_B, 'r')
    elif band == "R":
        f = open(LC_FILE_PATH_R, 'r')
    return [l[:-1] for l in f]


def get_var_training_lightcurve_paths(band="B"):
    """
    return val: file object with lightcurve paths in each line
    """
    if band == "B":
        f = open(LC_VAR_FILE_PATH_B, 'r')
    elif band == "R":
        f = open(LC_VAR_FILE_PATH_R, 'r')
    return [l[:-1] for l in f]


def get_var_lightcurve_paths():
    """
    return val: file object with lightcurve paths in each line
    """
    f = open(VAR_LC_FILE_PATH_B, 'r')
    return [l[:-1] for l in f]


def get_period(curva):
    from scipy.signal import lombscargle
    x = np.array(curva.index) - min(curva.index)
    mag_mean, mag_std = np.mean(curva['mag']), np.std(curva['mag'])
    y = (np.array(curva['mag']) - mag_mean)/mag_std
    T_tot = np.amax(x) - np.amin(x)

    f_N = 0.5*(1/get_avg_timestep(curva))
    f = np.arange(4/T_tot, 10, 0.1/T_tot)
    f = f*2*np.pi
    P = lombscargle(x, y, f)

    f_max = f[P.argmax()]/(2*np.pi)
    return 1/f_max


def plot_periodogram(curva):
    from scipy.signal import lombscargle
    import matplotlib.pyplot as plt
    x = np.array(curva.index) - min(curva.index)
    mag_mean, mag_std = np.mean(curva['mag']), np.std(curva['mag'])
    y = (np.array(curva['mag']) - mag_mean)/mag_std

    T_tot = np.amax(x) - np.amin(x)

    f_N = 0.5*(1/get_avg_timestep(curva))
    f = np.arange(1.0/10, 1.0, 0.01/T_tot)
    P = lombscargle(x, y, f)

    normval = x.shape[0]

    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'b+')
    plt.subplot(2, 1, 2)
    plt.plot(f, np.sqrt(4*(P/normval)), 'b+-')
    plt.show()


def get_folded_lightcurve(lc, period=None):
    lc_f = lc.copy()
    if period is None:
        period = get_period(lc_f)
    lc_f['phase'] = np.mod(lc_f.index, period) / period
    lc_f.index = range(len(lc_f))
    print "period: {0}".format(period)
    return lc_f.sort(columns=['phase']), period


def plot_folded_lightcurve(lc_f, period, show, path):
    import matplotlib.pyplot as plt
    macho_id = get_macho_id(path)
    x = np.array(lc_f['phase']) - min(lc_f['phase'])
    mag_mean, mag_std = np.mean(lc_f['mag']), np.std(lc_f['mag'])
    y = (np.array(lc_f['mag']) - mag_mean)/mag_std
    plt.plot(x, y, '.')
    plt.title("Period: {0}".format(period))
    plt.gca().invert_yaxis()
    if show:
        plt.show()
    else:
        plt.savefig("period_plots/{0}_{1}.png".format(macho_id, period))
    plt.close()
