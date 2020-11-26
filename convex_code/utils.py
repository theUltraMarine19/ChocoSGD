import glob
import os
# import pickle
import joblib


def to_percent(x):
    return round(100 * x, 2)


def pickle_it(var, name, directory):
    with open(os.path.join(directory, "{}.joblib".format(name)), 'wb') as f:
        joblib.dump(var, f)


def unpickle_dir(d):
    data = {}
    assert os.path.exists(d), "{} does not exists".format(d)
    for file in glob.glob(os.path.join(d, '*.joblib')):
        name = os.path.basename(file)[:-len('.joblib')]
        with open(file, 'rb') as f:
            var = joblib.load(f)
        data[name] = var
    return data
