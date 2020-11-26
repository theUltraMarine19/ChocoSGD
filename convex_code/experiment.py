import argparse
# import multiprocessing as mp
import ray
from ray import put, get
from ray.util.multiprocessing.pool import Pool
import os
# import pickle
import joblib
from sklearn.datasets import load_svmlight_file

import numpy as np

from logistic import LogisticDecentralizedSGD
from parameters import Parameters
from utils import pickle_it

# A_obj, y_obj = None, None
n_repeat = 5

def run_logistic(param):
    m = LogisticDecentralizedSGD(param)
    print(param.A_obj, param.y_obj)
    A = get(param.A_obj)
    y = get(param.y_obj)
    res = m.fit(A, y)
    print('{} - score: {}'.format(param, m.score(A, y)))
    return res


def run_experiment(directory, dataset_path, params, nproc=None):
    global A_obj, y_obj
    ray.init(address="auto")
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle_it(params, 'params', directory)

    print('load dataset')
    with open(dataset_path, 'rb') as f:
      A, y = joblib.load(f)
      A_obj = put(A)
      y_obj = put(y)
      for i in range(0, n_repeat):
          params[i].A_obj = A_obj
          params[i].y_obj = y_obj
          print("A.shape= ", A.shape, params[i].A_obj, params[i].y_obj)

    print('start experiment')
    pool = Pool(ray_address='auto')
    results = pool.map(run_logistic, params)
    print(results)

    pickle_it(results, 'results', directory)
    print('results saved in "{}"'.format(directory))
