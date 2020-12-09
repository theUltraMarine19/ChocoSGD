import argparse
import multiprocessing as mp
import os
import pickle
from sklearn.datasets import load_svmlight_file

import numpy as np

from logistic import LogisticDecentralizedSGD
from parameters import Parameters
from utils import pickle_it

from experiment import run_logistic, run_experiment

A, b = None, None

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('experiment', type=str)
  args = parser.parse_args()
  print(args)
  assert args.experiment in ['final']

  dataset_path = os.path.expanduser('../data/epsilon.joblib')
  n, d = 400000, 2000

###############################################
### RANDOM DATA PARTITION #####################
###############################################
  n_cores = 40
################### FINAL ################################

#   split_way = 'random'
#   split_name = split_way

#   num_epoch = 10
#   n_repeat = 5
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="decentralized-exact", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, quantization='full',
#                            n_cores=n_cores, method='plain',
#                            split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name,
#                            topology='ring', estimate='final'),
#       ]
#     run_experiment("dump/epsilon-final-decentralized-" + split_way + "-" +\
#                    str(n_cores) + "/", dataset_path, params, nproc=10)

#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="centralized", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, quantization='full',
#                            n_cores=n_cores, method='plain',
#                            split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name,
#                            topology='centralized', estimate='final')]
#     run_experiment("dump/epsilon-final-centralized-" + split_way + "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)


#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="decentralized-top-20", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.04,
#                            quantization='top', coordinates_to_keep=20,
#                            n_cores=n_cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)]
#     run_experiment("dump/epsilon-final-choco-top-20-" + split_way + "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)


#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="decentralized-random-20", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.01,
#                            quantization='random-biased', coordinates_to_keep=20,
#                            n_cores=n_cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)]
#     run_experiment("dump/epsilon-final-choco-random-20-" + split_way+ "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)


#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="decentralized-qsgd-8", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,
#                            quantization='qsgd-biased', num_levels=16,
#                            n_cores=n_cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)]
#     run_experiment("dump/epsilon-final-choco-qsgd-4bit-" + split_way + "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)

#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [Parameters(name="dcd-psgd-random-20",
#                            num_epoch=num_epoch, lr_type='decay',
#                            initial_lr=1e-15, tau=d, regularizer=1 / n,
#                            quantization='random-unbiased', coordinates_to_keep=20,
#                            n_cores=n_cores, method='dcd-psgd',
#                            split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name,
#                            topology='ring', estimate='final')]
#     run_experiment("dump/epsilon-final-dcd-random-20-" + split_way + "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)

#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [Parameters(name="ecd-psgd-random", num_epoch=num_epoch,
#                             lr_type='decay', initial_lr=1e-6, tau=d,
#                             regularizer=1 / n, consensus_lr=None,
#                             quantization='random-unbiased',
#                             coordinates_to_keep=20, n_cores=n_cores,
#                             method='ecd-psgd', split_data_random_seed=random_seed,
#                             distribute_data=True, split_data_strategy=split_name,
#                             topology='ring', estimate='final')]
#     run_experiment("dump/epsilon-final-ecd-random-20-" + split_way + "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)

# ###################### qsgd quantization #####################################

  
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                             lr_type='decay', initial_lr=0.01, tau=d,
#                             regularizer=1 / n, quantization='qsgd-unbiased',
#                             num_levels=16, n_cores=n_cores, method='dcd-psgd',
#                             split_data_random_seed=random_seed,
#                             distribute_data=True, split_data_strategy=split_name,
#                             topology='ring', estimate='final')]
#     run_experiment("dump/epsilon-final-dcd-qsgd-4bit-" + split_way + "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)

#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
#                             lr_type='decay', initial_lr=1e-06, tau=d,
#                             regularizer=1 / n, quantization='qsgd-unbiased',
#                             num_levels=16, n_cores=n_cores, method='ecd-psgd',
#                             split_data_random_seed=random_seed,
#                             distribute_data=True, split_data_strategy=split_name,
#                             topology='ring', estimate='final')]
#     run_experiment("dump/epsilon-final-ecd-qsgd-4bit-" + split_way + "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)




###############################################
### SORTED DATA PARTITION #####################
###############################################
################################ FINAL ####################################




  split_way = 'sorted'
  split_name = split_way
  if split_way == 'sorted':
    split_name = 'label-sorted'

  n_repeat = 5
  num_epoch = 10

  # # D-PSGD on ring topology (1 round of ring-reduce) with top 1%
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="decentralized-top-20", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='top',
  #                          coordinates_to_keep=20,
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # D-PSGD on ring topology (1 round of ring-reduce) with random 1%
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="decentralized-random", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='random-unbiased',
  #                          coordinates_to_keep=20, 
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # ChocoSGD on ring topology with top 1%
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="decentralized-top-20", num_epoch=num_epoch, 
  #                          lr_type='decay', initial_lr=0.1, tau=d, 
  #                          regularizer=1 / n, quantization='top', 
  #                          consensus_lr=0.04, coordinates_to_keep=20, 
  #                          n_cores=n_cores, method='choco', 
  #                          split_data_random_seed=random_seed, 
  #                          distribute_data=True, split_data_strategy=split_name, 
  #                          topology='ring', estimate='final',
  #                          random_seed=40 + random_seed)]
  #   run_experiment("dump/epsilon-final-choco-top-20-ring-" + split_way+ "-" + str(n_cores) + "/",
  #       dataset_path, params, nproc=10)

  # # ChocoSGD on ring topology with random-unbiased 1%
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="decentralized-random-un-20", num_epoch=num_epoch, 
  #                          lr_type='decay', initial_lr=0.1, tau=d, 
  #                          regularizer=1 / n, quantization='random-unbiased', 
  #                          consensus_lr=0.01, coordinates_to_keep=20, 
  #                          n_cores=n_cores, method='choco',
  #                          split_data_random_seed=random_seed, 
  #                          distribute_data=True, split_data_strategy=split_name, 
  #                          topology='ring', estimate='final',
  #                          random_seed=60 + random_seed)]
  #   run_experiment("dump/epsilon-final-choco-random-20-" + split_way+ "-" + str(n_cores) + "/",
  #         dataset_path, params, nproc=10)

  # # ChocoSGD on ring topology with random-biased 1%
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="decentralized-random-20", num_epoch=num_epoch, 
  #                          lr_type='decay', initial_lr=0.1, tau=d, 
  #                          regularizer=1 / n, quantization='random-biased', 
  #                          consensus_lr=0.01, coordinates_to_keep=20, 
  #                          n_cores=n_cores, method='choco',
  #                          split_data_random_seed=random_seed, 
  #                          distribute_data=True, split_data_strategy=split_name, 
  #                          topology='ring', estimate='final',
  #                          random_seed=60 + random_seed)]
  #   run_experiment("dump/epsilon-final-choco-random-20-" + split_way+ "-" + str(n_cores) + "/",
  #         dataset_path, params, nproc=10)

  # # DCD-PSGD on ring topology with random 1%
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [Parameters(name="dcd-psgd-random-20", num_epoch=num_epoch,
  #                           lr_type='decay', initial_lr=1e-15, tau=d,
  #                           regularizer=1 / n, quantization='random-unbiased',
  #                           coordinates_to_keep=20, 
  #                           n_cores=n_cores, method='dcd-psgd',
  #                           split_data_random_seed=random_seed, 
  #                           distribute_data=True, split_data_strategy=split_name, 
  #                           topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-dcd-random-20-" + split_way + "-" + str(n_cores) + "/",
  #                  dataset_path, params, nproc=10)

  # # ECD-PSGD on ring topology with random 1%
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [Parameters(name="ecd-psgd-random", num_epoch=num_epoch, 
  #                           lr_type='decay', initial_lr=1e-10, tau=d, 
  #                           regularizer=1 / n, quantization='random-unbiased', 
  #                           coordinates_to_keep=20,
  #                           n_cores=n_cores, method='ecd-psgd', 
  #                           split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-ecd-random-20-" + split_way + "-" + str(n_cores) + "/",
  #                  dataset_path, params, nproc=10)

###################### qsgd quantization #####################################

  # # D-PSGD on ring topology (1 round of ring-reduce) with qsgd 4-bit
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="dpsgd-qsgd-4-unbiased", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='qsgd-unbiased',
  #                          num_levels=16,
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # D-PSGD on ring topology (1 round of ring-reduce) with qsgd 8-bit
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="dpsgd-qsgd-8-unbiased", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='qsgd-unbiased',
  #                          num_levels=256,
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # ChocoSGD with 4-bit biased quantization
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="decentralized-qsgd-4-biased", num_epoch=num_epoch, 
  #                         lr_type='decay', initial_lr=0.1, tau=d, 
  #                         regularizer=1 / n, quantization='qsgd-biased', 
  #                         consensus_lr=0.34, num_levels=16, 
  #                         n_cores=n_cores, method='choco', 
  #                         split_data_random_seed=random_seed, 
  #                         distribute_data=True, split_data_strategy=split_name,
  #                         topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-choco-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
  #                  dataset_path, params, nproc=10)

  # # ChocoSGD with 4-bit unbiased quantization
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="decentralized-qsgd-4-unbiased", num_epoch=num_epoch, 
  #                         lr_type='decay', initial_lr=0.1, tau=d, 
  #                         regularizer=1 / n, quantization='qsgd-unbiased', 
  #                         consensus_lr=0.34, num_levels=16, 
  #                         n_cores=n_cores, method='choco', 
  #                         split_data_random_seed=random_seed, 
  #                         distribute_data=True, split_data_strategy=split_name,
  #                         topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-choco-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
  #                  dataset_path, params, nproc=10)

  # # DCD-PSGD with 4-bit unbiased qunatization
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [Parameters(name="dcd-psgd-qsgd-4", num_epoch=num_epoch, 
  #                           lr_type='decay', initial_lr=0.01, tau=d, 
  #                           regularizer=1 / n, quantization='qsgd-unbiased', 
  #                           num_levels=16, 
  #                           n_cores=n_cores, method='dcd-psgd', 
  #                           split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final', )]	
  #   run_experiment("dump/epsilon-final-dcd-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
  #                  dataset_path, params, nproc=10)

  # # ECD-PSGD with 4-bit unbiased qunatization
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [Parameters(name="ecd-psgd-qsgd-4", num_epoch=num_epoch, 
  #                           lr_type='decay', initial_lr=1e-12, tau=d, 
  #                           regularizer=1 / n, quantization='qsgd-unbiased', 
  #                           num_levels=16, 
  #                           n_cores=n_cores, method='ecd-psgd', 
  #                           split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-ecd-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
  #                  dataset_path, params, nproc=10)

  # # ChocoSGD with 8-bit biased quantization
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="decentralized-qsgd-8-biased", num_epoch=num_epoch, 
  #                         lr_type='decay', initial_lr=0.1, tau=d, 
  #                         regularizer=1 / n, quantization='qsgd-biased', 
  #                         consensus_lr=0.34, num_levels=256, 
  #                         n_cores=n_cores, method='choco', 
  #                         split_data_random_seed=random_seed, 
  #                         distribute_data=True, split_data_strategy=split_name,
  #                         topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-choco-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
  #                  dataset_path, params, nproc=10)

  # # DCD-PSGD with 8-bit unbiased qunatization
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [Parameters(name="dcd-psgd-qsgd-8", num_epoch=num_epoch, 
  #                           lr_type='decay', initial_lr=0.01, tau=d, 
  #                           regularizer=1 / n, quantization='qsgd-unbiased', 
  #                           num_levels=256, 
  #                           n_cores=n_cores, method='dcd-psgd', 
  #                           split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final', )] 
  #   run_experiment("dump/epsilon-final-dcd-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
  #                  dataset_path, params, nproc=10)

  # # ECD-PSGD with 8-bit unbiased qunatization
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [Parameters(name="ecd-psgd-qsgd-8", num_epoch=num_epoch, 
  #                           lr_type='decay', initial_lr=1e-12, tau=d, 
  #                           regularizer=1 / n, quantization='qsgd-unbiased', 
  #                           num_levels=256, 
  #                           n_cores=n_cores, method='ecd-psgd', 
  #                           split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-ecd-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
  #                  dataset_path, params, nproc=10)  

############## Comparison among all algorithms ###################################

  # D-PSGD on ring topology (1 round of ring-reduce)
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="dpgd-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='top',
  #                          coordinates_to_keep=20,
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # D-PSGD on ring topology (1 round of ring-reduce)
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="dpsgd-random", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='random-unbiased',
  #                          coordinates_to_keep=20,
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # D-PSGD on ring topology (1 round of ring-reduce)
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="dpsgd-qsgd", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='qsgd-unbiased',
  #                          num_levels=16,
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # All-Reduce SGD (Param-server like)
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="centralized-random", num_epoch=num_epoch, 
  #                          lr_type='decay', initial_lr=0.1, tau=d, 
  #                          regularizer=1 / n, quantization='random-unbiased',
  #                          coordinates_to_keep=20, 
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed, 
  #                          distribute_data=True, split_data_strategy=split_name, 
  #                          topology='centralized', estimate='final')]
  #   run_experiment("dump/epsilon-final-centralized-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # All-Reduce SGD (Param-server like)
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="centralized-top", num_epoch=num_epoch, 
  #                          lr_type='decay', initial_lr=0.1, tau=d, 
  #                          regularizer=1 / n, quantization='top',
  #                          coordinates_to_keep=20, 
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed, 
  #                          distribute_data=True, split_data_strategy=split_name, 
  #                          topology='centralized', estimate='final')]
  #   run_experiment("dump/epsilon-final-centralized-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # All-Reduce SGD (Param-server like)
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="centralized-qsgd", num_epoch=num_epoch, 
  #                          lr_type='decay', initial_lr=0.1, tau=d, 
  #                          regularizer=1 / n, quantization='qsgd-unbiased',
  #                          num_levels=16, 
  #                          n_cores=n_cores, method='plain',
  #                          split_data_random_seed=random_seed, 
  #                          distribute_data=True, split_data_strategy=split_name, 
  #                          topology='centralized', estimate='final')]
  #   run_experiment("dump/epsilon-final-centralized-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # EAMSGD on fully-connected topology (Sync)
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="EAMSGD-sync", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='full',
  #                          n_cores=n_cores, method='ea-sgd',
  #                          momentum=0.9, comm_period=1, elasticity=0.9/(n_cores*0.1), 
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='centralized', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # EAMSGD on fully-connected topology (Sync) -constant lr
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="EAMSGD-sync", num_epoch=num_epoch,
  #                          lr_type='constant', initial_lr=0.1,
  #                          regularizer=1 / n, quantization='full',
  #                          n_cores=n_cores, method='ea-sgd',
  #                          momentum=0.9, comm_period=1, elasticity=0.9/(n_cores*0.1), 
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='centralized', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # EAMSGD on fully-connected topology (Async)
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="EAMSGD-async", num_epoch=num_epoch,
  #                          lr_type='constant', initial_lr=0.1, 
  #                          regularizer=1 / n, quantization='full',
  #                          n_cores=n_cores, method='ea-sgd',
  #                          momentum=0.9, comm_period=16, elasticity=0.9/(n_cores*0.1), 
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='centralized', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # SGP With momentum on ring topology
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="SGP-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='top',
  #                          coordinates_to_keep=20,
  #                          n_cores=n_cores, method='SGP',
  #                          momentum=0.9,
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # SGP With momentum on ring topology
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="SGP-random", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='random-unbiased',
  #                          coordinates_to_keep=20,
  #                          n_cores=n_cores, method='SGP',
  #                          momentum=0.9,
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  #  # SGP With momentum on ring topology
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="SGP-qsgd", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, quantization='qsgd-unbiased',
  #                          num_levels=16,
  #                          n_cores=n_cores, method='SGP',
  #                          momentum=0.9,
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # # SGP With momentum on ring topology with different lr
  # if args.experiment in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #     params += [
  #               Parameters(name="SGP-ImageNet", num_epoch=num_epoch,
  #                          lr_type='epoch-decay', epoch_decay_lr=0.93, initial_lr=0.1,
  #                          regularizer=1 / n, quantization='full',
  #                          n_cores=n_cores, method='SGP',
  #                          momentum=0.9,
  #                          split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name,
  #                          topology='ring', estimate='final')]
  #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                  + "/", dataset_path, params, nproc=10)

  # AD-PSGD on ring topology
  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="AD-PSGD", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, quantization='full',
                           # coordinates_to_keep=20,
                           n_cores=n_cores, method='ad-psgd',
                           split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)

  # AD-PSGD on ring topology
  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="AD-PSGD-random", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, quantization='random-unbiased',
                           coordinates_to_keep=20,
                           n_cores=n_cores, method='ad-psgd',
                           split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)

  # AD-PSGD on ring topology
  if args.experiment in ['final']:
    params = []
    for random_seed in np.arange(1, n_repeat + 1):
      params += [
                Parameters(name="AD-PSGD-qsgd", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, quantization='qsgd-unbiased',
                           num_levels=16,
                           n_cores=n_cores, method='ad-psgd',
                           split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='ring', estimate='final')]
    run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
                   + "/", dataset_path, params, nproc=10)


############################# Topo and #workers ###################################

  # for workers in [36, 25, 16, 9]:
  #   for topo in ['centralized', 'ring', 'torus', 'disconnected', 'partly-connected']:
    

  #     # D-PSGD on ring topology (1 round of ring-reduce)
  #     if args.experiment in ['final']:
  #       params = []
  #       for random_seed in np.arange(1, n_repeat + 1):
  #         params += [
  #                   Parameters(name="dpsgd-exact-"+topo+str(workers)+"-", num_epoch=num_epoch,
  #                              lr_type='decay', initial_lr=0.1, tau=d,
  #                              regularizer=1 / n, quantization='full',
  #                              n_cores=workers, method='plain',
  #                              split_data_random_seed=random_seed,
  #                              distribute_data=True, split_data_strategy=split_name,
  #                              topology=topo, estimate='final')]
  #       run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                      + "/", dataset_path, params, nproc=10)

  #     # ChocoSGD with 4-bit biased quantization
  #     if args.experiment in ['final']:
  #       params = []
  #       for random_seed in np.arange(1, n_repeat + 1):
  #         params += [
  #                   Parameters(name="Choco-qsgd-"+topo+str(workers)+"-", num_epoch=num_epoch, 
  #                             lr_type='decay', initial_lr=0.1, tau=d, 
  #                             regularizer=1 / n, quantization='qsgd-biased', 
  #                             consensus_lr=0.34, num_levels=16, 
  #                             n_cores=workers, method='choco', 
  #                             split_data_random_seed=random_seed, 
  #                             distribute_data=True, split_data_strategy=split_name,
  #                             topology=topo, estimate='final')]
  #       run_experiment("dump/epsilon-final-choco-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
  #                      dataset_path, params, nproc=10)

  #     # ChocoSGD with random-biased 1%
  #     if args.experiment in ['final']:
  #       params = []
  #       for random_seed in np.arange(1, n_repeat + 1):
  #         params += [
  #                   Parameters(name="Choco-random-"+topo+str(workers)+"-", num_epoch=num_epoch, 
  #                              lr_type='decay', initial_lr=0.1, tau=d, 
  #                              regularizer=1 / n, quantization='random-biased', 
  #                              consensus_lr=0.01, coordinates_to_keep=20, 
  #                              n_cores=workers, method='choco',
  #                              split_data_random_seed=random_seed, 
  #                              distribute_data=True, split_data_strategy=split_name, 
  #                              topology=topo, estimate='final',
  #                              random_seed=60 + random_seed)]
  #       run_experiment("dump/epsilon-final-choco-random-20-" + split_way+ "-" + str(n_cores) + "/",
  #             dataset_path, params, nproc=10)

  #     # SGP With momentum
  #     if args.experiment in ['final']:
  #       params = []
  #       for random_seed in np.arange(1, n_repeat + 1):
  #         params += [
  #                   Parameters(name="SGP-"+topo+str(workers)+"-", num_epoch=num_epoch,
  #                              lr_type='decay', initial_lr=0.1, tau=d,
  #                              regularizer=1 / n, quantization='full',
  #                              n_cores=workers, method='SGP',
  #                              momentum=0.9,
  #                              split_data_random_seed=random_seed,
  #                              distribute_data=True, split_data_strategy=split_name,
  #                              topology=topo, estimate='final')]
  #       run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
  #                      + "/", dataset_path, params, nproc=10)

      # # AD-PSGD
      # if args.experiment in ['final']:
      #   params = []
      #   for random_seed in np.arange(1, n_repeat + 1):
      #     params += [
      #               Parameters(name="AD-PSGD-"+topo+str(workers)+"-", num_epoch=num_epoch,
      #                          lr_type='decay', initial_lr=0.1, tau=d,
      #                          regularizer=1 / n, quantization='full',
      #                          n_cores=workers, method='ad-psgd',
      #                          split_data_random_seed=random_seed,
      #                          distribute_data=True, split_data_strategy=split_name,
      #                          topology=topo, estimate='final')]
      #   run_experiment("dump/epsilon-final-decentralized-ring-" + split_way+ "-" + str(n_cores)\
      #                  + "/", dataset_path, params, nproc=10)







