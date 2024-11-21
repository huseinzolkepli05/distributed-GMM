from sklearn.mixture import BayesianGaussianMixture
from dask.distributed import LocalCluster
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import dask.dataframe
import pickle
import click
import os

def function(array_file, num_clusters, save_file):

    arr = np.expand_dims(np.load(array_file), 1)
    print(f'training data shape {arr.shape}')

    before = time.time()
    
    model = BayesianGaussianMixture(
        n_components = num_clusters,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.001,
        max_iter=100,n_init=1, random_state=42)
    
    model = model.fit(arr)

    with open(save_file, 'wb') as fopen:
        pickle.dump(model, fopen)

    print(f'done! Time taken {time.time() - before} seconds')

@click.command()
@click.option('--array_file', default='./sample.npy', help='sampled npy file')
@click.option('--num_clusters', default=10, help='number of cluster')
@click.option('--save_file', default='./save.pkl', help='model save name')
def cli(array_file, num_clusters, save_file):
    function(array_file, num_clusters, save_file)

if __name__ == '__main__':
    cli()