from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import mean_squared_error 
from dask.distributed import LocalCluster
from glob import glob
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import dask.dataframe
import pickle
import click
import os

@click.command()
@click.option('--partition_directory', default='./save', help='partition directory')
@click.option('--model_filename', default='./save.pkl', help='model filename')
@click.option('--transform_directory', default='./save_transform', help='transform directory to save')
def function(partition_directory, model_filename, transform_directory):

    cluster = LocalCluster() 
    client = cluster.get_client()
    print('running local cluster', cluster)
    
    df = dask.dataframe.read_parquet(partition_directory)

    def apply(df, partition_info=None):

        if partition_info is not None:
            partition_id = partition_info['number']

            with open(model_filename, 'rb') as fopen:
                gm = pickle.load(fopen)

            n_components = gm.n_components
            current = df.values

            data_t = np.zeros([len(current), 1])

            with open(os.path.join(transform_directory, f'{partition_id}.pkl'), 'rb') as fopen:
                features, re_ordered_phot, comp = pickle.load(fopen)

            data = np.concatenate([features, re_ordered_phot], axis=1)
            
            st = 0
            
            u = data[:, st]
            u = np.clip(features, -1, 1)[:,0]

            v = data[:, st + 1:st + 1 + np.sum(comp)]
            v_t = np.ones((current.shape[0], n_components)) * -100
            v_t[:, comp] = v
            v = v_t

            means = gm.means_.reshape([-1])
            stds = np.sqrt(gm.covariances_).reshape([-1])
            p_argmax = np.argmax(v, axis=1)
            std_t = stds[p_argmax]
            mean_t = means[p_argmax]
            
            tmp = u * 4 * std_t + mean_t
            
            data_t[:, st] = tmp

            error = mean_squared_error(current[:, 0], data_t[:, 0]) 

            print(f'partition id {partition_id}, MSE {error}')

    a = df.map_partitions(apply)
    
    before = time.time()

    a.compute()

    print(f'done! Time taken {time.time() - before} seconds')

if __name__ == '__main__':
    function()