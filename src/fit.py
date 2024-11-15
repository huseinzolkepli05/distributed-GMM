from sklearn.mixture import BayesianGaussianMixture
from dask.distributed import LocalCluster
from tqdm import tqdm
import time
import pandas as pd
import dask.dataframe
import pickle
import click
import os

@click.command()
@click.option('--partition_directory', default='./save', help='partition directory')
@click.option('--num_clusters', default=10, help='number of cluster')
@click.option('--save_file', default='./save.pkl', help='model save name')
def function(partition_directory, num_clusters, save_file):

    df = dask.dataframe.read_parquet(partition_directory)

    before = time.time()
    
    model = None
    for i in tqdm(range(df.npartitions)):
        part = df.get_partition(i).compute()

        if model is None:
            model = BayesianGaussianMixture(
                        n_components = num_clusters,
                        weight_concentration_prior_type='dirichlet_process',
                        weight_concentration_prior=0.001,
                        max_iter=100,n_init=1, random_state=42)
        
        model = model.fit(part)
    
    with open(save_file, 'wb') as fopen:
        pickle.dump(model, fopen)

    print(f'done! Time taken {time.time() - before} seconds')

if __name__ == '__main__':
    function()