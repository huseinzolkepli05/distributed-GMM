from sklearn.mixture import BayesianGaussianMixture
from dask.distributed import LocalCluster
from tqdm import tqdm
import pandas as pd
import dask.dataframe
import pickle
import click
import os

@click.command()
@click.option('--partition_directory', default='./save', help='partition directory')
@click.option('--num_clusters', default=10, help='number of cluster')
@click.option('--save_directory', default='./save_fit', help='fit directory to save')
def function(partition_directory, num_clusters, save_directory):

    os.makedirs(save_directory, exist_ok = True)

    cluster = LocalCluster() 
    client = cluster.get_client()
    print('running local cluster', cluster)
    
    df = dask.dataframe.read_parquet(partition_directory)
    
    def apply(df, partition_info=None):

        if partition_info is not None:
            partition_id = partition_info['number']

            model = BayesianGaussianMixture(
                n_components = num_clusters, 
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.001,
                max_iter=100,n_init=1, random_state=42
            )
            print(f'currently fitting {partition_id} with df shape {df.shape}')
            model = model.fit(df.values)

            filename = os.path.join(save_directory, f'{partition_id}.pkl')
            with open(filename, 'wb') as fopen:
                pickle.dump(model, fopen)
            
            print(f'done fitting {partition_id}')
    
    a = df.map_partitions(apply)
    a.compute()

    print('done!')

if __name__ == '__main__':
    function()