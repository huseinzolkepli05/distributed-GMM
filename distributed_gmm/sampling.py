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

def function(partition_directory, max_sample_size_partition, max_sample_size, save_file, save_directory):
    os.makedirs(save_directory, exist_ok = True)

    cluster = LocalCluster()
    client = cluster.get_client()
    print('running local cluster', cluster)

    df = dask.dataframe.read_parquet(partition_directory)

    def apply(df, partition_info=None):

        if partition_info is not None:
            partition_id = partition_info['number']
            arr = df.values[:, 0]

            if arr.shape[0] > max_sample_size_partition:
                percentiles = np.linspace(0, 100, max_sample_size_partition)
                arr = np.percentile(arr, percentiles)
            
            filename = os.path.join(save_directory, f'{partition_id}.npy')
            np.save(filename, arr)
            

    a = df.map_partitions(apply)
    
    before = time.time()

    a.compute()

    np_files = glob(os.path.join(save_directory, '*.npy'))
    arr = np.concatenate([np.load(f) for f in np_files])
    print(f'combined resample from partition {arr.shape}')
    if arr.shape[0] > max_sample_size:
        percentiles = np.linspace(0, 100, max_sample_size)
        arr = np.percentile(arr, percentiles)
        print(f'final resample {arr.shape}')
    
    np.save(save_file, arr)

    print(f'done! Time taken {time.time() - before} seconds')

@click.command()
@click.option('--partition_directory', default='./save', help='partition directory')
@click.option('--max_sample_size_partition', default=1000, help='max sample row size for each partition, it can be less but not more than definition')
@click.option('--max_sample_size', default=10000, help='max sample size after merging from all partition samples, will perform another samples')
@click.option('--save_directory', default='./sample', help='transform directory to save')
@click.option('--save_file', default='./sample.npy', help='save sampled file as npy format')
def cli(partition_directory, max_sample_size_partition, max_sample_size, save_file, save_directory):
    function(partition_directory, max_sample_size_partition, max_sample_size, save_file, save_directory)

if __name__ == '__main__':
    cli()