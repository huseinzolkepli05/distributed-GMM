from dask.distributed import LocalCluster
from tqdm import tqdm
import dask
import random
import pandas as pd
import click
import os

@click.command()
@click.option('--row_size', default=1000000, help='number of rows to generate')
@click.option('--partition_size', default=100000, help='size of partition')
@click.option('--save_directory', default='./save', help='save directory')
def function(row_size, partition_size, save_directory):
    if row_size % partition_size != 0:
        raise Exception('`row_size` must divisible by `partition_size`.')

    try:
        os.remove(save_directory)
    except:
        pass
    os.makedirs(save_directory, exist_ok = True)
    
    cluster = LocalCluster() 
    client = cluster.get_client()
    print('running local cluster', cluster)

    def loop(index):
        a = [random.uniform(0.0, 15.0) for _ in tqdm(range(partition_size))]
        df = pd.DataFrame({'A': a})
        df.to_parquet(os.path.join(save_directory, f'{index}.parquet'))
        return 1
    
    output = []
    for i in range((row_size // partition_size)):
        a = dask.delayed(loop)(i)
        output.append(a)
    
    total = dask.delayed(sum)(output)
    total.compute()
    print('done!')

if __name__ == '__main__':
    function()