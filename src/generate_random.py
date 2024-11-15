from dask.distributed import LocalCluster
from tqdm import tqdm
import time
import dask
import random
import pandas as pd
import click
import os

@click.command()
@click.option('--row_size', default=1000000, help='number of rows to generate')
@click.option('--partition_size', default=100000, help='size of partition')
@click.option('--batch_size', default=100, help='batch size to send to dask to prevent big graph warning')
@click.option('--save_directory', default='./save', help='save directory')
def function(row_size, partition_size, batch_size, save_directory):
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
        a = [random.uniform(0.0, 15.0) for _ in range(partition_size)]
        df = pd.DataFrame({'A': a})
        df.to_parquet(os.path.join(save_directory, f'{index}.parquet'))
        return 1
    
    before = time.time()
    
    max_size = (row_size // partition_size)
    for i in tqdm(range(0, max_size, batch_size)):
        output = []
        for k in range(i, i + batch_size, 1):
            a = dask.delayed(loop)(k)
            output.append(a)
        total = dask.delayed(sum)(output)
        total.compute()

    print(f'done! Time taken {time.time() - before} seconds')

if __name__ == '__main__':
    function()