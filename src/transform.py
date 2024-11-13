from sklearn.mixture import BayesianGaussianMixture
from dask.distributed import LocalCluster
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import dask.dataframe
import pickle
import click
import os

@click.command()
@click.option('--partition_directory', default='./save', help='partition directory')
@click.option('--model_filename', default='./save.pkl', help='model filename')
@click.option('--save_directory', default='./save_transform', help='transform directory to save')
@click.option('--eps', default=0.005, help='epsilon')
def function(partition_directory, model_filename, save_directory, eps):
    os.makedirs(save_directory, exist_ok = True)

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

            old_comp = gm.weights_ > eps
            mode_freq = (pd.Series(gm.predict(df)).value_counts().keys())
            comp = []
            for i in range(n_components):
                if (i in (mode_freq)) & old_comp[i]:
                    comp.append(True)
                else:
                    comp.append(False)

            means = gm.means_.reshape((1, n_components))
            stds = np.sqrt(gm.covariances_).reshape((1, n_components))

            current = df.values
            features = np.empty(shape=(len(current),n_components))
            features = (current - means) / (4 * stds) 
            n_opts = sum(comp)
            
            opt_sel = np.zeros(len(current), dtype='int')
            probs = gm.predict_proba(current)
            probs = probs[:, comp]
            for i in range(len(current)):
                pp = probs[i] + 1e-6
                pp = pp / sum(pp)
                opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
                
            probs_onehot = np.zeros_like(probs)
            probs_onehot[np.arange(len(probs)), opt_sel] = 1
            idx = np.arange((len(features)))
            features = features[:, comp]
            features = features[idx, opt_sel].reshape([-1, 1])
            features = np.clip(features, -.99, .99) 
            
            re_ordered_phot = np.zeros_like(probs_onehot)  
            col_sums = probs_onehot.sum(axis=0)
            n = probs_onehot.shape[1]
            largest_indices = np.argsort(-1*col_sums)[:n]
            for id,val in enumerate(largest_indices):
                re_ordered_phot[:,id] = probs_onehot[:,val]
            
            filename = os.path.join(save_directory, f'{partition_id}.pkl')
            with open(filename, 'wb') as fopen:
                pickle.dump([features, re_ordered_phot, comp], fopen)

    a = df.map_partitions(apply)
    a.compute()

    print('done!')

if __name__ == '__main__':
    function()



