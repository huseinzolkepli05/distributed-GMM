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
@click.option('--model_directory', default='./save_fit', help='model directory')
@click.option('--save_directory', default='./save_transform', help='transform directory to save')
@click.option('--eps', default=0.005, help='epsilon')
def function(partition_directory, model_directory, save_directory, eps):
    os.makedirs(save_directory, exist_ok = True)

    cluster = LocalCluster() 
    client = cluster.get_client()
    print('running local cluster', cluster)
    
    df = dask.dataframe.read_parquet(partition_directory)

    def apply(df, partition_info=None):

        if partition_info is not None:
            partition_id = partition_info['number']

            files = glob(os.path.join(model_directory, '*.pkl'))
            models = []
            for f in files:
                with open(f, 'rb') as fopen:
                    models.append(pickle.load(fopen))

            n_components = models[0].n_components

            comps, means, covariances = [], [], []
            weights_sum = np.zeros(n_components)
            means_sum = np.zeros((n_components, models[0].means_.shape[1]))
            covariances_sum = np.zeros_like(models[0].covariances_)
            for gm in models:
                old_comp = gm.weights_ > eps
                mode_freq = (pd.Series(gm.predict(df)).value_counts().keys())
                comp = []
                for i in range(n_components):
                    if (i in (mode_freq)) & old_comp[i]:
                        comp.append(True)
                    else:
                        comp.append(False)
                comps.append(comp)

                weights_sum += gm.weights_
                means_sum += gm.weights_[:, np.newaxis] * gm.means_
                covariances_sum += gm.weights_[:, np.newaxis, np.newaxis] * gm.covariances_
            
            weights_sum /= weights_sum.sum()
            means_merged = means_sum / weights_sum[:, np.newaxis]
            covariances_merged = covariances_sum / weights_sum[:, np.newaxis, np.newaxis]

            means = means_merged.reshape((1, n_components))
            stds = np.sqrt(covariances_merged).reshape((1, n_components))
            comp = [all(column) for column in zip(*comps)]

            current = df.values
            features = np.empty(shape=(len(current),n_components))
            features = (current - means) / (4 * stds) 
            n_opts = sum(comp)
            print(n_opts)
            
            probs = []
            for gm in models:
                probs.append(gm.predict_proba(current.reshape([-1, 1])))
            
            probs = np.mean(np.array(probs), axis = 0)
            probs = probs[:, comp]
            opt_sel = np.zeros(len(current), dtype='int')
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
                pickle.dump([features, re_ordered_phot], fopen)

    a = df.map_partitions(apply)
    a.compute()

    print('done!')

if __name__ == '__main__':
    function()



