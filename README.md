# distributed-GMM

Distributed GMM to scale to any size of data. **Currently only support continuous column**.

## how to

Before that, make sure meet all requirements,

```bash
pip3 install -r requirements.txt
```

### Simulating huge data

#### Generate randomly

```bash
python3 src/generate_random.py --help
```

```
Usage: generate_random.py [OPTIONS]

Options:
  --row_size INTEGER        number of rows to generate
  --partition_size INTEGER  size of partition
  --save_directory TEXT     save directory
  --help                    Show this message and exit.
```

Generate 10k rows and partition each 1k rows,

```bash
python3 src/generate_random.py --row_size 10000 --partition_size 1000 --save_directory './save'
```

```
running local cluster LocalCluster(b9d09010, 'tcp://127.0.0.1:44637', workers=5, threads=20, memory=78.33 GiB)
100%|███████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 2665927.67it/s]
100%|███████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 2945231.37it/s]

done!
```

**If the model is very memory consuming like O^2, it is better to make sure the partition size is small because each partitions will train the model, this can explode the memory pretty quickly**.

### Fit GMM incrementally

```bash
python3 src/fit.py --help
```

```
Usage: fit.py [OPTIONS]

Options:
  --partition_directory TEXT  partition directory
  --num_clusters INTEGER      number of cluster
  --save_file TEXT            model save name
  --help                      Show this message and exit.
```

```bash
python3 src/fit.py
```

```
10it [00:10,  1.07s/it]
done!
```

### Transform distributedly

```bash
python3 src/transform.py --help
```

```
Usage: transform.py [OPTIONS]

Options:
  --partition_directory TEXT  partition directory
  --model_filename TEXT       model filename
  --save_directory TEXT       transform directory to save
  --eps FLOAT                 epsilon
  --help                      Show this message and exit.
```

```bash
python3 src/transform.py
```

```
running local cluster LocalCluster(35e7be23, 'tcp://127.0.0.1:39533', workers=5, threads=20, memory=78.33 GiB)
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but BayesianGaussianMixture was fitted with feature names
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but BayesianGaussianMixture was fitted with feature names
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but BayesianGaussianMixture was fitted with feature names
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but BayesianGaussianMixture was fitted with feature names
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but BayesianGaussianMixture was fitted with feature names
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but BayesianGaussianMixture was fitted with feature names
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but BayesianGaussianMixture was fitted with feature names
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but BayesianGaussianMixture was fitted with feature names
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but BayesianGaussianMixture was fitted with feature names
  warnings.warn(
done!
```

### Inverse transform distributedly

```
python3 src/inverse_transform.py --help
```

```
Usage: inverse_transform.py [OPTIONS]

Options:
  --partition_directory TEXT  partition directory
  --model_filename TEXT       model filename
  --transform_directory TEXT  transform directory to save
  --help                      Show this message and exit.
```

```
python3 src/inverse_transform.py
```

```
running local cluster LocalCluster(3180dc9f, 'tcp://127.0.0.1:46785', workers=5, threads=20, memory=78.33 GiB)
partition id 9, MSE 23.618487447468592
partition id 3, MSE 24.376712829199654
partition id 6, MSE 23.58759148082721
partition id 2, MSE 24.628141359062816
partition id 7, MSE 4.967358512563559e-33
partition id 8, MSE 5.460396578326691e-33
partition id 1, MSE 3.6115038317149445e-33
partition id 5, MSE 8.32001735975286e-33
partition id 4, MSE 5.1645737388688115e-33
partition id 0, MSE 6.877881017395697e-33
done!
```