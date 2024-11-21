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
python3 distributed_gmm/generate_random.py --help
```

```
Usage: generate_random.py [OPTIONS]

Options:
  --row_size INTEGER        number of rows to generate
  --partition_size INTEGER  size of partition
  --batch_size INTEGER      batch size to send to dask to prevent big graph
                            warning
  --save_directory TEXT     save directory
  --help                    Show this message and exit.
```

Generate 1M rows and partition each 5k rows,

```bash
python3 distributed_gmm/generate_random.py --row_size 1000000 --partition_size 5000 --save_directory './save'
```

```
running local cluster LocalCluster(44a4c84d, 'tcp://127.0.0.1:43245', workers=5, threads=20, memory=78.33 GiB)
100%|████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  9.52it/s]
done! Time taken 0.21091151237487793 seconds
```

Took 0.21091151237487793 seconds to generate random 1M rows.

**If the model is very memory consuming like O^2, it is better to make sure the partition size is small because each partitions will train the model, this can explode the memory pretty quickly**.

### Sampling distributedly

This is to reduce the amount of data to train the model by simply generate samples using `np.percentile` from 0-100 for each partitions,

```bash
python3 distributed_gmm/sampling.py --help
```

```
Usage: sampling.py [OPTIONS]

Options:
  --partition_directory TEXT      partition directory
  --max_sample_size_partition INTEGER
                                  max sample row size for each partition, it
                                  can be less but not more than definition
  --max_sample_size INTEGER       max sample size after merging from all
                                  partition samples, will perform another
                                  samples
  --save_directory TEXT           transform directory to save
  --save_file TEXT                save sampled file as npy format
  --help                          Show this message and exit.
```

```bash
python3 distributed_gmm/sampling.py
```

```
running local cluster LocalCluster(d3d8b88b, 'tcp://127.0.0.1:41921', workers=5, threads=20, memory=78.33 GiB)
combined resample from partition (200000,)
final resample (10000,)
done! Time taken 0.257307767868042 seconds
```

Took 0.257307767868042 seconds to resample data from 1M rows to 10k rows.

### Fit GMM from the sampled data

```bash
python3 distributed_gmm/fit.py --help
```

```
Usage: fit.py [OPTIONS]

Options:
  --array_file TEXT       sampled npy file
  --num_clusters INTEGER  number of cluster
  --save_file TEXT        model save name
  --help                  Show this message and exit.
```

```bash
python3 distributed_gmm/fit.py
```

```
training data shape (10000, 1)
/home/husein/.local/lib/python3.10/site-packages/sklearn/mixture/_base.py:268: ConvergenceWarning: Initialization 1 did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.
  warnings.warn(
done! Time taken 0.43245673179626465 seconds
```

Took 0.43245673179626465 seconds to train on 10k sampled rows.

### Transform distributedly

```bash
python3 distributed_gmm/transform.py --help
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
python3 distributed_gmm/transform.py
```

```
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but BayesianGaussianMixture was fitted without feature names
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but BayesianGaussianMixture was fitted without feature names
  warnings.warn(
done! Time taken 4.768050909042358 seconds
```

Took 4.768050909042358 seconds to distributedly transformed 1M rows.

### Inverse transform distributedly

```
python3 distributed_gmm/inverse_transform.py --help
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
python3 distributed_gmm/inverse_transform.py
```

```
partition id 101, MSE 16.204095271461085
partition id 106, MSE 55.785496029013395
partition id 100, MSE 48.710000906949595
partition id 105, MSE 53.49456939221145
partition id 102, MSE 57.51461710739251
partition id 0, MSE 44.393166078498794
partition id 104, MSE 51.0450072769448
partition id 107, MSE 32.531861655189246
done! average MSE 39.41494702500879, Time taken 0.26247501373291016 seconds
```

Took 0.26247501373291016 seconds to distributedly inverse transformed 1M rows.

## Estimate time taken for 1B rows

Using default 5 workers 20 threads,

```
running local cluster LocalCluster(0ab1dbbf, 'tcp://127.0.0.1:41463', workers=5, threads=20, memory=78.33 GiB)
```

**If you check the code, `before` variable put literally behind few lines from the actual compute, this is to make sure we compute the actual time taken just for the execution**.

### Generate random rows

(0.21091151237487793 / 1e6) * 1e9 = 210.91151237487793

210.91151237487793 seconds or 3.5151918729146323 minutes.

#### But actual execution might be more faster

```
python3 distributed_gmm/generate_random.py --row_size 1000000000 --partition_size 5000 --save_directory './save'
```

```
running local cluster LocalCluster(d8d462d2, 'tcp://127.0.0.1:39569', workers=5, threads=20, memory=78.33 GiB)
  1%|▉                                                                                   | 21/2000 [00:02<03:28,  9.51it/s]
```

### Sampling

(0.257307767868042 / 1e6) * 1e9 = 257.307767868042

257.307767868042 seconds or 4.2884627978007 minutes or 0.071474379963345 hours.

**This actually might be more faster like generating the random rows**.

### Fit GMM

0.43245673179626465 seconds.

### Transform

(4.768050909042358 / 1e6) * 1e9 = 4768.050909042358

4768.050909042358 seconds or 79.46751515070598 minutes or 1.3244585858450997 hours.

**This actually might be more faster like generating the random rows**.

### Inverse Transform

(0.26247501373291016 / 1e6) * 1e9 = 262.47501373291016

262.47501373291016 seconds or 4.374583562215169 minutes.

**This actually might be more faster like generating the random rows**.

### Total

210.91151237487793 + 257.307767868042 + 0.43245673179626465 + 4768.050909042358 + 262.47501373291016 = 5499.177659749985

5499.177659749985 seconds or 91.65296099583308 minutes or 1.5275493499305515 hours.

## how to deploy on premise with zero internet?

<img src="airflow-v2.png" width="50%">

### Why Airflow?

1. Everything run as DAG and separate.

If we put everything as one script, failure at certain points must rerun the entire script.

But if we run as DAG and each node run one function, failure at certain nodes, we just simply rerun that nodes.

2. Better logging system.

Each DAG nodes have their own logging files, easier for debugging.

3. RBAC UI.

Airflow UI natively RBAC, for an example, we want all users able to check the logs but not all able to rerun the DAG.

### Installation

1. Make sure already provided VM with necessary Linux OS, Redhat or Debian, both are fine, bring a base image with already installed necessary Python version,

For debian,

```bash
sudo apt update
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-dev -y
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
python3 -m pip install --upgrade pip
```

2. Fetch all necessary wheels based on the requirements.txt,

Either this one you want to bundle together with the base image or you want to separate install, it depends on certain on-premise, some on-premises want to scan the whl files first and make sure everything able to install with 0 internet,

```bash
mkdir whl_dir
FILE="requirements.txt"
while IFS= read -r line || [[ -n "$line" ]]; do
    echo "Downloading: $line"
    pip3 download "$line"
done < "$FILE"
mv *.whl whl_dir
python3 setup.py sdist bdist_wheel
mv dist/*.whl whl_dir
zip -r whl_dir.zip whl_dir
```

3. After that, you can import this library pretty easy,

```python
from distributed_gmm import (
    generate_random,
    fit,
    transform,
    inverse_transform,
)

if __name__ == '__main__':
    generate_random.function(100000, 5000, 100, './save')
```

Do not forget to run inside `if __name__ == '__main__'` because by default dask cluster use multiprocessing and this required a fork or else you will get an error,

```
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
```

Unless we set `processes=False` to use threads based, https://docs.dask.org/en/stable/deploying-python.html#reference