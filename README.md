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

### Fit GMM distributedly

```bash
python3 src/fit.py --help
```

```
Usage: fit.py [OPTIONS]

Options:
  --partition_directory TEXT  partition directory
  --num_clusters INTEGER      number of cluster
  --save_directory TEXT       fit directory to save
  --help                      Show this message and exit.
```

```bash
python3 src/fit.py
```

```
running local cluster LocalCluster(9f9401b5, 'tcp://127.0.0.1:36123', workers=5, threads=20, memory=78.33 GiB)
currently fitting 9 with df shape (1000, 1)
currently fitting 3 with df shape (1000, 1)
currently fitting 2 with df shape (1000, 1)
currently fitting 7 with df shape (1000, 1)
currently fitting 8 with df shape (1000, 1)
currently fitting 4 with df shape (1000, 1)
currently fitting 1 with df shape (1000, 1)
currently fitting 6 with df shape (1000, 1)
currently fitting 5 with df shape (1000, 1)
currently fitting 0 with df shape (1000, 1)
/home/husein/.local/lib/python3.10/site-packages/sklearn/mixture/_base.py:268: ConvergenceWarning: Initialization 1 did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.
  warnings.warn(
done fitting 2
done fitting 7
/home/husein/.local/lib/python3.10/site-packages/sklearn/mixture/_base.py:268: ConvergenceWarning: Initialization 1 did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.
  warnings.warn(
done fitting 4
/home/husein/.local/lib/python3.10/site-packages/sklearn/mixture/_base.py:268: ConvergenceWarning: Initialization 1 did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.
  warnings.warn(
/home/husein/.local/lib/python3.10/site-packages/sklearn/mixture/_base.py:268: ConvergenceWarning: Initialization 1 did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.
  warnings.warn(
done fitting 0
done fitting 8
done fitting 9
done fitting 3
/home/husein/.local/lib/python3.10/site-packages/sklearn/mixture/_base.py:268: ConvergenceWarning: Initialization 1 did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.
  warnings.warn(
done fitting 5
done fitting 6
done fitting 1
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
  --model_directory TEXT      model directory
  --save_directory TEXT       transform directory to save
  --eps FLOAT                 epsilon
  --help                      Show this message and exit.
```