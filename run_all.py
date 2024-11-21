from distributed_gmm import (
    generate_random,
    fit,
    transform,
    inverse_transform,
)

if __name__ == '__main__':
    generate_random.function(100000, 5000, 100, './save')