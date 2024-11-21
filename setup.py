import setuptools


__packagename__ = 'distributed-GMM'

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='0.1',
    python_requires='>=3.10',
    description='Distributed GMM to scale to any size of data',
    author='huseinzol05',
    url='https://github.com/mesolitica/distributed-GMMe',
)