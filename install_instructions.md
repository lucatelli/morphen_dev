# Installation Instructions
This code depends upon multiple astronomical and signal processing packages and specific applications/interfaces required for functionalities and optmimisations. All these are explained bellow.

## The easy way
### Using a Singularity Container (preferred way)
We provide a definition file and a pre-build singularity container with all dependencies 
installed. 

Notes: 
1. The GPU implementation of JAX was not tested in this container. The definition 
file will install JAX with CPU support.
2. In the definition file, the option to build `wsclean` with GPU support may not work in 
   a different system that I am using.  
2. The CASA viewer application (`imview` from `casaviewer`), is 
   not working due to a `fusermount` issue.


### Using a conda environment file
***TODO***

## The manual way (good for different systems)
For those who want to install everything from scratch, here are the instructions.

### Install Miniconda3
Install `miniconda` with:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

After, create a new conda environment (e.g. python 3.8 is more appropriate to be used with CASA):
```
conda create -n morphen python=3.8
conda update --all
```

### CASA within Conda
At the moment, some libraries depend on the modular CASA pacakge (in Python).
To use the collection of functions inside `morphen`, the recommended way to 
install the modular version of CASA is using `conda` environments. 
Another reason is that this is the easiest way to merge CASA functions with other 
python interfaces, such as `numpy`, `scipy`, `astropy` and `Jax`. 

  


To install casacore within conda, follow the instructions in this page: 
https://github.com/conda-forge/casacore-feedstock

```
conda config --add channels conda-forge
conda update --all
```

Install `casacore` and `casatools`.
Search for different versions:
```
conda search casacore
```


Install CASA related packages with pip:
```
pip install casatools==6.4.4.31
pip install casatasks==6.4.4.31 casaplotms casadata casaviewer casampi casaplotserver casalogger
```

In order to use casaviewer, you have to downgrade `protobuf` to version 3.20.3:
```
pip install protobuf==3.20.3
```

Then, install python related packages:
```
pip install ipython notebook jupyter tqdm matplotlib corner sympy cmasher coloredlogs 
pip install numpy==1.24.3 astropy==5.2.2  pandas scipy==1.10.1
```

[//]: # (logging )

## Image Shape Analysis
Some utilities are used to quantify image structure. For that, python packages that are used are:
```commandline
pip install scikit-image==0.21.0 scikit-learn==1.2.2
```
## Photometry and Source Detection
```commandline
pip install petrofit #--no-deps #avoid issue with deprecated sklearn (already installed with scikit-learn)
pip install photutils==1.7.0 sep fitsio
```

## Image Fitting Libraries
Image fitting is performed with the `LMFIT` package, alongside `scipy` and the Monte Carlo `emcee` package. 
```commandline
pip install lmfit==1.1.0 emcee==3.1.4 h5py==3.8.0 corner arviz==0.15.1
conda install arviz
```

Note: At the momment, for an unknown reason, minimisation using Jax is not happening in the desired way if using the last stable 
version of `LMFIT (> v 1.2.1)`, but it does for `v 1.1.0`. So, while the issue is not identified, 
please use the indicated particular version of `LMFIT`.

## Jax Interface for Optmisation and GPU Processing
Some functions are decorated within the Jax framework to be optmised for CPU or to run on 
Nvidia GPUs (through CUDA).
Jax can be installed with `cuda-12.0`. The recommended way is to install within a conda 
environment (the same created before). See more instruction in https://github.com/google/jax#installation. 

### When Nvidia GPU is available
#### Cuda 12
```commandline
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
#### Cuda 11
```commandline
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


### CPU Only
```commandline
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

Note: In either case, the code can be run independently of the system. Jax will automatically detect if a GPU is 
available or not. If not, the code will run on CPU, but will be optmised and benefit from multi-core processing. 

### Comments on Performance
Using Jax, run time can be reduced by a factor of 10-20 if running in a CPU, or by a factor of 100-500 if running in a GPU!
However, more detailed benchmarks are required. If you would like to contribute, please contact us. 
![img.png](img.png)



