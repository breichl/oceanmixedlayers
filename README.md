# oceanmixedlayers

This package includes varoius methods for computing the ocean surface mixed layer depth from temperature, salinity, and density fields.  

## Installation (existing environment)  

To install from source:  
```
pip install -e .
```

### Creating a clean conda install with minimal external packages  

This package utilizes numpy and gsw for functionality of the mld software.  Notebooks and other tests may require additional packages, such as xarray and matplotlib.  To create a clean conda environment with the minimum required packages follow:  
```
conda create -n oceanmixedlayers numpy  
conda activate oceanmixedlayers  
conda install -c conda-forge gsw  
```  

You can then activate the oml environment and follow the installation instructions into an existing environment as above.  


### Optional additional packages to use all scripts and notebooks:  

```
conda install matplotlib jupyter netcdf4 xarray ipykernel  
```

A reminder, Jupyter environments can be added with:  
```
python -m ipykernel install --user --name myenv --display-name "oceanmixedlayers"  
```

## Instructions for using the installed package  

Examples and short tests are given in notebook form in the tests folder.  Some examples require downloading the Argo profile database (see ftp://usgodae.org/pub/outgoing/argo/). Idealized profiles can be constructed for testing the interfaces as well, without need for obtaining external data.  

The most useful example for usual implementation is probably in tests/Argo_Examples, which takes the Argo profiles and computes the gridded data for each of the algorithms included here.  
