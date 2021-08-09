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
conda create env -n oml numpy  
conda activate oml  
conda install -c conda-forge gsw  
```  

You can then activate the oml environment and follow the installation instructions into an existing environment as above.

## Instructions for using the installed package  

Examples and short tests are given in notebook form in the tests folder.  Some examples require downloading the Argo profile database (see ftp://usgodae.org/pub/outgoing/argo/). Idealized profiles can be constructed for testing the interfaces as well, without need for obtaining external data.
