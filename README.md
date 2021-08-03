# oceanmixedlayers

This package includes varoius methods for computing the ocean surface mixed layer depth from temperature, salinity, and density fields.  

## Installation (existing environment)  

To install from source:  
```
pip install -e .
```

### Creating a clean conda install with minimal external packages  

This package utilizes numpy, matplotlib, xarray, jupyter, netcdf4, and gsw for full functionality.  To create a clean conda environment with these packages follow:  
```
conda create env -n oml numpy matplotlib xarray jupyter netcdf4  
conda activate oml  
conda install -c conda-forge gsw  
```  

You can then activate the oml environment and follow the installation instructions into an existing environment as above.

## Instructions for using the installed package  

Examples and short tests are given in notebook form in the tests folder.  Some examples require first downloading the Argo profile database (see ftp://usgodae.org/pub/outgoing/argo/). Idealized profiles can be constructed for testing the interfaces as well.
