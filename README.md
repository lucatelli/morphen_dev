```
      %&&&&&+   +&&&&&&&*+                                                                 
      #@@@@@+  *#@@@@@#                                                                 *+ 
     *@@#*@@% *@@@%%@@+   %#&&@#%   &##&&#&%  +#@#&&##%+ +&&    &@@* &###&%%%*  +%&+   &@% 
    +@@@% &@@%@@#+ #@%  %@@*   @@& +@@%  +@@%  @@&  +@@% &@%   +@@%  &@#+      +@@@&  *@#  
    %@@#  +@@@@#  %@@  %@@*    @@& &@@&&&#&%  %@@%*%@#% &@@##&&@@@  +@@#&&%*+  #@#&@& #@*  
   +@@@*   &@@#   #@%  *@@%   &@#++@@%**#@%   @@&**+    &@*   %@@%  &@@+      &@@  *@#@&   
    %@&    *@&   &#&    +&##&#&%  &#&   *@@% %##+      *#%    ##&  %####&%%%*%@@&   *##+   
            +    +                       +%@&%***                          +&@@&+          
                     ++**%%%%%%%***++      +***+                         +%@@#%            
             +*%%&#@@@#&&%%%%%%%%&&#@@###@@@@@@#*                      *&@@#*              
         +*&#@@@#&%*+                *@@@@#&&@@@@                  +%#@@#%+                
      +&#@@@&*+                      #@@@*+  #@@@&%*++    +++**%&#@@#&*                    
    +&@@@&*                          &@@@@@@@@@@%%&####@@@@@##&&%*+                        
   %@@@%+                             +%&&&##&%+                                           
  %@@#+                                                                                    
  %&+                                                                                   
```


# `morphen`: Overview, features and limitations
***Readme file under development!***
## What is `morphen`?
`morphen` is a collection of python-based astronomical functionalities for image analysis and 
processing. The three main functionalities of `morphen` are:
1. Image Analysis ([image_analysis/README.md](image_analysis/README.md))
2. Multi-Sersic Image Fitting Decomposition ([image_decomposition/README.md](image_decomposition/README.md))
3. Radio Interferometric Self-calibration ([selfcal/README.md](selfcal/README.md))



You will be able to measure basic image morphology and photometry. `morphen` also comes with a 
state-of-the-art python-based image fitting implementation based on the Sersic 
function. Particularly to radio astronomy, these tools involve pure python, but also are 
integrated with CASA (https://casa.nrao.edu/) in order to work with common `casatasks` as well 
`WSClean` -- a code for fast interferometric imaging (https://wsclean.readthedocs.io/en/latest/).

## Getting Started
Some specifics of what you can do with `morphen` includes:
- Perform morphological analysis of astronomical images (general.)
- Basic source extraction and photometry (general).
- Perform a multi-component Sersic image decomposition to astronomical images of galaxies (general).
- Perform self-calibration and imaging with `WSClean` and `CASA` (radio astronomy).
- Use information from distinct interferometric arrays to perform a joint separation of distinct 
  physical mechanisms of the radio emission (radio astronomy).
- Experimental: some functionalities are applicable to general astronomical data, but more 
  testing is required beyond radio astronomy.

While in development, these modules will be kept in the same place. Stable releases will be 
provided for the full module. However, we plan to release these separated functionalities 
as standalone repositories in the near future.


Currently, there is no option to install `morphen` (via `pip` or `conda`). 
However, its usage is simple. The code can be used as a module, interactively via Jupyter notebooks,
or via the command line interface (see "Important notes" below). For now, we recommend  
using it via Jupyter notebooks (see below for examples). 


The modular file `morphen.py` is the on-development module that allows you 
to do such tasks, like a normal package installed via `pip`. For that, need to download the 
entire repository. The `libs` directory, specifically 
the `libs/libs.py` file, contains the core functionalities in which `morphen.py` is based.
Examples can be found in the following directories: 
- `notebooks/`: contains some more general examples of how to use the code.
- `image_analysis/`: contains examples of how to use the image analysis functionalities.
- `image_decomposition/`: contains examples of how to use the Sersic image decomposition 
  functionalities.

### Important notes
1. The functionalities presented in the examples notebooks are stable. We are in extensive 
   development, and we are setting milestones for optimizations, bug fixes, better 
   documentation, and new functionalities for a larger scope of the code. 
2. The command line option is still under development and not all argument options are 
   available. However, using it via jupyter is somehow stable (check the notebooks for examples). 
3. This readme file is under development. I am also currently adding more basic usages 
to Jupyter notebooks guides. 
4. Installation instructions for all the dependencies are provided in the `install_instructions.md` 
file.


## Features
### Image Analysis
In the directory [```image_analysis/```](image_analysis/), the notebook 
[```morphen.ipynb```](image_analysis/morphen.ipynb) contain sets examples of how to perform 
basic image analysis, such as image statistics, photometry,
shape analysis, etc. Check also [```image_analysis/README.md```](image_analysis/README.md) file for more details.

[//]: # (Collective direct results from this code are published here: `<<ADD LINK>>`.)





### Image Fitting Decomposition

We introduce a Python-based image fitting implementation using the Sersic function.
This implementation is designed to be robust, fast with GPU acceleration using 
JaX (https://jax.readthedocs.io/en/latest/index.html) and easy to use 
(semi-automated). 
The physical motivation behind this implementation is to provide an interface to easily perform a 
multi-component decomposition constrained around prior knowledge from the data itself, without 
the need of creating complicated configuration files to set model parameters. 
This helps mitigate issues when trying to fit multiple-component models to the data. 
Prior photometry is measured from the data using the `PetroFit` code 
(https://petrofit.readthedocs.io/en/latest/index.html) and the 
`photutils` package (https://photutils.readthedocs.io/en/stable/) and used as initial 
constraints for the minimisation.
Examples of how to use it can be found in the Notebook 
[```image_decomposition/morphen_sersic.ipynb```](image_decomposition/morphen_sersic.ipynb)


The decomposition was first designed for radio interferometric images, but can be used with any 
other type of images, such as optical images. However, application to optical data is still a work in 
progress as we require better PSF modeling, especially for HST and JWST observations. 
For now, you already can check some basic examples in the notebook
[```image_decomposition/morphen_sersic_optical.ipynb```](image_decomposition/morphen_sersic_optical.ipynb)


[//]: # (More details can be found in the )

[//]: # ([```image_decomposition/README.md```]&#40;image_decomposition/README.md&#41; file.)




[//]: # (Parameter preparation )

[//]: # (for minimisation is fully-automated, but the user has to define the number of model )

[//]: # (components to be fitted to the data.)

[//]: # (It uses the LMFIT package )

[//]: # (with a GPU optmisation layer &#40;Jax&#41;. )



[//]: # (It uses the LMFIT package with an )
[//]: # (object-oriented implementation, easy to use and manageable number of n-components. )

### Radio Interferometric Related Tasks



#### Interferometric Imaging With `wsclean`
Directory [```imaging/```](imaging/) contains a python script called 
[```imaging/imaging_with_wsclean_v3.py```](imaging/imaging_with_wsclean_v3.py) which is just a support code 
for easy use to call wsclean on the command line. See the intructions file of 
how to use it: [```imaging/wsclean_imaging.md```](imaging/wsclean_imaging.md)



#### Imaging with `wsclean` and self-calibration

File [```selfcal/imaging_with_wsclean.py```](selfcal/imaging_with_wsclean.py) is a wrapper
to call `wsclean` on the command line, with pre-defined parameters already set. You can 
use it to perform imaging with `wsclean` in a simple way and change parameters as 
required. Note that not all `WSClean` arguments are available in this wrapper.
Arguments that are not implemented can be simply passed with the argument 
`--opt_args` in `imaging_with_wsclean.py`. This script is standalone and can be downloaded 
and used separately from the `morphen` package.


In previous versions of this module (not available in this repo), all self-calibration 
routines were done with CASA. However, some changes were made and in this repo we 
provide for the first time an automated way to perform self-calibration, which uses `WSClean` as 
imager and `CASA` to compute the complex gain corrections (phases and amplitudes). 

To check how to use it, see the 
[```selfcal/README.md```](selfcal/README.md) file and examples in 
[```selfcal/selfcalibration.ipynb```](selfcal/selfcalibration.ipynb).
This self-calibration pipeline was tested in multiple datasets with the VLA from 1.4 GHz to 33 
GHz and with e-MERLIN at 5 GHz, for a wide range of sources total flux densities.

[//]: # (In about 50% of the cases, the pipeline was able to converge to a good solution, for the other )

[//]: # (cases, after further inspection, good solutions performing a second run of the pipeline. )

The file [```selfcal/auto_selfcal_wsclean.py```](selfcal/auto_selfcal_wsclean.py) 
is a script to perform self-calibration with `wsclean` and `CASA`. Is fully automated, 
but is still in development. Check the 
[```selfcal/README.md```](selfcal/README.md) file for more details.

#### Selfcalibration and Imaging with `CASA`.
(DOC NOT READY)

#### Interferometric Decomposition
Interferometric decomposition is a technique introduced by Lucatelli et al. (2024) to 
disentangle the radio emission using combined images from distinct interferometric arrays.

***More details will be provided soon.***

#### CASA Utilities
(IN DEV)

### Origin of `morphen`
The idea of `morphen` predates back to 2018 alongside the development of 
`morfometryka` (https://iopscience.iop.org/article/10.1088/0004-637X/814/1/55/pdf) and 
$$\kappa$$urvature (https://academic.oup.com/mnras/article/489/1/1161/5543965). The aim was to 
expand the functionalities of `morfometryka` (such as automated bulge-disk decomposition) and some optimisations.
Development was on pause, but soon after I started working with radio astronomy, it was 
clear that we needed a set of automated tools for radio interferometric data processing and analysis,
from basic plotting to more complicated tasks, such as self-calibration and a robust image 
decomposition. 


Alongside, it was also clear that the reproducibility in radio astronomy is a challenge, and 
we were in need of a package towards reproducibility of scientific results. Hence, the ideas of 
`morphen` were brought back to be incorporated within radio astronomy.

## How to contribute
This is an open-source project. We are welcoming all kinds of contributions, suggestions and bug 
reports. Feel free to open an issue or contact us. 
