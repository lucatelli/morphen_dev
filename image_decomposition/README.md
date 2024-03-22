# Image Fitting Decomposition

## Introduction
The `morphen` package provides a set of tools to perform image fitting decomposition, 
which uses the Sérsic function to model the surface brightness distribution of a source. 
The Sérsic modeling within `morphen` is semi-automated and designed to be time-effective. 
For cases where multiple model components have to be used, the code offers an interface 
to construct the model components in a semi-automated way. The user can inspect the results and 
repeat the process, if required, in a fashionable way.

There are other existing packages that perform image decomposition, such as GALFIT, 
and IMFIT. The largest differences between these two codes and the set of tools within `morphen` 
are:
1. Physical constraints: The initial condition of each model parameters is physically constrained 
   from the data itself,and not arbitrary. These prior information are determined from a source 
   extraction run in the 
   data, which includes also deblending. The location of these sub-regions and their basic 
   properties (peak position and intensity, effective radius, orientation, and axis ratio) are 
   used to construct a list of model components. The number of detected regions is used specify 
   the total number of model components to be fitted to the data. The code is flexible in the 
   sence that the user can specify if a certain parameter must be fixed or not, and its range. 
   There is no need to create configuration files, a dictionary of properties is generated on 
   the fly. This turns the process easy to analyse, and re-run the minimisation if required, 
   even if the number of components is changed.
2. Easy-to-Use: In interactive mode, the interface allows the user to inspect the source detected 
   regions. In general, some structures are simple, usually well modelled by a single Sersic function, 
   however, other may not. During the inspection, the user can easily identify those more 
   complicated structures. From that, it can be specified which detected structure requires 
   more than one model component to be fitted. This is easily done via the Jupyter Notebook, see 
   a comprehensive example in
   [```image_decomposition/morphen_sersic.ipynb```](image_decomposition/morphen_sersic.ipynb)
3. Speed: By default, the minimisation uses scipy from the JaX package, with some decorated 
   functions for convolution. JaX allows processes to run on Nvidia GPUs (through cuda), which 
   improves 
   runtime very significantly for convolution operations. Furthermore, if no GPU is present, 
   JaX can still benefit from parallel processing, using the same decorated function. In either 
   case, nothing needs to be done, the selection of which device is determined by how you have 
   installed JaX (https://jax.readthedocs.io/en/latest/installation.html). Note that GPU 
   acceleration for Sersic modeling is not available in other traditional packages.
4. Limitations: The fitting functionality of `morphen` is based only on a combination of Sersic 
   functions, hence more complicated models are not available yet. 


## Current Stage 
### Limitations
1. The code is not designed to fit multiple sources at once, for example, a large 
   image containing multiple sources. You must pre-process the data beforehand, ideally cutouts 
   with all the emission withing the image size and cleaned from any other sources (e.g. 
   background or foreground objects).
2. Source detection: the source detections uses the SEP package (detection and deblending). The 
   default parameters specified in the function `source_extraction` should work reasonably well. 
   However, it is know that detection and deblending are not perfect, and the user may need to 
   inspect each detection run to ensure that unwanted regions are added to the minimisation.
3. The minimisation is reasonably fast for a model with up to 5 components, and it becomes 
   slower as the number of components increases. The limitation is due to how parameters of each 
   component are passed into LMFIT, which is not optimised for large number of components and 
   LFMIT functions are not implemented in JaX. 
4. The current version (0.3) was not tested extensively in other wavelengths other than radio, 
   such as optical or infrared. The limitations are imposed exclusively by the PSF modelling, 
   which depends on the PSF structure itself. Radio PSFs are predominantly Gaussian, which makes 
   the PSF simpler. For optical and infrared images, we are working on a (soon) future update to 
   account for such complexities of the PSF. Some working examples using ground-based telescopes 
   can be found in [```image_decomposition/morphen_sersic_optical.ipynb```](image_decomposition/morphen_sersic_optical.ipynb).
5. In traditional codes they offer a wide range of mathematical functions to model the 
   surface brightness distribution of a source. The current version of `morphen` only offers the 
   Sérsic function. Therefore, it will not be possible to model structures whith, for 
   example, ring morphologies or truncated profiles. The implementation of other functions is not planned yet. 
6. PSF modelling for optical observations is at a very early stage, so it is not optimal yet.
7. Background estimation for optical observations requires more work. Currently it uses 
   SEP (or `photutils`) to estimate the background, with a large filter (e.g. the bkg will be 
   almost flat). The final bkg map used is just a randomised version of that map. 

## Known Issues
### Major
- `LMFIT` versions above `1.1.0`: When using any version of the `LMFIT` package above `1.1.0`, 
  the minimisation does not work. The issue is still under investigation. We then strictly 
  recommend using version `1.1.0` of `LMFIT` until further notice.

### Minor
- Axes units in offset mode can be wrong when using pixel units as the projection.
- Integrated fluxes are not correct when using optical data, as the current version only sums up 
  the pixel values.
- sub-component model images and residual images generated after fit contain discontinuities  if 
  a mask was used for fit. This is due to the mask being applied to the model image and residual

[//]: # (### 1.2 Current Stage )

[//]: # (The current stage of the Sersic image decomposition functionality of `morphen` is: )

[//]: # (***semi-automated***. )

[//]: # (The manual portion of the code involves the following steps, which can be performed via a )

[//]: # (Jupyter Notebook interface:)

[//]: # (   1. An inspection to define the number of total model components to be fitted to the )

[//]: # (      imaging data.)

[//]: # (   2. Set basic constraint properties, for example, if a specific parameter should be )

[//]: # (      fixed or not.)



## Code Core Functionalities and Usage
### Getting started
As a basic guide, we refer to the following Jupyter Notebooks:
[```image_decomposition/morphen_sersic.ipynb```](image_decomposition/morphen_sersic.ipynb) for 
radio images and [```image_decomposition/morphen_sersic_optical.ipynb```](image_decomposition/morphen_sersic_optical.ipynb)
for optical images. These contains the basic functionalities that are explained below. 

In this tutorial, we are going to import the modules in the following way:

```python
# asume that we are in the directory `morphen/image_decomposition/`
sys.path.append('../libs/')
sys.path.append('../')
import morphen as mp
import libs as mlibs
```

The image fitting implementation consists of multiple steps, from loading the data, source 
detection and the fitting part itself.

We first start by loading the data via the `mp.read_data` function. Typical examples are 
provided below.

For radio images:
```python
root_path = '../../data_examples/data_examples_fitting/vla_only/UGC5101_X/'
prefix_images = '*MFS-image.fits'
imagelist = mlibs.glob.glob(root_path+prefix_images) 
input_data=mp.read_data(filename=imagelist[0],
                        residualname=imagelist[0].replace('MFS-image.fits','MFS-residual.fits'))
```
Note that we do not require to provide the PSF image, as it will be computed automatically 
from the restoring beam of the image. It is also mandatory to provide the residual image, i.e., 
the residual map generated during interferometric deconvolution. We have assumed in this example 
that images were generated by `WSClean` and both image and residual are in the same directory.
If you have `CASA` generated images, please convert those to `FITS` using the `CASA` task 
`exportfits`.

For optical images:
```python
root_path = '../../data_examples_dev/optical/efigi/lenticulars/showcase/'
psf_root_path = '../../data_examples_dev/optical/efigi/'
imagename = root_path + 'PGC0060343_r_seg.fits' #use this for the showcase
psf_name = psf_root_path+'psf_efigi_s13.fits'
input_data=mp.read_data(filename=imagename,
                        psfname=psf_name)

```
In this case, we require to provide the PSF image.

### Source Detection
The source detection is under development, but it is based on the `SEP` and/or `PetroFit/photutils` 
packages. In principle any algorithm can be adapted into `morphen`. The complete update documentation showing 
how to do that will be provided soon. The source detection is called via the `source_extraction` 
and the package to be used is specified via the argument `algorithm` (e.g. `SEP` or `PF`); 
default is `SEP`.

Source detection is crucial for the fitting minimisation to work well. It consists 
in finding the relevant regions of emission so that basic properties are computed 
from them. Currently, critical properties that are computed prior to the minimisation are:
- The position `(x0,y0)` of each relevant structure. 
- Its effective circular radius (or half-to-total radius) `R50`, which is the region that 
  encloses half of the total flux/luminosity of the structure. 
- Its orientation `PA` and elongation `q = R50/R50b`, where`R50b` is the half-to-total radius 
  perpendicular to `R50` (or semi-minor axis); 

These properties are used as initial conditions and constraints during the minimisation for 
every model component. 

In the source detection step, each detected structure is assigned a unique ID. An example 
is shown in Figure 1 below.

<div align="center">

![img_3.png](img_3.png)

*Figure 1: Source detection exemplification.*

</div>

However, the code still does not recognise in which case a detected structure is single or 
multi-component. Typical examples are:
1. Radio observations: a radio emission with a core-compact structure surrounded by extended 
   emission. The entire emission will be labelled by a unique ID.
2. Optical: in a similar way, i) a lenticular galaxy, containing a bulge and a disk and ii) a 
   spiral/lenticular galaxy containing a bulge, a bar and an extended disk/spiral arms. 

We require to identify which detected structure is multi-component, since
those structures will require more than one Sersic function (model 
component) to be fitted to the data. Currently, this step is not automated, and it has
to be done manually by the user (using the Jupyter Notebook interface). Therefore, the manual 
portion of the code is to define which detected structure requires more than one model 
component to be fitted to the data. 

***In a future version, the determination of when a structure ID is multi-component 
or not this will be done automatically.*** 

In a typical run, one can invoke the function `mp.source_extraction` in dry run mode 
(`dry_run=True`) so that only the regions will be displayed, without any photometry performed. 
Then, if the detection was good, call the function again with `dry_run=False` to perform the 
photometry and obtain basic properties. Such quantities are stored in an object `SE`, e.g. 
````python
SE = mp.source_extraction(*args,dry_run=False)
````
The object `SE` will be passed to the minimisation functions `mp.sersic_multifit_radio` (for 
radio images) or `mp.sersic_multifit_general` (for optical images). 

Note that `mp.source_extraction` is used in either case (radio or optical images).

### Background estimation
***TO-DO***
Providing a rms bkg map is not mandatory as it will be computed automatically if not provided. 
However, the code accepts a user-provided rms bkg map, for example `bkg_rms_map`.

### Running the minimisation
With the `input_data` and `SE` objects, the minimisation is called with the function

#### Radio Images 
```python
smfr = mp.sersic_multifit_radio(input_data,
                                SE, #source extraction object, from previous step
                                convolution_mode='GPU',
                                which_residual = 'shuffled', #natural or shuffled (natural is experimental!)
                                # bkg_rms_map = sep_bkg.back(),
                                fix_geometry=True, #for stability purposes, keep True for now. 
                                comp_ids=['1'],# which component label is compact?
                                dr_fix=[3,50],#for each component, radial element size to fix (x0,y0) positions
                                fix_value_n=[0.5,0.5],#for each component, the Sersic index value to be fixed. 
                                fix_n=[True,True],#for each component, fix or not the Sersic index. 
                                aspect='elliptical',#elliptical or circular gaussian for beam convolution? 
                                z = mlibs.find_z_NED('UGC5101'))
```

#### Optical Images 
```python
smfg = mp.sersic_multifit_general(input_data,
                                SE, #source extraction object, from previous step
                                convolution_mode='GPU',self_bkg=True,
                                which_residual = 'user',
                                bkg_rms_map = bkg_rms_map,
                                tr_solver='exact', #'lsmr or exact'
                                fix_geometry=True, #for stability purposes, keep True for now. 
                                comp_ids=['1'],# which component label is compact/bulge?
                                dr_fix=[5,30,300],#for each component, radial element size to fix (x0,y0) positions
                                fix_value_n=[1.0,0.5,1.0],#for each component, the Sersic index value to be fixed. 
                                fix_n=[False,False,False],#for each component, fix or not the Sersic index. 
                                z = 0.1 #just an arbitrary value for now.
                                )
```

## 3 Basic Examples
The fitting approach 

### Source Detection
The source detection is under development, but it is based on the `SEP` package.
Other experiments are being made to use `PetroFit` (specifically `Photutils`), but in 
principle any algorithm can be adapted into `morphen`. 

Structure detection is crucial for the fitting minimisation to work well. It consists 
in finding the relevant regions of radio emission so that basic properties are computed 
from it, such as the position `(x0,y0)`, orientation `PA`, the half-to-total radius 
`R50`, and the axis ratio `q = R50/R50b`. These properties are used as initial 
conditions and constraints during the minimisation.

Each detected structure is assigned a unique ID. However, the code still does not 
recognise in which case a detected structure is single or multi-component, for 
example: i) a radio emission with a core-compact structure plus extended emission 
around it; ii) or a lenticular galaxy, containing a bulge and a disk. From that, the  
manual portion of the code is to define which detected structure requires more than 
one model component to be fitted to the data. 

***In a future version this will be done automatically.*** 

### Summary of the Method
As a guide, a brief summary involving the manual steps is given below:
 - After the source extraction is performed, some basic properties are computed over 
   the detected components, such as Petrosian radius, position angle, axis ratio and 
   the half-to-total radii, and the intensity at the half-to-total radii. These 
   quantities are used as initial conditions and constraints during the minimisation. 
   Therefore, you have to manually specify which detected structure requires more than 
   one model component to be fitted. This is easily done via the Jupyter Notebook 
   interface, which turns the process easy to analyse the minimisation results and 
   outputs, and if required, repeat the process.
 - By default, the code uses the source detrection positions as initial conditions 
   for the model components to be minimised, and also constrains the variation of 
   such parameters to a small range (e.g. `dr_fix ~ 5-10`). If you require two
   or more model components to fit a specific structure, but the structure is very 
   asymmetric, you may want to increase the range that (`x0,y0`) can vary, e.g.  
   `dr_fix ~ 50`.  In the same way as before, you may want to fix the model coordinate 
   positions, and adjust the ranges as required, using `fix_x0_y0=[True,True]` and  
   `dr_fix = [5,50]`. It is not advised to set `fix_x0_y0 = [False, False]` because 
   it is pointless, the source detection already is providing good initial values for 
   these parameters, and there is no need to search a parameter space far away from 
   the detection values, which may cause long run time scenarios.
 - Choose if the Sersic Index of the model components are free or fixed; if fixed, 
   which value it should be fixed to. For clarification, core-compact 
   radio structures are very well modelled by Gaussian functions, e.g. sersic index 
   `n=0.5` and  diffuse radio structures are well modelled by a disk-exponential 
   distribution, e.g. a Sersic index `n=1.0`, but sometimes are also well modelled
   by a Gaussian function. So, you may want to inspect which one represent best your 
   data. More details are provided in the Example section bellow. To handle how this  
   parameter is fixed or not, and to which value, you can use a list. For example,  if 
   you are modelling the emission with 2 model components (`COMP_1`, `COMP_2`), 
   and you would like to fix the `n` of `COMP_1` to `n=0.5` and keep the one for 
   `COMP_2` free, you can provide `fix_n=[True,False]` alongside `fix_value_n = [0.5,1.
   0]`. By default, the last element of the list `fix_value_n` will be ignored because 
   its associated element in `fix_n` is `False`. The list must contain the same shape as 
   the number of total model components to be fitted to the data. By default, the 
   code will set `fix_n` to `True` for all detected structures `ID_*` and 
   `fix_value_n` to `0.5` similarly. 

*The morfometryka algorithm is not public available yet, and it is under development. 
In the near future, functions from `morphen` will migrate to `morfometryka` and vice 
versa. 

### Additional Features
- Which geometry to be used during optimisation, standard elliptical geometry or 
  generalised elliptical geometry. This can be set with `fix_geometry=False` or 
  `fix_geometry=True` (see Eq. 11 of the paper). By default, the code uses the standard 
  elliptical geometry (i.e. C=0).


## Examples
### Fitting Radio Data

% As described in Sec.~\ref{sec:image_fitting}, to construct the list of model 
components before the minimisation, we have used a source extraction algorithm 
(\textsc{SEP}\footnote{\url{https://github.com/kbarbary/sep}}) and a Petrosian 
analysis package (\textsc{PetroFit}) to estimate the initial conditions for 
the $50\%$ flux radii (i.e. $R_n$ in Eq. \ref{eq:sersic_law}) and associated 
$50\%$ contour levels at each component ($I_n$) as well the coordinates and 
orientation of each component.  To maximise run-time efficiency and minimise 
fitting issues due to the complexity of the problem, the coordinates $(x_0,y_0)$ 
of the components are kept almost fixed to their detection positions, with a 
free interval of $+/-$5 pixels. Note that this is not true for the larger 
scale components (e.g. \texttt{COMP\_4} and \texttt{COMP\_5}), where, despite 
their initial conditions being close to companion components, larger bounds are 
provided. This has proved effective in reducing minimisation issues. 


### Fitting Optical Data
A basic example of the code's usage is presented in this notebook: [```image_sersic_decomposition.ipynb```](imaging/imaging_with_wsclean_v3.py) 

### Source Detection
![img_3.png](img_3.png)

### Running the minimisation
The minimisation is called with the function `do_fit2D` providing the required inputs: 

```python
## We use 2 solver methods for better convergence. 
save_name_append='_ls_n110D'
result_mini, mini,result_1,result_extra,model_dict, \
image_results_conv,image_results_deconv, \
                    smodel2D, model_temp = do_fit2D(imagename=crop_image,residualname=None,
                                                   init_constraints=sources_photometies_new,psf_name=psf_name,
                                                   params_values_init = None,#imfit_conf_values[0:-1],
#                                                    fix_n = False,fix_x0_y0=[False,False,False],
                                                   ncomponents=n_components_new,constrained=True,self_bkg=True,rms_map=rms_map,
                                                   fix_n=[False,False,False,False,False,True,False],
#                                                     mask_region=mask_component,
                                                   fix_value_n = [3.0,1.0,1.0,1.0,1.0,1.0,1.0],
                                                    fix_x0_y0=[True,True,True,True,True,True,True],
                                                    dr_fix = [3.5,3.5,3.5,3.5,3.5,7.5,7.5],
                                                   # n_to_fix = [0.5,None,1.0,None]
                                                   convolution_mode='GPU',fix_geometry=False,workers=6,
                                                   #do not fix the n of the extra component
                                                   method1 = 'least_squares',method2 = 'least_squares',
                                                   init_params = 0.2,final_params = 5.0,loss='cauchy',tr_solver='exact',
                                                   save_name_append=save_name_append)
```

![img_1.png](img_1.png)
![img_2.png](img_2.png)

# Limitations With Radio Images
Radio images require us that we create associated PSF images with the same 
size as the radio image. For larger images, performance issues may arrise due 
to the convolution operations between images and PSFs with large sizes. 
That it is why is wise to perform cutouts of radio 
maps accordingly in order to reduce associated sizes.



## Notes on performance
Both `scipy.optimize` and `LMFIT` minimiser are single-core, in exception when using `JaX`'s 
`Scipy`. Even with that, the run-time is comparable to `GALFIT` and `IMFIT` which contain 
multi-thread processing. That said, one would run `morphen` fitting routine on multiple data sets.
That will speed up image decomposition by a significant factor.



## Dependencies
The image decomposition within `morphen` depends on some astrophysical packages such 
as `astropy`, `petrofit`, `sep`, `photutils`, `morfometryka_core`* and optmisation 
algorithms, such as `LMFIT` (a wrapper of `scipy`) and `Jax` -- a computing layer for  
multi-thread CPU and GPU processing. If you have GPU, it benefits from both the CPU 
and GPU.
<br>

***WARNING: Many functions from `scipy` are not implemented in `jax.scipy` so, some 
of them (e.g. convolution operations) were made manually with in-built Jax 
functions and they may not be stable/optimised. As of now, every week new functions
from `scipy` are implemented in `jax.scipy`, so some changes are expected in the near  
future.*** 

[//]: # ()
[//]: # ()
[//]: # (## Documentation)

[//]: # ( - [```sersic.md```]&#40;sersic.md&#41;)

[//]: # ( - )