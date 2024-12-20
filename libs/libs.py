"""
                                                          ..___|**_
                                                  .|||||||||*+@+*__*++.
                                              _||||.           .*+;].,#_
                                         _|||*_                _    .@@@#@.
                                   _|||||_               .@##@#| _||_
       Morphen                |****_                   .@.,/\..@_.
                             #///#+++*|    .       .@@@;#.,.\@.
                              .||__|**|||||*||*+@#];_.  ;,;_
 Geferson Lucatelli                            +\*_.__|**#
                                              |..      .]]
                                               ;@       @.*.
                                                #|       _;]];|.
                                                 ]_          _+;]@.
                                                 _/_             |]\|    .  _
                                              ...._@* __ .....     ]]+ ..   _
                                                  .. .       . .. .|.|_ ..

"""
__versions__ = ('0.3.1alpha-1', '0.4.0alpha-1', '')
__codenames__ = ('Pelicoto', 'Saurinho', '')
__version__ = '0.4.0alpha-1'
__codename__ = 'Saurinho'
__author__ = 'Geferson Lucatelli'
__email__ = 'geferson.lucatelli@postgrad.manchester.ac.uk; gefersonlucatelli@gmail.com'
__date__ = '2024 11 13'
print(__doc__)
print('Version',__version__, '('+__codename__+')')
print('By',__author__)
print('Date',__date__)
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.text import Text
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
from matplotlib import rcParams


import numpy as np
from sympy import *
import casatasks
from casatasks import *
import casatools
# from casatools import *
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import astropy.io.fits as pf
from casatools import image as IA

import lmfit
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
import emcee
from prettytable import PrettyTable

import string
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl_
import glob
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import sep
import fitsio

from astropy.stats import mad_std
from scipy.ndimage import gaussian_filter
from astropy import visualization
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from skimage.measure import perimeter_crofton
from scipy import ndimage
from scipy.ndimage import morphology
from scipy.ndimage import shift
from scipy.special import huber
from skimage.morphology import disk, square
from skimage.morphology import dilation

from scipy.optimize import leastsq, fmin, curve_fit
import scipy.ndimage as nd
import scipy
from scipy.stats import circmean, circstd
from scipy.signal import savgol_filter

from astropy.cosmology import FlatLambdaCDM
import numpy as np
from astropy import units as u
from astropy import coordinates
import pandas as pd
import sys
import pickle
import time
import corner
import re
from tqdm import tqdm
import itertools

import dynesty
from dynesty import plotting as dyplot
import corner



import numpy as np
from scipy import ndimage
from sklearn.neighbors import KNeighborsClassifier


# import pymc3 as pm
# try:
from petrofit import make_radius_list
from petrofit import Petrosian
from petrofit import source_photometry
from petrofit import make_catalog, plot_segments
from petrofit import plot_segment_residual
from petrofit import order_cat
# except:
#     pass
import copy
# from copy import copy
import astropy.io.fits as fits
import matplotlib.ticker as mticker
import coloredlogs
import logging
import warnings
warnings.filterwarnings('ignore', module='photutils')
warnings.filterwarnings('ignore', module='astropy')

from functools import partial
try:
    import jax
    from jax import jit, vmap
    from jax.numpy.fft import fft2, ifft2, fftshift
    import jax.numpy as jnp
    import jax.scipy as jscipy
except:
    print('Jax was not imported/installed correctly, Sersic Fitting Will FAIL! ')
    print('Jax/GPU Libraries not imported.')
    pass
#setting the GPU memory fraction to be used of 25% should be fine!
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.25'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=6'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


# os.environ["NUM_CPUS"] = "12"  # Set the desired number of CPU threads here
# # os.environ["XLA_FLAGS"] = "--xla_cpu_threads=2"
os.environ["TF_XLA_FLAGS"] = "--xla_cpu_threads=8"  # Set the desired number of CPU
# # threads here
#
os.environ['MKL_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ["NUM_INTER_THREADS"]="8"
# os.environ["NUM_INTRA_THREADS"]="12"
# #
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=8")


# sys.path.append('../../scripts/analysis_scripts/')
sys.path.append('../analysis_scripts/')
from fitEllipse import main_test2

# import analysisUtils as au
"""
# from astroquery.ipac.ned import Ned
# result_table = Ned.query_object("MCG12-02-001")
# z = result_table['Redshift'].data.data
"""
#redshift for some sources.
z_d = {'VV705': 0.04019,
       'UGC5101':0.03937,
       'UGC8696':0.03734,
       'MCG12' : 0.015698,
       'VV250':0.03106}


# def print_logger_header(title, logger):
#     separator = "-" * len(title)
#     logger.info(separator)
#     logger.info(title)
#     logger.info(separator)

def print_logger_header(title, logger):
    width = len(title) + 4  # Add padding to the width
    top_bottom_border = "+" + "-" * width + "+"
    side_border = "| " + title + " |"

    logger.info(top_bottom_border)
    logger.info(side_border)
    logger.info(top_bottom_border)

"""
#Config
"""
def reset_rc_params():
    """
    Global configuration for matplotlib.pyplot
    """
    mpl.rcParams.update({'font.size': 16,
                         "text.usetex": False,  #
                         "font.family": "sans-serif",
                         'mathtext.fontset': 'stix',
                         "font.family": "sans",
                         'font.weight': 'medium',
                         'font.family': 'STIXGeneral',
                         'xtick.labelsize': 16,
                         'figure.figsize': (6, 4),
                         'ytick.labelsize': 16,
                         'axes.labelsize': 16,
                         'xtick.major.width': 1,
                         'ytick.major.width': 1,
                         'axes.linewidth': 1.5,
                         'axes.edgecolor': 'orange',
                         'lines.linewidth': 2,
                         'legend.fontsize': 14,
                         'grid.linestyle': '--',
                         # 'grid.color':'black',
                         'axes.grid.which': 'major',
                         'axes.grid.axis': 'both',
                         'axes.spines.right': True,
                         'axes.grid': True,
                         'axes.titlesize': 16,
                         'legend.framealpha': 1.0
                         })
    pass


def deprecated(old_name, new_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(f"'{old_name}' is deprecated and "
                          f"will be removed in a future version. "
                          f"Use '{new_name}' instead.",
                          category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

"""
 __  __       _   _
|  \/  | __ _| |_| |__
| |\/| |/ _` | __| '_ \
| |  | | (_| | |_| | | |
|_|  |_|\__,_|\__|_| |_|

 _____                 _   _
|  ___|   _ _ __   ___| |_(_) ___  _ __  ___
| |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
|  _|| |_| | | | | (__| |_| | (_) | | | \__ \
|_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/

"""

def gaussian2D(x0, y0, a, fwhm, q, c, PA, size):
    """
    Creates a 2D gaussian model.

    Parameters
    ----------
    x0,y0 : float float
        center position in pixels
    a : float
        amplitude of the gaussian function, arbitrary units
        [0, inf]
    fwhm : float
        full width at half maximum of the gaussian, in pixels
        [0, inf]
    q : float
        axis ratio, q = b/a; e = 1 -q
        q in [0,1]
    c : float
        geometric parameter that controls how boxy the ellipse is
        c in [-2, 2]
    PA : float
        position angle in degrees of the meshgrid
        [-180, +180]
    size : tuple float
        size of the 2D image data array

    Returns
    -------
    numpt.ndarray 2D
        2D gaussian function image
    """
    # print(size)
    x, y = np.meshgrid(np.arange((size[1])), np.arange((size[0])))
    x, y = rotation(PA, x0, y0, x, y)
    r = (abs(x) ** (c + 2.0) + ((abs(y)) / (q)) ** (c + 2.0)) ** (1.0 / (c + 2.0))
    # mask = 1./np.sqrt(2.*np.pi*sigma**2.) * np.exp(-r2/(2.*sigma**2.))
    gaussian_2D_model = a * np.exp(-4 * (np.log(2)) * (r) / (fwhm ** 2.0))
    return (gaussian_2D_model)

def rotation(PA, x0, y0, x, y):
    """
    Rotate an input image array. It can be used to modify
    the position angle (PA).

    Params
    ------
        x0,y0: center position
        PA: position angle of the meshgrid
        x,y: meshgrid arrays
    Returns
    -------
        tuple float
            rotated meshgrid arrays
    """
    # gal_center = (x0+0.01,y0+0.01)
    x0 = x0 + 0.25
    y0 = y0 + 0.25
    # convert to radians
    t = (PA * np.pi) / 180.0
    return ((x - x0) * np.cos(t) + (y - y0) * np.sin(t),
            -(x - x0) * np.sin(t) + (y - y0) * np.cos(t))


def Rn_R0(R0,n):
    """
    Determine the effective radius from the scale radius and the sersic index.
    """
    return(R0 * (bn_cpu(n)**n))
    

def bn_cpu(n):
    """
    bn function from Cioti .... (1997);
    Used to define the relation between Rn (half-light radii) and total
    luminosity

    Parameters:
        n: sersic index
    """
    return 2. * n - 1. / 3. + 0 * ((4. / 405.) * n) + ((46. / 25515.) * n ** 2.0)

def sersic2D(xy, x0, y0, PA, ell, n, In, Rn,cg=0.0):
    """
    Parameters
    ----------
    xy : tuple float
        meshgrid arrays
    x0,y0 : float float
        center position in pixels
    PA : float
        position angle in degrees of the meshgrid
        [-180, +180]
    ell : float
        ellipticity, e = 1 - q
        ell in [0,1]
    n : float
        sersic index
        n in [0, inf]
   Rn : float
        half-light radius
        Rn in [0, inf]
    In : float
        intensity at Rn
        In in [0, inf]
    cg : float
        geometric parameter that controls how boxy the ellipse is
        c in [-2, 2]
    Returns
    -------
    model : 2D array
        2D sersic function image
    """
    q = 1 - ell
    x, y = xy
    # x,y   = np.meshgrid(np.arange((size[1])),np.arange((size[0])))
    xx, yy = rotation(PA, x0, y0, x, y)
    # r     = (abs(xx)**(c+2.0)+((abs(yy))/(q))**(c+2.0))**(1.0/(c+2.0))
    r = np.sqrt((abs(xx) ** (cg+2.0) + ((abs(yy)) / (q)) ** (cg+2.0)))
    model = In * np.exp(-bn_cpu(n) * ((r / (Rn)) ** (1.0 / n) - 1.))
    return (model)

def FlatSky_cpu(data_level, a):
    """
    Parameters
    ----------
    data_level : float
        data level, usually the std of the image.
        data_level in [0, inf]
    a : float
        flat sky level factor, to multiply the data_level.
        a in [0, inf]
    Returns
    -------
    float
        flat sky level

    """
    return (a * data_level)


# def break_tangent(r_brk,rs):
#     B = 2.65-4.98*(r_brk/(r_brk-rs))
#     return(B)
# def profile_break_tangent(r,r_brk,rs):
#     func = 0.5*(np.tanh( (2.0-break_tangent(r_brk,rs))*(r/r_brk)+break_tangent(r_brk,rs)
#                       ) +1 )
#     return(func)


def deconvolve_fft(image, psf):
    """
    Simple deconvolution in input array image. 
    
    CAUTION: This is just indented to simulate how a convolved residual map 
    would look like as deconvolved. It is not a real deconvolution.
    
    This was designed to provide a residual map to be used as input for the 
    Sersic fitting. Instead of providing the convolved residual map, it is 
    more correct to provide a deconvolved residual map.

    Parameters
    ----------
    image : 2D array
        Input image array.
    psf : 2D array
        Input psf array.
    Returns
    -------
    deconvolved_scaled : 2D array
        Deconvolved image array.
    deconvolved_norm : 2D array
        Deconvolved image array, normalised.

    
    """
    padded_shape = (image.shape[0] + psf.shape[0] - 1,
                    image.shape[1] + psf.shape[1] - 1)

    # Pad both image and psf to the new shape
    pad_shape = [(0, ts - s) for s, ts in zip(image.shape, padded_shape)]
    image_padded = np.pad(image, pad_shape, mode='constant')
    pad_shape = [(0, ts - s) for s, ts in zip(psf.shape, padded_shape)]
    psf_padded = np.pad(psf, pad_shape, mode='constant')
    
    
    image_fft = scipy.fftpack.fft2(image_padded)
    psf_fft = scipy.fftpack.fft2(psf_padded)
    deconvolved_fft_full = image_fft / psf_fft
    
    deconvolved_fft = deconvolved_fft_full[psf.shape[0] // 2:image.shape[0] + psf.shape[0] // 2,
            psf.shape[1] // 2:image.shape[1] + psf.shape[1] // 2]
    deconvolved = np.abs(scipy.fftpack.ifft2(deconvolved_fft))
    deconvolved_norm = deconvolved/np.sum(deconvolved)
    deconvolved_scaled = (image/np.mean(image)) * deconvolved_norm
    return deconvolved_scaled, deconvolved_norm

"""
 __  __       _   _
|  \/  | __ _| |_| |__
| |\/| |/ _` | __| '_ \
| |  | | (_| | |_| | | |
|_|  |_|\__,_|\__|_| |_|

 _____                 _   _
|  ___|   _ _ __   ___| |_(_) ___  _ __  ___
| |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
|  _|| |_| | | | | (__| |_| | (_) | | | \__ \
|_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


  ____ ____  _   _        _____             _     _          _ 
 / ___|  _ \| | | |      | ____|_ __   __ _| |__ | | ___  __| |
| |  _| |_) | | | |      |  _| | '_ \ / _` | '_ \| |/ _ \/ _` |
| |_| |  __/| |_| |      | |___| | | | (_| | |_) | |  __/ (_| |
 \____|_|    \___/       |_____|_| |_|\__,_|_.__/|_|\___|\__,_|

"""
try:
    @jit
    def bn(n):
        """
        bn function from Cioti .... (1997);
        Used to define the relation between Rn (half-light radii) and total
        luminosity

        Parameters:
            n: sersic index
        """
        return 2. * n - 1. / 3. + 0 * ((4. / 405.) * n) + ((46. / 25515.) * n ** 2.0)
except:
    def bn(n):
        """
        bn function from Cioti .... (1997);
        Used to define the relation between Rn (half-light radii) and total
        luminosity

        Parameters:
            n: sersic index
        """
        return 2. * n - 1. / 3. + 0 * ((4. / 405.) * n) + ((46. / 25515.) * n ** 2.0)

try:
    @jit
    def sersic2D_GPU(xy, x0=256, y0=256, PA=10, ell=0.9,
                    n=1.0, In=0.1, Rn=10.0, cg=0.0):
        """
        Using Jax >> 10x to 100x faster.

        Parameters
        ----------
        xy : tuple float
            meshgrid arrays
        x0,y0 : float float
            center position in pixels
        PA : float
            position angle in degrees of the meshgrid
            [-180, +180]
        ell : float
            ellipticity, e = 1 - q
            ell in [0,1]
        n : float
            sersic index
            n in [0, inf]
        Rn : float
            half-light radius
            Rn in [0, inf]
        In : float
            intensity at Rn
            In in [0, inf]
        cg : float
            geometric parameter that controls how boxy the ellipse is
            c in [-2, 2]
        Returns
        -------
        model : 2D Jax array
            2D sersic function image
        """

        q = 1 - ell
        x, y = xy

        xx, yy = rotation_GPU(PA, x0, y0, x, y)
        # r     = (abs(xx)**(c+2.0)+((abs(yy))/(q))**(c+2.0))**(1.0/(c+2.0))
        r = jnp.sqrt((abs(xx) ** (cg + 2.0) + ((abs(yy)) / (q)) ** (cg + 2.0)))
        model = In * jnp.exp(-bn(n) * ((r / (Rn)) ** (1.0 / n) - 1.))
        return (model)
except:
    def sersic2D_GPU(xy, x0=256, y0=256, PA=10, ell=0.9,
                    n=1.0, In=0.1, Rn=10.0, cg=0.0):
        """
        Using Jax >> 10x to 100x faster.

        Parameters
        ----------
        xy : tuple float
            meshgrid arrays
        x0,y0 : float float
            center position in pixels
        PA : float
            position angle in degrees of the meshgrid
            [-180, +180]
        ell : float
            ellipticity, e = 1 - q
            ell in [0,1]
        n : float
            sersic index
            n in [0, inf]
        Rn : float
            half-light radius
            Rn in [0, inf]
        In : float
            intensity at Rn
            In in [0, inf]
        cg : float
            geometric parameter that controls how boxy the ellipse is
            c in [-2, 2]
        Returns
        -------
        model : 2D Jax array
            2D sersic function image
        """

        q = 1 - ell
        x, y = xy

        xx, yy = rotation_GPU(PA, x0, y0, x, y)
        # r     = (abs(xx)**(c+2.0)+((abs(yy))/(q))**(c+2.0))**(1.0/(c+2.0))
        r = jnp.sqrt((abs(xx) ** (cg + 2.0) + ((abs(yy)) / (q)) ** (cg + 2.0)))
        model = In * jnp.exp(-bn(n) * ((r / (Rn)) ** (1.0 / n) - 1.))
        return (model)

def sersic2D_GPU_new(xy, params):
    """
    Using Jax >> 10x to 100x faster.

    Parameters
    ----------
    xy : tuple float
        meshgrid arrays
    x0,y0 : float float
        center position in pixels
    PA : float
        position angle in degrees of the meshgrid
        [-180, +180]
    ell : float
        ellipticity, e = 1 - q
        ell in [0,1]
    n : float
        sersic index
        n in [0, inf]
    Rn : float
        half-light radius
        Rn in [0, inf]
    In : float
        intensity at Rn
        In in [0, inf]
    cg : float
        geometric parameter that controls how boxy the ellipse is
        c in [-2, 2]
    Returns
    -------
    model : 2D Jax array
        2D sersic function image
    """
    print(params)
    # print(params.shape)
    x0, y0, PA, ell, n, In, Rn, cg = params
    q = 1 - ell
    x, y = xy

    xx, yy = rotation_GPU(PA, x0, y0, x, y)
    # r     = (abs(xx)**(c+2.0)+((abs(yy))/(q))**(c+2.0))**(1.0/(c+2.0))
    r = jnp.sqrt((abs(xx) ** (cg + 2.0) + ((abs(yy)) / (q)) ** (cg + 2.0)))
    model = In * jnp.exp(-bn(n) * ((r / (Rn)) ** (1.0 / n) - 1.))
    return (model)

try:
    @jit
    def rotation_GPU(PA, x0, y0, x, y):
        """
        Rotate an input image array. It can be used to modify
        the position angle (PA).

        Using Jax >> 10-100x faster.

        Params:
            x0,y0: center position
            PA: position angle of the meshgrid
            x,y: meshgrid arrays
        """
        # gal_center = (x0+0.01,y0+0.01)
        x0 = x0 + 0.25
        y0 = y0 + 0.25
        # convert to radians
        t = (PA * jnp.pi) / 180.0
        return ((x - x0) * jnp.cos(t) + (y - y0) * jnp.sin(t),
                -(x - x0) * jnp.sin(t) + (y - y0) * jnp.cos(t))

    @jit
    def FlatSky(background_data, a):
        """
        A simple model for the background.

        Parameters
        ----------
        background_data : 2D array
            Input background array.
        a : float
            flat sky level factor, to multiply the background_data.
        """
        return (a * background_data)

    @jit
    def _fftconvolve_jax(image, psf):
        """
        2D Image convolution using the analogue of scipy.signal.fftconvolve,
        but with Jax. This function is decorated to speed up things.
        """
        return jax.scipy.signal.fftconvolve(image, psf, mode='same')

except:
    def rotation_GPU(PA, x0, y0, x, y):
        """
        Rotate an input image array. It can be used to modify
        the position angle (PA).

        Using Jax >> 10-100x faster.

        Params:
            x0,y0: center position
            PA: position angle of the meshgrid
            x,y: meshgrid arrays
        """
        # gal_center = (x0+0.01,y0+0.01)
        x0 = x0 + 0.25
        y0 = y0 + 0.25
        # convert to radians
        t = (PA * jnp.pi) / 180.0
        return ((x - x0) * jnp.cos(t) + (y - y0) * jnp.sin(t),
                -(x - x0) * jnp.sin(t) + (y - y0) * jnp.cos(t))

    def FlatSky(background_data, a):
        """
        A simple model for the background.

        Parameters
        ----------
        background_data : 2D array
            Input background array.
        a : float
            flat sky level factor, to multiply the background_data.
        """
        return (a * background_data)

    def _fftconvolve_jax(image, psf):
        """
        2D Image convolution using the analogue of scipy.signal.fftconvolve,
        but with Jax. This function is decorated to speed up things.
        """
        return jax.scipy.signal.fftconvolve(image, psf, mode='same')

"""
 ____  _     _
|  _ \(_)___| |_ __ _ _ __   ___ ___  ___
| | | | / __| __/ _` | '_ \ / __/ _ \/ __|
| |_| | \__ \ || (_| | | | | (_|  __/\__ \
|____/|_|___/\__\__,_|_| |_|\___\___||___/
  ____                                     _                         
 / ___|   ___    ___   _ __ ___     ___   | |   ___     __ _   _   _ 
| |      / _ \  / __| | '_ ` _ \   / _ \  | |  / _ \   / _` | | | | |
| |___  | (_) | \__ \ | | | | | | | (_) | | | | (_) | | (_| | | |_| |
 \____|  \___/  |___/ |_| |_| |_|  \___/  |_|  \___/   \__, |  \__, |
                                                       |___/   |___/ 

"""



def luminosity_distance_cosmo(z,Om0=0.308):
    h = 67.8  # * (h1 + h2) / 2
    cosmo = FlatLambdaCDM(H0=h, Om0=Om0)
    D_L = cosmo.luminosity_distance(z=z).value
    # print('D_l = ', D_L)  # 946.9318492873492 Mpc
    return(D_L)

def comoving_distance_cosmo(z,Om0=0.308):
    h = 67.8  # * (h1 + h2) / 2
    cosmo = FlatLambdaCDM(H0=h, Om0=Om0)
    D_C = cosmo.comoving_distance(z=z).value
    # print('D_l = ', D_L)  # 946.9318492873492 Mpc
    return(D_C)

def angular_distance_cosmo(z, Om0=0.308):
    h = 67.8# * (h1 + h2) / 2
    cosmo = FlatLambdaCDM(H0=h, Om0=Om0)
    d_A = cosmo.angular_diameter_distance(z=z)
    # print('D_a = ', d_A)  # 946.9318492873492 Mpc
    return(d_A)

def arcsec_to_pc(z, cell_size, Om0=0.308):
    h = 67.8# * (h1 + h2) / 2
    cosmo = FlatLambdaCDM(H0=h, Om0=Om0)
    d_A = cosmo.angular_diameter_distance(z=z)
    # print('D_a = ', d_A)  # 946.9318492873492 Mpc
    theta = 1 * u.arcsec
    distance_pc = (theta * d_A).to(u.pc, u.dimensionless_angles())
    # unit is Mpc only now
    # print('Linear Distance = ', distance_pc)  # 3.384745689510495 Mpc
    return (distance_pc)

def pixsize_to_pc(z, cell_size, Om0=0.308):
    h = 67.8  # * (h1 + h2) / 2
    cosmo = FlatLambdaCDM(H0=h, Om0=Om0)
    d_A = cosmo.angular_diameter_distance(z=z)
    # print('D_a = ', d_A)  # 946.9318492873492 Mpc
    theta = cell_size * u.arcsec
    distance_pc = (theta * d_A).to(u.pc, u.dimensionless_angles())  # unit is Mpc only now

    # print('Linear Distance = ', distance_pc)  # 3.384745689510495 Mpc
    return (distance_pc.value)



def cosmo_stats(imagename,z,results=None):
    """
    Get beam shape info in physical units.
    """
    if results is None:
        results = {}
        results['#imagename'] = os.path.basename(imagename)
    pc_scale, bmaj, bmin, BA_pc = beam_physical_area(imagename, z=z)
    results['arcsec_to_pc'] = pc_scale.value
    results['bmaj_pc'] = bmaj.value
    results['bmin_pc'] = bmin.value
    results['BA_pc'] = BA_pc.value
    return(results)

def find_z_NED(source_name):
    """
    Find the redshift of a source (by name) from NED.

    Parameters
    ----------
    source_name : str
        Source name.
            Example: 'VV705'

    Returns
    -------
    redshift_NED : float, None
    """
    from astroquery.ipac.ned import Ned
    result_table = Ned.query_object(source_name)
    redshift_NED = result_table['Redshift'].data.data
    if redshift_NED.shape[0] == 0:
        return None
    else:
        return redshift_NED[0]


"""
 ___
|_ _|_ __ ___   __ _  __ _  ___
 | || '_ ` _ \ / _` |/ _` |/ _ \
 | || | | | | | (_| | (_| |  __/
|___|_| |_| |_|\__,_|\__, |\___|
                     |___/
  ___                       _   _
 / _ \ _ __   ___ _ __ __ _| |_(_) ___  _ __  ___
| | | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \/ __|
| |_| | |_) |  __/ | | (_| | |_| | (_) | | | \__ \
 \___/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|___/
      |_|
"""


def circle_area(r):
    """
    Return the pixel area of a circle given the radius.

    Parameters
    ----------
    r : float, ndarray
        Radius of the circle.

    Returns
    -------
    float, ndarray
        Area of the circle.
    """
    return (np.pi * r * r)


def area_to_radii(A):
    return (np.sqrt(A / np.pi))

def radii_to_area(radii,radii_err=None):
    area_from_radii = np.pi*radii*radii
    if radii_err is not None:
        area_from_radii_err = 2.0*np.pi*radii*radii_err
        # return(area_from_radii,area_from_radii_err)
    else:
        area_from_radii_err = 0.0
    return(area_from_radii,area_from_radii_err)

def pix_area_to_kpc_area(area_pix,pix_to_pc):
    kpc_area = (area_pix * (pix_to_pc**2.0)/(1000**2.0))
    return(kpc_area)
    
def load_fits_data(image):
    '''
        Name origin:
        ctn > casa to numpy
        Function that read fits files, using casa IA.open or astropy.io.fits.
        Note: For some reason, IA.open returns a rotated mirroed array, so we need
        to undo it by a rotation.
        '''
    if isinstance(image, str) == True:
        try:
            ia = IA()
            ia.open(image)
            try:
                numpy_array = ia.getchunk()[:, :, 0, 0]
            except:
                numpy_array = ia.getchunk()[:, :]
            ia.close()
            # casa gives a mirroed and 90-degree rotated image :(
            data_image = np.rot90(numpy_array)[::-1, ::]
            return (data_image)
        except:
            try:
                _data_image = pf.getdata(image)
                if len(_data_image.shape) == 4:
                    data_image = _data_image[0][0]
                else:
                    data_image = _data_image
                return (data_image)
            except:
                print('Input image is not a fits file.')
                return(ValueError)
    else:
        raise ValueError("Input image is not a string.")

ctn = deprecated("ctn","load_fits_data")(load_fits_data)


def cvi(imname):
    try:
        os.system('casaviewer ' + imname)
    except:
        try:
            os.system('~/casaviewer ' + imname)
        except:
            pass


def normalise_in_log(profile):
    def normalise(x):
        return (x- x.min())/(x.max() - x.min())
    y = profile.copy()
    nu_0 = normalise(np.log(y))
    return nu_0


def shuffle_2D(image):
    height, width = image.shape

    # Reshape the image to a 1D array
    image_flat = image.copy().reshape(-1)

    # Shuffle the pixels randomly
    np.random.shuffle(image_flat)

    # Reshape the shuffled 1D array back to a 2D image
    shuffled_image = image_flat.reshape(height, width)

    # Print the shuffled image
    return(shuffled_image)


def estimate_circular_aperture(image, cellsize, std=3):
    if isinstance(image, str) == True:
        g = load_fits_data(image)
    else:
        g = image
    gmask = g[g > std * mad_std(g)]
    npix = len(gmask)
    barea = beam_area2(image, cellsize)
    nbeams = npix / barea
    circ_radii = np.sqrt(npix / np.pi)
    return (circ_radii)


def beam_area(Omaj, Omin, cellsize):
    '''
    Computes the estimated projected beam area (theroetical),
    given the semi-major and minor axis
    and the cell size used during cleaning.
    Return the beam area in pixels.
    '''
    BArea = ((np.pi * Omaj * Omin) / (4 * np.log(2))) / (cellsize ** 2.0)
    return (BArea)


def beam_area2(image, cellsize=None):
    '''
    Computes the estimated projected beam area (theroetical),
    given the semi-major and minor axis
    and the cell size used during cleaning.
    Return the beam area in pixels.
    '''
    if cellsize is None:
        try:
            cellsize = get_cell_size(image)
        except:
            print('Unable to read cell size from image header. '
                  'Please, provide the cell size of the image!')
            pass
    imhd = imhead(image)
    Omaj = imhd['restoringbeam']['major']['value']
    Omin = imhd['restoringbeam']['minor']['value']
    BArea = ((np.pi * Omaj * Omin) / (4 * np.log(2))) / (cellsize ** 2.0)
    return (BArea)

def getfreqs(fitslist):
    freqs = []
    for fitsfile in fitslist:
        hdu = fits.open(fitsfile)
        hdr = hdu[0].header
        freq = hdr['CRVAL3']
        freqs.append(freq)
    _freqs = np.array(freqs)
    return _freqs

def beam_shape(image):
    '''
    Return the beam shape (bmin,bmaj,pa) from given image.
    It uses CASA's function `imhead`.

    '''
    import numpy as np
    from astropy import units as u
    cell_size = get_cell_size(image)
    imhd = imhead(image)
    Omaj = imhd['restoringbeam']['major']['value']
    Omin = imhd['restoringbeam']['minor']['value']
    PA = imhd['restoringbeam']['positionangle']['value']
    freq = imhd['refval'][2] / 1e9
    """
    bmaj,bmin,PA,freq = beam_shape(crop_image)
    """
    bmaj = Omaj*u.arcsec
    bmin = Omin*u.arcsec
    freq_ = freq * u.GHz

    fwhm_to_sigma = 1./(8*np.log(2))**0.5
    BAarcsec = 2.*np.pi*(bmaj*bmin*fwhm_to_sigma**2)
    # # BA
    # equiv = u.brightness_temperature(freq_)
    # (0.0520*u.Jy/BA).to(u.K, equivalencies=equiv)
    return (Omaj,Omin,PA,freq,BAarcsec)

def sort_list_by_beam_size(imagelist, residuallist=None,return_df=False):
    """
    Sort a list of images files by beam size.

    If no residual list is provided, it will assume that the residual files are in the same
    directory as the image files, and that the residual files have the same name prefix as the
    image files.


    Parameters
    ----------
    imagelist : list
        List of image files.
    residuallist : list, optional
        List of associated residual files.
    return_df : bool, optional
        Return a pandas dataframe with the beam sizes.

    Returns
    -------
    imagelist_sort : list
        Sorted list of image files.
    residuallist_sort : list
        Sorted list of residual files.
    df_beam_sizes_sorted : pandas dataframe
        Dataframe with the beam sizes.
    """
    beam_sizes_list = []
    for i in tqdm(range(len(imagelist))):
        beam_sizes = {}
        aO, bO, _, _, _ = beam_shape(imagelist[i])
        beam_size_arcsec = np.sqrt(aO*bO)
        # beam_size_px, _, _ = get_beam_size_px(imagelist[i])
        beam_sizes['imagename'] = imagelist[i]
        if residuallist is not None:
            beam_sizes['residualname'] = residuallist[i]
        else:
            beam_sizes['residualname'] = \
                imagelist[i].replace('/MFS_images/','/MFS_residuals/')\
                            .replace('-image','-residual')
        beam_sizes['id'] = i
        beam_sizes['B_size_arcsec'] = beam_size_arcsec
        beam_sizes_list.append(beam_sizes)

    df_beam_sizes = pd.DataFrame(beam_sizes_list)
    df_beam_sizes_sorted = df_beam_sizes.sort_values('B_size_arcsec')
    imagelist_sort = np.asarray(df_beam_sizes_sorted['imagename'])
    residuallist_sort = np.asarray(df_beam_sizes_sorted['residualname'])
    # i = 0
    # for image in imagelist_sort:
    #     print(i,'>>',os.path.basename(image))
    #     i=i+1
    if return_df==True:
        return(imagelist_sort,residuallist_sort,df_beam_sizes_sorted)
    if return_df==False:
        return (imagelist_sort, residuallist_sort)


def sort_list_by_freq(imagelist, return_df=False):
    """
    Sort a list of images files by frequency.

    If no residual list is provided, it will assume that the residual files are in the same
    directory as the image files, and that the residual files have the same name prefix as the
    image files.

    Parameters
    ----------
    imagelist : list
        List of image files.
    residuallist : list, optional
        List of associated residual files.
    return_df : bool, optional
        Return a pandas dataframe with the beam sizes.

    Returns
    -------
    imagelist_sort : list
        Sorted list of image files.
    residuallist_sort : list
        Sorted list of residual files.
    df_beam_sizes_sorted : pandas dataframe
        Dataframe with the beam sizes.
    """
    freqs = getfreqs(imagelist)

    # Get indices that would sort the frequency array
    sorted_indices = np.argsort(freqs)
    # Sort the frequency array and image list using the sorted indices
    sorted_freq_array = freqs[sorted_indices]
    sorted_image_list = [imagelist[i] for i in sorted_indices]

    residuallist = []
    for i in range(len(sorted_image_list)):
        residuallist_i = \
            sorted_image_list[i].replace('/MFS_images/', '/MFS_residuals/') \
                .replace('-image.', '-residual.').replace('-image-pb', '-residual-pb')
        residuallist.append(residuallist_i)

    imagelist_sort = np.asarray(sorted_image_list)
    residuallist_sort = np.asarray(residuallist)
    i = 0
    for image in imagelist_sort:
        print(i, '>>', os.path.basename(image))
        i = i + 1
    return (imagelist_sort, residuallist_sort, sorted_freq_array)


def get_beam_size_px(imagename):
    aO,bO,_,_,_ = beam_shape(imagename)
    cs = get_cell_size(imagename)
    aO_px = aO/cs
    bO_px = bO/cs
    beam_size_px = np.sqrt(aO_px * bO_px)
    return(beam_size_px,aO_px,bO_px)

def beam_physical_area(imagename,z):
    '''
    Return the beam shape (bmin,bmaj,pa) given an image.
    '''
    import numpy as np
    from astropy import units as u
    cell_size = get_cell_size(imagename)
    imhd = imhead(imagename)
    Omaj = imhd['restoringbeam']['major']['value']
    Omin = imhd['restoringbeam']['minor']['value']
    """
    bmaj,bmin,PA,freq = beam_shape(crop_image)
    """
    pc_scale = arcsec_to_pc(z=z,cell_size=cell_size)
    bmaj = Omaj*pc_scale
    bmin = Omin*pc_scale

    fwhm_to_sigma = 1./(8*np.log(2))**0.5
    BAarcsec = 2.*np.pi*(bmaj*bmin*fwhm_to_sigma**2)

    return (pc_scale,bmaj,bmin,BAarcsec)




def cut_image(img, center=None, size=(1024, 1024),
              cutout_filename=None, special_name=''):
    """
    Cut images keeping updated header/wcs.
    This function is a helper to cut both image and its associated residual
    (from casa or wsclean).
    It saves both images with a cutout prefix.
    If the centre is not given, the peak position will be selected.
    If the size is not defined, cut a standard size of (1024 x 1024).
    It updates the wcs of the croped image.

    To do: crop a fration of the image.
    """
    if center is None:
        """
        Better to include momments instead of peak.
        """
        imst = imstat(img)
        position = (imst['maxpos'][0], imst['maxpos'][1])
    else:
        position = center

    hdu = pf.open(img)[0]
    wcs = WCS(hdu.header, naxis=2)

    cutout = Cutout2D(hdu.data[0][0], position=position, size=size, wcs=wcs)
    hdu.data = cutout.data
    hdu.header.update(cutout.wcs.to_header())
    if cutout_filename is None:
        cutout_filename = img.replace('-image.fits', '-image_cutout' +
                                      special_name + '.fits')
    hdu.writeto(cutout_filename, overwrite=True)

    # do the same for the residual image
    hdu2 = pf.open(img)[0]
    wcs2 = WCS(hdu2.header, naxis=2)

    hdu_res = pf.open(img.replace('-image.fits', '-residual.fits'))[0]
    # plt.imshow(hdu_res.data[0][0])
    wcs_res = WCS(hdu_res.header, naxis=2)
    cutout_res = Cutout2D(hdu_res.data[0][0],
                          position=position, size=size, wcs=wcs2)
    hdu2.data = cutout_res.data
    hdu2.header.update(cutout_res.wcs.to_header())
    # if cutout_filename is None:
    cutout_filename_res = img.replace('-image.fits', '-residual_cutout' +
                                      special_name + '.fits')
    hdu2.writeto(cutout_filename_res, overwrite=True)
    return(cutout_filename,cutout_filename_res)


def do_cutout(image, box_size=(200,200), center=None, return_='data'):
    """
    Fast cutout of a numpy array.

    Returs: numpy data array or a box for that cutout, if asked.
    """
    if isinstance(box_size, int):
        box_size = (box_size,box_size)

    if center is None:
        if isinstance(image, str) == True:
            # imhd = imhead(image)
            st = imstat(image)
            print('  >> Center --> ', st['maxpos'])
            xin, xen, yin, yen = (st['maxpos'][0] - box_size[0],
                                  st['maxpos'][0] + box_size[0],
                                  st['maxpos'][1] - box_size[1],
                                  st['maxpos'][1] + box_size[1])
            data_cutout = load_fits_data(image)[xin:xen, yin:yen]

        else:
            try:
                max_x, max_y = np.where(load_fits_data(image) == load_fits_data(image).max())
                xin = max_x[0] - box_size[0]
                xen = max_x[0] + box_size[0]
                yin = max_y[0] - box_size[1]
                yen = max_y[0] + box_size[1]
            except:
                max_x, max_y = np.where(image == image.max())
                xin = max_x[0] - box_size[0]
                xen = max_x[0] + box_size[0]
                yin = max_y[0] - box_size[1]
                yen = max_y[0] + box_size[1]

            data_cutout = image[xin:xen, yin:yen]


    else:
        xin, xen, yin, yen = (center[0] - box_size[0], center[0] + box_size[0],
                              center[1] - box_size[1], center[1] + box_size[1])
        if isinstance(image, str) == True:
            data_cutout = load_fits_data(image)[xin:xen, yin:yen]
        else:
            data_cutout = image[xin:xen, yin:yen]

    if return_ == 'data':
        return (data_cutout)
    if return_ == 'box':
        box = xin, xen, yin, yen  # [xin:xen,yin:yen]
        return (box)


# def do_cutout_2D(image_data, box_size=300, center=None, return_='data'):
#     """
#     Fast cutout of a numpy array.
#
#     Returs: numpy data array or a box for that cutout, if asked.
#     """
#
#     if center is None:
#         x0, y0= nd.maximum_position(image_data)
#         print('  >> Center --> ', x0, y0)
#         if x0-box_size>1:
#             xin, xen, yin, yen = x0 - box_size, x0 + box_size, \
#                                  y0 - box_size, y0 + box_size
#         else:
#             print('Box size is larger than image!')
#             return ValueError
#     else:
#         xin, xen, yin, yen = center[0] - box_size, center[0] + box_size, \
#             center[1] - box_size, center[1] + box_size
#     if return_ == 'data':
#         data_cutout = image_data[xin:xen, yin:yen]
#         return (data_cutout)
#     if return_ == 'box':
#         box = xin, xen, yin, yen  # [xin:xen,yin:yen]
#         return(box)

def do_cutout_2D(image_data, box_size=(300,300), center=None, return_='data'):
    """
    Fast cutout of a numpy array.

    Returs: numpy data array or a box for that cutout, if asked.
    """
    if isinstance(box_size, int):
        box_size = (box_size,box_size)

    if center is None:
        x0, y0= nd.maximum_position(image_data)
        print('  >> Center --> ', x0, y0)
        if x0-box_size[0]>1:
            xin, xen, yin, yen = x0 - box_size[0], x0 + box_size[0], \
                                 y0 - box_size[1], y0 + box_size[1]
        else:
            print('Box size is larger than image!')
            return ValueError
    else:
        xin, xen, yin, yen = center[0] - box_size[0], center[0] + box_size[0], \
            center[1] - box_size[1], center[1] + box_size[1]
    if return_ == 'data':
        data_cutout = image_data[xin:xen, yin:yen]
        return (data_cutout)
    if return_ == 'box':
        box = xin, xen, yin, yen  # [xin:xen,yin:yen]
        return(box)




def copy_header(image, image_to_copy, file_to_save=None):
    """
    For image files with no wcs, copy the header from a similar/equal image to
    the wanted file.
    Note: This is intended to be used to copy headers from images to their
    associated models and residuals.
    Note: Residual CASA images do not have Jy/Beam units, so this function
        can be used to copy the header/wcs information to the wanted file
        in order to compute the total flux in residual maps after the
        header has been copied.
    """
    from astropy.io import fits
    if file_to_save is None:
        file_to_save = image_to_copy.replace('.fits', 'header.fits')
    # Open the source image and get its header
    with fits.open(image) as hdu1:
        header = hdu1[0].header
        # Open the target image and update its header
        with fits.open(image_to_copy, mode='update') as hdu2:
            hdu2[0].header.update(header)
            hdu2.flush()
            hdu2.close()
    pass


def tcreate_beam_psf(imname, cellsize=None,size=(128,128),app_name='',
                     aspect=None):
    """
    From an interferometric image, reconstruct the restoring beam as a PSF
    gaussian image, with the same size as the original image.

    Parameters
    ----------
    imname : str
        Image name.
    cellsize : float
        Cell size of the image in arcsec.
    size : tuple
        Size of the PSF image.
    app_name : str
        Name to append to the PSF image.
    aspect : str
        Aspect ratio of the PSF image. If 'equal', the PSF will be circular.
        If None, the PSF will be elliptical.
    Returns
    -------
    psf_name : str
        PSF image name.

    """
    if cellsize is None:
        try:
            cellsize = get_cell_size(imname)
        except:
            print('Please, provide a cellsize for the image.')
            return (ValueError)

    msmd = casatools.msmetadata()
    ms = casatools.ms()
    tb = casatools.table()
    cl = casatools.componentlist()
    ia = IA()
    qa = casatools.quanta()
    # ia.open(image)

    imst = imstat(imname)
    imhd = imhead(imname)
    tb.close()
    direction = "J2000 10h00m00.0s -30d00m00.0s"
    cl.done()
    freq = str(imhd['refval'][2] / 1e9) + 'GHz'
    if (aspect=='circular') or (aspect=='equal'):
        print('WARNING: Using circular Gaussian for Gaussian beam convolution.')
        minoraxis = str(imhd['restoringbeam']['minor']['value']) + str(
            imhd['restoringbeam']['major']['unit'])
        majoraxis = str(imhd['restoringbeam']['minor']['value']) + str(
            imhd['restoringbeam']['major']['unit'])

    if (aspect == 'elliptical') or (aspect is None):
    # else:
        print('INFO: Using Elliptical Gaussian for Gaussian beam convolution.')
        minoraxis = str(imhd['restoringbeam']['minor']['value']) + str(
            imhd['restoringbeam']['minor']['unit'])
        majoraxis = str(imhd['restoringbeam']['major']['value']) + str(
            imhd['restoringbeam']['major']['unit'])
    # print(f"++==>>  PSF major/minor axis = ', {majoraxis} X {minoraxis}")

    pa = str(imhd['restoringbeam']['positionangle']['value']) + str(
        imhd['restoringbeam']['positionangle']['unit'])
    cl.addcomponent(dir=direction, flux=1.0, fluxunit='Jy', freq=freq,
                    shape="Gaussian", majoraxis=majoraxis, minoraxis=minoraxis,
                    positionangle=pa)
    ia.fromshape(imname.replace('.fits', '_beampsf'+app_name+'.im'),
                 [size[0],size[1],1,1], overwrite=True)
    cs = ia.coordsys()
    cs.setunits(['rad', 'rad', '', 'Hz'])
    cell_rad = qa.convert(qa.quantity(str(cellsize) + "arcsec"), "rad")['value']
    cs.setincrement([-cell_rad, cell_rad], 'direction')
    cs.setreferencevalue([qa.convert("10h", 'rad')['value'],
                          qa.convert("-30deg", 'rad')['value']],
                         type="direction")
    cs.setreferencevalue(freq, 'spectral')
    ia.setcoordsys(cs.torecord())
    ia.setbrightnessunit("Jy/pixel")
    ia.modify(cl.torecord(), subtract=False)
    exportfits(
        imagename=imname.replace('.fits', '_beampsf'+app_name+'.im'),
        fitsimage=imname.replace('.fits', '_beampsf'+app_name+'.fits'),
        overwrite=True)
    psf_name = imname.replace('.fits', '_beampsf'+app_name+'.fits')
    cl.close()
    return(psf_name)


def create_box_around_peak(imagename, fractions=None):
    """
    Create a box with 25% (or specified fraction) of the image
    around the peak.
    """
    ihl = imhead(imagename, mode='list')
    ih = imhead(imagename)
    st = imstat(imagename)

    M = ihl['shape'][0]
    N = ihl['shape'][1]
    if fractions is None:
        frac_X = int(0.25 * M)
        frac_Y = int(0.25 * N)
    else:
        frac_X = int(fractions[0] * M)
        frac_Y = int(fractions[1] * N)
    # slice_pos_X = 0.15 * M
    # slice_pos_Y = 0.85 * N
    slice_pos_X = st['maxpos'][0]
    slice_pos_Y = st['maxpos'][1]

    box_edge = np.asarray([slice_pos_X - frac_X,
                           slice_pos_Y - frac_Y,
                           slice_pos_X + frac_X,
                           slice_pos_Y + frac_Y]).astype(int)

    box_edge_str = str(box_edge[0]) + ',' + str(box_edge[1]) + ',' + \
                   str(box_edge[2]) + ',' + str(box_edge[3])

    return (box_edge_str)


def create_box(imagename, fracX=0.15, fracY=0.15):
    """
    Create a box with 20% of the image
    at an edge (upper left) of the image.
    To be used with casa tasks.
    """
    ihl = imhead(imagename, mode='list')
    ih = imhead(imagename)

    M = ihl['shape'][0]
    N = ihl['shape'][1]
    frac_X = int(fracX * M)
    frac_Y = int(fracY * N)
    slice_pos_X = int(0.02 * M)
    slice_pos_Y = int(0.98 * N)

    box_edge = np.asarray([slice_pos_X,
                           slice_pos_Y - frac_Y,
                           slice_pos_X + frac_X,
                           slice_pos_Y]).astype(int)

    box_edge_str = str(box_edge[0]) + ',' + str(box_edge[1]) + ',' + \
                   str(box_edge[2]) + ',' + str(box_edge[3])

    return (box_edge_str, ih)


def create_box_np(imagename, fracX=0.15, fracY=0.15):
    """
    Create a box with 20% of the image
    at an edge (upper left) of the image.
    To be used on a np.array.
    """
    ihl = imhead(imagename, mode='list')
    ih = imhead(imagename)

    M = ihl['shape'][0]
    N = ihl['shape'][1]

    frac_X = int(fracX * M)
    frac_Y = int(fracY * N)
    slice_pos_X = int(0.02 * M)
    slice_pos_Y = int(0.98 * N)

    cut = [[slice_pos_X, slice_pos_X + frac_X],
           [slice_pos_Y - frac_Y, slice_pos_Y]
           ]

    return (cut)

def sort_masks(image, mask_array, sort_by='flux'):
    unique_labels = np.unique(mask_array)
    region_labels = unique_labels[1:]  # 0 is the zero values
    mask_areas = []
    masks = []
    if sort_by == 'area':
        for i in range(len(region_labels)):
            mask_comp = (mask_array == region_labels[i])
            masks.append(mask_comp)
            area_mask = np.sum(mask_comp)
            mask_areas.append(area_mask)
    if sort_by == 'flux':
        for i in range(len(region_labels)):
            mask_comp = (mask_array == region_labels[i])
            masks.append(mask_comp)
            area_mask = np.sum(mask_comp * image)
            mask_areas.append(area_mask)

    mask_areas = np.asarray(mask_areas)
    sorted_indices_desc = np.argsort(mask_areas)[::-1]
    sorted_arr_desc = mask_areas[sorted_indices_desc]
    return (masks, sorted_indices_desc)


def rot180(imagem, x0, y0):
    R = np.matrix([[-1, 0], [0, -1]])
    img180 = nd.affine_transform(imagem, R,
                                 offset=(2. * (y0 - 0.5), 2. * (x0 - 0.5)))
    return img180


def estimate_area(data_mask, cellsize, Omaj, Omin):
    npix = np.sum(data_mask)
    barea = beam_area(Omaj, Omin, cellsize)
    nbeams = npix / barea
    circ_radii = np.sqrt(npix / np.pi)
    # print(npix)
    return (nbeams, circ_radii, npix)


def estimate_area_nbeam(image, mask, cellsize):
    npix = np.sum(mask)
    barea = beam_area2(image, cellsize)
    nbeams = npix / barea
    circ_radii = np.sqrt(npix / np.pi)
    # print(npix)
    return (circ_radii)

def estimate_area2(image,data_mask,cellsize):
    npix = np.sum(data_mask)
    barea= beam_area2(image,cellsize)
    nbeams = npix/barea
    circ_radii = np.sqrt(npix/np.pi)
    # print(npix)
    return(nbeams,circ_radii,npix)

def get_cell_size(imagename):
    """
    Get the cell size/pixel size in arcsec from an image header wcs.
    """
    hdu = pf.open(imagename)
    ww = WCS(hdu[0].header)
    pixel_scale = (ww.pixel_scale_matrix[1,1]*3600)
    cell_size =  pixel_scale.copy()
    return(cell_size)

def get_dilation_size(image):
    omaj, omin, _, _, _ = beam_shape(image)
    dilation_size = int(
        np.sqrt(omaj * omin) / (2 * get_cell_size(image)))
    return(dilation_size)


def mask_dilation(image, cell_size=None, sigma=6,rms=None,
                  dilation_size=None,iterations=2, dilation_type='disk',
                  PLOT=False,show_figure=False,logger=None,
                  fig_size_factor = 1,
                  special_name='',verbose=0):
    """
    Mask dilation function.

    Apply a binary dilation to a mask originated from the emission of a source.
    This was originally designed for radio images, where the dilation factor is
    proportional to the beam size. The dilation factor is computed as the geometric
    mean of the major and minor axis of the beam. The dilation factor is then
    converted to pixels using the cell size of the image. Note that the expansion
    occurs in two opposite directions, so the dilation size should be half of the
    size of the beam for one iteration, so that the final dilation will be a unit of
    the beam size. However, the exact size of the dilation will also be determinet by
    the number of iterations.


    Parameters
    ----------
    image : str
        Image name.
    cell_size : float
        Cell size of the image in arcsec.
    sigma : float
        Sigma level for the mask.
    rms : float
        RMS level for the mask.
    dilation_size : int
        Size of the dilation.
    iterations : int
        Number of iterations for the dilation.
    dilation_type : str
        Type of dilation. Options are 'disk' or 'square'.
    PLOT : bool
        Plot the mask dilation.
    show_figure : bool
        Show the figure.
    logger : logger
        Logger object.
    fig_size_factor : float
        Figure size factor.
    special_name : str
        Special name for the mask dilation.
    """

    if isinstance(image, str) == True:
        data = load_fits_data(image)
    else:
        data = image
    if rms is None:
        std = mad_std(data)
    else:
        std = rms

    if (dilation_size is None) or (dilation_size == '2x'):
        try:
            omaj, omin, _, _, _ = beam_shape(image)
            if dilation_size == '2x':
                dilation_size = int(
                    2*np.sqrt(omaj * omin) / (2 * get_cell_size(image)))
            else:
                dilation_size = int(
                    np.sqrt(omaj * omin) / (2 * get_cell_size(image)))
            if verbose >= 1:
                if logger is not None:
                    logger.debug(f" ==>  Mask dilation size is "
                                f"{dilation_size} [px]")
                else:
                    print(f" ==>  Mask dilation size is "
                        f"{dilation_size} [px]")
        except:
            if dilation_size is None:
                dilation_size = 5

    mask = (data >= sigma * std)
    mask3 = (data >= 3 * std)
    data_mask = mask * data

    if dilation_type == 'disk':
        data_mask_d = ndimage.binary_dilation(mask,
                                            structure=disk(dilation_size),
                                            iterations=iterations).astype(mask.dtype)

    if dilation_type == 'square':
        data_mask_d = ndimage.binary_dilation(mask,
                                            structure=square(dilation_size),
                                            iterations=iterations).astype(mask.dtype)

    if PLOT == True:
        fig = plt.figure(figsize=(int(15*fig_size_factor), int(4*fig_size_factor)))
        ax0 = fig.add_subplot(1, 4, 1)
        ax0.imshow((mask3), origin='lower',cmap='magma')
        ax0.set_title(r'mask $>$ ' + str(3) + '$\sigma_{\mathrm{mad}}$')
        ax0.axis('off')
        ax1 = fig.add_subplot(1, 4, 2)
        #         ax1.legend(loc='lower left')
        ax1.imshow((mask), origin='lower',cmap='magma')
        ax1.set_title(r'mask $>$ ' + str(sigma) + '$\sigma_{\mathrm{mad}}$')
        ax1.axis('off')
        ax2 = fig.add_subplot(1, 4, 3)
        ax2.imshow(data_mask_d, origin='lower',cmap='magma')
        ax2.set_title(r'D(mask)'f'[{dilation_size}|{iterations}]'f'{special_name}')
        ax2.axis('off')
        ax3 = fig.add_subplot(1, 4, 4)
        ax3 = eimshow(data * data_mask_d, ax=ax3, vmin_factor=0.01,CM='magma')
        ax3.set_title(r'D(mask) $\times$ data')
        #         ax3.imshow(np.log(data*data_mask_d))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        #         fig.tight_layout()
        ax3.axis('off')
        if show_figure == True:
            plt.show()
        else:
            plt.close()
    #         plt.savefig(image.replace('.fits','_masks.jpg'),dpi=300, bbox_inches='tight')

    # if cell_size is not None:
    #     if isinstance(image, str) == True:
    #         try:
    #             print((data * data_mask_d).sum() / beam_area2(image, cell_size))
    #             print((data * data_mask).sum() / beam_area2(image, cell_size))
    #             print((data).sum() / beam_area2(image, cell_size))
    #         except:
    #             print('Provide a cell size of the image.')
    return (mask, data_mask_d)




def mask_dilation_from_list(images, residuals, sigma=6, rms=None, dilation_size=None, iterations=2, dilation_type='disk',
                  PLOT=False, show_figure=True, logger=None, fig_size_factor=1, special_name='', verbose=0):
    """
    Mask dilation function for multi-frequency images.
    
    This function applies mask dilation across multiple images and computes a common mask
    that considers emission across all images. It avoids setting a smaller mask as the common mask.
    
    Parameters
    ----------
    images : list of str or ndarray
        List of image names or image data arrays.
    sigma : float
        Sigma level for the mask.
    rms : float
        RMS level for the mask.
    dilation_size : int
        Size of the dilation.
    iterations : int
        Number of iterations for the dilation.
    dilation_type : str
        Type of dilation. Options are 'disk' or 'square'.
    PLOT : bool
        Plot the mask dilation.
    show_figure : bool
        Show the figure.
    logger : logger
        Logger object.
    fig_size_factor : float
        Figure size factor.
    special_name : str
        Special name for the mask dilation.
    """

    # Function to compute mask for a single image
    def compute_mask(image, sigma, rms, dilation_size, iterations, dilation_type, verbose):
        if isinstance(image, str):
            data = load_fits_data(image)
        else:
            data = image
        if rms is None:
            std = mad_std(data)
        else:
            std = rms
        
        if (dilation_size is None) or (dilation_size == '2x'):
            try:
                omaj, omin, _, _, _ = beam_shape(image)
                if dilation_size == '2x':
                    dilation_size = int(2 * np.sqrt(omaj * omin) / (2 * get_cell_size(image)))
                else:
                    dilation_size = int(np.sqrt(omaj * omin) / (2 * get_cell_size(image)))
                if verbose >= 1:
                    if logger is not None:
                        logger.debug(f" ==> Mask dilation size is {dilation_size} [px]")
                    else:
                        print(f" ==> Mask dilation size is {dilation_size} [px]")
            except:
                dilation_size = 5

        mask = (data >= sigma * std)

        if dilation_type == 'disk':
            data_mask_d = ndimage.binary_dilation(mask, structure=disk(dilation_size), iterations=iterations).astype(mask.dtype)
        elif dilation_type == 'square':
            data_mask_d = ndimage.binary_dilation(mask, structure=square(dilation_size), iterations=iterations).astype(mask.dtype)

        return mask, data_mask_d

    # Compute individual masks for each image
    mask_list = []
    for kk in range(len(images)):
        rms = mad_std(load_fits_data(residuals[kk]))
        mask, dilated_mask = compute_mask(images[kk], sigma, rms, dilation_size, iterations, dilation_type, verbose)
        mask_list.append(dilated_mask)

    # Compute the common mask
    combined_mask = np.sum(mask_list, axis=0) >= len(mask_list) // 2  # Majority vote across masks

    # Optional plot of the common mask
    if PLOT:
        fig = plt.figure(figsize=(int(15 * fig_size_factor), int(4 * fig_size_factor)))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(combined_mask, origin='lower', cmap='magma')
        ax.set_title('Common Mask')
        ax.axis('off')
        if show_figure:
            plt.show()
        else:
            plt.close()

    return combined_mask




def mask_dilation_from_mask(image, mask_init, cell_size=None, sigma=3,rms=None,
                  dilation_size=None,iterations=2, dilation_type='disk',
                  PLOT=True,show_figure=True):
    """
    Apply a dilation to an existing mask.
    """
    from scipy import ndimage
    from scipy.ndimage import morphology
    from skimage.morphology import disk, square
    from skimage.morphology import dilation

    if isinstance(image, str) == True:
        data = load_fits_data(image)
    else:
        data = image
    if rms is None:
        std = mad_std(data)
    else:
        std = rms

    if dilation_size is None:
        try:
            omaj, omin, _, _, _ = beam_shape(image)
            dilation_size = int(
                np.sqrt(omaj * omin) / (2 * get_cell_size(image)))
        except:
            if dilation_size is None:
                dilation_size = 5

    data_init = data * mask_init
    mask3 = (data >= 3 * std)
    # std = mad_std(data[mask_init])
    mask = (data_init >= sigma * std)
    data_mask = mask * data

    if dilation_type == 'disk':
        data_mask_d = ndimage.binary_dilation(mask_init,
                                            structure=disk(dilation_size),
                                            iterations=iterations).\
                                                astype(mask_init.dtype)

    if dilation_type == 'square':
        data_mask_d = ndimage.binary_dilation(mask_init,
                                            structure=square(dilation_size),
                                            iterations=iterations).\
                                                astype(mask_init.dtype)

    if PLOT == True:
        fig = plt.figure(figsize=(15, 4))
        ax0 = fig.add_subplot(1, 4, 1)
        ax0.imshow((mask3), origin='lower')
        ax0.set_title(r'Mask above' + str(3) + '$\sigma$')
        ax0.axis('off')
        ax1 = fig.add_subplot(1, 4, 2)
        #         ax1.legend(loc='lower left')
        ax1.imshow((mask), origin='lower')
        ax1.set_title(r'Mask above' + str(sigma) + '$\sigma$')
        ax1.axis('off')
        ax2 = fig.add_subplot(1, 4, 3)
        ax2.imshow(data_mask_d, origin='lower')
        ax2.set_title(r'Dilated mask')
        ax2.axis('off')
        ax3 = fig.add_subplot(1, 4, 4)
        ax3 = fast_plot2(data * data_mask_d, ax=ax3, vmin_factor=0.1)
        ax3.set_title(r'Dilated mask $\times$ data')
        #         ax3.imshow(np.log(data*data_mask_d))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        #         fig.tight_layout()
        ax3.axis('off')
        if show_figure == True:
            plt.show()
        else:
            plt.close()

    if cell_size is not None:
        if isinstance(image, str) == True:
            try:
                print((data * data_mask_d).sum() / beam_area2(image, cell_size))
                print((data * data_mask).sum() / beam_area2(image, cell_size))
                print((data).sum() / beam_area2(image, cell_size))
            except:
                print('Provide a cell size of the image.')
    return (mask, data_mask_d)


def convert_seg_map_to_masks(seg_map):
    """
    Convert a segmentation map to a list of masks, where each mask corresponds to a unique segment.

    Parameters:
        seg_map (2D array): Segmentation map where different segments have different integer values,
                            and masked regions have a value of zero.

    Returns:
        list: A list of boolean masks, where each mask corresponds to a segment in the segmentation map.
    """
    # Find the unique segment labels, excluding zero (the background)
    unique_segments = np.unique(seg_map)
    unique_segments = unique_segments[unique_segments != 0]

    # Create a mask for each segment
    masks = [(seg_map == segment) for segment in unique_segments]

    return masks

def convert_masks_to_segmentation(masks_regions):
    # Assuming all masks have the same shape
    height, width = masks_regions[0].shape
    
    # Create an empty array for the segmentation image
    segmentation_image = np.zeros((height, width), dtype=int)
    
    # Loop over each mask and assign a unique integer value
    for idx, mask in enumerate(masks_regions, start=1):
        # idx+1 ensures the values start from 1 instead of 0
        segmentation_image[mask == 1] = idx
    
    return segmentation_image


def split_overlapping_masks(mask1, mask2):
    """
    # Compute the overlap between the masks

    GIven two masks, find the overlapping region between the two and
    split half of the pixels in this region attributing them to one maks and the
    other half to the other mask.

    # Check for overlap between the masks and split the overlapping region
    mask1_new, mask2_new = split_overlapping_masks(mask1, mask2)

    # Show the new masks
    print(mask1_new)
    print(mask2_new)

    """

    overlap_mask = (mask1 > 0) & (mask2 > 0)

    # If there is no overlap, return the original masks
    if not np.any(overlap_mask):
        return mask1, mask2

    # Split the overlapping region in half
    half_mask1 = mask1.copy()
    half_mask2 = mask2.copy()
    overlap_indices = np.where(overlap_mask)
    num_overlapping_pixels = len(overlap_indices[0])
    half_num_overlapping_pixels = num_overlapping_pixels // 2
    for i in range(num_overlapping_pixels):
        row, col = overlap_indices[0][i], overlap_indices[1][i]
        if i < half_num_overlapping_pixels:
            half_mask1[row, col] = mask1[row, col]
            half_mask2[row, col] = 0
        else:
            half_mask1[row, col] = 0
            half_mask2[row, col] = mask2[row, col]

    # Return the split masks
    return half_mask1, half_mask2




def split_overlapping_masks3(mask1, mask2, mask3):
    """
    # Compute the overlap between the masks

    The same as `split_overlapping_masks`, but for three masks.
    """
    overlap_mask1 = (mask1 > 0) & (mask2 > 0) & (mask3 == 0)
    overlap_mask2 = (mask1 > 0) & (mask3 > 0) & (mask2 == 0)
    overlap_mask3 = (mask2 > 0) & (mask3 > 0) & (mask1 == 0)

    # If there is no overlap, return the original masks
    if not np.any(overlap_mask1) and not np.any(overlap_mask2) and not np.any(
            overlap_mask3):
        return mask1, mask2, mask3

    # Split the overlapping region in half for each overlapping pair
    num_overlapping_pixels1 = np.sum(overlap_mask1)
    num_overlapping_pixels2 = np.sum(overlap_mask2)
    num_overlapping_pixels3 = np.sum(overlap_mask3)
    half_num_overlapping_pixels1 = num_overlapping_pixels1 // 2
    half_num_overlapping_pixels2 = num_overlapping_pixels2 // 2
    half_num_overlapping_pixels3 = num_overlapping_pixels3 // 2

    half_mask1 = mask1.copy()
    half_mask2 = mask2.copy()
    half_mask3 = mask3.copy()

    overlap_indices1 = np.where(overlap_mask1)
    for i in range(num_overlapping_pixels1):
        row, col = overlap_indices1[0][i], overlap_indices1[1][i]
        if i < half_num_overlapping_pixels1:
            half_mask1[row, col] = mask1[row, col]
            half_mask2[row, col] = 0
            half_mask3[row, col] = 0
        else:
            half_mask1[row, col] = 0
            half_mask2[row, col] = mask2[row, col]
            half_mask3[row, col] = 0

    overlap_indices2 = np.where(overlap_mask2)
    for i in range(num_overlapping_pixels2):
        row, col = overlap_indices2[0][i], overlap_indices2[1][i]
        if i < half_num_overlapping_pixels2:
            half_mask1[row, col] = mask1[row, col]
            half_mask2[row, col] = 0
            half_mask3[row, col] = 0
        else:
            half_mask1[row, col] = 0
            half_mask2[row, col] = 0
            half_mask3[row, col] = mask3[row, col]

    overlap_indices3 = np.where(overlap_mask3)
    for i in range(num_overlapping_pixels3):
        row, col = overlap_indices3[0][i], overlap_indices3[1][i]
        if i < half_num_overlapping_pixels3:
            half_mask1[row, col] = 0
            half_mask2[row, col] = mask2[row, col]
            half_mask3[row, col] = 0
        else:
            half_mask1[row, col] = 0
            half_mask2[row, col] = 0
            half_mask3[row, col] = mask3[row, col]

    # Return the split masks
    return half_mask1, half_mask2, half_mask3


def trail_vector(vx,vy,v0=np.asarray([1, 0])):
    import scipy.linalg as la
    v = np.asarray([vx, vy])
    norm_vec = np.linalg.norm(v)
    v_hat = v / norm_vec
    cosine_angle = np.dot(v0, v_hat) / (la.norm(v0) * la.norm(v_hat))
    angle_PA = np.degrees(np.arccos(cosine_angle))
    return(angle_PA,norm_vec)


def calculate_radii(image, mask):
    """
    Calculate circular radii of the emission in a 2D numpy image.

    Parameters:
        image (np.ndarray): The 2D numpy image containing the radio emission of a galaxy.
        sigma_level (float): The sigma level to use for the contour.
        background_std (float): The standard deviation of the background.

    Returns:
        float: The circular radii of the emission.
    """
    # Calculate the threshold level for the given sigma level
    #     threshold = sigma_level * background_std

    # Create a binary mask of the emission above the threshold level
    #     mask = image > threshold

    # Calculate the center of mass of the emission
    from scipy.ndimage import measurements
    com = measurements.center_of_mass(image * mask)

    # Calculate the distance of each pixel from the center of mass
    y, x = np.indices(mask.shape)
    distances = np.sqrt((x - com[1]) ** 2 + (y - com[0]) ** 2)

    # Calculate the median distance of the pixels above the threshold level
    median_distance = np.nanmedian(distances[mask])
    mean_distance = np.nanmean(distances[mask])
    std_distance = np.nanstd(distances[mask])
    # mad_std_distance = mad_std(distances[mask])

    return median_distance, mean_distance, std_distance

def pad_psf(imagename,psfnasme):
    import numpy as np
    psf_data = pf.getdata(psfnasme)
    image_data = load_fits_data(imagename)
    # Assume that 'psf' and 'image' are the original psf and image arrays, respectively
    psf_image_size = psf_data.shape[0] # assuming the psf is square
    image_size = image_data.shape[0] # assuming the image is square
    padding_size = (image_size - psf_image_size + 1) // 2

    # Create a new array of zeros with the desired padded size
    padded_psf = np.zeros((image_size, image_size))
    start_idx = image_size // 2 - psf_image_size // 2
    end_idx = start_idx + psf_image_size

    # Copy the original psf into the center of the new array
    padded_psf[start_idx:end_idx, start_idx:end_idx] = psf_data

    # Copy the original psf into the center of the new array
#     padded_psf[padding_size:-padding_size, padding_size:-padding_size] = psf_data
    pf.writeto(imagename.replace('.fits','_psf.fits'),padded_psf,overwrite=True)
    return(imagename.replace('.fits','_psf.fits'))


def get_frequency(imagename):
    """
    Get the frequency of a radio observation from the wcs of a fits image.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    # Open the FITS file
    with fits.open(imagename) as hdulist:
        header = hdulist[0].header

    # Extract WCS information
    wcs_info = WCS(header)

    for i in range(1, wcs_info.naxis + 1):
        if 'FREQ' in header.get(f'CTYPE{i}', ''):
            freq_ref = header.get(f'CRVAL{i}')
            frequency = freq_ref/1e9
    return(frequency)



def convolve_2D_smooth(imagename, imagename2=None,
                       mode='same', add_prefix='_convolve2D'):
    """
    CASA's way of convolving images.
    Very slow, but accurate.

    This function will be removed in the future, as all convolution operations
    will be migrated into pure python tasks.

    Parameters
    ----------
        imagename (str): The name of the image to convolve.
        imagename2 (str): The name of the image to use as the restoring beam.
        mode (str): The mode of convolution. Can be 'same' or 'transfer'.
        add_prefix (str): The prefix to add to the output image name.

    """
    if mode == 'same':
        imhd = imhead(imagename)
        imsmooth(imagename=imagename,
                 outfile=imagename.replace('.fits', '_convolved2D.fits'), overwrite=True,
                 major=imhd['restoringbeam']['major'],
                 minor=imhd['restoringbeam']['minor'],
                 pa=imhd['restoringbeam']['positionangle'])
        return (imagename.replace('.fits', '_convolved2D.fits'))

    if mode == 'transfer' and imagename2 != None:
        '''
        Use restoring beam from image1 to convolve image2.
        '''
        imhd = imhead(imagename2)
        outfile = imagename2.replace('.fits', add_prefix + '.fits')
        imsmooth(imagename=imagename,
                 outfile=outfile, overwrite=True,
                 major=imhd['restoringbeam']['major'],
                 minor=imhd['restoringbeam']['minor'],
                 pa=imhd['restoringbeam']['positionangle'])
        return (outfile)



def run_analysis_list(my_list,ref_residual,ref_image,z,mask_=None,rms=None,
                      sigma=6):
    results_conc_compact = []
    missing_data_im = []
    missing_data_re = []
#     z_d = {'VV705': 0.04019,'UGC5101':0.03937,'UGC8696':0.03734, 'VV250':0.03106}
    if rms is None:
        rms = mad_std(load_fits_data(ref_residual))
    if mask_ is None:
        _, mask_ = mask_dilation(ref_image,#imagelist_vla[k],
                                 sigma=sigma, iterations=2,
                                    dilation_size=None, PLOT=True)

    for i in tqdm(range(len(my_list))):
    # for i in tqdm(range(0,3)):
        crop_image = my_list[i]
        crop_residual = ref_residual#residuallist_vla[k]
        processing_results_model_compact = {} #store calculations only for source
        processing_results_model_compact['#modelname'] = os.path.basename(crop_image)

    #     processing_results_source,mask= measures(crop_image,crop_residual,z=zd,deblend=False,apply_mask=True,
    #                            results_final = processing_results_source,
    #                    plot_catalog = False,bkg_sub=False,bkg_to_sub = None,mask_component=None,
    #                    npixels=500,fwhm=121,kernel_size=121,sigma_mask=7.0,last_level=3.0,
    #                                              iterations=3,dilation_size=7,
    #                    do_PLOT=True,show_figure=True,add_save_name='')
        processing_results_model_compact, _, _ = measures(imagename=crop_image,
                                                residualname=crop_residual,
                                                z=z,rms=rms,
                                                mask = mask_,
#                                                 mask_component=mask_,
                                                do_petro=False,
                                                results_final=processing_results_model_compact,
                                                do_PLOT=True,dilation_size=None,
                                                apply_mask=False)
        results_conc_compact.append(processing_results_model_compact)
#     return(results_conc_compact)
    return(processing_results_model_compact)


"""
 _____ _ _      
|  ___(_) | ___ 
| |_  | | |/ _ \
|  _| | | |  __/
|_|   |_|_|\___|

  ___                       _   _             
 / _ \ _ __   ___ _ __ __ _| |_(_) ___  _ __  
| | | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \ 
| |_| | |_) |  __/ | | (_| | |_| | (_) | | | |
 \___/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|
      |_|   
#File Operations
"""


def get_list_names(root_path,prefix,which_data,source,sub_comp='',
                   version='v2',cutout_folder=''):
    path = root_path+\
           'data_analysis/LIRGI_sample/analysis_results/processing_images/'+\
           which_data+'/'+source+'/wsclean_images_'+\
           version+'/MFS_images/'+cutout_folder
    pathr = root_path+\
            'data_analysis/LIRGI_sample/analysis_results/processing_images/'+\
            which_data+'/'+source+'/wsclean_images_'+\
            version+'/MFS_residuals/'+cutout_folder

#     prefix = '*-MFS-image.fits'

    imlist = (glob.glob(path+prefix))
    imlist.sort()
    positives = []
    negatives = []
    for it in imlist:
        if '.-multiscale..-' in it:
            negatives.append(it)
#         if "taper_.-" in it:
#             negatives.append(it)
        else:
            positives.append(it)
    negatives.reverse()
#     negatives.sort()

    imagelist_sort = []
    for it in negatives:
        imagelist_sort.append(it)
    for it in positives:
        imagelist_sort.append(it)
    # em = [em[2],em[-1]]
    # i = 0
    # for image in imagelist_sort:
    #     print(i,'>>',os.path.basename(image))
    #     i=i+1

    imagelist_sort_res = []
    if cutout_folder == '':
        replacement = ['-image','-residual']
    else:
        if sub_comp =='':
            replacement = ['-image','-residual']
        else:
            replacement = ['image.cutout'+sub_comp+'.fits',
                           'residual.cutout'+sub_comp+'.fits']
    for i in range(len(imagelist_sort)):
        imagelist_sort_res.append(pathr +
                                  os.path.basename(imagelist_sort[i]).
                                  replace(replacement[0],replacement[1]))
    # i = 0
    # for image in imagelist_sort_res:
    #     print(i,'>>',os.path.basename(image))
    #     i=i+1
    return(np.asarray(imagelist_sort),np.asarray(imagelist_sort_res))




def get_fits_list_names(root_path,prefix='*.fits'):
    imlist = (glob.glob(root_path+prefix))
    imlist.sort()
    i = 0
    for image in imlist:
        print(i,'>>',os.path.basename(image))
        i=i+1
    return(imlist)

# def read_imfit_params(fileParams):
#     dlines = [line for line in open(fileParams) if len(line.strip()) > 0 and line[0] != "#"]
#     values = []
#     temp = []
#     for line in dlines:
#         if line.split()[0] == 'FUNCTION':
#             pass
#         else:
#             temp.append(float(line.split()[1]))
#         if line.split()[0] == 'r_e':
#             #         values['c1'] = {}
#             values.append(np.asarray(temp))
#             temp = []

#     if dlines[-2].split()[1] == 'FlatSky':
#         values.append(np.asarray(float(dlines[-1].split()[1])))
#     return (values)


def read_imfit_params(fileParams,return_names=False):
    dlines = [ line for line in open(fileParams) if len(line.strip()) > 0 and line[0] != "#" ]
    values=[]
    temp=[]
    param_names = []
    for line in dlines:
#         print(line)
        if line.split()[0]=='FUNCTION' or line.split()[0]=='GAIN' or line.split()[0]=='READNOISE':
            pass
        else:
#             print(float(line.split()[1]))
            temp.append(float(line.split()[1]))
            param_names.append(line.split()[0])
        if line.split()[0]=='R_e' or line.split()[0]=='r_e':
    #         values['c1'] = {}
            values.append(np.asarray(temp))
            temp = []

    if dlines[-2].split()[1]=='FlatSky':
        values.append(np.asarray(float(dlines[-1].split()[1])))
    if return_names == True:
        return(values,param_names)
    else:
        return(values)


# Testing functions, not used anywhere.
def shannon_entropy_2d(arr):
    rows, cols = arr.shape
    result = np.zeros((rows, cols))
    for i in range(2, rows-2):
        for j in range(2, cols-2):
            box = arr[i-2:i+3, j-2:j+3].ravel()
            p = box / np.nansum(box)
            entropy = -np.nansum(p * np.log2(p))
            result[i, j] = entropy
    return result

def cvi(imname):
    try:
        os.system('casaviewer ' + imname)
    except:
        try:
            os.system('~/casaviewer ' + imname)
        except:
            pass


def format_coords(dec_raw):
    deg, rest = dec_raw.split('.', 1)
    min, rest = rest.split('.', 1)
    sec = rest
    dec_formatted = f"{deg} {min} {sec}"
    return (dec_formatted)


def conver_str_coords(ra, dec):
    from astropy.coordinates import Angle, SkyCoord
    import astropy.units as u

    #     # Example hour-angle coordinates
    # #     ha_str = '13h15m34.9461s'
    # #     dec_str = '+62d07m28.6912s'

    #     # Convert the hour-angle and declination to angles
    #     ha = Angle(ha_str,unit='hourangle')
    #     print(ha)
    #     dec = Angle(dec_str,unit='deg')

    #     # Create a SkyCoord object with the coordinates and convert to ICRS frame
    #     coords = SkyCoord(ha, dec, unit=(u.hourangle, u.deg), frame='icrs')

    #     # Get the RA and Dec in degrees
    #     ra_deg = coords.ra.deg
    #     dec_deg = coords.dec.deg
    coor = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    ra_deg = coor.ra.degree
    dec_deg = coor.dec.degree
    # print(ra_deg, dec_deg)
    return (ra_deg, dec_deg)

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

def calculate_centroid_and_imagesize(coordinates, cellsize):
    """
    Calculate the centroid and imagesize for interferometric imaging given a list of celestial coordinates.
    
    
    # Example: Fake coordinates
    coordinates = [
        '20:37:31.075 +25:33:04.896',
        '20:37:28.675 +25:34:14.596',
        '20:37:34.475 +25:32:54.096',
        '20:37:30.875 +25:31:44.896',
    ]
    cellsize = 0.2  # Arcseconds

    # Calculate centroid and imagesize
    result = calculate_centroid_and_imagesize(coordinates, cellsize)

    # Extract phase centre and image dimensions
    phase_centre = SkyCoord(result["phase_centre"], unit=(u.hourangle, u.deg))
    ra_imagesize, dec_imagesize = result["imagesize"]

    # Visualize
    sky_coords = SkyCoord(coordinates, unit=(u.hourangle, u.deg))
    plt.figure(figsize=(8, 6))
    plt.scatter(sky_coords.ra.deg, sky_coords.dec.deg, color="blue", label="Source Positions")
    plt.scatter(phase_centre.ra.deg, phase_centre.dec.deg, color="red", marker="x", s=100, label="Centroid (Phase Centre)")

    # Draw rectangle
    margin_factor = 1.1
    ra_min = min(sky_coords.ra.deg) - (margin_factor - 1) * (max(sky_coords.ra.deg) - min(sky_coords.ra.deg))
    ra_max = max(sky_coords.ra.deg) + (margin_factor - 1) * (max(sky_coords.ra.deg) - min(sky_coords.ra.deg))
    dec_min = min(sky_coords.dec.deg) - (margin_factor - 1) * (max(sky_coords.dec.deg) - min(sky_coords.dec.deg))
    dec_max = max(sky_coords.dec.deg) + (margin_factor - 1) * (max(sky_coords.dec.deg) - min(sky_coords.dec.deg))

    plt.plot(
        [ra_min, ra_min, ra_max, ra_max, ra_min],
        [dec_min, dec_max, dec_max, dec_min, dec_min],
        color="orange", label="Imaging Region", linestyle="--"
    )

    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.legend()
    plt.title("Calculated Imaging Region and Centroid")
    plt.grid()
    plt.show()

    
    Parameters:
    coordinates (list): List of celestial coordinates as strings (e.g., '20:37:31.075 +25.33.04.896').
    cellsize (float): Pixel size in arcseconds.
    
    
    Returns:
    dict: Phase centre and imagesize with margins.
    """
    # Parse coordinates
    sky_coords = SkyCoord(coordinates, unit=(u.hourangle, u.deg))
    ra_values = sky_coords.ra.deg
    dec_values = sky_coords.dec.deg

    # Compute centroid
    ra_centroid = np.mean(ra_values)
    dec_centroid = np.mean(dec_values)
    phase_centre = SkyCoord(ra_centroid, dec_centroid, unit=u.deg)

    # Compute the angular extent
    ra_extent = (max(ra_values) - min(ra_values)) * np.cos(np.radians(dec_centroid))
    dec_extent = max(dec_values) - min(dec_values)

    # Add margins
    margin_factor = 1.1
    ra_extent_margined = ra_extent * margin_factor
    dec_extent_margined = dec_extent * margin_factor

    # Convert angular size to pixel size
    ra_imagesize = int(np.ceil(ra_extent_margined * 3600 / cellsize))
    dec_imagesize = int(np.ceil(dec_extent_margined * 3600 / cellsize))

    # Return results
    return {
        "phase_centre": phase_centre.to_string('hmsdms'),
        "phase_centre_formatted": phase_centre.to_string(style='hmsdms', sep=':', precision=3),
        "imagesize": (ra_imagesize, dec_imagesize),
    }



def find_offsets(reference_coords, target_coords):
    # Compute the pixel offsets between target and reference coordinates
    offset_x = reference_coords[0] - target_coords[0]
    offset_y = reference_coords[1] - target_coords[1]
    return offset_x, offset_y


def cutout_2D_radec_backup(imagename, residualname=None, ra_f=None, dec_f=None, cutout_size=1024,
                    special_name='',correct_shift=False,ref_cutout_image=None):
    from astropy.io import fits
    import os
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    # load image data and header
    if ra_f is None:
        imst = imstat(imagename)
        print('maxpos = ', imst['maxpos'])
        print('maxposf = ', imst['maxposf'])
        coords = imst['maxposf'].split(',')
        ra = coords[0]
        dec = format_coords(coords[1])
        # print(ra, dec)
        ra_f, dec_f = conver_str_coords(ra, dec)
        print('ra_f,dec_f = ', ra_f,dec_f)

    with fits.open(imagename) as hdul:
        image_data, header = hdul[0].data, hdul[0].header

        # create a WCS object from the header
        wcs = WCS(header, naxis=2)
        # wcs.wcs.radesys = 'icrs'

        # set the center and size of the cutout
        #     ra_f,dec_f = conver_str_coords(ra,dec)
        center_ra = ra_f  # center RA in degrees
        center_dec = dec_f  # center Dec in degrees

        # center = SkyCoord(ra=center_ra, dec=center_dec, unit='deg',from)
        center = SkyCoord(ra=center_ra * u.degree, dec=center_dec * u.degree,
                          frame='icrs')
        # print(center)
        # create a Cutout2D object
        cutout = Cutout2D(image_data[0][0], center, cutout_size, wcs=wcs)
        # apply shift
        if correct_shift == True:
            if ref_cutout_image is not None:
                ref_image_cutout_data = load_fits_data(ref_cutout_image)
                x_ref, y_ref = nd.maximum_position(ref_image_cutout_data)[::-1]
                reference_source_coords = (x_ref, y_ref)
                offset_x, offset_y = find_offsets(reference_source_coords,
                                                  nd.maximum_position(cutout.data)[::-1])
                print(f" !!!! Offsets of peak position are: {offset_x, offset_y}.")
                aligned_target_image = shift(cutout.data, (offset_y, offset_x),
                                             mode='constant')
                new_hdul = fits.HDUList(
                    [fits.PrimaryHDU(header=hdul[0].header, data=aligned_target_image)])
            else:
                #the code must stop
                print('No reference image was provided. '
                      'No shift correction will be applied.')
        else:
            new_hdul = fits.HDUList(
                [fits.PrimaryHDU(header=hdul[0].header, data=cutout.data)])

        new_hdul[0].header.update(cutout.wcs.to_header())
        savename_img = os.path.dirname(imagename) + '/' + os.path.basename(imagename).replace(
            '.fits', '.cutout' + special_name + '.fits')
        new_hdul.writeto(savename_img, overwrite=True)

    if residualname is not None:
        with fits.open(residualname) as hdul:
            image_data, header = hdul[0].data, hdul[0].header

            # create a WCS object from the header
            wcs = WCS(header, naxis=2)
            # wcs.wcs.radesys = 'icrs'

            # set the center and size of the cutout
            #     ra_f,dec_f = conver_str_coords(ra,dec)
            center_ra = ra_f  # center RA in degrees
            center_dec = dec_f  # center Dec in degrees

            # center = SkyCoord(ra=center_ra, dec=center_dec, unit='deg',from)
            center = SkyCoord(ra=center_ra * u.degree, dec=center_dec * u.degree,
                              frame='icrs')

            # create a Cutout2D object
            cutout = Cutout2D(image_data[0][0], center, cutout_size, wcs=wcs)
            if correct_shift == True:
                if ref_cutout_image is not None:
                    # ref_image_cutout_data = load_fits_data(ref_cutout_image)
                    # x_ref, y_ref = nd.maximum_position(ref_image_cutout_data)[::-1]
                    # reference_source_coords = (x_ref, y_ref)
                    # offset_x, offset_y = find_offsets(reference_source_coords,
                    #                                   nd.maximum_position(cutout.data)[::-1])
                    # print(f" !!!! Offsets of peak position are: {offset_x, offset_y}.")
                    aligned_target_image = shift(cutout.data, (offset_y, offset_x),
                                                 mode='constant')
                    new_hdul = fits.HDUList(
                        [fits.PrimaryHDU(header=hdul[0].header, data=aligned_target_image)])
                else:
                    # the code must stop
                    print('No reference image was provided. '
                          'No shift correction will be applied.')
            else:
                new_hdul = fits.HDUList(
                    [fits.PrimaryHDU(header=hdul[0].header, data=cutout.data)])

            # new_hdul = fits.HDUList(
            #     [fits.PrimaryHDU(header=hdul[0].header, data=cutout.data)])
            new_hdul[0].header.update(cutout.wcs.to_header())
            savename_res = os.path.dirname(residualname) + '/' + os.path.basename(
                residualname).replace('.fits', '.cutout' + special_name + '.fits')
            new_hdul.writeto(savename_res, overwrite=True)
    return (ra_f, dec_f,savename_img,wcs)

def cutout_2D_radec(imagename, residualname=None, ra_f=None, dec_f=None, cutout_size=1024,
                    special_name='', correct_shift=False, ref_cutout_image=None, pixel_coords=None):
    from astropy.io import fits
    import os
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    
    # load image data and header
    with fits.open(imagename) as hdul:
        image_data, header = hdul[0].data, hdul[0].header
        wcs = WCS(header, naxis=2)

        if pixel_coords is not None:
            # Convert pixel coordinates to RA/Dec
            ra_f, dec_f = wcs.pixel_to_world(*pixel_coords).ra.degree, wcs.pixel_to_world(*pixel_coords).dec.degree
            print(f"Converted pixel coordinates {pixel_coords} to RA/Dec: ({ra_f}, {dec_f})")
        
        elif ra_f is None or dec_f is None:
            imst = imstat(imagename)
            print('maxpos = ', imst['maxpos'])
            print('maxposf = ', imst['maxposf'])
            coords = imst['maxposf'].split(',')
            ra = coords[0]
            dec = format_coords(coords[1])
            ra_f, dec_f = conver_str_coords(ra, dec)
            print('ra_f, dec_f = ', ra_f, dec_f)
        
        # set the center and size of the cutout
        center = SkyCoord(ra=ra_f * u.degree, dec=dec_f * u.degree, frame='icrs')
        # print('Centre SkyCoord = ',center)
        # create a Cutout2D object
        cutout = Cutout2D(image_data[0][0], center, cutout_size, wcs=wcs)
        
        # apply shift
        if correct_shift:
            if ref_cutout_image is not None:
                ref_image_cutout_data = load_fits_data(ref_cutout_image)
                x_ref, y_ref = nd.maximum_position(ref_image_cutout_data)[::-1]
                reference_source_coords = (x_ref, y_ref)
                offset_x, offset_y = find_offsets(reference_source_coords,
                                                  nd.maximum_position(cutout.data)[::-1])
                print(f" !!!! Offsets of peak position are: {offset_x, offset_y}.")
                aligned_target_image = shift(cutout.data, (offset_y, offset_x),
                                             mode='constant')
                new_hdul = fits.HDUList(
                    [fits.PrimaryHDU(header=hdul[0].header, data=aligned_target_image)])
            else:
                # the code must stop
                print('No reference image was provided. '
                      'No shift correction will be applied.')
        else:
            new_hdul = fits.HDUList(
                [fits.PrimaryHDU(header=hdul[0].header, data=cutout.data)])

        new_hdul[0].header.update(cutout.wcs.to_header())
        savename_img = os.path.dirname(imagename) + '/' + os.path.basename(imagename).replace(
            '.fits', '.cutout.' + special_name + '.fits')
        new_hdul.writeto(savename_img, overwrite=True)

    if residualname is not None:
        with fits.open(residualname) as hdul:
            image_data, header = hdul[0].data, hdul[0].header
            wcs = WCS(header, naxis=2)

            center = SkyCoord(ra=ra_f * u.degree, dec=dec_f * u.degree, frame='icrs')

            cutout = Cutout2D(image_data[0][0], center, cutout_size, wcs=wcs)
            if correct_shift:
                if ref_cutout_image is not None:
                    aligned_target_image = shift(cutout.data, (offset_y, offset_x),
                                                 mode='constant')
                    new_hdul = fits.HDUList(
                        [fits.PrimaryHDU(header=hdul[0].header, data=aligned_target_image)])
                else:
                    print('No reference image was provided. '
                          'No shift correction will be applied.')
            else:
                new_hdul = fits.HDUList(
                    [fits.PrimaryHDU(header=hdul[0].header, data=cutout.data)])

            new_hdul[0].header.update(cutout.wcs.to_header())
            savename_res = os.path.dirname(residualname) + '/' + os.path.basename(
                residualname).replace('.fits', '.cutout.' + special_name + '.fits')
            new_hdul.writeto(savename_res, overwrite=True)
    
    return ra_f, dec_f, savename_img



def cutout_2D_radec_v2(imagename, residualname=None, ra_f=None, dec_f=None, cutout_size=1024,
                       cut_model=False,special_name='', correct_shift=False, 
                       ref_cutout_image=None, pixel_coords=None,
                       custom_save_path=None, custom_save_name=None):
    from astropy.io import fits
    import os
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    
    if os.path.exists(custom_save_path) is False:
        os.makedirs(custom_save_path)

    prefix_image_add = f"-{imagename.split('-image')[0].split('-')[-1]}"
    
    with fits.open(imagename) as hdul:
        image_data, header = hdul[0].data, hdul[0].header
        wcs = WCS(header, naxis=2)

        if pixel_coords is not None:
            ra_f, dec_f = wcs.pixel_to_world(*pixel_coords).ra.degree, wcs.pixel_to_world(*pixel_coords).dec.degree
            print(f"Converted pixel coordinates {pixel_coords} to RA/Dec: ({ra_f}, {dec_f})")
        elif ra_f is None or dec_f is None:
            imst = imstat(imagename)
            print('maxpos = ', imst['maxpos'])
            print('maxposf = ', imst['maxposf'])
            coords = imst['maxposf'].split(',')
            ra = coords[0]
            dec = format_coords(coords[1])
            ra_f, dec_f = conver_str_coords(ra, dec)
            print('ra_f, dec_f = ', ra_f, dec_f)
            
        center = SkyCoord(ra=ra_f * u.degree, dec=dec_f * u.degree, frame='icrs')
        # print('Centre SkyCoord = ', center)
        cutout = Cutout2D(image_data[0][0] if len(image_data.shape) == 4 else image_data, center, cutout_size, wcs=wcs)
        
        if len(image_data.shape) == 4:
            new_data = np.zeros((image_data.shape[0], image_data.shape[1], cutout_size, cutout_size))
            new_data[0,0] = cutout.data
        else:
            new_data = cutout.data

        if correct_shift:
            if ref_cutout_image is not None:
                ref_image_cutout_data = load_fits_data(ref_cutout_image)
                x_ref, y_ref = nd.maximum_position(ref_image_cutout_data)[::-1]
                reference_source_coords = (x_ref, y_ref)
                offset_x, offset_y = find_offsets(reference_source_coords,
                                                nd.maximum_position(cutout.data)[::-1])
                print(f" !!!! Offsets of peak position are: {offset_x, offset_y}.")
                if len(image_data.shape) == 4:
                    new_data[0,0] = shift(new_data[0,0], (offset_y, offset_x), mode='constant')
                else:
                    new_data = shift(new_data, (offset_y, offset_x), mode='constant')
            else:
                raise ValueError(f"No reference image was provided."
                                 f"No shift correction will be applied.")

        header.update(cutout.wcs.to_header())
        
        if len(image_data.shape) == 4:
            header['NAXIS3'] = image_data.shape[1]
            header['NAXIS4'] = image_data.shape[0]
            header['NAXIS1'] = cutout_size
            header['NAXIS2'] = cutout_size

        hdu = fits.PrimaryHDU(data=new_data, header=header)
        if custom_save_path is not None and custom_save_name is not None:
            savename_img = custom_save_path + '/' + custom_save_name+prefix_image_add+'-image.fits'
        else:
            savename_img = os.path.dirname(imagename) + '/' + os.path.basename(imagename).replace('.fits', '.cutout.' + special_name + '.fits')
        hdu.writeto(savename_img, overwrite=True)

    if residualname is not None:
        with fits.open(residualname) as hdul:
            image_data, header = hdul[0].data, hdul[0].header
            wcs = WCS(header, naxis=2)
            
            cutout = Cutout2D(image_data[0][0] if len(image_data.shape) == 4 else image_data, center, cutout_size, wcs=wcs)
            
            if len(image_data.shape) == 4:
                new_data = np.zeros((image_data.shape[0], image_data.shape[1], cutout_size, cutout_size))
                new_data[0,0] = cutout.data
            else:
                new_data = cutout.data

            if correct_shift and ref_cutout_image is not None:
                if len(image_data.shape) == 4:
                    new_data[0,0] = shift(new_data[0,0], (offset_y, offset_x), mode='constant')
                else:
                    new_data = shift(new_data, (offset_y, offset_x), mode='constant')
            header.update(cutout.wcs.to_header())
            if len(image_data.shape) == 4:
                header['NAXIS3'] = image_data.shape[1]
                header['NAXIS4'] = image_data.shape[0]
                header['NAXIS1'] = cutout_size
                header['NAXIS2'] = cutout_size

            hdu = fits.PrimaryHDU(data=new_data, header=header)
            
            if custom_save_path is not None and custom_save_name is not None:
                savename_res = custom_save_path + '/' + custom_save_name+prefix_image_add+'-residual.fits'
            else:
                savename_res = os.path.dirname(residualname) + '/' + os.path.basename(residualname).replace('.fits', '.cutout.' + special_name + '.fits')
            
            hdu.writeto(savename_res, overwrite=True)
    if cut_model:
        model_name = imagename.replace('-image','-model')
        with fits.open(model_name) as hdul:
            image_data, header = hdul[0].data, hdul[0].header
            wcs = WCS(header, naxis=2)
            
            cutout = Cutout2D(image_data[0][0] if len(image_data.shape) == 4 else image_data, center, cutout_size, wcs=wcs)
            
            if len(image_data.shape) == 4:
                new_data = np.zeros((image_data.shape[0], image_data.shape[1], cutout_size, cutout_size))
                new_data[0,0] = cutout.data
            else:
                new_data = cutout.data

            if correct_shift and ref_cutout_image is not None:
                if len(image_data.shape) == 4:
                    new_data[0,0] = shift(new_data[0,0], (offset_y, offset_x), mode='constant')
                else:
                    new_data = shift(new_data, (offset_y, offset_x), mode='constant')
            header.update(cutout.wcs.to_header())
            if len(image_data.shape) == 4:
                header['NAXIS3'] = image_data.shape[1]
                header['NAXIS4'] = image_data.shape[0]
                header['NAXIS1'] = cutout_size
                header['NAXIS2'] = cutout_size

            hdu = fits.PrimaryHDU(data=new_data, header=header)
            
            if custom_save_path is not None and custom_save_name is not None:
                savename_model = custom_save_path + '/' + custom_save_name+prefix_image_add+'-model.fits'
            else:
                savename_model = os.path.dirname(model_name) + '/' + os.path.basename(model_name).replace('.fits', '.cutout.' + special_name + '.fits')
            
            hdu.writeto(savename_model, overwrite=True)
   
    return ra_f, dec_f, savename_img


"""
 ____  _        _
/ ___|| |_ __ _| |_ ___
\___ \| __/ _` | __/ __|
 ___) | || (_| | |_\__ \
|____/ \__\__,_|\__|___/

"""

def quadrature_error(imagedata,residualdata):
    sum_squares = np.nansum(residualdata**2.0)
    mean_value = np.nanmean(imagedata)
    N_size = imagedata.size
    q_error = np.sqrt((sum_squares / (N_size - 1)) - (mean_value**2))
    return(q_error)

def rms_estimate(imagedata):
    mean_value = np.mean(imagedata)
    square_difference = (imagedata - mean_value) ** 2
    mean_squared_diff = np.mean(square_difference)
    RMS = np.sqrt(mean_squared_diff)
    return(RMS)

def mc_flux_error(imagename,image,residual, num_threads = 6, n_samples = 1000):
    """
    TEST: DO NOT USE THIS
    """
    def calculate_sum(image, residual, seed):
        # Set random seed
        np.random.seed(seed)

        # Randomly sample a subset of the data with replacement
        sample = np.random.choice(image.ravel(), size=image.size, replace=True)

        # Add residual map to sampled data
        sample_with_residual = sample + residual.ravel()

        # Calculate sum of sampled data
        sample_sum = np.sum(sample_with_residual)

        return sample_sum

    import concurrent.futures

    print(' >> Running MC flux error estimate....')
    # Create ThreadPoolExecutor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    # Submit tasks to ThreadPoolExecutor
    tasks = []
    for i in range(n_samples):
        task = executor.submit(calculate_sum, image, residual, i)
        tasks.append(task)

    # Retrieve results from tasks
    sums = []
    for task in tqdm(concurrent.futures.as_completed(tasks)):
        result = task.result()
        sums.append(result)

    # Calculate mean and standard deviation of sums
    sum_mean = np.mean(sums/beam_area2(imagename))
    sum_std = np.std(sums/beam_area2(imagename))

    # Print results
    print(f"Sum: {sum_mean:.4f}")
    print(f"Error: {sum_std:.4f}")

    # Plot histogram of sums
    plt.hist(sums/beam_area2(imagename), bins=20)
    plt.axvline(sum_mean, color='r', linestyle='--', label=f"Mean: {sum_mean:.2f}")
    plt.legend()
    plt.show()
    return(sum_mean, sum_std)

def emcee_flux_error(image,residual):
    """
    TEST: DO NOT USE THIS
    """
    import numpy as np
    import emcee
    # Define log likelihood function
    def log_likelihood(flux, image, residual, sigma):
        # Calculate model image with given flux
        model = image + flux * residual

        # Calculate chi-squared statistic
        chi2 = np.sum((model - image)**2 / sigma**2)

        # Calculate log likelihood
        log_likelihood = -0.5 * chi2

        return log_likelihood

    # Define log prior function
    def log_prior(flux):
        # Uniform prior on flux between 0 and 1000
        if 0 < flux < 1000:
            return 0.0

        return -np.inf

    # Define log probability function
    def log_probability(flux, image, residual, sigma):
        # Calculate log prior
        lp = log_prior(flux)
        if not np.isfinite(lp):
            return -np.inf

        # Calculate log likelihood
        ll = log_likelihood(flux, image, residual, sigma)

        # Calculate log posterior probability
        log_prob = lp + ll

        return log_prob

    # Generate example data
    # image = np.random.normal(10.0, 1.0, size=(100, 100))
    # residual = np.random.normal(0.0, 0.1, size=(100, 100))
    sigma = 0.1

    # Define number of dimensions and number of walkers
    ndim = 1
    nwalkers = 32

    # Initialize walkers with random values
    print('# Initialize walkers with random values')
    p0 = np.random.uniform(0.0, 100.0, size=(nwalkers, ndim))

    # Initialize sampler with log probability function
    print('# Initialize sampler with log probability function')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(image, residual, sigma),threads=6)

    # Burn-in phase to reach equilibrium
    print('# Burn-in phase to reach equilibrium')
    n_burn = 100
    pos, prob, state = sampler.run_mcmc(p0, n_burn, store=False)

    # Reset sampler and run production phase
    print('# Reset sampler and run production phase')
    sampler.reset()
    n_prod = 1000
    sampler.run_mcmc(pos, n_prod)

    # Extract samples from sampler
    print('# Extract samples from sampler')
    samples = sampler.get_chain()

    # Calculate mean and standard deviation of flux estimates
    print('# Calculate mean and standard deviation of flux estimates')
    flux_samples = samples[:, 0]
    flux_mean = np.mean(flux_samples)
    flux_std = np.std(flux_samples)

    # Calculate total flux estimate and error
    print('# Calculate total flux estimate and error')
    total_flux = np.sum(image + flux_mean * residual)
    total_flux_err = flux_std * np.sum(residual)

    print("Total flux estimate:", total_flux)
    print("Total flux error:", total_flux_err)
#     print("Total flux estimate:", total_flux/beam_area2(imagelist[idx]))
#     print("Total flux error:", total_flux_err/beam_area2(imagelist[idx]))
    def plot_flux_estimate(image, residual, flux_samples):
        """
        plot_flux_estimate(image,residual,flux_samples)
        """
        # Calculate total flux estimate and error
        flux_mean = np.mean(flux_samples)
        flux_std = np.std(flux_samples)
        total_flux = np.sum(image + flux_mean * residual)/beam_area2(imagelist[idx])
        total_flux_err = flux_std * np.sum(residual)/beam_area2(imagelist[idx])

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

    #     # Plot image
    #     ax.imshow(image, cmap='viridis', origin='lower', alpha=0.8)
    #     ax.set_xlabel('X pixel')
    #     ax.set_ylabel('Y pixel')
    #     ax.set_title('Flux estimate for 2D image')

    #     # Add colorbar
    #     cbar = plt.colorbar(ax.imshow(image, cmap='viridis', origin='lower', alpha=0.8))
    #     cbar.ax.set_ylabel('Pixel value')

        # Plot flux samples as scatter plot
        ax.scatter(flux_samples/beam_area2(imagelist[idx]),
                   total_flux -
                   (flux_samples/beam_area2(imagelist[idx])).sum(axis=1),
                   cmap='viridis', alpha=0.5)

        # Add mean and error bars
        ax.axvline(flux_mean/beam_area2(imagelist[idx]), color='red',
                   label='Flux estimate')
        ax.axhline(total_flux - np.sum(flux_samples/beam_area2(imagelist[idx])) +
                   flux_mean/beam_area2(imagelist[idx]) * len(flux_samples),
                   color='red', linestyle='--', label='Total flux estimate')
        ax.fill_betweenx([total_flux - total_flux_err, total_flux +
                          total_flux_err], flux_mean/beam_area2(imagelist[idx]) -
                         flux_std/beam_area2(imagelist[idx]),
                         flux_mean/beam_area2(imagelist[idx]) +
                         flux_std/beam_area2(imagelist[idx]),
                         alpha=0.2, color='red', label='Total flux error')

        # Add legend
        ax.legend(loc='lower right')

        # Show plot
        plt.show()


def cdiff(data):
    '''
    Manual 'fix' of numpy's diff function.
    '''
    diff_corr = np.diff(data,append=data[-1])
    diff_corr[-1] = diff_corr[-2]
    return(diff_corr)

"""
 ____
/ ___|  ___  _   _ _ __ ___ ___
\___ \ / _ \| | | | '__/ __/ _ \
 ___) | (_) | |_| | | | (_|  __/
|____/ \___/ \__,_|_|  \___\___|

 ____       _            _   _
|  _ \  ___| |_ ___  ___| |_(_) ___  _ __
| | | |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \
| |_| |  __/ ||  __/ (__| |_| | (_) | | | |
|____/ \___|\__\___|\___|\__|_|\___/|_| |_|

"""

def get_ell_size_factor(psf_current, psf_large=50, ell_large=2.0, psf_small=4,
                        ell_small=7):
    """
    Rough linear relation between the restoring beam size (psf) with the scale factor
    of the ellipse to
    be drawn on the detection map.
    """
    return ell_large + (ell_small - ell_large) *  ((psf_current - psf_large) / (
            psf_small - psf_large))


"""
 ___                            
|_ _|_ __ ___   __ _  __ _  ___ 
 | || '_ ` _ \ / _` |/ _` |/ _ \
 | || | | | | | (_| | (_| |  __/
|___|_| |_| |_|\__,_|\__, |\___|
                     |___/      
    _                _           _     
   / \   _ __   __ _| |_   _ ___(_)___ 
  / _ \ | '_ \ / _` | | | | / __| / __|
 / ___ \| | | | (_| | | |_| \__ \ \__ \
/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
                       |___/    

"""


def get_radial_profiles(imagelist, labels_profiles=None, save_fig=None):
    '''
    Given a list of images (e.g. target images, psf, etc), plot the radial
    profile of each one altogether and retur the radiis and profiles for all.
    '''
    profiles = []
    radiis = []
    for IMAGE in imagelist:
        RR, IR = get_profile(IMAGE)
        profiles.append(IR)
        radiis.append(RR)
    return (profiles, radiis)

def get_peak_pos(imagename):
    st = imstat(imagename=imagename)
    maxpos = st['maxpos'][0:2]
    # print('Peak Pos=', maxpos)
    return (maxpos)


def get_profile(imagename, center=None,binsize=1,interpnan=True,stddev=False):
    if isinstance(imagename, str) == True:
        data_2D = load_fits_data(imagename)
    else:
        data_2D = imagename

    if stddev:
        nr, radius, profile, profile_std = azimuthalAverage(data_2D,return_nr = True,
                                                interpnan=interpnan,center=center,
                                                stddev = stddev,
                                                binsize=binsize)
        return (radius, profile, profile_std)
    else:
        nr, radius, profile = azimuthalAverage(data_2D,return_nr = True,
                                                interpnan=interpnan,center=center,
                                                binsize=binsize)
        return (radius, profile)
    


def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False,
                     binsize=1.0, weights=None, steps=False, interpnan=False, left=None, right=None,
                     mask=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.
    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        y0max, x0max = nd.maximum_position((image))
        # center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
        center = np.array([x0max, y0max])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        mask = np.ones(image.shape, dtype='bool')
    # obsolete elif len(mask.shape) > 1:
    # obsolete     mask = mask.ravel()

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)
    nbins = int(np.round(np.nanmax(r) / binsize) + 1)
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    # nr = np.bincount(whichbin)[1:]
    nr = np.histogram(r, bins, weights=mask.astype('int'))[0]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or range(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        # Find out which radial bin each point in the map belongs to
        whichbin = np.digitize(r.flat, bins)
        # This method is still very slow; is there a trick to do this with histograms?
        radial_prof_std = np.array([np.nanstd(image.flat[mask.flat * (whichbin == b)]) for b in range(1, nbins + 1)])
    
    radial_prof = np.histogram(r, bins, weights=(image * weights * mask))[0] / \
                    np.histogram(r, bins, weights=(mask * weights))[0]

    if interpnan:
        radial_prof = np.interp(bin_centers, bin_centers[radial_prof == radial_prof],
                                radial_prof[radial_prof == radial_prof], left=left, right=right)

    if steps:
        xarr = np.array(zip(bins[:-1], bins[1:])).ravel()
        yarr = np.array(zip(radial_prof, radial_prof)).ravel()
        return xarr, yarr
    elif returnradii:
        if stddev:
            return bin_centers, radial_prof, radial_prof_std
        else:
            return bin_centers, radial_prof
    elif return_nr:
        if stddev:
            return nr, bin_centers, radial_prof, radial_prof_std
        else:
            return nr, bin_centers, radial_prof
    else:
        return radial_prof


def azimuthalAverageBins(image, azbins, symmetric=None, center=None, **kwargs):
    """ Compute the azimuthal average over a limited range of angles
    kwargs are passed to azimuthalAverage """
    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    theta = np.arctan2(x - center[0], y - center[1])
    theta[theta < 0] += 2 * np.pi
    theta_deg = theta * 180.0 / np.pi

    if isinstance(azbins, np.ndarray):
        pass
    elif isinstance(azbins, int):
        if symmetric == 2:
            azbins = np.linspace(0, 90, azbins)
            theta_deg = theta_deg % 90
        elif symmetric == 1:
            azbins = np.linspace(0, 180, azbins)
            theta_deg = theta_deg % 180
        elif azbins == 1:
            return azbins, azimuthalAverage(image, center=center, returnradii=True, **kwargs)
        else:
            azbins = np.linspace(0, 359.9999999999999, azbins)
    else:
        raise ValueError("azbins must be an ndarray or an integer")

    azavlist = []
    for blow, bhigh in zip(azbins[:-1], azbins[1:]):
        mask = (theta_deg > (blow % 360)) * (theta_deg < (bhigh % 360))
        rr, zz = azimuthalAverage(image, center=center, mask=mask, returnradii=True, **kwargs)
        azavlist.append(zz)

    return azbins, rr, azavlist


def radialAverage(image, center=None, stddev=False, returnAz=False, return_naz=False,
                  binsize=1.0, weights=None, steps=False, interpnan=False, left=None, right=None,
                  mask=None, symmetric=None):
    """
    Calculate the radially averaged azimuthal profile.
    (this code has not been optimized; it could be speed boosted by ~20x)
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    stddev - if specified, return the radial standard deviation instead of the average
    returnAz - if specified, return (azimuthArray,azimuthal_profile)
    return_naz   - if specified, return number of pixels per azimuth *and* azimuth
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and azimuthal
        profile so you can plot a step-form azimuthal profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.
    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])
    theta = np.arctan2(x - center[0], y - center[1])
    theta[theta < 0] += 2 * np.pi
    theta_deg = theta * 180.0 / np.pi
    maxangle = 360

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        # mask is only used in a flat context
        mask = np.ones(image.shape, dtype='bool').ravel()
    elif len(mask.shape) > 1:
        mask = mask.ravel()

    # allow for symmetries
    if symmetric == 2:
        theta_deg = theta_deg % 90
        maxangle = 90
    elif symmetric == 1:
        theta_deg = theta_deg % 180
        maxangle = 180

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)
    nbins = int(np.round(maxangle / binsize))
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    # Find out which azimuthal bin each point in the map belongs to
    whichbin = np.digitize(theta_deg.flat, bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or range(1,nbins+1) )
    # azimuthal_prof.shape = bin_centers.shape
    if stddev:
        azimuthal_prof = np.array([image.flat[mask * (whichbin == b)].std() for b in range(1, nbins + 1)])
    else:
        azimuthal_prof = np.array(
            [(image * weights).flat[mask * (whichbin == b)].sum() / weights.flat[mask * (whichbin == b)].sum() for b in
             range(1, nbins + 1)])

    # import pdb; pdb.set_trace()

    if interpnan:
        azimuthal_prof = np.interp(bin_centers,
                                   bin_centers[azimuthal_prof == azimuthal_prof],
                                   azimuthal_prof[azimuthal_prof == azimuthal_prof],
                                   left=left, right=right)

    if steps:
        xarr = np.array(zip(bins[:-1], bins[1:])).ravel()
        yarr = np.array(zip(azimuthal_prof, azimuthal_prof)).ravel()
        return xarr, yarr
    elif returnAz:
        return bin_centers, azimuthal_prof
    elif return_naz:
        return nr, bin_centers, azimuthal_prof
    else:
        return azimuthal_prof


def radialAverageBins(image, radbins, corners=True, center=None, **kwargs):
    """ Compute the radial average over a limited range of radii """
    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])

    if isinstance(radbins, np.ndarray):
        pass
    elif isinstance(radbins, int):
        if radbins == 1:
            return radbins, radialAverage(image, center=center, returnAz=True, **kwargs)
        elif corners:
            radbins = np.linspace(0, r.max(), radbins)
        else:
            radbins = np.linspace(0, np.max(np.abs(np.array([x - center[0], y - center[1]]))), radbins)
    else:
        raise ValueError("radbins must be an ndarray or an integer")

    radavlist = []
    for blow, bhigh in zip(radbins[:-1], radbins[1:]):
        mask = (r < bhigh) * (r > blow)
        az, zz = radialAverage(image, center=center, mask=mask, returnAz=True, **kwargs)
        radavlist.append(zz)

    return radbins, az, radavlist


def get_image_statistics(imagename,cell_size=None,
                         mask_component=None,mask=None,
                         residual_name=None,region='', dic_data=None,
                         sigma_mask=6,apply_mask=True,
                         fracX=0.1, fracY=0.1):
    """
    Get some basic image statistics.



    """
    if dic_data is None:
        dic_data = {}
        dic_data['#imagename'] = os.path.basename(imagename)

    if cell_size is None:
        cell_size = get_cell_size(imagename)

    image_data = load_fits_data(imagename)
    if mask is not None:
        image_data = np.nan_to_num(image_data*mask,nan=0)
        apply_mask = False
    #     dic_data['imagename'] = imagename
    if apply_mask == True:
        omask, mask = mask_dilation(imagename, sigma=sigma_mask)
        image_data = np.nan_to_num(image_data*mask,nan=0)
    if mask_component is not None:
        image_data = np.nan_to_num(image_data * mask_component,nan=0)
        if mask is None:
            mask = mask_component
        # mask = mask_component
    if (mask_component is None) and (apply_mask == False) and (mask is None):
        unity_mask = np.ones(load_fits_data(imagename).shape)
        omask, mask = unity_mask, unity_mask
        image_data = np.nan_to_num(load_fits_data(imagename) * mask,nan=0)

    stats_im = imstat(imagename=imagename, region=region)

    box_edge, imhd = create_box(imagename, fracX=fracX, fracY=fracY)
    stats_box = imstat(imagename=imagename, box=box_edge)

    # determine the flux peak and positions of image
    flux_peak_im = stats_im['max'][0]
    flux_min_im = stats_im['min'][0]
    dic_data['max_im'] = flux_peak_im
    dic_data['min_im'] = flux_min_im
    """
    x0max,y0max = peak_center(image_data)
    dic_data['x0'], dic_data['y0'] = x0max,y0max
    # determine momentum centres.
    x0m, y0m, _, _ = momenta(image_data, PArad_0=None, q_0=None)
    dic_data['x0m'], dic_data['y0m'] = x0m, y0m

    #some geometrical measures
    # calculate PA and axis-ratio
    PA, q, x0col, y0col, PAm, qm, PAmi, qmi, PAmo, qmo,\
        x0median,y0median,\
        x0median_i,y0median_i,x0median_o,y0median_o = cal_PA_q(image_data)

    dic_data['PA'], dic_data['q'] = PA, q
    dic_data['PAm'], dic_data['qm'] = PAm, qm
    dic_data['PAm'], dic_data['qm'] = PAm, qm
    dic_data['PAmi'], dic_data['qmi'] = PAmi, qmi
    dic_data['PAmo'], dic_data['qmo'] = PAmo, qmo
    dic_data['x0m_i'], dic_data['y0m_i'] = x0median_i, y0median_i
    dic_data['x0m_o'], dic_data['y0m_o'] = x0median_o, y0median_o
    """

    # determine the rms and std of residual and of image
    rms_im = stats_im['rms'][0]
    rms_box = stats_box['rms'][0]
    sigma_im = stats_im['sigma'][0]
    sigma_box = stats_box['sigma'][0]

    dic_data['rms_im'] = rms_im
    dic_data['rms_box'] = rms_box
    dic_data['sigma_im'] = sigma_im
    dic_data['sigma_box'] = sigma_box

    # determine the image and residual flux
    flux_im = stats_im['flux'][0]
    flux_box = stats_box['flux'][0]
    dic_data['flux_im'] = flux_im
    dic_data['flux_box'] = flux_box
    sumsq_im = stats_im['sumsq'][0]
    sumsq_box = stats_box['sumsq'][0]

    q_sq = sumsq_im / sumsq_box
    q_flux = flux_im / flux_box
    # flux_ratio = flux_re/flux_im
    dic_data['q_sq'] = q_sq
    dic_data['q_flux'] = q_flux

    snr = flux_im / rms_box
    snr_im = flux_im / rms_im

    dr_e = []
    frac_ = np.linspace(0.05, 0.85, 10)
    frac_image = 0.10
    '''
    Each loop below run a sliding window,
    one to the x direction and the other to y-dircetion.
    This is to get a better estimate (in multiple regions)
    of the background rms and therefore SNR
    Each window has a fraction frac_image of the image size.
    '''
    for frac in frac_:
        box, _ = create_box(imagename, fracX=frac, fracY=frac_image)
        st = imstat(imagename, box=box)
        snr_tmp = flux_peak_im / st['rms'][0]
        dr_e.append(snr_tmp)

    dr_e2 = []
    for frac in frac_:
        box, _ = create_box(imagename, fracX=frac_image, fracY=frac)
        st = imstat(imagename, box=box)
        snr_tmp = flux_peak_im / st['rms'][0]
        dr_e2.append(snr_tmp)
    #average of the SNR -- DINAMIC RANGE
    DR_SNR_E = (np.mean(dr_e) + np.mean(dr_e2)) / 2

    dic_data['snr'] = snr
    dic_data['snr_im'] = snr_im
    dic_data['DR_SNR_E'] = DR_SNR_E

    DR_pk_rmsbox = flux_peak_im / rms_box
    DR_pk_rmsim = flux_peak_im / rms_im
    dic_data['DR_pk_rmsbox'] = DR_pk_rmsbox
    dic_data['DR_pk_rmsim'] = DR_pk_rmsim

    dic_data['bmajor'] = imhd['restoringbeam']['major']['value']
    dic_data['bminor'] = imhd['restoringbeam']['minor']['value']
    dic_data['positionangle'] = imhd['restoringbeam']['positionangle']['value']



    # if residual_name is not None:
    #     data_res = load_fits_data(residual_name)
    #     flux_res_error = 3 * np.sum(data_res * mask) \
    #                      / beam_area2(imagename, cell_size)
    #     # rms_res =imstat(residual_name)['flux'][0]
    #     flux_res = np.sum(load_fits_data(residual_name)) / beam_area2(imagename, cell_size)
    #
    #     res_error_rms =np.sqrt(
    #         np.sum((abs(data_res * mask -
    #                     np.mean(data_res * mask))) ** 2 * np.sum(mask))) / \
    #                    beam_area2(imagename,cell_size)
    #
    #     try:
    #         total_flux_tmp = dic_data['total_flux_mask']
    #     except:
    #         total_flux_tmp = flux_im
    #         total_flux_tmp = total_flux(image_data,imagename,mask=mask)
    #
    #     # print('Estimate #1 of flux error (based on sum of residual map): ')
    #     # print('Flux = ', total_flux_tmp * 1000, '+/-',
    #     #       abs(flux_res_error) * 1000, 'mJy')
    #     # print('Fractional error flux = ', flux_res_error / total_flux_tmp)
    #     print('-----------------------------------------------------------------')
    #     print('Estimate of flux error (based on rms of '
    #           'residual x area): ')
    #     print('Flux = ', total_flux_tmp * 1000, '+/-',
    #           abs(res_error_rms) * 1000, 'mJy')
    #     print('Fractional error flux = ', res_error_rms / total_flux_tmp)
    #     print('-----------------------------------------------------------------')
    #
    #     dic_data['max_residual'] = np.max(data_res * mask)
    #     dic_data['min_residual'] = np.min(data_res * mask)
    #     dic_data['flux_residual'] = flux_res
    #     dic_data['flux_error_res'] = abs(flux_res_error)
    #     dic_data['flux_error_res_2'] = abs(res_error_rms)
    #     dic_data['mad_std_residual'] = mad_std(data_res)
    #     dic_data['rms_residual'] = rms_estimate(data_res)

    #     print(' Flux=%.5f Jy/Beam' % flux_im)
    #     print(' Flux peak (image)=%.5f Jy' % flux_peak_im, 'Flux peak (residual)=%.5f Jy' % flux_peak_re)
    #     print(' flux_im/sigma_im=%.5f' % snr_im, 'flux_im/sigma_re=%.5f' % snr)
    #     print(' rms_im=%.5f' % rms_im, 'rms_re=%.5f' % rms_re)
    #     print(' flux_peak_im/rms_im=%.5f' % peak_im_rms, 'flux_peak_re/rms_re=%.5f' % peak_re_rms)
    #     print(' sumsq_im/sumsq_re=%.5f' % q)
    return (dic_data)


def level_statistics(img, cell_size=None, mask_component=None,
                    sigma=6, do_PLOT=False, crop=False,data_2D = None,data_res=None,
                    box_size=256, bkg_to_sub=None, apply_mask=True,
                    mask=None,rms=None,
                    results=None, dilation_size=None, iterations=2,
                    add_save_name='', SAVE=False, show_figure=False, ext='.jpg'):
    """
    Function old name: plot_values_std

    Slice the intensity values of an image into distinct regions.
    Then, compute information for each bin level of the emission.
    The implemented splitting is:

        1. Inner region: peak intensity -> 0.1 * peak intensity
        2. Mid-region: 0.1 * peak intensity -> 10 * rms
        3. Low-region: 10 * rms -> 6 * rms
        4. Uncertain region: 6 * rms -> 3 * rms

    Parameters
    ----------
    img : str
        Path to the image.
    cell_size : float, optional
        Cell size of the image. The default is None. In that case, the function
        get_cell_size will attempt to estimate it from the header of the image.
    mask_component : array, optional
        The default is None. This is designed to be used when an image is complex,
        and one would like to study multiple components of the emission separately,
        each one at a time.
    sigma : float, optional
        The default is 6. This is the number of standard deviations to be used during
        mask dilation.
    do_PLOT : bool, optional
        The default is False. If True, the function will plot and save the image.
    crop : bool, optional
        The default is False. If True, the function will crop the image to a square
        of size box_size.
    box_size : int, optional
        The default is 256. This is the size of the square to be used if crop=True.
    data_2D : array, optional
        The default is None. If not None, the function will use this array instead
        and consider header information from img to be used with the array data_2D.

    """
    if cell_size is None:
        cell_size = get_cell_size(img)
    if data_2D is not None:
        g_ = data_2D
    else:
        g_ = load_fits_data(img)
    g = g_.copy()
    if rms is None:
        std = mad_std(g_)
    else:
        std = rms

    if bkg_to_sub is not None:
        g = g - bkg_to_sub
    if mask_component is not None:
        g = g * mask_component

    beam_area_ = beam_area2(img)

    if mask is not None:
        g = g * mask
        apply_mask = False  # do not calculate the mask again, in case is True.
        g = g * mask
    if apply_mask == True:
        _, mask_dilated = mask_dilation(img, cell_size=cell_size, sigma=sigma,
                                        dilation_size=dilation_size, rms=rms,
                                        iterations=iterations,
                                        PLOT=False)
        g = g * mask_dilated


    g = np.nan_to_num(g,nan=0)
    
    if mask_component is not None:
        levels = np.geomspace(g.max(), (1 * std), 5)
        levels_top = np.geomspace(g.max(), g.max() * 0.1, 3)
        try:
            levels_mid = np.geomspace(g.max() * 0.1, (10 * std), 5)
        except:
            levels_mid = np.asarray([0])
        try:
            levels_low = np.geomspace(10 * std, (6.0  * std), 2)
            levels_uncertain = np.geomspace(6.0 * std, (3.0 * std), 3)
        except:
            levels_low = np.asarray([0])
            levels_uncertain = np.asarray([0])

    else:
        if apply_mask is not False:
            # print('asdasd', g.max(), std)
            levels = np.geomspace(g.max(), (1 * std), 5)
            levels_top = np.geomspace(g.max(), g.max() * 0.1, 3)
            try:
                levels_mid = np.geomspace(g.max() * 0.1, (10 * std), 5)
            except:
                levels_mid = np.asarray([0])
            try:
                levels_low = np.geomspace(10 * std, (6.0  * std), 2)
                levels_uncertain = np.geomspace(6 * std, (3.0 * std), 3)
            except:
                levels_low = np.asarray([0])
                levels_uncertain = np.asarray([0])
        else:
            levels = np.geomspace(g.max(), (3 * std), 5)
            levels_top = np.geomspace(g.max(), g.max() * 0.1, 3)
            # levels_mid = np.geomspace(g.max() * 0.1, (10 * std + dl), 5)
            try:
                levels_mid = np.geomspace(g.max() * 0.1, (10 * std), 5)
            except:
                levels_mid = np.asarray([0])
            try:
                levels_low = np.geomspace(10 * std, (6.0  * std), 2)
                levels_uncertain = np.geomspace(3 * std, (1.0 * std), 3)
            except:
                levels_low = np.asarray([0])
                levels_uncertain = np.asarray([0])

    # pix_inner = g[g >= levels_top[-1]]
    # pix_mid = g[np.where((g < levels_top[-1]) & (g >= levels_mid[-1]))]
    # pix_low = g[np.where((g < levels_mid[-1]) & (g >= levels_low[-1]))]
    # pix_uncertain = g[np.where((g < levels_low[-1]) & (g >= levels_uncertain[-1]))]
    pix_inner_mask = (g >= levels_top[-1])
    pix_inner = g * pix_inner_mask
    pix_mid_mask = (((g < levels_top[-1]) & (g >= levels_mid[-1])))
    pix_mid = g * pix_mid_mask
    pix_low_mask = (((g < levels_mid[-1]) & (g >= levels_low[-1])))
    pix_low = g * pix_low_mask
    pix_uncertain_mask = (((g < levels_low[-1]) & (g >= levels_uncertain[-1])))
    pix_uncertain = g * pix_uncertain_mask
    inner_flux = pix_inner.sum() / beam_area_
    mid_flux = pix_mid.sum() / beam_area_
    low_flux = pix_low.sum() / beam_area_
    uncertain_flux = pix_uncertain.sum() / beam_area_

    total_flux = low_flux + mid_flux + inner_flux + uncertain_flux
    ext_flux = low_flux + mid_flux + uncertain_flux
    pix_area = len(g[g >= 3 * std])
    number_of_beams = pix_area / beam_area_
    n_beams_inner = np.sum(pix_inner_mask) / beam_area_
    n_beams_mid = np.sum(pix_mid_mask) / beam_area_
    n_beams_low = np.sum(pix_low_mask) / beam_area_
    n_beams_uncertain = np.sum(pix_uncertain_mask) / beam_area_
    if results is None:
        results = {}
        results['#imagename'] = os.path.basename(img)

    # print('Low Flux (extended) Jy                    > ', low_flux, ' >> ratio=',
    #       low_flux / total_flux)
    # print('Mid Flux (outer core + inner extended) Jy > ', mid_flux, ' >> ratio=',
    #       mid_flux / total_flux)
    # print('Inner Flux (core) Jy                      > ', inner_flux,
    #       ' >> ratio=', inner_flux / total_flux)
    # print('Uncertain Flux (<5std)                    > ', uncertain_flux,
    #       ' >> ratio=', uncertain_flux / total_flux)
    # print('Total Flux Jy                             > ', total_flux)
    # print('Total area (in # ob beams)                > ', number_of_beams)
    # print('Total inner area (in # ob beams)          > ', n_beams_inner)
    # print('Total mid area (in # ob beams)            > ', n_beams_mid)
    # print('Total low area (in # ob beams)            > ', n_beams_low)
    # print('Total uncertain area (in # ob beams)      > ', n_beams_uncertain)
    # print('Inner Flux (core) fraction                > ',
    #       inner_flux / total_flux)
    # print('Outer Flux (ext)  fraction                > ', ext_flux / total_flux)

    results['peak_of_flux'] = np.max(g)
    results['total_flux'] = total_flux
    results['inner_flux'] = inner_flux
    results['low_flux'] = low_flux
    results['mid_flux'] = mid_flux
    results['uncertain_flux'] = uncertain_flux

    results['inner_flux_f'] = inner_flux / total_flux
    results['low_flux_f'] = low_flux / total_flux
    results['mid_flux_f'] = mid_flux / total_flux
    results['uncertain_flux_f'] = uncertain_flux / total_flux

    results['number_of_beams'] = number_of_beams
    results['n_beams_inner'] = n_beams_inner
    results['n_beams_mid'] = n_beams_mid
    results['n_beams_low'] = n_beams_low
    results['n_beams_uncertain'] = n_beams_uncertain

    if do_PLOT == True:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
        vmin = 1 * std
        vmax = g_.max()
        # norm = visualization.simple_norm(g, stretch='log')#, max_percent=max_percent_lowlevel)
        norm = simple_norm(g, stretch='asinh', asinh_a=0.005, min_cut=vmin,
                           max_cut=vmax)

        if crop == True:
            try:
                xin, xen, yin, yen = do_cutout(img, box_size=box_size,
                                               center=center, return_='box')
                g = g[xin:xen, yin:yen]
            except:
                try:
                    max_x, max_y = np.where(g == g.max())
                    xin = max_x[0] - box_size
                    xen = max_x[0] + box_size
                    yin = max_y[0] - box_size
                    yen = max_y[0] + box_size
                    g = g[xin:xen, yin:yen]
                except:
                    pass
        try:
            im_plot = ax.imshow(g, cmap='magma_r', norm=norm, alpha=1.0,
                                origin='lower')
            ax.contour(g, levels=levels_top[::-1], colors='lime',
                       alpha=1.0)  # cmap='Reds', linewidths=0.75)
            ax.contour(g, levels=levels_mid[::-1], colors='yellow',
                       linewidths=0.75)
            ax.contour(g, levels=levels_low[::-1],
                       colors='#56B4E9')  # cmap='Greens', linewidths=0.75)
            ax.contour(g, levels=levels_uncertain[::-1], colors='grey',
                       linewidths=0.4)
        except:
            pass
        # im_plot.colorbar()
        #         plt.subplots_adjust(wspace=0, hspace=0)
        #         fig.tight_layout()

        if SAVE is not None:
            plt.savefig(
                img.replace('.fits', '_std_levels') + add_save_name + ext,
                dpi=300,
                bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()
    return (results)


def compute_asymetries(imagename,mask,mask_component=None,
                       bkg_to_sub=None,
                       centre=None,results=None):
    if results is None:
        results = {}
        results['#imagename'] = os.path.basename(imagename)

    if isinstance(imagename, str) == True:
        image_data = load_fits_data(imagename)
    else:
        image_data = imagename

    if bkg_to_sub is not None:
        image_data = image_data - bkg_to_sub
    if mask_component is not None:
        image_data = image_data * mask_component
    if centre is None:
        pass
    if centre is not None:
        x0A, y0A = centre
    if (mask is None) and (mask_component is not None):
        mask = mask_component
    else:
        unity_mask = np.ones(image_data.shape) == 1
        omask, mask = unity_mask, unity_mask

    try:
        BGrandom, BGmedian, BGmin, BGstd, BGx0, BGy0 \
            = background_asymmetry(image_data, mask, pre_clean=False)
    except:
        print('Error computing background assymetry.')
        BGrandom, BGmedian, BGmin, BGstd, BGx0, BGy0 = 0.0, 0.0, 0.0, 0.0, x0A, y0A

    x0A0fit, y0A0fit = fmin(assimetria0, (x0A, y0A),
                            args=(image_data, mask,), disp=0)
    x0A1fit, y0A1fit = fmin(assimetria1, (x0A, y0A),
                            args=(image_data, mask,), disp=0)
    A0 = assimetria0((x0A0fit, y0A0fit), image_data, mask) - BGmedian
    A1 = assimetria1((x0A1fit, y0A1fit), image_data, mask)
    results['A_BK_median'] = BGmedian
    results['A0'] = A0
    results['A1'] = A1
    results['x0A0fit'] = x0A0fit
    results['y0A0fit'] = y0A0fit
    results['x0A1fit'] = x0A1fit
    results['y0A1fit'] = y0A1fit
    return(results)


def convex_shape(mask):
    from scipy.spatial import ConvexHull
    indices = np.transpose(np.nonzero(mask))
    hull = ConvexHull(indices)
    convex_area = hull.area
    convex_perimeter = hull.volume
    return(convex_area,convex_perimeter)

def shape_measures(imagename, residualname, z, mask_component=None, sigma_mask=6,
             last_level=3.0, vmin_factor=1.0, plot_catalog=False,data_2D=None,
             npixels=128, fwhm=81, kernel_size=21, dilation_size=None,
             main_feature_index=0, results_final={}, iterations=2,
             fracX=0.10, fracY=0.10, deblend=False, bkg_sub=False,
             bkg_to_sub=None, rms=None,
             apply_mask=True, do_PLOT=False, SAVE=True, show_figure=True,
             mask=None,do_measurements='all',
             add_save_name=''):
    """
    Main function that perform other function calls responsible for
    all relevant calculations on the images.
    """
    cell_size = get_cell_size(imagename)
    """
    One beam area is one element resolution, so we avoind finding sub-components 
    that are smaller than the beam area. 
    """
    min_seg_pixels = beam_area2(imagename)

    if mask is not None:
        mask = mask
        apply_mask = False

    if apply_mask == True:
        _, mask_dilated = mask_dilation(imagename, cell_size=cell_size,
                                        sigma=sigma_mask,
                                        dilation_size=dilation_size,
                                        iterations=iterations, rms=rms,
                                        PLOT=True)
        mask = mask_dilated
    else:
        mask = None

    if data_2D is not None:
        data_2D = data_2D
    else:
        data_2D = load_fits_data(imagename)

    results_final = None #start dict to store measurements.

    levels, fluxes, agrow, plt, \
        omask2, mask2, results_final = make_flux_vs_std(imagename,
                                                        cell_size=cell_size,
                                                        residual=residualname,
                                                        mask_component=mask_component,
                                                        last_level=last_level,
                                                        sigma_mask=sigma_mask,
                                                        apply_mask=False,
                                                        data_2D = data_2D,
                                                        mask=mask,
                                                        rms=rms,
                                                        vmin_factor=vmin_factor,
                                                        results=results_final,
                                                        show_figure=show_figure,
                                                        bkg_to_sub=bkg_to_sub,
                                                        add_save_name=add_save_name,
                                                        SAVE=SAVE)
    error_petro = False
    if z is not None:
        z = z
    else:
        z = 0.01

    results_final['error_petro'] = error_petro
    if mask_component is not None:
        r, ir = get_profile(load_fits_data(imagename) * mask_component)
    else:
        r, ir = get_profile(imagename)
    #     rpix = r / cell_size
    rpix = r.copy()
    r_list_arcsec = rpix * cell_size
    Rp_arcsec = results_final['C90radii'] * cell_size
    R50_arcsec = results_final['C50radii'] * cell_size
    pix_to_pc = pixsize_to_pc(z=z, cell_size=cell_size)
    r_list_pc = rpix * pix_to_pc
    Rp_pc = results_final['C90radii'] * pix_to_pc
    R50_pc = results_final['C50radii'] * pix_to_pc
    Rp_arcsec = results_final['C90radii'] * cell_size
    R50_arcsec = results_final['C50radii'] * cell_size
    r_list_arcsec = rpix * cell_size
    Rp_arcsec = results_final['C90radii'] * cell_size
    R50_arcsec = results_final['C50radii'] * cell_size


    results_final = cosmo_stats(imagename=imagename, z=z, results=results_final)

    results_final['pix_to_pc'] = pix_to_pc
    results_final['cell_size'] = cell_size
    # if error_petro == True:
    # results_final['area_beam_2Rp'] = area_beam[
    #     int(results_final['C90radii'])]
    # results_final['area_beam_R50'] = area_beam[
    #     int(results_final['C50radii'])]

    df = pd.DataFrame.from_dict(results_final, orient='index').T
    df.to_csv(imagename.replace('.fits', add_save_name + '_area_stats.csv'),
              header=True,
              index=False)

    return (results_final, mask)


def compute_flux_density(imagename, residualname, mask=None, sigma=6,
                         systematic_error_fraction=0.05):
    beam_area = beam_area2(imagename)
    image_data = load_fits_data(imagename)
    rms = mad_std(load_fits_data(residualname))
    if mask is None:
        _, mask = mask_dilation(imagename, show_figure=False,
                                PLOT=False, rms=rms, sigma=sigma,
                                      iterations=2)
    total_flux_density = np.nansum(image_data * mask) / beam_area

    # compute error in flux density
    data_res = load_fits_data(residualname)
    res_error_rms = np.sqrt(np.nansum(
        (abs(data_res * mask - np.nanmean(data_res * mask))) ** 2 * np.nansum(
            mask))) / beam_area

    total_flux_density_residual = np.nansum(data_res * mask) / beam_area

    # Calculate systematic error as a fraction of the total flux density of the image
    systematic_error = systematic_error_fraction * total_flux_density

    # Calculate total flux density error in quadrature
    total_flux_density_error = np.sqrt(
        systematic_error ** 2 + (rms / beam_area) ** 2 + total_flux_density_residual ** 2)

    # res_error_rms = 3 * np.nansum(data_res * mask)/beam_area

    print('-----------------------------------------------------------------')
    print('Estimate of flux error (based on rms of '
          'residual x area): ')
    print('Flux Density = ', total_flux_density * 1000, '+/-',
          total_flux_density_error * 1000, 'mJy')
    print('Fractional error flux = ', total_flux_density_error / total_flux_density)
    print('-----------------------------------------------------------------')
    # print(f"Flux Density = {total_flux_density*1000:.2f} mJy")
    return (total_flux_density, total_flux_density_error)

def measures(imagename, residualname, z, mask_component=None, sigma_mask=6,
             last_level=2.0, vmin_factor=1.0, plot_catalog=False,
             data_2D=None,data_res=None,
             npixels=128, fwhm=81, kernel_size=21, dilation_size=None,
             main_feature_index=0, results_final={}, iterations=2,
             fracX=0.10, fracY=0.10, deblend=False, bkg_sub=False,
             bkg_to_sub=None, rms=None,do_petro=True,
             crop=False, box_size=256,
             apply_mask=True, do_PLOT=False, SAVE=False, show_figure=True,
             mask=None,do_measurements='',compute_A=False,
             add_save_name='',logger=None,verbose=0):
    """
    Main function that perform other function calls responsible for
    all relevant calculations on the images.
    """
    cell_size = get_cell_size(imagename)
    """
    One beam area is one element resolution, so we avoind finding sub-components
    that are smaller than the beam area.
    """
    min_seg_pixels = beam_area2(imagename)

    if mask is not None:
        mask = mask
        apply_mask = False
        omask = None
        if verbose >= 1:
            if logger is not None:
                logger.info(f"  >> Using provided mask.")
            else:
                print('     >> INFO: Using provided mask.')

    if apply_mask == True:
        # if logger is not None:
        #     logger.info(f"  CALC >> Performing mask dilation.")
        # else:
        #     print('     >> CALC: Performing mask dilation.')
        original_mask, mask_dilated = mask_dilation(imagename, cell_size=cell_size,
                                        sigma=sigma_mask,
                                        dilation_size=dilation_size,
                                        iterations=iterations, rms=rms,
                                        PLOT=do_PLOT,verbose=verbose)
        mask = mask_dilated
        omask = original_mask
    # else:
    #     print('     >> WARN: Not using any mask.')
    #     mask = None
    # """
    # Basic background estimation.
    # """
    # # bkg_ = sep_background(crop_image,apply_mask=False,mask=mask,
    # #                       bw=11, bh=11, fw=12, fh=12)
    # bkg_ = sep_background(imagename,apply_mask=True,mask=None,
    #                       bw=11, bh=11, fw=12, fh=12)
    # bkg_to_sub = bkg_.back()

    if data_2D is not None:
        data_2D = data_2D
    else:
        data_2D = load_fits_data(imagename)

    # if verbose >= 1:
    #     if logger is not None:
    #         logger.info(f"  CALC >> Performing level statistics.")
    #     else:
    #         print('     >> CALC: Performing level statistics.')

    # results_final = level_statistics(img=imagename, cell_size=cell_size,
    #                                 mask_component=mask_component,
    #                                 mask=mask, apply_mask=False,
    #                                 data_2D=data_2D,
    #                                 sigma=sigma_mask, do_PLOT=do_PLOT,
    #                                 results=results_final, bkg_to_sub=bkg_to_sub,
    #                                 show_figure=False,
    #                                 rms=rms,
    #                                 add_save_name=add_save_name,
    #                                 SAVE=SAVE, ext='.jpg')
    if verbose >=1:
        if logger is not None:
            logger.info(f"  CALC >> Computing image properties.")
        else:
            print('     >> CALC: Computing image properties.')

    levels, fluxes, agrow, \
        omask2, mask2, results_final = compute_image_properties(imagename,
                                                        residual=residualname,
                                                        cell_size=cell_size,
                                                        mask_component=mask_component,
                                                        last_level=last_level,
                                                        sigma_mask=sigma_mask,
                                                        apply_mask=False,
                                                        crop=crop,box_size=box_size,
                                                        mask=mask,
                                                        rms=rms,
                                                        data_2D=data_2D,
                                                        data_res=data_res,
                                                        vmin_factor=vmin_factor,
                                                        # results=results_final,
                                                        show_figure=show_figure,
                                                        bkg_to_sub=bkg_to_sub,
                                                        verbose=verbose,
                                                        add_save_name=add_save_name,
                                                        SAVE=SAVE,logger=logger)
    # r_list, area_arr, area_beam, p, \
    #     flux_arr, results_final = do_petrofit(imagename, cell_size,
    #                                           mask_component=mask_component,
    #                                           PLOT=do_PLOT,
    #                                           sigma_mask=sigma_mask,
    #                                           dilation_size=dilation_size,
    #                                           npixels=npixels, fwhm=fwhm,
    #                                           kernel_size=kernel_size,
    #                                           results=results_final,
    #                                           apply_mask=apply_mask)
    omask = omask2.copy()
    error_petro = False
    if do_petro == True:
        try:
            if verbose >= 1:
                if logger is not None:
                    logger.info(f"  CALC >> Computing Petrosian properties.")
                else:
                    print('     >> CALC: Computing Petrosian properties.')
            r_list, area_arr, area_beam, p, flux_arr, error_arr, results_final, cat, \
                segm, segm_deblend, sorted_idx_list = \
                compute_petrosian_properties(data_2D, imagename,
                                                mask_component=mask_component,
                                                global_mask=mask,
                                                source_props=results_final,
                                                apply_mask=False,
                                                # error = ,
                                                sigma_level=sigma_mask,
                                                bkg_sub=bkg_sub, bkg_to_sub=bkg_to_sub,
                                                vmin=vmin_factor, plot=do_PLOT,
                                                deblend=deblend,
                                                fwhm=fwhm, kernel_size=kernel_size,
                                                show_figure=show_figure,
                                                verbose = verbose,
                                                add_save_name=add_save_name,
                                                npixels=npixels,logger=logger)
            error_petro = False
        except Exception as e:
            if logger is not None:
                logger.warning(f"  -->> ERROR when computing Petrosian properties. "
                                f"Will flag error_petro as True.")
            else:
                print("     -->> ERROR when computing Petrosian properties. Will "
                        "flag error_petro as True.")
            error_petro = True
    else:
        error_petro = True

    results_final['error_petro'] = error_petro
    if mask_component is not None:
        centre = nd.maximum_position(data_2D * mask_component)[::-1]
        r, ir = get_profile(load_fits_data(imagename) * mask_component)
    else:
        centre = nd.maximum_position(data_2D)[::-1]
        r, ir = get_profile(data_2D,center=centre)
    #     rpix = r / cell_size
    rpix = r.copy()

    if z is not None:
        z = z
    else:
        z = 0.01
    if error_petro == False:
        r_list_arcsec = r_list * cell_size
        Rp_arcsec = results_final['Rp'] * cell_size
        R50_arcsec = results_final['R50'] * cell_size
        pix_to_pc = pixsize_to_pc(z=z, cell_size=cell_size)
        r_list_pc = r_list * pix_to_pc
        Rp_pc = results_final['Rp'] * pix_to_pc
        R50_pc = results_final['R50'] * pix_to_pc
        Rp_arcsec = results_final['Rp'] * cell_size
        R50_arcsec = results_final['R50'] * cell_size
        r_list_arcsec = r_list * cell_size
        Rp_arcsec = results_final['Rp'] * cell_size
        R50_arcsec = results_final['R50'] * cell_size
    if error_petro == True:
        r_list_arcsec = rpix * cell_size
        Rp_arcsec = results_final['C95radii'] * cell_size
        R50_arcsec = results_final['C50radii'] * cell_size
        pix_to_pc = pixsize_to_pc(z=z, cell_size=cell_size)
        r_list_pc = rpix * pix_to_pc
        Rp_pc = results_final['C95radii'] * pix_to_pc
        R50_pc = results_final['C50radii'] * pix_to_pc
        Rp_arcsec = results_final['C95radii'] * cell_size
        R50_arcsec = results_final['C50radii'] * cell_size
        r_list_arcsec = rpix * cell_size
        Rp_arcsec = results_final['C95radii'] * cell_size
        R50_arcsec = results_final['C50radii'] * cell_size

    
    if do_measurements=='all':
        try:
            x0c, y0c = results_final['x0m'], results_final['y0m']
            if compute_A == True:
                print('--==>> Computing asymetries...')
                results_final = compute_asymetries(imagename=imagename,
                                                mask=mask,
                                                bkg_to_sub=bkg_to_sub,
                                                mask_component=mask_component,
                                                centre=(x0c, y0c),
                                                results=results_final)

            # idx_R50 = np.where(flux_arr < 0.5 * results_final['flux_rp'])[0][-1]
            # idx_Rp = np.where(r_list < 2 * results_final['Rp'])[0][-1]
            # idx_Cradii = np.where(r_list < results_final['Cradii'])[0][-1]
            # idx_C50radii = np.where(r_list < results_final['C50radii'])[0][-1]

            # idx_R50 = np.where(flux_arr < 0.5 * results_final['flux_rp'])[0][-1]
            # flux_arr[idx_R50], r_list[idx_R50], results_final['R50']
            # if mask_component is None:
            if verbose >= 1:
                print('--==>> Computing image statistics...')
            results_final = get_image_statistics(imagename=imagename,
                                                mask_component=mask_component,
                                                mask=mask,
                                                sigma_mask=sigma_mask,
                                                apply_mask=False,
                                                residual_name=residualname, fracX=fracX,
                                                fracY=fracY,
                                                dic_data=results_final,
                                                cell_size=cell_size)
        except:
            pass
    

    results_final = cosmo_stats(imagename=imagename, z=z, results=results_final)

    results_final['pix_to_pc'] = pix_to_pc
    results_final['cell_size'] = cell_size
    if error_petro == False:
        results_final['area_beam_2Rp'] = area_beam[int(results_final['Rp'])]
        results_final['area_beam_R50'] = area_beam[int(results_final['R50'])]
    if error_petro == True:
        results_final['area_beam_2Rp'] = results_final['A95']/beam_area2(imagename)
        results_final['area_beam_R50'] = results_final['A50']/beam_area2(imagename)

    df = pd.DataFrame.from_dict(results_final, orient='index').T
    df.to_csv(imagename.replace('.fits', add_save_name + '_stats.csv'),
              header=True,
              index=False)
    return (results_final, mask, omask)



def check_flux_duplicates(cumulative_flux, radii):
    """
    Check if the array of fluxes contain duplicated values.
    """
    cumulative_flux_norm = cumulative_flux / cumulative_flux[-1]
    
    unique_elements = np.unique(cumulative_flux_norm)
    has_duplicates = len(unique_elements) != len(cumulative_flux_norm)
    if has_duplicates:
        # raise ValueError("Cumulative flux contains duplicates.")
        # print("Cumulative flux contains duplicates.")
        # print(".... attempting removal of duplicates.")
        unique_flux, indices = np.unique(cumulative_flux_norm, return_inverse=True)
        average_radii = np.zeros_like(unique_flux, dtype=float)
        average_cumulative_flux = np.zeros_like(unique_flux, dtype=float)
        # Calculate average of y-values corresponding to each unique x-value
        for i in range(len(unique_flux)):
            average_radii[i] = np.mean(radii[indices == i])
            average_cumulative_flux[i] = np.mean(cumulative_flux[indices == i])
            
        cumulative_flux_norm = unique_flux.copy()
        radii = average_radii.copy()
        cumulative_flux = average_cumulative_flux.copy()
        
    return(cumulative_flux,radii)


def find_fractional_radius(cumulative_flux, radii, 
                           fraction,do_plot=False):
    """
    Calculate the radius that encloses a given fraction of the total flux.
    
    Parameters:
    - cumulative_flux: numpy array of cumulative flux densities
    - radii: numpy array of radii corresponding to the cumulative flux densities
    - fraction: the fraction of the total flux for which to calculate the radius
    
    Returns:
    - radius_at_fraction: the radius that encloses the specified fraction of the total flux
    """
    from scipy.interpolate import interp1d
    # Ensure the fraction is between 0 and 1
    if not (0 < fraction < 1):
        raise ValueError("Fraction must be between 0 and 1.")

    cumulative_flux,radii = check_flux_duplicates(cumulative_flux,radii)
    cumulative_flux_norm = cumulative_flux / cumulative_flux[-1]
    # Calculate the total flux to find the target flux for the given fraction
    total_flux = cumulative_flux_norm[-1]
    total_flux_true= cumulative_flux[-1]
    fraction_flux = total_flux * fraction
    fraction_flux_true = total_flux_true * fraction
    
    # Use spline interpolation to create a smooth function of cumulative flux vs radius
    spline = interp1d(cumulative_flux_norm, radii, kind='linear')
    
    # Use the interpolated function to find the radius for the target flux
    radius_at_fraction = spline(fraction_flux)

    area_at_fraction = radii_to_area(radius_at_fraction)[0]

    if fraction >=0.95:
        dr_err_up = 0.025
    else:
        dr_err_up = 0.05
    if fraction <= 0.1:
        dr_err_lo = 0.025
    else:
        dr_err_lo = 0.05
    
    # Calculate the lower and upper fractions for the error range
    lower_fraction = max(fraction - dr_err_lo, 0)
    upper_fraction = min(fraction + dr_err_up, 1)

    # Calculate the flux values for the lower and upper fractions
    lower_fraction_flux = lower_fraction * total_flux
    upper_fraction_flux = upper_fraction * total_flux

    # Calculate the radii at the lower and upper fractions using the spline interpolation
    lower_radius = spline(lower_fraction_flux)
    upper_radius = spline(upper_fraction_flux)

    # Calculate the error range as the difference between the lower and upper radii
    radius_error_range = (abs(upper_radius-radius_at_fraction),abs(lower_radius-radius_at_fraction))
    # print(f"Fractional Radius with error range: {radius_at_fraction:.2f} +/- {radius_error_range[1]:.2f}/{radius_error_range[0]:.2f}")
    fraction_radius_error_mean = np.mean(radius_error_range)
    # Calculate the areas at the lower and upper radii
    lower_area = radii_to_area(lower_radius)[0]
    upper_area = radii_to_area(upper_radius)[0]

    # Calculate the area error range as the difference between the lower and upper areas
    area_error_range = (abs(upper_area - area_at_fraction), abs(lower_area - area_at_fraction))
    # print(f"Fractional Area with error range: {area_at_fraction:.2f} +/- {area_error_range[1]:.2f}/{area_error_range[0]:.2f}")
    # print(area_error_range[0],area_error_range[1],np.mean(area_error_range))
    fraction_area_error_mean = np.mean(area_error_range)
    if do_plot:
        # Plot the data points
        plt.plot(radii, cumulative_flux_norm, 'o', label='Data Points')
        
        # Create a smooth range of flux values for plotting the interpolation
        flux_range = np.linspace(np.min(cumulative_flux_norm), np.max(cumulative_flux_norm), 1000)
        
        # Plot the spline interpolation
        plt.plot(spline(flux_range), flux_range, '-', label='Spline Interpolation')
        
        # Plot a cross at the fractional radius
        plt.plot(radius_at_fraction, fraction_flux, 'rx', markersize=10, label=f'Fractional Radius ({fraction*100:.1f}%)')
        plt.axvline(lower_radius)
        plt.axvline(radius_at_fraction)
        plt.axvline(upper_radius)
        
        # Add labels and legend
        plt.ylabel('Cumulative Flux')
        plt.xlabel('Radius')
        plt.legend()
        
        # Show the plot
        plt.show()
    
    return(radius_at_fraction, fraction_flux_true, area_at_fraction, 
           fraction_radius_error_mean, fraction_area_error_mean)

def R50_to_fwhm(R50,R50_err,q_ratio=1,scale = 1):
    """
    Calculate the beam size in arcseconds for a given R50 value.
    (Lucatelli et al., 2024)
    """
    theta_maj = 2*R50*scale
    theta_min = theta_maj*q_ratio
    
    theta_maj_err = 2*R50_err*scale
    theta_min_err = theta_maj_err*q_ratio
    
    theta = (theta_maj*theta_min)**0.5
    theta_err = (theta_maj_err*theta_min_err)**0.5
    return(theta_maj,theta_maj_err,
           theta_min,theta_min_err,
           theta,theta_err)


def concentration_index_errors(C80radii, C20radii, 
                               C90radii, C50radii, 
                               C80radii_err, C20radii_err, 
                               C90radii_err, C50radii_err):
    """
    Given fractional radii and their errors, calculate the concentration indices
    and their errors.
    
    Parameters:
    - C80radii: the radius enclosing 80% of the total flux
    - C20radii: the radius enclosing 20% of the total flux
    - C90radii: the radius enclosing 90% of the total flux
    - C50radii: the radius enclosing 50% of the total flux
    - C80radii_err: the error in the radius enclosing 80% of the total flux
    - C20radii_err: the error in the radius enclosing 20% of the total flux
    - C90radii_err: the error in the radius enclosing 90% of the total flux
    - C50radii_err: the error in the radius enclosing 50% of the total flux
    
    Returns:
    - C1: the concentration index C1
    - C1_err: the error in the concentration index C1
    - C2: the concentration index C2
    - C2_err: the error in the concentration index C2
    """
    C1 = np.log10(C80radii / C20radii)
    C2 = np.log10(C90radii / C50radii)

    dC1_dC80radii = 1 / (C80radii * np.log(10))
    dC1_dC20radii = -1 / (C20radii * np.log(10))
    C1_err = np.sqrt((dC1_dC80radii * C80radii_err)**2 + (dC1_dC20radii * C20radii_err)**2)

    dC2_dC90radii = 1 / (C90radii * np.log(10))
    dC2_dC50radii = -1 / (C50radii * np.log(10))
    C2_err = np.sqrt((dC2_dC90radii * C90radii_err)**2 + (dC2_dC50radii * C50radii_err)**2)
    # print(f"C1 = {C1:.2f} +/- {C1_err:.2f}")
    # print(f"C2 = {C2:.2f} +/- {C2_err:.2f}")
    return C1, C1_err, C2, C2_err


# def deprecated(new_func_name):
#     def decorator(old_func):
#         def wrapper(*args, **kwargs):
#             warnings.warn(f"'{old_func.__name__}' is deprecated and will be "
#                           f"removed in a future version. "
#                           f"Use '{new_func_name}' instead.",
#                           category=DeprecationWarning, stacklevel=2)
#             return old_func(*args, **kwargs)
#         return wrapper
#     return decorator



def compute_image_properties(img, residual, cell_size=None, mask_component=None,
                             aspect=1, last_level=1.5, mask=None, 
                             bins = 64,
                             data_2D=None,data_res=None,
                             dilation_size=None, iterations=2,
                             dilation_type='disk',do_fit_ellipse=True,
                             sigma_mask=6, rms=None, results=None, bkg_to_sub=None,
                             apply_mask=True, vmin_factor=3, vmax_factor=0.5,
                             crop=False, box_size=256,
                             SAVE=True, add_save_name='', show_figure=True,
                             save_csv = False,
                             ext='.jpg',logger=None,verbose=0):
    """
    Params
    ------
    img : str
        The path to the image to be analyzed.
    residual : str
        The path to the residual image associated with the image.
    cell_size : float, optional
        The default is None. The size of the pixel in arcseconds.
        If None, `get_cell_size` will be used to get from the header.
    mask_component : 2D np array, optional
        The default is None. If not None, this is the mask for a specific component
        when performing a multi-component source analysis.
    aspect : float, optional (experimental)
        The default is 1. The aspect ratio of the image for plotting.
    last_level : float, optional
        New threshold level (as multiple of sigma_mad) to be used inside the existing mask.
    bins : int, optional
        Number of bins to slice intensity levels of the image. 
    mask : 2D np array, optional
        The default is None. If not None, the function `mask_dilation` will determine it with
        default parameters.
    data_2D : 2D np array, optional
        The default is None. This can be used to pass a 2D array directly to the function.
        For example, when calculating properties from an array without reading from a file,
        e.g. a model image, you can use this. But, to obtain meaningful physical units,
        you must provide the corresponding image file where this array was derived from.
    dilation_size : int, optional
        The default is None. The size of the dilation to be used in the mask dilation.
        If None, the default value will be the size of the restoring beam.
    iterations : int, optional
        The default is 2. The number of iterations to be used in the mask dilation.
        If signs of over-dilation are present, you can set to 1.
    dilation_type : str, optional
        The default is 'disk'. The type of dilation to be used in the mask dilation.
    sigma_mask : float, optional
        The default is 6. The sigma level to be used in the mask dilation.
    rms : float, optional
        The default is None. The rms value to be used in the mask dilation.
        If None, the function `mad_std` will be used to calculate it from the residual image,
        if provided. If the residual image is not provided, the function will use the image itself.
        But, in that case, the result may not be accurate (overestimated) if the image size is
        comparable in size to the size of the source structure.
    results : dict, optional
        The default is None. A dictionary to store the results.
        You can pass an existing external dictionary, so the results will be appended to it.
    bkg_to_sub : 2D np array, optional (EXPERIMENTAL)
        The default is None. The background to be subtracted from the image.
    apply_mask : bool, optional
        The default is True. If True, the mask dilation will be calcualted from the image.
    vmin_factor : float, optional
        The default is 3. The factor (as a multiple of sigma_mad) to be used in the vmin
        calculation for the image plot.
    vmax_factor : float, optional
        The default is 0.5. The factor (as a multiple of peak brightness) to be used in the vmax
        calculation for the image plot.
    crop : bool, optional
        The default is False. If True, the image will be cropped to a box_size.
    box_size : int, optional
        The default is 256. The size of the box to be used in the image cropping.
    SAVE : bool, optional
        The default is True. If True, the image plot will be saved to a file.
    add_save_name : str, optional
        The default is ''. A string to be added to the image plot file name.
    show_figure : bool, optional
        The default is True. If True, the image plot will be shown.
    ext : str, optional
        The default is '.jpg'. The file extension to be used in the image plot file name.
    logger : logging.Logger, optional
        The default is None. A logger object to be used to log messages.

    """
    if results is None:
        results = {}
        results['#imagename'] = os.path.basename(img)

    from skimage.draw import disk
    if data_2D is not None:
        g_ = data_2D
    else:
        g_ = load_fits_data(img)
    # res_ = load_fits_data(residual)
    
    g = np.nan_to_num(g_.copy(),nan=0)
    # res = res_.copy()

    if bkg_to_sub is not None:
        g = g - bkg_to_sub


    if cell_size is None:
        try:
            cell_size = get_cell_size(img)
        except:
            cell_size = 1.0

    g_hd = imhead(img)
    freq = g_hd['refval'][2] / 1e9
    # print(freq)
    omaj = g_hd['restoringbeam']['major']['value']
    omin = g_hd['restoringbeam']['minor']['value']
    # beam_area_ = beam_area(omaj, omin, cellsize=cell_size)
    beam_area_ = beam_area2(img)

    if rms is not None:
        std = rms
    else:
        if residual is not None:
            std = mad_std(np.nan_to_num(load_fits_data(residual), nan=0))
        else:
            std = mad_std(np.nan_to_num(g_.copy(),nan=0))
    if mask is not None:
        mask = mask
        omask = mask
        g = g * mask  # *(g_>3*mad_std(g_)) + 0.1*mad_std(g_)
        # res = res * mask  # *(g_>3*mad_std(g_)) + 0.1*mad_std(g_)
        # total_flux = np.sum(g_ * (g_ > 3 * std)) / beam_area_
        total_flux = np.sum(g) / beam_area_
        total_flux_nomask = np.sum(g_) / beam_area_
        apply_mask = False

    if apply_mask == True:
        if data_res is None:
            omask, mask = mask_dilation(img, sigma=sigma_mask,
                                        dilation_size=dilation_size,
                                        iterations=iterations,
                                        dilation_type=dilation_type,
                                        show_figure=show_figure,
                                        verbose=verbose)
        else:
            omask, mask = mask_dilation(data_2D, sigma=sigma_mask,
                                        dilation_size=dilation_size,
                                        iterations=iterations,
                                        dilation_type=dilation_type,
                                        show_figure=show_figure,
                                        verbose=verbose)
    

        # eimshow(mask)
        g = g * mask  # *(g_>3*mad_std(g_)) + 0.1*mad_std(g_)
        # res = res * mask
        # total_flux = np.sum(g_ * (g_ > 3 * std)) / beam_area_
        total_flux = np.nansum(g) / beam_area_
        total_flux_nomask = np.nansum(g_) / beam_area_
        # g = g_

    if mask_component is not None:
        #
        g = g * mask_component

        # res = res * mask_component

        total_flux_nomask = np.nansum(g) / beam_area_
        # total_flux = np.sum(g * (g > 3 * std)) / beam_area_
        total_flux = np.nansum(g) / beam_area_
        if mask is not None:
            mask = mask * mask_component
            omask = mask * mask_component
        else:
            mask = mask_component
            omask = mask_component

    # if (apply_mask is None) and  (mask is None):
    #     mask = mask_component
        # g = g_
    if (mask_component is None) and (apply_mask == False) and (mask is None):
        # g = g_
        total_flux = np.nansum(g * (g > 3 * std)) / beam_area_
        total_flux_nomask = np.nansum(g) / beam_area_

        unity_mask = np.ones(g.shape) == 1
        omask, mask = unity_mask, unity_mask

    # total_flux_nomask = np.sum(g_) / beam_area_
    # if (mask is None) and (mask_component is not None):
    #     mask = mask_component


    
    results['total_flux_nomask'] = total_flux_nomask
    results['total_flux_mask'] = total_flux
    results['peak_of_flux'] = np.nanmax(g)
    results['bmajor'] = omaj
    results['bminor'] = omin
    results['freq'] = freq*1e9
    
    
    # flux_mc, flux_error_m = mc_flux_error(img, g,
    #               res,
    #               num_threads=6, n_samples=1000)
    #
    # results['flux_mc'] = flux_mc
    # results['flux_error_mc'] = flux_error_m

    if residual is not None:
        systematic_error_fraction = 0.05
        if data_res is None: 
            data_res = load_fits_data(residual)
        else:
            data_res = data_res
        
        data_res = np.nan_to_num(data_res,nan=0)
        flux_res_error = 3 * np.sum(data_res * mask) / beam_area_
        # rms_res =imstat(residual_name)['flux'][0]

        total_flux_density_residual = np.nansum(data_res * mask) / beam_area_
        res_error_rms =np.sqrt(
            np.nansum((abs(data_res * mask -
                        np.nanmean(data_res * mask))) ** 2 * np.nansum(mask))) / beam_area_

        # Calculate systematic error as a fraction of the total flux density of the image
        systematic_error = systematic_error_fraction * results['total_flux_mask']

        # Calculate total flux density error in quadrature
        total_flux_density_error = np.sqrt(
            systematic_error ** 2 + (std / beam_area_) ** 2 + total_flux_density_residual ** 2)
        # print('total_flux_density_residual = ',total_flux_density_residual)
        # print('systematic_error = ', systematic_error)
        # print('(std / beam_area_) = ', (std / beam_area_))
        if verbose >= 1:
            print('-----------------------------------------------------------------')
            print('Flux Density and error (based on rms of residual x area): ')
            print(f"Flux Density = {results['total_flux_mask'] * 1000:.2f} "
                  f"+/- {res_error_rms * 1000:.2f} mJy")
            print(f"Fractional error = {(res_error_rms / results['total_flux_mask']):.2f}")
            print('Flux Density and error (quadrature |fract_err + res_err + rms|): ')
            print(f"Flux Density = {results['total_flux_mask'] * 1000:.2f} "
                  f"+/- {total_flux_density_error * 1000:.2f} mJy")
            print(f"Fractional error = {(total_flux_density_error / results['total_flux_mask']):.2f}")
            print('-----------------------------------------------------------------')

        results['max_residual'] = np.nanmax(data_res * mask)
        results['min_residual'] = np.nanmin(data_res * mask)
        results['flux_residual'] = total_flux_density_residual
        results['flux_error_res'] = abs(flux_res_error)
        results['flux_error_res_2'] = abs(res_error_rms)
        results['flux_error_res_3'] = abs(total_flux_density_error)
        results['mad_std_residual'] = mad_std(data_res)
        results['rms_residual'] = rms_estimate(data_res)

    try:
        # this should not be linspace, should be spaced in a logarithmic sense!!
        # print('np.nanmax(g)',np.nanmax(g))
        # print('last_level * std',last_level * std)
        # plt.figure()
        # plt.imshow(g)
        # plt.show()
        levels = np.geomspace(np.nanmax(g), last_level * std,bins)
        levels2 = np.geomspace(np.nanmax(g), last_level * std,100)
        levels_ellipse = np.geomspace(np.nanmax(g), last_level * std,32)
    except:
        levels = np.geomspace(np.nanmax(g), last_level * np.nanstd(g),bins)
        levels2 = np.geomspace(np.nanmax(g), last_level * np.nanstd(g),100)
        levels_ellipse = np.geomspace(np.nanmax(g), last_level * np.nanstd(g),32)

    fluxes = []
    areas = []
    for i in range(len(levels)):
        if i == 0:
            # condition = (g >= levels[i])
            # flux = g[np.where(condition)].sum() / beam_area_
            condition = (g >= levels[i])
            flux = np.nansum(g * (condition)) / beam_area_

            area = np.nansum(condition)
            fluxes.append(flux)
            areas.append(area)
        else:
            # condition = (g < levels[i - 1]) & (g >= levels[i])
            # flux = g[np.where(condition)].sum() / beam_area_
            condition = ((g < levels[i - 1]) & (g >= levels[i]))
            flux = np.nansum(g * (condition)) / beam_area_
            area = np.nansum(condition)
            # area = np.sum((g >= levels[i]))
            fluxes.append(flux)
            areas.append(area)
    fluxes = np.asarray(fluxes)
    areas = np.asarray(areas)
    #     print()
    #     plt.scatter(levels[:],fluxes[:])#/np.sum(fluxes))

    """
    Growth curve.
    """
    
    agrow = areas.copy()
    results['total_flux_levels'] = np.nansum(fluxes)


    # agrow_beam = agrow / beam_area_
    Lgrow = np.nancumsum(fluxes)
    # Lgrow_norm = Lgrow / fluxes.sum()
    _radii = []
    for i in range(0, len(levels)):
        mask_at_level_i = g >= levels[i]
        circular_radii = np.sqrt(np.nansum(mask_at_level_i) / np.pi)
        # print(circular_radii)
        _radii.append(circular_radii)
    radii = np.asarray(_radii)
    # print(Lgrow_norm)
    Lgrow,radii = check_flux_duplicates(Lgrow,radii)
    Lgrow_norm = Lgrow / np.nansum(fluxes)

    ####################################################################
    ## THIS NEEDS A BETTER IMPLEMENTATION USING SPLINE!!!!  ############
    ####################################################################
    mask_L20 = Lgrow_norm < 0.2
    mask_L50 = Lgrow_norm < 0.5
    mask_L80 = Lgrow_norm < 0.8
    try:
        mask_L90 = (Lgrow_norm > 0.89) & (Lgrow_norm < 0.91)
        mask_L95 = (Lgrow_norm > 0.95) & (Lgrow_norm < 0.97)
    except:
        mask_L90 = (Lgrow_norm > 0.85) & (
                    Lgrow_norm < 0.95)  # in case there not enough pixels
        mask_L95 = (Lgrow_norm > 0.92) & (
                    Lgrow_norm < 0.97)  # in case there not enough pixels

    mask_L20_idx = [i for i, x in enumerate(mask_L20) if x]
    mask_L50_idx = [i for i, x in enumerate(mask_L50) if x]
    mask_L80_idx = [i for i, x in enumerate(mask_L80) if x]
    mask_L90_idx = [i for i, x in enumerate(mask_L90) if x]
    mask_L95_idx = [i for i, x in enumerate(mask_L95) if x]
    """
    The tries and exceptions bellow are temporary solutions to avoid 
    errors when there are not enough pixels or no signal at all, when computing 
    the growth curve.    
    """
    try:
        sigma_20 = levels[mask_L20_idx[-1]]
        flag20 = False
    except:
        flag20 = True
        try:
            sigma_20 = levels[mask_L50_idx[-1]]
        except:
            sigma_20 = last_level * std

    try:
        sigma_50 = levels[mask_L50_idx[-1]]
        flag50 = False
    except:
        flag50 = True
        sigma_50 = last_level * std
    try:
        sigma_80 = levels[mask_L80_idx[-1]]
        sigma_90 = levels[mask_L90_idx[-1]]
        sigma_95 = levels[mask_L95_idx[-1]]
        flag9095 = False
    except:
        sigma_80 = last_level * std
        sigma_90 = last_level * std
        sigma_95 = last_level * std
        flag9095 = True

    # up_low_std = (np.max(gal) / (3 * mad_std(gal))) * 0.7
    # inner_shell_mask = ((gal<(up_low_std+5)*mad_std(gal)) & (gal>up_low_std*mad_std(gal)))
    # inner_shell_mask = ((g <sigma_50) & (g >sigma_50*0.95))
    # outer_shell_mask = ((g < sigma_90) & (g > sigma_90 * 0.95))

    inner_mask = (g > (sigma_50)) * mask
    outer_mask90 = (g > (sigma_90)) * mask
    inner_perimeter = perimeter_crofton(inner_mask, 4)
    outer_perimeter90 = perimeter_crofton(outer_mask90, 4)
    outer_perimeter = perimeter_crofton(mask, 4)

    # Geometry
    x0max, y0max = peak_center(g * mask)
    results['x0'], results['y0'] = x0max, y0max
    # determine momentum centres.
    x0m, y0m, _, _ = momenta(g * mask, PArad_0=None, q_0=None)
    results['x0m'], results['y0m'] = x0m, y0m

    if do_fit_ellipse:
        try:
            # some geometrical measures
            # calculate PA and axis-ratio
            # region_split = [i for i, x in enumerate(levels > sigma_50) if x][-1]
            region_split = [i for i, x in enumerate(levels_ellipse > sigma_50) if x][-1]
            
            PA, q, x0col, y0col, PAm, qm, \
                PAmi, qmi, PAmo, qmo, \
                x0median, y0median, \
                x0median_i, y0median_i, \
                x0median_o, y0median_o,profiles = cal_PA_q(g * mask, Isequence=levels_ellipse,
                                                region_split=region_split,
                                                # SAVENAME=img.replace('.fits','_ellipsefit') + ext
                                                SAVENAME=img.replace('.fits','_ellipsefit')
                                                )
        except:
                PA, q, x0col, y0col, PAm, qm = 0.0, 0.0, x0max,y0max, 0.0, 0.0
                PAmi, qmi, PAmo, qmo = 0.0, 0.0, 0.0, 0.0
                x0median, y0median = x0max,y0max
                x0median_i, y0median_i = x0max,y0max
                x0median_o, y0median_o = x0max,y0max
                profiles = None

        results['PA'], results['q'] = PA, q
        results['PAm'], results['qm'] = PAm, qm
        results['PAmi'], results['qmi'] = PAmi, qmi
        results['PAmo'], results['qmo'] = PAmo, qmo
        results['x0m_i'], results['y0m_i'] = x0median_i, y0median_i
        results['x0m_o'], results['y0m_o'] = x0median_o, y0median_o

        vx = results['x0'] - results['x0m']
        vy = results['y0'] - results['y0m']
        TvPA, Tvlenght = trail_vector(vx=vx, vy=vy, v0=np.asarray([1, 0]))
        results['TvPA'] = TvPA
        results['Tvlenght'] = Tvlenght

    try:
        L20_norm = Lgrow_norm[mask_L20_idx[-1]]  # ~ 0.2
        L20 = Lgrow[mask_L20_idx[-1]]
    except:
        try:
            L20_norm = Lgrow_norm[mask_L50_idx[-1]]  # ~ 0.2
            L20 = Lgrow[mask_L50_idx[-1]]
        except:
            L20 = 0.0
            L20_norm = 0.9999
    try:
        L50_norm = Lgrow_norm[mask_L50_idx[-1]]  # ~ 0.5
        L50 = Lgrow[mask_L50_idx[-1]]
    except:
        L50_norm = 0.9999
        L50 = 0.0
    try:
        L80_norm = Lgrow_norm[mask_L80_idx[-1]]  # ~ 0.8
        L80 = Lgrow[mask_L80_idx[-1]]
    except:
        L80_norm = 0.9999
        L80 = 0.0
    try:
        """
        Not enough pixels
        """
        L90_norm = Lgrow_norm[mask_L90_idx[-1]]  # ~ 0.9
        L90 = Lgrow[mask_L90_idx[-1]]
        L95_norm = Lgrow_norm[mask_L95_idx[-1]]  # ~ 0.9
        L95 = Lgrow[mask_L95_idx[-1]]
        flagL9095 = False

    except:
        flagL9095 = True
        try:
            try:
                L90_norm = Lgrow_norm[mask_L80_idx[-1]]  # ~ 0.9
                L90 = Lgrow[mask_L80_idx[-1]]
                L95_norm = Lgrow_norm[mask_L80_idx[-1]]  # ~ 0.9
                L95 = Lgrow[mask_L80_idx[-1]]
            except:
                L90_norm = Lgrow_norm[-1]  # ~ 0.9
                L90 = Lgrow[-1]
                L95_norm = Lgrow_norm[-1]  # ~ 0.9
                L95 = Lgrow[-1]

        except:
            L90_norm = 0.9999
            L90 = fluxes.sum()
            L95_norm = 0.9999
            L95 = fluxes.sum()


    try:
        TB20 = T_B(omaj, omin, freq, L20)
    except:
        TB20 = 0.0
    TB50 = T_B(omaj, omin, freq, L50)
    TB80 = T_B(omaj, omin, freq, L80)
    TB90 = T_B(omaj, omin, freq, L90)

    TB = T_B(omaj, omin, freq, total_flux)

    levels_20 = np.asarray([sigma_20])
    levels_50 = np.asarray([sigma_50])
    levels_80 = np.asarray([sigma_80])
    levels_90 = np.asarray([sigma_90])
    levels_95 = np.asarray([sigma_95])

    levels_3sigma = np.asarray([3 * std])

    g20 = ((g * mask) > sigma_20)
    g50 = ((g * mask) > sigma_50)
    g80 = ((g * mask) > sigma_80)
    g90 = ((g * mask) > sigma_90)
    g95 = ((g * mask) > sigma_95)

    try:
        C20radii, L20, npix20, C20radii_err, npix20_err = \
            find_fractional_radius(Lgrow, radii, fraction = 0.20)
        L20_norm = 0.20
    except:
        if logger is not None:
            logger.warning(f" !!==> Not enough pixels to calculate L20-R20."
                        f"       Using flux/size of first pixel.")
        C20radii = radii[0]
        L20 = Lgrow[0]
        L20_norm = Lgrow_norm[0]
        npix20 = radii_to_area(C20radii)[0]
        C20radii_err, npix20_err = C20radii/2, npix20/2
    
    try:
        if logger is not None:
            logger.warning(f" !!==> Not enough pixels to calculate L50-R50."
                           f"       Using flux/size of first pixel.")
        C50radii, L50, npix50, C50radii_err, npix50_err = \
            find_fractional_radius(Lgrow, radii, fraction = 0.50)
        L50_norm = 0.50
    except:
        C50radii = radii[0]
        L50 = Lgrow[0]
        L50_norm = Lgrow_norm[0]
        npix50 = radii_to_area(C50radii)[0]
        C50radii_err, npix50_err = C50radii/2, npix50/2
    
    C80radii, L80, npix80, C80radii_err, npix80_err = \
        find_fractional_radius(Lgrow, radii, fraction = 0.80)
    C90radii, L90, npix90, C90radii_err, npix90_err = \
        find_fractional_radius(Lgrow, radii, fraction = 0.90)
    C95radii, L95, npix95, C95radii_err, npix95_err = \
        find_fractional_radius(Lgrow, radii, fraction = 0.95)
    C99radii, L99, npix99, C99radii_err, npix99_err = \
        find_fractional_radius(Lgrow, radii, fraction = 0.99)

    # _frac_fluxes = np.asarray([0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
    #                            0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 0.99])
    # _frac_fluxes = np.asarray([0.85, 0.9 , 0.95, 0.99])
    # frac_radii  = {}
    # frac_fluxes = {}
    # for frac_flux in _frac_fluxes:
    #     # try:
    #     radius_percent, _, _ = find_fractional_radius(Lgrow, radii, fraction = frac_flux)
    #     frac_radii[f'R{int(frac_flux*100)}'] = radius_percent
    #     frac_fluxes[f'S{int(frac_flux*100)}'] = frac_flux
        # except:
        #     pass
    # print(frac_radii.keys())
    # print(frac_fluxes)
    # print(np.sqrt(((frac_radii['R99'] - frac_radii['R95'])**2.0 + (frac_radii['R90'] - frac_radii['R95'])**2.0  + (frac_radii['R95'] - frac_radii['R85'])**2.0)/3.0))
    
    # plt.figure()
    # plt.plot(frac_fluxes,frac_radii,'o')
    # plt.show()
    # np.mean(frac_radii)

    L80_norm = 0.80
    L90_norm = 0.90
    L95_norm = 0.95

    
    A20 = npix20 / beam_area_
    A50 = npix50 / beam_area_
    A80 = npix80 / beam_area_
    A90 = npix90 / beam_area_
    A95 = npix95 / beam_area_
    A99 = npix99 / beam_area_
    A20_err = npix20_err / beam_area_
    A50_err = npix50_err / beam_area_
    A80_err = npix80_err / beam_area_
    A90_err = npix90_err / beam_area_
    A95_err = npix95_err / beam_area_
    A99_err = npix99_err / beam_area_

    # A20, C20radii, npix20 = estimate_area((g > sigma_20) * mask, cell_size, omaj,
    #                                       omin)
    # A50, C50radii, npix50 = estimate_area((g > sigma_50) * mask, cell_size, omaj,
    #                                       omin)
    # A80, C80radii, npix80 = estimate_area((g > sigma_80) * mask, cell_size, omaj,
    #                                       omin)
    # A90, C90radii, npix90 = estimate_area((g > sigma_90) * mask, cell_size, omaj,
    #                                       omin)
    # A95, C95radii, npix95 = estimate_area((g > sigma_95) * mask, cell_size, omaj,
    #                                       omin)

    # if flag20 == True:
    #     if flag50 == False:
    #         A20, C20radii, npix20 = A50 / 2, C50radii / 2, npix50 / 2
    #     else:
    #         try:
    #             A20, C20radii, npix20 = A80 / 3, C80radii / 3, npix80 / 3
    #         except:
    #             A20, C20radii, npix20 = A95 / 4, C95radii / 4, npix95 / 4

    try:
        results['conv_P20'], results['conv_A20'] = convex_shape(g20)
    except:
        results['conv_P20'], results['conv_A20'] = 5,5

    try:
        results['conv_P50'], results['conv_A50'] = convex_shape(g50)
        results['conv_P80'], results['conv_A80'] = convex_shape(g80)
        results['conv_P90'], results['conv_A90'] = convex_shape(g90)
        results['conv_P95'], results['conv_A95'] = convex_shape(g95)
        results['conv_PT'], results['conv_AT'] = convex_shape(mask)
    except:
        results['conv_P50'], results['conv_A50'] = 5,5
        results['conv_P80'], results['conv_A80'] = 5,5
        results['conv_P90'], results['conv_A90'] = 5,5
        results['conv_P95'], results['conv_A95'] = 5, 5
        results['conv_PT'], results['conv_AT'] = 5,5


    # This is a more robust computation of source size.
    try:
        R20med,R20mean,R20std = calculate_radii(g, g20)
    except:
        R20med, R20mean, R20std = 1,1,1
    try:
        R50med,R50mean,R50std = calculate_radii(g, g50)
    except:
        R50med, R50mean, R50std = 1,1,1
    try:
        R80med,R80mean,R80std = calculate_radii(g, g80)
    except:
        R80med, R80mean, R80std = 1,1,1
    try:
        R90med,R90mean,R90std = calculate_radii(g, g90)
    except:
        R90med, R90mean, R90std = 1,1,1
    try:
        R95med,R95mean,R95std = calculate_radii(g, g95)
    except:
        R95med, R95mean, R95std = 1,1,1
    try:
        RTmed,RTmean,RTstd = calculate_radii(g, mask)
    except:
        RTmed, RTmean, RTstd = 1,1,1

    results['R20med'], results['R20mean'], \
        results['R20std'] = R20med,R20mean,R20std
    results['R50med'], results['R50mean'], \
        results['R50std'] = R50med,R50mean,R50std
    results['R80med'], results['R80mean'], \
        results['R80std'] = R80med,R80mean,R80std
    results['R90med'], results['R90mean'], \
        results['R90std'] = R90med,R90mean,R90std
    results['R95med'], results['R95mean'], \
        results['R95std'] = R95med,R95mean,R95std
    results['RTmed'], results['RTmean'], \
        results['RTstd'] = RTmed,RTmean,RTstd

    # # print(C20radii, C50radii, C80radii, C90radii)
    # C1 = np.log10(C80radii / C20radii)
    # C2 = np.log10(C90radii / C50radii)
    
    C1, C1_err, C2, C2_err = concentration_index_errors(C80radii, C20radii, C90radii, C50radii, 
                                                        C80radii_err, C20radii_err, C90radii_err, C50radii_err)

    AC1 = np.log10(A80 / A20)
    AC2 = np.log10(A90 / A50)

    CAC1 = np.log10(area_to_radii(results['conv_A80']) / area_to_radii(results['conv_A20']))
    CAC2 = np.log10(area_to_radii(results['conv_A90']) / area_to_radii(results['conv_A50']))



    area_total, Cradii, npix_total = estimate_area(mask, cell_size, omaj, omin)
    o_area_total, o_Cradii, o_npix_total = estimate_area(omask, cell_size, omaj,
                                                         omin)

    mask_outer = (g < sigma_50) & (g > last_level * std) * mask
    A50_100, C50_100radii, npix50_100 = estimate_area(mask_outer, cell_size,
                                                      omaj, omin)
    # gaussianity = 1 / (L50 / g.max())
    gaussianity_L50 = g.max() / L50
    gaussianity = sigma_50 / g.max()

    mask_outer_full = (g < sigma_50) * mask
    #     plt.imshow(mask_outer_full)
    A50_full, C50_full_radii, npix50_full = estimate_area(mask_outer_full,
                                                          cell_size, omaj, omin)

    radii_ratio = C50radii / C50_100radii
    radii_ratio_full = C50radii / C50_full_radii
    area_ratio = A50 / A50_100
    area_ratio_full = A50 / A50_full

    results['max'] = np.nanmax(g)
    results['std_image'] = std
    try:
        results['std_residual'] = mad_std(load_fits_data(residual))
    except:
        pass

    
    results['L50'] = L50
    results['sigma_50'] = sigma_50
    results['sigma_90'] = sigma_90
    results['sigma_95'] = sigma_95
    results['o_area_total'] = o_area_total
    results['o_Cradii'] = o_Cradii
    results['o_npix_total'] = o_npix_total
    results['area_total'] = area_total
    results['Cradii'] = Cradii
    results['npix_total'] = npix_total
    results['inner_perimeter'] = inner_perimeter
    results['outer_perimeter'] = outer_perimeter
    results['outer_perimeter90'] = outer_perimeter90

    results['TB20'] = TB20
    results['TB50'] = TB50
    results['TB80'] = TB80
    results['TB'] = TB

    #     results['nbeams_total'] = nbeams_total

    results['C1'] = C1
    results['C2'] = C2
    results['C1_err'] = C1_err
    results['C2_err'] = C2_err
    results['AC1'] = AC1
    results['AC2'] = AC2
    results['CAC1'] = CAC1
    results['CAC2'] = CAC2

    results['L20'] = L20
    results['A20'] = A20
    results['C20'] = sigma_20 / std
    results['C20radii'] = C20radii
    results['npix20'] = npix20
    results['C20radii_err'] = C20radii_err
    results['npix20_err'] = npix20_err

    results['L50'] = L50
    results['A50'] = A50
    results['C50'] = sigma_50 / std
    results['C50radii'] = C50radii
    results['C50radii_err'] = C50radii_err
    results['npix50'] = npix50
    results['npix50_err'] = npix50_err
    
    results['C50theta'] = C50radii * np.sqrt(6 * np.log(2))
    results['C50theta_err'] = C50radii_err * np.sqrt(6 * np.log(2))
    results['C50theta_dec'] = np.sqrt(results['C50theta']**2.0 - (results['bmajor']/cell_size)**2.0)
    # results['C50theta_dec_err'] = np.sqrt(results['C50theta_err']**2.0 - (results['bmajor']/cell_size)**2.0)
    results['C50theta_dec_err'] = results['C50theta_dec'] * results['C50theta_err']
    
    
    

    results['L80'] = L80
    results['A80'] = A80
    results['C80'] = sigma_80 / std
    results['C80radii'] = C80radii
    results['npix80'] = npix80
    results['C80radii_err'] = C80radii_err
    results['npix80_err'] = npix80_err

    results['L90'] = L90
    results['A90'] = A90
    results['C90'] = sigma_90 / std
    results['C90radii'] = C90radii
    results['npix90'] = npix90
    results['C90radii_err'] = C90radii_err
    results['npix90_err'] = npix90_err
    results['flag20'] = flag20
    results['flag50'] = flag50
    results['flag9095'] = flag9095
    results['flagL9095'] = flagL9095


    results['L95'] = L95
    results['A95'] = A95
    results['C95'] = sigma_95 / std
    results['C95radii'] = C95radii
    results['npix95'] = npix95
    results['C95radii_err'] = C95radii_err
    results['npix95_err'] = npix95_err

    results['L99'] = L99
    results['A99'] = A99
    results['C99radii'] = C99radii
    results['npix99'] = npix99
    results['C99radii_err'] = C99radii_err
    results['npix99_err'] = npix99_err
    
    results['A20_err'] = A20_err
    results['A50_err'] = A50_err
    results['A80_err'] = A80_err
    results['A90_err'] = A90_err
    results['A95_err'] = A95_err
    results['A99_err'] = A99_err


    results['gaussianity'] = gaussianity
    results['gaussianity_L50'] = gaussianity_L50

    results['A50_100'] = A50_100
    results['C50_100radii'] = C50_100radii
    results['npix50_100'] = npix50_100

    results['A50_full'] = A50_full
    results['C50_full_radii'] = C50_full_radii
    results['npix50_full'] = npix50_full

    results['radii_ratio'] = radii_ratio
    results['radii_ratio_full'] = radii_ratio_full
    results['area_ratio'] = area_ratio
    results['area_ratio_full'] = area_ratio_full
    results['beam_area'] = beam_area_

    if logger is not None:
        print_logger_header(title="Basic Source Properties",
                            logger=logger)
        logger.debug(f" ==>  Peak of Flux="
                     f"{results['peak_of_flux']*1000:.2f} [mJy/beam]")
        logger.debug(f" ==>  Total Flux Inside Mask='"
                     f"{results['total_flux_mask']*1000:.2f} [mJy]")
        logger.debug(f" ==>  Total Flux Image="
                     f"{results['total_flux_nomask'] * 1000:.2f} [mJy]")
        logger.debug(f" ==>  Half-Light Radii="
                     f"{results['C50radii']:.2f} [px]")
        logger.debug(f" ==>  Total Source Size="
                     f"{results['C95radii']:.2f} [px]")
        if do_fit_ellipse:
            logger.debug(f" ==>  Source Global Axis Ratio="
                        f"{results['qm']:.2f}")
            logger.debug(f" ==>  Source Global PA="
                        f"{results['PAm']:.2f} [degrees]")
            logger.debug(f" ==>  Inner Axis Ratio="
                        f"{results['qmi']:.2f}")
            logger.debug(f" ==>  Outer Axis Ratio="
                        f"{results['qmo']:.2f}")
            logger.debug(f" ==>  Inner PA="
                        f"{results['PAmi']:.2f} [degrees]")
            logger.debug(f" ==>  Outer PA="
                        f"{results['PAmo']:.2f} [degrees]")



    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(radii, Lgrow / results['total_flux_mask'],
                label='Norm Masked Flux')
    ax1.axhline(0, ls='-.', color='black')
    # ax1.axvline(g.max()*0.5,label=r'$0.5\times \max$',color='purple')
    # ax1.axvline(g.max()*0.1,label=r'$0.1\times \max$',color='#E69F00')
    ax1.axvline(C50radii,
                label=r"$R_{50}\sim $"f"{C50radii*cell_size:0.3f}+/-"f"{C50radii_err*cell_size:0.3f}''",
                ls='-.', color='lime')
    ax1.axhline(L50_norm, ls='-.', color='lime')
    # ax1.plot(C50radii, 
    #            0.5, 'rx', markersize=10, 
    #         label=r"$R_{50}\sim $"f"{C50radii*cell_size:0.3f}''",
    #         ls='-.', color='lime')
    # ax1.axhline(L50_norm, ls='-.', color='lime')
    ax1.axvline(C95radii,
                label=r"$R_{95}\sim $"f"{C95radii*cell_size:0.3f}+/-"f"{C95radii_err*cell_size:0.3f}''",
                # ls='--',
                color='#56B4E9')
    # ax1.axhline(0.95, ls='-.', color='#56B4E9')
    # ax1.axvline(std * 6, label=r"6.0$\times \sigma_{\mathrm{mad}}$", color='black')
    # if last_level<3:
    #     ax1.axvline(std * 3, label=r"3.0$\times \sigma_{\mathrm{mad}}$", color='brown')

    ax1.set_title("Total Integrated Flux Density \n "
                  r"($\sigma_{\mathrm{mad}}$ levels) = "
                  f"{1000*np.sum(fluxes):.2f} $\pm$ {1000*total_flux_density_error:.2f} mJy")
    #     ax1.axvline(mad_std(g)*1,label=r'$1.0\times$ std',color='gray')
    # ax1.axvline(levels[-1], label=r"Mask Dilation",
    #             color='cyan')

    ax1.set_xlabel(fr'Aperture Circular Radius $R$ [px]')
    ax1.set_ylabel(fr"$S(\leq R)$")
    # ax1.semilogx()
    ax1.grid(alpha=0.5)
    # plt.xlim(1e-6,)
    ax1.legend(loc='lower right')


    # fig = plt.figure(figsize=(10, 4))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.scatter(levels[:], np.cumsum(fluxes) / results['total_flux_mask'],
    #             label='Mask Norm Flux')

    # ax1.axhline(0, ls='-.', color='black')
    # # ax1.axvline(g.max()*0.5,label=r'$0.5\times \max$',color='purple')
    # # ax1.axvline(g.max()*0.1,label=r'$0.1\times \max$',color='#E69F00')
    # ax1.axvline(sigma_50,
    #             label=r"$R_{50}\sim $"f"{C50radii*cell_size:0.3f}''",
    #             ls='-.', color='lime')
    # ax1.axhline(L50_norm, ls='-.', color='lime')
    # ax1.axvline(sigma_95,
    #             label=r"$R_{95}\sim $"f"{C95radii*cell_size:0.3f}''",
    #             # ls='--',
    #             color='#56B4E9')
    # ax1.axvline(std * 6, label=r"6.0$\times \sigma_{\mathrm{mad}}$", color='black')
    # if last_level<3:
    #     ax1.axvline(std * 3, label=r"3.0$\times \sigma_{\mathrm{mad}}$", color='brown')

    # ax1.set_title("Total Integrated Flux Density \n "
    #               r"($\sigma_{\mathrm{mad}}$ levels) = "
    #               f"{1000*np.sum(fluxes):.2f} mJy")
    # #     ax1.axvline(mad_std(g)*1,label=r'$1.0\times$ std',color='gray')
    # ax1.axvline(levels[-1], label=r"Mask Dilation",
    #             color='cyan')

    # ax1.set_xlabel('Levels [Jy/Beam]')
    # ax1.set_ylabel("Fraction of Integrated Flux per level")
    # ax1.semilogx()
    # ax1.grid(alpha=0.5)
    # # plt.xlim(1e-6,)
    # ax1.legend(loc='lower left')

    ax2 = fig.add_subplot(1, 2, 2)

    vmin = vmin_factor * std
    #     print(g)
    vmax = vmax_factor * g.max()
    norm = simple_norm(g, stretch='asinh', asinh_a=0.05, min_cut=vmin,
                       max_cut=vmax)

    if crop == True:
        try:
            xin, xen, yin, yen = do_cutout(img, box_size=box_size, center=center,
                                           return_='box')
            g = g[xin:xen, yin:yen]
        except:
            try:
                max_x, max_y = np.where(g == g.max())
                xin = max_x[0] - box_size
                xen = max_x[0] + box_size
                yin = max_y[0] - box_size
                yen = max_y[0] + box_size
                g = g[xin:xen, yin:yen]
            except:
                pass

    im_plot = ax2.imshow(g, cmap='magma_r', origin='lower', alpha=1.0,
                         norm=norm,
                         aspect=aspect)  # ,vmax=vmax, vmin=vmin)#norm=norm

    # levels_50 = np.asarray([-3*sigma_50,sigma_50,3*sigma_50])

    try:
        ax2.contour(g, levels=levels_50, colors='lime', linewidths=2.5,
                    alpha=1.0)  # cmap='Reds', linewidths=0.75)
        # ax2.contour(g, levels=levels_90, colors='white', linewidths=2.0,
        #             # linestyles='--',
        #             alpha=1.0)  # cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=levels_95, colors='#56B4E9', linewidths=2.0,
                    # linestyles='--',
                    alpha=1.0)  # cmap='Reds', linewidths=0.75)
        #         ax2.contour(g, levels=levels_3sigma,colors='#D55E00',
        #         linewidths=1.5,alpha=1.0)#cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=[last_level * std], colors='cyan', linewidths=0.6,
                    alpha=1.0,
                    # linestyles='--'
                    )  # cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=[6.0 * std], colors='black', linewidths=1.5,
                    alpha=0.9,
                    # linestyles='--'
                    )  # cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=[3.0 * std], colors='brown', linewidths=1.2,
                    alpha=0.9,
                    # linestyles='--'
                    )  # cmap='Reds', linewidths=0.75)
        # ax2.contour(g, levels=[5.0 * std], colors='brown', linewidths=0.6,
        #             alpha=0.3,
        #             # linestyles='--'
        #             )  # cmap='Reds', linewidths=0.75)

    except:
        print('Not plotting contours!')
    ax2.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    if SAVE is not None:
        plt.savefig(img.replace('.fits', '_Lgrow_levels')+add_save_name + ext,
                    dpi=300,bbox_inches='tight')
    if show_figure == True:
        plt.show()
    else:
        plt.close()

    if save_csv == True:
        import csv
        # df_temp = pd.DataFrame(results).T
        # df_temp.to_csv(img.replace('.fits', '_image_properties')+add_save_name + '.csv',
        #                header=True,index=False)

        with open(img.replace('.fits', '_image_properties')+add_save_name + '.csv',
                  'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
    return (levels, fluxes, agrow, omask, mask, results)


def structural_morphology(imagelist, residuallist,
                          indices, masks_deblended,
                          zd, big_mask=None, data_2D=None,sigma_mask=6.0,
                          iterations = 2,
                          do_PLOT=False, show_figure=False,
                          sigma_loop_init=6.0, do_measurements='all',
                          verbose=0):
    """
    From the emission  of a given source and its deblended components,
    run in each component the morphometry analysis.

    A list of images is accepted with a common deblended mask for all.

    This was originally intended to be used with multi-resolution images.
    First, a common-representative image is processed with a source detection and
    deblending algorithm. Then, those detected/deblended regions are used to
    run a forced morphometry on all multi-resolution images.

    << Finish documentation >>

    """
    results_conc = []
    missing_data_im = []
    missing_data_re = []
    masks = []
    for i in tqdm(range(len(imagelist))):
        try:
            crop_image = imagelist[i]
            data_2D = load_fits_data(crop_image)

            if residuallist is not None:
                crop_residual = residuallist[i]
                residual_2D = load_fits_data(crop_residual)
                std = mad_std(residual_2D)
                if verbose >= 1:
                    print('Using RMS from residual')
            else:
                crop_residual = None
                residual_2D = None
                std = mad_std(data_2D)
                if verbose >= 1:
                    print('Using RMS from data ')
            #         crop_residual = residuallist[i]
            cell_size = get_cell_size(crop_image)
            std = mad_std(data_2D)
            npixels = int(2*beam_area2(crop_image))
            #             residual_2D = load_fits_data(crop_residual)



            processing_results_source = {}  # store calculations only for source
            processing_results_source['#imagename'] = os.path.basename(
                crop_image)
            # first, run the analysis for the entire source structure
            processing_results_source, mask, _ = measures(imagename=crop_image,
                                                          residualname=crop_residual,
                                                          z=zd, deblend=False,
                                                          apply_mask=True,
                                                        #   mask = big_mask,
                                                          results_final=processing_results_source,
                                                          plot_catalog=False,
                                                          rms=std,
                                                          bkg_sub=False,
                                                          bkg_to_sub=None,
                                                          mask_component=None,
                                                          npixels=npixels, fwhm=121,
                                                          kernel_size=121,
                                                          sigma_mask=sigma_mask,
                                                          last_level=1.5,
                                                          iterations=iterations,
                                                          dilation_size=None,
                                                          do_measurements=do_measurements,
                                                          do_PLOT=do_PLOT,
                                                          show_figure=show_figure,
                                                          add_save_name='',
                                                          verbose=verbose)
            flag_subcomponent = 0
            processing_results_source['freq'] = getfreqs([crop_image])[0]
            processing_results_source['comp_ID'] = str(0)
            processing_results_source['flag_subcomponent'] = flag_subcomponent
            # ref_mask = mask*big_mask
            
            results_conc.append(processing_results_source)
            # bkg_ = sep_background(crop_image, apply_mask=True, mask=None, bw=11,
            #                       bh=11, fw=12, fh=12)

            omaj, omin, _, _, _ = beam_shape(crop_image)
            dilation_size = int(0.5*np.sqrt(omaj * omin) / (2 * get_cell_size(crop_image)))
            # print('dilation_size=', dilation_size)
            # masks_expanded = {}
            
            if len(indices) > 1:
                for j in range(len(indices)):
                    # ii = str(i+1)
                    sigma_loop = sigma_loop_init  # reset the loop
                    processing_results_components = {}  # store calculation only for individual components of the soruce
                    processing_results_components['#imagename'] = os.path.basename(crop_image)
                    

                    mask_component = masks_deblended[j]
                    data_component = mask_component*data_2D.copy()
                    add_save_name = 'comp_' + str(j+1)
                    # print('Component id ', processing_results_components['comp_ID'])
                    try:
                        # mask_new = mask_component.copy()
                        _, mask_new = \
                            mask_dilation_from_mask(data_2D,
                                                    mask_component,
                                                    sigma=sigma_loop,
                                                    PLOT=False,iterations=iterations,
                                                    dilation_size=dilation_size,
                                                    show_figure=False)

                        # dilated masks must not overlap >> non conservation of flux
                        # mask_new = mask_new * big_mask
                        for l in range(len(indices)):
                            if l != j:
                                mask_new[masks_deblended[l]] = False
                            else:
                                pass
                        # masks_expanded[f'mask_ex_{j}'] = mask_new
                        # plt.figure()
                        # plt.imshow(mask_new*big_mask)
                        # plt.show()
                        processing_results_components, mask, _ = \
                            measures(crop_image, crop_residual, z=zd,
                                        deblend=False, apply_mask=False,
                                        plot_catalog=False, bkg_sub=False,
                                        mask = big_mask,
                                        mask_component=mask_new, rms=std,
                                        iterations=iterations, npixels=50, fwhm=121,
                                        kernel_size=121, sigma_mask=sigma_loop,
                                        last_level=1.5,
                                        # bkg_to_sub = bkg_.back(),
                                        dilation_size=dilation_size,
                                        add_save_name=add_save_name,
                                        do_measurements=do_measurements,
                                        do_PLOT=do_PLOT, show_figure=show_figure,
                                        verbose=verbose,
                                        results_final=processing_results_components)
                        flag_subcomponent = 0
                        # print('Component id ', processing_results_components['comp_ID'])
                        processing_results_components['freq'] = getfreqs([crop_image])[0]
                        processing_results_components['comp_ID'] = str(j+1)
                        processing_results_components['flag_subcomponent'] = flag_subcomponent
                    except:
                        try:
                            error_mask = True
                            while error_mask and sigma_loop > 1.0:
                                try:
                                    # mask_new = mask_component.copy()
                                    _, mask_new = \
                                        mask_dilation_from_mask(load_fits_data(crop_image),
                                                                mask_component,
                                                                rms=std,
                                                                sigma=sigma_loop,
                                                                iterations=3,
                                                                PLOT=False,
                                                                dilation_size=dilation_size,
                                                                show_figure=False)

                                    # dilated masks must not overlap >> non conservation of flux
                                    for l in range(len(indices)):
                                        if l != j:
                                            mask_new[masks_deblended[l]] = False
                                        else:
                                            pass
                                    
                                    # masks_expanded[f'mask_ex_{j}'] = mask_new
                                    
                                    if sigma_loop >= 3.0:
                                        last_level = 1.5
                                    if sigma_loop < 3.0:
                                        last_level = sigma_loop - 0.5

                                    (processing_results_components, mask,
                                     _) = measures(crop_image, crop_residual,
                                                   z=zd, deblend=False,
                                                   apply_mask=False,
                                                   plot_catalog=False,
                                                   bkg_sub=False,
                                                   mask = big_mask,
                                                   mask_component=mask_new,
                                                   rms=std, iterations=3,
                                                   npixels=1000, fwhm=121,
                                                   kernel_size=121,
                                                   sigma_mask=sigma_loop,
                                                   last_level=last_level,
                                                   add_save_name=add_save_name,
                                                   dilation_size=dilation_size,
                                                   do_measurements=do_measurements,
                                                   do_PLOT=do_PLOT, 
                                                   show_figure=show_figure,
                                                   verbose=verbose,
                                                   results_final=processing_results_components)
                                    # print('Component id ', processing_results_components['comp_ID'])
                                    processing_results_components['subreg_sigma'] = sigma_loop
                                    error_mask = False
                                    flag_subcomponent = 1
                                    processing_results_components['freq'] = getfreqs([crop_image])[0]
                                    processing_results_components['comp_ID'] = str(j+1)
                                    processing_results_components['flag_subcomponent'] = flag_subcomponent
                                except Exception as e:
                                    # Handle the error, and decrease p by 0.5
                                    print(
                                        f"Error occurred with sigma={sigma_loop}: {e}")
                                    sigma_loop -= 0.5
                                    print("Reducing sigma to=", sigma_loop)

                            if not error_mask:
                                print("Function call successful with sigma_mad=",
                                      sigma_loop)
                            else:
                                print(
                                    "Unable to call function with any value of sigma_mad.")
                        except:
                            print(
                                'Last attempt to perform morphometry, '
                                'with mininum threshold allowed.')
                            processing_results_components, mask, _ = measures(
                                crop_image, crop_residual, z=zd, deblend=False,
                                apply_mask=False,mask = big_mask,
                                plot_catalog=False, bkg_sub=False,
                                # bkg_to_sub = bkg_.back(),
                                mask_component=mask_new, rms=std,
                                dilation_size=dilation_size, iterations=2,
                                npixels=int(beam_area2(crop_image)),
                                fwhm=81, kernel_size=81, sigma_mask=2.0,
                                last_level=0.5,
                                add_save_name=add_save_name,
                                do_measurements=do_measurements,
                                do_PLOT=do_PLOT, show_figure=show_figure,
                                verbose=verbose,
                                results_final=processing_results_components)
                            processing_results_components['subreg_sigma'] = 1.0
                            processing_results_components['comp_ID'] = str(j+1)
                            processing_results_components['freq'] = getfreqs([crop_image])[0]
                            flag_subcomponent = 1
                            processing_results_components['flag_subcomponent'] = flag_subcomponent

                    results_conc.append(processing_results_components)
                    # masks.append(masks_expanded)
                processing_results_source['ncomps'] = len(indices)
            else:
                processing_results_source['ncomps'] = 1
        except:
            print('Some imaging data is missing!')
            missing_data_im.append(os.path.basename(crop_image))
            missing_data_re.append(os.path.basename(crop_residual))
            pass
    return (pd.DataFrame(results_conc), processing_results_source, missing_data_im)



make_flux_vs_std = deprecated("make_flux_vs_std",
                              "compute_image_properties")(compute_image_properties)

"""
 __  __                  _                          _              
|  \/  | ___  _ __ _ __ | |__   ___  _ __ ___   ___| |_ _ __ _   _ 
| |\/| |/ _ \| '__| '_ \| '_ \ / _ \| '_ ` _ \ / _ \ __| '__| | | |
| |  | | (_) | |  | |_) | | | | (_) | | | | | |  __/ |_| |  | |_| |
|_|  |_|\___/|_|  | .__/|_| |_|\___/|_| |_| |_|\___|\__|_|   \__, |
                  |_|                                        |___/
                  
#Morphometry 
"""

def background_asymmetry(img, mask, pre_clean=False):
    """
    <<<Morfometryka-core part>>>
    """
    def measure_asymmetry_patch(pos, patch, img):
        (x0, y0) = pos
        rot_cell = rot180(patch, x0, y0)
        sub = patch - rot_cell
        return np.sum(abs(sub)) / np.sum(abs(mask * img))

    Mo,No = img.shape
    gridsize = Mo // 10  # 10% of the size of the image
    n_pix = gridsize ** 2
    xcells = Mo // gridsize
    ycells = No // gridsize
    asymmetry_grid = np.zeros((xcells, ycells))
    gal_area = mask.sum()

    for xi in range(xcells):
        for yi in range(ycells):
            cell_mask = mask[xi * gridsize:(xi + 1) * gridsize,
                        yi * gridsize:(yi + 1) * gridsize]

            if cell_mask.sum() > 0:
                asymmetry_grid[xi, yi] = 0
                continue

            cell_img = img[xi * gridsize:(xi + 1) * gridsize, yi * gridsize:(yi + 1) * gridsize]
            x0, y0 = fmin(measure_asymmetry_patch, (gridsize // 2, gridsize // 2), args=(cell_img, img), disp=0)
            asymmetry_grid[xi, yi] = (gal_area / n_pix) * measure_asymmetry_patch((x0, y0), cell_img, img)
            del cell_img

    linear = asymmetry_grid[np.where(asymmetry_grid != 0)].ravel()

    if len(linear) > 0:
        BGrandom = np.random.choice(linear, 1)[0]
        BGmedian = np.median(linear)
        BGmin = linear.min()
        BGstd = np.std(linear)
        position = np.where(asymmetry_grid == linear.min())
        x0 = position[1][0] * gridsize + gridsize // 2
        y0 = position[0][0] * gridsize + gridsize // 2

    elif pre_clean == False:
        # measure background asymmetry with original pre-clean image if it fails for the clean one
        return background_asymmetry(img, mask, pre_clean=True)
    else:
        '''
           This is a fallback for when something goes wrong with background asymmetry estimates.
           It should also appear as a QF.
        '''
        BGrandom = 0
        BGmedian = 0
        BGmin = 0
        BGstd = 0
        x0 = 0
        y0 = 0

    return BGrandom, BGmedian, BGmin, BGstd, x0, y0


def assimetria0(pos, img, mask, box=False):
    """
    <<<Morfometryka-core part>>>
    """
    # print(' @ - Computing Asymetry 0')
    (x0, y0) = pos
    # psfmask = psfmask(psfsigma, *img.shape, x0, y0)
    if (box):
        boxmask = np.zeros_like(img)
        try:
            radii_px = np.ceil(self.P.Rp * self.NRp)
        except:
            radii_px = np.ceil(self.P.Rp * 1.5)
        boxmask[int(x0 - radii_px):int(x0 + radii_px), int(y0 - radii_px):int(y0 + radii_px)] = 1
        imgorig = boxmask * img
        imgsub = boxmask * (img - rot180(img, x0, y0))
        A = np.sum(abs(imgsub)) / np.sum(abs(imgorig))
    else:
        imgorig = img * mask
        imgsub = (img - rot180(img, x0, y0)) * mask
        A = np.sum(abs(imgsub)) / np.sum(abs(imgorig))

    del imgorig, imgsub
    return A


def assimetria1(pos, img, mask,use_mask=True):
    """
    <<<Morfometryka-core part>>>
    """
    # print(' @ - Computing Asymetry 0')
    x0, y0 = pos
    A1img = np.abs(img - rot180(img, x0, y0)) / (np.sum(np.abs(img)))
    if use_mask==True:
        return np.sum(mask * A1img)
    else:
        AsySigma = 3.00
        A1mask = A1img > np.median(A1img) + AsySigma * mad_std(A1img)
        return np.sum(mask * A1mask * A1img)


def geo_mom(p, q, I, centered=True, normed=True, complex=False, verbose=False):
    """
    <<<Morfometryka-core part>>>
    return the central moment M_{p,q} of image I
    http://en.wikipedia.org/wiki/Image_moment
    F.Ferrari 2012, prior to 4th JPAS
    """


    M, N = I.shape
    x, y = np.meshgrid(np.arange(N), np.arange(M))

    M_00 = np.nansum(I)

    if centered:
        # centroids
        x_c = (1 / M_00) * np.nansum(x * I)
        y_c = (1 / M_00) * np.nansum(y * I)

        x = x - x_c
        y = y - y_c

        if verbose:
            print('centroid  at', x_c, y_c)

    if normed:
        NORM = M_00 ** (1 + (p + q) / 2.)
    else:
        NORM = 1.0

    if complex:
        XX = (x + y * 1j)
        YY = (x - y * 1j)
    else:
        XX = x
        YY = y

    M_pq = (1 / NORM) * np.nansum(XX ** p * YY ** q * I)

    return M_pq


def q_PA(image):
    """
    <<<Morfometryka-core part>>>
    Adapted version of momenta from main mfmtk.
    """
    m00 = geo_mom(0, 0, image, centered=0, normed=0)
    m10 = geo_mom(1, 0, image, centered=0, normed=0)
    m01 = geo_mom(0, 1, image, centered=0, normed=0)
    m11 = geo_mom(1, 1, image, centered=0, normed=0)
    m20 = geo_mom(2, 0, image, centered=0, normed=0)
    m02 = geo_mom(0, 2, image, centered=0, normed=0)

    mu20 = geo_mom(2, 0, image, centered=1, normed=0)
    mu02 = geo_mom(0, 2, image, centered=1, normed=0)
    mu11 = geo_mom(1, 1, image, centered=1, normed=0)

    # centroids
    x0col = m10 / m00
    y0col = m01 / m00

    # manor, minor and axis ratio
    lam1 = np.sqrt(abs((1 / 2.) * (mu20 + mu02 + np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2))) / m00)
    lam2 = np.sqrt(abs((1 / 2.) * (mu20 + mu02 - np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2))) / m00)
    a = max(lam1, lam2)
    b = min(lam1, lam2)

    PA = (1 / 2.) * np.arctan2(2 * mu11, (mu20 - mu02))
    if PA < 0:
        PA = PA + np.pi
    PAdeg = np.rad2deg(PA)
    a = a
    b = b
    q = b / a
    return PAdeg, b / a, x0col, y0col


def peak_center(image):
    """
    <<<Morfometryka-core part>>>
    """
    y0max, x0max = nd.maximum_position((image))
    try:
        """
        For very small images of emission, this function breaks. In that
        case, just return the the max position.
        """
        # size of peak region to consider in interpolarion for x_peak
        dp = 2
        CenterOffset = 10
        No, Mo = image.shape
        peakimage = image[y0max - dp:y0max + dp, x0max - dp:x0max + dp]
        m00 = geo_mom(0, 0, peakimage, centered=0, normed=0)
        m10 = geo_mom(1, 0, peakimage, centered=0, normed=0)
        m01 = geo_mom(0, 1, peakimage, centered=0, normed=0)

        x0peak = x0max + m10 / m00 - dp
        y0peak = y0max + m01 / m00 - dp

        # check if center is galaxy center, i.e., should be near the image center
        # otherwise apply a penalty to pixel value proportional to the center distance^2
        if np.sqrt((x0peak - No / 2.) ** 2 + (
                y0peak - Mo / 2.) ** 2) > CenterOffset:
            # define a penalty as we move from the center
            xx, yy = np.meshgrid(np.arange(No) - No / 2.,
                                 np.arange(Mo) - Mo / 2.)
            rr2 = xx ** 2 + yy ** 2
            y0peak, x0peak = nd.maximum_position((image / rr2))
        return (x0peak, y0peak)
    except:
        return (x0max, y0max)


def momenta(image, PArad_0=None, q_0=None):
    '''
    <<<Morfometryka-core part>>>
    Calculates center of mass, axis lengths and position angle
    '''

    m00 = geo_mom(0, 0, image, centered=0, normed=0)
    m10 = geo_mom(1, 0, image, centered=0, normed=0)
    m01 = geo_mom(0, 1, image, centered=0, normed=0)
    m11 = geo_mom(1, 1, image, centered=0, normed=0)
    m20 = geo_mom(2, 0, image, centered=0, normed=0)
    m02 = geo_mom(0, 2, image, centered=0, normed=0)

    mu20 = geo_mom(2, 0, image, centered=1, normed=0)
    mu02 = geo_mom(0, 2, image, centered=1, normed=0)
    mu11 = geo_mom(1, 1, image, centered=1, normed=0)

    # centroids
    x0col = m10 / m00
    y0col = m01 / m00

    # manor, minor and axis ratio
    lam1 = np.sqrt(abs((1 / 2.) *
                       (mu20 + mu02 + np.sqrt((mu20 - mu02) ** 2 +
                                              4 * mu11 ** 2))) / m00)
    lam2 = np.sqrt(abs((1 / 2.) *
                       (mu20 + mu02 - np.sqrt((mu20 - mu02) ** 2 +
                                              4 * mu11 ** 2))) / m00)
    a = max(lam1, lam2)
    b = min(lam1, lam2)

    PA = (1 / 2.) * np.arctan2(2 * mu11, (mu20 - mu02))
    if PA < 0:
        PA = PA + np.pi

    # self.PArad = PA
    # self.PAdeg = np.rad2deg(self.PArad)
    # self.q = self.b/self.a
    # self.PArad = PA
    # self.PAdeg = np.rad2deg(self.PArad)
    # self.q = self.b/self.a

    # mofified by lucatelli.
    """This will force mfmtk to do photometry for the given input PA
    this can be useful when we want to study how the light profile changes
    as function of PA or q. This was indented to explore the difference between the
    profiles of elliptical and spiral galaxies, which may not change soo much for the
    former while it may for the later.
    """

    if PArad_0 is None:
        PArad = PA  # + np.pi/2
    else:
        PArad = PArad_0

    PAdeg = np.rad2deg(PArad)

    if q_0 is None:
        q = b / a
    else:
        q = q_0
    return (x0col, y0col, q, PAdeg)


def cal_PA_q(gal_image_0,Isequence = None,region_split=None,SAVENAME=None):
    '''
    <<<Morfometryka-core part>>>
    Estimates inner and outer PA nad q=(b/a)
    '''
    # mean Inner q,  mean outer q,  mean Inner PA,  mean Outer PA
    # from fitEllipse import main_test2
    
    # qmi, qmo, PAmi, PAmo, qm, PAm,\
    #     x0median,y0median,x0median_i,y0median_i,\
    #     x0median_o,y0median_o = main_test2(gal_image_0,
    #                                        Isequence = Isequence,
    #                                        region_split=region_split,
    #                                        SAVENAME=SAVENAME)
    
    from fit_ellipse import fit_ellipse_to_galaxy
    qmi, qmo, PAmi, PAmo, qm, PAm,x0median,y0median,x0median_i,y0median_i,\
        x0median_o,y0median_o,profiles = \
            fit_ellipse_to_galaxy(gal_image_0,
                                  Isequence = Isequence,
                                  region_split=region_split,
                                #   plot_results=True,
                                #   plot_profiles=True,
                                  save_name=SAVENAME)

    # global PA,  global q
    PA, q, x0col, y0col = q_PA(gal_image_0)

    # print("Initial PA and q = ", PA, q)
    # print("Median PA and q = ", PAm, qm)
    # print("Inner-Mean PA and q = ", PAmi, qmi)
    # print("Outer-Mean PA and q = ", PAmo, qmo)
    return (PA, q, x0col, y0col, PAm, qm, PAmi, qmi, PAmo, qmo,
            x0median,y0median,x0median_i,y0median_i,x0median_o,y0median_o,profiles)



def savitzky_golay_2d(z, window_size, order, derivative=None):
    """
    <<<Morfometryka-core part>>>
    http://nbviewer.ipython.org/github/pv/SciPy-CookBook/blob/master/ipython/SavitzkyGolay.ipynb
    """

    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size ** 2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size ** 2, )

    # build matrix of system of equation
    A = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(np.flipud(z[1:half_size + 1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size - 1:-1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1:half_size + 1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size - 1:-1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1:half_size + 1, 1:half_size + 1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(np.flipud(np.fliplr(z[-half_size - 1:-1, -half_size - 1:-1])) - band)

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(np.flipud(Z[half_size + 1:2 * half_size + 1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(np.fliplr(Z[-half_size:, half_size + 1:2 * half_size + 1]) - band)

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')


def standartize(image, q, PArad, x0, y0):
    """
    <<<Morfometryka-core part>>>
    make a standard galaxy, id est, PA=0, q=1
    arguments are 'image' to be standartized and  its 'S' stamp and P phot classes
    """

    ##### rotate array
    R = np.array([[np.cos(PArad), np.sin(PArad)], [-np.sin(PArad), np.cos(PArad)]])

    ##### shear array
    S = np.diag([q, 1.])
    # SERSIC fit values
    # S = np.diag([self.Ss.qFit2D, 1.])

    # affine transform matrix, rotate then scale
    transform = np.dot(R, S)

    # where to transform about
    centro_i = (x0, y0)
    # contro_o: where to put center after
    centro_o = np.array(image.shape) / 2

    myoffset = centro_i - np.dot(transform, centro_o)
    bval = np.mean(image[-2:])
    stangal = nd.affine_transform(image, transform, offset=myoffset, order=2, cval=bval)

    return stangal


def polarim(image, origin=None, log=False):
    """
    <<<Morfometryka-core part>>>
    Reprojects a 2D numpy array ("image") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image.
    http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
    refactored by FF, 2013-2014 (see transpolar.py)
    """

    if origin is None:
        origin = np.array(image.shape) / 2.

    def cart2polar(x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        #         max_radii = np.sqrt(x.max()**2+y.max()**2)
        #         rscale = x.max()/max_radii
        #         tscale = y.max()/(2*np.pi)
        theta = np.arctan2(y, x)
        return r, theta

    def polar2cart(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def cart2logpolar(x, y, M=1):
        alpha = 0.01
        r = np.sqrt(x ** 2 + y ** 2)
        rho = M * np.log(r + alpha)
        theta = np.arctan2(y, x)
        return rho, theta

    def logpolar2cart(rho, theta, M=1):
        x = np.exp(rho / M) * np.cos(theta)
        y = np.exp(rho / M) * np.sin(theta)
        return x, y

    ny, nx = image.shape
    if origin is None:
        x0, y0 = (nx // 2, ny // 2)
        origin = (x0, y0)
    else:
        x0, y0 = origin

    # Determine that the min and max r and theta coords will be...
    x, y = np.meshgrid(np.arange(nx) - x0, np.arange(ny) - y0)  # ,sparse=True )

    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0]  # We need to shift the origin back to
    yi += origin[1]  # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi))  # (map_coordinates requires a 2xn array)

    zi = nd.map_coordinates(image, coords, order=1)  # ,prefilter=False)
    galpolar = zi.reshape((nx, ny))

    r_polar = r_i
    theta_polar = theta_i

    return galpolar, r_polar, theta_polar, x, y


def Gradindex(image_data, Rp=None):
    """
    <<<Morfometryka-core part>>>
    Gradient Index
    Calculates an index based on the image gradient magnitude and orientation

    SGwindow and SGorder are Savitsky-Golay filter parameters
    F. Ferrari, 2014
    """

    def sigma_func(params):
        '''
        <<<Morfometryka-core part>>>
        calculates the sigma psi with different parameters
        called by the minimization routine '''
        (x0, y0, q, PA) = params
        #### creates standardized image
        # using segmentation geometric parameters
        # galnostarsstd = standartize(P.galnostars, S.q, S.PArad, S.y0col, S.x0col)
        # using Sersic geometric parameters
        galnostarsstd = standartize(image_data, q, PA, x0, y0)

        # creates polar imagem
        galpolar, r_polar, theta_polar, _, _ = polarim(galnostarsstd)
        #     print(galpolar)
        # print '%.5f %.5f %.5f %.5f' % (x0, y0, q, PA)

        if Rp is None:
            galpolarpetro = galpolar[0: -1, :]
        else:
            galpolarpetro = galpolar[0: 2 * int(Rp), :]
        #     galpolarpetro =  galpolar[ 0   : int(config.NRp * P.Rp), : ]

        # circular_area_radius
        #     if min(galpolarpetro.shape) <= SGwindow:
        #         SGorder -= 1

        try:
            # dx,dy = np.gradient(savgol_filter(galpolarpetro, SGwindow, SGorder, 1))
            dx, dy = savitzky_golay_2d(galpolarpetro, SGwindow, SGorder, 'both')
        except:
            SGwindow = 5
        # dx,dy = np.gradient(savgol_filter(galpolarpetro, SGwindow, SGorder, 1))
        dx, dy = savitzky_golay_2d(galpolarpetro, SGwindow, 1, 'both')

        mag = np.sqrt(dx ** 2 + dy ** 2)
        # mag = dxdy
        magmask = mag > (np.median(mag))
        ort = np.arctan2(dy, dx)
        # ort = np.arctan(dxdy)
        ortn = (ort + np.pi) % (np.pi)

        psi = circmean(ortn[magmask])
        sigma_psi = circstd(ortn[magmask])

        return sigma_psi

    def sigma_func_eval(params):
        '''
        <<<Morfometryka-core part>>>
        calculates the sigma psi with different parameters
        called by the minimization routine '''
        (x0, y0, q, PA) = params
        #### creates standardized image
        # using segmentation geometric parameters
        # galnostarsstd = standartize(P.galnostars, S.q, S.PArad, S.y0col, S.x0col)
        # using Sersic geometric parameters
        galnostarsstd = standartize(image_data, q, PA, x0, y0)

        # creates polar imagem
        galpolar, r_polar, theta_polar, _, _ = polarim(galnostarsstd)
        #     print(galpolar)
        # print '%.5f %.5f %.5f %.5f' % (x0, y0, q, PA)

        #     galpolarpetro =  galpolar[ 0   : int(config.NRp * P.Rp), : ]
        if Rp is None:
            galpolarpetro = galpolar[0: -1, :]
        else:
            galpolarpetro = galpolar[0: 2 * int(Rp), :]
        # circular_area_radius
        #     if min(galpolarpetro.shape) <= SGwindow:
        #         SGorder -= 1

        try:
            # dx,dy = np.gradient(savgol_filter(galpolarpetro, SGwindow, SGorder, 1))
            dx, dy = savitzky_golay_2d(galpolarpetro, SGwindow, SGorder, 'both')
        except:
            SGwindow = 5
        # dx,dy = np.gradient(savgol_filter(galpolarpetro, SGwindow, SGorder, 1))
        dx, dy = savitzky_golay_2d(galpolarpetro, SGwindow, 1, 'both')

        mag = np.sqrt(dx ** 2 + dy ** 2)
        # mag = dxdy
        magmask = mag > (np.median(mag))
        ort = np.arctan2(dy, dx)
        # ort = np.arctan(dxdy)
        ortn = (ort + np.pi) % (np.pi)

        psi = circmean(ortn[magmask])
        sigma_psi = circstd(ortn[magmask])

        return sigma_psi, ort, ortn, dx, dy, mag

    # SAVITSKY-GOLAY parameters
    # polynom order
    SGorder = 5

    # SG window is galaxy_size/10 and must be odd
    # CAN'T BE RELATIVE TO IMAGE SIZE... MUST BE RELATIVE TO GALAXY SIZE
    # SGwindow = int(S.Mo/10.)
    # SGwindow = int(P.Rp/2.)
    SGwindow = 5
    #     SGwindow = int(circular_area_radius/2)
    if SGwindow % 2 == 0:
        SGwindow = SGwindow + 1

    #     image_data=load_fits_data(imagelist[1])

    PA, q, x0col, y0col, PAm, qm, PAmi, qmi, PAmo, qmo,profiles = cal_PA_q(image_data)

    x0sigma, y0sigma, qsigma, PAsigma = \
        fmin(sigma_func, (x0col, y0col, qm, np.deg2rad(PAm)), ftol=0.1, xtol=1.0, disp=0)
    sigma_psi, ort, ortn, dx, dy, mag = sigma_func_eval((x0sigma, y0sigma, qsigma, PAsigma))

    #     x0sigma, y0sigma, qsigma, PAsigma = \
    #                 fmin(sigma_func, (Ss.x0Fit2D, Ss.y0Fit2D, Ss.qFit2D, np.deg2rad(Ss.PAFit2D)), ftol=0.1, xtol=1.0, disp=0)

    #     #if qsigma > 1:
    #     #   qsigma = 1/qsigma
    #     #   PAsigma = PAsigma - np.pi/2.

    #     x0sigma, y0sigma, qsigma, PAsigma = \
    #                 fmin(sigma_funcseg, (Ss.x0Fit2D, Ss.y0Fit2D, Ss.qFit2D, np.deg2rad(Ss.PAFit2D)), ftol=0.1, xtol=1.0, disp=0)
    return (sigma_psi, ort, ortn, dx, dy, mag, PAm, qm)

"""
 ____              _
/ ___|  __ ___   _(_)_ __   __ _
\___ \ / _` \ \ / / | '_ \ / _` |
 ___) | (_| |\ V /| | | | | (_| |
|____/ \__,_| \_/ |_|_| |_|\__, |
                           |___/
"""


def save_results_csv(result_mini, save_name, ext='.csv', save_corr=True,
                     save_params=True):
    values = result_mini.params.valuesdict()
    if save_corr:
        try:
            covariance = result_mini.covar
            covar_df = pd.DataFrame(covariance, index=values.keys(),
                                    columns=values.keys())
            covar_df.to_csv(save_name + '_mini_corr' + ext, index_label='parameter')
        except:
            print('Error saving covariance matrix. Skiping...')

    if save_params:
        try:
            stderr = [result_mini.params[name].stderr for name in values.keys()]
            df = pd.DataFrame({'value': list(values.values()), 'stderr': stderr},
                              index=values.keys())
            df.to_csv(save_name + '_mini_params' + ext, index_label='parameter')
        except:
            print('Errors not present in mini, saving only parameters.')
            df = pd.DataFrame(result_mini.params.valuesdict(), index=['value'])
            df.T.to_csv(save_name + '_mini_params' + ext,
                        index_label='parameter')



"""
 ____  _               _          
|  _ \| |__  _   _ ___(_) ___ ___ 
| |_) | '_ \| | | / __| |/ __/ __|
|  __/| | | | |_| \__ \ | (__\__ \
|_|   |_| |_|\__, |___/_|\___|___/
             |___/  
             
#Physics
"""
def T_B(theta_maj, theta_min, freq, I):
    # https://science.nrao.edu/facilities/vla/proposing/TBconv
    # https://www.cv.nrao.edu/~sransom/web/Ch2.html
    # theta_maj/min are gaussian beam maj and min axes in units of arcsecs
    # freq is the central frequency in units of GHz
    # I is the total flux measured by the beam - peak flux
    # divifing I/theta_maj & min converts I(mJy/beam) to S/solid angle
    brightness_temp = 1000*1222 * I / ((freq ** 2) * (theta_maj * theta_min))

    return brightness_temp/1e5


def Tb_source(Snu,freq,theta1,theta2,z,
              Snu_err=None,theta1_err=None,theta2_err=None):
    """
    Compute the brightness temperature, provided the deconvolved model parameters
    having semi-major and semi-minor FWHM's theta1 and theta2.
    
    Parameters
    ----------
    Snu : float
        The flux density at the frequency nu.
    freq : float
        The frequency at which the flux density is measured.
    theta1 : float
        The deconvolved FWHM axis of the source.
    theta2 : float
        The deconvolved FWHM axis of the source.
    z : float
        The redshift of the source.
    
    Returns
    -------
    Tb : float
        The brightness temperature of the source divided by 1e5.
    """
    const = 1.8e9 * (1+z)*1000
    Tb = ((const * Snu)/(freq*freq*(theta1*1000)*(theta2*1000)))/1e5
    if (Snu_err is not None) and (theta1_err is not None) and (theta2_err is not None):
        Tb_err = Tb * np.sqrt((Snu_err/Snu)**2.0 
                              + (theta1_err/theta1)**2.0 
                              + (theta2_err/theta2)**2.0)
        return(Tb,Tb_err)
    else:
        return(Tb)

def Olim(Omaj,SNR):
    Olim_ = Omaj * np.sqrt((4*np.log(2)/np.pi) * np.log(SNR/(SNR-1)))
    return(Olim_)

def deconv_R50(R50conv,theta12):
    return(np.sqrt(4*R50conv**2.0 - theta12**2.0 )/2)

def phi_source(OH,SpH,OL,SpL):
    """
    ## Source Sizes
    If the circular Gaussian source is imaged with two different resolutions
    $\theta_H$ and $\theta_L$, the ratio of the image peak brightnesses is
    \begin{equation}
        \frac{S_p^{H}}{S_p^{L}} =
        \left(
        1 + \frac{\phi^2}{\theta_L^2}
        \right)
        \left(
        1 + \frac{\phi^2}{\theta_H^2}
        \right)^{-1}
    \end{equation}
    This equation can be solved for the source size
    \begin{align}
        \phi =
        \left[
            \frac{\theta_L^2 \theta_H^2 (S_p^L - S_p^H)}{\theta_L^2 S_p^H - \theta_H^2 S_p^L}
        \right]^{1/2}
    \end{align}

    """
    nume = ((OL**2) * (OH**2)) * (SpL - SpH)
    deno = OL*OL*SpH - OH*OH*SpL
    phi_size = np.sqrt(nume/deno)
    return(phi_size)

def get_size_params(df,pix_to_pc):
    Bb = (df['bmin_pc']/pix_to_pc)*df['cell_size']
    Ba = (df['bmaj_pc']/pix_to_pc)*df['cell_size']
    max_im = df['max']
    Bsize = np.sqrt(Bb*Ba)
    return(Bsize,max_im)

def LT(In,Rn,n):
    """
    Total luminosity of a Sersic Function

    """
    bn = 2*n - 1.0/3.0 + 4/(405*n)
    num = 2*np.pi*In*Rn*Rn*n*np.exp(bn)*scipy.special.gamma(2*n)
    den = bn**(2*n)
    return(num/den)


def D_Cfit(z):
    h = 0.687
    H0 = 100 * h  # km/(s * Mpc)
    c = 299792.458  # km/s
    # comoving distance Dc
    D_H0 = c / H0  # Mpc
    a = 1.0 / (1 + z)
    n1 = a / (1 - a)
    n2 = (1 - a) / (0.785 + a)
    n3 = (1 - a) / ((0.312 + a) ** 2.0)
    D_C = D_H0 / (n1 + 0.2278 + 0.2070 * n2 - 0.0158 * n3)
    DC_MPC = D_C
    return (DC_MPC)


def compute_Lnu(flux, z, alpha):
    dist_conversion_factor = 3.08567758128 * 1e24  # m #3.08567758128*(10**24) #cm
    # lum_conversion_factor =  10**(-26) # W/(m^2 Hz Jy)
    lum_conversion_factor = 1e-23
    # D_L = luminosity_distance_cosmo(z=z)
    D_C = comoving_distance_cosmo(z=z) * dist_conversion_factor
    # D_Lnu = D_C * ((1 + z) ** ((alpha + 1) / 2)) * dist_conversion_factor
    D_L = D_C * (1+z)
    # flux is in Jy
    L_nu = (4.0 * np.pi * (D_L ** 2.0)/((1+z)**(alpha+1))) * flux * lum_conversion_factor
    # print(f"Comoving distance DC= {D_C/dist_conversion_factor} Mpc")
    # print(f"Luminosity distance DL= {D_L/dist_conversion_factor} Mpc")
    # print(f"Spetral Luminosity Lnu = {L_nu}")
    # Ld = 4.0 * np.pi * D_L**2.0 * flux
    # L_nu_error = 0.0
    return (L_nu)

def compute_Lnu_v2(flux, z, alpha,freq,nu0=1.0):
    dist_conversion_factor = 3.08567758128 * 1e24  # m #3.08567758128*(10**24) #cm
    # lum_conversion_factor =  10**(-26) # W/(m^2 Hz Jy)
    lum_conversion_factor = 1e-23
    # D_L = luminosity_distance_cosmo(z=z)
    D_C = comoving_distance_cosmo(z=z) * dist_conversion_factor
    # D_Lnu = D_C * ((1 + z) ** ((alpha + 1) / 2)) * dist_conversion_factor
    D_L = D_C * (1+z)
    # flux is in Jy
    L_nu = (4.0 * np.pi * (D_L ** 2.0)/((1+z)**(alpha+1))) * ((freq/nu0)**alpha) *flux * lum_conversion_factor
    # print(f"Comoving distance DC= {D_C/dist_conversion_factor} Mpc")
    # print(f"Luminosity distance DL= {D_L/dist_conversion_factor} Mpc")
    # print(f"Spetral Luminosity Lnu = {L_nu}")
    # Ld = 4.0 * np.pi * D_L**2.0 * flux
    # L_nu_error = 0.0
    return (L_nu)


def tabatabaei(nu, al, z):
    thermal_frac = 1. / (1. + 13. * ((nu * (1 + z)) ** (0.1 + al)))
    return (thermal_frac)


# Calculate a luminosity, star formation rate, and uncertainties given a flux density
# Using equation relating synchrotron emission to star formation rate given in Murphy et. al (2011)
# Also using Condon & Matthews (2018) to calculate spectral luminosity distance
def calc_params(flux, flux_error, redshift, redshift_error, freqs, alphas=-0.85,
                allowthermal=False):
    # Defining symbols (sympy)
    z = Symbol('z')  # redshift
    zunc = Symbol('zunc')  # redshift uncertainty
    nu = Symbol('nu')  # frequency
    nuunc = Symbol('nuunc')  # frequency uncertainty
    al = Symbol('al')  # non-thermal spectral index alpha
    alunc = Symbol('alunc')  # alpha uncertainty
    f = Symbol('f')  # flux
    f_unc = Symbol('f_unc')  # flux uncertainty
    Ho = Symbol('Ho')  # hubble constant
    Ho_unc = Symbol('Ho_unc')  # hubble uncertainty

    # define speed of light and Hubble distance
    c = 299792.458  # km/s
    Dho = c / Ho

    # Define symbolic formluas for desired quantities
    # convenience definition from Murphy et al. paper
    a = 1 / (1 + z)
    # Comoving distance formula
    # 3E24 factor converts between cm and Mpc
    Dc = (Dho / (a / (1 - a) + 0.2278 + 0.2070 * (1 - a) / (
                0.785 + a) - 0.0158 * (1 - a) / (
                             (0.312 + a) ** 2))) * 3.08567758128 * (
                     10 ** 24)  # cm
    # luminosity distance formula
    Dl = (1 + z) * Dc  # cm
    # spectral luminosity distance
    Dl_nu = Dl * ((1 + z) ** (-(al + 1) / 2))
    # inverse square law to get luminosity
    Lumform = (4 * np.pi * f * (10 ** -23) * Dl_nu ** 2)  # ergs/s
    # SFR formula in solar masses/yr (Murphy et. al)
    kroupa_to_salpeter = 1.5
    # SFRform = kroupa_to_salpeter*(6.64e-29*(nu**(-al))*Lumform)
    if allowthermal:
        L_NT = Lumform / (1 + 1 / 10. * ((1 + z) * nu) ** (-0.1 - al))
        SFRform = 6.64e-29 * (nu ** (-al)) * L_NT
    else:
        SFRform = 6.64e-29 * (nu ** (-al)) * Lumform
    # luminosity uncertainty formula - simple error propagation
    Lum_stat_unc = ((diff(Lumform, f) * f_unc) ** 2) ** 0.5
    Lum_syst_unc = ((diff(Lumform, z) * zunc) ** 2 + (
                diff(Lumform, Ho) * Ho_unc) ** 2) ** 0.5
    # SFR uncertainty formula
    SFR_stat_uncertainty = ((diff(SFRform, f) * f_unc) ** 2) ** .5
    SFR_syst_uncertainty = ((diff(SFRform, z) * zunc) ** 2 + (
                diff(SFRform, Ho) * Ho_unc) ** 2 + (
                                        diff(SFRform, al) * alunc) ** 2) ** .5

    # Define constants
    Hubble = 67.8  # km/s/Mpc
    Hubble_unc = 2
    # freqs = 1.51976491105  # GHz
    freqsigs = 0
    alphasig = 0.05

    output = []

    # substitute in values into symbolic expressions
    # SFR values
    SF = SFRform.subs({nu: freqs, al: alphas, f: flux, z: redshift, Ho: Hubble})
    SF_stat = SFR_stat_uncertainty.subs(
        {nu: freqs, al: alphas, f: flux, z: redshift,
         f_unc: flux_error, Ho: Hubble})
    SF_syst = SFR_syst_uncertainty.subs(
        {nu: freqs, al: alphas, f: flux, z: redshift, zunc: redshift_error,
         alunc: alphasig, f_unc: flux_error, Ho: Hubble, Ho_unc: Hubble_unc})
    # luminosity values
    Lum = Lumform.subs({f: flux, z: redshift, al: alphas, Ho: Hubble})
    Lum_stat = Lum_stat_unc.subs(
        {f: flux, z: redshift, f_unc: flux_error, Ho: Hubble, al: alphas})
    Lum_syst = Lum_syst_unc.subs(
        {f: flux, z: redshift, f_unc: flux_error, zunc: redshift_error,
         Ho: Hubble, Ho_unc: Hubble_unc})

    output.append(Lum)
    output.append(Lum_stat)
    output.append(SF)
    output.append(SF_stat)

    return output


# def compute_SFR_general(flux, frequency, z, alpha, alpha_NT=-0.85, flux_error=None,
#                    calibration_kind='Murphy12', return_with_error=False):
#     '''
#         To do:
#             [ ] - Implement error estimates
#             [ ] - Check the thermal contribution
#             [ ] - Use spectral index from the image
#             [ ] - Explore different Te's (electron temperature)
#     '''

#     if calibration_kind == 'Murphy11':
#         Lnu_NT, Lnu_NT_error = compute_Lnu(flux, z,
#                                            alpha)  # 0.0014270422727500343
#         SFR = 6.64 * (1e-29) * ((frequency) ** (-alpha_NT)) * Lnu_NT
#         if flux_error is not None:
#             Lnu_NT_error, Lnu_NT_error2 = compute_Lnu(flux_error, z, alpha)
#             SFR_error = 6.64 * (1e-29) * ((frequency) ** (-alpha_NT)) * Lnu_NT_error
#         else:
#             SFR_error = 0.0

#     if calibration_kind == 'Tabatabaei2017':
#         '''
#         There is something wrong for this kind!
#         '''
#         Lnu_NT, Lnu_NT_error = compute_Lnu(flux, z,
#                                            alpha)  # 0.0014270422727500343
#         SFR = 1.11 * 1e-37 * 1e9 * frequency * Lnu_NT
#         if flux_error is not None:
#             Lnu_NT_error, Lnu_NT_error2 = compute_Lnu(flux_error, z, alpha)
#             SFR_error = 1.11 * 1e-37 * 1e9 * frequency * Lnu_NT_error
#         else:
#             SFR_error = 0.0

#     if calibration_kind == 'Murphy12':
#         Te = 1e4
#         Lnu_NT, Lnu_NT_error = compute_Lnu(flux, z, alpha)
#         SFR = 1e-27 * ( \
#                     (2.18 * ((Te / (1e4)) ** 0.45) * (frequency ** (-0.1)) + \
#                      15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
#             ) * Lnu_NT

#         if flux_error is not None:
#             Lnu_NT_error, Lnu_NT_error2 = compute_Lnu(flux_error, z, alpha)
#             SFR_error = 1e-27 * ( \
#                         (2.18 * ((Te / (1e4)) ** 0.45) * (frequency ** (-0.1)) + \
#                          15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
#                 ) * Lnu_NT_error
#         else:
#             SFR_error = 0.0
#     # print('SFR =', SFR, '+/-', SFR_error, 'Mo/yr')
#     if return_with_error:
#         return (SFR, SFR_error)
#     else:
#         return (SFR)

def compute_SFR_general(flux, frequency, z, alpha, alpha_NT=-0.85, 
                        nu0 = 1.0,
                        flux_error=None,
                        calibration_kind='Murphy12'):
    '''
        To do:
            [ ] - Implement error estimates
            [ ] - Check the thermal contribution
            [ ] - Use spectral index from the image
            [ ] - Explore different Te's (electron temperature)
    '''

    Lnu,Lnu_error,SFR,SFR_error = None, None, None, None
    if calibration_kind == 'Murphy11':
        Lnu_NT = compute_Lnu(flux, z, alpha)  # 0.0014270422727500343
        SFR = 6.64 * (1e-29) * ((frequency) ** (-alpha_NT)) * Lnu_NT
        if flux_error is not None:
            Lnu_NT_error = compute_Lnu(flux_error, z, alpha)
            SFR_error = 6.64 * (1e-29) * ((frequency) ** (-alpha_NT)) * Lnu_NT_error
        else:
            SFR_error = 0.0

    # if calibration_kind == 'Tabatabaei2017':
    #     '''
    #     There is something wrong for this kind!
    #     '''
    #     Lnu_NT = compute_Lnu(flux, z, alpha)  # 0.0014270422727500343
    #     SFR = 1.11 * 1e-37 * 1e9 * frequency * Lnu_NT
    #     if flux_error is not None:
    #         Lnu_NT_error = compute_Lnu(flux_error, z, alpha)
    #         SFR_error = 1.11 * 1e-37 * 1e9 * frequency * Lnu_NT_error
    #     else:
    #         SFR_error = 0.0

    # if calibration_kind == 'Murphy12':
    #     Te = 1e4
    #     Lnu_NT = compute_Lnu(flux, z, alpha_NT)
    #     SFR = 1e-27 * ( \
    #                 (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
    #                  15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
    #         ) * Lnu_NT

    #     if flux_error is not None:
    #         Lnu_NT_error = compute_Lnu(flux_error, z, alpha_NT)
    #         SFR_error = 1e-27 * ( \
    #                     (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
    #                      15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
    #             ) * Lnu_NT_error
    #     else:
    #         SFR_error = 0.0
    
    if calibration_kind == 'Murphy12':
        Te = 1e4
        Lnu = compute_Lnu(flux, z, alpha_NT)
        SFR = 1*1e-27 * ( \
                    (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
                     15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
            ) * Lnu

        if flux_error is not None:
            Lnu_error = compute_Lnu(flux_error, z, alpha_NT)
            SFR_error = 1*1e-27 * ( \
                        (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
                         15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
                ) * Lnu_error
        else:
            SFR_error = 0.0

    if calibration_kind == 'Tabatabaei2017':
        Te = 1e4
        Lnu = compute_Lnu(flux, z, alpha_NT)
        SFR = 1*1e-27 * ( \
                    (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
                     15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
            ) * Lnu

        if flux_error is not None:
            Lnu_error = compute_Lnu(flux_error, z, alpha_NT)
            SFR_error = 1*1e-27 * ( \
                        (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
                         15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
                ) * Lnu_error
        else:
            SFR_error = 0.0
    
    if calibration_kind == 'Murphy12_NT':
        Lnu = compute_Lnu(flux, z, alpha_NT)
        SFR = 6.64*(1e-29) * ((frequency/nu0)**(-alpha_NT)) * Lnu
        
        if flux_error is not None:
            Lnu_error = compute_Lnu(flux_error, z, alpha_NT)
            SFR_error = 6.64*1e-29 * ((frequency/nu0)**(-alpha_NT)) * Lnu_error
        else:
            SFR_error = 0.0
            
    if calibration_kind == 'Murphy12_T':
        Te = 1e4
        Lnu = compute_Lnu(flux, z, alpha)
        SFR = 4.59*1e-28 * ((frequency/nu0)**(-alpha)) * ((Te / (1e4)) ** (-0.45)) * Lnu
        
        if flux_error is not None:
            Lnu_error = compute_Lnu(flux_error, z, alpha)
            SFR_error = 4.59*1e-28 * ((frequency/nu0)**(-alpha)) * ((Te / (1e4)) ** (-0.45)) * Lnu_error
        else:
            SFR_error = 0.0
    # print(f'nu =', frequency)
    # print(f'alpha =', alpha)
    # print(f'SFR =', SFR, '+/-', SFR_error, 'Mo/yr')
    print(f'Lnu =', Lnu, '+/-', Lnu_error, 'erg/s/Hz')
    # print('SFR =', SFR, '+/-', SFR_error, 'Mo/yr')
    return (SFR, SFR_error)



def compute_SFR_general_v2(flux, frequency, z, alpha, alpha_NT=-0.85, 
                        nu0 = 1.0,
                        flux_error=None,
                        calibration_kind='Murphy12'):
    '''
        To do:
            [ ] - Implement error estimates
            [ ] - Check the thermal contribution
            [ ] - Use spectral index from the image
            [ ] - Explore different Te's (electron temperature)
    '''
    Lnu,Lnu_error,SFR,SFR_error = None, None, None, None
    
    if calibration_kind == 'Murphy11':
        Lnu_NT = compute_Lnu(flux, z, alpha)  # 0.0014270422727500343
        SFR = 6.64 * (1e-29) * ((frequency) ** (-alpha_NT)) * Lnu_NT
        if flux_error is not None:
            Lnu_NT_error = compute_Lnu(flux_error, z, alpha)
            SFR_error = 6.64 * (1e-29) * ((frequency) ** (-alpha_NT)) * Lnu_NT_error
        else:
            SFR_error = 0.0

    # if calibration_kind == 'Tabatabaei2017':
    #     '''
    #     There is something wrong for this kind!
    #     '''
    #     Lnu_NT = compute_Lnu(flux, z, alpha)  # 0.0014270422727500343
    #     SFR = 1.11 * 1e-37 * 1e9 * frequency * Lnu_NT
    #     if flux_error is not None:
    #         Lnu_NT_error = compute_Lnu(flux_error, z, alpha)
    #         SFR_error = 1.11 * 1e-37 * 1e9 * frequency * Lnu_NT_error
    #     else:
    #         SFR_error = 0.0

    # if calibration_kind == 'Murphy12':
    #     Te = 1e4
    #     Lnu_NT = compute_Lnu(flux, z, alpha_NT)
    #     SFR = 1e-27 * ( \
    #                 (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
    #                  15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
    #         ) * Lnu_NT

    #     if flux_error is not None:
    #         Lnu_NT_error = compute_Lnu(flux_error, z, alpha_NT)
    #         SFR_error = 1e-27 * ( \
    #                     (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
    #                      15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
    #             ) * Lnu_NT_error
    #     else:
    #         SFR_error = 0.0
    
    if calibration_kind == 'Murphy12':
        Te = 1e4
        # Lnu_tot = compute_Lnu_v2(flux, z, alpha_NT,frequency,nu0)
        Lnu = compute_Lnu(flux, z, alpha_NT)
        SFR = 1*1e-27 * ( \
                    (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
                     15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
            ) * Lnu

        if flux_error is not None:
            # Lnu_tot_error = compute_Lnu_v2(flux_error, z, alpha_NT,frequency,nu0)
            Lnu_error = compute_Lnu(flux, z, alpha_NT)
            SFR_error = 1*1e-27 * ( \
                        (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
                         15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
                ) * Lnu_error
        else:
            SFR_error = 0.0

    if calibration_kind == 'Tabatabaei2017':
        Te = 1e4
        # Lnu_tot = compute_Lnu_v2(flux, z, alpha_NT,frequency,nu0)
        Lnu = compute_Lnu(flux, z, alpha_NT)
        SFR = 1*1e-27 * ( \
                    (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
                     15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
            ) * Lnu

        if flux_error is not None:
            # Lnu_tot_error = compute_Lnu_v2(flux_error, z, alpha_NT,frequency,nu0)
            Lnu_error = compute_Lnu(flux, z, alpha_NT)
            SFR_error = 1*1e-27 * ( \
                        (2.18 * ((Te / (1e4)) ** (0.45)) * (frequency ** (-0.1)) + \
                         15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
                ) * Lnu_error
        else:
            SFR_error = 0.0
    
    if calibration_kind == 'Murphy12_NT':
        Lnu = compute_Lnu_v2(flux, z, alpha_NT,frequency,nu0)
        SFR = 6.64*(1e-29) * ((frequency)**(-alpha_NT)) * Lnu
        
        if flux_error is not None:
            Lnu_error = compute_Lnu_v2(flux_error, z, alpha_NT,frequency,nu0)
            SFR_error = 6.64*1e-29 * ((frequency)**(-alpha_NT)) * Lnu_error
        else:
            SFR_error = 0.0
            
    if calibration_kind == 'Murphy12_T':
        Te = 1e4
        Lnu = compute_Lnu_v2(flux, z, alpha,frequency,nu0)
        SFR = 4.59*1e-28 * ((frequency)**(-alpha)) * ((Te / (1e4)) ** (-0.45)) * Lnu
        
        if flux_error is not None:
            Lnu_error = compute_Lnu_v2(flux_error, z, alpha,frequency,nu0)
            SFR_error = 4.59*1e-28 * ((frequency)**(-alpha)) * ((Te / (1e4)) ** (-0.45)) * Lnu_error
        else:
            SFR_error = 0.0
    print(f'nu =', frequency)
    print(f'alpha =', alpha)
    print(f'SFR =', SFR, '+/-', SFR_error, 'Mo/yr')
    print(f'Lnu =', Lnu, '+/-', Lnu_error, 'erg/s/Hz')
    return (SFR, SFR_error)



"""
 ____  _           _                       _              
|  _ \| |__   ___ | |_ ___  _ __ ___   ___| |_ _ __ _   _ 
| |_) | '_ \ / _ \| __/ _ \| '_ ` _ \ / _ \ __| '__| | | |
|  __/| | | | (_) | || (_) | | | | | |  __/ |_| |  | |_| |
|_|   |_| |_|\___/ \__\___/|_| |_| |_|\___|\__|_|   \__, |
                                                    |___/
                                                    
 ____      _                 _
|  _ \ ___| |_ _ __ ___  ___(_) __ _ _ __
| |_) / _ \ __| '__/ _ \/ __| |/ _` | '_ \
|  __/  __/ |_| | | (_) \__ \ | (_| | | | |
|_|   \___|\__|_|  \___/|___/_|\__,_|_| |_|
#Petrosian
"""


def do_petrofit(image, cell_size, mask_component=None, fwhm=8, kernel_size=5, npixels=32,
                main_feature_index=0, sigma_mask=7, dilation_size=10,deblend=False,
                apply_mask=True, PLOT=True, show_figure = True, results=None):
    # from petrofit.photometry import order_cat
    # from petrofit.photometry import make_radius_list

    # from petrofit source_photometry
    # from petrofit import make_catalog, plot_segments
    # from petrofit import plot_segment_residual
    # from petrofit import order_cat

    if results is None:
        results = {}
        results['#imagename'] = os.path.basename(image)

    # sigma = fwhm * gaussian_fwhm_to_sigma
    # kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size)
    data_2D_ = load_fits_data(image)
    #mad std must be computed on the original data, not on the masked array.
    std = np.std(data_2D_)

    if mask_component is not None:
        data_2D_ = data_2D_ * mask_component
    if apply_mask == True:
        omask, mask = mask_dilation(image, cell_size=cell_size,
                                    sigma=sigma_mask, dilation_size=dilation_size,
                                    PLOT=False)

        # data_2D = data_2D*(data_2D>=3*mad_std(data_2D))
        data_2D = data_2D_ * mask
        # std = np.std(data_2D)
    else:
        data_2D = data_2D_
        mask = np.ones(data_2D.shape)
        # std = mad_std(data_2D)

    plt.figure()

    cat, segm, segm_deblend = make_catalog(
        data_2D,
        threshold=1.0 * std,
        deblend=deblend,
        # kernel_size=kernel_size,
        # fwhm=fwhm,
        npixels=npixels,
        plot=PLOT, vmax=data_2D.max(), vmin=3 * std
    )

    # Display source properties
    print("Num of Targets:", len(cat))

    # Convert to table
    cat_table = cat.to_table()

    vmax = data_2D.max()
    vmin = 3 * std

    if PLOT == True:
        plt.figure()
        plot_segments(segm, image=data_2D, vmax=vmax, vmin=vmin)
        try:
            plt.figure()
            plot_segment_residual(segm, data_2D, vmax=vmax * 0.01)
            plt.figure()
            plot_segments(segm_deblend, image=data_2D, vmax=vmax, vmin=vmin)
        except:
            pass

    # Sort and get the largest object in the catalog
    sorted_idx_list = order_cat(cat, key='area', reverse=True)
    idx = sorted_idx_list[main_feature_index]  # index 0 is largest
    source = cat[idx]  # get source from the catalog

    #     try:
    #         r,ir = get_profile(image)
    #     except:
    # radii_last = 3*estimate_circular_aperture(crop_image,cellsize=0.008,std=3)
    # radii_last
    r,ir = get_profile(image)
    # radii_last = r[-1]/cell_size
    # radii_last = r[-1]
    # radii_last = 2*results['outer_perimeter']/(2.0*np.pi)
    if mask_component is not None:
        radii_last = int(3 * np.sqrt((np.sum(mask_component) / np.pi)))
    else:
        radii_last = np.sqrt(data_2D.shape[0]**2.0 + data_2D.shape[1]**2.0)/2
        # radii_last = 2 * estimate_area_nbeam(image, mask, cellsize=cell_size)
    print('Creating radii list with max value of =', radii_last)
    #     r_list = np.arange(4, radii_last, 4)
    #     r_list = make_radius_list(
    #         max_pix=radii_last, # Max pixel to go up to
    #         n=int(len(r)) # the number of radii to produce
    #     )

    r_list = make_radius_list(
        max_pix=radii_last,  # Max pixel to go up to
        n=int(radii_last)  # the number of radii to produce
    )

    #     max_pix = 3*estimate_area_nbeam(image,mask,cellsize=cell_size)
    #     # cell_size = 0.008
    #     true_resolution = 0.05
    #     n_points = max_pix * (cell_size/true_resolution)

    #     r_list = make_radius_list(
    #         max_pix=max_pix, # Max pixel to go up to
    #         n=int(n_points) # the number of radii to produce
    #     )
    # print(repr(r_list))

    flux_arr, area_arr, error_arr = source_photometry(
        # Inputs
        source,  # Source (`photutils.segmentation.catalog.SourceCatalog`)
        data_2D,  # Image as 2D array
        #     segm_deblend, # Deblended segmentation map of image
        segm,  # Deblended segmentation map of image
        r_list,  # list of aperture radii
        # Options
        cutout_size=2 * max(r_list),  # Cutout out size, set to double the max radius
        bg_sub=False,  # Subtract background
        sigma=1, sigma_type='clip',  # Fit a 2D plane to pixels within 3 sigma of the mean
        plot=PLOT, vmax=0.3 * data_2D.max(), vmin=3 * std,  # Show plot with max and min defined above
    )

    beam_A = beam_area2(image, cellsize=cell_size)
    S_flux = flux_arr / beam_A
    area_beam = area_arr / beam_A

    from petrofit.petrosian import Petrosian
    p = Petrosian(r_list, area_arr, flux_arr)

    from copy import copy

    p_copy = copy(p)
    p_copy.eta = 0.2
    p_copy.epsilon = 2

    # print('eta =', p_copy.eta)
    # print('epsilon =', p_copy.epsilon)
    # print('r_half_light (old vs new) = {:0.2f} vs {:0.2f}'.format(p.r_half_light, p_copy.r_half_light))
    # print('r_total_flux (old vs new) = {:0.2f} vs {:0.2f}'.format(p.r_total_flux, p_copy.r_total_flux))



    # print('R50 = ', p_copy.r_half_light)
    # print('Rp=', p_copy.r_petrosian)

    results['R50'] = p_copy.r_half_light
    results['Rp'] = p_copy.r_petrosian
    results['R50_2'] = p.r_half_light
    results['Rp_2'] = p.r_petrosian
    results['flux_rp'] = S_flux.max()
    results['r_total_flux_2'] = p.r_total_flux
    results['total_flux_rp_2'] = p.total_flux/beam_A

    results['R20p'] = p_copy.fraction_flux_to_r(fraction=0.2)
    results['R50p'] = p_copy.r_half_light
    results['R80p'] = p_copy.fraction_flux_to_r(fraction=0.8)
    results['R90p'] = p_copy.fraction_flux_to_r(fraction=0.9)
    results['R20p_2'] = p.fraction_flux_to_r(fraction=0.2)
    results['R50p_2'] = p.r_half_light
    results['R80p_2'] = p.fraction_flux_to_r(fraction=0.8)
    results['R90p_2'] = p.fraction_flux_to_r(fraction=0.9)

    C1p = np.log10(results['R80p'] / results['R20p'])
    C2p = np.log10(results['R90p'] / results['R50p'])

    results['C1p'] = C1p
    results['C2p'] = C2p

    results['r_total_flux'] = p_copy.r_total_flux
    results['total_flux_rp'] = p_copy.total_flux/beam_A
    results['r_total_flux_2'] = p.r_total_flux
    results['total_flux_rp_2'] = p.total_flux/beam_A

    if PLOT == True:
        plt.figure()
        #         plt.plot(r_list,S_flux/S_flux[-1])
        #         plt.xlim(0,2*p.r_petrosian)
        #         plt.semilogx()
        p_copy.plot(plot_r=True)
        plt.savefig(image.replace('.fits', '_Rpetro_flux.jpg'), dpi=300, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()

    return (r_list, area_arr, area_beam, p_copy, S_flux, results)


def petrosian_metrics(source, data_2D, segm, mask_source,global_mask=None,
                 i='1', petro_properties={},sigma_type='clip',eta_value=0.2,
                 rlast=None, sigma=3, vmin=3, bkg_sub=False,error=None,
                 plot=False):
    if rlast is None:
        if mask_source is not None:
            _, area_convex_mask = convex_shape(mask_source)
            # rlast = int(2 * np.sqrt((np.sum(mask_source) / np.pi)))
            rlast = int(1.5*area_to_radii(area_convex_mask))
        else:
            if global_mask is not None:
                _, area_convex_mask = convex_shape(global_mask)
                # rlast = int(2 * np.sqrt((np.sum(global_mask) / np.pi)))
                rlast = int(1.5*area_to_radii(area_convex_mask))
            else:
                rlast = np.sqrt(data_2D.shape[0]**2.0 + data_2D.shape[1]**2.0)/2
    else:
        rlast = rlast

    r_list = make_radius_list(max_pix=rlast,  # Max pixel to go up to
                              n=int(rlast)  # the number of radii to produce
                             )
    cutout_size = 2 * max(r_list)

    # if (bkg_to_sub is not None) and (bkg_sub == False):
    #     data_2D = data_2D - bkg_to_sub

    flux_arr, area_arr, error_arr = source_photometry(source=source,
                                                      image=data_2D,
                                                      segm_deblend=segm,
                                                      r_list=r_list,
                                                      error=error,
                                                      cutout_size=cutout_size,
                                                      bg_sub=bkg_sub, sigma=sigma,
                                                      sigma_type=sigma_type,
                                                      plot=plot, vmax=0.3 * data_2D.max(),
                                                      vmin=vmin * mad_std(data_2D)
                                                      )
    #     fast_plot2(mask_source * data_2D)
    p = Petrosian(r_list, area_arr, flux_arr)
    from copy import copy
    p_015 = copy(p)
    if eta_value is None:
        p_015.eta = 0.2
    else:
        p_015.eta = eta_value

    p_015.epsilon = 2
    R50 = p_015.r_half_light
    Snu = p_015.total_flux
    Rp = p_015.r_petrosian
    try:
        Rpidx = int(2 * Rp)
    except:
        Rpidx = int(r_list[-1])
    petro_properties['R50'] = R50
    petro_properties['Snu'] = Snu
    petro_properties['Rp'] = Rp
    petro_properties['Rpidx'] = Rpidx
    petro_properties['rlast'] = rlast
    return (petro_properties, flux_arr, area_arr, error_arr, p_015, r_list)

def compute_petrosian_properties(data_2D, imagename, mask_component=None,
                                 global_mask=None,
                                 i=0, source_props=None,apply_mask = False,
                                 sigma_level=3, bkg_sub=False,error=None,
                                 vmin=1.0, plot=False, deblend=False,
                                 show_figure = True,plot_catalog=False,
                                 segm_reg= 'mask',vmax=0.1,bkg_to_sub=None,
                                 fwhm=121, kernel_size=81, npixels=128,
                                 verbose=0,
                                 add_save_name='',logger=None):
    # if mask:
    if source_props is None:
        source_props = {}
        source_props['#imagename'] = os.path.basename(imagename)
    if imagename is not None:
        beam_area_ = beam_area2(imagename, cellsize=None)
    if npixels is None:
        npixels = int(beam_area_)

    ii = str(i + 1)
    std = mad_std(data_2D)
    data_component = data_2D.copy()
    # if apply_mask == True:
    #     omask, mask = mask_dilation(image, cell_size=cell_size,
    #                                 sigma=sigma_mask, dilation_size=dilation_size,
    #                                 PLOT=False)
    if (bkg_to_sub is not None) and (bkg_sub == False):
        data_component = data_component  - bkg_to_sub
    if global_mask is not None:
        '''
        Global mask:
            mask that describe the full extension of the source.
        '''
        data_component = data_component * global_mask
        # sigma_level = 1.0
    if mask_component is not None:
        """
        mask_component: 2D bolean array
            If source has multiple components, this mask is the mask of
            only one component, or more, like
            mask_component = mask_component1 + mask_component2.
        """
        # sigma_level = 1.0
        data_component = data_component * mask_component


    # data_component = data_2D
    # eimshow(data_component)

    try:
        cat, segm, segm_deblend = make_catalog(image=data_component,
                                               threshold=3 * std,
                                               deblend=deblend,
                                               # kernel_size=kernel_size,
                                               # fwhm=fwhm,
                                               npixels=npixels,
                                               # because we already deblended it!
                                               plot=plot_catalog,
                                               vmax=vmax*data_component.max(),
                                               vmin=vmin * std)
    except:
        try:
            cat, segm, segm_deblend = make_catalog(image=data_component,
                                                   threshold=0.5 * std,
                                                   deblend=deblend,
                                                   # kernel_size=kernel_size,
                                                   # fwhm=fwhm,
                                                   npixels=npixels,
                                                   # because we already deblended it!
                                                   plot=plot_catalog,
                                                   vmax=vmax*data_component.max(),
                                                   vmin=vmin * std)
        except:
            cat, segm, segm_deblend = make_catalog(image=data_component,
                                                   threshold=0.01 * std,
                                                   deblend=deblend,
                                                   # kernel_size=kernel_size,
                                                   # fwhm=fwhm,
                                                   npixels=npixels,
                                                   # because we already deblended it!
                                                   plot=plot_catalog,
                                                   vmax=vmax * data_component.max(),
                                                   vmin=vmin * std)

    sorted_idx_list = order_cat(cat, key='area', reverse=True)
    idx = sorted_idx_list[0]  # index 0 is largest
    source = cat[idx]



    # if plot == True:
    #     # plt.figure()
    #     plot_segments(segm, image=data_component, vmax=vmax, vmin=vmin)
    #     try:
    #         # plt.figure()
    #         plot_segment_residual(segm, image=data_component, vmax=vmax * 0.01)
    #         # plt.figure()
    #         # plot_segments(data_component, image=data_2D, vmax=vmax, vmin=vmin)
    #     except:
    #         pass

    # source = cat[0]

    source_props['PA'] = source.orientation.value
    source_props['q'] = 1 - source.ellipticity.value
    source_props['area'] = source.area.value
    source_props['Re'] = source.equivalent_radius.value
    source_props['x0c'] = source.xcentroid
    source_props['y0c'] = source.ycentroid

    if segm_reg == 'deblended':
        segm_mask = segm_deblend
    if segm_reg == 'mask':
        segm_mask = segm




    # help function to be used if iteration required.
    source_props, flux_arr, area_arr, error_arr, p, r_list = \
        petrosian_metrics(source=source,
                          data_2D=data_component,
                          segm=segm_mask,global_mask=global_mask,
                          mask_source=mask_component,i=ii,
                          petro_properties=source_props,
                          rlast=None, sigma=sigma_level,
                          vmin=vmin, bkg_sub=bkg_sub,error=error,
                          plot=plot)

    """
    Check if Rp is larger than last element (rlast) of the R_list. If yes,
    we need to run petro_params again, with a larger rlast, at least r_last>=Rp.
    If not, R50 will be np.nan as well Snu.
    """

    if (source_props['rlast'] < 2 * source_props['Rp']) or \
            (np.isnan(p.r_total_flux)):
        if verbose>0:
            print('WARNING: Number of pixels for petro region is to small. '
                  'Looping over until good condition is satisfied.')
            print(f"Rlast        = {source_props['rlast']}")
            print(f"2*Rp           =  {2*source_props['Rp']}")
            print(f"r_total_flux = {p.r_total_flux}")


        if (mask_component is not None):
            _, area_convex_mask = convex_shape(mask_component)
            # rlast = int(2 * np.sqrt((np.sum(mask_source) / np.pi)))
            Rlast_new = int(3.0 * area_to_radii(area_convex_mask))
        else:
            if global_mask is not None:
                _, area_convex_mask = convex_shape(global_mask)
                # rlast = int(2 * np.sqrt((np.sum(mask_source) / np.pi)))
                Rlast_new = int(3.0 * area_to_radii(area_convex_mask))
            else:
                Rlast_new = 2 * source_props['Rp'] + (2 * source_props['Rp']-source_props['rlast']+1)
        # print(f'Rlast (new) = {Rlast_new}')
        source_props, flux_arr, area_arr, error_arr, p, r_list = \
            petrosian_metrics(source=source,data_2D=data_component,
                              segm=segm_mask,global_mask=global_mask,
                              mask_source=mask_component,eta_value=0.20,
                              i=ii,petro_properties=source_props,
                              rlast=Rlast_new,sigma=sigma_level,
                              vmin=vmin,bkg_sub=bkg_sub,error=error,
                              plot=plot)

    if (source_props['rlast'] < 2 * source_props['Rp']) or \
            (np.isnan(p.r_total_flux)):
        if verbose>0:
            print('WARNING: Number of pixels for petro region is to small. '
                  'Looping over until good condition is satisfied.')
            print(f"Rlast        = {source_props['rlast']}")
            print(f"2*Rp         =  {2*source_props['Rp']}")
            print(f"r_total_flux = {p.r_total_flux}")
        Rlast_new = 2 * source_props['Rp'] + (2 * source_props['Rp']-source_props['rlast']+1)
        # print(f'Rlast (new) = {Rlast_new}')
        source_props, flux_arr, area_arr, error_arr, p, r_list = \
            petrosian_metrics(source=source,data_2D=data_component,
                              segm=segm_mask,global_mask=global_mask,
                              mask_source=mask_component,eta_value=0.25,
                              i=ii,petro_properties=source_props,
                              rlast=Rlast_new,sigma=sigma_level,
                              vmin=vmin,bkg_sub=bkg_sub,error=error,
                              plot=plot)


    beam_A = beam_area2(imagename)
    S_flux = flux_arr / beam_A
    area_beam = area_arr / beam_A

    # from copy import copy
    # p_copy = copy(p)
    # p_copy.eta = 0.2
    # p_copy.epsilon = 2

    # print('eta =', p.eta)
    # print('epsilon =', p.epsilon)
    # print('r_half_light (old vs new) = {:0.2f}'.
    #       format(p.r_half_light))
    # print('r_total_flux (old vs new) = {:0.2f}'.
    #       format(p.r_total_flux))


    """
    Now, estimate the effective intensity.
    """
    r, ir = get_profile(data_component, binsize=1.0)
    try:
        I50 = ir[int(source_props['R50'])]
    except:
        source_props['R50'] = source_props['Re'] / 2
        I50 = ir[int(source_props['R50'])]
        # I50 = ir[0]*0.1
    source_props['I50'] = I50

    # print('R50 = ', p.r_half_light)
    # print('Rp=', p.r_petrosian)

    source_props['R50'] = p.r_half_light
    source_props['Rp'] = p.r_petrosian
    # source_props['R50_2'] = p.r_half_light
    # source_props['Rp_2'] = p.r_petrosian
    source_props['flux_rp'] = S_flux.max()
    source_props['r_total_flux'] = p.r_total_flux
    source_props['total_flux_rp'] = p.total_flux/beam_A

    # source_props['R20p'] = p_copy.fraction_flux_to_r(fraction=0.2)
    # source_props['R50p'] = p_copy.r_half_light
    # source_props['R80p'] = p_copy.fraction_flux_to_r(fraction=0.8)
    # source_props['R90p'] = p_copy.fraction_flux_to_r(fraction=0.9)
    source_props['R20p'] = p.fraction_flux_to_r(fraction=0.2)
    source_props['R50p'] = p.r_half_light
    source_props['R80p'] = p.fraction_flux_to_r(fraction=0.8)
    source_props['R90p'] = p.fraction_flux_to_r(fraction=0.9)

    C1p = np.log10(source_props['R80p'] / source_props['R20p'])
    C2p = np.log10(source_props['R90p'] / source_props['R50p'])

    source_props['C1p'] = C1p
    source_props['C2p'] = C2p

    # source_props['r_total_flux'] = p.r_total_flux
    # source_props['total_flux_rp'] = p.total_flux/beam_A
    # source_props['r_total_flux_2'] = p.r_total_flux
    # source_props['total_flux_rp_2'] = p.total_flux/beam_A

    if plot == True:
        plot_flux_petro(imagename, flux_arr, r_list, add_save_name)
        plt.figure()
        #         plt.plot(r_list,S_flux/S_flux[-1])
        #         plt.xlim(0,2*p.r_petrosian)
        #         plt.semilogx()
        p.plot(plot_r=True)
        plt.savefig(imagename.replace('.fits', '_Rpetro_flux'+
                                      add_save_name+'.jpg'),
                    dpi=300, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()

        plt.figure()
        p.plot_cog()
        plt.savefig(imagename.replace('.fits', '_cog_flux'+
                                      add_save_name+'.jpg'),
                    dpi=300, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()

    # if (global_mask is not None) and (imagename is not None):
    #     data_comp_mask = data_component * global_mask
    #     total_comp_flux = np.sum(data_comp_mask) / beam_area_
    #     source_props['c' + ii + '_total_flux'] = total_comp_flux
    return(r_list, area_arr, area_beam, p, flux_arr, error_arr, source_props,
           cat, sorted_idx_list, segm, segm_deblend)

def compute_petro_source(data_2D, mask_component=None, global_mask=None,
                         obs_type = 'radio',
                         npixels=None, nlevels=1, contrast=1,
                         imagename=None, i=0, source_props={},positions=None,
                         sigma_level=3, bkg_sub=False,
                         vmin=1, plot=False, deblend=False):
    """
    Perform petrosian photometry of a source. It can be used for the full structure of the source
    or for individual regions (incl. deblended).

    Parameters:
    -----------
    data_2D: 2D numpy array
        Image data.
    mask_component: 2D numpy array
        Mask of a given region (e.g. used when analysing multiple distinc regions of a source.
    global_mask: 2D numpy array
        Mask of the full source.
    imagename: str
        Name of the image. Used when one wants to point the original .fits header (wcs) of the
        image where the data_2D comes from.
    i: int
        Index of the component (when analysing multiple regions).
    source_props: dict
        Dictionary to store the properties of the source.
    positions: list
        List of positions of the sources.
    sigma_level: float
        Level of the sigma to be used for the thresholding.
    bkg_sub: bool
        If True, subtract the background.
    vmin: float
        Minimum value of the image.
    plot: bool
        If True, plot the results.
    deblend: bool
        If True, deblend the sources.
    """
    verbose = 0
    # if mask:
    if imagename is not None:
        try:
            beam_area_ = beam_area2(imagename, cellsize=None)
        except:
            beam_area_ = 1.0

    if isinstance(i, str) == True:
        # print(' ++==>> Computing global source properties.')
        ii = i
    else:
        ii = str(i + 1)
    
    # if isinstance(data_2D, str) == True:
    #     std = mad_std(load_fits_data(data_2D))
    # else:
    if mad_std(data_2D) == 0:
        std = mad_std(data_2D[(data_2D>0)])
    else:
        std = mad_std(data_2D)
    
    if mask_component is not None:
        data_component = data_2D * mask_component
        if npixels is None:
            npixels = int(np.nansum(mask_component)/50)
    else:
        mask_component = np.ones(data_2D.shape)
        data_component = data_2D
        if npixels is None:
            npixels = int(data_2D.shape[0]/50)
    
    # print(' ++==>> Making catalog...')
    cat, segm, segm_deblend = make_catalog(image=data_component,
                                           threshold=sigma_level * std,
                                           deblend=False,# because we already deblended it!
                                        #    npixels=npixels,
                                           npixels=npixels, nlevels=nlevels, contrast=contrast,
                                           plot=plot, vmax=np.nanmax(data_component),
                                           vmin=vmin * std)

    sorted_idx_list = order_cat(cat, key='area', reverse=True)
    # print(len(sorted_idx_list))
    source = cat[sorted_idx_list[0]]

    do_fit_ellipse = True
    if do_fit_ellipse:
        levels_ellipse = np.geomspace(np.nanmax(data_component), sigma_level * std, 32)
        try:
            sigma_50 = levels_ellipse[levels_ellipse < 0.2*np.nanmax(levels_ellipse)][0]
        except:
            sigma_50 = levels_ellipse[levels_ellipse < 0.8*np.nanmax(levels_ellipse)][0]
        region_split = [i for i, x in enumerate(levels_ellipse > sigma_50) if x][-1]
        
        PA, q, x0col, y0col, PAm, qm, \
            PAmi, qmi, PAmo, qmo, \
            x0median, y0median, \
            x0median_i, y0median_i, \
            x0median_o, y0median_o,profiles = cal_PA_q(data_component, 
                                                       Isequence=levels_ellipse,
                                                       region_split=region_split,
                                                       SAVENAME=f'{imagename}_{i}'.replace('.fits','_ellipsefit')
                                                    #    SAVENAME=img.replace('.fits','_ellipsefit') + ext
                                            )

    source_props['c' + ii + '_PA'] = source.orientation.value
    source_props['c' + ii + '_q'] = 1 - source.ellipticity.value
    source_props['c' + ii + '_area'] = source.area.value
    source_props['c' + ii + '_Re'] = source.equivalent_radius.value
    source_props['c' + ii + '_x0c'] = source.xcentroid
    source_props['c' + ii + '_y0c'] = source.ycentroid
    source_props['c' + ii + '_label'] = source.label

    # help function to be used if iteration required.
    # print(' ++==>> Computing petrosian parameters...')
    source_props, p = petro_params(source=source, data_2D=data_component, segm=segm,
                                mask_source=mask_component,positions=positions,
                                i=ii, petro_properties=source_props,
                                rlast=None, sigma=sigma_level,
                                vmin=vmin, bkg_sub=bkg_sub,
                                plot=plot)
    
    # print('Rlast', source_props['c' + ii + '_rlast'])
    # print('2Rp', 2 * source_props['c' + ii + '_Rp'])
    """
    Check if Rp is larger than last element (rlast) of the R_list. If yes,
    we need to run petro_params again, with a larger rlast, at least r_last>=Rp.
    If not, R50 will be np.nan.
    """
    if (source_props['c' + ii + '_rlast'] < 2 * source_props['c' + ii + '_Rp']) \
            or (np.isnan(p.r_total_flux)):
        if verbose>0:
            print('WARNING: Number of pixels for petro region is to small. '
                  'Looping over until good condition is satisfied.')
        # Rlast_new = 2 * source_props['c' + ii + '_Rp'] + 3

        _, area_convex_mask = convex_shape(mask_component)
        # rlast = int(2 * np.sqrt((np.sum(mask_source) / np.pi)))
        Rlast_new = int(3.0 * area_to_radii(area_convex_mask))
        # print('Rlast', source_props['c' + ii + '_rlast'])
        # print('2Rp', 2 * source_props['c' + ii + '_Rp'])
        # print('Rlast_new',Rlast_new)
        # print(' ++==>> Re-computing petrosian parameters...')
        source_props, p = petro_params(source=source, data_2D=data_component,
                                    segm=segm, mask_source=mask_component,
                                    i=ii, petro_properties=source_props,
                                    rlast=Rlast_new, sigma=sigma_level,
                                    vmin=vmin,
                                    bkg_sub=bkg_sub, plot=plot)

        if (source_props['c' + ii + '_rlast'] < 2 * source_props['c' + ii + '_Rp']) \
                or (np.isnan(p.r_total_flux)):
            if verbose>0:
                print('WARNING: Number of pixels for petro region is to small. '
                    'Looping over until good condition is satisfied.')
            # Rlast_new = 2 * source_props['Rp'] + 3
            # print(' ++==>> Re-computing petrosian parameters...')

            Rlast_new = 2 * source_props['c' + ii + '_Rp'] + 3
            # print('Rlast', source_props['c' + ii + '_rlast'])
            # print('2Rp', 2 * source_props['c' + ii + '_Rp'])
            # print('Rlast_new',Rlast_new)
            
            source_props, p = petro_params(source=source, data_2D=data_2D,
                                        segm=segm,mask_source=mask_component,
                                        i=ii, petro_properties=source_props,
                                        rlast=Rlast_new, sigma=sigma_level,
                                        vmin=vmin,eta_value=0.25,
                                        bkg_sub=bkg_sub, plot=plot)



    """
    Now, estimate the effective intensity.
    """
    r, ir = get_profile(data_component, binsize=1.0)
    # if obs_type == 'radio':
    #     I50 = np.nanmax(ir)
    # else:
    try:
        I50 = ir[int(source_props['c' + ii + '_R50'])]
    except:
        source_props['c' + ii + '_R50'] = source_props['c' + ii + '_Re'] / 2
        I50 = ir[int(source_props['c' + ii + '_R50'])]
        # I50 = ir[0]*0.1

    source_props['c' + ii + '_I50'] = I50

    if (global_mask is not None) and (imagename is not None):
        data_comp_mask = data_component * global_mask
        total_comp_flux = np.sum(data_comp_mask) / beam_area_
        source_props['c' + ii + '_total_flux'] = total_comp_flux
    return (source_props)

def petro_cat(data_2D, fwhm=24, npixels=128, kernel_size=15,
              nlevels=30, contrast=0.001,bkg_sub=False,
              sigma_level=20, vmin=5,
              deblend=True, plot=False):
    """
    Use PetroFit class to create catalogues.
    """
    cat, segm, segm_deblend = make_catalog(
        image=data_2D,
        threshold=sigma_level * mad_std(data_2D),
        # kernel_size=kernel_size,
        # fwhm=fwhm,
        nlevels=nlevels,
        deblend=deblend,
        npixels=npixels,contrast=contrast,
        plot=plot, vmax=data_2D.max(), vmin=vmin * mad_std(data_2D)
    )

    sorted_idx_list = order_cat(cat, key='area', reverse=True)
    #     idx = sorted_idx_list[main_feature_index]  # index 0 is largest
    #     source = cat[idx]  # get source from the catalog
    return (cat, segm, sorted_idx_list)


def petro_params(source, data_2D, segm, mask_source, positions=None,
                 i='1', petro_properties={},sigma_type='clip',eta_value=None,
                 rlast=None, sigma=3, vmin=3, bkg_sub=True, plot=False):
    if rlast is None:
        rlast = int(np.sqrt((np.sum(mask_source) / np.pi)))
        # _, area_convex_mask = convex_shape(mask_source)
        # Rlast_convex = int(1.0 * area_to_radii(area_convex_mask))        
        # print('Estimate for rlast =', rlast)
        # print('Estimate for rlast (convex) =', Rlast_convex)

    else:
        rlast = rlast
        

    r_list = make_radius_list(max_pix=rlast,  # Max pixel to go up to
                              n=int(rlast)  # the number of radii to produce
                             )
    cutout_size = 2 * max(r_list)
    flux_arr, area_arr, error_arr = source_photometry(source, data_2D, segm,
                                                      r_list, cutout_size=cutout_size,
                                                      # position2=positions,
                                                      bg_sub=bkg_sub, sigma=sigma,
                                                      sigma_type=sigma_type,
                                                      plot=plot, vmax=0.3 * data_2D.max(),
                                                      vmin=vmin * mad_std(data_2D)
                                                      )
    #     fast_plot2(mask_source * data_2D)
    p = Petrosian(r_list, area_arr, flux_arr)

    if eta_value is None:
        R50 = p.r_half_light
        R20 = p.fraction_flux_to_r(fraction=0.2)
        R80 = p.fraction_flux_to_r(fraction=0.8)
        C1p = np.log10(R80 / R20)
        Snu = p.total_flux
        Rp = p.r_petrosian
        p_return = p
        if plot == True:
            plt.figure()
            p.plot(plot_r=plot)
            plt.figure()
            p.plot_cog()
        #     print('    R50 =', R50)
        #     print('     Rp =', Rp)


    if eta_value is not None:
        from copy import copy
        p_new = copy(p)
        p_new.eta = eta_value
        R50 = p_new.r_half_light
        R20 = p_new.fraction_flux_to_r(fraction=0.2)
        R80 = p_new.fraction_flux_to_r(fraction=0.8)
        C1p = np.log10(R80 / R20)
        Snu = p_new.total_flux
        Rp = p_new.r_petrosian
        p_return = p_new
        if plot == True:
            plt.figure()
            p_new.plot(plot_r=plot)
            p_new.plot_cog()
        #     print('    R50 =', R50)
        #     print('     Rp =', Rp)

    try:
        Rpidx = int(2 * Rp)
    except:
        Rpidx = int(r_list[-1])
    petro_properties['c' + i + '_R50'] = R50
    petro_properties['c' + i + '_R20'] = R20
    petro_properties['c' + i + '_R80'] = R80
    petro_properties['c' + i + '_C1'] = C1p
    petro_properties['c' + i + '_Snu'] = Snu
    petro_properties['c' + i + '_Rp'] = Rp
    petro_properties['c' + i + '_Rpidx'] = Rpidx
    petro_properties['c' + i + '_rlast'] = rlast

    return (petro_properties, p_return)


def source_props(data_2D, source_props={},sigma_mask = 5,
                 fwhm=24, npixels=128, kernel_size=15, nlevels=30,
                 contrast=0.001,sigma_level=20, vmin=5,bkg_sub=False,
                 deblend=True,PLOT=False,apply_mask=False):
    '''
    From a 2D image array, perform simple source extraction, and calculate basic petrosian
    properties.
    '''
    verbose = 0
    if apply_mask:
        _, mask = mask_dilation(data_2D, PLOT=False,
                                sigma=sigma_mask, iterations=2, dilation_size=10)
        data_2D = data_2D*mask

    cat, segm, sorted_idx_list = petro_cat(data_2D, fwhm=fwhm, npixels=npixels,
                                           kernel_size=kernel_size,bkg_sub=bkg_sub,
                                           nlevels=nlevels, contrast=contrast,
                                           sigma_level=sigma_level, vmin=vmin,
                                           deblend=deblend, plot=PLOT)
    #     i = 0
    for i in range(len(sorted_idx_list)):
        ii = str(i + 1)
        seg_image = cat[sorted_idx_list[i]]._segment_img.data
        # seg_image = np.logical_not(cat[sorted_idx_list[i]].segment_ma.mask)
        source = cat[sorted_idx_list[i]]
        source_props['c' + ii + '_PA'] = source.orientation.value
        source_props['c' + ii + '_q'] = 1 - source.ellipticity.value
        source_props['c' + ii + '_area'] = source.area.value
        source_props['c' + ii + '_Re'] = source.equivalent_radius.value
        source_props['c' + ii + '_x0c'] = source.xcentroid
        source_props['c' + ii + '_y0c'] = source.ycentroid
        source_props['c' + ii + '_label'] = source.label

        label_source = source.label
        # plt.imshow(seg_image==label_source)
        mask_source = seg_image == label_source
        # mask_source = seg_image
        source_props, p = petro_params(source=source, data_2D=data_2D, segm=segm,
                                    mask_source=mask_source,
                                    i=ii, petro_properties=source_props,
                                    rlast=None, sigma=sigma_level,
                                    vmin=vmin, bkg_sub=bkg_sub,
                                    plot=PLOT)

        #         print(Rp_props['rlast'],2*Rp_props['Rp'])
        if ((source_props['c' + ii + '_rlast'] < 2 * source_props['c' + ii + '_Rp'])) \
                or (np.isnan(p.r_total_flux)):
            if verbose>0:
                print('WARNING: Number of pixels for petro region is to small. '
                    'Looping over until good condition is satisfied.')
            # Rlast_new = 2 * source_props['c' + ii + '_Rp'] + 3

            _, area_convex_mask = convex_shape(mask_source)
            # rlast = int(2 * np.sqrt((np.sum(mask_source) / np.pi)))
            Rlast_new = int(3.0 * area_to_radii(area_convex_mask))

            source_props, p = petro_params(source=source, data_2D=data_2D,
                                        segm=segm,mask_source=mask_source,
                                        i=ii, petro_properties=source_props,
                                        rlast=Rlast_new, sigma=sigma_level,
                                        vmin=vmin,
                                        bkg_sub=bkg_sub, plot=PLOT)

        if (source_props['c' + ii + '_rlast'] < 2 * source_props['c' + ii + '_Rp']) \
                or (np.isnan(p.r_total_flux)):
            if verbose>0:
                print('WARNING: Number of pixels for petro region is to small. '
                    'Looping over until good condition is satisfied.')
            # Rlast_new = 2 * source_props['Rp'] + 3
            Rlast_new = 2 * source_props['c' + ii + '_Rp'] + 3
            source_props, p = petro_params(source=source, data_2D=data_2D,
                                        segm=segm,mask_source=mask_source,
                                        i=ii, petro_properties=source_props,
                                        rlast=Rlast_new, sigma=sigma_level,
                                        vmin=vmin,eta_value=0.25,
                                        bkg_sub=bkg_sub, plot=PLOT)

        r, ir = get_profile(data_2D * mask_source, binsize=1.0)
        try:
            I50 = ir[int(source_props['c' + ii + '_R50'])]
        except:
            source_props['c' + ii + '_R50'] = source_props['c' + ii + '_Re']/2
            I50 = ir[int(source_props['c' + ii + '_R50'])]
            # I50 = ir[0]*0.1

        source_props['c' + ii + '_I50'] = I50

    source_props['ncomps'] = len(sorted_idx_list)
    return (source_props,cat, segm)


"""
 ____                           
/ ___|  ___  _   _ _ __ ___ ___ 
\___ \ / _ \| | | | '__/ __/ _ \
 ___) | (_) | |_| | | | (_|  __/
|____/ \___/ \__,_|_|  \___\___|

 _____      _                  _   _             
| ____|_  _| |_ _ __ __ _  ___| |_(_) ___  _ __  
|  _| \ \/ / __| '__/ _` |/ __| __| |/ _ \| '_ \ 
| |___ >  <| |_| | | (_| | (__| |_| | (_) | | | |
|_____/_/\_\\__|_|  \__,_|\___|\__|_|\___/|_| |_|


"""


def sep_background(imagename,mask=None,apply_mask=False,show_map=False,
                   bw=64, bh=64, fw=5, fh=5, use_beam_fraction=False,
                   b_factor=2,f_factor=2):
    """
    Use SEP to estimate the background of an image.

    Parameters
    ----------
    imagename : str
        Path to the image.
    mask : array
        Mask to be applied to the image.
    apply_mask : bool
        If True, calculate the dilated mask from the image.
    show_map : bool
        If True, show the background map.
    bw : int
        Box width for the background estimation.
    bh : int
        Box height for the background estimation.
    fw : int
        Filter width for the background estimation.
    fh : int
        Filter height for the background estimation.
    use_beam_fraction : bool
        If True, use the beam fraction sizes for the sizes of the boxes and
        filters (bw, bh, fw, fh).
    bfactor : int (optional)
        Factor to multiply the box sizes (bw, bh) by.
    factor : int (optional)
        Factor to multiply the filter sizes (fw, fh) by.

    Returns
    -------
    bkg : sep.Background
        Background object.
    """
    '''
    If using astropy.io.fits, you get an error (see bug on sep`s page).
    '''
    _data_2D = fitsio.read(imagename)
    if len(_data_2D.shape) == 4:
        data_2D = _data_2D[0][0]
    else:
        data_2D = _data_2D

    if use_beam_fraction:
        bspx = get_beam_size_px(imagename)
        bspx_x, bspx_y = int(bspx[1]), int(bspx[2])
        bspx_avg = int(bspx[0])
        print(f"Beam Size in px=({bspx_x},{bspx_y})")
        print(f"Average beam Size in px=({bspx_avg})")
        bw, bh = int(bspx_x*b_factor), int(bspx_y*b_factor)
        fw, fh = int(bspx_avg*f_factor), int(bspx_avg*f_factor)

    if (mask is None) and (apply_mask==True):
        _, mask = mask_dilation(imagename, PLOT=False,
                                sigma=3, iterations=2, dilation_size=10)
        bkg = sep.Background(data_2D, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh)

    else:
        bkg = sep.Background(data_2D,bw=bw, bh=bh, fw=fw, fh=fh)
    bkg_rms = bkg.rms()
    bkg_image = bkg.back()
    if show_map == True:
        plt.imshow(bkg_image,origin='lower')
        plt.title(f"max(bkg)/max(data)={(bkg_image.max()/data_2D.max()):.6f}")
        plt.colorbar()
        # plt.clf()
        # plt.close()
    return(bkg)

def distances_from_reference(x_positions, y_positions, reference_coordinate):
    # Reference coordinate
    x_ref, y_ref = reference_coordinate

    # Calculate distances to the reference coordinate
    distances = np.sqrt((x_positions - x_ref)**2 + (y_positions - y_ref)**2)

    # Return the distances sorted
#     sorted_distances = np.sort(distances)

    return distances

def sep_source_ext(imagename, residualname=None, 
                   sigma=6.0, iterations=2, dilation_size=None,
                   deblend_nthresh=100, deblend_cont=0.005, maskthresh=0.0,
                   gain=1.0, filter_kernel=None, mask=None,
                   segmentation_map=False, clean_param=1.0, clean=True,
                   minarea=20, filter_type='matched', 
                   sort_by='distance',
                   bw=64, bh=64, fw=3, fh=3, ell_size_factor=2,
                   apply_mask=False, sigma_mask=6, minarea_factor=1.0,
                   npixels=None,
                   show_bkg_map=True, show_detection=False):
    """
    Simple source extraction algorithm (using SEP https://sep.readthedocs.io/en/v1.1.x/).

    Parameters
    ----------
    imagename : str
        Path to the image.
    sigma : float
        Sigma level for detection.
    iterations : int
        Number of iterations for the mask dilation.
    dilation_size : int
        Size of the dilation kernel.
    deblend_nthresh : int
        Number of thresholds for deblending.
    deblend_cont : float
        Minimum contrast ratio for deblending.
    maskthresh : float
        Threshold for the mask.
    gain : float
        Gain of the image.
    filter_kernel : array
        Filter kernel for the convolution.
    mask : array
        Mask to be applied to the image.
    segmentation_map : bool
        If True, returns the segmentation map.
    clean_param : float
        Cleaning parameter.
    clean : bool
        If True, clean the image.
    minarea : int
        Minimum area for detection.
    filter_type : str
        Type of filter to be used.
    sort_by : str
        Sort the output by flux or area.
    bw : int
        Box width for the background estimation.
    bh : int
        Box height for the background estimation.
    fw : int
        Filter width for the background estimation.
    fh : int
        Filter height for the background estimation.
    ell_size_factor : int
        Size of the ellipse to be plotted.
    apply_mask : bool
        If True, apply the mask to the image.
    sigma_mask : float
        Sigma level for the mask.
    minarea_factor : float

    """

    # filter_kernel_5x5 = np.array([
    #     [1, 1, 1, 1, 1],
    #     [1, 2, 2, 2, 1],
    #     [1, 2, 3, 2, 1],
    #     [1, 2, 2, 2, 1],
    #     [1, 1, 1, 1, 1]])

    data_2D = fitsio.read(imagename)
    if len(data_2D.shape) == 4:
        data_2D = data_2D[0][0]
        
    if residualname is not None:
        residual_2D = fitsio.read(residualname)
        if len(residual_2D.shape) == 4:
            residual_2D = residual_2D[0][0]
    
    if residualname is not None:
        m, s = np.mean(data_2D), mad_std(residual_2D)
    else:
        m, s = np.mean(data_2D), mad_std(data_2D)

    if apply_mask and mask is None:
        _, mask = mask_dilation(data_2D, sigma=sigma_mask, iterations=iterations,
                                rms=s,
                                PLOT=True,show_figure=True,
                                dilation_size=dilation_size)
        # data_2D = data_2D_ * mask

    if npixels is None:
        npixels = int(minarea * minarea_factor)
    
    print('+++++++++++++++++++++++')
    print('SEP Filter sizes:')
    print('    bw,bh=(', int(bw), int(bh),')')
    print('    fw,fh=(', int(fw), int(fh),')')
    print(f'   rms={s}')
    print(f'   npixels={npixels}')
    print(f'   deblend_nthresh={deblend_nthresh}')
    print(f'   deblend_cont={deblend_cont}')
    print(f'   clean={clean}')
    print(f'   clean_param={clean_param}')
    print('+++++++++++++++++++++++')
    bkg = sep.Background(data_2D, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh)
    # print(bkg.globalback)
    # print(bkg.globalrms)
    bkg_image = bkg.back()
    bkg_rms = bkg.rms()
    
    data_sub = data_2D  - bkg_image

    if show_bkg_map == True:
        plt.figure()
        # display bkg map.
        plt.imshow(data_sub, interpolation='nearest', cmap='gray', vmin=1*s,
                    vmax=0.8*np.max(data_sub), origin='lower')
        plt.colorbar()
        plt.show()
        plt.figure()
        plt.imshow(bkg_image)
        plt.show()
        # fast_plot2(bkg_rms)

    if mask is not None:
        data_sub = data_sub * mask
    else:
        data_sub = data_sub.copy()
    
    
    if segmentation_map == True:
        objects, seg_maps = sep.extract(data_sub, thresh=sigma * s,
                                        minarea=npixels, filter_type=filter_type,
                                        deblend_nthresh=deblend_nthresh,
                                        deblend_cont=deblend_cont,
                                        filter_kernel=filter_kernel,
                                        maskthresh=maskthresh, gain=gain,
                                        clean=clean, clean_param=clean_param,
                                        segmentation_map=segmentation_map,
                                        err=None, mask=None)
        print('++==>> INFO: Total number of Sources/Structures (deblended) = ', len(objects))
    else:
        objects = sep.extract(data_sub, thresh=sigma * s,
                              minarea=npixels, filter_type=filter_type,
                              deblend_nthresh=deblend_nthresh,
                              deblend_cont=deblend_cont, filter_kernel=filter_kernel,
                              maskthresh=maskthresh, gain=gain,
                              clean=clean, clean_param=clean_param,
                              segmentation_map=segmentation_map,
                              err=None, mask=None)

    # len(objects)
    from matplotlib.patches import Ellipse
    from skimage.draw import ellipse

    # m, s = np.mean(data_sub), np.std(data_sub)
    if show_detection == True:
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                       vmin=s, vmax=0.2*np.nanmax(data_sub), origin='lower')

    masks_regions = []

    if mask is not None:
        ell_size_factor = 0.05 * np.sqrt(np.sum(mask) / np.pi)
    
    y, x = np.indices(data_2D.shape[:2])
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=2 * ell_size_factor * objects['a'][i],
                    height=2 * ell_size_factor * objects['b'][i],
                    angle=objects['theta'][i] * 180. / np.pi)

        xc = objects['x'][i]
        yc = objects['y'][i]
        a = ell_size_factor * objects['a'][i]
        b = ell_size_factor * objects['b'][i]
        theta = objects['theta'][i]
        rx = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
        ry = (y - yc) * np.cos(theta) - (x - xc) * np.sin(theta)

        inside = ((rx / a) ** 2 + (ry / b) ** 2) <= 1
        mask_ell = np.zeros_like(data_2D)
        mask_ell[inside] = True
        if show_detection == True:
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)
        masks_regions.append(mask_ell)

    #         plt.savefig('components_SEP.pdf',dpi=300, bbox_inches='tight')
    flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
                                         3.0, err=bkg.globalrms, gain=1.0)
    # for i in range(len(objects)):
    #     print("object {:d}: flux = {:f} +/- {:f}".format(i, flux[i], fluxerr[i]))
    # objects['b'] / objects['a'], np.rad2deg(objects['theta'])
    # print(objects)
    # sort regions from largest size to smallest size.
    # print(objects['x'], objects['y'])
    _masks_regions = [mask == 1.0 for mask in masks_regions]
    masks_regions = _masks_regions
    mask_areas = []
    mask_fluxes = []
    for mask_comp in masks_regions:
        area_mask = np.sum(mask_comp)
        sum_mask = np.sum(mask_comp * data_2D)
        mask_areas.append(area_mask)
        mask_fluxes.append(sum_mask)
    mask_areas = np.asarray(mask_areas)
    mask_fluxes = np.asarray(mask_fluxes)
    if sort_by == 'area':
        sorted_indices_desc = np.argsort(mask_areas)[::-1]
        sorted_arr_desc = mask_areas[sorted_indices_desc]
    if sort_by == 'flux':
        sorted_indices_desc = np.argsort(mask_fluxes)[::-1]
        sorted_arr_desc = mask_fluxes[sorted_indices_desc]
    if sort_by == 'distance':
        ref_centre = data_2D.shape[0] / 2, data_2D.shape[1] / 2
        distances = distances_from_reference(objects['x'],
                                             objects['y'],
                                             ref_centre
                                             )
        sorted_indices_desc = np.argsort(distances)
        sorted_arr_desc = distances[sorted_indices_desc]
        
    
    
    objects_sorted = {}
    objects_sorted['xc'] = np.asarray([1] * len(objects))
    objects_sorted['yc'] = np.asarray([1] * len(objects))
    for i in range(len(objects)):
        objects_sorted['xc'][i] = objects['x'][sorted_indices_desc[i]]
        objects_sorted['yc'][i] = objects['y'][sorted_indices_desc[i]]

    if show_detection == True:
        for i in range(len(objects)):
            xc = objects['x'][sorted_indices_desc[i]]
            yc = objects['y'][sorted_indices_desc[i]]
            label = str('ID' + str(i + 1))
            label_x = xc + 3 * ell_size_factor
            label_y = yc + 20 * ell_size_factor
            line_end_y = label_y - 5 
            
            text = Text(label_x, label_y, label, ha='center', va='center', color='red')
            ax.add_artist(text)
            ax.plot([xc, label_x], [yc, line_end_y], color='red', alpha=0.4)

        plt.axis('off')
        # plt.show()
        plt.savefig(imagename + '_SEP.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    if segmentation_map == True:
        return (masks_regions, sorted_indices_desc, bkg_image, seg_maps, objects_sorted)
    else:
        return (masks_regions, sorted_indices_desc, bkg_image,  objects_sorted)

# def sep_source_ext(imagename, sigma=10.0, iterations=2, dilation_size=None,
#                    deblend_nthresh=100, deblend_cont=0.005, maskthresh=0.0,
#                    gain=1, filter_kernel=None, mask=None,
#                    segmentation_map=False, clean_param=1.0, clean=True,
#                    minarea=20, filter_type='matched', sort_by='flux',
#                    bw=64, bh=64, fw=3, fh=3, ell_size_factor=2,
#                    apply_mask=False,sigma_mask=6,minarea_factor=1.0,
#                    show_bkg_map=False, show_detection=False):
#     """
#     Simple source extraction algorithm (using SEP https://sep.readthedocs.io/en/v1.1.x/).
#
#
#     """
#     import sep
#     import fitsio
#     import matplotlib.pyplot as plt
#     from matplotlib.text import Text
#     from matplotlib import rcParams
#
#     data_2D = fitsio.read(imagename)
#     if len(data_2D.shape) == 4:
#         data_2D = data_2D[0][0]
#     m, s = np.mean(data_2D), mad_std(data_2D)
#     bkg = sep.Background(data_2D)
#
#     if apply_mask:
#         if mask is not None:
#             data_2D = data_2D * mask
#         else:
#             _, mask = mask_dilation(data_2D, sigma=sigma_mask, iterations=iterations,
#                                     dilation_size=dilation_size)
#             data_2D = data_2D * mask
#
#     # else:
#     #     mask = None
#     bkg = sep.Background(data_2D, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh)
#     # print(bkg.globalback)
#     # print(bkg.globalrms)
#     bkg_image = bkg.back()
#     bkg_rms = bkg.rms()
#
#     if show_bkg_map == True:
#         plt.figure()
#         #display bkg map.
#         plt.imshow(data_2D, interpolation='nearest', cmap='gray', vmin=m - s,
#                    vmax=m + s, origin='lower')
#         plt.colorbar()
#         plt.close()
#         plt.figure()
#         plt.imshow(bkg_image)
#         plt.close()
#     # fast_plot2(bkg_rms)
#     data_sub = data_2D - bkg
#     if segmentation_map == True:
#         npixels = int(minarea * minarea_factor)
#         objects, seg_maps = sep.extract(data_sub, thresh=sigma,
#                                         minarea=npixels, filter_type=filter_type,
#                                         deblend_nthresh=deblend_nthresh,
#                                         deblend_cont=deblend_cont, filter_kernel=filter_kernel,
#                                         maskthresh=maskthresh, gain=gain,
#                                         clean=clean, clean_param=clean_param,
#                                         segmentation_map=segmentation_map,
#                                         err=bkg.globalrms, mask=mask)
#     else:
#         npixels = int(minarea * minarea_factor)
#         objects = sep.extract(data_sub, thresh=sigma,
#                               minarea=npixels, filter_type=filter_type,
#                               deblend_nthresh=deblend_nthresh,
#                               deblend_cont=deblend_cont, filter_kernel=filter_kernel,
#                               maskthresh=maskthresh, gain=gain,
#                               clean=clean, clean_param=clean_param,
#                               segmentation_map=segmentation_map,
#                               err=bkg.globalrms, mask=mask)
#
#     # len(objects)
#     from matplotlib.patches import Ellipse
#     from skimage.draw import ellipse
#
#     m, s = np.mean(data_sub), np.std(data_sub)
#     if show_detection == True:
#         fig, ax = plt.subplots()
#         im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
#                        vmin=m - s, vmax=m + s, origin='lower')
#
#     masks_regions = []
#
#     y, x = np.indices(data_2D.shape[:2])
#     for i in range(len(objects)):
#         e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
#                     width=2 * ell_size_factor * objects['a'][i],
#                     height=2 * ell_size_factor * objects['b'][i],
#                     angle=objects['theta'][i] * 180. / np.pi)
#
#         xc = objects['x'][i]
#         yc = objects['y'][i]
#         a = ell_size_factor * objects['a'][i]
#         b = ell_size_factor * objects['b'][i]
#         theta = objects['theta'][i]
#         rx = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
#         ry = (y - yc) * np.cos(theta) - (x - xc) * np.sin(theta)
#
#         inside = ((rx / a) ** 2 + (ry / b) ** 2) <= 1
#         mask_ell = np.zeros_like(data_2D)
#         mask_ell[inside] = True
#         if show_detection == True:
#             e.set_facecolor('none')
#             e.set_edgecolor('red')
#             ax.add_artist(e)
#         masks_regions.append(mask_ell)
#
#     #         plt.savefig('components_SEP.pdf',dpi=300, bbox_inches='tight')
#     flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
#                                          3.0, err=bkg.globalrms, gain=1.0)
#     for i in range(len(objects)):
#         print("object {:d}: flux = {:f} +/- {:f}".format(i, flux[i], fluxerr[i]))
#     objects['b'] / objects['a'], np.rad2deg(objects['theta'])
#
#     # sort regions from largest size to smallest size.
#     mask_areas = []
#     mask_fluxes = []
#     for mask_comp in masks_regions:
#         area_mask = np.sum(mask_comp)
#         sum_mask = np.sum(mask_comp * data_2D)
#         mask_areas.append(area_mask)
#         mask_fluxes.append(sum_mask)
#     mask_areas = np.asarray(mask_areas)
#     mask_fluxes = np.asarray(mask_fluxes)
#     if sort_by == 'area':
#         sorted_indices_desc = np.argsort(mask_areas)[::-1]
#         sorted_arr_desc = mask_areas[sorted_indices_desc]
#     if sort_by == 'flux':
#         sorted_indices_desc = np.argsort(mask_fluxes)[::-1]
#         sorted_arr_desc = mask_fluxes[sorted_indices_desc]
#
#     objects_sorted = {}
#     objects_sorted['xc'] = np.asarray([1] * len(objects))
#     objects_sorted['yc'] = np.asarray([1] * len(objects))
#     for i in range(len(objects)):
#         objects_sorted['xc'][i] = objects['x'][sorted_indices_desc[i]]
#         objects_sorted['yc'][i] = objects['y'][sorted_indices_desc[i]]
#
#     if show_detection == True:
#         for i in range(len(objects)):
#             xc = objects['x'][sorted_indices_desc[i]]
#             yc = objects['y'][sorted_indices_desc[i]]
#             label = str('ID' + str(i + 1))
#             text = Text(xc + 10 * ell_size_factor, yc + 3 * ell_size_factor, label, ha='center', va='center', color='red')
#             ax.add_artist(text)
#
#         plt.axis('off')
#         # plt.show()
#         plt.savefig(imagename + '_SEP.jpg', dpi=300, bbox_inches='tight')
#         plt.show()
#
#     if segmentation_map == True:
#         return (masks_regions, sorted_indices_desc, seg_maps, objects_sorted)
#     else:
#         return (masks_regions, sorted_indices_desc, objects_sorted)


"""
  ___        _             _           _   _                 
 / _ \ _ __ | |_ _ __ ___ (_)___  __ _| |_(_) ___  _ __  ___ 
| | | | '_ \| __| '_ ` _ \| / __|/ _` | __| |/ _ \| '_ \/ __|
| |_| | |_) | |_| | | | | | \__ \ (_| | |_| | (_) | | | \__ \
 \___/| .__/ \__|_| |_| |_|_|___/\__,_|\__|_|\___/|_| |_|___/
      |_|                                                    
"""


def ellipse_fitting(image, center=None):
    """
    TTESTING: DO NOT USE!
    """
    if center is None:
        center = np.array([image.shape[0] // 2, image.shape[1] // 2])
    y, x = np.indices((image.shape))
    x -= center[0]
    y -= center[1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    def fit_function(params, x, y, r):
        a, b, pa = params
        pa = np.deg2rad(pa)
        xr = (x * np.cos(pa) + y * np.sin(pa)) / b
        yr = (-x * np.sin(pa) + y * np.cos(pa)) / a
        return np.sqrt(xr ** 2 + yr ** 2)

    def residuals(params, x, y, r, image):
        r_fit = fit_function(params, x, y, r)
        residual = (image - ndi.map_coordinates(image, [r_fit], order=1)) ** 2
        return residual.ravel()

    scores = []
    fitted_params = []

    for r0 in np.unique(r):
        print(r0)
        mask = (r >= r0) & (r < r0 + 1)
        x_ = x[mask].ravel()
        y_ = y[mask].ravel()
        image_ = image[mask].ravel()
        r_ = r[mask].ravel()
        params_init = np.array([r0, r0, 0.0])
        params_fit = opt.least_squares(residuals, params_init, args=(x_, y_, r_, image_))
        #         print(opt.least_squares(residuals, params_init, args=(x_, y_, r_, image_)))
        score = np.sum((fit_function(params_fit.x, x_, y_, r_) - r_) ** 2) / len(r_)
        scores.append(score)
        fitted_params.append(params_fit.x)
    return fitted_params, scores

def ellipse_fit(image, dr):
    """
    TTESTING: DO NOT USE!
    """
    # Get the image shape
    shape = image.shape

    # Define the center of the image
    center = (shape[0]//2, shape[1]//2)

    # Define the maximum radius to consider
    max_r = np.min(shape)//2

    # Define the number of steps to consider for each radial step
    n_steps = 20

    # Define the radial distances
    r = np.arange(0, max_r, dr)

    # Pre-allocate arrays for the fit parameters
    a = np.zeros_like(r)
    b = np.zeros_like(r)
    theta = np.zeros_like(r)
    x0 = np.zeros_like(r)
    y0 = np.zeros_like(r)

    # Define the optimization function to minimize
    def optimize_ellipse(params, x, y, image):
        a, b, theta, x0, y0 = params
        X, Y = np.meshgrid(x, y)
        X_rot = (X-x0)*np.cos(theta) + (Y-y0)*np.sin(theta)
        Y_rot = -(X-x0)*np.sin(theta) + (Y-y0)*np.cos(theta)
        Z = (X_rot/a)**2 + (Y_rot/b)**2
        Z = np.where(Z<=1, 1, 0)
        Z = Z * image
        Z_sum = np.sum(Z)
        return Z_sum

    # Loop over each radial distance
    for i, ri in enumerate(r):
        # Define the angular step size
        d_theta = 2*np.pi / n_steps

        # Define the initial guess for the fit parameters
        x = np.arange(center[0]-ri, center[0]+ri+1, 1)
        y = np.arange(center[1]-ri, center[1]+ri+1, 1)
        X, Y = np.meshgrid(x, y)
        params0 = (ri, ri, 0, center[0], center[1])

        # Optimize the fit parameters
        res = opt.minimize(optimize_ellipse, params0, args=(x, y, image))
        params_fit = res.x
        a[i], b[i], theta[i], x0[i], y0[i] = params_fit

    # Return the fit parameters
    return a, b, theta, x0,


"""

                        ___
                       |_ _|_ __ ___   __ _  __ _  ___
                        | || '_ ` _ \ / _` |/ _` |/ _ \
                        | || | | | | | (_| | (_| |  __/
                       |___|_| |_| |_|\__,_|\__, |\___|
                                            |___/
        ____                                           _ _   _
       |  _ \  ___  ___ ___  _ __ ___  _ __   ___  ___(_) |_(_) ___  _ __
       | | | |/ _ \/ __/ _ \| '_ ` _ \| '_ \ / _ \/ __| | __| |/ _ \| '_ \
       | |_| |  __/ (_| (_) | | | | | | |_) | (_) \__ \ | |_| | (_) | | | |
       |____/ \___|\___\___/|_| |_| |_| .__/ \___/|___/_|\__|_|\___/|_| |_|
                                      |_|

"""

def setup_model_components(n_components=2):
    """
        Set up a single sersic component or a composition of sersic components.

        Uses the LMFIT objects to easily create model components.

        fi_ is just a prefix to distinguish the set of parameters for each component.

    """
    if n_components == 1:
        smodel2D = Model(sersic2D, prefix='f1_') + Model(FlatSky, prefix='s_')
    if n_components > 1:
        smodel2D = Model(sersic2D, prefix='f1_')
        for i in range(2, n_components + 1):
            smodel2D = smodel2D + Model(sersic2D, prefix='f' + str(i) + '_')
        smodel2D = smodel2D + Model(FlatSky, prefix='s_')
    return (smodel2D)


def construct_model_parameters(n_components, params_values_init_IMFIT=None,
                               init_constraints=None,observation_type='radio',
                               constrained=True, fix_n=False, fix_value_n=False,
                               fix_x0_y0=False,dr_fix = None,fix_geometry=True,
                               fix_max_value_Rn = False,
                               init_params=0.25, final_params=4.0):
    """
    This function creates a single or multi-component Sersic model to be fitted
    onto an astronomical image.

    It uses the function setup_model_components to create the model components and specify/constrain
    the parameters space in which each parameter will vary during the fit.

    DEV NOTES:

        Note that this function handles parameter/model generation in four different ways:
            -- free parameters (params_values_init_IMFIT=None, init_constraints=None,
            constrained=False)
            -- constrained parameters from IMFIT (params_values_init_IMFIT=np.array of IMFIT
            parameters, init_constraints=None, constrained=True)
            -- initial parameter from a source extraction object and no constraints
            (params_values_init_IMFIT=None, init_constraints=SE.object,
            constrained=False)
            -- initial and constrained parameters from a source extraction object
            (params_values_init_IMFIT=None, init_constraints=SE.object, constrained=True)

        These are the four possible combinations of parameters and constraints that can be used.
        However, only the last one was tested extensively and is currently being used as default.
        It showed to be the most robust and reliable way to fit the model to the data.
        The other methods need some more testing and improvements.


    Note:

    Parameters
    ----------
    n_components : int, optional
        Number of components to be fitted. The default is None.
    params_values_init_IMFIT : list, optional
        List of initial parameters from a IMFIT config file to be used as initial guess for the fit.
        The default is None.
    init_constraints : dict, optional
        Dictionary containing initial constraints to be used as initial guess
        for the fit. The default is None.
    constrained : bool, optional
        If True, then the fit will be constrained. The default is True.
    fix_n : bool, optional
        If True, then the Sersic index will be fixed to 0.5. The default is False.
    fix_value_n : float, optional
        If True, then the Sersic index will be fixed to this value. The default is False.
    fix_x0_y0 : bool, optional
        If True, then the centre position will be fixed to the initial guess
        value. The default is False.
    dr_fix : float, optional
        If True, then the centre position will be fixed to the initial guess
        value. The default is False.
    fix_geometry : bool, optional
        If True, then the geometry of the components will be fixed to the
        initial guess value. The default is True.

    ----------------------------
    These will be removed in a future version.
    init_params : float, optional
        Initial parameter value. The default is 0.25.
    final_params : float, optional
        Final parameter value. The default is 4.0.
    """
    
    if n_components is None:
        n_components = len(params_values_init_IMFIT) - 1

    smodel2D = setup_model_components(n_components=n_components)
    # print(smodel2D)
    model_temp = Model(sersic2D)
    dr = 10


    # params_values_init_IMFIT = [] #grid of parameter values, each row is the
    # parameter values of a individual component

    if params_values_init_IMFIT is not None:
        """This takes the values from an IMFIT config file as init
        params and set number of components. This is useful to use results 
        from IMFIT, for example. 
        
        WARNING: This portion of the code was not revised and tested properly 
        since it was implemented. It will remain here for practical reasons 
        and for future improvements and experiments. 
        """
        for i in range(0, n_components):
            # x0, y0, PA, ell, n, In, Rn = params_values_init_IMFIT[i]
            x0, y0, PA, ell, n, In, Rn = params_values_init_IMFIT[i]
            if fix_x0_y0 is not False:
                fix_x0_y0_i = fix_x0_y0[i]
                dr_fix_i = dr_fix[i]
            else:
                fix_x0_y0_i = False
                dr_fix_i = False

            if fix_n is not False:
                fix_n_i = fix_n[i]
            else:
                fix_n_i = False

            ii = str(i + 1)
            if constrained == True:
                for param in model_temp.param_names:
                    # apply bounds to each parameter.
                    smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                            value=eval(param),
                                            min=init_params * eval(param),
                                            max=final_params * eval(param))

                    # still, some of them must be treated in particular.
                    if param == 'n':
                        if fix_n_i == True:
                            print('++==>> Fixing sersic index of component',i+1,' to 0.5')
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=0.5, min=0.49, max=0.51)
                        else:
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=eval(param), min=0.3,
                                max=8.0)
                    if param == 'x0':
                        if fix_x0_y0_i is not False:
                            """
                            Fix centre position by no more than dr_fix.
                            """
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=eval(param),
                                min=eval(param) - dr_fix_i,
                                max=eval(param) + dr_fix_i)
                        else:
                            if (init_constraints is not None) and (
                                    init_constraints['ncomps'] == n_components):
                                """
                                If initial constraints using Petro analysis are
                                provided, then use!
                                """
                                ddxx = 3  # the offset on x direction from Petro centre.
                                x0 = init_constraints['c' + ii + '_x0c']
                                x0_max = x0 + ddxx
                                x0_min = x0 - ddxx
                                print('Limiting ', param)
                                smodel2D.set_param_hint(
                                    'f' + str(i + 1) + '_' + param,
                                    value=x0,
                                    min=x0_min,
                                    max=x0_max)
                            else:
                                """
                                Then, consider that input File is good, then
                                give some bound
                                around those values.
                                """
                                print('Limiting ', param)
                                smodel2D.set_param_hint(
                                    'f' + str(i + 1) + '_' + param,
                                    value=eval(param),
                                    min=eval(param) - dr,
                                    max=eval(param) + dr)
                    if param == 'y0':
                        if fix_x0_y0_i is not False:
                            """
                            Fix centre position by no more than dr_fix_i.
                            """
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=eval(param),
                                min=eval(param) - dr_fix_i,
                                max=eval(param) + dr_fix_i)
                        else:
                            if (init_constraints is not None) and (
                                    init_constraints['ncomps'] == n_components):
                                """
                                If initial constraints is using Petro analysis
                                are provided, then use!
                                """
                                ddyy = 3  # the offset on x direction from Petro centre.
                                y0 = init_constraints['c' + ii + '_y0c']
                                y0_max = y0 + ddyy
                                y0_min = y0 - ddyy
                                print('Limiting ', param)
                                smodel2D.set_param_hint(
                                    'f' + str(i + 1) + '_' + param,
                                    value=y0,
                                    min=y0_min,
                                    max=y0_max)
                            else:
                                """
                                Then, consider that input File is good, then give
                                some bound around those values.
                                """
                                print('Limiting ', param)
                                smodel2D.set_param_hint(
                                    'f' + str(i + 1) + '_' + param,
                                    value=eval(param),
                                    min=eval(param) - dr,
                                    max=eval(param) + dr)
                    if param == 'ell':
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=eval(param), min=0.001,
                                                max=0.8)
                    if param == 'PA':
                        if (init_constraints is not None) and (
                                init_constraints['ncomps'] == n_components):
                            _PA = init_constraints['c' + ii + '_PA']
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=_PA, min=_PA - 60,
                                max=_PA + 60)
                        else:
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=eval(param), min=-50.0,
                                max=190.0)
                    if param == 'In':
                        if (init_constraints is not None) and (
                                init_constraints['ncomps'] == n_components):
                            I50 = init_constraints['c' + ii + '_I50']
                            I50_max = I50 * 10
                            I50_min = I50 * 0.1
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=I50_max, min=I50_min, max=I50_max)
                        else:
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=eval(param),
                                min=init_params * eval(param),
                                max=10 * final_params * eval(param))
            if constrained == False:
                for param in model_temp.param_names:
                    smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                            value=eval(param), min=0.000001)
                    if param == 'n':
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=0.5, min=0.3, max=8)
                    if param == 'PA':
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=45, min=-50.0, max=190)
                    if param == 'ell':
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=eval(param), min=0.001,
                                                max=0.99)
                    if param == 'In':
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=eval(param), min=0.0000001,
                                                max=10.0)
                    if param == 'Rn':
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=eval(param), min=0.5,
                                                max=300.0)
                    if param == 'x0':
                        print('Limiting ', param)
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=eval(param),
                                                min=eval(param) - dr * 5,
                                                max=eval(param) + dr * 5)
                    if param == 'y0':
                        # print('Limiting ',param)
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=eval(param),
                                                min=eval(param) - dr * 5,
                                                max=eval(param) + dr * 5)

        smodel2D.set_param_hint('s_a', value=1, min=0.99, max=1.01)
        # smodel2D.set_param_hint('s_a', value=1, min=0.3, max=6.0)
    else:
        if init_constraints is not None:
            """
            This is the default option to use, and the more robust.
            """
            if constrained == True:
                """
                This is the default option to use, and the more robust.
                """
                for j in range(init_constraints['ncomps']):
                    if fix_n is not False:
                        fix_n_j = fix_n[j]
                        fix_value_n_j = fix_value_n[j]
                    else:
                        fix_n_j = False
                        fix_value_n_j = 1.0 # will be skipped.

                    if fix_max_value_Rn is not False:
                        fix_max_value_Rn_j = fix_max_value_Rn[j]
                    else:
                        fix_max_value_Rn_j = False
                        

                    if fix_x0_y0 is not False:
                        fix_x0_y0_j = fix_x0_y0[j]
                        dr_fix_j = dr_fix[j]
                    else:
                        fix_x0_y0_j = False
                        dr_fix_j = False

                    if fix_geometry is not False:
                        fix_geometry_j = fix_geometry[j]
                    else:
                        fix_geometry_j = False

                    jj = str(j + 1)
                    for param in model_temp.param_names:
                        #                         smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                        #                                                 value=eval(param), min=0.000001)
                        if (param == 'n'):
                            if (fix_n_j == True):
                                print(f'++==>> Fixing sersic index of component {j+1} to {fix_value_n_j}.')
                                dn = 0.01
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=fix_value_n_j,
                                    min=fix_value_n_j-dn, max=fix_value_n_j+dn)
                            else:
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=0.5, min=0.3, max=8.0)

                        """
                        Constraining PA and q from the pre-analysis of the image
                        (e.g. petro analysys) is not robust, since that image is
                        already convolved with the restoring beam, which can be
                        rotated. So the PA and q of a DECONVOLVED_MODEL
                        (the actual minimization problem here) can be different
                        from the PA and q of a CONVOLVED_MODEL
                        (as well Rn_conv > Rn_decon; In_conv< In_deconv).
                        So, at least we give some large bound.
                        """
                        if param == 'PA':
                            dO = 90
                            _PA = init_constraints['c' + jj + '_PA']
                            PA_max = _PA + dO
                            PA_min = _PA - dO
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=_PA, min=PA_min, max=PA_max)
                        if param == 'ell':
                            ell = 1 - init_constraints['c' + jj + '_q']
                            ell_min = ell * 0.2
                            #                         if ell + dell <= 1.0:
                            # if ell * 1.0 <= 0.5:
                            if ell <= 0.15:
                                ell_max = 0.15
                            else:
                                ell_max = 0.75
                                
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=ell, min=ell_min, max=ell_max)
                            # print(f" -- Init value for ell: {ell}")
                            # print(f" -- min value for ell: {ell_min}")
                            # print(f" -- max value for ell: {ell_max}")
                            # smodel2D.set_param_hint(
                            #     'f' + str(j + 1) + '_' + param,
                            #     value=ell, min=0.01, max=0.9)


                        if param == 'cg':
                            if fix_geometry_j == True:
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=0.0, min=-0.01, max=0.01)
                            else:
                                print('Using general elliptical geometry during '
                                      'fitting... may take longer.')
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=0.0, min=-2.0, max=2.0)

                        if param == 'In':
                            I50 = init_constraints['c' + jj + '_I50']
                            """
                            A high value of I50 is required because the
                            deconvolved model has a higher peak intensity
                            (and therefore the same for the I50 region) than the
                            convolved model. The PSF convolution attenuates significantly
                            the peak intensity.
                            
                            To-do:
                            1. Use a more robust approach, from the theoretical prediction of a 
                            deconvolved signal from a convolved signal with a Gaussian 
                            kernel.
                            """
                            if observation_type == 'radio':
                                I50_max = I50 * 100
                                I50_min = I50 * 0.1
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=I50, min=I50_min, max=I50_max)
                            else:
                                I50_max = I50 * 25
                                I50_min = I50 * 0.05
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=I50, min=I50_min, max=I50_max)
                        if param == 'Rn':
                            if fix_max_value_Rn_j is not False:
                                """
                                Fix the value of Rn to a maximum value.
                                This is useful if one wants to fit a delta function.
                                That is, Rn=1.0 ~ delta functio, 1 pixel component.
                                """
                                print(f" ++==>> Limiting {param}_{jj} to {fix_max_value_Rn_j}")
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=fix_max_value_Rn_j*0.9, 
                                    min=0.1, 
                                    max=fix_max_value_Rn_j)
                                # smodel2D.set_param_hint(
                                #     'f' + str(j + 1) + '_' + param,
                                #     value=fix_max_value_Rn_j, 
                                #     min=fix_max_value_Rn_j-0.01, 
                                #     max=fix_max_value_Rn_j+0.01)
                                
                            else:
                                R50 = init_constraints['c' + jj + '_R50']
                                # R50_max = R50 * 4.0
                                # R50_max = init_constraints['c' + jj + '_Rp']
                                if observation_type == 'radio':
                                    R50_max = R50 * 1.0
                                    R50_min = R50 * 0.01 #should be small.
                                    smodel2D.set_param_hint(
                                        'f' + str(j + 1) + '_' + param,
                                        value=R50, min=R50_min, max=R50_max)
                                else:
                                    R50_max = R50 * 2.0
                                    R50_min = R50 * 0.5 #should be small.
                                    smodel2D.set_param_hint(
                                        'f' + str(j + 1) + '_' + param,
                                        value=R50, min=R50_min, max=R50_max)

                        if param == 'x0':
                            if fix_x0_y0_j is not False:
                                """
                                Fix centre position by no more than dr_fix.
                                """
                                x0c = init_constraints['c' + jj + '_x0c']
                                x0_max = x0c + dr_fix_j
                                x0_min = x0c - dr_fix_j
                                print(f" ++==>> Limiting {param}={x0c}+/-{dr_fix_j}")
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=x0c,
                                    min=x0_min,
                                    max=x0_max)
                            else:
                                ddxx = 10
                                x0c = init_constraints['c' + jj + '_x0c']
                                x0_max = x0c + ddxx
                                x0_min = x0c - ddxx
                                print(f" ++==>> Limiting {param}={x0c}+/-{ddxx}")
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=x0c,
                                    min=x0_min,
                                    max=x0_max)
                        if param == 'y0':
                            if fix_x0_y0_j is not False:
                                """
                                Fix centre position by no more than dr_fix.
                                """
                                y0c = init_constraints['c' + jj + '_y0c']
                                y0_max = y0c + dr_fix_j
                                y0_min = y0c - dr_fix_j
                                print(f" ++==>> Limiting {param}={y0c}+/-{dr_fix_j}")
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=y0c,
                                    min=y0_min,
                                    max=y0_max)
                            else:
                                ddyy = 10
                                y0c = init_constraints['c' + jj + '_y0c']
                                y0_max = y0c + ddyy
                                y0_min = y0c - ddyy
                                print(f" ++==>> Limiting {param}={y0c}+/-{ddyy}")
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=y0c,
                                    min=y0_min,
                                    max=y0_max)
            if constrained == False:
                for j in range(init_constraints['ncomps']):
                    jj = str(j + 1)
                    for param in model_temp.param_names:
                        smodel2D.set_param_hint('f' + str(j + 1) + '_' + param,
                                                value=eval(param), min=0.000001,
                                                max=0.5)
                        if param == 'n':
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=0.5, min=0.3, max=8)
                        if param == 'PA':
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=45, min=-50.0, max=190)
                        if param == 'ell':
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=0.2, min=0.001, max=0.9)
                        if param == 'In':
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=0.1, min=0.0000001, max=10.0)
                        if param == 'Rn':
                            Rp = init_constraints['c' + jj + '_x0c']
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=10, min=2.0, max=2 * Rp)

                        """
                        This is not contrained, but at least is a good idea to
                        give some hints to the centre (x0,y0).
                        """
                        if param == 'x0':
                            ddxx = 20
                            x0c = init_constraints['c' + jj + '_x0c']
                            x0_max = x0c + ddxx
                            x0_min = x0c - ddxx
                            print('Limiting ', param)
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=x0c,
                                min=x0_min,
                                max=x0_max)
                        if param == 'y0':
                            ddyy = 20
                            y0c = init_constraints['c' + jj + '_y0c']
                            y0_max = y0c + ddyy
                            y0_min = y0c - ddyy
                            print('Limiting ', param)
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=y0c,
                                min=y0_min,
                                max=y0_max)

            smodel2D.set_param_hint('s_a', value=1, min=0.99, max=1.01)
            # smodel2D.set_param_hint('s_a', value=1, min=0.3, max=6.0)
        else:
            '''
            Run a complete free-optimization.
            '''
            try:
                for j in range(n_components):
                    jj = str(j + 1)
                    for param in model_temp.param_names:
                        smodel2D.set_param_hint('f' + str(j + 1) + '_' + param,
                                                value=0.5, min=0.000001)
                        if param == 'n':
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=0.5, min=0.3, max=6)
                smodel2D.set_param_hint('s_a', value=1, min=0.99, max=1.01)
                # smodel2D.set_param_hint('s_a', value=1, min=0.3, max=6.0)
            except:
                print('Please, if not providing initial parameters file,')
                print('provide basic information for the source.')
                return (ValueError)

    params = smodel2D.make_params()
    # print(smodel2D.param_hints)
    
    # Create a PrettyTable object
    table = PrettyTable()
    table.field_names = ["Parameter", "Value", "Min", "Max"]

    # Add rows to the table with formatted values
    for key, val in smodel2D.param_hints.items():
        table.add_row([
            key, 
            f"{val['value']:.3f}", 
            f"{val['min']:.3f}", 
            f"{val['max']:.3f}"
        ])

    # Print the table
    print(table)
    
    return (smodel2D, params)


def constrain_nelder_mead_params(params,
                                 max_factor = 1.03,
                                 min_factor = 0.97):
    """
    Constrain Nelder-Mead optimised parameters.
    Since Nelder-Mead is robust, we can feed these values
    into Least-Squares.

    This is a workaround since nelder-mead does not provide statistical errors.
    SO, this is an attempt to produce a set of parameter distributions around the best fit ones.
    """
    params_copy = params.copy()
    for name, param in params_copy.items():
        value = param.value
        if param.value > 0:
            param.max = value * max_factor
        if param.value < 0:
            param.max = value * min_factor
        if param.value > 0:
            param.min = value * min_factor
        if param.value < 0:
            param.min = value * max_factor
        if param.value == 0:
            param.max = 0.01
            param.min = -0.01
    return(params_copy)


def generate_random_params_uniform_old(params, param_errors):
    # Generate a set of random numbers from a normal distribution with mean 0 and standard deviation 1
    # try:
    #     # Scale the random numbers by the standard errors of the parameters
    param_errors_corr = param_errors.copy()
    random_nums = np.random.uniform(-5, 5, size=len(params))
    scaled_random_nums = random_nums * param_errors
    random_params = params + scaled_random_nums
    return random_params

def generate_random_params_uniform(params, param_errors, sigma_errors=5.0):
    # Generate a set of random numbers from a normal distribution with mean 0 and standard deviation 1

    random_params = np.random.uniform(params-sigma_errors*param_errors, 
                                    params+sigma_errors*param_errors, 
                                    size=len(params))
    return random_params


def generate_random_params_normal(params, param_errors, sigma_errors=3.0):
    # Generate a set of random numbers from a normal distribution with mean 0 and standard deviation 1
    param_errors_corr = param_errors.copy()

    #     ndim_params = int((len(params)-1)/(len(params[0:8])))
    #     weights = np.asarray([3,3,5,0.05,0.1,0.00001,5,0.01])
    #     weights_m = np.tile(weights, (ndim_params, 1))
    #     weights_f = weights_m.flatten()
    #     weights_f = np.append(weights_f,np.asarray([0.1]))
    # #     np.random.seed(123)

    #     # Generate a random distribution of values between -1 and 1
    #     random_noise = np.random.uniform(low=-1, high=1, size=len(weights_f))
    # #     random_noise = np.random.random(len(weights_f)) * weights_f

    random_params = np.random.normal(params, 
                                    #  abs(sigma_errors*params*0.3), 
                                     sigma_errors*param_errors, 
                                     size=len(params))
    
    # scaled_random_nums = random_nums * param_errors  # + random_noise
    #     random_nums = np.random.normal(0.0, 0.1, size=len(params))
    # scaled_random_nums = random_nums * params
    # random_params = scaled_random_nums
    # random_params = params + scaled_random_nums
    return random_params


def generate_random_params_tukeylambda(params, param_errors):
    from scipy.stats import tukeylambda
    # Generate a set of random numbers from a tukeylambda distribution.
    param_errors_corr = param_errors.copy()
    random_nums = tukeylambda.rvs(0.5, size=len(params)) * 10
    scaled_random_nums = random_nums * param_errors
    #     random_nums =  tukeylambda.rvs(0.5, size=len(params)) *0.1
    #     scaled_random_nums = random_nums * params
    random_params = params + scaled_random_nums
    return random_params

def add_extra_component(petro_properties, copy_from_id):
    """
    Create another component from a dictionary (petro_properties) having
    photometric properties for N detected components in an image.

    Params
    ------
    petro_properties: dict
        Contain parameters of a number o N components obtained
        by a petrosian analysis of all detected sources.
        Example (these are actually the keys() from the dictionary):
        ['c1_PA', 'c1_q', 'c1_area', 'c1_Re',
        'c1_x0c', 'c1_y0c', 'c1_label', 'c1_R50',
        'c1_Snu', 'c1_Rp', 'c1_Rpidx', 'c1_rlast',
        'c1_I50']
    copy_from: int
        From which component copy parameters from.
        This is useful, for example, the source has two components detected,
        1 compact and the other a structure that can not be modelled by a single
        sersic function. Then, we need one function to model the compact structure,
        but 2 sersic functions to model the other structure.

        Assume that we have a blob surrounded by a disky emission ( detected
        as one source). Both are placed on the same region, on top of each other
        (e.g. from optical, we can call for example] a bulge and
        a disk). We need two functions to model this region.

        So, if component i=1 is the blob (or the bulge) we copy the parameters from it and
        create a second component. We just have to ajust some of the parameters.
        E.g. the effective radius of this new component, is in principle, larger than the original component.
        As well, the effective intensity will be smaller because we are adding a component
        further away from the centre. Other quantities, however, are uncertain, such as the Sersic index, position angle
        etc, but may be (or not!) close to those of component i.

    """

    from collections import OrderedDict
    dict_keys = list(petro_properties.keys())
    unique_list = list(OrderedDict.fromkeys(
        [elem.split('_')[1] for elem in dict_keys if '_' in elem]))
    #     print(unique_list)
    print(unique_list)
    petro_properties_copy = petro_properties.copy()
    new_comp_id = petro_properties['ncomps'] + 1
    for k in range(len(unique_list)):
        #         print(unique_list[k])
        # do not change anything for other parameters.
        petro_properties_copy['c' + str(new_comp_id) + '_' + unique_list[k]] = \
        petro_properties_copy['c' + str(copy_from_id) + '_' + unique_list[k]]
        if unique_list[k] == 'R50':
            # multiply the R50 value by a factor, e.g., 5.0
            # factor = 5.0
            factor = np.min([petro_properties['cg_Rp'],2*petro_properties['cg_Rp'] / abs(petro_properties['cg_Rp']-petro_properties_copy['c' + str(copy_from_id) + '_' + unique_list[k]])])
            print(factor )
            petro_properties_copy[
                'c' + str(new_comp_id) + '_' + unique_list[k]] = \
            petro_properties_copy[
                'c' + str(copy_from_id) + '_' + unique_list[k]] * factor
        if unique_list[k] == 'I50':
            # multiply the I50 value by a factor, e.g., 0.2
            factor = 0.01
            petro_properties_copy[
                'c' + str(new_comp_id) + '_' + unique_list[k]] = \
            petro_properties_copy[
                'c' + str(copy_from_id) + '_' + unique_list[k]] * factor
        if unique_list[k] == 'Rp':
            petro_properties_copy[
                'c' + str(new_comp_id) + '_' + unique_list[k]] = petro_properties['cg_Rp']
                    
    # update number of components
    petro_properties_copy['ncomps'] = petro_properties_copy['ncomps'] + 1
    return (petro_properties_copy)



def sorted_detected_coordinates(reference_x, 
                                reference_y, 
                                detected_x, 
                                detected_y, 
                                reference_coordinate, 
                                tolerance=2):
    """
    This function will sort a new set of detected coordinates 
    by distance to a reference position in the same order as a reference set 
    of coordinates. 
    
    Consider for example, that in a high-resolution image, we have a set of
    detected structures such as ID1, ID2, ID3. 
    
    If in a new image (e.g. a low-resolution image) we detect the same structures
    in addition to other structures, we will have for example: 
        ID1_new, ID2_new, ID3_new, ID4_new, ID5_new.
    However, the order of the detected structures can be any. For example,
    ID1_new may be the same as ID1, but not for the others. 
    
    This function will sort the new coordinates in the same order they appear 
    in the reference coordinates, in addition to the extra detected coordinates.
    
    A typical use case is: 
    - Perform a source detection in a VLA image (reference image), at 33 GHz.
    - Perform a source detection in an e-MERLIN image, at 6 GHz.
    The number of Structures may differ, but some of them are the same.
    This allow us to connect the labels of the structures in both images.
    
    """
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # Reference coordinate
    # Reference coordinate
    x_ref, y_ref = reference_coordinate

    # Calculate distances to the reference coordinate for both sets of coordinates
    reference_distances = np.sqrt((reference_x - x_ref)**2 + (reference_y - y_ref)**2)
    detected_distances = np.sqrt((detected_x - x_ref)**2 + (detected_y - y_ref)**2)

    # Sort the reference and detected indices by distance
    sorted_reference_indices = np.argsort(reference_distances)
    sorted_detected_indices = np.argsort(detected_distances)

    # Initialize lists to store the sorted detected coordinates, distances, and indices
    sorted_detected_coordinates = []
    sorted_detected_distances = []
    sorted_detected_original_indices = []

    # Match reference coordinates to detected coordinates by closest distance
    for ref_idx in sorted_reference_indices:
        if len(sorted_detected_indices) > 0:
            closest_idx = sorted_detected_indices[0]
            sorted_detected_coordinates.append([detected_x[closest_idx], detected_y[closest_idx]])
            sorted_detected_distances.append(detected_distances[closest_idx])
            sorted_detected_original_indices.append(closest_idx)
            # Remove the matched index to avoid duplicate matching
            # sorted_detected_indices = np.delete(sorted_detected_indices, 0)

    # Add any remaining detected coordinates that were not matched
    for det_idx in sorted_detected_indices:
        sorted_detected_coordinates.append([detected_x[det_idx], detected_y[det_idx]])
        sorted_detected_distances.append(detected_distances[det_idx])
        sorted_detected_original_indices.append(det_idx)

    return (np.array(sorted_detected_coordinates), 
            np.array(sorted_detected_distances), 
            np.array(sorted_detected_indices))
    

 

def phot_source_ext(imagename, residual=None, sigma=1.0, iterations=2, dilation_size=None,
                    deblend_nthresh=5, deblend_cont=1e-6, maskthresh=0.0,
                    gain=1, filter_kernel=None, mask=None,
                    segmentation_map=False, clean_param=1.0, clean=True,
                    minarea=100, minarea_factor=1, npixels = None,
                    filter_type='matched', 
                    sort_by='distance', threshold_mode='sigma',
                    bw=64, bh=64, fw=3, fh=3, ell_size_factor=None,
                    apply_mask=False, sigma_mask=6,
                    show_bkg_map=False, show_detection=False,
                    SE_ref=None):
    """
    Simple source extraction algorithm (using SEP https://sep.readthedocs.io/en/v1.1.x/).


    """

    data_2D = load_fits_data(imagename)
    if len(data_2D.shape) == 4:
        data_2D = data_2D[0][0]
    # m, s = np.mean(data_2D), np.std(data_2D)
    if residual is not None:
        m, s = np.mean(data_2D), mad_std(residual)
    else:
        m, s = np.mean(data_2D), mad_std(data_2D)
    # bkg = 0.0
    if apply_mask:
        _, mask = mask_dilation(data_2D, sigma=sigma_mask, iterations=iterations,
                                rms=s,
                                PLOT=True,show_figure=True,
                                dilation_size=dilation_size)


    bkg = sep.Background(data_2D, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh)
    # print(bkg.globalback)
    # print(bkg.globalrms)

    bkg_image = bkg.back()
    bkg_rms = bkg.rms()
    
    data_sub = data_2D - bkg_image

    if show_bkg_map == True:
        # plt.figure()
        # # display bkg map.
        # plt.imshow(data_2D, interpolation='nearest', cmap='gray', vmin=3*s,
        #            vmax=0.2*np.max(data_2D), origin='lower')
        # plt.colorbar()
        # plt.close()
        # plt.show()
        # plt.figure()
        # plt.imshow(bkg_image)
        # plt.colorbar()
        # plt.show()
        # plt.close()
        
        fig = plt.figure(figsize=(18, 6))
        ax0 = fig.add_subplot(1, 3, 2)
        ax0 = eimshow(data_sub,ax=ax0,fig=fig,
                            plot_colorbar=True,
                            # cbar_orientation = 'horizontal',
                            show_axis='off',
                            vmin_factor=0.5,
                            cbar_n_points = 3,
                            add_contours=True)
        ax0.set_title(f'data - bkg')
        
        ax1 = fig.add_subplot(1, 3, 1)
        ax1 = eimshow(data_2D,ax=ax1,fig=fig,
                            plot_colorbar=True,
                            # cbar_orientation = 'horizontal',
                            show_axis='off',
                            vmin_factor=0.5,
                            cbar_n_points=3,
                            plot_title = f'data',
                            add_contours=True)
        ax2 = fig.add_subplot(1, 3, 3)
        ax2 = eimshow(bkg_image,ax=ax2,fig=fig,
                            plot_colorbar=True,
                            # cbar_orientation = 'horizontal',
                            show_axis='off',
                            # vmin_factor=0.5,
                            # vmax_factor=1.0,
                            cbar_n_points=3,
                            plot_title = f'bkg',
                            add_contours=False)
        plt.show()
        plt.clf()
        plt.close()

    
    if mask is not None:
        data_sub = data_sub * mask
    else:
        data_sub = data_sub
        
    if (threshold_mode == 'residual') and residual is not None:
        threshold = sigma * residual
    elif threshold_mode == 'sigma':
        threshold = sigma * s
            
    # else:
    #     mask = None
    # print(data_sub)
    if npixels is None:
        npixels = int(minarea * minarea_factor)
    # print(' INFO: Uinsg min number of pixels of :', npixels)
    cat, segm, seg_maps = make_catalog(image=data_sub,
                                       threshold=threshold,
                                       deblend=True, 
                                       contrast=deblend_cont,
                                       nlevels=deblend_nthresh,
                                       npixels=npixels,
                                       figsize=(20, 20),
                                       plot=show_detection, vmin=1.0 * s)
    # cat, segm, seg_maps = make_catalog(image=data_sub,
    #                                    threshold=sigma * s,
    #                                    deblend=True, 
    #                                    contrast=deblend_cont,
    #                                    nlevels=deblend_nthresh,
    #                                    npixels=npixels,
    #                                    figsize=(20, 20),
    #                                    plot=show_detection, vmin=1.0 * s)
    if sort_by == 'flux':
        indices = list(order_cat(cat, key='segment_flux', reverse=True))
    if sort_by == 'area':
        indices = list(order_cat(cat, key='area', reverse=True))
    if sort_by == 'distance':
        ref_centre = data_2D.shape[0] / 2, data_2D.shape[1] / 2
        if SE_ref is not None:
            coords, distances, indices = \
                sorted_detected_coordinates(SE_ref.objects['xc'],
                                            SE_ref.objects['yc'],
                                            cat.xcentroid, 
                                            cat.ycentroid, 
                                            ref_centre)
            print(indices)
        else:
            
            distances = distances_from_reference(cat.xcentroid, 
                                            cat.ycentroid,
                                            ref_centre)
            indices = np.argsort(distances)
        
    masks_deblended = []
    for k in range(len(indices)):
        # print(k)
        masks_deblended.append(seg_maps == seg_maps.labels[indices[k]])


    # len(objects)
    from matplotlib.patches import Ellipse
    from skimage.draw import ellipse

    # m, s = np.mean(data_sub), np.std(data_sub)
    if show_detection == True:
        fig, ax = plt.subplots(figsize=(10, 10))
        norm = simple_norm(data_sub, stretch='sqrt', asinh_a=0.02, min_cut=s,
                    max_cut=0.2*np.nanmax(data_sub))
        im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                       norm=norm, origin='lower')
        # im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
        #                vmin=s, vmax=0.2*np.nanmax(data_sub), origin='lower')

    masks_regions = []

    if ell_size_factor is None:
        if mask is not None:
            # ell_size_factor = np.sqrt(np.sum(mask) / (np.pi))/cat[0].equivalent_radius.value
            ell_size_factor = 0.05*np.sqrt(np.sum(mask) / (np.pi))
        else:
            ell_size_factor = 0.5
    
    y, x = np.indices(data_2D.shape[:2])
    
    # objects = cat
    for i in range(len(cat)):
        source = cat[i]
        seg_mask = (seg_maps.data == i + 1)
        e = Ellipse(xy=(source.centroid[0], source.centroid[1]),
                    width=1 * ell_size_factor * source.equivalent_radius.value,
                    height=1 * ell_size_factor * (
                                1 - source.ellipticity.value) * source.equivalent_radius.value,
                    # angle=source.orientation.value * 180. / np.pi
                    angle=source.orientation.value
                    )

        xc = source.centroid[0]
        yc = source.centroid[1]
        a = ell_size_factor * source.equivalent_radius.value
        b = ell_size_factor * (
                    1 - source.ellipticity.value) * source.equivalent_radius.value
        theta = source.orientation.value
        rx = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
        ry = (y - yc) * np.cos(theta) - (x - xc) * np.sin(theta)

        inside = ((rx / a) ** 2 + (ry / b) ** 2) <= 1
        mask_ell = np.zeros_like(data_2D)
        mask_ell[inside] = True
        if show_detection == True:
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)
        masks_regions.append(seg_mask)

    #         plt.savefig('components_SEP.pdf',dpi=300, bbox_inches='tight')
    # flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
    #                                      3.0, err=bkg.globalrms, gain=1.0)
    # for i in range(len(objects)):
    #     print("object {:d}: flux = {:f} +/- {:f}".format(i, flux[i], fluxerr[i]))
    # objects['b'] / objects['a'], np.rad2deg(objects['theta'])

    # sort regions from largest size to smallest size.
    mask_areas = []
    mask_fluxes = []
    for mask_comp in masks_regions:
        area_mask = np.sum(mask_comp)
        sum_mask = np.sum(mask_comp * data_2D)
        mask_areas.append(area_mask)
        mask_fluxes.append(sum_mask)
    mask_areas = np.asarray(mask_areas)
    mask_fluxes = np.asarray(mask_fluxes)
    if sort_by == 'area':
        sorted_indices_desc = np.argsort(mask_areas)[::-1]
        sorted_arr_desc = mask_areas[sorted_indices_desc]
    if sort_by == 'flux':
        sorted_indices_desc = np.argsort(mask_fluxes)[::-1]
        sorted_arr_desc = mask_fluxes[sorted_indices_desc]
    if sort_by == 'distance':
        ref_centre = data_2D.shape[0] / 2, data_2D.shape[1] / 2
        if SE_ref is not None:
            coords, distances, sorted_indices_desc = \
                sorted_detected_coordinates(SE_ref.objects['xc'],
                                            SE_ref.objects['yc'],
                                            cat.xcentroid, 
                                            cat.ycentroid, 
                                            ref_centre)
            # sorted_indices_desc = np.argsort(distances)
            # sorted_arr_desc = distances[sorted_indices_desc]
                
        else:
            distances = distances_from_reference(cat.xcentroid, 
                                                cat.ycentroid,
                                                ref_centre)
            sorted_indices_desc = np.argsort(distances)
            # sorted_arr_desc = distances[sorted_indices_desc]

    objects_sorted = {}
    objects_sorted['xc'] = np.asarray([1] * len(cat))
    objects_sorted['yc'] = np.asarray([1] * len(cat))
    for i in range(len(cat)):
        source = cat[sorted_indices_desc[i]]
        objects_sorted['xc'][i] = source.centroid[0]
        objects_sorted['yc'][i] = source.centroid[1]

    if show_detection == True:
        for i in range(len(cat)):
            source = cat[sorted_indices_desc[i]]
            xc = source.centroid[0]
            yc = source.centroid[1]
            label = str('ID' + str(i + 1))
            
            label_x = xc + 3 * ell_size_factor
            label_y = yc + 20 * ell_size_factor
            
            line_end_y = label_y - 5 
            
            text = Text(label_x, label_y, label, ha='center', va='center', color='red')
            ax.add_artist(text)
            
            ax.plot([xc, label_x], [yc, line_end_y], color='red', alpha=0.4)

        plt.axis('off')
        plt.savefig(imagename + '_SEP_phot.jpg', dpi=300, bbox_inches='tight')
        plt.show()



    if segmentation_map == True:
        return (masks_deblended, sorted_indices_desc, bkg,
                seg_maps, objects_sorted)
    else:
        return (masks_deblended, sorted_indices_desc, bkg,
                objects_sorted)

def prepare_fit(ref_image, ref_res, z, ids_to_add=[1],
                bw=51, bh=51, fw=15, fh=15, sigma=15, ell_size_factor=2.0,
                deblend_cont=1e-7, deblend_nthresh=15,
                minarea=None,minarea_factor=1.0,npixels=None,
                sigma_mask=6,mask=None,dilation_size=None,
                show_detection=True,use_extraction_positions=False,
                clean_param=0.9,clean=True,sort_by='distance',
                apply_mask=False, mask_grow_iterations=3,
                obs_type = 'radio',algorithm='SEP',force_circular=True,
                show_petro_plots=False):
    """
    Prepare the imaging data to be modeled.

    This function runs a source extraction, computes basic petrosian properties
    from the data for each detected source as well as shape morphology
    (e.g. position angle, axis ration, effective intensity and radii).
    """
    crop_image = ref_image
    crop_residual = ref_res
    data_2D = load_fits_data(crop_image)
    if crop_residual is not None:
        residual_2D = load_fits_data(crop_residual)
    else:
        residual_2D = None
    if minarea is None:
        try:
            minarea = int(beam_area2(crop_image))
        except:
            minarea = data_2D.shape[0]/30
    pix_to_pc = pixsize_to_pc(z=z,
                              cell_size=get_cell_size(crop_image))
    #     eimshow(crop_image, vmin_factor=5)
    try:
        std_res = mad_std(load_fits_data(crop_residual))
    except:
        std_res = mad_std(data_2D)

    if mask is not None:
        mask_detection = mask
        apply_mask = False

    if apply_mask:
        _, mask = mask_dilation(crop_image, sigma=sigma_mask, dilation_size=None,
                                iterations=2, rms=std_res)


    mask_detection = mask.copy()

    # if apply_mask == True:
    #     mask_detection = mask
    # else:
    #     mask_detection = None #np.ones(data_2D.shape)
    # # plt.figure()

    # _, mask = mask_dilation(crop_image, sigma=6, dilation_size=None,
    #                         iterations=2)
    if algorithm == 'SEP':
        masks, indices, bkg, seg_maps, objects = \
            sep_source_ext(crop_image, 
                           residualname=crop_residual,
                           bw=bw,bh=bh,fw=fw, fh=fh,
                           minarea=minarea,
                           minarea_factor=minarea_factor,
                           segmentation_map=True,
                           filter_type='matched',
                           mask = mask_detection,
                           deblend_nthresh=deblend_nthresh,
                           deblend_cont=deblend_cont,
                           clean_param=clean_param,
                           clean=clean,
                           sort_by=sort_by,
                           npixels = npixels,
                           dilation_size=dilation_size,
                           iterations = mask_grow_iterations,
                           sigma=sigma,sigma_mask=sigma_mask,
                           ell_size_factor=ell_size_factor,
                           apply_mask=apply_mask,
                           show_detection=show_detection)
    if algorithm == 'PF':
        masks, indices, bkg, seg_maps, objects = \
            phot_source_ext(crop_image, 
                           residual=residual_2D,
                           bw=bw,
                           bh=bh,
                           fw=fw, fh=fh,
                           minarea=minarea,
                           minarea_factor=minarea_factor,
                           segmentation_map=True,
                           filter_type='matched', mask=mask,
                           deblend_nthresh=deblend_nthresh,
                           deblend_cont=deblend_cont,
                           clean_param=clean_param,
                           clean=clean,
                           sort_by=sort_by,
                           sigma=sigma,sigma_mask=sigma_mask,
                           iterations = mask_grow_iterations,
                           ell_size_factor=ell_size_factor,
                           apply_mask=apply_mask,
                           show_detection=show_detection)



    sigma_level = 3
    vmin = 3
    # i = 0 #to be used in indices[0], e.g. first component
    sources_photometries = {}  # init dict to store values.
    # if use_extraction_positions == True:
    #     for i in range(len(indices)):
    #         # ii = str(i+1)
    #         positions = np.array([objects['xc'][i], objects['yc'][i]])
    #         mask_component = masks[indices[i]]
    #         data_component = data_2D * mask_component
    #         sources_photometries = compute_petro_source(data_component,
    #                                                     mask_component=mask_component,
    #                                                     sigma_level=1,positions=positions,
    #                                                     i=i, plot=show_petro_plots,
    #                                                     source_props=sources_photometries)

    # else:
    
    
    sources_photometries = compute_petro_source(data_2D = data_2D,
                                                imagename = crop_image,
                                                sigma_level=6,
                                                nlevels=1, contrast=1,
                                                deblend=False, npixels=None,
                                                i='g', mask_component = mask_detection,
                                                plot=show_petro_plots,
                                                source_props=sources_photometries)
    
    for i in tqdm(range(len(indices))):
        # ii = str(i+1)
        mask_component = masks[i]
        data_component = data_2D * mask_component
        sources_photometries = compute_petro_source(data_2D = data_component,
                                                    imagename = crop_image,
                                                    mask_component=mask_component,
                                                    obs_type = obs_type,
                                                    sigma_level=1,nlevels=1, contrast=1,
                                                    deblend=False, npixels=None,
                                                    i=i, plot=show_petro_plots,
                                                    source_props=sources_photometries)


    if obs_type == 'radio':
        """
        PSF image is contained within the header of the original image. 
        A new psf file will be created (as `psf_name`). 
        """
        # omaj, omin, _, _, _ = beam_shape(crop_image)
        # dilation_size = int(
        #     np.sqrt(omaj * omin) / (2 * get_cell_size(crop_image)))
        # psf_image_size = dilation_size*6
        # psf_image_size = (2 * psf_image_size) // 2 +1
        psf_image_size = int(data_2D.shape[0])
        # print('++==>> PSF IMAGE SIZE is', psf_image_size)
        # creates a psf from the beam shape.
        psf_name = tcreate_beam_psf(crop_image, size=(
            psf_image_size, psf_image_size))  # ,app_name='_'+str(psf_image_size)+'x'+str(psf_image_size)+'')
    if obs_type == 'other':
        """
        Provide a psf file.
        """
        psf_name = None

    n_components = len(indices)
    n_IDs = len(indices)
    sources_photometries['ncomps'] = n_components
    sources_photometries['nIDs'] = n_IDs

    
    print("# of structures (IDs) to be fitted =", n_IDs)
    # sources_photometies_new = sources_photometies
    # n_components_new = n_components
    if ids_to_add is not None:
        for id_to_add in ids_to_add:
            sources_photometries = add_extra_component(sources_photometries,
                                                       copy_from_id=id_to_add)
    
    if force_circular:
        print(f"<> Forcing components to be circular.")
        # for i in range(len(indices)):
        for i in range(sources_photometries['ncomps']):
            if i+1 <= n_IDs:
                sources_photometries[f'c{i+1}_q'] = 0.99
            print(f"<> q_{i+1} {sources_photometries[f'c{i+1}_q']}")
    else:
        print(f"<> Leaving components to be elliptical.")
        for i in range(sources_photometries['ncomps']):
            if i+1 > n_IDs:
                sources_photometries[f'c{i+1}_q'] = 0.6
                # sources_photometries[f'c{i+1}_R50'] = sources_photometries[f'cg_Rp']
                # sources_photometries[f'c{i+1}_Rp'] = sources_photometries[f'cg_Rp']
            print('<> q = ',sources_photometries[f'c{i+1}_q'])

    # update variable `n_components`.
    n_components = sources_photometries['ncomps']
    print("# of model components (COMPS) to be fitted =", n_components)
    return (sources_photometries, n_components, n_IDs, masks, indices, objects,
            psf_name, mask, bkg)

def do_fit2D(imagename, params_values_init_IMFIT=None, ncomponents=None,
             init_constraints=None, data_2D_=None, residualdata_2D_=None,
             residualname=None,which_residual='shuffled',observation_type = 'radio',
             init_params=0.25, final_params=4.0, constrained=True,
             fix_n=True, fix_value_n=False, 
             fix_max_value_Rn=False,
             dr_fix=3,
             fix_x0_y0=False, psf_name=None, convolution_mode='GPU',
             convolve_cutout=False, cut_size=512, self_bkg=False, rms_map=None,
             fix_geometry=True, contrain_nelder=False, workers=6,mask_region = None,
             special_name='', method1='least_squares', method2='least_squares',
             reduce_fcn='neglogcauchy',loss="cauchy",tr_solver="exact",x_scale = 'jac',
             ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=2,max_nfev=200000,
             regularize  = True, f_scale = 1.0,
             maxiter = 30000, maxfev = 30000, xatol = 1e-12,
             fatol = 1e-12, return_all = True, disp = True,
             de_options=None,parameters_mini_init=None,
             save_name_append='',logger=None):
    """
    Perform a Robust and Fast Multi-Sersic Decomposition with GPU acceleration.
    tr_solver:

    Parameters
    ----------
    imagename: str
        Name of the image to be fitted.
    params_values_init_IMFIT: list
        Initial parameters values for the model.
    ncomponents: int
        Number of components to be fitted.
    init_constraints: dict
        Initial constraints for the model.
    data_2D_: 2D array
        Image to be fitted.
    residualdata_2D_: 2D array
        Residual image to be fitted.
    residualname: str
        Name of the residual image to be fitted.
    which_residual: str
        Which residual to be used for the fitting.
        Options: 'shuffled' or 'natural'.
    init_params: float
        Initial parameters for the model.
    final_params: float
        Final parameters for the model.
    constrained: bool
        If True, use initial constraints for the model.
    fix_n: bool
        If True, fix the Sersic index of the model.
    fix_value_n: float
        If True, fix the Sersic index of the model to this value.
    dr_fix: float
        If True, fix the centre position of the model.
    fix_x0_y0: bool
        If True, fix the centre position of the model.
    psf_name: str
        Name of the PSF image to be used for the convolution.
    convolution_mode: str
        If 'GPU', use GPU acceleration for the convolution.
    convolve_cutout: bool
        If True, convolve the image cutout with the PSF.
    cut_size: int
        Size of the cutout image.
    self_bkg: bool
        If True, use the image background as the residual background.
    rms_map: 2D array
        RMS map to be used for the fitting.
    fix_geometry: bool
        If True, fix the geometry of the model.
    contrain_nelder: bool
        If True, constrain the Nelder-Mead optimised parameters.
    workers: int
        Number of workers to be used for the fitting.
    mask_region: 2D array
        Mask to be used for the fitting.
    special_name: str
        Special name to be used for the output files.
    method1: str
        Method to be used for the fitting.
    method2: str
        Method to be used for the fitting.
    reduce_fcn: str

    loss: str

    tr_solver: str

    x_scale: str

    ftol: float

    xtol: float

    gtol: float

    verbose: int

    max_nfev: int

    regularize: bool
        If True, regularize the model.
    f_scale: float

    maxiter: int

    maxfev: int

    xatol: float

    fatol: float

    return_all: bool

    disp: bool

    de_options: dict

    save_name_append: str

    logger: logger


    returns
    -------
    result: dict
        Dictionary containing the results of the fitting.


    """
    try:
        from jax import jit
    except:
        convolution_mode = 'CPU'
        
    if de_options is None:
        de_options = {'disp': True, 'workers': 6,
                      'max_nfev': 20000, 'vectorized': True,
                      # 'strategy': 'randtobest1bin',
                      'mutation': (0.5, 1.5),
                      'recombination': [0.2, 0.9],
                      'init': 'random', 'tol': 0.00001,
                      'updating': 'deferred',
                      'popsize': 600}

    startTime = time.time()
    try:
        logger.info(f"Fitting image: {imagename}")
    except:
        pass
    FlatSky_level = None
    if data_2D_ is None:
        data_2D = pf.getdata(imagename)
    else:
        data_2D = data_2D_

    if mask_region is not None:
        """
        
        """
        logger.debug(f" ==> Using provided mask region to constrain fit. ")
        logger.warning(f" !!==> Fitting with a mask is experimental! ")
        # data_2D = data_2D * mask_region
        if convolution_mode == 'GPU':
            mask_for_fit = jnp.array(mask_region)
        else:
            mask_for_fit = mask_region
    else:
        mask_for_fit = None

    if convolution_mode == 'GPU':
        data_2D_gpu = jnp.array(data_2D)

    if psf_name is not None:
        PSF_CONV = True
        try:
            PSF_DATA_raw = pf.getdata(psf_name)
            if len(PSF_DATA_raw.shape) == 4:
                PSF_DATA_raw = PSF_DATA_raw[0][0]
        except:
            PSF_DATA_raw = load_fits_data(psf_name)

        if convolution_mode == 'GPU':
            if logger is not None:
                logger.debug(f"---------------------------------------")
                logger.debug(f" <<< PERFORMING CONVOLUTION WITH JAX >>> ")
                logger.debug(f"---------------------------------------")
            PSF_DATA = jnp.array(PSF_DATA_raw)

        if convolution_mode == 'CPU':
            PSF_DATA = PSF_DATA_raw
        # PSF_DATA = pf.getdata(
        #     imagename.replace('-image.cutout.fits', '-beampsf.cutout.fits'))
    else:
        PSF_CONV = False
        PSF_DATA = None

    if residualname is not None and which_residual != 'user':
        """
        This is important for radio image fitting.

        It uses the shuffled version of the residual cleaned image
        originated from the interferometric deconvolution.

        This ensures that the best model created here will be on top
        of that rms noise so that flux conservation is maximized.

        However, this residual is not added as model + shuffled_residual
        only, but instead by a multiplication factor,
        e.g. model + const* shuffled_residual, and const will be minimized
        as well during the fitting (here, called `s_a`).
        """
        if residualdata_2D_ is not None:
            residual_2D = residualdata_2D_
        else:
            residual_2D = pf.getdata(residualname)

        if which_residual == 'shuffled':
            if logger is not None:
                logger.debug(f" ==> Using clean shuffled background for optmization... ")
            residual_2D_to_use = shuffle_2D(residual_2D)
        if which_residual == 'natural':
            if logger is not None:
                logger.debug(f" ==> Using clean background for optmization... ")
            """            
            if psf_name is not None:
                if logger is not None:
                    logger.debug(f" ====> Deconvolving residual map... ")
                residual_2D_to_use, _ = deconvolve_fft(residual_2D,
                                                            PSF_DATA_raw/PSF_DATA_raw.sum())
            else:
                residual_2D_to_use = residual_2D
            """
            residual_2D_to_use = residual_2D

        FlatSky_level = mad_std(residual_2D_to_use)
        #         background = residual_2D #residual_2D_to_use
        if convolution_mode == 'GPU':
            background = jnp.array(residual_2D_to_use)
        else:
            background = residual_2D_to_use

    else:
        if which_residual == 'user':
            if rms_map is None:
                print('--==>> A rms map/background mode was selected (user)')
                print('       but no rms/background map was provided.')
                print('       Please, provide a rms/background map.')
                print('||==>> Stopping code now.')
                raise ValueError("rms_map should not be None")
            else:
                logger.debug(f" ==> Using provided RMS map. ")
                background_map=rms_map
                background = background_map.copy()
        else:
            if self_bkg == True:
                if logger is not None:
                    logger.warning(f" ==> No residual/background provided. Using image bkg map... ")
                background_map = sep_background(imagename)
                background = shuffle_2D(background_map.back())
            else:
                if logger is not None:
                    logger.warning(f" ==> Using only flat sky for rms bkg.")
                FlatSky_level = mad_std(data_2D)
                background = FlatSky_level

    if residualdata_2D_ is not None:
        residual_2D = residualdata_2D_
    else:
        try:
            residual_2D = pf.getdata(residualname)
        except:
            residual_2D = background

    size = data_2D.shape
    if convolution_mode == 'GPU':
        x,y = jnp.meshgrid(jnp.arange((size[1])), jnp.arange((size[0])))
        xy = jnp.stack([x, y], axis=0)
    else:
        xy = np.meshgrid(np.arange((size[1])), np.arange((size[0])))

    if convolve_cutout is True:
        """
        WARNING: DO NOT USE FOR NOW!
        
        Instead of convolving the entire image,
        convolve only a box.
        Can be 10x faster.

        Issue:
        It causes the flat sky level to be much higher than the real value.
        
        Need further investigation and proper implementation.
        """
        x0c, y0c = int(size[0] / 2), int(size[1] / 2)

    #     FlatSky_level = background#mad_std(data_2D)


    # if convolution_mode == 'GPU':

    # FlatSky_level = mad_std(data_2D)
    if convolution_mode == 'GPU':
        residual_2D = jnp.array(residual_2D)

    nfunctions = ncomponents

    def min_residual_2D(params):
        dict_model = {}
        model = 0
        for i in range(1, nfunctions + 1):
            model = model + sersic2D(xy, params['f' + str(i) + '_x0'],
                                     params['f' + str(i) + '_y0'],
                                     params['f' + str(i) + '_PA'],
                                     params['f' + str(i) + '_ell'],
                                     params['f' + str(i) + '_n'],
                                     params['f' + str(i) + '_In'],
                                     params['f' + str(i) + '_Rn'],
                                     params['f' + str(i) + '_cg'], )
        # print(model.shape)
        # model = model + FlatSky_cpu(FlatSky_level, params['s_a'])*background
        # model = model + FlatSky_cpu(background, params['s_a'])
        MODEL_2D_conv = scipy.signal.fftconvolve(model, PSF_DATA, 'same') + \
                        FlatSky_cpu(background, params['s_a'])
        residual = data_2D - MODEL_2D_conv
        return np.ravel(residual)

    # @partial(jit, static_argnums=2)
    # def build_model(xy,_params,nfunctions):
    #     params = func(_params[:-1])
    #     model = 0
    #     for i in range(0, nfunctions):
    #         params_i = params[i]
    #         print(params_i[5])
    #         model = model + sersic2D_GPU(xy, params_i[0],
    #                                      params_i[1],
    #                                      params_i[2],
    #                                      params_i[3],
    #                                      params_i[4],
    #                                      params_i[5],
    #                                      params_i[6],
    #                                      params_i[7])
    #     return model
    """
    def extract_params(params):
        param_array = jnp.array(
            [params[f'f{i}_{name}'].value for i in range(1, nfunctions + 1) for name in
             ['x0', 'y0', 'PA', 'ell', 'n', 'In', 'Rn', 'cg']])
        param_matrix = np.split(param_array, nfunctions)
        return param_matrix
    # @partial(jit, static_argnums=2)
    # @jit
    # def build_model(xy,param_matrix):
    #     model = 0
    #     for model_params in param_matrix:
    #         model = model + sersic2D_GPU(xy, model_params[0],
    #                                      model_params[1],
    #                                      model_params[2],
    #                                      model_params[3],
    #                                      model_params[4],
    #                                      model_params[5],
    #                                      model_params[6],
    #                                      model_params[7])
    #     return model
    # batched_sersic2D_GPU = vmap(sersic2D_GPU, in_axes=(None, 0))
    # @jit
    # def build_model(xy,param_matrix):
    #     # Use the batched version of sersic2D_GPU to compute all models in parallel
    #     models = batched_sersic2D_GPU(xy, param_matrix)
    #     # Sum the models to get the final model
    #     total_model = models.sum(axis=0)
    #     return total_model


    # @jit
    # def sersic2D_GPU_vectorized(xy, params):
    #     # Assuming sersic2D_GPU is compatible with JAX and can work with batched inputs
    #     # Unpack params directly within the function if necessary
    #     return sersic2D_GPU_new(xy, params)
    #
    # batched_sersic2D_GPU = vmap(sersic2D_GPU_vectorized, in_axes=(None, 0))
    #
    # @jit
    # def build_model(xy, param_matrix):
    #     # Compute all models in parallel
    #     models = batched_sersic2D_GPU(xy, param_matrix)
    #     # Sum the models to get the final model
    #     total_model = models.sum(axis=0)
    #     return total_model
    def min_min_residual_2D_GPU(params):
        model = 0
        for i in range(1, nfunctions + 1):
            model = model + sersic2D_GPU(xy,
                                         params['f' + str(i) + '_x0'].value,
                                         params['f' + str(i) + '_y0'].value,
                                         params['f' + str(i) + '_PA'].value,
                                         params['f' + str(i) + '_ell'].value,
                                         params['f' + str(i) + '_n'].value,
                                         params['f' + str(i) + '_In'].value,
                                         params['f' + str(i) + '_Rn'].value,
                                         params['f' + str(i) + '_cg'].value)
        # # param_matrix = extract_params(params)
        # param_matrix = func(jnp.array(list(params.valuesdict().values()))[:-1])
        # model = build_model(xy,param_matrix)
        # print(params.values)
        # print(params.valuesdict().values())
        # _params = convert_params_to_numpy(np.asarray(list(params.valuesdict().values())))
        # print(jnp.array(list(params.valuesdict().values())))
        # model = build_model(xy,
        #                     jnp.array(list(params.valuesdict().values())),
        #                     nfunctions)
        # print(model.shape)
        # MODEL_2D_conv = convolve_on_gpu(model, PSF_DATA)
        # MODEL_2D_conv = jax_convolve(model, PSF_DATA)
        # model = model + FlatSky(background, params['s_a'].value)
        # MODEL_2D_conv = _fftconvolve_jax(model, PSF_DATA) + \
        #                 FlatSky(background,params['s_a'].value)
        MODEL_2D_conv = _fftconvolve_jax(model+
                                         FlatSky(background,params['s_a'].value),
                                         PSF_DATA)
        residual = data_2D_gpu - MODEL_2D_conv

        # return np.asarray(residual).copy()
        # return np.asarray(residual).copy().flatten()
        # return np.asarray(residual+0.01*abs(jnp.nanmin(residual))).copy()
        return np.asarray(residual).copy()
    """
    def convert_params_to_numpy(_params):
        return list(_params)

    try:
        # @partial(jit, static_argnums=1)
        @jit
        def func(x):
            return jnp.split(x, nfunctions)

        @jit
        def build_model(xy,param_matrix):
            model = 0
            for model_params in param_matrix:
                model = model + sersic2D_GPU(xy, model_params[0],
                                            model_params[1],
                                            model_params[2],
                                            model_params[3],
                                            model_params[4],
                                            model_params[5],
                                            model_params[6],
                                            model_params[7])
            return model
    
    except:
        # @partial(jit, static_argnums=1)
        
        def func(x):
            return jnp.split(x, nfunctions)

        
        def build_model(xy,param_matrix):
            model = 0
            for model_params in param_matrix:
                model = model + sersic2D_GPU(xy, model_params[0],
                                            model_params[1],
                                            model_params[2],
                                            model_params[3],
                                            model_params[4],
                                            model_params[5],
                                            model_params[6],
                                            model_params[7])
            return model
    
    def min_residual_2D_GPU(params):
        model = 0
        for i in range(1, nfunctions + 1):
            model = model + sersic2D_GPU(xy,
                                         params['f' + str(i) + '_x0'].value,
                                         params['f' + str(i) + '_y0'].value,
                                         params['f' + str(i) + '_PA'].value,
                                         params['f' + str(i) + '_ell'].value,
                                         params['f' + str(i) + '_n'].value,
                                         params['f' + str(i) + '_In'].value,
                                         params['f' + str(i) + '_Rn'].value,
                                         params['f' + str(i) + '_cg'].value)

        # # param_matrix = extract_params(params)
        # param_matrix = func(jnp.array(list(params.valuesdict().values()))[:-1])
        # model = build_model(xy,param_matrix)

        MODEL_2D_conv = _fftconvolve_jax(model+
                                         FlatSky(background,params['s_a'].value),
                                         PSF_DATA)
        # residual = ((data_2D_gpu[mask_for_fit] - MODEL_2D_conv[mask_for_fit])/
        #             (1000*(abs(residual_2D[mask_for_fit])+1.0e-6)))
        residual = (data_2D_gpu[mask_for_fit] - MODEL_2D_conv[mask_for_fit])
        return np.asarray(residual).copy()
        # weights = 1/((background[mask_for_fit])/data_2D_gpu[mask_for_fit])
        # weightned_residual = (data_2D_gpu[mask_for_fit] - MODEL_2D_conv[mask_for_fit]) * jnp.sqrt(weights)
        # return np.asarray(weightned_residual).copy()
    
    if convolution_mode == 'GPU':
        @jit
        def convolve_on_gpu(image, psf):
            """
            This was before jax.scipy implementing fftconvolve.
            It provides the same result, at the same speed.

            This function also accepts PSFs with a different shape of the image.

            """
            # Calculate the new padded shape
            padded_shape = (image.shape[0] + psf.shape[0] - 1,
                            image.shape[1] + psf.shape[1] - 1)

            # Pad both image and psf to the new shape
            pad_shape = [(0, ts - s) for s, ts in zip(image.shape, padded_shape)]
            image_padded = jnp.pad(image, pad_shape, mode='constant')
            pad_shape = [(0, ts - s) for s, ts in zip(psf.shape, padded_shape)]
            psf_padded = jnp.pad(psf, pad_shape, mode='constant')
            # psf_padded = pad_for_convolution(psf, padded_shape)
            image_fft = jnp.fft.fft2(image_padded)
            psf_fft = jnp.fft.fft2(psf_padded)

            conv_fft = image_fft * psf_fft

            # Get the real part of the inverse FFT and crop to the original image size
            result_full = jnp.real(jnp.fft.ifft2(conv_fft))
            return result_full[psf.shape[0] // 2:image.shape[0] + psf.shape[0] // 2,
                   psf.shape[1] // 2:image.shape[1] + psf.shape[1] // 2]

        jax_convolve = jit(convolve_on_gpu)


    smodel2D, params = construct_model_parameters(
        params_values_init_IMFIT=params_values_init_IMFIT, n_components=nfunctions,
        init_constraints=init_constraints,observation_type=observation_type,
        fix_n=fix_n, fix_value_n=fix_value_n,
        fix_max_value_Rn = fix_max_value_Rn,
        fix_x0_y0=fix_x0_y0, dr_fix=dr_fix, fix_geometry=fix_geometry,
        init_params=init_params, final_params=final_params,
        constrained=constrained)

    if convolution_mode == 'CPU':
        mini = lmfit.Minimizer(min_residual_2D, params, max_nfev=200000,
                               nan_policy='omit', reduce_fcn=reduce_fcn)
    if convolution_mode == 'GPU':
        mini = lmfit.Minimizer(min_residual_2D_GPU, params, max_nfev=200000,
                               nan_policy='omit', reduce_fcn=reduce_fcn)

    # initial minimization.

    print(' >> Using', method1, ' solver for first optimisation run... ')
    # take parameters from previous run, and re-optimize them.
    #     me``rthod2 = 'ampgo'#'least_squares'
    #     method2 = 'least_squares'
    result_extra = None
    if method1 == 'nelder':
        # very robust, but takes time....
        #         print(' >> Using', method1,' solver for first optimisation run... ')
        result_1 = mini.minimize(method='nelder',
                                 #                                  xatol = 1e-12, fatol = 1e-12, disp=True,
                                 #                                  adaptive = True,max_nfev = 30000,
                                 options={'maxiter': maxiter, 'maxfev': maxfev,
                                          'xatol': xatol, 'fatol': fatol,
                                          'return_all': return_all,
                                          'disp': disp}
                                 )


    if method1 == 'least_squares':
        # faster, but usually not good for first run.
        # if results_previous_run is not None:
        print(' >> Using',tr_solver,'for tr solver, with regularize set to',regularize,
              ' Loss is',loss,'.')
        
        if parameters_mini_init is not None:
            print(f'  ++==>> Using initial mini parameters from a previous run.')
            # try:
            result_1 = mini.minimize(method='least_squares',
                                     params=parameters_mini_init,
                                    max_nfev=max_nfev, x_scale=x_scale, f_scale=f_scale,
                                    tr_solver=tr_solver,
                                    tr_options={'regularize': regularize,
                                                },
                                    ftol=ftol, xtol=xtol, gtol=gtol, verbose=verbose,
                                    loss=loss)  # ,f_scale=0.5, max_nfev=5000, verbose=2) 
        else:
            result_1 = mini.minimize(method='least_squares',
                                    max_nfev=max_nfev, x_scale=x_scale, f_scale=f_scale,
                                    tr_solver=tr_solver,
                                    tr_options={'regularize': regularize,
    #                                              'min_delta':1e-14, 'eta':0.05,
    #                                              'xtol':1e-14, 'gtol':1e-14,
    #                                              'ftol':1e-14
                                                },
                                    ftol=ftol, xtol=xtol, gtol=gtol, verbose=verbose,
                                    loss=loss)  # ,f_scale=0.5, max_nfev=5000, verbose=2)

    if method1 == 'differential_evolution':
        # de is giving some issues, I do not know why.
        result_1 = mini.minimize(method='differential_evolution',
                                 options={'disp': True, 'workers': workers,
                                          'max_nfev': max_nfev, 'vectorized': True,
                                          'strategy': 'randtobest1bin',
                                          'mutation': (0.5, 1.5),
                                          'recombination': [0.2, 0.9],
                                          'init': 'random', 'tol': 0.00001,
                                          'updating': 'deferred',
                                          'popsize': 600})
        # result_1 = mini.minimize(method='differential_evolution', popsize=600,
        #                          disp=True,  # init = 'random',
        #                          # mutation=(0.5, 1.5), recombination=[0.2, 0.9],
        #                          max_nfev=20000,
        #                          workers=1, updating='deferred', vectorized=True)

    print(' >> Using', method2, ' solver for second optimisation run... ')

    second_run_params = result_1.params
    if (contrain_nelder == True) and (method2 == 'nelder'):
        """
        It seems that least_squares is ignoring the best-parameters provided by
        Nelder-mead, which means that it is lookig the parameter space far away
        from the optimised Nelder-Mead ones.

        So, with this condition, we force a much smaller searching region, but
        it assumes that Nelder opt was good (which is not always true).

        YOU MUST CHECK YOUR RESULTS!!!!

        """
        print('Constraining Nelder-Mead Parameters for method', method2)
        params_constrained = constrain_nelder_mead_params(result_1.params,
                                                          max_factor=1.03,
                                                          min_factor=0.97)
        # UPDATE THE SECOND RUN PARAMETERS TO BE THE CONSTRAINED ONES.
        second_run_params = params_constrained

    if method2 == 'nelder':
        result = mini.minimize(method='nelder', params=second_run_params,
                               options={'maxiter': maxiter, 'maxfev': maxfev,
                                        'xatol': xatol, 'fatol': fatol,
                                        'disp': disp})

    if method2 == 'ampgo':
        # ampgo is not workin well/ takes so long ???
        result = mini.minimize(method='ampgo', params=second_run_params,
                               maxfunevals=10000, totaliter=30, disp=True,
                               maxiter=5, glbtol=1e-8)

    if method2 == 'least_squares':
        # faster, usually converges and provide errors.
        # Very robust if used in second opt from first opt parameters.
        result = mini.minimize(method='least_squares', 
                               params=second_run_params,
                               max_nfev=max_nfev,
                               tr_solver=tr_solver,
                               tr_options={'regularize': regularize,
#                                            'min_delta': 1e-14, 'eta': 0.05,
#                                            'xtol': 1e-14, 'gtol': 1e-14,
#                                            'ftol': 1e-14
                                          },
                               x_scale=x_scale,  f_scale=f_scale,
                               ftol=ftol, xtol=xtol, gtol=gtol, verbose=verbose,
                               loss=loss)  # ,f_scale=0.5, max_nfev=5000, verbose=2)

    if method2 == 'differential_evolution':
        # result = mini.minimize(method='differential_evolution',
        #                        params=second_run_params,
        #                        options={'maxiter': 30000, 'workers': -1,
        #                                 'tol': 0.001, 'vectorized': True,
        #                                 'strategy': 'randtobest1bin',
        #                                 'updating': 'deferred', 'disp': True,
        #                                 'seed': 1}
        #                        )
        result = mini.minimize(method='differential_evolution',
                               params=second_run_params,
                               options=de_options
                               )

    params = result.params

    model_temp = Model(sersic2D)
    xy = np.meshgrid(np.arange((size[1])), np.arange((size[0])))
    model = 0
    model_dict = {}
    image_results_conv = []
    image_results_deconv = []
    total_image_results_deconv = []
    total_image_results_conv = []
    bkg_images = []
    flat_sky_total = FlatSky_cpu(background, params['s_a'].value)
    bkg_comp_i = flat_sky_total.copy()
    for i in range(1, ncomponents + 1):
        model_temp = sersic2D_GPU(xy, params['f' + str(i) + '_x0'].value,
                              params['f' + str(i) + '_y0'].value,
                              params['f' + str(i) + '_PA'].value,
                              params['f' + str(i) + '_ell'].value,
                              params['f' + str(i) + '_n'].value,
                              params['f' + str(i) + '_In'].value,
                              params['f' + str(i) + '_Rn'].value,
                              params['f' + str(i) + '_cg'].value)

        model = model + model_temp
        #to each individual component, add the bkg map.
        model_dict['model_c' + str(i)] = np.asarray(model_temp+bkg_comp_i).copy()
        if PSF_CONV == True:
            if convolution_mode == 'GPU':
                # model_dict['model_c' + str(i) + '_conv'] = np.asarray(jax_convolve(model_temp, PSF_DATA)).copy()
                # model_dict['model_c' + str(i) + '_conv'] = (
                #     np.asarray(_fftconvolve_jax(model_temp,PSF_DATA).copy()+bkg_comp_i))
                # to each individual component, add the bkg map.
                model_dict['model_c' + str(i) + '_conv'] = (
                    np.asarray(_fftconvolve_jax(model_temp+bkg_comp_i,PSF_DATA).copy()))
            if convolution_mode == 'CPU':
                model_dict['model_c' + str(i) + '_conv'] = (
                        scipy.signal.fftconvolve(model_temp+bkg_comp_i, PSF_DATA_raw,'same'))

        else:
            model_dict['model_c' + str(i) + '_conv'] = model_temp+bkg_comp_i

        pf.writeto(imagename.replace('.fits', '') +
                   "_" + "model_component_" + str(i) +
                   special_name + save_name_append + '.fits',
                   model_dict['model_c' + str(i) + '_conv'], overwrite=True)
        copy_header(imagename, imagename.replace('.fits', '') +
                    "_" + "model_component_" + str(i) +
                    special_name + save_name_append + '.fits',
                    imagename.replace('.fits', '') +
                    "_" + "model_component_" + str(
                        i) + special_name + save_name_append + '.fits')
        pf.writeto(imagename.replace('.fits', '') +
                   "_" + "dec_model_component_" + str(i) +
                   special_name + save_name_append + '.fits',
                   model_dict['model_c' + str(i)], overwrite=True)
        copy_header(imagename, imagename.replace('.fits', '') +
                    "_" + "dec_model_component_" + str(i) +
                    special_name + save_name_append + '.fits',
                    imagename.replace('.fits', '') +
                    "_" + "dec_model_component_" + str(i) +
                    special_name + save_name_append + '.fits')

        image_results_conv.append(imagename.replace('.fits', '') +
                                  "_" + "model_component_" + str(i) +
                                  special_name + save_name_append + '.fits')
        image_results_deconv.append(imagename.replace('.fits', '') +
                                    "_" + "dec_model_component_" + str(i) +
                                    special_name + save_name_append + '.fits')

    #     model = model
    model_dict['model_total_dec'] = np.asarray(model+flat_sky_total) # +FlatSky_cpu(background,
    # params['s_a'].value)

    if PSF_CONV == True:
        # model_dict['model_total_conv'] = scipy.signal.fftconvolve(model,
        #                                                           PSF_DATA_raw,
        #                                                           'same')  # + FlatSky(FlatSky_level, params['s_a'])
        if convolution_mode == 'GPU':
            # model_dict['model_total_conv'] = np.asarray(jax_convolve(model,
            #                                                          PSF_DATA)).copy()
            # model_conv = _fftconvolve_jax(model, PSF_DATA).copy() + FlatSky_cpu(background, params['s_a'].value
            model_conv = _fftconvolve_jax(model+flat_sky_total,PSF_DATA).copy()
        if convolution_mode == 'CPU':
            model_conv = scipy.signal.fftconvolve(model+flat_sky_total, PSF_DATA_raw,'same')
            model_dict['model_total_conv'] = model_conv
    else:
        model_dict['model_total_conv'] = model + flat_sky_total


    # model_dict['best_residual'] = data_2D - model_dict['model_total']
    # bkg_comp_total
    model_dict['model_total_conv'] = np.asarray(model_conv)
    model_dict['best_residual_conv'] = np.asarray(data_2D) - model_dict['model_total_conv']


    model_dict['deconv_bkg'] = np.asarray(flat_sky_total)
    # bkg_comp_total
    model_dict['conv_bkg'] = np.asarray(_fftconvolve_jax(flat_sky_total,PSF_DATA).copy())

    pf.writeto(imagename.replace('.fits', '') +
               "_" + "conv_model" + special_name + save_name_append + '.fits',
               model_dict['model_total_conv'], overwrite=True)

    total_image_results_conv.append(imagename.replace('.fits', '') +
               "_" + "conv_model" + special_name + save_name_append + '.fits')

    pf.writeto(imagename.replace('.fits', '') +
               "_" + "dec_model" + special_name + save_name_append + '.fits',
               model_dict['model_total_dec'], overwrite=True)

    total_image_results_deconv.append(imagename.replace('.fits', '') +
               "_" + "dec_model" + special_name + save_name_append + '.fits')


    pf.writeto(imagename.replace('.fits', '') +
               "_" + "residual" + special_name + save_name_append + ".fits",
               model_dict['best_residual_conv'], overwrite=True)
    copy_header(imagename, imagename.replace('.fits', '') +
                "_" + "conv_model" + special_name + save_name_append + '.fits',
                imagename.replace('.fits', '') +
                "_" + "conv_model" + special_name + save_name_append + '.fits')
    copy_header(imagename, imagename.replace('.fits', '') +
                "_" + "dec_model" + special_name + save_name_append + '.fits',
                imagename.replace('.fits', '') +
                "_" + "dec_model" + special_name + save_name_append + '.fits')
    copy_header(imagename, imagename.replace('.fits', '') +
                "_" + "residual" + special_name + save_name_append + '.fits',
                imagename.replace('.fits', '') +
                "_" + "residual" + special_name + save_name_append + '.fits')

    pf.writeto(imagename.replace('.fits', '') +
               "_" + "dec_model" + special_name + save_name_append + '.fits',
               model_dict['model_total_dec'], overwrite=True)
    copy_header(imagename, imagename.replace('.fits', '') +
                "_" + "dec_model" + special_name + save_name_append + '.fits',
                imagename.replace('.fits', '') +
                "_" + "dec_model" + special_name + save_name_append + '.fits')

    pf.writeto(imagename.replace('.fits', '') +
               "_" + "deconv_bkg" + special_name + save_name_append + '.fits',
               model_dict['deconv_bkg'], overwrite=True)
    copy_header(imagename, imagename.replace('.fits', '') +
                "_" + "deconv_bkg" + special_name + save_name_append + '.fits',
                imagename.replace('.fits', '') +
                "_" + "deconv_bkg" + special_name + save_name_append + '.fits')

    pf.writeto(imagename.replace('.fits', '') +
               "_" + "conv_bkg" + special_name + save_name_append + '.fits',
               model_dict['conv_bkg'], overwrite=True)
    copy_header(imagename, imagename.replace('.fits', '') +
                "_" + "conv_bkg" + special_name + save_name_append + '.fits',
                imagename.replace('.fits', '') +
                "_" + "conv_bkg" + special_name + save_name_append + '.fits')
    bkg_images.append(imagename.replace('.fits', '') +
               "_" + "deconv_bkg" + special_name + save_name_append + '.fits')
    bkg_images.append(imagename.replace('.fits', '') +
               "_" + "conv_bkg" + special_name + save_name_append + '.fits')


    # # initial minimization.
    # method1 = 'differential_evolution'
    # print(' >> Using', method1, ' solver for first optimisation run... ')
    # # take parameters from previous run, and re-optimize them.
    # #     method2 = 'ampgo'#'least_squares'
    # method2 = 'least_squares'



    image_results_conv.append(imagename.replace('.fits', '') +
                              "_" + "conv_model" +
                              special_name + save_name_append + '.fits')
    image_results_deconv.append(imagename.replace('.fits', '') +
                                "_" + "dec_model" +
                                special_name + save_name_append + '.fits')
    image_results_conv.append(imagename.replace('.fits', '') +
                              "_" + "residual" +
                              special_name + save_name_append + ".fits")

    # save mini results (full) to a pickle file.
    with open(imagename.replace('.fits',
                                '_' + 'fit' +
                                special_name + save_name_append + '.pickle'),
              "wb") as f:
        pickle.dump(result, f)

    with open(imagename.replace('.fits',
                                '_' + 'fit' +
                                special_name + save_name_append + '_modeldict.pickle'),
              "wb") as f:
        pickle.dump(model_dict, f)


    exec_time = time.time() - startTime
    print('Exec time fitting=', exec_time, 's')



    # save results to csv file.
    try:
        save_results_csv(result_mini=result,
                         save_name=image_results_conv[-2].replace('.fits', ''),
                         ext='.csv',
                         save_corr=True, save_params=True)
    except:
        print('Error Saving Results to a csv file!!!')
        pass

    return (result, mini, result_1, result_extra, model_dict, image_results_conv,
            image_results_deconv, bkg_images, smodel2D, model_temp)


def return_and_save_model(mini_results, imagename, ncomponents, background=0.0,
                          save_results=False,save_name_append=''):
    params = mini_results.params
    data_2D = load_fits_data(imagename)
    model_temp = Model(sersic2D)
    model = 0
    PSF_CONV = True
    size = load_fits_data(imagename).shape
    FlatSky_level = mad_std(data_2D)
    xy = np.meshgrid(np.arange((size[0])), np.arange((size[1])))
    model_dict = {}
    image_results_conv = []
    image_results_deconv = []
    for i in range(1, ncomponents + 1):
        model_temp = sersic2D(xy, params['f' + str(i) + '_x0'],
                              params['f' + str(i) + '_y0'],
                              params['f' + str(i) + '_PA'],
                              params['f' + str(i) + '_ell'],
                              params['f' + str(i) + '_n'],
                              params['f' + str(i) + '_In'],
                              params['f' + str(i) + '_Rn'],
                              params['f' + str(i) + '_cg'],) + \
                     background / ncomponents + FlatSky(FlatSky_level,params['s_a']) / ncomponents
        # print(model_temp[0])
        model = model + model_temp
        # print(model)
        model_dict['model_c' + str(i)] = model_temp

        if PSF_CONV == True:
            model_dict['model_c' + str(i) + '_conv'] = scipy.signal.fftconvolve(
                model_temp, PSF_DATA,
                'same')  # + FlatSky(FlatSky_level, params['s_a'])/ncomponents
        else:
            model_dict['model_c' + str(i) + '_conv'] = model_temp

        if save_results is True:
            pf.writeto(imagename.replace('.fits', '') + "_" + str(
                ncomponents) + "C_model_component_" + str(
                i) + special_name + save_name_append + '.fits',
                       model_dict['model_c' + str(i) + '_conv'], overwrite=True)
            copy_header(imagename, imagename.replace('.fits', '') + "_" + str(
                ncomponents) + "C_model_component_" + str(
                i) + special_name + save_name_append + '.fits',
                        imagename.replace('.fits', '') + "_" + str(
                            ncomponents) + "C_model_component_" + str(
                            i) + special_name + save_name_append + '.fits')
            pf.writeto(imagename.replace('.fits', '') + "_" + str(
                ncomponents) + "C_dec_model_component_" + str(
                i) + special_name + save_name_append + '.fits',
                       model_dict['model_c' + str(i)], overwrite=True)
            copy_header(imagename, imagename.replace('.fits', '') + "_" + str(
                ncomponents) + "C_dec_model_component_" + str(
                i) + special_name + save_name_append + '.fits',
                        imagename.replace('.fits', '') + "_" + str(
                            ncomponents) + "C_dec_model_component_" + str(
                            i) + special_name + save_name_append + '.fits')

            image_results_conv.append(imagename.replace('.fits', '') + "_" + str(
                ncomponents) + "C_model_component_" + str(
                i) + special_name + save_name_append + '.fits')
            image_results_deconv.append(
                imagename.replace('.fits', '') + "_" + str(
                    ncomponents) + "C_dec_model_component_" + str(
                    i) + special_name + save_name_append + '.fits')

    #     model = model
    model_dict['model_total'] = model  # + FlatSky(FlatSky_level, params['s_a'])

    if PSF_CONV == True:
        model_dict['model_total_conv'] = scipy.signal.fftconvolve(model,
                                                                  PSF_DATA,
                                                                  'same')  # + FlatSky(FlatSky_level, params['s_a'])
    else:
        model_dict['model_total_conv'] = model

    model_dict['best_residual'] = data_2D - model_dict['model_total']
    model_dict['best_residual_conv'] = data_2D - model_dict['model_total_conv']

    if save_results == True:
        pf.writeto(imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_model" + special_name + save_name_append + '.fits',
                   model_dict['model_total_conv'], overwrite=True)
        pf.writeto(imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_residual" + special_name + save_name_append + ".fits",
                   model_dict['best_residual_conv'], overwrite=True)
        copy_header(imagename, imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_model" + special_name + save_name_append + '.fits',
                    imagename.replace('.fits', '') + "_" + str(
                        ncomponents) + "C_model" + special_name + save_name_append + '.fits')
        copy_header(imagename, imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_residual" + special_name + save_name_append + '.fits',
                    imagename.replace('.fits', '') + "_" + str(
                        ncomponents) + "C_residual" + special_name + save_name_append + '.fits')

        pf.writeto(imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_dec_model" + special_name + save_name_append + '.fits',
                   model_dict['model_total'], overwrite=True)
        copy_header(imagename, imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_dec_model" + special_name + save_name_append + '.fits',
                    imagename.replace('.fits', '') + "_" + str(
                        ncomponents) + "C_dec_model" + special_name + save_name_append + '.fits')
        # initial minimization.

        image_results_conv.append(imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_model" + special_name + save_name_append + '.fits')
        image_results_deconv.append(imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_dec_model" + special_name + save_name_append + '.fits')
        image_results_conv.append(imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_residual" + special_name + save_name_append + ".fits")
        with open(imagename.replace('.fits', '_' + str(
                ncomponents) + 'C_de_fit' + special_name + save_name_append + '.pickle'),
                  "wb") as f:
            pickle.dump(mini_results, f)
    return (model_dict, image_results_conv, image_results_deconv)


def compute_model_properties(model_list,  # the model list of each component
                             which_model,  # `convolved` or `deconvolved`?
                             residualname,
                             rms,  # the native rms from the data itself.
                             mask_region = None,
                             z=None,
                             sigma_mask=5.0,
                             last_level = 1.0,
                             vmin_factor=1.0,
                             iterations = 2,
                             verbose=0):
    """
    Helper function function to calculate model component properties.

    For each model component fitted to a data using the sercic profile, perform morphometry on each
    component image, both deconvolved and convolved images.
    """
    model_properties = {}
    kk = 1
    # if which_model == 'conv':
    #     rms_model = rms
    #     dilation_size = 2
    # if which_model == 'deconv':
    #     rms_model = rms/len(model_list)
    #     dilation_size = 2
    if which_model == 'conv':
        dilation_size = dilation_size = get_dilation_size(model_list[0])
    if which_model == 'deconv':
        dilation_size = 2

    if verbose >= 1:
        print(f' ++==>> Dilation size = {dilation_size}')
        print(f' ++==>> Iterations = {iterations}')
        show_figure = True
    else:
        show_figure = False

    for model_component in model_list:
        try:
            print('Computing properties of model component: ', os.path.basename(model_component))
            model_component_data = load_fits_data(model_component)
            if which_model == 'conv':
                rms_model = rms
            if which_model == 'deconv':
                rms_model = mad_std(model_component_data) + rms
            # if verbose >= 1:
            #     print(' --==>> STD RMS of model component: ', rms_model)
            #     print(' --==>> STD RMS of model bkg: ', rms)
            #     print(' --==>> Ratio rms_model/rms_bkg: ', rms_model/rms)

            _, mask_component = mask_dilation(model_component,
                                            rms=rms_model,
                                            sigma=sigma_mask, 
                                            dilation_size=dilation_size,
                                            iterations=iterations, 
                                            PLOT=show_figure)
            
            # print('number of pixesl in model mask = ', np.sum(mask_component))
            # print('number of pixesl in model mask * regions mask = ', np.sum(mask_component*mask_region))
            # print('number of pixesl in regions mask = ', np.sum(mask_region))

            properties, _, _ = measures(imagename=model_component,
                                            residualname=residualname,
                                            z=z,
                                            sigma_mask=sigma_mask,
                                            last_level = last_level,
                                            vmin_factor=vmin_factor,
                                            dilation_size=dilation_size,
                                            mask = mask_region,
                                            mask_component=mask_component,
                                            show_figure = show_figure,
                                            apply_mask=False,
                                            # data_2D=load_fits_data(model_component),
                                            rms=rms_model)

            model_properties[f"model_c_{which_model}_{kk}_props"] = properties.copy()
            # model_properties[f"model_c_{which_model}_{kk}_props"]['model_file'] = model_component
            model_properties[f"model_c_{which_model}_{kk}_props"]['comp_ID'] = kk
            model_properties[f"model_c_{which_model}_{kk}_props"][
                'model_file'] = os.path.basename(model_component)
            kk = kk + 1
        except:
            empty_properties = {key: np.nan for key in model_properties[f"model_c_{which_model}_{kk-1}_props"].keys()}
            model_properties[f"model_c_{which_model}_{kk}_props"] = empty_properties.copy()
            model_properties[f"model_c_{which_model}_{kk}_props"]['comp_ID'] = kk
            model_properties[f"model_c_{which_model}_{kk}_props"][
                'model_file'] = os.path.basename(model_component)
            print('Error computing properties of model component: ', os.path.basename(model_component))
            kk = kk + 1

    return (model_properties)


def evaluate_compactness(deconv_props, conv_props):
    """
    Determine if a model component of a radio structure is compact or extended.

    It uses the concentration index determined from the areas A20, A50, A80 and A90.

    """
    import pandas as pd
    ncomps = deconv_props['comp_ID'].shape[0]

    Spk_ratio = np.asarray(conv_props['peak_of_flux']) / np.asarray(deconv_props['peak_of_flux'])
    I50_ratio = np.asarray(deconv_props['I50']) / np.asarray(conv_props['I50'])
    class_criteria = {}
    for i in range(ncomps):
        class_criteria[f"comp_ID_{i + 1}"] = {}
        if Spk_ratio[i] < 0.5:
            class_criteria[f"comp_ID_{i + 1}"]['Spk_class'] = 'C'
        if Spk_ratio[i] >= 0.5:
            class_criteria[f"comp_ID_{i + 1}"]['Spk_class'] = 'D'

        if (I50_ratio[i] < 0.5) or (np.isnan(I50_ratio[i])): #nan because the component is too
            # small.
            class_criteria[f"comp_ID_{i + 1}"]['I50_class'] = 'C'
        if I50_ratio[i] >= 0.5:
            class_criteria[f"comp_ID_{i + 1}"]['I50_class'] = 'D'

        AC1_conv_check = conv_props['AC1'].iloc[i]
        AC2_conv_check = conv_props['AC2'].iloc[i]
        AC1_deconv_check = deconv_props['AC1'].iloc[i]
        AC2_deconv_check = deconv_props['AC2'].iloc[i]
        # print(AC1_conv_check)

        # if (AC1_conv_check >= 1.0) or (AC1_conv_check == np.inf):
        #     class_criteria[f"comp_ID_{i+1}"]['AC1_conv_class'] = 'C'
        # if (AC2_conv_check >= 1.0) or AC2_conv_check == np.inf:
        #     class_criteria[f"comp_ID_{i+1}"]['AC2_conv_class'] = 'C'
        if (AC1_deconv_check >= 1.0) or (np.isnan(AC1_deconv_check)):
            class_criteria[f"comp_ID_{i + 1}"]['AC1_deconv_class'] = 'C'
        if AC1_deconv_check < 1.0:
            class_criteria[f"comp_ID_{i + 1}"]['AC1_deconv_class'] = 'D'
        if AC2_deconv_check >= 0.75:
            class_criteria[f"comp_ID_{i + 1}"]['AC2_deconv_class'] = 'C'
        if AC2_deconv_check < 0.75:
            class_criteria[f"comp_ID_{i + 1}"]['AC2_deconv_class'] = 'D'

    class_results_df = pd.DataFrame(class_criteria)
    for i in range(ncomps):
        dessision_compact = np.sum(class_results_df[f"comp_ID_{i + 1}"] == 'C')
        # dessision_diffuse = np.sum(class_results_df[f"comp_ID_{i+1}"]=='D')
        if dessision_compact > 2:
            class_criteria[f"comp_ID_{i + 1}"]['final_class'] = 'C'
        if dessision_compact < 2:
            class_criteria[f"comp_ID_{i + 1}"]['final_class'] = 'D'
        if dessision_compact == 2:
            try:
                class_criteria[f"comp_ID_{i + 1}"]['final_class'] = \
                    class_criteria[(f"comp_ID_{i + 1}")]['Spk_class']
            except:
                class_criteria[f"comp_ID_{i + 1}"]['final_class'] = 'D'
    return (class_criteria)

def format_nested_data(nested_data):
    """
    Format to a data frame a list of nested dictionaries.

    Parameters
    ----------
    nested_data : list of dictionaries
        The list of dictionaries to be formatted.

    Returns
    -------
    df : pandas.DataFrame
        The formatted data frame.
    """
    processed_data = []
    for item in nested_data:
        for model_name, props in item.items():
            props['model_name'] = model_name  # Add the model name to the dictionary
            processed_data.append(props)
    df = pd.DataFrame(processed_data)
    return(df)


def compute_model_stats_GPU(params, imagename, residualname, psf_name,
                            ncomponents, num_simulations=2000,
                            sigma=6, iterations=2,
                            save_results=False, special_name=''):
    """
    Function to run a Monte Carlo simulation from the best optimized fit parameters.
    This can be used to compute a proper error for the total flux on the model.
    Be aware that this function is computing expensive as it requires
    running a 2D convolution on each iteration. This function is optimized to run
    on a CUDA GPU.
    """

    BA = beam_area2(imagename)
    data_2D_cpu = load_fits_data(imagename)
    data_2D = jnp.asarray(data_2D_cpu)
    residual_2D_cpu = load_fits_data(residualname)
    residual_2D = jnp.asarray(residual_2D_cpu)
    residual_2D_shuffled = shuffle_2D(residual_2D_cpu)
    background = jnp.asarray(residual_2D_shuffled)
    model_temp = Model(sersic2D)
    PSF_DATA_raw = load_fits_data(psf_name)

    PSF_DATA = jnp.asarray(PSF_DATA_raw)
    omaj, omin, _, _, _ = beam_shape(imagename)
    dilation_size = int(np.sqrt(omaj * omin) / (2 * get_cell_size(imagename)))
    _, mask_cpu = mask_dilation(imagename, sigma=sigma, iterations=iterations,
                                dilation_size=dilation_size, PLOT=True)
    mask = jnp.asarray(mask_cpu)
    values = params.valuesdict()
    stderr = jnp.asarray(
        [params[name].stderr for name in values.keys()])
    #     dfstderr = pd.DataFrame({'value': list(values.values()), 'stderr': stderr},
    #                       index=values.keys())
    params_values = jnp.asarray(list(values.values()))
    #     random_params = generate_random_params(params_values,stderr)

    size = load_fits_data(imagename).shape
    FlatSky_level = mad_std(data_2D_cpu)
    xy = jnp.meshgrid(jnp.arange((size[0])), jnp.arange((size[1])))
    model_dict = {}
    image_results_conv = []
    image_results_deconv = []

    #     num_simulations = 50

    # Generate a set of random parameters for each simulation and compute the total flux
    #     total_fluxes = []
    #     random_params_list = []
    print('Running MCMC on best-fit parameters (using cuda gpu).')

    def mcmc_sim(random_params):
        #         params_values, stderr = params_result
        #     for l in tqdm(range(num_simulations)):
        #         random_params_list.append(random_params)
        model = 0
        #         model_comps = []
        sub_com_res_results = jnp.zeros(ncomponents)
        sub_com_flux_results = jnp.zeros(ncomponents)
        for i in range(1, ncomponents + 1):
            # 8 means that we have 8 parameters per model-component.
            # e.g. In, Rn, n, q, c, PA, x0 and y0,
            mcmc_params = random_params[int(8 * (i - 1)):int(8 * (i))]
            model_temp = sersic2D(xy,
                                  mcmc_params[0],
                                  # params['f' + str(i) + '_x0'],
                                  mcmc_params[1],
                                  # params['f' + str(i) + '_y0'],
                                  mcmc_params[2],
                                  # params['f' + str(i) + '_PA'],
                                  mcmc_params[3],
                                  # params['f' + str(i) + '_ell'],
                                  mcmc_params[4],  # params['f' + str(i) + '_n'],
                                  mcmc_params[5],
                                  # params['f' + str(i) + '_In'],
                                  mcmc_params[6],
                                  # params['f' + str(i) + '_Rn'],
                                  mcmc_params[7]) + background / ncomponents + \
                         FlatSky(FlatSky_level, random_params[-1]) / ncomponents
            # print(model_temp[0])
            model_temp_dec = model_temp
            model_temp_conv = jax.scipy.signal.fftconvolve(model_temp_dec,
                                                             PSF_DATA, 'same')
            # cp.cuda.Stream.null.synchronize()
            #             model_comps.append(model_temp_conv)
            model = model + model_temp_conv
            total_flux_comp_i = jnp.sum(model_temp_conv * mask) / BA
            sub_com_flux_results[i - 1] = total_flux_comp_i

            total_res_flux_sub_comp_i = jnp.sum((data_2D - model) * mask) / BA
            sub_com_res_results[i - 1] = total_res_flux_sub_comp_i

        model_total_conv = model
        total_flux_random_model = jnp.sum(model_total_conv * mask) / BA
        residual_model = jnp.sum(((data_2D - model_total_conv) * mask) ** 2.0)
        residual_nrss_model = calc_nrss(data_2D * mask, model_total_conv * mask)

        if np.isnan(total_flux_random_model) == True:
            good_model = False
        else:
            good_model = True

        #         total_fluxes.append(total_flux_random_model)
        return (
            total_flux_random_model, good_model, residual_model,
            residual_nrss_model,
            sub_com_flux_results, sub_com_res_results)

    random_params_list = []
    for i in range(num_simulations):
        random_params = generate_random_params(params_values, stderr)
        random_params_list.append(random_params)
    random_params_list = jnp.asarray(random_params_list)

    #     with Pool(max_pool) as p:
    #         results = list(
    #             tqdm(
    #                 p.imap(mcmc_sim,random_params_list),
    #                 total=num_simulations
    #             )
    #         )

    results = []
    for i in tqdm(range(len(random_params_list))):
        results.append(mcmc_sim(random_params_list[i]))

    total_fluxes_all_gpu = []
    residuals_gpu = []
    residuals_nrss_gpu = []
    flags_gpu = []
    sub_comp_fluxes_gpu = []
    sub_comp_residuals_gpu = []
    for dict_temp in results:
        total_fluxes_all_gpu.append(dict_temp[0])
        flags_gpu.append(dict_temp[1])
        residuals_gpu.append(dict_temp[2])
        residuals_nrss_gpu.append(dict_temp[3])
        sub_comp_fluxes_gpu.append(dict_temp[4])
        sub_comp_residuals_gpu.append(dict_temp[5])

    total_fluxes_all = jnp.asarray(total_fluxes_all_gpu)
    flags = jnp.asarray(flags_gpu).get()
    residuals = jnp.asarray(residuals_gpu)
    residuals_nrss = jnp.asarray(residuals_nrss_gpu)
    sub_comp_fluxes = jnp.asarray(sub_comp_fluxes_gpu)[flags]
    sub_comp_residuals = jnp.asarray(sub_comp_residuals_gpu)[flags]

    total_fluxes = total_fluxes_all[flags]
    random_params_list = random_params_list[flags]
    residuals = residuals[flags]
    residuals_nrss = residuals_nrss[flags]
    add_flags = np.where(abs(total_fluxes) > 100 * mad_std(abs(total_fluxes)))[0]
    total_fluxes = np.delete(total_fluxes, add_flags, axis=0)
    random_params_list = np.delete(random_params_list, add_flags, axis=0)
    residuals = np.delete(residuals, add_flags, axis=0)
    residuals_nrss = np.delete(residuals_nrss, add_flags, axis=0)
    sub_comp_fluxes = np.delete(sub_comp_fluxes, add_flags, axis=0)
    sub_comp_residuals = np.delete(sub_comp_residuals, add_flags, axis=0)

    mean_flux = jnp.mean(total_fluxes)
    std_flux = jnp.std(total_fluxes)
    print("Estimated total flux = %.4f +/- %.4f" % (mean_flux, std_flux))
    labels_names = list(mini_results.params.valuesdict().keys())
    #     labels = ["Parameter {}".format(i+1) for i in range(len(params))] + ["Total Flux"]
    labels = ["Parameter " + i for i in labels_names] + ["Total Flux"]

    data_results = np.column_stack([random_params_list, total_fluxes])
    #     fig = corner.corner(data_results, labels=labels, quantiles=[0.16, 0.5, 0.84],
    #                         show_titles=True, title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 12})
    #     fig.suptitle("Monte Carlo Simulation Results", fontsize=16, y=1.0)

    #     model_dict['best_residual'] = data_2D - model_dict['model_total']
    #     model_dict['best_residual_conv'] = data_2D - model_dict['model_total_conv']
    return (
        total_fluxes, random_params_list, data_results, residuals,
        residuals_nrss,
        sub_comp_fluxes, sub_comp_residuals)


def run_mcmc_mini(imagename, psf_name, mini_results, residualname=None, rms_map=None,
                  n_components=None, nwalkers_multiplier=20, backend_name="filename.h5",
                  burn_in_phase=1000, production_steps=6000):
    """
    GPU Optimized function to run a MCMC of a multi-dimensional sersic model.
    It uses JAX.
    This function is not feasible to be run on a CPU.

    Please, only use if you have a GPU, if not, it can take days....

    """
    import emcee
    import corner

    if residualname is not None:
        residual_2D = pf.getdata(residualname)
        residual_2D_shuffled = shuffle_2D(residual_2D)
        background = jnp.array(residual_2D_shuffled)
    else:
        background = jnp.array(rms_map)

    omaj, omin, _, _, _ = beam_shape(imagename)
    dilation_size = int(
        np.sqrt(omaj * omin) / (2 * get_cell_size(imagename)))

    rms_std_res = mad_std(rms_map)
    _, mask_region = mask_dilation(imagename,
                                   rms=rms_std_res,
                                   sigma=6.0, dilation_size=dilation_size,
                                   iterations=5, PLOT=True)

    data_2D = load_fits_data(imagename) * mask_region
    data_2D_gpu = jnp.array(data_2D)
    PSF_DATA = jnp.array(load_fits_data(psf_name))
    size = data_2D.shape
    xy = jnp.meshgrid(jnp.arange((size[1])), jnp.arange((size[0])))
    FlatSky_level_GPU = jnp.array(mad_std(data_2D))

    # Read input parameters and std
    params = mini_results.copy()
    #     opt_params = jnp.asarray(list(params.valuesdict().values()))
    opt_params = jnp.array(list(params.valuesdict().values()))
    stderr = jnp.array([params[name].stderr for name in params.valuesdict().keys()])

    if n_components is None:
        n_components = int((stderr.shape[0] - 1) / 8)

    nfunctions = n_components
    
    
    @jit
    def convolve_on_gpu(image, psf):
        image_fft = fft2(image)
        psf_fft = fft2(psf)
        conv_fft = image_fft * psf_fft
        return fftshift(jnp.real(ifft2(conv_fft)))

    @jit
    def log_likelihood(random_params):
        for i in range(1, nfunctions + 1):
            mcmc_params = random_params[int(8 * (i - 1)):int(8 * (i))]
            model = sersic2D_GPU(xy,
                                 mcmc_params[0],
                                 # params['f' + str(i) + '_x0'],
                                 mcmc_params[1],
                                 # params['f' + str(i) + '_y0'],
                                 mcmc_params[2],
                                 # params['f' + str(i) + '_PA'],
                                 mcmc_params[3],
                                 # params['f' + str(i) + '_ell'],
                                 mcmc_params[4],  # params['f' + str(i) + '_n'],
                                 mcmc_params[5],
                                 # params['f' + str(i) + '_In'],
                                 mcmc_params[6],
                                 # params['f' + str(i) + '_Rn'],
                                 mcmc_params[7]) + background / nfunctions + \
                    FlatSky(FlatSky_level_GPU, random_params[-1]) / nfunctions

        MODEL_2D_conv = convolve_on_gpu(model, PSF_DATA)
        #         print(model)
        residual = data_2D_gpu - MODEL_2D_conv
        chi2 = jnp.sum(residual ** 2) / jnp.sum(data_2D_gpu)

        #         inv_sigma2 = 1.0/(1+MODEL_2D_conv**2*jnp.exp(2))
        #     #     return -0.5*(chi2*inv_sigma2 - np.log(inv_sigma2)))
        dof = len(model) - len(params)
        log_like = -0.5 * (chi2 + dof * jnp.log(2 * jnp.pi))
        total_flux_model = jnp.sum(MODEL_2D_conv)
        total_flux_res = jnp.sum(residual)
        #         log_like = chi2
        #         log_like = -0.5*(jnp.sum(residual ** 2*inv_sigma2 - jnp.log(inv_sigma2)))
        return log_like, (total_flux_model, total_flux_res)

    def ln_prob(p, params):
        # Calculate log-prior probability
        log_prior_prob = log_prior(p, params)
        if jnp.isinf(log_prior_prob):
            return log_prior_prob

        # Update parameter values based on proposed step
        for i, (param_name, param) in enumerate(params.items()):
            param.value = p[i]

        params_model = jnp.array(list(params.valuesdict().values()))
        log_likelihood_prob = log_likelihood(params_model)[0]
        return log_prior_prob + log_likelihood_prob

    # @jit
    #     def log_prior(p, params):
    #         out_of_bounds = []
    #         for i, (param_name, param) in enumerate(params.items()):
    #             if '_cg' in param_name:
    #                 if (p[i] < -0.2 or p[i] > 0.2):
    #                     out_of_bounds.append(param_name)
    #             if 's_a' in param_name:
    #                 if (p[i] < 0.0 or p[i] > 10.0):
    #                     out_of_bounds.append(param_name)
    #             else:
    #                 if (p[i] < param.min or p[i] > param.max):
    #                     out_of_bounds.append(param_name)
    #         if len(out_of_bounds) == 0:
    #             return 0.0  # Return zero for in-bounds parameters
    #         else:
    #             return -jnp.inf  # Return negative infinity for out-of-bounds parameters

    #     def log_prior(p, params):
    #         for i, (param_name, param) in enumerate(params.items()):
    # #             if (p[i] < param.min or p[i] > param.max):
    #             if (p[i] < param.min or p[i] > param.max):
    #                 return -jnp.inf  # Return negative infinity for out-of-bounds parameters
    #         return 0.0  # Return zero for in-bounds parameters
    #     def log_prior(p, params):
    #         for i, (param_name, param) in enumerate(params.items()):
    #             if (p[i] < 0.4 or p[i] > param.max+1) and '_n' in param_name:
    #                 return -jnp.inf  # Return negative infinity for out-of-bounds parameters
    #             if (p[i] < param.min or p[i] > param.max*10.0) and '_In' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min or p[i] > param.max) and '_Rn' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < 0.0 or p[i] > 0.8) and '_ell' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min or p[i] > param.max) and '_PA' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min*0.9 or p[i] > param.max*1.1) and '_x0' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min*0.9 or p[i] > param.max*1.1) and '_y0' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < 0.0 or p[i] > 10.0) and 's_a' in param_name:
    #                 return -jnp.inf
    #         return 0.0  # Return zero for in-bounds parameters

    def log_prior(p, params):
        out_of_bounds = []
        for i, (param_name, param) in enumerate(params.items()):
            pvalue = param.value
            if (p[i] < pvalue * 0.8 or p[i] > pvalue * 1.25) and '_n' in param_name:
                out_of_bounds.append(param_name)
            if (p[i] < pvalue * 0.8 or p[i] > pvalue * 1.25) and '_In' in param_name:
                out_of_bounds.append(param_name)
            if (p[i] < pvalue * 0.8 or p[i] > pvalue * 1.25) and '_Rn' in param_name:
                out_of_bounds.append(param_name)
            if (p[i] < 0.0 or p[i] > pvalue + 0.2) and '_ell' in param_name:
                out_of_bounds.append(param_name)
            if (p[i] < pvalue - 20 or p[i] > pvalue + 20) and '_PA' in param_name:
                out_of_bounds.append(param_name)
            if (p[i] < pvalue - 10 or p[i] > pvalue + 10) and '_x0' in param_name:
                out_of_bounds.append(param_name)
            if (p[i] < pvalue - 10 or p[i] > pvalue + 10) and '_y0' in param_name:
                out_of_bounds.append(param_name)
            if (p[i] < pvalue - 0.1 or p[i] > pvalue + 0.1) and '_cg' in param_name:
                out_of_bounds.append(param_name)
            if (p[i] < 0.0 or p[i] > pvalue + 1) and 's_a' in param_name:
                out_of_bounds.append(param_name)
        if len(out_of_bounds) == 0:
            return 0.0  # Return zero for in-bounds parameters
        else:
            return -jnp.inf  # Return negative infinity for out-of-bounds parameters

    #     def log_prior(p, params):
    #         out_of_bounds = []
    #         for i, (param_name, param) in enumerate(params.items()):
    #             if (p[i] < param.min-0.1 or p[i] > param.max+0.1) and '_n' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min or p[i] > param.max) and '_In' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min or p[i] > param.max) and '_Rn' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min or p[i] > param.max) and '_ell' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min or p[i] > param.max) and '_PA' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min-10 or p[i] > param.max+10) and '_x0' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min-10 or p[i] > param.max+10) and '_y0' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min-0.1 or p[i] > param.max+0.1) and '_cg' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < 0.0 or p[i] > 10) and 's_a' in param_name:
    #                 out_of_bounds.append(param_name)
    #         if len(out_of_bounds) == 0:
    #             return 0.0  # Return zero for in-bounds parameters
    #         else:
    #             return -jnp.inf  # Return negative infinity for out-of-bounds parameters

    #     def log_prior(p, params):
    #         out_of_bounds = []
    #         for i, (param_name, param) in enumerate(params.items()):
    #             if (p[i] < param.min-0.1 or p[i] > param.max+0.1) and '_n' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min or p[i] > param.max) and '_In' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min or p[i] > param.max) and '_Rn' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min or p[i] > param.max) and '_ell' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min or p[i] > param.max) and '_PA' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min-30 or p[i] > param.max+30) and '_x0' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min-30 or p[i] > param.max+30) and '_y0' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < param.min-0.5 or p[i] > param.max+0.5) and '_cg' in param_name:
    #                 out_of_bounds.append(param_name)
    #             if (p[i] < 0.0 or p[i] > 10) and 's_a' in param_name:
    #                 out_of_bounds.append(param_name)
    #         if len(out_of_bounds) == 0:
    #             return 0.0  # Return zero for in-bounds parameters
    #         else:
    #             return -jnp.inf  # Return negative infinity for out-of-bounds parameters

    #     def log_prior(p, params):
    #         for i, (param_name, param) in enumerate(params.items()):
    #             if (p[i] < 0.25 or p[i] > param.max+1) and '_n' in param_name:
    #                 return -jnp.inf  # Return negative infinity for out-of-bounds parameters
    #             if (p[i] < param.min or p[i] > param.max*10.0) and '_In' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min or p[i] > param.max) and '_Rn' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < 0.0 or p[i] > 1.0) and '_ell' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min or p[i] > param.max) and '_PA' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min*0.9 or p[i] > param.max*1.1) and '_x0' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min*0.9 or p[i] > param.max*1.1) and '_y0' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < 0.0 or p[i] > 10.0) and 's_a' in param_name:
    #                 return -jnp.inf
    #         return 0.0  # Return zero for in-bounds parameters
    #     def log_prior(p, params):
    #         for i, (param_name, param) in enumerate(params.items()):
    #             if (p[i] < 0.25 or p[i] > 6.0) and '_n' in param_name:
    #                 return -jnp.inf  # Return negative infinity for out-of-bounds parameters
    #             if (p[i] < 0.0 or p[i] > 10.0) and '_In' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < 1.0 or p[i] > param.max) and '_Rn' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < 0.0 or p[i] > 1.0) and '_ell' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < -180. or p[i] > 180) and '_PA' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min*0.9 or p[i] > param.max*1.1) and '_x0' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < param.min*0.9 or p[i] > param.max*1.1) and '_y0' in param_name:
    #                 return -jnp.inf
    #             if (p[i] < 0.0 or p[i] > 10.0) and 's_a' in param_name:
    #                 return -jnp.inf
    #         return 0.0  # Return zero for in-bounds parameters

    # Define the log-posterior function for the MCMC
    # @jit
    def log_posterior(params):
        lp = log_prior(params)
        if not jnp.isfinite(lp):
            return -jnp.inf  # Return negative infinity for out-of-bounds parameters
        return lp + log_likelihood(params)[0]

    # prepare the mcmc
    nwalkers = nwalkers_multiplier * len(params)
    ndim = len(params)
    print('    ### Initi of MCMC ###')
    print('--------------------------------')
    print('    >> Number of Sersics:', n_components)
    print('    >> Number of parameters:', ndim)
    print('    >> Number of walkers:   ', nwalkers)
    print('--------------------------------')
    print('    >> Burn-in:   ', burn_in_phase)
    print('    >> Main-Phase Steps:   ', production_steps)
    p0 = []
    for i in range(nwalkers):
        # walker_params = []
        random_params = jnp.array(
            generate_random_params_uniform(np.asarray(list(params.valuesdict().values())), stderr))
        p0.append(random_params)

    #     print(p0)
    backend = emcee.backends.HDFBackend(backend_name)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, args=(params,), threads=1,
                                    backend=backend)
    sampler.run_mcmc(p0, production_steps, progress=True)
    #     samples = sampler.chain[:, burn_in_phase:, :].reshape((-1, ndim))
    #     backend.store(chain=sampler.get_chain(), log_prob=sampler.get_log_prob(), blobs=sampler.get_blobs())
    samples = sampler.get_chain(discard=burn_in_phase, flat=True)
    #     RESIDUAL = []
    #     TOTAL_FLUX = []
    #     for result in sampler.sample(p0,ndim):
    #     # Unpack the result tuple and save the residual and total flux to the backend
    #         residual, total_flux = result
    #         RESIDUAL.append(residual)
    #         TOTAL_FLUX.append(total_flux)

    #     sampler.backend.write_to_hdf5(backend_name)
    #     backend.close()

    print('--------------------------------')
    print('    Parameter Errors            ')
    # Compute the median and 1-sigma error bars for each parameter
    params_median = np.median(samples, axis=0)
    params_err_minus = np.percentile(samples, 16, axis=0) - params_median
    params_err_plus = np.percentile(samples, 84, axis=0) - params_median
    params_err = 0.5 * (params_err_minus + params_err_plus)

    # Print the results
    for i, param_name in enumerate(params):
        print(f"{param_name} = {params_median[i]} +/- {params_err[i]}")

    idx_cut = []  # only select n, In and Rn parameters
    for i in range(0, n_components):
        idx_cut.append(int(i * 8 + 4))
        idx_cut.append(int(i * 8 + 5))
        idx_cut.append(int(i * 8 + 6))

    labels = [f"{param_name}" for param_name in params]
    # Plot the corner plot
    #     fig = corner.corner(samples[:,idx_cut],
    #                         labels=np.asarray(labels)[idx_cut],
    #                         truths=list(params_median[idx_cut]))

    #     plt.show()

    #     # Plot the chains
    #     fig, axes = plt.subplots(len(idx_cut), figsize=(10, 27), sharex=True)
    #     labels = [f"{param_name}" for param_name in params]
    #     for i in range(0,len(idx_cut)):
    #     # for i in idx_cut:
    #         ax = axes[i]
    #         ax.plot(sampler.get_chain()[:, :, idx_cut[i]], color="k", alpha=0.1)
    #         ax.set_xlim(0, len(sampler.get_chain()))
    #         ax.set_ylabel(labels[idx_cut[i]])
    #     axes[-1].set_xlabel("Step")

    return (samples, sampler, backend)

def run_image_fitting(imagelist, residuallist, sources_photometries,
                      n_components, comp_ids=[], mask=None, mask_for_fit=None,
                      use_mask_for_fit=False,
                      which_residual='shuffled',
                      save_name_append='', z=None, aspect=None,
                      convolution_mode='GPU', workers=6,
                      method1='least_squares', method2='least_squares',
                      loss="cauchy", tr_solver="exact",
                      init_params=0.25, final_params=4.0,sigma=6,
                      fix_n=[True, True, True, True, True, True, False],
                      fix_x0_y0 = [True, True, True, True, True, True, True, True],
                      fix_value_n=[0.5, 0.5, 0.5, 1.0], 
                      fix_max_value_Rn = [False, False, False, False],
                      fix_geometry=True,
                      dr_fix=[10, 10, 10, 10, 10, 10, 10, 10],logger=None,
                      parameters_mini_init = None,
                      self_bkg=False, bkg_rms_map=None,verbose=0):
    """
    Support function to run the image fitting to a image or to a list of images.

    Note. This function was implemented to  help with my own research, but it may be useable in
    some contexts. It is not a general function, and it is not well documented.

    What it does:
    For a multi-component source, this function will handle information individually for
    each component and store information in a dictionary.



    """

    results_fit = []
    lmfit_results = []
    lmfit_results_1st_pass = []
    errors_fit = []
    data_properties_list = []
    models = []
    list_results_compact_conv_morpho = []   # store the morphological properties
                                            # of the sum of all convolved compact components
    list_results_compact_deconv_morpho = [] # store the morphological properties
                                            # of the sum of all deconvolved compact components
    list_results_ext_conv_morpho = []       # store the morphological properties
                                            # of the sum of all convolved extended components
    list_results_ext_deconv_morpho = []     # store the morphological properties
                                            # of the sum of all deconvolved extended components
    list_individual_deconv_props = []       # store the morphological properties
                                            # of each deconvolved component.
    list_individual_conv_props = []         # store the morphological properties
                                            # of each convolved component.


    for i in range(len(imagelist)):
        #         model_dict_results = {}
        # try:
        crop_image = imagelist[i]
        crop_residual = residuallist[i]
        print('  ++==>>  Fitting', os.path.basename(crop_image))
        #             dict_results['#imagename'] = crop_image
        data_2D = load_fits_data(crop_image)
        res_2D = load_fits_data(crop_residual)
        rms_std_data = mad_std(data_2D)
        rms_std_res = mad_std(res_2D)
        if verbose >= 2:
            print('rms data = ', rms_std_data * 1e6,
                    '; rms res = ', rms_std_res * 1e6,
                    '; ratio = ', rms_std_res / rms_std_data)

        sigma_level = 3
        vmin = 3
        # i = 0 #to be used in indices[0], e.g. first component
        # omaj, omin, _, _, _ = beam_shape(crop_image)
        # dilation_size = int(
        # np.sqrt(omaj * omin) / (2 * get_cell_size(crop_image)))


        if verbose >= 1:
            do_PLOT = True
            PLOT = True
            SAVE = True
            show_figure = True
        else:
            do_PLOT = False
            PLOT = False
            SAVE = False
            show_figure = False


        if use_mask_for_fit == True:
            print('++==>> USING MASK FOR FITTTTTTTTTTTTTTTT ')
            if mask_for_fit is None:
                print('++==>> CALCULATING MASK FOR FIT IS NONE')
                _, mask_for_fit = mask_dilation(crop_image,
                                                rms=rms_std_res,
                                                sigma=sigma, dilation_size=None,
                                                iterations=6, PLOT=True)
            else:
                print('++==>> USING PROVIDED MASK FOR FIT IS NOT NONE')
                mask_for_fit = mask_for_fit
                plt.figure()
                plt.imshow(mask_for_fit,origin='lower')
                plt.show()
                plt.clf()
                plt.close()

        _, mask_region_i = mask_dilation(crop_image,
                                        rms=rms_std_res,
                                        sigma=5.0, dilation_size=None,
                                        iterations=4, PLOT=True)
        
        mask_region = mask_for_fit.copy() * mask_region_i
        
        data_properties, _, _ = \
            measures(imagename=crop_image,
                     residualname=crop_residual, z=z,
                     sigma_mask=6.0,
                     last_level=1.0, vmin_factor=1.0,
                     dilation_size=None,
                     results_final={},
                     rms=rms_std_res,
                     apply_mask=True,
                     iterations = 3,
                     mask=mask_region_i*mask_region,
                     do_PLOT=do_PLOT, SAVE=SAVE,
                     do_petro = True,
                     show_figure=show_figure,
                     do_measurements='partial',
                     add_save_name='_data',
                     verbose=verbose)
        data_properties_list.append(data_properties)
        # psf_image_size = dilation_size*6
        # psf_image_size = (2 * psf_image_size) // 2 +1

        psf_zise = int(get_beam_size_px(crop_image)[0])
        psf_image_size = int(psf_zise * 10)
        # psf_image_size = int(data_2D.shape[0])
        if verbose >= 2:
            print('++==>> PSF BEAM SIZE is >=> ', psf_zise)
            print('++==>> PSF IMAGE SIZE is ', psf_image_size)

        # psf_image_size = int(load_fits_data(crop_image).shape[0])

        # creates a psf from the beam shape.
        psf_name = tcreate_beam_psf(crop_image, size=(psf_image_size, psf_image_size),
                                    aspect=aspect,
                                    # aspect=None,
                                    )  # ,app_name='_'+str(psf_image_size)+'x'+str(psf_image_size)+'')

        result_mini, mini, result_1, result_extra, model_dict, \
            image_results_conv, image_results_deconv, bkg_images, \
            smodel2D, model_temp = \
                do_fit2D(imagename=crop_image,
                            residualname=crop_residual,
                            which_residual=which_residual,
                            init_constraints=sources_photometries,
                            psf_name=psf_name,
                            params_values_init_IMFIT=None,# imfit_conf_values[0:-1],
                            #fix_n = False,fix_x0_y0=[False,False,False],
                            ncomponents=n_components, constrained=True,
                            fix_n=fix_n,
                            mask_region=mask_for_fit,
                            fix_value_n=fix_value_n,
                            fix_max_value_Rn=fix_max_value_Rn,
                            fix_x0_y0=fix_x0_y0,
                            dr_fix=dr_fix,
                            self_bkg=self_bkg, rms_map=bkg_rms_map,
                            convolution_mode=convolution_mode,
                            fix_geometry=fix_geometry,
                            workers=workers,
                            method1=method1, method2=method2,
                            loss=loss, tr_solver=tr_solver,
                            init_params=init_params, final_params=final_params,
                            parameters_mini_init = parameters_mini_init,
                            save_name_append=save_name_append,logger=logger)

        # print(result_mini.params)
        models.append(model_dict)
        lmfit_results.append(result_mini.params)
        lmfit_results_1st_pass.append(result_1.params)
        special_name = save_name_append
        bkg_deconv = bkg_images[0]
        bkg_conv = bkg_images[1]

        # _, mask_dilated_new = mask_dilation_from_mask(crop_image,
        #                                               mask_region,
        #                                               rms=rms_std_res,
        #                                               iterations=10)

        rms_bkg_deconv = mad_std(load_fits_data(bkg_deconv))
        deconv_model_properties = compute_model_properties(model_list=image_results_deconv[:-1],
                                                           which_model='deconv',
                                                           residualname=crop_residual,
                                                           rms=rms_std_res,
                                                           mask_region = mask_region,
                                                           verbose=verbose
                                                           )
        rms_bkg_conv = mad_std(load_fits_data(bkg_conv))
        conv_model_properties = compute_model_properties(model_list=image_results_conv[:-2],
                                                         which_model='conv',
                                                         residualname=crop_residual,
                                                         rms=rms_std_res,
                                                         mask_region = mask_region,
                                                         verbose=verbose
                                                         )
        list_individual_deconv_props.append(deconv_model_properties)
        list_individual_conv_props.append(conv_model_properties)

        deconv_props = pd.DataFrame(deconv_model_properties).T
        conv_props = pd.DataFrame(conv_model_properties).T
        try:
            class_results = evaluate_compactness(deconv_props, conv_props)
        except:
            #NEED-A-FIX
            class_results = {}
            pass
        # for l in class_results.keys():
        #     ID = 1
        #     deconv_props.loc[l, 'comp_ID'] = ID


        deconv_props.to_csv(image_results_deconv[-1].replace('.fits','_component_properties.csv'),
                            header=True,index=False)
        conv_props.to_csv(image_results_conv[-2].replace('.fits','_component_properties.csv'),
                          header=True, index=False)

        # comp_ids = []
        # print('*************************************')
        # print(class_results)
        try:
            if comp_ids == []:
                ID = 1
                for key in class_results.keys():
                    if class_results[key]['final_class'] == 'C':
                        comp_ids.append(str(ID))
                    ID = ID + 1
            if comp_ids == []:
                comp_ids = ['1']
        except:
            #NEED-A-FIX
            pass

        all_comps_ids = np.arange(1, n_components + 1).astype('str')
        mask_compact_ids = np.isin(all_comps_ids, np.asarray(comp_ids))
        ext_ids = list(all_comps_ids[~mask_compact_ids])
        print('  ++>> Total component IDs modelled:', all_comps_ids)
        print('  ++>> IDs attributed to compact structures:', comp_ids)
        print('  ++>> IDs attributed to extended structures:', ext_ids)

        compact_model = 0
        extended_model = 0
        compact_model_deconv = 0
        extended_model_deconv = 0
        for lc in comp_ids:
            compact_model = (compact_model +
                                model_dict['model_c' + lc + '_conv']-model_dict['conv_bkg'])
            compact_model_deconv = (compact_model_deconv +
                                    model_dict['model_c' + lc]-model_dict['deconv_bkg'])
        # if ext_ids is not None:
        if ext_ids == []:
            extended_model = 0
            extended_model_deconv = 0
            nfunctions = 1
        else:
            for le in ext_ids:
                extended_model = (extended_model +
                                    model_dict['model_c' + le + '_conv']-model_dict['conv_bkg'])
                extended_model_deconv = (extended_model_deconv +
                                            model_dict['model_c' + le]-model_dict['deconv_bkg'])
                nfunctions = None
                # extended_model = extended_model - model_dict['conv_bkg'] * (len(ext_ids) - 1)
                # extended_model_deconv = extended_model_deconv - model_dict['deconv_bkg'] * (
                #             len(ext_ids) - 1)
            extended_model = extended_model + model_dict['conv_bkg']
            extended_model_deconv = extended_model_deconv + model_dict['deconv_bkg']

        # compact_model = compact_model - model_dict['conv_bkg']*(len(comp_ids)-1)
        # compact_model_deconv = compact_model_deconv - model_dict['deconv_bkg']*(len(comp_ids)-1)
        compact_model = compact_model + model_dict['conv_bkg']
        compact_model_deconv = compact_model_deconv + model_dict['deconv_bkg']
        extended_data = load_fits_data(crop_image) - compact_model
        
        if ext_ids != []:
            pf.writeto(crop_image.replace('.fits', '') +
                    "_" + "dec_ext_model" + save_name_append + ".fits",
                    extended_model_deconv, overwrite=True)
            copy_header(crop_image, crop_image.replace('.fits', '') +
                        "_" + "dec_ext_model" + save_name_append + ".fits",
                        crop_image.replace('.fits', '') +
                        "_" + "dec_ext_model" + save_name_append + ".fits")

            pf.writeto(crop_image.replace('.fits', '') +
                       "_" + "ext_model" + save_name_append + ".fits",
                       extended_model, overwrite=True)
            copy_header(crop_image, crop_image.replace('.fits', '') +
                        "_" + "ext_model" + save_name_append + ".fits",
                        crop_image.replace('.fits', '') +
                        "_" + "ext_model" + save_name_append + ".fits")

        pf.writeto(crop_image.replace('.fits', '') +
                "_" + "dec_compact" + save_name_append + ".fits",
                compact_model_deconv, overwrite=True)
        copy_header(crop_image, crop_image.replace('.fits', '') +
                    "_" + "dec_compact" + save_name_append + ".fits",
                    crop_image.replace('.fits', '') +
                    "_" + "dec_compact" + save_name_append + ".fits")

        
        exteded_file_name = crop_image.replace('.fits', '') + \
                            special_name + '_extended.fits'
        pf.writeto(exteded_file_name,extended_data,overwrite=True)
        copy_header(crop_image,exteded_file_name)
        
        compact_file_name = crop_image.replace('.fits', '') + \
                            special_name + '_conv_compact.fits'
        pf.writeto(compact_file_name,compact_model,overwrite=True)
        copy_header(crop_image,compact_file_name)
        
        
        
        decomp_results = plot_decomp_results(imagename=crop_image,
                                                compact=compact_model,
                                                extended_model=extended_model,
                                                rms=rms_std_res,
                                                nfunctions=nfunctions,
                                                special_name=special_name)
        
        decomp_results['compact_model_image'] = compact_file_name
        decomp_results['extended_model_image'] = exteded_file_name

        plot_fit_results(crop_image, model_dict, image_results_conv,
                            sources_photometries,
                            bkg_image = load_fits_data(bkg_conv),
                            crop=False, box_size=200,
                            mask=mask_region,
                            vmax_factor=0.3, vmin_factor=1.0)
        # plt.xlim(0,3)
        plot_slices(load_fits_data(crop_image), load_fits_data(crop_residual), model_dict,
                    image_results_conv[-2], sources_photometries)
        
        parameter_results = result_mini.params.valuesdict().copy()

        try:
            for param in result_mini.params.valuesdict().keys():
                parameter_results[param+'_err'] = result_mini.params[param].stderr
        except:
            pass

        
        parameter_results['#imagename'] = os.path.basename(crop_image)
        parameter_results['residualname'] = os.path.basename(crop_residual)
        parameter_results['beam_size_px'] = psf_zise

        # print('++++++++++++++++++++++++++++++++++++++++')
        # print('++++++++++++++++++++++++++++++++++++++++')
        # print('++++++++++++++++++++++++++++++++++++++++')
        # print(compact_model)
        # print(mask)
        # print(mask_region)
        # print('++++++++++++++++++++++++++++++++++++++++')
        # print('++++++++++++++++++++++++++++++++++++++++')
        # print('++++++++++++++++++++++++++++++++++++++++')

        # _rms_model = mad_std(compact_model)
        # print('**************************')
        # print('**************************')
        # print('RMS MODEL COMPACT CONV:', _rms_model)
        # if _rms_model < 1e-6:
        #     rms_model = mad_std(compact_model[compact_model>1e-6])
        # else:
        #     rms_model = _rms_model


        iterations = 2
        rms_model = mad_std(compact_model) + rms_std_res
        rms_compact_conv = rms_bkg_conv # * len(comp_ids)
        _, mask_region_conv_comp = mask_dilation(compact_model,
                                        rms=rms_model,
                                        # rms=rms_compact_conv,
                                        sigma=5.0,
                                        dilation_size=get_dilation_size(crop_image),
                                        # dilation_size=2,
                                        iterations=2,
                                        PLOT=PLOT,
                                        special_name=' compact conv ')#*mask_region_i



        # print('++++ Computing properties of convolved compact model.')

        # if np.sum(mask * mask_region_conv_comp) < np.sum(mask_region_conv_comp):
        #     _mask = mask_dilated_new * mask_region_conv_comp
        # else:
        #     _mask = mask


        results_compact_conv_morpho, _, _ = \
            measures(imagename=crop_image,
                     residualname=crop_residual, z=z,
                     sigma_mask=5.0,
                     last_level=1.0, vmin_factor=1.0,
                     data_2D=compact_model,
                     dilation_size=None,
                     results_final={},
                     rms=rms_model,
                    #  rms=rms_compact_conv,
                     apply_mask=False, do_PLOT=do_PLOT, SAVE=SAVE,
                     do_petro = True,
                     show_figure=show_figure,
                     mask_component=mask_region_conv_comp,
                     mask=mask_region, do_measurements='partial',
                     add_save_name='_compact_conv',verbose=verbose)

        list_results_compact_conv_morpho.append(results_compact_conv_morpho)



        # _rms_model = mad_std(compact_model_deconv)
        # print('**************************')
        # print('**************************')
        # print('RMS MODEL COMPACT DECONV:', _rms_model)
        # if _rms_model < 1e-6:
        #     rms_model = mad_std(compact_model_deconv[compact_model_deconv>1e-6])
        # else:
        #     rms_model = _rms_model

        rms_model = mad_std(compact_model_deconv) + rms_std_res
        rms_compact_deconv = rms_bkg_deconv # / len(comp_ids)
        _, mask_region_deconv_comp = mask_dilation(compact_model_deconv,
                                        rms=rms_model,
                                        # rms=rms_compact_deconv,
                                        sigma=5.0,
                                        dilation_size=2,
                                        # dilation_size=2,
                                        iterations=2,
                                        PLOT=PLOT,
                                        special_name=' compact deconv ')#*mask_region_i

        # if np.sum(mask * mask_region_deconv_comp) < np.sum(mask_region_deconv_comp):
        #     _mask = mask_dilated_new * mask_region_deconv_comp
        # else:
        #     _mask = mask

        # print('++++ Computing properties of deconvolved compact model.')
        try:
            results_compact_deconv_morpho, _, _ = \
                measures(imagename=crop_image,
                         residualname=crop_residual, z=z,
                         sigma_mask=5.0,
                         last_level=1.0, vmin_factor=1.0,
                         data_2D=compact_model_deconv,
                         dilation_size=None,
                         results_final={},
                         rms=rms_model,
                        #  rms=rms_compact_deconv,
                         apply_mask=False, do_PLOT=do_PLOT, SAVE=SAVE,
                         do_petro = True,
                         show_figure=show_figure,
                         mask_component=mask_region_deconv_comp,
                         mask=mask_region, do_measurements='partial',
                         add_save_name='_compact_deconv')

            list_results_compact_deconv_morpho.append(results_compact_deconv_morpho)
        except:
            empty_results = {key: np.nan for key in results_compact_conv_morpho.keys()}
            list_results_compact_deconv_morpho.append(empty_results)

        if nfunctions == 1:
            """
            Consider that the single component fitted represents a 
            compact component. Hence, extended emission is considered
            to be only the residual after removing that component. 
            """
            try:
                results_ext_conv_morpho, _, _ = \
                    measures(imagename=crop_image,
                             residualname=crop_residual, z=z,
                             sigma_mask=6.0,
                             last_level=1.5, vmin_factor=1.0,
                             data_2D=(load_fits_data(crop_image) - compact_model) * mask_region,
                             dilation_size=None,
                             results_final={},
                             rms=rms_std_res,
                             apply_mask=False, do_PLOT=do_PLOT, SAVE=SAVE,
                             do_petro=False,
                             show_figure=show_figure,
                             # mask_component=mask_region_deconv_comp,
                             mask=mask, do_measurements='partial',
                             add_save_name='_extended_conv')

                list_results_ext_conv_morpho.append(results_ext_conv_morpho)
                results_ext_deconv_morpho = results_ext_conv_morpho
                list_results_ext_deconv_morpho.append(results_ext_deconv_morpho)
            except:
                empty_results = {key: np.nan for key in results_compact_conv_morpho.keys()}
                list_results_ext_conv_morpho.append(empty_results)
                list_results_ext_deconv_morpho.append(empty_results)


        else:
            rms_model = mad_std(extended_model) + rms_std_res
            rms_ext_conv = rms_bkg_conv # * len(ext_ids)
            # if _rms_model < 1e-6:
            #     rms_model = mad_std(extended_model[extended_model>1e-6])
            # else:
            #     rms_model = _rms_model
            _, mask_region_conv_ext = mask_dilation(extended_model,
                                rms=rms_model,
                                # rms=rms_ext_conv,
                                sigma=5.0,
                                dilation_size=get_dilation_size(crop_image),
                                # dilation_size=2,
                                iterations = 2,
                                PLOT=PLOT,
                                special_name=' extended conv ')#*mask_region_i
            # print('++++ Computing properties of convolved extended model.')

            # if np.sum(mask * mask_region_conv_ext) < np.sum(mask_region_conv_ext):
            #     _mask = mask_dilated_new * mask_region_conv_ext
            # else:
            #     _mask = mask
            try:
                results_ext_conv_morpho, _, _ = \
                    measures(imagename=crop_image,
                             residualname=crop_residual, z=z,
                             sigma_mask=6.0,
                             last_level=1.5, vmin_factor=1.0,
                             data_2D=(load_fits_data(crop_image) - compact_model) * mask_region,
                             dilation_size=None,
                             results_final={},
                             rms=rms_std_res,
                             apply_mask=False, do_PLOT=do_PLOT, SAVE=SAVE,
                             do_petro=True,
                             show_figure=show_figure,
                            #  mask_component=mask_region_deconv_comp,
                             mask=mask_region, do_measurements='partial',
                             add_save_name='_extended_conv')
                # results_ext_conv_morpho, _,_ = \
                #     measures(imagename=crop_image,
                #              residualname=crop_residual, z=z,
                #              sigma_mask=5.0,
                #              last_level=1.5, vmin_factor=1.0,
                #              data_2D=extended_model,
                #              dilation_size=None,
                #              results_final={},
                #              rms=rms_model,
                #             #  rms=rms_ext_conv,
                #              apply_mask=False, do_PLOT=do_PLOT, SAVE=SAVE,
                #              do_petro=True,
                #              show_figure=show_figure,
                #              mask_component=mask_region_conv_ext,
                #              mask=mask_region, do_measurements='partial',
                #              add_save_name='_extended_conv')
                list_results_ext_conv_morpho.append(results_ext_conv_morpho)
            except:
                empty_results = {key: np.nan for key in results_compact_conv_morpho.keys()}
                list_results_ext_conv_morpho.append(empty_results)

            rms_model = mad_std(extended_model_deconv) + rms_std_res
            rms_ext_deconv = rms_bkg_deconv # / len(ext_ids)
            # print('**************************')
            # print('**************************')
            # print('RMS MODEL EXTENDED DECONV:', _rms_model)
            # if _rms_model < 1e-6:
            #     rms_model = mad_std(extended_model_deconv[extended_model_deconv>1e-6])
            # else:
            #     rms_model = _rms_model

            _, mask_region_deconv_ext = mask_dilation(extended_model_deconv,
                                rms=rms_model,
                                # rms = rms_ext_deconv,
                                sigma=5.0,
                                dilation_size=get_dilation_size(crop_image),
                                # dilation_size=2,
                                iterations = 2,
                                PLOT=PLOT,
                                special_name=' extended deconv ')#*mask_region_i

            # if np.sum(mask * mask_region_deconv_ext) < np.sum(mask_region_deconv_ext):
            #     _mask = mask_dilated_new * mask_region_deconv_ext
            # else:
            #     _mask = mask

            try:
                # print('++++ Computing properties of deconvolved extended model.')
                results_ext_deconv_morpho, _,_ = \
                    measures(imagename=crop_image,
                             residualname=crop_residual, z=z,
                             sigma_mask=5.0,
                             last_level=1.0, vmin_factor=1.0,
                             data_2D=extended_model_deconv,
                             dilation_size=None,
                             results_final={},
                             rms=rms_model,
                            #  rms=rms_ext_deconv,
                             apply_mask=False, do_PLOT=do_PLOT, SAVE=SAVE,
                             do_petro=True,
                             show_figure=show_figure,
                             mask_component=mask_region_deconv_ext,
                             mask=mask_region, do_measurements='partial',
                             add_save_name='_extended_deconv')

                list_results_ext_deconv_morpho.append(results_ext_deconv_morpho)
            except:
                empty_results = {key: np.nan for key in results_compact_conv_morpho.keys()}
                list_results_ext_deconv_morpho.append(empty_results)

        all_results = {**parameter_results, **decomp_results}
        results_fit.append(all_results)

        # except:
            # print('Error on fitting', os.path.basename(crop_image))
            # errors_fit.append(crop_image)

    return (pd.DataFrame(results_fit),result_mini,mini,
            lmfit_results, lmfit_results_1st_pass, errors_fit, models,
            pd.DataFrame(data_properties_list),
            pd.DataFrame(list_results_compact_conv_morpho),
            pd.DataFrame(list_results_compact_deconv_morpho),
            pd.DataFrame(list_results_ext_conv_morpho),
            pd.DataFrame(list_results_ext_deconv_morpho),
            pd.DataFrame(list_individual_deconv_props[0]).T,
            pd.DataFrame(list_individual_conv_props[0]).T,
            image_results_conv, image_results_deconv,bkg_images,
            class_results,
            compact_model)




def interferometric_decomposition(image1, image2, image3=None,
                                  residual1=None, residual2=None, residual3=None,
                                  iterations=2,
                                  dilation_size=None, std_factor=1.0,sub_bkg=False,
                                  ref_mask=None,sigma=6):
    """
    Peform an interferometric image decomposition using the e-MERLIN and JVLA.

    Parameters
    ----------
    image1 : str
        Path to an e-MERLIN image.
    image2 : str
        Path to a combined image.
    image3 : str, optional
        Path to a JVLA image.
    residual1 : str, optional
        Path to the residual image of the e-MERLIN image.
    residual2 : str, optional
        Path to the residual image of the combined image.
    residual3 : str, optional
        Path to the residual image of the JVLA image.
    iterations : int, optional
        Number of iterations to perform the sigma masking.
    dilation_size : int, optional
        Size of the dilation kernel.
    std_factor : float, optional
        Factor to multiply the standard deviation of the residual image.
    ref_mask : array, optional
        Reference JVLA mask to be used in the masking process.

    """
    def fit_sigma(image1_data, image2_data):
        '''
        Functiont to optmize sigma masking.
        This minimizes the negative values when subtracting e-MERLIN and VLA images.
        '''

        def opt_sigma_old(params):
            """
            This the old implementation of the minimisation function.

            Note: This function works very well, and it is exact. However, it is very slow.
            """
            sigma_level = params['sigma_level']
            # print('Current sigma is:', sigma_level)
            gomask, gmask = mask_dilation(g, cell_size=None,
                                          iterations=iterations,
                                          sigma=sigma_level,
                                          dilation_size=dilation_size_i,
                                          PLOT=False)

            I1mask = gmask * g  # - convolve_fft(gmask,kernel)
            I1mask_name = image_cut_i.replace('.fits', '_I1mask.fits')
            pf.writeto(I1mask_name,
                       I1mask, overwrite=True)

            copy_header(image_cut_i, I1mask_name, I1mask_name)

            M12 = convolve_2D_smooth(I1mask_name,
                                     mode='transfer',
                                     imagename2=image_cut_j,
                                     add_prefix='_M12')
            R12_ = g_next - load_fits_data(M12)  # + np.mean(g[g>3*mad_std(g)])
            #             residual_mask = R12_ - std_factor*std_level
            residual_mask = R12_ - std_factor * np.std(R12_)
            return (residual_mask)

        def opt_sigma(params):
            '''
            Improved version, do not use casa, so do not need to write and read files.
            Using convolve_fft seems to give **almost** the same results as CASA's
            function smooth.

            ** There is a small difference in the relative amplitude when convolving image2 with
            the beam of image1. The factor is exactly the same as the ratio of the beam areas:
            beam_area2(image2) / beam_area2(image1)
            '''
            sigma_level = params['sigma_level']
            # std_factor = params['std_factor']
            # print(f"Current opt values = sigma={sigma_level.value}, std_factor={std_factor.value}")
            print(f"Current opt values = sigma={sigma_level.value}")
            _, dilated_mask = mask_dilation(image1_data, cell_size=None,
                                                 iterations=2,
                                          sigma=sigma_level,
                                          dilation_size=dilation_size_i,
                                          PLOT=False)
            masked_image = dilated_mask * image1_data * ref_mask
            M12 = _fftconvolve_jax(masked_image, PSF_DATA_j)
            # M12 = scipy.signal.fftconvolve(masked_image, PSF_DATA, mode='same')
            mask_M12 = (M12 > 1e-6)
            _R12 = (image2_data - M12 + offset_2) * ref_mask
            # R12 = _R12 + abs(jnp.nanmedian(_R12[_R12<0]))
            # R12 = (image2_data - (M12 + 0*std_factor * bkg_2))*ref_mask
            # residual_mask = 1000*(R12_ - std_factor * std_level) * ref_mask  # *mask_M12#avoid
            return (np.array(_R12).copy())

        bounds_i, bounds_f = 6, 100
        sigma_i, sigma_f = bounds_i, bounds_f
        std_fac_bounds_i, std_fac_bounds_f = 0.99, 1.01
        std_fac_i, std_fac_f = std_fac_bounds_i, std_fac_bounds_f
        pars0 = 6.0, 1.0
        sigma_0, std_fac_0 = pars0
        fit_params = lmfit.Parameters()
        fit_params.add("sigma_level", value=sigma_0, min=sigma_i, max=sigma_f)
        # fit_params.add("std_factor", value=std_fac_0, min=std_fac_i, max=std_fac_f)
        #         fit_params.add("offset", value=1.0, min=0.5, max=2)
        solver_method = 'nelder'
        #         mini = lmfit.Minimizer(opt_sigma,fit_params,max_nfev=5000,nan_policy='omit',reduce_fcn='neglogcauchy')
        mini = lmfit.Minimizer(opt_sigma, fit_params, max_nfev=50000,
                               nan_policy='omit', reduce_fcn='neglogcauchy')
        result_1 = mini.minimize(method=solver_method,tol=1e-12,
                                 options={'xatol': 1e-12,
                                          'fatol': 1e-8,
                                          'adaptive': True})

        result = mini.minimize(method=solver_method, params=result_1.params,
                               tol=1e-8,
                               options={'xatol': 1e-12,
                                        'fatol': 1e-12,
                                        'adaptive': True})
        # result_1 = mini.minimize(method='least_squares',
        #                          tr_solver="exact",
        #                          tr_options={'regularize': True},
        #                          # x_scale='jac', loss="cauchy",
        #                          ftol=1e-14, xtol=1e-14, gtol=1e-14,
        #                          f_scale=1,
        #                          max_nfev=200000,
        #                          verbose=2)
        #
        #
        # result = mini.minimize(method='least_squares',
        #                        params=result_1.params,
        #                        tr_solver="exact",
        #                        tr_options={'regularize': True},
        #                        # x_scale='jac', loss="cauchy",
        #                        ftol=1e-14, xtol=1e-14, gtol=1e-14,
        #                        f_scale=1,
        #                        max_nfev=200000,
        #                        verbose=2)

        # result_1 = mini.minimize(method='differential_evolution',
        #                          # tr_solver="exact",
        #                          # tr_options={'regularize': False},
        #                          # # x_scale='jac', loss="cauchy",
        #                          # ftol=1e-14, xtol=1e-14, gtol=1e-14,
        #                          # f_scale=1,
        #                          # max_nfev=200000,
        #                          verbose=2)
        #
        #
        # result = mini.minimize(method='differential_evolution',
        #                        params=result_1.params,
        #                        # tr_solver="exact",
        #                        # tr_options={'regularize': False},
        #                        # # x_scale='jac', loss="cauchy",
        #                        # ftol=1e-14, xtol=1e-14, gtol=1e-14,
        #                        # f_scale=1,
        #                        # max_nfev=200000,
        #                        verbose=2)
        #         result = mini.minimize(method=solver_method,params=result_1.params,
        #                                  max_nfev=30000, #x_scale='jac',  # f_scale=0.5,
        #                                  tr_solver="exact",
        #                                  tr_options={'regularize': True},
        #                                  ftol=1e-14, xtol=1e-14, gtol=1e-14, verbose=2)

        parFit = result.params['sigma_level'].value #, result.params['std_factor'].value
        Err_parFit = result.params['sigma_level'].stderr #, result.params['std_factor'].stderr
        # resFit = result.residual
        # chisqr = result.chisqr
        # redchi = result.redchi
        # pars = parFit
        # sigma_opt_fit, std_factor_opt_fit = pars
        return (parFit, Err_parFit, result)

    # set image names, they must have the same cell_size,image size AND MUST BE ALIGNED!!!
    image_cut_i = image1  # highest resolution image
    image_cut_j = image2  # intermediate resolution image

    # read data files
    image1_data = load_fits_data(image_cut_i)
    image2_data = load_fits_data(image_cut_j)

    if sub_bkg:
        if residual2 is not None:
            bkg_2 = load_fits_data(residual2)
            offset_2 = 0.5 * bkg_2
        else:
            bkg_1 = sep_background(image_cut_i, apply_mask=False,
                                   show_map=False,use_beam_fraction=True).back()
            bkg_2 = sep_background(image_cut_j, apply_mask=False,
                                   show_map=False,use_beam_fraction=True).back()
            offset_2 = 0.5 * bkg_2

    else:
        if residual2 is not None:
            bkg_2 = load_fits_data(residual2)
            offset_2 = 0.5 * bkg_2
        else:
            bkg_1 = mad_std(image1_data)
            bkg_2 = mad_std(image2_data)
            offset_2 = 0.5*np.std(image2_data)


    psf_name_j = tcreate_beam_psf(image_cut_j,
                                  size=image1_data.shape,
                                  aspect=None)

    correction_factor_ij = beam_area2(image_cut_j) / beam_area2(image_cut_i)
    PSF_DATA_j = load_fits_data(psf_name_j)*correction_factor_ij  # this will result similar results as

    omaj_i, omin_i, _, _, _ = beam_shape(image_cut_i)
    dilation_size_i = int(np.sqrt(omaj_i * omin_i) / (2 * get_cell_size(image_cut_i)))
    print('Usig mask dilution size of (for image i)=', dilation_size_i)

    results = {}
    results_short = {}

    results['imagename_i'] = os.path.basename(image_cut_i)
    results['imagename_j'] = os.path.basename(image_cut_j)

    parFit, Err_parFit, result_mini = fit_sigma(image1_data, image2_data)
    print('Optmized Sigma=', parFit, '+/-', Err_parFit)
    # sigma_level, std_factor = parFit
    sigma_level = parFit

    #Dilate the mask referent to image1 (high-res image).
    I01mask, I1mask = mask_dilation(image1_data, cell_size=get_cell_size(image_cut_i),
                                    iterations=iterations,
                                    sigma=sigma_level,
                                    dilation_size=dilation_size_i, PLOT=True)

    I1mask_data = I1mask * image1_data * ref_mask # - convolve_fft(gmask,kernel)
    # gg = gmask - gaussian_filter(gmask,150)
    I1mask_name = image_cut_i.replace('.fits', '_I1mask.fits')
    pf.writeto(I1mask_name, I1mask_data, overwrite=True)
    copy_header(image_cut_i, I1mask_name, I1mask_name)

    pf.writeto(image_cut_i.replace('.fits', '_mask_bool.fits'), I1mask * 1.0,
               overwrite=True)
    copy_header(image_cut_i, image_cut_i.replace('.fits', '_mask_bool.fits'),
                image_cut_i.replace('.fits', '_mask_bool.fits'))


    M12_data = scipy.signal.fftconvolve(I1mask_data, PSF_DATA_j, mode='same') #+offset_2
    M12 = image_cut_j.replace('.fits', '_M12opt.fits')
    pf.writeto(M12, M12_data, overwrite=True)
    copy_header(image_cut_j, M12, M12)

    # subtract image2 from theta2*image1_mask and save images
    R12_ = load_fits_data(image_cut_j) - M12_data # + 1*
    # std_factor*std_level# std_factor*std_level# + np.mean(g[g>3*mad_std(g)])
    R12 = image_cut_j.replace('.fits', '_R12opt.fits')
    pf.writeto(R12, R12_, overwrite=True)
    copy_header(image_cut_j, R12, R12)
    ###################################
    ###################################
    ###################################
    std_1 = mad_std(load_fits_data(residual1))
    std_2 = mad_std(load_fits_data(residual2))
    _, I1mask = mask_dilation(image1, PLOT=False, dilation_type='disk', rms=std_1,
                                     sigma=result_mini.params['sigma_level'].value,
                                     iterations=2, dilation_size=None)
    _, mask_I1 = mask_dilation(image1, PLOT=False, dilation_type='disk', rms=std_1,
                                     sigma=6.0,
                                     iterations=2, dilation_size=None)
    _, mask_I2 = mask_dilation(image2, PLOT=False, dilation_type='disk', rms=std_2,
                                     sigma=6.0,
                                     iterations=2, dilation_size=None)
    _, mask_M12 = mask_dilation(M12, PLOT=False, dilation_type='disk', rms=std_2,
                                      sigma=6, iterations=2, dilation_size=None)
    _, mask_R12 = mask_dilation(R12, PLOT=False, dilation_type='disk', rms=std_2,
                                      sigma=6, iterations=2, dilation_size=None)

    results['I1_name'] = image1
    results['I2_name'] = image2
    results['R12_name'] = R12
    results['M12_name'] = M12

    print(f"  ++==>> Computing image statistics on Image1...")
    I1_props = compute_image_properties(img=image1,
                                        residual=residual1,
                                        rms=std_1,
                                        apply_mask=False,
                                        show_figure=False,
                                        mask=mask_I1,
                                        last_level=2.0)[-1]

    print(f"  ++==>> Computing image statistics on Image-Mask1...")
    I1mask_props = compute_image_properties(img=I1mask_name,
                                            residual=residual1,
                                            rms=std_1,
                                            apply_mask=False,
                                            show_figure=False,
                                            mask=I1mask,
                                            last_level=2.0)[-1]

    print(f"  ++==>> Computing image statistics on Image2...")
    I2_props = compute_image_properties(img=image2,
                                        residual=residual2,
                                        rms=std_2,
                                        apply_mask=False,
                                        show_figure=False,
                                        mask=mask_I2,
                                        last_level=2.0)[-1]

    print(f"  ++==>> Computing image statistics on R12...")
    R12_props = compute_image_properties(img=R12,
                                         residual=residual2,
                                         rms=std_2,
                                         apply_mask=False,
                                         show_figure=False,
                                         mask=mask_R12,
                                         last_level=2.0)[-1]
    print(f"  ++==>> Computing image statistics on M12...")
    M12_props = compute_image_properties(img=M12,
                                         residual=residual2,
                                         rms=std_2,
                                         show_figure=False,
                                         apply_mask=False,
                                         mask=mask_M12,
                                         last_level=2.0)[-1]

    """
    Store measured properties:
        - total flux density on: I1, I1mask, R12, M12 and I2
        - peak brightness on: I1, I1mask, R12, M12 and I2
        - sizes on: I1, I1mask, R12, M12 and I2
    """

    results['S_I1'] = I1_props['total_flux_mask']
    results['S_I1mask'] = I1mask_props['total_flux_mask']
    results['S_R12'] = R12_props['total_flux_mask']
    results['S_M12'] = M12_props['total_flux_mask']
    results['S_I2'] = I2_props['total_flux_mask']

    results['Speak_I1'] = I1_props['peak_of_flux']
    results['Speak_I1mask'] = I1mask_props['peak_of_flux']
    results['Speak_R12'] = R12_props['peak_of_flux']
    results['Speak_M12'] = M12_props['peak_of_flux']
    results['Speak_I2'] = I2_props['peak_of_flux']

    results['C50radii_I1'] = I1_props['C50radii']
    results['C50radii_I1mask'] = I1mask_props['C50radii']
    results['C50radii_R12'] = R12_props['C50radii']
    results['C50radii_M12'] = M12_props['C50radii']
    results['C50radii_I2'] = I2_props['C50radii']

    results['C95radii_I1'] = I1_props['C95radii']
    results['C95radii_I1mask'] = I1mask_props['C95radii']
    results['C95radii_R12'] = R12_props['C95radii']
    results['C95radii_M12'] = M12_props['C95radii']
    results['C95radii_I2'] = I2_props['C95radii']


    results['diff_SI2_SI1'] = results['S_I2'] - results['S_I1']
    results['ratio_SI1_SI2'] = results['S_I1'] / results['S_I2']
    results['diff_SpeakI2_SpeakI1'] = results['Speak_I2'] - results['Speak_I1']
    results['ratio_SpeakI1_SpeakI2'] = results['Speak_I1'] / results['Speak_I2']

    # omask_i, mask_i = mask_dilation(image_cut_i,
    #                                 cell_size=get_cell_size(image_cut_i),
    #                                 sigma=sigma, iterations=2,
    #                                 dilation_size=dilation_size_i, PLOT=True)

    omaj_j, omin_j, _, _, _ = beam_shape(image_cut_j)
    dilation_size_j = int(
        np.sqrt(omaj_j * omin_j) / (2 * get_cell_size(image_cut_j)))
    print('Usig mask dilution size of (for image j)=', dilation_size_j)

    omask_j, mask_j = mask_dilation(image_cut_j,
                                    cell_size=get_cell_size(image_cut_j),
                                    iterations=2, sigma=sigma,
                                    dilation_size=dilation_size_j, PLOT=False)

    plot_interferometric_decomposition(I1mask_name, image_cut_j,
                                       M12, R12,
                                       vmax_factor=2.0, vmin_factor=0.5,
                                       run_phase='1st',
                                       crop=True, NAME=image_cut_i,
                                       SPECIAL_NAME='_I1_2_M12_R12')

    print('####################################################')
    print('----------------- **** REPORT **** -----------------')
    print('####################################################')
    print(f"Flux Density I1 (high-res)              = {results['S_I1'] * 1000:.2f} mJy")
    print(f"Flux Density I1mask (high-res compact)  = {results['S_I1mask'] * 1000:.2f} mJy")
    print(f"Flux Density M12 (mid-res compact conv) = {results['S_M12'] * 1000:.2f} mJy")
    print(f"Flux Density I2 (mid-res)               = {results['S_I2'] * 1000:.2f} mJy")
    print(f"Flux Density R12 (mid-res extended)     = {results['S_R12'] * 1000:.2f} mJy")
    print('------------------ ************** ------------------')
    print(f"Diff Flux Density I2-I1                 = {results['diff_SI2_SI1'] * 1000:.2f} mJy")
    print(f"Ratio Flux Density I1/I2                = {results['ratio_SI1_SI2']:.2f}")
    print(f"Diff Peak SpeakI2 - SpeakI1             = "
          f"{results['diff_SpeakI2_SpeakI1']*1000:.2f} mJy/beam")
    print(f"Ratio Peak SpeakI1 / SpeakI2            = "
          f"{results['ratio_SpeakI1_SpeakI2']:.2f}")
    print('####################################################')

    if image3 is not None:

        def image_sum(image_cut_k, M13, M23,offset_3):
            def opt_res(params):
                # print(f"Params: a={params['a'].value},"
                #       f"b={params['b'].value},"
                #       f"c={params['c'].value}")
                optimisation = (params['a'] * M23_data + params['b'] * M13_data + params['c'] *
                                offset_3)
                I3ext = (image3_data - optimisation)
                #     I3ext = image_data - (params['a']*M23_data-params['c']*zero_off)- (params['b']*M13_data-params['d']*zero_off)
                # return (I3ext+abs(np.nanmin(I3ext)))
                return (I3ext)

            def opt_sub_3(image_data, M23_data, M13_data, bkg_3):
                bounds_i = 0.1, 0.1, -1.0  # ,-3.0
                bounds_f = 10.01, 10.01, 1.0  # ,+3.0
                pars0 = 1.0, 1.0, 0.5  # ,0.0
                ai, bi, ci = bounds_i
                af, bf, cf = bounds_f
                a0, b0, c0 = pars0
                fit_params = lmfit.Parameters()
                fit_params.add("a", value=a0, min=ai, max=af)
                fit_params.add("b", value=b0, min=bi, max=bf)
                fit_params.add("c", value=c0, min=ci, max=cf)
                #     fit_params.add("d", value=d0, min=di, max=df)
                solver_method = 'least_squares'
                mini = lmfit.Minimizer(opt_res, fit_params, max_nfev=50000,
                                       nan_policy='omit', reduce_fcn='neglogcauchy')

                result_1 = mini.minimize(method='nelder', tol=1e-8,
                                         options={'xatol': 1e-8,
                                                  'fatol': 1e-8,
                                                  'adaptive': True})

                result = mini.minimize(method='nelder', params=result_1.params,
                                       tol=1e-8,
                                       options={'xatol': 1e-8,
                                                'fatol': 1e-8,
                                                'adaptive': True})

                #             result_1 = mini.minimize(method=solver_method)
                # result_1 = mini.minimize(method=solver_method,
                #                          tr_solver="exact",
                #                          tr_options={'regularize': True},
                #                          x_scale='jac', loss="cauchy",
                #                          ftol=1e-12, xtol=1e-12, gtol=1e-12,
                #                          f_scale=1.0,
                #                          max_nfev=5000, verbose=2)
                # result = mini.minimize(method=solver_method,
                #                        params=result_1.params,
                #                        tr_solver="exact",
                #                        tr_options={'regularize': True},
                #                        x_scale='jac', loss="cauchy",
                #                        ftol=1e-12, xtol=1e-12, gtol=1e-12,
                #                        f_scale=1.0,
                #                        max_nfev=5000, verbose=2)

                parFit = result.params['a'].value, result.params['b'].value, \
                    result.params['c'].value  # ,result.params['d'].value
                Err_parFit = result.params['a'].stderr, result.params['b'].stderr, \
                    result.params['c'].stderr  # ,result.params['d'].stderr
                # resFit = result.residual
                # chisqr = result.chisqr
                # redchi = result.redchi
                # pars = parFit
                # res_opt = pars
                return (result, parFit, Err_parFit)

            image3_data = mask_k * load_fits_data(image_cut_k)
            M13_data = load_fits_data(M13)
            M23_data = load_fits_data(M23)

            result_mini_I3, parFit, Err_parFit = \
                opt_sub_3(image3_data, M23_data, M13_data, offset_3)
            a, b, c = parFit
            a_err, b_err, c_err = Err_parFit
            print(f"Linear Image Combination Optmisation: "
                  f"        (a,b,c)={parFit}+/-{Err_parFit}")

            model_total = a * load_fits_data(M23) + b * load_fits_data(M13) + 1 * c * offset_3
            M123 = image_cut_j.replace('.fits', '_M123_opt.fits')
            pf.writeto(M123, model_total, overwrite=True)
            copy_header(image_cut_k, M123, M123)

            model_comp = 0 * a * load_fits_data(M23) + b * load_fits_data(M13) + 0 * c * offset_3 / 2
            Mcomp = image_cut_j.replace('.fits', '_M13_opt.fits')
            pf.writeto(Mcomp, model_comp, overwrite=True)
            copy_header(image_cut_k, Mcomp, Mcomp)

            model_extended = a * load_fits_data(M23)# + c * bkg_3 / 2
            Mext = image_cut_j.replace('.fits', '_M23_opt.fits')
            pf.writeto(Mext, model_extended, overwrite=True)
            copy_header(image_cut_k, Mext, Mext)

            I3re = load_fits_data(image_cut_k) - (a * load_fits_data(M23) + b * load_fits_data(M13) + 0 * c * offset_3)
            I3_RT = image_cut_j.replace('.fits', '_RT.fits')
            I3ext = load_fits_data(image_cut_k) - (0 * a * load_fits_data(M23) + b * load_fits_data(M13) +  0 * c * offset_3 / 2)
            I3ext_name = image_cut_j.replace('.fits', '_residual_extended.fits')
            I3comp = load_fits_data(image_cut_k) - (a * load_fits_data(M23) + 0 * b * load_fits_data(M13) + 0 * c * offset_3 / 2)
            I3comp_name = image_cut_j.replace('.fits', '_residual_comp.fits')

            pf.writeto(I3_RT, I3re, overwrite=True)
            pf.writeto(I3ext_name, I3ext, overwrite=True)
            pf.writeto(I3comp_name, I3comp, overwrite=True)
            copy_header(image_cut_k, I3_RT, I3_RT)
            copy_header(image_cut_k, I3ext_name, I3ext_name)
            copy_header(image_cut_k, I3comp_name, I3comp_name)

            return (M123, Mcomp, Mext, I3_RT, I3ext_name, I3comp_name, result_mini_I3, parFit, Err_parFit)


        print('Running decomposition for Image3...')
        image_cut_k = image3  # lowest resolution image
        results['imagename_k'] = os.path.basename(image_cut_k)

        omaj_k, omin_k, _, _, _ = beam_shape(image_cut_k)
        dilation_size_k = int(
            np.sqrt(omaj_k * omin_k) / (2 * get_cell_size(image_cut_k)))
        print('Usig mask dilution size of (for image k)=', dilation_size_k)

        omask_k, mask_k = mask_dilation(image_cut_k,
                                        cell_size=get_cell_size(image_cut_k),
                                        sigma=6, iterations=2,
                                        dilation_size=dilation_size_k,
                                        PLOT=False)

        image3_data = load_fits_data(image_cut_k)
        bak = beam_area2(image_cut_k, cellsize=get_cell_size(image_cut_k))
        #         image_cut_k = imagelist[k]#cut_image2(imagelist[k],size=(1024,1024))

        psf_name_k = tcreate_beam_psf(image_cut_j,
                                      size=image1_data.shape,
                                      aspect=None)

        correction_factor_jk = beam_area2(image_cut_k) / beam_area2(image_cut_j)
        correction_factor_ik = beam_area2(image_cut_k) / beam_area2(image_cut_i)
        PSF_DATA_k = load_fits_data(psf_name_k)  # this will result similar results as

        if sub_bkg:
            bkg_3 = sep_background(image_cut_k, apply_mask=False,
                                   show_map=False, use_beam_fraction=True).back()
            offset_3 = 0.5 * bkg_3

        else:
            if residual3 is not None:
                bkg_3 = load_fits_data(residual3)
                offset_3 = 0.5 * bkg_3
            else:
                bkg_3 = mad_std(image3_data)
                offset_3 = 0.5 * bkg_3


        #read the result of R12, but without the offest component!
        R12_data = load_fits_data(R12)# - offset_2
        R12_mask = mask_j  # (R12_data>1*mad_std(R12_data))
        # gg = gmask - convolve_fft(gmask,kernel)
        R12_data_mask = R12_mask * R12_data  # - gaussian_filter(gmask,10)

        #Generate new file for a masked region of R12.
        R12mask = image_cut_j.replace('.fits', '_R12_mask.fits')
        pf.writeto(R12mask, R12_data_mask, overwrite=True)
        copy_header(image_cut_j, R12mask, R12mask)

        # Convolve R12mask with the beam of image3.
        M23_data = scipy.signal.fftconvolve(R12_data_mask,
                                            PSF_DATA_k*correction_factor_jk,
                                            mode='same')
        M23 = image_cut_j.replace('.fits', '_M23.fits')
        pf.writeto(M23, M23_data, overwrite=True)
        copy_header(image_cut_k, M23, M23)

        # # Convolving I1mask with the beam of image3.
        # M13_data = scipy.signal.fftconvolve(I1mask_data,
        #                                     PSF_DATA_k*correction_factor_ik,
        #                                     mode='same')
        # M13 = image_cut_k.replace('.fits', '_M13.fits')
        # pf.writeto(M13, M13_data, overwrite=True)
        # copy_header(image_cut_k, M13, M13)

        # Convolving I1mask with the beam of image3.
        M13_data = scipy.signal.fftconvolve(M12_data,
                                            PSF_DATA_k*correction_factor_jk,
                                            mode='same')
        M13 = image_cut_k.replace('.fits', '_M13.fits')
        pf.writeto(M13, M13_data, overwrite=True)
        copy_header(image_cut_k, M13, M13)

        M123_opt, M13_opt, M23_opt, I3_RT, \
            I3ext_name, I3comp_name, result_mini_I3, parFit, Err_parFit = \
            image_sum(image_cut_k, M13, M23,offset_3)

        a, b, c = parFit
        a_err, b_err, c_err = Err_parFit

        results['I3_name'] = image3
        results['M23_name'] = M23_opt
        results['M13_name'] = M13_opt
        results['M123_name'] = M123_opt
        results['I3ext_name'] = I3ext_name
        results['I3_RT_name'] = I3_RT

        results['a'] = a
        results['b'] = b
        results['c'] = c
        results['a_err'] = a_err
        results['b_err'] = b_err
        results['c_err'] = c_err

        std_3 = mad_std(load_fits_data(residual3))

        _, mask_M13 = mask_dilation(M13_opt, PLOT=False, dilation_type='disk', rms=std_3,
                                    sigma=6.0, iterations=2, dilation_size=None)
        _, mask_I3 = mask_dilation(image3, PLOT=False, dilation_type='disk', rms=std_3,
                                   sigma=6.0,
                                   iterations=2, dilation_size=None)
        _, mask_I3ext = mask_dilation(I3ext_name, PLOT=False, dilation_type='disk', rms=std_3,
                                   sigma=6.0,
                                   iterations=2, dilation_size=None)

        print(f"  ++==>> Computing image statistics on Image3...")
        I3_props = compute_image_properties(img=image3,
                                            residual=residual3,
                                            rms=std_3,
                                            apply_mask=False,
                                            show_figure=False,
                                            mask=mask_I3,
                                            last_level=2.0)[-1]
        print(f"  ++==>> Computing image statistics on extended emission...")
        I3ext_props = compute_image_properties(img=I3ext_name,
                                               residual=residual3,
                                               rms=std_3,
                                               apply_mask=False,
                                               show_figure=False,
                                               mask=mask_I3,
                                               last_level=2.0)[-1]

        print(f"  ++==>> Computing image statistics on M23...")
        M23_props = compute_image_properties(img=M23_opt,
                                               residual=residual3,
                                               rms=std_3,
                                               apply_mask=False,
                                               show_figure=False,
                                               mask=mask_I3,
                                               last_level=2.0)[-1]
        print(f"  ++==>> Computing image statistics on M13...")
        M13_props = compute_image_properties(img=M13_opt,
                                             residual=residual3,
                                             rms=std_3,
                                             apply_mask=False,
                                             show_figure=False,
                                             mask=mask_M13,
                                             last_level=2.0)[-1]
        print(f"  ++==>> Computing image statistics on residual RT...")
        I3_RT_props = compute_image_properties(img=I3_RT,
                                             residual=residual3,
                                             rms=std_3,
                                             apply_mask=False,
                                             show_figure=True,
                                             mask=mask_I3,
                                             last_level=2.0)[-1]

        results['S_I3'] = I3_props['total_flux_mask']
        results['S_I3ext'] = I3ext_props['total_flux_mask']
        results['S_M23'] = M23_props['total_flux_mask']
        results['S_M13'] = M13_props['total_flux_mask']
        results['S_I3RT'] = I3_RT_props['total_flux_mask']

        results['Speak_I3'] = I3_props['peak_of_flux']
        results['Speak_I3ext'] = I3ext_props['peak_of_flux']
        results['Speak_M23'] = M23_props['peak_of_flux']
        results['Speak_M13'] = M13_props['peak_of_flux']
        results['Speak_I3RT'] = I3_RT_props['peak_of_flux']

        results['C50radii_I3'] = I3_props['C50radii']
        results['C50radii_I3ext'] = I3ext_props['C50radii']
        results['C50radii_M23'] = M23_props['C50radii']
        results['C50radii_M13'] = M13_props['C50radii']
        results['C50radii_I3RT'] = I3_RT_props['C50radii']

        results['C95radii_I3'] = I3_props['C95radii']
        results['C95radii_I3ext'] = I3ext_props['C95radii']
        results['C95radii_M23'] = M23_props['C95radii']
        results['C95radii_M13'] = M13_props['C95radii']
        results['C95radii_I3RT'] = I3_RT_props['C95radii']

        results['diff_SI3_SI1'] = results['S_I3'] - results['S_I1']
        results['ratio_SI1_SI3'] = results['S_I1'] / results['S_I3']
        results['diff_SpeakI3_SpeakI1'] = results['Speak_I3'] - results['Speak_I1']
        results['ratio_SpeakI1_SpeakI3'] = results['Speak_I1'] / results['Speak_I3']

        print('####################################################')
        print('----------------- **** REPORT **** -----------------')
        print('####################################################')
        print(f"Flux Density I3 (low-res)             = {results['S_I3'] * 1000:.2f} mJy")
        print(f"Flux Density Comp M13 (low-res)       = {results['S_M13'] * 1000:.2f} mJy")
        print(f"Flux Density Ext M23 (low-res)        = {results['S_M23'] * 1000:.2f} mJy")
        print(f"Flux Density Extended (I3 low-res)    = {results['S_I3ext'] * 1000:.2f} mJy")
        print(f"Flux Density Residual (RT low-res)    = {results['S_I3RT'] * 1000:.2f} mJy")
        print('------------------ ************** ------------------')
        print(f"Diff Flux Density I3-I1                 = {results['diff_SI3_SI1'] * 1000:.2f} mJy")
        print(f"Ratio Flux Density I1/I3                = {results['ratio_SI1_SI3']:.2f}")
        print(f"Diff Peak SpeakI3 - SpeakI1             = "
              f"{results['diff_SpeakI3_SpeakI1'] * 1000:.2f} mJy/beam")
        print(f"Ratio Peak SpeakI1 / SpeakI3            = "
              f"{results['ratio_SpeakI1_SpeakI3']:.2f}")
        print('####################################################')

        #
        # results['S_I3_23_res_full_py'] = np.sum(I3_residual_23) / bak
        # results['S_I3_23_res_mask_py'] = np.sum(mask_k * I3_residual_23) / bak
        # results['a'] = a
        # results['b'] = b
        # results['c'] = c
        #
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(image3.replace('.fits', '_interf_decomposition.csv'),
                  header=True,
                  index=False)

        plot_interferometric_decomposition(R12mask, image_cut_k,
                                           M123_opt,  # M23,
                                           I3_RT,
                                           vmin0=mad_std(load_fits_data(image_cut_j)),
                                           vmax_factor=0.5, vmin_factor=2.0,
                                           run_phase='2nd',
                                           crop=False, box_size=512,
                                           NAME=image_cut_j,
                                           SPECIAL_NAME='_R12_I2_M123_I3re')

        plot_interferometric_decomposition(R12mask, image_cut_k, M13_opt,
                                           # M23,
                                           I3ext_name,
                                           vmax_factor=0.5, vmin_factor=2.0,
                                           crop=False, box_size=512,
                                           run_phase='compact',
                                           NAME=image_cut_j,
                                           SPECIAL_NAME='_R12_I3_Mcomp_I3ext')

    if image3 is None:
        return (result_mini, results, results_short, I1mask_data, I1mask_name, R12, M12)
    else:
        return (result_mini, result_mini_I3, results, I1mask, I1mask_name, R12, M12, M123_opt,
                M13_opt, M23_opt, I3_RT, I3ext_name)

image_decomposition = deprecated("image_decomposition",
                              "interferometric_decomposition")(interferometric_decomposition)

def perform_interferometric_decomposition(imagelist_em, imagelist_comb,
                                          imagelist_vla, residuallist_vla,
                                          idx_em, idx_vla, idxs_em, z,
                                          std_factor=0.5, ref_mask=None,
                                          sigma=6):

    """
    Perform interferometric decomposition of the images in the list.

    Parameters
    ----------
    imagelist_em : list
        List of e-merlin images.
    imagelist_comb : list
        List of combined images.
    imagelist_vla : list
        List of jvla images.
    residuallist_vla : list
        List of jvla residual images.
    idx_em : int
        Index of the e-merlin image.
    idx_vla : int
        Index of the jvla image.
    idxs_em : list
        List of indices of the e-merlin images.
    z : float
        Redshift of the source.
    std_factor : float; default = 0.5
        Factor to multiply the standard deviation of the image.
        Do not modify this parameter, unless you know what you are doing. See
        Fig. 5 in the paper for more details.
    ref_mask : array
        Reference mask.
    sigma : float; default = 6
        Sigma level for the mask dilation.

    """
    int_results = []
    I1_MASK_LIST = []
    R12_MASK_LIST = []
    R12_LIST = []
    M12_LIST = []
    M123_LIST = []
    M23_Mext_opt_LIST = []
    I3RE_RT_LIST = []
    M13_Mcomp_opt_LIST = []
    I3RE_LIST = []
    I3EXT_LIST = []
    IMAGELIST_ERROR = []
    imagelist_em_short = imagelist_em[idxs_em]
    rms = mad_std(load_fits_data(residuallist_vla[idx_vla]))
    EXTENDED_PROPS = []
    EXTENDED_2nd_PROPS = []
    COMP_EM_PROPS = []
    COMP_PROPS = []
    COMP_VLA_PROPS = []
    for l in range(0, len(imagelist_em_short)):
        for j in tqdm(range(0, len(imagelist_comb))):
            try:
                #                 i = idx_em # e-merlin image
                k = idx_vla  # almost pure jvla image, but needs to have the same cellsize as of the e-merlin one
                result_mini, results, results_short, Imask, I1mask_name, I1mask_2, R12, R12conv, M12, \
                    M123_opt, Mcomp_opt, Mext_opt, I3re_name, I3ext_name, I3comp_name, I3_residual_23_name = image_decomposition(
                    #                 image1 = imagelist_comb[2],
                    image1=imagelist_em_short[l],
                    image2=imagelist_comb[j],
                    image3=imagelist_vla[k],
                    std_factor=std_factor, dilation_size=None, iterations=2,
                    ref_mask=ref_mask, sigma=sigma
                )
                int_results.append(results)
                I1_MASK_LIST.append(I1mask_name)
                R12_MASK_LIST.append(R12conv)
                M12_LIST.append(M12)
                R12_LIST.append(R12)
                M123_LIST.append(M123_opt)
                M23_Mext_opt_LIST.append(Mext_opt)
                M13_Mcomp_opt_LIST.append(Mcomp_opt)
                I3RE_RT_LIST.append(I3re_name)
                I3EXT_LIST.append(I3ext_name)

                rms_i = mad_std(load_fits_data(imagelist_em_short[l]))
                rms_j = mad_std(load_fits_data(R12))

                _, mask_extended = mask_dilation(imagelist_comb[j],
                                                 # imagelist_vla[k],
                                                 sigma=sigma, iterations=2,
                                                 dilation_size=None, PLOT=True)
                #                 _, mask_extended_2nd = mask_dilation(I3ext_name,#imagelist_vla[k],
                #                              sigma=6, iterations=2,
                #                                 dilation_size=None, PLOT=True)

                _, mask_compact = mask_dilation(M12, rms=rms,
                                                # we have to give a rms for model-based images
                                                sigma=sigma, iterations=2,
                                                dilation_size=None, PLOT=True)
                results_extended = run_analysis_list([R12], residuallist_vla[k],
                                                     imagelist_vla[k],
                                                     z,
                                                     mask_extended, rms_j, sigma=sigma)
                results_extended_2nd = run_analysis_list([I3ext_name],
                                                         residuallist_vla[k],
                                                         imagelist_vla[k],
                                                         z,
                                                         ref_mask, rms, sigma=sigma)

                results_compact = run_analysis_list([M12], residuallist_vla[k],
                                                    imagelist_vla[k],
                                                    z,
                                                    mask_compact, rms_j, sigma=sigma)
                results_compact_vla = run_analysis_list([Mcomp_opt],
                                                        residuallist_vla[k],
                                                        imagelist_vla[k],
                                                        z,
                                                        ref_mask, rms, sigma=sigma)

                results_compact_EM = run_analysis_list([I1mask_name],
                                                       residuallist_vla[k],
                                                       imagelist_vla[k],
                                                       z,
                                                       mask_compact, rms_i, sigma=sigma)
                EXTENDED_PROPS.append(results_extended)
                EXTENDED_2nd_PROPS.append(results_extended_2nd)
                COMP_EM_PROPS.append(results_compact_EM)
                COMP_PROPS.append(results_compact)
                COMP_VLA_PROPS.append(results_compact_vla)

            except:
                print('Error on minimising image:', imagelist_comb[j])
                IMAGELIST_ERROR.append(imagelist_comb[j])
    return (
    int_results, EXTENDED_PROPS, EXTENDED_2nd_PROPS, COMP_PROPS, COMP_EM_PROPS,
    COMP_VLA_PROPS,I1_MASK_LIST,
    R12_MASK_LIST, M12_LIST, R12_LIST, M123_LIST, M23_Mext_opt_LIST,
    M13_Mcomp_opt_LIST, I3RE_RT_LIST, I3EXT_LIST)



def two_point_fth(nu2,alpha,nu1=33, alpha_nt=-0.85):
    if nu2>=nu1:
        raise ValueError(f"nu2 = {nu2} should be less than nu1 = {nu1}")
    if alpha<alpha_nt:
        print('W: ++==>> alpha is smaller than alpha_nt!')
        print('W: ++==>> Setting alpha_nt = alpha - 0.1')
        alpha_nt = alpha - 0.1
    num = (nu2/nu1)**alpha - (nu2/nu1)**alpha_nt
    den = (nu2/nu1)**(-0.1) - (nu2/nu1)**alpha_nt
    return(num/den)



def RC_function_S2(nu, A1l, A2, alpha_nt, nu0):
    return A1l * ((nu / nu0) ** (-0.1)) + A2 * (nu0 ** (alpha_nt)) * ((nu / nu0) ** (alpha_nt))


def RC_function_SY_FF(nu, A_sy, A_ff, alpha_nt, nu0):
    # return A_ff * (nu0 ** (-0.1)) * ((nu / nu0) ** (-0.1)) + A_sy * (nu0 ** (alpha_nt)) * ((nu / nu0) ** (alpha_nt))
    return A_ff * ((nu / nu0) ** (-0.1)) + A_sy * ((nu / nu0) ** (alpha_nt))


def RC_function_SY_ffa(nu,Snu0,fth_nu0,alpha_nt,nu_tau_t,f_cov,nu0=10):
    tau_nu = (nu/nu_tau_t)**(-2.1)
    return((1 - fth_nu0) * Snu0 * (1-f_cov*(1-np.exp(-tau_nu))) * ((nu/nu0)**alpha_nt))

def RC_function_FF_ffa(nu,Snu0,fth_nu0,nu_tau_t,nu0=10):
    tau_nu = (nu/nu_tau_t)**(-2.1)
    return(fth_nu0 * Snu0 * ((1-np.exp(-tau_nu))/(tau_nu)) * ((nu/nu0)**(-0.1)))
    

def RC_function_SY_FF_ffa(nu, Snu0, fth_nu0, alpha_nt, nu_tau_t, f_cov=1.0, nu0=10):
    # return A_ff * (nu0 ** (-0.1)) * ((nu / nu0) ** (-0.1)) + A_sy * (nu0 ** (alpha_nt)) * ((nu / nu0) ** (alpha_nt))
    # tau_nu = (nu/nu_tau_t)**(-2.1)
    # S_ff_abs = fth_nu0 * Snu0 * ((1-np.exp(-tau_nu))/(np.exp(-tau_nu))) * ((nu/nu0)**(-0.1))
    S_ff_abs = RC_function_FF_ffa(nu,Snu0,fth_nu0,nu_tau_t,nu0)
    # S_sy_abs = (1 - fth_nu0) * Snu0 * (1-f_cov*(1-np.exp(-tau_nu))) * ((nu/nu0)**alpha_nt)
    S_sy_abs = RC_function_SY_ffa(nu,Snu0,fth_nu0,alpha_nt,nu_tau_t,f_cov,nu0)
    S_total_abs = S_ff_abs + S_sy_abs
    return S_total_abs


def RC_function_SY_ffa_v2(nu,A_sy,alpha_nt,nu_tau_t,f_cov,nu0=10):
    tau_nu = (nu/nu_tau_t)**(-2.1)
    return((10**A_sy) * (1-f_cov*(1-np.exp(-tau_nu))) * ((nu/nu0)**alpha_nt))

def RC_function_FF_ffa_v2(nu,A_ff,nu_tau_t,nu0=10):
    tau_nu = (nu/nu_tau_t)**(-2.1)
    return((10**A_ff) * ((1-np.exp(-tau_nu))/(tau_nu)) * ((nu/nu0)**(-0.1)))

def RC_function_SY_FF_ffa_v2(nu, A_sy, A_ff, alpha_nt, nu_tau_t, f_cov=1.0, nu0=10):
    # return A_ff * (nu0 ** (-0.1)) * ((nu / nu0) ** (-0.1)) + A_sy * (nu0 ** (alpha_nt)) * ((nu / nu0) ** (alpha_nt))
    # tau_nu = (nu/nu_tau_t)**(-2.1)
    # S_ff_abs = fth_nu0 * Snu0 * ((1-np.exp(-tau_nu))/(np.exp(-tau_nu))) * ((nu/nu0)**(-0.1))
    S_ff_abs = RC_function_FF_ffa_v2(nu,A_ff,nu_tau_t,nu0)
    # S_sy_abs = (1 - fth_nu0) * Snu0 * (1-f_cov*(1-np.exp(-tau_nu))) * ((nu/nu0)**alpha_nt)
    S_sy_abs = RC_function_SY_ffa_v2(nu,A_sy,alpha_nt,nu_tau_t,f_cov,nu0)
    S_total_abs = S_ff_abs + S_sy_abs
    return S_total_abs

def RC_function_linear(nu, A1l, alpha,nu0=10):
    return A1l * ((nu/nu0) ** (alpha))


def power_law_phys_model(nu, Reff, S0peak, alpha,nu0=1):
    """
    Physical motivated power law model for the radio emission. 
    
    It uses the effective radius of the emission as a parameter to be minimised.
    This assumes a Gaussian model, where the total flux density is given by 
    SG = (3 * pi * Speak * Reff ** 2)/2
    Lucatelli et al 2024. 
    """
    return 0.5 * 3 * np.pi * S0peak * (Reff ** 2.0) * ((nu/nu0) ** (alpha))


def calc_specidx_region_linear(freqs, fluxes, fluxes_err, ndim=3):
    from scipy.stats import t
    x = freqs / 1e9
    nu0 = np.mean(x)
    y = fluxes
    yerr = fluxes_err

    # plt.figure(figsize=(8, 6))
    # plt.errorbar(x, y,
    #              yerr=yerr,
    #              fmt='o', label='Observed data', color='k', ecolor='gray', alpha=0.5)
    # plt.xlabel('Frequency [GHz]')
    # plt.ylabel('Flux Density [mJy]')
    # plt.legend()
    # plt.semilogy()
    # plt.semilogx()
    # plt.show()

    def log_likelihood_linear(theta, nu, y):
        A1l, alpha = theta
        model = RC_function_linear(nu, A1l, alpha,nu0)
        # sigma2 = (np.std(y)) ** 2
        # sigma2 = yerr ** 2
        # likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        sigma2 = yerr ** 2
        # sigma2 = (np.sqrt((yerr)**2.0+0.1))**2.0
        likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        # likelihood = (y - model) / (y+np.sqrt((yerr)**2.0+0.1))#okay 
        return likelihood
        # return np.sum(y-model)**2.0

    # def log_likelihood_linear(theta, nu, y):
    #     dof = 10  # degrees of freedom for the t-distribution; adjust as needed
    #     A1l, alpha = theta
    #     model = RC_function_linear(nu, A1l, alpha)
    #     residuals = y - model
    #     log_likelihood = np.sum(t.logpdf(residuals / yerr, df=dof, scale=yerr))
    #     return log_likelihood

    def prior_transform_linear(utheta):
        ua1l, ualpha = utheta  # nu0 is not a parameter to be estimated, so it's removed from here
        a1l = -2.0 + ua1l * (5000.0)  # Transforming ua1l to be in the range -5 to 50
        alpha = -3.0 + ualpha * (3.0)  # Transforming ualpha_nt to be in the range -2 to 1.5
        return a1l, alpha

    ndim = 2
    dsampler = dynesty.DynamicNestedSampler(log_likelihood_linear, prior_transform_linear,
                                            bound='balls',
                                            ndim=ndim, nlive=30000, sample='rwalk',  # hslice, rwalk
                                            logl_args=(x, y))

    # dsampler.run_nested(dlogz_init=0.05, nlive_init=1500, nlive_batch=500,
    #                     # use_stop=False,
    #                 # maxiter_init=10000, maxiter_batch=1000, maxbatch=10
    #                    )
    dsampler.run_nested(nlive_init=5000, nlive_batch=1000)
    results = dsampler.results

    results.summary()

    try:
        lnz_truth = ndim * -np.log(2 * 10.)  # analytic evidence solution
        fig, axes = dyplot.runplot(results, lnz_truth=lnz_truth)

        fig, axes = dyplot.traceplot(results, truths=np.zeros(ndim),
                                     truth_color='black', show_titles=True,
                                     trace_cmap='magma', connect=True,
                                     connect_highlight=range(5))
    except:
        pass

    try:
        # # initialize figure
        # fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        # axes = axes.reshape((3, 3))  # reshape axes
        #
        # fg, ax = dyplot.cornerplot(results, color='blue', truths=np.zeros(ndim),
        #                            truth_color='black', show_titles=True,
        #                            max_n_ticks=3, quantiles=None,
        #                            fig=(fig, axes[:, :3]))
        fig, axes = dyplot.cornerplot(results,
                                      show_titles=True,  # Show titles with parameter values
                                      title_kwargs={'y': 1.03},  # Adjust title position
                                      labels=[r'$A_1$', r'$\alpha$'],  # Label axes
                                      label_kwargs=dict(fontsize=16),
                                      # Adjust label font size
                                      # quantiles = [0.1, 0.5, 0.9],
                                      # title_quantiles = [0.1, 0.5, 0.9],
                                      max_n_ticks=4, use_math_text=True, verbose=True,
                                      title_fmt='.3f',
                                      fig=plt.subplots(2, 2, figsize=(5, 5)))
    except:
        pass

    # Extract the best-fit parametershttps://dynesty.readthedocs.io/en/latest/examples.html
    best_fit_indices = np.argmax(results.logl)
    best_fit_params = results.samples[best_fit_indices]
    A1l_best, alpha_nt_best = best_fit_params

    # Calculate the best-fit model
    x_resample = np.linspace(np.min(x)*0.7, np.max(x)*1.3, 100)
    best_fit_model = RC_function_linear(x_resample, A1l_best, alpha_nt_best,nu0)

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y, yerr=yerr, fmt='o', label='Observed data', color='k', ecolor='gray',
                 alpha=0.5)
    plt.plot(x_resample, best_fit_model, label='Best-fit model', color='red')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Flux Density [Jy]')
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()

    try:
        # Extracting samples for the corner plot
        samples = results.samples

        # Creating a corner plot
        fig = corner.corner(samples,
                            labels=[r'$A_1$', r'$\alpha_{\rm nt}$'],
                            truths=[A1l_best, alpha_nt_best],
                            quantiles=[0.025, 0.5, 0.975], show_titles=True,
                            title_kwargs={"fontsize": 12})
        plt.show()
    except:
        pass
    return (results, dsampler)


def calc_specidx_region_S2(freqs, fluxes, fluxes_err, ndim=3):
    x = freqs / 1e9
    nu0 = np.mean(x)
    y = fluxes
    yerr = fluxes_err

    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y,
                 yerr=yerr,
                 fmt='o', label='Observed data', color='k', ecolor='gray', alpha=0.5)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Flux Density [mJy]')
    plt.legend()
    plt.semilogy()
    plt.semilogx()
    plt.show()

    def log_likelihood(theta, nu, y):
        A1l, A2, alpha_nt = theta
        model = RC_function_S2(nu, A1l, A2, alpha_nt, nu0)
        sigma2 = yerr ** 2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

    def prior_transform(utheta):
        ua1l, ua2, ualpha_nt = utheta  # nu0 is not a parameter to be estimated, so it's removed from here
        a1l = 0 + ua1l * (15)  # Transforming ua1l to be in the range -5 to 50
        a2 = 0 + ua2 * (15)  # Transforming ua2 to be in the range -5 to 50
        alpha_nt = -2.0 + ualpha_nt * (2.0)  # Transforming ualpha_nt to be in the range -2 to 1.5
        return a1l, a2, alpha_nt

    dsampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, bound='balls',
                                            ndim=ndim, nlive=15000, sample='rwalk',  # hslice, rwalk
                                            logl_args=(x, y))

    # dsampler.run_nested(dlogz_init=0.05, nlive_init=1500, nlive_batch=500,
    #                     # use_stop=False,
    #                 # maxiter_init=10000, maxiter_batch=1000, maxbatch=10
    #                    )
    dsampler.run_nested(nlive_init=2500, nlive_batch=1000)
    # dsampler.run_nested()
    results = dsampler.results

    results.summary()

    try:
        lnz_truth = ndim * -np.log(2 * 10.)  # analytic evidence solution
        fig, axes = dyplot.runplot(results, lnz_truth=lnz_truth)

        fig, axes = dyplot.traceplot(results, truths=np.zeros(ndim),
                                     truth_color='black', show_titles=True,
                                     trace_cmap='magma', connect=True,
                                     connect_highlight=range(5))
    except:
        pass

    try:
        # initialize figure
        fig, axes = dyplot.cornerplot(results,
                                      show_titles=True,  # Show titles with parameter values
                                      title_kwargs={'y': 1.03},  # Adjust title position
                                      labels=[r'$A_1$',r'$A_2$', r'$\alpha$'],  # Label axes
                                      label_kwargs=dict(fontsize=16),
                                      # Adjust label font sibze
                                      # quantiles = [0.16, 0.5, 0.84],title_quantiles = [0.16, 0.5, 0.84],
                                      max_n_ticks=4, use_math_text=True, verbose=True,
                                      title_fmt='.3f',
                                      fig=plt.subplots(3, 3, figsize=(7, 7)))
    except:
        pass

    # Extract the best-fit parametershttps://dynesty.readthedocs.io/en/latest/examples.html
    best_fit_indices = np.argmax(results.logl)
    best_fit_params = results.samples[best_fit_indices]
    A1l_best, A2_best, alpha_nt_best = best_fit_params

    # Calculate the best-fit model
    x_resample = np.linspace(np.min(x)*0.5, np.max(x)*2, 100)
    best_fit_model = RC_function_S2(x_resample, A1l_best, A2_best, alpha_nt_best, nu0)

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y, yerr=yerr, fmt='o', label='Observed data', color='k', ecolor='gray',
                 alpha=0.5)
    plt.plot(x_resample, best_fit_model, label='Best-fit model', color='red')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Flux Density')
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()

    try:
        # Extracting samples for the corner plot
        samples = results.samples

        # Creating a corner plot
        fig = corner.corner(samples,
                            labels=[r'$A_1$', r'$A_2$', r'$\alpha_{\rm nt}$'],
                            truths=[A1l_best, A2_best, alpha_nt_best],
                            quantiles=[0.025, 0.5, 0.975], show_titles=True,
                            title_kwargs={"fontsize": 12})
        plt.show()
    except:
        pass
    return (results, dsampler)

def RC_function_Sq(nu,S0,alpha,q):
    return S0*(nu**(alpha))*np.exp(q * (np.log(nu))**2.0)

def calc_specidx_region_Sq(freqs, fluxes, fluxes_err, ndim=3):
    x = freqs / 1e9
    nu0 = np.mean(x)
    y = fluxes
    yerr = fluxes_err
    S0_init = 2*np.max(y)

    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y,
                 yerr=yerr,
                 fmt='o', label='Observed data', color='k', ecolor='gray', alpha=0.5)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Flux Density [mJy]')
    plt.legend()
    plt.semilogy()
    plt.semilogx()
    plt.show()

    def log_likelihood(theta, nu, y):
        S0, alpha, q = theta
        model = RC_function_Sq(nu, S0, alpha, q)
        sigma2 = yerr ** 2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

    def prior_transform(utheta):
        us0, ualpha, uq = utheta
        # from here
        s0 = 0 + us0 * (S0_init)
        alpha = -2.0 + ualpha * (2.0)
        q = -0.9 + uq * (0.9)
        return s0, alpha, q


    dsampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, bound='balls',
                                            ndim=ndim, nlive=30000, sample='rwalk',  # hslice, rwalk
                                            logl_args=(x, y))

    # dsampler.run_nested(dlogz_init=0.05, nlive_init=1500, nlive_batch=500,
    #                     # use_stop=False,
    #                 # maxiter_init=10000, maxiter_batch=1000, maxbatch=10
    #                    )
    dsampler.run_nested(nlive_init=5000, nlive_batch=1000)
    # dsampler.run_nested()
    results = dsampler.results

    results.summary()

    try:
        lnz_truth = ndim * -np.log(2 * 10.)  # analytic evidence solution
        fig, axes = dyplot.runplot(results, lnz_truth=lnz_truth)

        fig, axes = dyplot.traceplot(results, truths=np.zeros(ndim),
                                     truth_color='black', show_titles=True,
                                     trace_cmap='magma', connect=True,
                                     connect_highlight=range(5))
    except:
        pass

    try:
        # initialize figure
        fig, axes = dyplot.cornerplot(results,
                                      show_titles=True,  # Show titles with parameter values
                                      title_kwargs={'y': 1.03},  # Adjust title position
                                      labels=[r'$S_{0}$',r'$\alpha$', r'$q$'],  # Label axes
                                      label_kwargs=dict(fontsize=16),
                                      # Adjust label font size
                                      # quantiles = [0.16, 0.5, 0.84],title_quantiles = [0.16, 0.5, 0.84],
                                      max_n_ticks=4, use_math_text=True, verbose=True,
                                      title_fmt='.3f',
                                      fig=plt.subplots(3, 3, figsize=(7, 7)))
    except:
        pass

    # Extract the best-fit parametershttps://dynesty.readthedocs.io/en/latest/examples.html
    best_fit_indices = np.argmax(results.logl)
    best_fit_params = results.samples[best_fit_indices]
    S0_best, alpha_best, q_best = best_fit_params

    # Calculate the best-fit model
    x_resample = np.linspace(x[0], x[-1], 100)
    best_fit_model = RC_function_Sq(x_resample, S0_best, alpha_best, q_best)

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y, yerr=yerr, fmt='o', label='Observed data', color='k', ecolor='gray',
                 alpha=0.5)
    plt.plot(x_resample, best_fit_model, label='Best-fit model', color='red')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Flux Density')
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()

    try:
        # Extracting samples for the corner plot
        samples = results.samples

        # Creating a corner plot
        fig = corner.corner(samples,
                            labels=[r'$S_{0}$',r'$\alpha$', r'$q$'],
                            truths=[S0_best, alpha_best, q_best],
                            quantiles=[0.025, 0.5, 0.975], show_titles=True,
                            title_kwargs={"fontsize": 12})
        plt.show()
    except:
        pass
    return (results, dsampler)


def general_mcmc(x_data, y_data, yerr_data, 
                 result_1, param_labels, 
                 model_func, nwalkers=None,
                 burn_in = 2500,thin=5,
                 nsteps=7500, sigma_errors=1.0, 
                 prior_sigma=15,
                 quantiles = [2.5, 50, 97.5]):
    """
    General MCMC simulation for a model with n parameters.
    
    Parameters:
    - x_data: Independent variable data
    - y_data: Dependent variable data
    - yerr_data: Error in dependent variable data
    - result_1: Minimization result object from lmfit
    - param_labels: List of parameter names
    - model_func: The model function to fit
    - nwalkers: Number of walkers for the MCMC sampler (default is 25 * number of parameters)
    - nsteps: Number of MCMC steps (default is 7500)
    - sigma_errors: Number of standard deviations of the best parameters distribution, where
      p +/- sigma * p_err 
    - prior_sigma: Number of standard deviations for prior (default is 15.0)
    
    Returns:
    - samples: MCMC samples for the parameters
    """
    # Extract best-fit parameters and their standard errors from the minimization result
    best_fit_params = np.asarray([result_1.params[label].value for label in param_labels])
    params_stderr = np.asarray([result_1.params[label].stderr for label in param_labels])
    
    mask = ~np.isnan(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]
    yerr_data = yerr_data[mask]

    # Number of dimensions (parameters)
    ndim = len(best_fit_params)
    if nwalkers is None:
        nwalkers = int(25 * ndim)
    # Define the log-probability function with priors
    # def log_prob(params, x, y, yerr, best_fit_params, params_stderr):
    #     # Prior: within +/- 10 sigma
    #     if not all(best_fit_params[i] - prior_sigma*params_stderr[i] < params[i] < best_fit_params[i] + prior_sigma*params_stderr[i] for i in range(ndim)):
    #         return -np.inf
    #     # Calculate the model predictions
    #     model_pred = model_func(params, x)
    #     # Calculate the log-likelihood
    #     sigma2 = yerr**2
    #     log_likelihood = -0.5 * np.sum((y - model_pred)**2 / sigma2 + np.log(sigma2))
    #     return log_likelihood

    # def log_prob(params, x, y, yerr, best_fit_params, params_stderr):
    #     # Check for NaNs in the input data or parameters
    #     if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(yerr)) or np.any(np.isnan(params)):
    #         return -np.inf  # Return negative infinity to ignore samples with NaNs
        
    #     # Prior: within +/- 10 sigma
    #     if not all(best_fit_params[i] - prior_sigma*params_stderr[i] < params[i] < best_fit_params[i] + prior_sigma*params_stderr[i] for i in range(ndim)):
    #         return -np.inf
        
    #     # Calculate the model predictions
    #     model_pred = model_func(params, x)
        
    #     # Calculate the log-likelihood
    #     sigma2 = yerr**2
    #     log_likelihood = -0.5 * np.sum((y - model_pred)**2 / sigma2 + np.log(sigma2))
        
    #     return log_likelihood
    
    def log_prob(params, x, y, yerr, best_fit_params, params_stderr):
        # Check for NaN in the parameters
        if np.any(np.isnan(params)):
            return -np.inf

        # Prior: within +/- prior_sigma * sigma
        if not all(best_fit_params[i] - prior_sigma * params_stderr[i] < params[i] < best_fit_params[i] + prior_sigma * params_stderr[i] for i in range(ndim)):
            return -np.inf

        # Calculate the model predictions
        model_pred = model_func(params, x)

        # Check for NaNs in model predictions
        if np.any(np.isnan(model_pred)):
            return -np.inf

        # Calculate sigma squared
        sigma2 = yerr ** 2

        # Check for zeros or NaNs in sigma2
        if np.any(sigma2 <= 0) or np.any(np.isnan(sigma2)):
            return -np.inf

        # Check for NaNs in y
        if np.any(np.isnan(y)):
            return -np.inf

        # Calculate the log-likelihood
        residuals = y - model_pred
        if np.any(np.isnan(residuals)):
            return -np.inf

        log_likelihood = -0.5 * np.sum((residuals) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

        # Check for NaN in log_likelihood
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    # Initialize the walkers
    p0 = []
    for i in range(nwalkers):
        walker_params = []
        random_params = np.asarray(generate_random_params_normal(best_fit_params, 
                                                                 params_stderr,
                                                                 sigma_errors=sigma_errors))
        p0.append(list(random_params))
    p0 = np.asarray(p0) 

    # Set up the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x_data, 
                                                                    y_data, 
                                                                    yerr_data, 
                                                                    best_fit_params, 
                                                                    params_stderr))

    # Run MCMC
    sampler.run_mcmc(p0, nsteps, progress=True)

    # Analyze the samples
    samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    
    param_dict = {}
    for i, label in enumerate(param_labels):
        q = np.percentile(samples[:, i], [quantiles[0]*100,
                                          quantiles[1]*100,
                                          quantiles[2]*100])
        param_dict[label] = {
            'best': q[1],
            'lower_bound': q[0],
            'upper_bound': q[2],
            'lower': (q[1] - q[0]),
            'upper': (q[2] - q[1])
        }
    
    # # Plotting the samples
    # fig = corner.corner(samples, 
    #                     labels=param_labels,
    #                     show_titles=True, quantiles=[0.025, 0.5, 0.975],
    #                     truths=best_fit_params)
    # fig.show()
    
    return samples,param_dict


def general_mcmc_physical(x_data, y_data, yerr_data, 
                 result_1, param_labels, 
                 model_func, S0peak, nu0=1.0,
                 nwalkers=None,
                 burn_in = 2500,thin=5,
                 nsteps=7500, sigma_errors=1.0, prior_sigma=15):
    """
    General MCMC simulation for a model with n parameters.
    
    Parameters:
    - x_data: Independent variable data
    - y_data: Dependent variable data
    - yerr_data: Error in dependent variable data
    - result_1: Minimization result object from lmfit
    - param_labels: List of parameter names
    - model_func: The model function to fit
    - nwalkers: Number of walkers for the MCMC sampler (default is 200)
    - nsteps: Number of MCMC steps (default is 7500)
    - sigma_errors: Number of standard deviations of the best parameters distribution, where
      p +/- sigma * p_err 
    - prior_sigma: Number of standard deviations for prior (default is 15.0)
    
    Returns:
    - samples: MCMC samples for the parameters
    """
    # Extract best-fit parameters and their standard errors from the minimization result
    best_fit_params = np.asarray([result_1.params[label].value for label in param_labels])
    params_stderr = np.asarray([result_1.params[label].stderr for label in param_labels])
    
    mask = ~np.isnan(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]
    yerr_data = yerr_data[mask]

    # Number of dimensions (parameters)
    ndim = len(best_fit_params)
    if nwalkers is None:
        nwalkers = int(100 * ndim)
    # Define the log-probability function with priors
    def log_prob(params, x, y, yerr, best_fit_params, params_stderr):
        # Prior: within +/- 10 sigma
        if not all(best_fit_params[i] - prior_sigma*params_stderr[i] < params[i] < best_fit_params[i] + prior_sigma*params_stderr[i] for i in range(ndim)):
            return -np.inf
        # Calculate the model predictions
        model_pred = model_func(params, x, S0peak, nu0)
        # Calculate the log-likelihood
        sigma2 = yerr**2
        log_likelihood = -0.5 * np.sum((y - model_pred)**2 / sigma2 + np.log(sigma2))
        return log_likelihood

    # Initialize the walkers
    p0 = []
    for i in range(nwalkers):
        walker_params = []
        random_params = np.asarray(generate_random_params_normal(best_fit_params, 
                                                                 params_stderr,
                                                                 sigma_errors=sigma_errors))
        p0.append(list(random_params))
    p0 = np.asarray(p0) 

    # Set up the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x_data, 
                                                                    y_data, 
                                                                    yerr_data, 
                                                                    best_fit_params, 
                                                                    params_stderr))

    # Run MCMC
    sampler.run_mcmc(p0, nsteps, progress=True)

    # Analyze the samples
    samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    
    param_dict = {}
    for i, label in enumerate(param_labels):
        q = np.percentile(samples[:, i], [2.5, 50, 97.5])
        param_dict[label] = {
            'best': q[1],
            'lower': (q[1] - q[0]),
            'upper': (q[2] - q[1])
        }
    
    # # Plotting the samples
    # fig = corner.corner(samples, 
    #                     labels=param_labels,
    #                     show_titles=True, quantiles=[0.025, 0.5, 0.975],
    #                     truths=best_fit_params)
    # fig.show()
    
    return samples,param_dict


def power_law_phys_fit(freqs,
                          fluxes,
                          fluxes_err,
                          S0peak,
                          _nu0=1.0,
                          basename_save=None,log_plot=True,
                          save_name_append = None,
                          plot_errors_shade = False,
                          do_mcmc_fit = False,
                          mcmc_version = 'general',
                          title_text = None,
                          add_fit_legend = True,
                          verbose=0):
    """
    Peform a fit to the radio spectrum using a linear model.

    Parameters
    ----------
    freqs : array
        Frequency array in Hz.
    fluxes : array
        Flux density array in mJy.
    fluxes_err : array
        Flux density error array in mJy.
    basename_save : str
        Basename to save the output files.
    verbose : int
        Verbosity level.

    """
    x = freqs / 1e9
    y = fluxes
    yerr = fluxes_err
    weights = 1.0 / yerr
    if _nu0 != 1.0:
        nu0 = _nu0/1e9
    else:
        nu0 = _nu0
        
    def min_func(params):
        Reff = params['Reff']
        alpha = params['alpha']
        model = power_law_phys_model(x, Reff, S0peak, alpha, nu0)
        res = (y - model) / (np.log(yerr))
        return res.copy()


    fit_params = lmfit.Parameters()
    fit_params.add("Reff", value=10.0, min=-0.01, max=10)
    fit_params.add("alpha", value=-0.9, min=-4.0, max=4.0)

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                             loss="huber", tr_solver="exact",
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                           loss="huber", tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 3),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # fig, ax = plt.subplots()
    x_resample = np.linspace(np.min(x)*0.7, np.max(x)*1.3, 500)
    ax1.errorbar(x, y, yerr=yerr, 
                 fmt='o',label='Data', color='k', ecolor='gray',alpha=0.5)
    
    if do_mcmc_fit == True:
        nwalkers = int(2 * 100)
        burn_in = 2500
        if mcmc_version == 'lmfit':
            """ 
            This will be removed in the future.
            """
            results_emcee = mini.emcee(burn=burn_in,
                                    steps=4000,
                                    thin=5, nwalkers=nwalkers,
                                    is_weighted=False,
                                    float_behavior='posterior',
                                    params=result.params, workers=-1)

            _samples_emcee = results_emcee.flatchain[burn_in:]
            _Reff = np.asarray(_samples_emcee['Reff'])
            _alpha = np.asarray(_samples_emcee['alpha'])
            model_samples = np.array(
                [power_law_phys_model(x_resample, _Reff[i], S0peak, _alpha[i], nu0) for i in
                range(_samples_emcee.shape[0])])
            samples_emcee = np.asarray(_samples_emcee[['Reff', 'alpha']])
            param_dict = {}
            for i, label in enumerate(['Reff', 'alpha']):
                q = np.percentile(samples_emcee[:, i], [2.5, 50, 97.5])
                param_dict[label] = {
                    'best': q[1],
                    'lower': (q[1] - q[0]),
                    'upper': (q[2] - q[1])
                }
            print(param_dict)
        if mcmc_version == 'general':
            """
            This seems to be more robust as we have more control over
            the MCMC process.
            """
            def RC_powerlaw_physical(params,x, S0peak, x0):
                return 0.5*3.0*np.pi*S0peak*(params[0]**2.0) * ((x/x0)**params[1])
                
            samples_emcee, param_dict = general_mcmc_physical(x_data = x, 
                                                    y_data = y, 
                                                    yerr_data = yerr, 
                                                    result_1 = result, 
                                                    param_labels = ["Reff", "alpha"], 
                                                    model_func = RC_powerlaw_physical,
                                                    S0peak = S0peak, 
                                                    nu0=nu0,
                                                    burn_in = burn_in,
                                                    nwalkers = nwalkers)
            _Reff = samples_emcee.T[0]
            _alpha = samples_emcee.T[1]

            model_samples = np.array(
                [power_law_phys_model(x_resample, 
                                    _Reff[i], S0peak,
                                    _alpha[i], nu0) for i in range(_Reff.shape[0])])
            print(param_dict)

        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)

    model_resample = power_law_phys_model(x_resample,
                                    result.params['Reff'].value,
                                    S0peak,
                                    result.params['alpha'].value,
                                    nu0)

    model_best = power_law_phys_model(x,
                                    result.params['Reff'].value,
                                    S0peak,
                                    result.params['alpha'].value,
                                    nu0)

    ax1.plot(x_resample,model_resample,
             color='red', ls='-.', label='Best-fit model')
    ax2.plot(x, (y-model_best)/model_best,
             color='green', ls='dotted', label='Residual')
    ax2.set_ylim(-1.0,1.0)

    if do_mcmc_fit == True:
        ax1.plot(x_resample, model_mean,
                 color='purple', linestyle=(0, (5, 10)), label='MCMC Mean')

    # plt.fill_between(x_resample, lower_bound, upper_bound, color='lightgray', alpha=0.5,
    #                  label='Parameter uncertainty')
    if plot_errors_shade:
        if do_mcmc_fit == True:
            ax1.fill_between(x_resample,
                             model_mean - 3*model_std,
                             model_mean + 3*model_std, color='lightgray',
                            alpha=0.7)
        else:
            # Define the number of Monte Carlo samples
            num_samples = 5000

            # Generate random samples from parameter distributions
            Reff_samples = np.random.normal(result.params['Reff'].value, 
                                            result.params['Reff'].stderr,
                                            num_samples)
            alpha_samples = np.random.normal(result.params['alpha'].value,
                                             result.params['alpha'].stderr,
                                             num_samples)

            # Compute model predictions for each sample
            model_predictions = np.zeros((num_samples, len(x_resample)))
            for i in range(num_samples):
                model_predictions[i] = power_law_phys_model(x_resample, Reff_samples[i],
                                                            S0peak,
                                                            alpha_samples[i],nu0)

            median_prediction = np.median(model_predictions, axis=0)
            std_prediction = np.std(model_predictions, axis=0)

            ax1.fill_between(x_resample, median_prediction - 1*std_prediction,
                             median_prediction + 1*std_prediction,
                             color='lightgray', alpha=0.5,
                             # label='Uncertainty (1-sigma)'
                             )
    # plt.ylim(1e-3,1.2*np.max(y))
    if add_fit_legend == True:
        ax1.legend(loc=(0.05, 0.05),frameon=True,prop={'size': 11})
    if np.nanmax(y) < 5:
        ax1.set_ylim(0.1*np.nanmin(y),10.0*np.nanmax(y))
    else:
        ax1.set_ylim(0.1*np.nanmin(y),3.0*np.nanmax(y))
    ax2.set_xlabel(r'$\nu$ [GHz]')
    # plt.ylabel('Integrated Flux Density [mJy]')
    ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
    text_x, text_y = 0.65, 0.37
    text = (r"$\alpha_{\rm nth}"+f"= {(result.params['alpha'].value):.2f}\pm "
            rf"{(result.params['alpha'].stderr):.2f}$")
    
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = plt.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        bbox=text_bbox_props, transform=fig.transFigure)
    if do_mcmc_fit == True:
        if mcmc_version == 'general':
            text_x, text_y = 0.65, 0.77
            text = (r"$\alpha_{\rm nth}^{\rm MCMC}"+f"= {(param_dict['alpha']['best']):.2f}"
                    rf"_{{-{param_dict['alpha']['lower']:.2f}}}^{{+{param_dict['alpha']['upper']:.2f}}}$")
            text_bbox = plt.text(text_x, text_y, text,
                                # ha='center', va='center',
                                fontsize=12, color='black',
                                # bbox=text_bbox_props, 
                                transform=fig.transFigure)
    
    if title_text is not None:
        ax1.set_title(title_text)

    if log_plot == True:
        ax1.semilogx()
        ax1.semilogy()

    plt.subplots_adjust(hspace=0.05)
    ax2.legend(loc=(0.7, 0.55),prop={'size': 14},frameon=False)
    # legend()

    if basename_save is not None:
        if save_name_append is None:
            save_name_append = '_RC_alpha_fit_linear'
        else:
            save_name_append = save_name_append + '_RC_alpha_fit_linear'
        plt.savefig(basename_save.replace('.fits','_')+save_name_append+'.jpg', dpi=600,
                    bbox_inches='tight')
    # plt.show()
    corner_kwargs = {
        'bins': 30,
        'hist_bin_factor': 1.0,
        'color': 'purple',  # Change the color of the scatter plots and histograms
        'hist_kwargs': {
            'color': 'green',      # Histogram color
            'edgecolor': 'black',  # Edge color of the bins
            'linewidth': 1.5       # Width of the bin edges
        },
        'scatter_kwargs': {
            'alpha': 0.6,          # Transparency of points
            's': 10,               # Size of points
            'color': 'purple',     # Color of points
            'edgecolor': 'none'    # Edge color of points
        },
        'contour_kwargs': {
            'colors': 'blue',       # Color of contour lines
            'linewidths': 1.2       # Width of contour lines
        }
    }
    if do_mcmc_fit == True:
        try:
            from scipy.stats import gaussian_kde
            # fig_c = plt.figure(figsize=(2,2))
            fig_c = plt.figure()
            _ = corner.corner(samples_emcee,
                                    labels=[r'$R_{\rm eff}$',r'$\alpha$'],
                                    truths=[result.params['Reff'].value,
                                            result.params['alpha'].value],
                                    show_titles=True,
                                    quantiles=[0.025, 0.5, 0.975],
                                    # **corner_kwargs
                                    # fig=fig_c
                                    )
            print(samples_emcee.shape)
            if basename_save is not None:
                save_name_append_corner = save_name_append + '_corner'
                plt.savefig(basename_save.replace('.fits','_')+save_name_append_corner+'.jpg', 
                            dpi=600,bbox_inches='tight')
                
            print('++==>> Parameter Results (MCMC sampling).')
            print(lmfit.fit_report(results_emcee.params))
        except:
            pass
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))
    return(mini,result)




def do_fit_spec_RC_linear(freqs,
                          fluxes,
                          fluxes_err,
                          nu0=None,
                          basename_save=None,log_plot=True,
                          save_name_append = None,
                          plot_errors_shade = False,
                          do_mcmc_fit = False,
                          sigma_shade=3,
                          sigma_errors = 1.0,
                          quantiles = [0.16, 0.5, 0.84],
                          mcmc_version = 'general',
                          burn_in = 1000,
                          nsteps = 5000,
                          thin = 5,
                          title_text = None,
                          add_fit_legend = True,
                          verbose=0):
    """
    Peform a fit to the radio spectrum using a linear model.

    Parameters
    ----------
    freqs : array
        Frequency array in Hz.
    fluxes : array
        Flux density array in mJy.
    fluxes_err : array
        Flux density error array in mJy.
    basename_save : str
        Basename to save the output files.
    verbose : int
        Verbosity level.

    """
    x = freqs / 1e9
    y = fluxes
    if nu0 is None:
        nu0 = np.nanmean(x)
    yerr = fluxes_err
    weights = 1.0 / yerr
    epsilon = 1e-16
    def min_func(params):
        A1 = params['A1']
        alpha = params['alpha']
        model = RC_function_linear(x, A1, alpha,nu0)
        relative_error = yerr / (np.abs(y) + epsilon)
        weights = 1 / (relative_error + epsilon)
        weightned_residual = (y - model) * np.sqrt(weights)
        # res = (y - model)/yerr
        # loss = huber_loss(res)
        # res = (y - model)
        # res = (y - model) / (y+np.sqrt((yerr)**2.0+0.1))#okay
        # res = (y - model) / (y * (np.log(yerr)))  # okay
        # res = (y - model) / (y+np.log(abs(yerr))) # okay 9
        # res = (y  - model) / (np.sqrt(abs(y+yerr))) # okay 8
        # res = (y  - model) / (np.log(abs(y+yerr))) # okay 8
        # residuals = (y - model)/np.sqrt(abs(y+yerr))
        # res = (y - model) / (np.log(yerr))
        # res = 0.5 * (residuals**2 + np.sqrt(2 * np.pi * yerr**2))
        # res = (y - model) / (yerr)  # okay
        # res = (y - model) / (y + yerr)  # okay
        # res = data - RC_function_S2(nu, A1l, alpha)
        return weightned_residual.copy()


    fit_params = lmfit.Parameters()
    fit_params.add("A1", value=10.0, min=-5, max=10000)
    fit_params.add("alpha", value=-0.9, min=-4.0, max=4.0)

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                             loss="cauchy", tr_solver="exact",
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                           loss="cauchy", tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))
    # result_1 = mini.minimize(method='nelder',
    #                          options={'maxiter': 30000, 'maxfev': 30000,
    #                                   'xatol': 1e-12, 'fatol': 1e-12,
    #                                   'return_all': True,'adaptive': True,
    #                                   'disp': True}
    #                      )
    # second_run_params = result_1.params

    # result = mini.minimize(method='nelder',params=second_run_params,
    #                        options={'maxiter': 30000, 'maxfev': 30000,
    #                                 'xatol': 1e-12, 'fatol': 1e-12,
    #                                 'return_all': True,'adaptive': True,
    #                                 'disp': True}
    #                  )
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 3),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # fig, ax = plt.subplots()
    x_resample = np.linspace(np.min(x)*0.9, np.max(x)*1.1, 500)
    ax1.errorbar(x, y, yerr=yerr, 
                 fmt='o',label='Data', color='k', ecolor='gray',alpha=0.5)

    # nwalkers = int(len(y) * 2 * 20)
    
    
    if do_mcmc_fit == True:
        nwalkers = int(2 * 25)
        if mcmc_version == 'lmfit':
            """ 
            This will be removed in the future.
            """
            results_emcee = mini.emcee(burn=burn_in,
                                    steps=4000,
                                    thin=5, nwalkers=nwalkers,
                                    is_weighted=False,
                                    float_behavior='posterior',
                                    params=result.params, workers=-1)

            _samples_emcee = results_emcee.flatchain[burn_in:]
            _A1 = np.asarray(_samples_emcee['A1'])
            _alpha = np.asarray(_samples_emcee['alpha'])
            model_samples = np.array(
                [RC_function_linear(x_resample, _A1[i], _alpha[i],nu0) for i in
                range(_samples_emcee.shape[0])])
            samples_emcee = np.asarray(_samples_emcee[['A1', 'alpha']])
            param_dict = {}
            for i, label in enumerate(['A1', 'alpha']):
                q = np.percentile(samples_emcee[:, i], [2.5, 50, 97.5])
                param_dict[label] = {
                    'best': q[1],
                    'lower': (q[1] - q[0]),
                    'upper': (q[2] - q[1])
                }
            print(param_dict)
        if mcmc_version == 'general':
            """
            This seems to be more robust as we have more control over
            the MCMC process.
            """
            def RC_powerlaw(params,x):
                return params[0] * ((x/nu0)**params[1])
                
            samples_emcee, param_dict = general_mcmc(x_data = x, 
                                                    y_data = y, 
                                                    yerr_data = yerr, 
                                                    result_1 = result, 
                                                    param_labels = ["A1", "alpha"], 
                                                    burn_in = burn_in,
                                                    nwalkers = nwalkers,
                                                    nsteps = nsteps,
                                                    thin = thin,
                                                    sigma_errors = sigma_errors,
                                                    model_func = RC_powerlaw,
                                                    quantiles=quantiles)
            _A1 = samples_emcee.T[0]
            _alpha = samples_emcee.T[1]

            model_samples = np.array(
                [RC_function_linear(x_resample, 
                                    _A1[i], 
                                    _alpha[i],nu0) for i in range(_A1.shape[0])])
            print(param_dict)

        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)

    model_resample = RC_function_linear(x_resample,
                                    result.params['A1'].value,
                                    result.params['alpha'].value,
                                    nu0)

    model_best = RC_function_linear(x,
                                    result.params['A1'].value,
                                    result.params['alpha'].value,
                                    nu0)

    ax1.plot(x_resample,model_resample,
             color='red', ls='-.', label='Best-fit model')
    ax2.plot(x, (y-model_best)/model_best,
             color='green', ls='dotted', label='Residual')
    ax2.set_ylim(-1.0,1.0)

    if do_mcmc_fit == True:
        ax1.plot(x_resample, model_mean,
                 color='purple', linestyle=(0, (5, 10)), label='MCMC Mean')

    # plt.fill_between(x_resample, lower_bound, upper_bound, color='lightgray', alpha=0.5,
    #                  label='Parameter uncertainty')
    if plot_errors_shade:
        if do_mcmc_fit == True:
            ax1.fill_between(x_resample,
                             model_mean - sigma_shade*model_std,
                             model_mean + sigma_shade*model_std, color='lightgray',
                            alpha=0.7)
        else:
            # Define the number of Monte Carlo samples
            num_samples = 5000

            # Generate random samples from parameter distributions
            A1_samples = np.random.normal(result.params['A1'].value, result.params['A1'].stderr,
                                          num_samples)
            alpha_samples = np.random.normal(result.params['alpha'].value,
                                             result.params['alpha'].stderr,
                                             num_samples)

            # Compute model predictions for each sample
            model_predictions = np.zeros((num_samples, len(x_resample)))
            for i in range(num_samples):
                model_predictions[i] = RC_function_linear(x_resample, A1_samples[i],
                                                          alpha_samples[i],nu0)

            median_prediction = np.median(model_predictions, axis=0)
            std_prediction = np.std(model_predictions, axis=0)

            ax1.fill_between(x_resample, median_prediction - sigma_shade*std_prediction,
                             median_prediction + sigma_shade*std_prediction,
                             color='lightgray', alpha=0.5,
                             # label='Uncertainty (1-sigma)'
                             )
    # plt.ylim(1e-3,1.2*np.max(y))
    if add_fit_legend == True:
        ax1.legend(loc=(0.05, 0.05),frameon=True,prop={'size': 11})
    if np.nanmax(y) < 5:
        ax1.set_ylim(0.1*np.nanmin(y),10.0*np.nanmax(y))
    else:
        ax1.set_ylim(0.1*np.nanmin(y),3.0*np.nanmax(y))
    ax2.set_xlabel(r'$\nu$ [GHz]')
    # plt.ylabel('Integrated Flux Density [mJy]')
    ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
    text_x, text_y = 0.65, 0.37
    text = (r"$\alpha"+f"= {(result.params['alpha'].value):.2f}\pm "
            rf"{(result.params['alpha'].stderr):.2f}$")
    
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = plt.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        bbox=text_bbox_props, transform=fig.transFigure)
    if do_mcmc_fit == True:
        if mcmc_version == 'general':
            text_x, text_y = 0.65, 0.77
            text = (r"$\alpha^{\rm MCMC}"+f"= {(param_dict['alpha']['best']):.2f}"
                    rf"_{{-{param_dict['alpha']['lower']:.2f}}}^{{+{param_dict['alpha']['upper']:.2f}}}$")
            text_bbox = plt.text(text_x, text_y, text,
                                # ha='center', va='center',
                                fontsize=12, color='black',
                                # bbox=text_bbox_props, 
                                transform=fig.transFigure)
    
    if title_text is not None:
        ax1.set_title(title_text)

    if log_plot == True:
        ax1.semilogx()
        ax1.semilogy()

    plt.subplots_adjust(hspace=0.05)
    ax2.legend(loc=(0.7, 0.55),prop={'size': 14},frameon=False)
    # legend()

    if basename_save is not None:
        if save_name_append is None:
            save_name_append = '_RC_alpha_fit_linear'
        else:
            save_name_append = save_name_append + '_RC_alpha_fit_linear'
        plt.savefig(basename_save.replace('.fits','_')+save_name_append+'.jpg', dpi=600,
                    bbox_inches='tight')
    # plt.show()
    corner_kwargs = {
        'bins': 30,
        'hist_bin_factor': 1.0,
        'color': 'purple',  # Change the color of the scatter plots and histograms
        'hist_kwargs': {
            'color': 'green',      # Histogram color
            'edgecolor': 'black',  # Edge color of the bins
            'linewidth': 1.5       # Width of the bin edges
        },
        'scatter_kwargs': {
            'alpha': 0.6,          # Transparency of points
            's': 10,               # Size of points
            'color': 'purple',     # Color of points
            'edgecolor': 'none'    # Edge color of points
        },
        'contour_kwargs': {
            'colors': 'blue',       # Color of contour lines
            'linewidths': 1.2       # Width of contour lines
        }
    }
    if do_mcmc_fit == True:
        try:
            from scipy.stats import gaussian_kde
            # fig_c = plt.figure(figsize=(2,2))
            fig_c = plt.figure()
            _ = corner.corner(samples_emcee,
                                    labels=[r'$A_{\rm sy}$',r'$\alpha$'],
                                    truths=[result.params['A1'].value,
                                            result.params['alpha'].value],
                                    show_titles=True,
                                    quantiles=quantiles,
                                    # **corner_kwargs
                                    # fig=fig_c
                                    )
            print(samples_emcee.shape)
            if basename_save is not None:
                save_name_append_corner = save_name_append + '_corner'
                plt.savefig(basename_save.replace('.fits','_')+save_name_append_corner+'.jpg', 
                            dpi=600,bbox_inches='tight')
                
            print('++==>> Parameter Results (MCMC sampling).')
            print(lmfit.fit_report(results_emcee.params))
        except:
            pass
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))
    return(mini,result)


def do_fit_spec_RC_curv(freqs,
                          fluxes,
                          fluxes_err,
                          basename_save=None,log_plot=True,
                          save_name_append = None,
                          plot_errors_shade = False,
                          do_mcmc_fit = False,
                          title_text = None,
                          verbose=0):
    """
    Peform a fit to the radio spectrum using a curved model.

    Parameters
    ----------
    freqs : array
        Frequency array in Hz.
    fluxes : array
        Flux density array in mJy.
    fluxes_err : array
        Flux density error array in mJy.
    basename_save : str
        Basename to save the output files.
    verbose : int
        Verbosity level.

    """
    x = freqs / 1e9
    nu0 = np.mean(x)
    y = fluxes
    yerr = fluxes_err
    S0_init = 2*np.max(y)

    def min_func(params):
        S0 = params['S0']
        alpha = params['alpha']
        q = params['q']
        model = RC_function_Sq(x, S0, alpha, q)
        # res = (y - model) / (y + np.sqrt((yerr) ** 2.0 + 0.001))  # okay
        res = (y - model) / (y*(np.log(yerr)))  # okay
        return res.copy()

    fit_params = lmfit.Parameters()
    fit_params.add("S0", value=10.0, min=-5, max=5000)
    fit_params.add("alpha", value=-0.9, min=-5.0, max=5.0)
    fit_params.add("q", value=0.0, min=-0.9, max=0.9)

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                             loss="cauchy", tr_solver="exact",
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                           loss="cauchy", tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # fig = plt.figure(figsize=(8, 4))
    x_resample = np.linspace(np.min(x)*0.25, np.max(x)*1.3, 500)
    ax1.errorbar(x, y, yerr=yerr, fmt='o',
                 label='Data', color='k', ecolor='gray',
                 alpha=0.5)

    nwalkers = int(len(y) * 3 * 20)
    if do_mcmc_fit == True:
        burn_in = 500
        results_emcee = mini.emcee(burn=burn_in,
                                   steps=5000,
                                   thin=5, nwalkers=nwalkers,
                                   is_weighted=False,
                                   float_behavior='posterior',
                                   params=result.params, workers=-1)

        samples_emcee = results_emcee.flatchain[burn_in:]
        _S0 = np.asarray(samples_emcee['S0'])
        _alpha = np.asarray(samples_emcee['alpha'])
        _q = np.asarray(samples_emcee['q'])

        model_samples = np.array(
            [RC_function_Sq(x_resample, _S0[i], _alpha[i], _q[i]) for i in
             range(samples_emcee.shape[0])])

        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)

    model_resample = RC_function_Sq(x_resample,
                                    result.params['S0'].value,
                                    result.params['alpha'].value,
                                    result.params['q'].value)

    model_best = RC_function_Sq(x,
                                    result.params['S0'].value,
                                    result.params['alpha'].value,
                                    result.params['q'].value)


    ax1.plot(x_resample, model_resample,
             color='red', ls='-.', label='Best-fit model')
    ax2.plot(x, (y-model_best)/model_best,
             color='green', ls='dotted', label='Residual')
    ax2.set_xlabel(r'$\nu$ [GHz]')
    # plt.fill_between(x_resample, lower_bound, upper_bound, color='lightgray', alpha=0.5,
    #                  label='Parameter uncertainty')
    if plot_errors_shade:
        if do_mcmc_fit == True:
            ax1.plot(x_resample, model_mean,
                     color='purple', linestyle=(0, (5, 10)), label='MCMC Mean')
            ax1.fill_between(x_resample,
                             model_mean - 3*model_std,
                             model_mean + 3*model_std, color='lightgray',
                            alpha=0.5)
        else:
            # Define the number of Monte Carlo samples
            num_samples = 1000

            # Generate random samples from parameter distributions
            S0_samples = np.random.normal(result.params['S0'].value, result.params['S0'].stderr,
                                          num_samples)
            alpha_samples = np.random.normal(result.params['alpha'].value,
                                             result.params['alpha'].stderr,
                                             num_samples)
            q_samples = np.random.normal(result.params['q'].value, result.params['q'].stderr,
                                         num_samples)

            # Compute model predictions for each sample
            model_predictions = np.zeros((num_samples, len(x_resample)))
            for i in range(num_samples):
                model_predictions[i] = RC_function_Sq(x_resample, S0_samples[i],
                                                      alpha_samples[i],
                                                      q_samples[i])

            median_prediction = np.median(model_predictions, axis=0)
            std_prediction = np.std(model_predictions, axis=0)

            ax1.fill_between(x_resample,
                             median_prediction - 1*std_prediction,
                             median_prediction + 1*std_prediction,
                             color='lightgray', alpha=0.5,
                             # label='Uncertainty (1-sigma)'
                             )
    # plt.ylim(1e-3,1.2*np.max(y))
    ax1.legend()
    ax1.set_ylim(0.1*np.nanmin(y),3.0*np.nanmax(y))
    # plt.ylabel('Integrated Flux Density [mJy]')
    ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
    text_x, text_y = 0.65, 0.37
    text = (rf"$\alpha = {(result.params['alpha'].value):.2f}\pm "
            rf"{(result.params['alpha'].stderr):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = ax1.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        bbox=text_bbox_props, transform=fig.transFigure)

    text_x, text_y = 0.48, 0.37
    text = (rf"$q = {(result.params['q'].value):.2f}\pm "
            rf"{(result.params['q'].stderr):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = ax1.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        bbox=text_bbox_props, transform=fig.transFigure)

    if title_text is not None:
        ax1.set_title(title_text)

    if log_plot == True:
        ax1.semilogx()
        ax1.semilogy()
    plt.subplots_adjust(hspace=0.05)
    ax2.legend()
    if basename_save is not None:
        if save_name_append is None:
            save_name_append = '_RC_alpha_fit_curved'
        else:
            save_name_append = save_name_append + '_RC_alpha_fit_curved'
        plt.savefig(basename_save.replace('.fits','_')+save_name_append+'.jpg', dpi=600,
                    bbox_inches='tight')
    plt.show()

    plt.figure()

    if do_mcmc_fit == True:
        _ = corner.corner(samples_emcee[['S0','alpha','q']],
                                labels=[r'$S_{0}$',
                                        r'$\alpha$',
                                        r'$q$'],
                                truths=[result.params['S0'].value,
                                        result.params['alpha'].value,
                                        result.params['q'].value],
                                show_titles=True,
                                quantiles=[0.025, 0.5, 0.975])
        plt.show()
        print('++==>> Parameter Results (MCMC sampling).')
        print(lmfit.fit_report(results_emcee.params))
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))


    return mini,result,samples_emcee

def RC_function_turnover(nu,nut1,A,B,alpha):
    tau = (nu/nut1)**(-2.1)
    return ((1-np.exp(-tau))*(B+A*(nu/nut1)**(0.1+alpha))*((nu/nut1)**2.0))

def do_fit_spec_RC_turnover(freqs,
                          fluxes,
                          fluxes_err,
                          basename_save=None,log_plot=True,
                          save_name_append = None,
                          plot_errors_shade = False,
                          do_mcmc_fit = False,
                          title_text = None,
                          verbose=0):
    """
    Peform a fit to the radio spectrum using a curved model.

    Parameters
    ----------
    freqs : array
        Frequency array in Hz.
    fluxes : array
        Flux density array in mJy.
    fluxes_err : array
        Flux density error array in mJy.
    basename_save : str
        Basename to save the output files.
    verbose : int
        Verbosity level.

    """
    x = freqs / 1e9
    y = fluxes
    yerr = fluxes_err

    def minimisation_function(data,model,error,delta=0.1):
        sigma2 = error ** 2
        # return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        chi2 = np.sum((model - data)**2 / sigma2)
        # Calculate log likelihood
        log_likelihood = -0.5 * chi2
        return log_likelihood


    def min_func(params):
        nut1 = params['nut1']
        A = params['A']
        B = params['B']
        alpha = params['alpha']
        model = RC_function_turnover(x, nut1, A, B, alpha)
        # res = (y - model) / (y + np.sqrt((yerr) ** 2.0 + 0.001))  # okay
        log_likelihood = (y - model) / (y*(np.log(abs(yerr))))  # okay
        # res = (y - model) / (y * (np.sqrt(abs(yerr))))  # okay
        # sigma2 = yerr ** 2
        # chi2 = ((y - model)**2 / sigma2)
        # log_likelihood = -0.5 * chi2
        
        # log_likelihood=0.5 * ((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        
        # res = (y - model) / (y*np.log(abs(yerr) + 1.0))  # okay 8
        return log_likelihood.copy()

    fit_params = lmfit.Parameters()
    fit_params.add("nut1", value=1.0, min=0.01, max=10)
    fit_params.add("alpha", value=-0.9, min=-2.0, max=-0.2)
    fit_params.add("A", value=10.0, min=0.00001, max=5000)
    fit_params.add("B", value=10.0, min=0.00001, max=5000)

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                             loss="cauchy", tr_solver="exact",
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                           loss="cauchy", tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # fig = plt.figure(figsize=(8, 4))
    x_resample = np.linspace(np.min(x)*0.25, np.max(x)*1.3, 500)
    ax1.errorbar(x, y, yerr=yerr, fmt='o',
                 label='Data', color='k', ecolor='gray',
                 alpha=0.5)

    nwalkers = int(4 * 20)
    if do_mcmc_fit == True:
        burn_in = 500
        results_emcee = mini.emcee(burn=burn_in,
                                   steps=5000,
                                   thin=5, nwalkers=nwalkers,
                                   is_weighted=False,
                                   float_behavior='posterior',
                                   params=result.params, workers=-1)

        samples_emcee = results_emcee.flatchain[burn_in:]
        _nut1 = np.asarray(samples_emcee['nut1'])
        _alpha = np.asarray(samples_emcee['alpha'])
        _A = np.asarray(samples_emcee['A'])
        _B = np.asarray(samples_emcee['B'])
        # RC_function_turnover(nu, nut1, A, B, alpha)
        model_samples = np.array(
            [RC_function_turnover(x_resample, _nut1[i], _A[i], _B[i], _alpha[i]) for i in
             range(samples_emcee.shape[0])])

        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)

    model_resample = RC_function_turnover(x_resample,
                                    result.params['nut1'].value,
                                    result.params['A'].value,
                                    result.params['B'].value,
                                    result.params['alpha'].value)

    model_best = RC_function_turnover(x,
                                    result.params['nut1'].value,
                                    result.params['A'].value,
                                    result.params['B'].value,
                                    result.params['alpha'].value)


    ax1.plot(x_resample, model_resample,
             color='red', ls='-.', label='Best-fit model')
    ax2.plot(x, (y-model_best)/model_best,
             color='green', ls='dotted', label='Residual')
    ax2.set_xlabel(r'$\nu$ [GHz]')
    # plt.fill_between(x_resample, lower_bound, upper_bound, color='lightgray', alpha=0.5,
    #                  label='Parameter uncertainty')
    if plot_errors_shade:
        if do_mcmc_fit == True:
            ax1.plot(x_resample, model_mean,
                     color='purple', linestyle=(0, (5, 10)), label='MCMC Mean')
            ax1.fill_between(x_resample,
                             model_mean - 3*model_std,
                             model_mean + 3*model_std, color='lightgray',
                            alpha=0.5)
        else:
            # Define the number of Monte Carlo samples
            num_samples = 1000

            # Generate random samples from parameter distributions
            nut1_samples = np.random.normal(result.params['nut1'].value, result.params['nut1'].stderr,
                                          num_samples)
            alpha_samples = np.random.normal(result.params['alpha'].value,
                                             result.params['alpha'].stderr,
                                             num_samples)
            A_samples = np.random.normal(result.params['A'].value, result.params['A'].stderr,
                                         num_samples)

            B_samples = np.random.normal(result.params['B'].value, result.params['B'].stderr,
                                         num_samples)

            # Compute model predictions for each sample
            model_predictions = np.zeros((num_samples, len(x_resample)))
            for i in range(num_samples):
                model_predictions[i] = RC_function_turnover(x_resample,
                                                            nut1_samples[i],
                                                            A_samples[i],
                                                            B_samples[i],
                                                            alpha_samples[i])

            median_prediction = np.median(model_predictions, axis=0)
            std_prediction = np.std(model_predictions, axis=0)

            ax1.fill_between(x_resample,
                             median_prediction - 1*std_prediction,
                             median_prediction + 1*std_prediction,
                             color='lightgray', alpha=0.5,
                             # label='Uncertainty (1-sigma)'
                             )
    # plt.ylim(1e-3,1.2*np.max(y))
    ax1.legend()
    ax1.set_ylim(0.1*np.nanmin(y),3.0*np.nanmax(y))
    # plt.ylabel('Integrated Flux Density [mJy]')
    ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
    text_x, text_y = 0.65, 0.37
    text = (rf"$\alpha = {(result.params['alpha'].value):.2f}\pm "
            rf"{(result.params['alpha'].stderr):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = ax1.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        bbox=text_bbox_props, transform=fig.transFigure)

    text_x, text_y = 0.48, 0.37
    text = (rf"$\nu_t = {(result.params['nut1'].value):.2f}\pm "
            rf"{(result.params['nut1'].stderr):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = ax1.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        bbox=text_bbox_props, transform=fig.transFigure)

    if title_text is not None:
        ax1.set_title(title_text)

    if log_plot == True:
        ax1.semilogx()
        ax1.semilogy()
    plt.subplots_adjust(hspace=0.05)
    ax2.legend()
    if basename_save is not None:
        if save_name_append is None:
            save_name_append = '_RC_alpha_fit_turnover'
        else:
            save_name_append = save_name_append + '_RC_alpha_fit_turnover'
        plt.savefig(basename_save.replace('.fits','_')+save_name_append+'.jpg', dpi=600,
                    bbox_inches='tight')
    plt.show()

    plt.figure()

    if do_mcmc_fit == True:
        _ = corner.corner(samples_emcee[['nut1','A','B','alpha']],
                                labels=[r'$\nu_{t,1}$',
                                        r'$A$',
                                        r'$B$',
                                        r'$\alpha$'],
                                truths=[result.params['nut1'].value,
                                        result.params['A'].value,
                                        result.params['B'].value,
                                        result.params['alpha'].value],
                                show_titles=True,
                                quantiles=[0.025, 0.5, 0.975])
        plt.show()
        print('++==>> Parameter Results (MCMC sampling).')
        print(lmfit.fit_report(results_emcee.params))
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))


    return mini,result


    
    
def huber_loss(residuals, delta=1.0):
    return huber(delta, residuals)

def huber_like_weighting(yerr, k=1.345):
    return np.where(yerr > k, k / yerr, 1)


def fit_spec_SY_FF(freqs,
                   fluxes,
                   fluxes_err,
                   nu0=None,
                   basename_save=None,log_plot=True,
                   save_name_append = None,
                   plot_errors_shade = False,
                   sigma_shade=3,
                   quantiles = [0.16, 0.5, 0.84],
                   do_mcmc_fit = False,
                   mcmc_version = 'general',
                   burn_in = 1000,
                   nsteps = 5000,
                   thin = 5,
                   sigma_errors = 1.0,
                   title_text = None,
                   add_fit_legend = True,
                   plot_fit_results=True,
                   verbose=0):
    """
    Peform a fit to the radio spectrum using SY + FF model.

    Parameters
    ----------
    freqs : array
        Frequency array in Hz.
    fluxes : array
        Flux density array in mJy.
    fluxes_err : array
        Flux density error array in mJy.
    basename_save : str
        Basename to save the output files.
    verbose : int
        Verbosity level.

    """
    
    # if sigma_shade == 1 or sigma_shade == 1.0:
    #     quantiles = [0.16, 0.5, 0.84]
    # elif sigma_shade == 2 or sigma_shade == 2.0:
    #     quantiles = [0.025, 0.5, 0.975]
    # elif sigma_shade == 3 or sigma_shade == 3.0:
    #     quantiles = [0.00015, 0.5, 0.9985]
    # else:
    #     raise ValueError("Unsupported sigma value. Use sigma = 1, 2, or 3.")
    
    
    x = freqs / 1e9
    y = fluxes
    yerr = fluxes_err
    weights = 1.0 / yerr
    if nu0 is None:
        nu0 = np.mean(x)
    print(f' ++==>> Using reference frequency of {nu0} GHz.')



        
    def min_func(params):
        A_sy = params['A_sy']
        A_ff = params['A_ff']
        alpha_nt = params['alpha_nt']
        model = RC_function_SY_FF(x, A_sy, A_ff, alpha_nt, nu0)
        # res = (y - model) / (np.log(yerr))
        
        epsilon = 1e-8
        relative_error = yerr / (np.abs(y) + epsilon)
        weights = 1 / (relative_error + epsilon)
        res = (y - model) * np.sqrt(weights)
        
        # res = (y - model) * np.sqrt(2 / np.log1p(yerr**2))
        # res = (y - model)/yerr
        # loss = huber_loss(res)
        
        return res.copy()


    fit_params = lmfit.Parameters()
    fit_params.add("A_sy", value=0.5, min=0.0, max=5000)
    fit_params.add("A_ff", value=0.5, min=0.0, max=5000)
    fit_params.add("alpha_nt", value=-0.9, min=-2.5, max=2.5)

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                             loss="cauchy", tr_solver="exact",
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                           loss="cauchy", tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))

    x_resample = np.linspace(np.min(x)*0.7, np.max(x)*1.3, 500)
    
    
    if do_mcmc_fit == True:
        nwalkers = int(3 * 25)
        
        if mcmc_version == 'lmfit':
            """ 
            This will be removed in the future.
            """
            results_emcee = mini.emcee(burn=burn_in,
                                    steps=nsteps,
                                    thin=thin, nwalkers=nwalkers,
                                    is_weighted=False,
                                    float_behavior='posterior',
                                    params=result.params, workers=-1)

            _samples_emcee = results_emcee.flatchain[burn_in:]
            _A_sy = np.asarray(_samples_emcee['A_sy'])
            _A_ff = np.asarray(_samples_emcee['A_ff'])
            _alpha_nt = np.asarray(_samples_emcee['alpha_nt'])
            model_samples = np.array(
                [RC_function_SY_FF(x_resample, _A_sy[i], _A_ff[i], _alpha_nt[i], nu0) for i in
                range(_samples_emcee.shape[0])])
            samples_emcee = np.asarray(_samples_emcee[['A_sy', 'A_ff', 'alpha_nt']])
            param_dict = {}
            for i, label in enumerate(['A_sy', 'A_ff','alpha_nt']):
                q = np.percentile(samples_emcee[:, i], [2.5, 50, 97.5])
                param_dict[label] = {
                    'best': q[1],
                    'lower': (q[1] - q[0]),
                    'upper': (q[2] - q[1])
                }
            print(param_dict)
        if mcmc_version == 'general':
            """
            This seems to be more robust as we have more control over
            the MCMC process.
            """
            def RC_SY_FF(params,x):
                return (params[1] * ((x/nu0)**(-0.1)) + params[0] * ((x/nu0)**params[2]))
                
            samples_emcee, param_dict = general_mcmc(x_data = x, 
                                                    y_data = y, 
                                                    yerr_data = yerr, 
                                                    result_1 = result, 
                                                    param_labels = ["A_sy", "A_ff", 'alpha_nt'], 
                                                    burn_in = burn_in,
                                                    nwalkers = nwalkers,
                                                    nsteps = nsteps,
                                                    thin = thin,
                                                    sigma_errors = sigma_errors,
                                                    model_func = RC_SY_FF,
                                                    quantiles=quantiles)
            _A_sy = samples_emcee.T[0]
            _A_ff = samples_emcee.T[1]
            _alpha_nt = samples_emcee.T[2]

            model_samples = np.array(
                [RC_function_SY_FF(x_resample, 
                                    _A_sy[i], _A_ff[i], 
                                    _alpha_nt[i], nu0) for i in range(_A_sy.shape[0])])
            print(param_dict)

        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)

    model_resample = RC_function_SY_FF(x_resample,
                                    result.params['A_sy'].value,
                                    result.params['A_ff'].value,
                                    result.params['alpha_nt'].value,
                                    nu0)

    model_best = RC_function_SY_FF(x,
                                    result.params['A_sy'].value,
                                    result.params['A_ff'].value,
                                    result.params['alpha_nt'].value,
                                    nu0)



    A_sy_term = RC_function_SY_FF(x_resample,
                            result.params['A_sy'].value,
                            result.params['A_ff'].value*0,
                            result.params['alpha_nt'].value,nu0)
    
    
    A_ff_term = RC_function_SY_FF(x_resample,
                            result.params['A_sy'].value*0,
                            result.params['A_ff'].value,
                            result.params['alpha_nt'].value,nu0)
    
    A_ff_term_pred = RC_function_SY_FF(x,
                            result.params['A_sy'].value*0,
                            result.params['A_ff'].value,
                            result.params['alpha_nt'].value,nu0)
    
    thermal_fraction_freq = A_ff_term_pred / model_best

    if plot_fit_results:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 4),
                                    gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.errorbar(x, y, yerr=yerr, 
                    fmt='o',label='Data', color='k', ecolor='gray',alpha=0.5)

        ax1.plot(x_resample,model_resample,
                color='red', ls='-.', label='Fit')
        ax2.plot(x, (y-model_best)/model_best,
                color='green', ls='dotted', label='Residual')
        ax2.set_ylim(-1.0,1.0)


        ax1.plot(x_resample, A_sy_term,
                '-', label='sy')
        
        ax1.plot(x_resample, A_ff_term,
                '-', label='ff')

        if do_mcmc_fit == True:
            ax1.plot(x_resample, model_mean,
                    color='purple', linestyle=(0, (5, 10)), label='MCMC')

        # plt.fill_between(x_resample, lower_bound, upper_bound, color='lightgray', alpha=0.5,
        #                  label='Parameter uncertainty')
        if plot_errors_shade:
            if do_mcmc_fit == True:
                ax1.fill_between(x_resample,
                                model_mean - sigma_shade*model_std,
                                model_mean + sigma_shade*model_std, color='lightgray',
                                alpha=0.7)
            else:
                # Define the number of Monte Carlo samples
                num_samples = 5000

                # Generate random samples from parameter distributions
                A_sy_samples = np.random.normal(result.params['A_sy'].value, 
                                                result.params['A_sy'].stderr,
                                                num_samples)
                A_ff_samples = np.random.normal(result.params['A_ff'].value, 
                                                result.params['A_ff'].stderr,
                                                num_samples)
                alpha_nt_samples = np.random.normal(result.params['alpha_nt'].value,
                                                result.params['alpha_nt'].stderr,
                                                num_samples)

                # Compute model predictions for each sample
                model_predictions = np.zeros((num_samples, len(x_resample)))
                for i in range(num_samples):
                    model_predictions[i] = RC_function_SY_FF(x_resample, 
                                                            A_sy_samples[i],
                                                            A_ff_samples[i],
                                                            alpha_nt_samples[i],
                                                            nu0)

                median_prediction = np.median(model_predictions, axis=0)
                std_prediction = np.std(model_predictions, axis=0)

                ax1.fill_between(x_resample, median_prediction - sigma_shade*std_prediction,
                                median_prediction + sigma_shade*std_prediction,
                                color='lightgray', alpha=0.5,
                                # label='Uncertainty (1-sigma)'
                                )
        # plt.ylim(1e-3,1.2*np.max(y))
        if add_fit_legend == True:
            ax1.legend(loc=(0.05, 0.05),frameon=True,prop={'size': 11},ncol=2)
        if np.nanmax(y) < 5:
            ax1.set_ylim(0.1*np.nanmin(y),10.0*np.nanmax(y))
        else:
            ax1.set_ylim(0.1*np.nanmin(y),3.0*np.nanmax(y))
        ax2.set_xlabel(r'$\nu$ [GHz]')
        # plt.ylabel('Integrated Flux Density [mJy]')
        ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
        
        text_x, text_y = 0.62, 0.72
        text = (r"$\alpha_{\rm nt}"+f"= {(result.params['alpha_nt'].value):.2f}\pm "
                rf"{(result.params['alpha_nt'].stderr):.2f}$")
        text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
        text_bbox = plt.text(text_x, text_y, text,
                            # ha='center', va='center',
                            fontsize=12, color='black',
                            bbox=text_bbox_props, transform=fig.transFigure)
        
        if do_mcmc_fit == True:
            if mcmc_version == 'general':
                text_x, text_y = 0.62, 0.82
                text = (r"$\alpha_{\rm nt}^{\rm MCMC}"+f"= {(param_dict['alpha_nt']['best']):.2f}"
                        rf"_{{-{param_dict['alpha_nt']['lower']:.2f}}}^{{+{param_dict['alpha_nt']['upper']:.2f}}}$")
                text_bbox = plt.text(text_x, text_y, text,
                                    # ha='center', va='center',
                                    fontsize=12, color='black',
                                    # bbox=text_bbox_props, 
                                    transform=fig.transFigure)
        
        if title_text is not None:
            ax1.set_title(title_text)

        if log_plot == True:
            ax1.semilogx()
            ax1.semilogy()

        plt.subplots_adjust(hspace=0.05)
        ax2.legend(loc=(0.7, 0.55),prop={'size': 14},frameon=False,ncol=2)
        # legend()

        if basename_save is not None:
            if save_name_append is None:
                save_name_append = '_RC_sy_ff_fit'
            else:
                save_name_append = save_name_append + '_RC_sy_ff_fit'
            plt.savefig(basename_save.replace('.fits','_')+save_name_append+'.jpg', dpi=600,
                        bbox_inches='tight')
        # plt.show()
        corner_kwargs = {
            'bins': 30,
            'hist_bin_factor': 1.0,
            'color': 'purple',  # Change the color of the scatter plots and histograms
            'hist_kwargs': {
                'color': 'green',      # Histogram color
                'edgecolor': 'black',  # Edge color of the bins
                'linewidth': 1.5       # Width of the bin edges
            },
            'scatter_kwargs': {
                'alpha': 0.6,          # Transparency of points
                's': 10,               # Size of points
                'color': 'purple',     # Color of points
                'edgecolor': 'none'    # Edge color of points
            },
            'contour_kwargs': {
                'colors': 'blue',       # Color of contour lines
                'linewidths': 1.2       # Width of contour lines
            }
        }
        if do_mcmc_fit == True:
            try:
                from scipy.stats import gaussian_kde
                # fig_c = plt.figure(figsize=(2,2))
                fig_c = plt.figure()
                _ = corner.corner(samples_emcee,
                                        labels=[r'$A_{\rm sy}$',
                                                r'$A_{\rm ff}$',
                                                r'$\alpha_{\rm nt}$'],
                                        truths=[result.params['A_sy'].value,
                                                result.params['A_ff'].value,
                                                result.params['alpha_nt'].value],
                                        show_titles=True,
                                        quantiles=quantiles,
                                        # **corner_kwargs
                                        # fig=fig_c
                                        )
                print(samples_emcee.shape)
                if basename_save is not None:
                    save_name_append_corner = save_name_append + '_corner'
                    plt.savefig(basename_save.replace('.fits','_')+save_name_append_corner+'.jpg', 
                                dpi=600,bbox_inches='tight')
                    
                print('++==>> Parameter Results (MCMC sampling).')
                print(lmfit.fit_report(results_emcee.params))
            except:
                pass
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))
    if do_mcmc_fit:
        return(mini,result,thermal_fraction_freq,samples_emcee, param_dict)
    else:
        return(mini,result,thermal_fraction_freq)



def fit_spec_SY_FF_FFA(freqs,
                   fluxes,
                   fluxes_err,
                   nu0=None,
                   basename_save=None,log_plot=True,
                   save_name_append = None,
                   plot_errors_shade = False,
                   sigma_shade=3,
                   quantiles = [0.16, 0.5, 0.84],
                   do_mcmc_fit = False,
                   mcmc_version = 'general',
                   burn_in = 1000,
                   nsteps = 5000,
                   thin = 5,
                   sigma_errors = 1.0,
                   title_text = None,
                   add_fit_legend = True,
                   plot_fit_results=True,
                   verbose=0):
    """
    Peform a fit to the radio spectrum using SY + FF model in combination with 
    free-free absorption.

    Parameters
    ----------
    freqs : array
        Frequency array in Hz.
    fluxes : array
        Flux density array in mJy.
    fluxes_err : array
        Flux density error array in mJy.
    basename_save : str
        Basename to save the output files.
    verbose : int
        Verbosity level.

    """
    
    # if sigma_shade == 1 or sigma_shade == 1.0:
    #     quantiles = [0.16, 0.5, 0.84]
    # elif sigma_shade == 2 or sigma_shade == 2.0:
    #     quantiles = [0.025, 0.5, 0.975]
    # elif sigma_shade == 3 or sigma_shade == 3.0:
    #     quantiles = [0.00015, 0.5, 0.9985]
    # else:
    #     raise ValueError("Unsupported sigma value. Use sigma = 1, 2, or 3.")
    
    
    x = freqs / 1e9
    y = fluxes
    yerr = fluxes_err
    weights = 1.0 / yerr
    if nu0 is None:
        nu0 = np.mean(x)
    print(f' ++==>> Using reference frequency of {nu0} GHz.')



        
    def min_func(params):
        Snu0 = params['Snu0']
        fth_nu0 = params['fth_nu0']
        alpha_nt = params['alpha_nt']
        nu_tau_t = params['nu_tau_t']
        f_cov = params['f_cov']
        model = RC_function_SY_FF_ffa(x, Snu0, fth_nu0, alpha_nt, nu_tau_t, f_cov, nu0)
        # res = (y - model) / (np.log(yerr))
        epsilon = 1e-8
        relative_error = yerr / (np.abs(y) + epsilon)
        weights = 1 / (relative_error + epsilon)
        res = (y - model) * np.sqrt(weights)
        # res = (y - model) * np.sqrt(2 / np.log1p(yerr**2))
        # res = (y - model)/yerr
        # loss = huber_loss(res)
        
        return res.copy()


    fit_params = lmfit.Parameters()
    fit_params.add("Snu0", value=10.0, min=0.5, max=5000)
    fit_params.add("fth_nu0", value=0.1, min=0.01, max=0.99)
    fit_params.add("alpha_nt", value=-0.9, min=-2.5, max=2.5)
    fit_params.add("nu_tau_t", value=1.0, min=0.1, max=10.0)
    # fit_params.add("f_cov", value=0.99, min=0.0, max=1.0)
    fit_params.add("f_cov", value=1.0, min=0.9999, max=1.0)
    # fit_params.add("f_cov", value=1.0, vary=False)
    

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                             loss="cauchy", tr_solver="exact",
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                           loss="cauchy", tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))

    x_resample = np.linspace(np.min(x)*0.7, np.max(x)*1.3, 500)
    
    
    if do_mcmc_fit == True:
        nwalkers = int(5 * 25)
        
        if mcmc_version == 'lmfit':
            """ 
            This will be removed in the future.
            """
            results_emcee = mini.emcee(burn=burn_in,
                                    steps=nsteps,
                                    thin=thin, nwalkers=nwalkers,
                                    is_weighted=False,
                                    float_behavior='posterior',
                                    params=result.params, workers=-1)

            _samples_emcee = results_emcee.flatchain[burn_in:]
            _Snu0 = np.asarray(_samples_emcee['Snu0'])
            _fth_nu0 = np.asarray(_samples_emcee['fth_nu0'])
            _alpha_nt = np.asarray(_samples_emcee['alpha_nt'])
            _nu_tau_t = np.asarray(_samples_emcee['nu_tau_t'])
            _f_cov = np.asarray(_samples_emcee['f_cov'])
            model_samples = np.array(
                [RC_function_SY_FF_ffa(x_resample, _Snu0[i], _fth_nu0[i], _alpha_nt[i], _nu_tau_t, _f_cov, nu0) for i in
                range(_samples_emcee.shape[0])])
            samples_emcee = np.asarray(_samples_emcee[['Snu0', 'fth_nu0', 'alpha_nt', 'nu_tau_t', 'f_cov']])
            param_dict = {}
            for i, label in enumerate(['Snu0', 'fth_nu0','alpha_nt', 'nu_tau_t', 'f_cov']):
                q = np.percentile(samples_emcee[:, i], [2.5, 50, 97.5])
                param_dict[label] = {
                    'best': q[1],
                    'lower': (q[1] - q[0]),
                    'upper': (q[2] - q[1])
                }
            print(param_dict)
        if mcmc_version == 'general':
            """
            This seems to be more robust as we have more control over
            the MCMC process.
            """
            def RC_SY_FF_FFA(params,x):
                tau_nu = (x/params[3])**(-2.1)
                S_ff_abs = params[1] * params[0] * ((1-np.exp(-tau_nu))/(tau_nu)) * ((x/nu0)**(-0.1))
                S_sy_abs = (1 - params[1]) * params[0] * (1-params[4]*(1-np.exp(-tau_nu))) * ((x/nu0)**params[2])
                S_total_abs = S_ff_abs + S_sy_abs
                return S_total_abs
                
            samples_emcee, param_dict = general_mcmc(x_data = x, 
                                                    y_data = y, 
                                                    yerr_data = yerr, 
                                                    result_1 = result, 
                                                    param_labels = ["Snu0", "fth_nu0", 'alpha_nt', 'nu_tau_t', 'f_cov'], 
                                                    burn_in = burn_in,
                                                    nwalkers = nwalkers,
                                                    nsteps = nsteps,
                                                    thin = thin,
                                                    sigma_errors = sigma_errors,
                                                    model_func = RC_SY_FF_FFA,
                                                    quantiles=quantiles)
            _Snu0 = samples_emcee.T[0]
            _fth_nu0 = samples_emcee.T[1]
            _alpha_nt = samples_emcee.T[2]
            _nu_tau_t = samples_emcee.T[3]
            _f_cov = samples_emcee.T[4]

            model_samples = np.array(
                [RC_function_SY_FF_ffa(x_resample, 
                                    _Snu0[i], _fth_nu0[i], _alpha_nt[i], 
                                    _nu_tau_t[i], _f_cov[i], nu0) for i in range(_Snu0.shape[0])])
            print(param_dict)

        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)

    model_resample = RC_function_SY_FF_ffa(x_resample,
                                        result.params['Snu0'].value,
                                        result.params['fth_nu0'].value,
                                        result.params['alpha_nt'].value,
                                        result.params['nu_tau_t'].value,
                                        result.params['f_cov'].value,
                                    nu0)

    model_best = RC_function_SY_FF_ffa(x,
                                        result.params['Snu0'].value,
                                        result.params['fth_nu0'].value,
                                        result.params['alpha_nt'].value,
                                        result.params['nu_tau_t'].value,
                                        result.params['f_cov'].value,
                                    nu0)



    Sy_term = RC_function_SY_ffa(x_resample,
                            result.params['Snu0'].value,
                            result.params['fth_nu0'].value,
                            result.params['alpha_nt'].value,
                            result.params['nu_tau_t'].value,
                            result.params['f_cov'].value,
                            nu0)
    
    
    ff_term = RC_function_FF_ffa(x_resample,
                            result.params['Snu0'].value,
                            result.params['fth_nu0'].value,
                            result.params['nu_tau_t'].value,
                            nu0)
    
    ff_term_pred = RC_function_FF_ffa(x,
                            result.params['Snu0'].value,
                            result.params['fth_nu0'].value,
                            result.params['nu_tau_t'].value,
                            nu0)
    
    thermal_fraction_freq = ff_term_pred / model_best

    if plot_fit_results:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 4),
                                    gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.errorbar(x, y, yerr=yerr, 
                    fmt='o',label='Data', color='k', ecolor='gray',alpha=0.5)

        ax1.plot(x_resample,model_resample,
                color='red', ls='-.', label='Fit')
        ax2.plot(x, (y-model_best)/model_best,
                color='green', ls='dotted', label='Residual')
        ax2.set_ylim(-1.0,1.0)


        ax1.plot(x_resample, Sy_term,
                '-', label='sy')
        
        ax1.plot(x_resample, ff_term,
                '-', label='ff')

        if do_mcmc_fit == True:
            ax1.plot(x_resample, model_mean,
                    color='purple', linestyle=(0, (5, 10)), label='MCMC')

        # plt.fill_between(x_resample, lower_bound, upper_bound, color='lightgray', alpha=0.5,
        #                  label='Parameter uncertainty')
        if plot_errors_shade:
            if do_mcmc_fit == True:
                ax1.fill_between(x_resample,
                                model_mean - sigma_shade*model_std,
                                model_mean + sigma_shade*model_std, color='lightgray',
                                alpha=0.7)
            else:
                # Define the number of Monte Carlo samples
                num_samples = 5000

                # Generate random samples from parameter distributions
                Snu0_samples = np.random.normal(result.params['Snu0'].value, 
                                                result.params['Snu0'].stderr,
                                                num_samples)
                fth_nu0_samples = np.random.normal(result.params['fth_nu0'].value, 
                                                result.params['fth_nu0'].stderr,
                                                num_samples)
                alpha_nt_samples = np.random.normal(result.params['alpha_nt'].value,
                                                result.params['alpha_nt'].stderr,
                                                num_samples)
                nu_tau_t_samples = np.random.normal(result.params['nu_tau_t'].value,
                                                result.params['nu_tau_t'].stderr,
                                                num_samples)
                f_cov_samples = np.random.normal(result.params['f_cov'].value,
                                                result.params['f_cov'].stderr,
                                                num_samples)

                # Compute model predictions for each sample
                model_predictions = np.zeros((num_samples, len(x_resample)))
                for i in range(num_samples):
                    model_predictions[i] = RC_function_SY_FF_ffa(x_resample, 
                                                            Snu0_samples[i],
                                                            fth_nu0_samples[i],
                                                            alpha_nt_samples[i],
                                                            nu_tau_t_samples[i],
                                                            f_cov_samples[i],
                                                            nu0)

                median_prediction = np.median(model_predictions, axis=0)
                std_prediction = np.std(model_predictions, axis=0)

                ax1.fill_between(x_resample, median_prediction - sigma_shade*std_prediction,
                                median_prediction + sigma_shade*std_prediction,
                                color='lightgray', alpha=0.5,
                                # label='Uncertainty (1-sigma)'
                                )
        # plt.ylim(1e-3,1.2*np.max(y))
        if add_fit_legend == True:
            ax1.legend(loc=(0.05, 0.05),frameon=True,prop={'size': 11},ncol=2)
        if np.nanmax(y) < 5:
            ax1.set_ylim(0.1*np.nanmin(y),10.0*np.nanmax(y))
        else:
            ax1.set_ylim(0.1*np.nanmin(y),3.0*np.nanmax(y))
        ax2.set_xlabel(r'$\nu$ [GHz]')
        # plt.ylabel('Integrated Flux Density [mJy]')
        ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
        
        text_x, text_y = 0.62, 0.72
        text = (r"$\alpha_{\rm nt}"+f"= {(result.params['alpha_nt'].value):.2f}\pm "
                rf"{(result.params['alpha_nt'].stderr):.2f}$")
        text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
        text_bbox = plt.text(text_x, text_y, text,
                            # ha='center', va='center',
                            fontsize=12, color='black',
                            bbox=text_bbox_props, transform=fig.transFigure)
        
        if do_mcmc_fit == True:
            if mcmc_version == 'general':
                text_x, text_y = 0.62, 0.82
                text = (r"$\alpha_{\rm nt}^{\rm MCMC}"+f"= {(param_dict['alpha_nt']['best']):.2f}"
                        rf"_{{-{param_dict['alpha_nt']['lower']:.2f}}}^{{+{param_dict['alpha_nt']['upper']:.2f}}}$")
                text_bbox = plt.text(text_x, text_y, text,
                                    # ha='center', va='center',
                                    fontsize=12, color='black',
                                    # bbox=text_bbox_props, 
                                    transform=fig.transFigure)
        
        if title_text is not None:
            ax1.set_title(title_text)

        if log_plot == True:
            ax1.semilogx()
            ax1.semilogy()

        plt.subplots_adjust(hspace=0.05)
        ax2.legend(loc=(0.7, 0.55),prop={'size': 14},frameon=False,ncol=2)
        # legend()

        if basename_save is not None:
            if save_name_append is None:
                save_name_append = '_RC_sy_ff_ffa_fit'
            else:
                save_name_append = save_name_append + '_RC_sy_ff_ffa_fit'
            plt.savefig(basename_save.replace('.fits','_')+save_name_append+'.jpg', dpi=600,
                        bbox_inches='tight')
        # plt.show()
        corner_kwargs = {
            'bins': 30,
            'hist_bin_factor': 1.0,
            'color': 'purple',  # Change the color of the scatter plots and histograms
            'hist_kwargs': {
                'color': 'green',      # Histogram color
                'edgecolor': 'black',  # Edge color of the bins
                'linewidth': 1.5       # Width of the bin edges
            },
            'scatter_kwargs': {
                'alpha': 0.6,          # Transparency of points
                's': 10,               # Size of points
                'color': 'purple',     # Color of points
                'edgecolor': 'none'    # Edge color of points
            },
            'contour_kwargs': {
                'colors': 'blue',       # Color of contour lines
                'linewidths': 1.2       # Width of contour lines
            }
        }
        if do_mcmc_fit == True:
            try:
                from scipy.stats import gaussian_kde
                # fig_c = plt.figure(figsize=(2,2))
                fig_c = plt.figure()
                _ = corner.corner(samples_emcee,
                                        labels=[r'$S_{\nu_{0}}$',
                                                r'$f_{\rm th}^{\nu_0}}$',
                                                r'$\alpha_{\rm nt}$',
                                                r'$\nu_{\rm t}$',
                                                r'$\Xi$'
                                                ],
                                        truths=[result.params['Snu0'].value,
                                                result.params['fth_nu0'].value,
                                                result.params['alpha_nt'].value,
                                                result.params['nu_tau_t'].value,
                                                result.params['f_cov'].value
                                                ],
                                        show_titles=True,
                                        quantiles=quantiles,
                                        # **corner_kwargs
                                        # fig=fig_c
                                        )
                print(samples_emcee.shape)
                if basename_save is not None:
                    save_name_append_corner = save_name_append + '_corner'
                    plt.savefig(basename_save.replace('.fits','_')+save_name_append_corner+'.jpg', 
                                dpi=600,bbox_inches='tight')
                    
                print('++==>> Parameter Results (MCMC sampling).')
                print(lmfit.fit_report(results_emcee.params))
            except:
                pass
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))
    if do_mcmc_fit:
        return(mini,result,thermal_fraction_freq,samples_emcee, param_dict)
    else:
        return(mini,result,thermal_fraction_freq)



def fit_spec_SY_FF_FFA_v2(freqs,
                   fluxes,
                   fluxes_err,
                   nu0=None,
                   basename_save=None,log_plot=True,
                   save_name_append = None,
                   plot_errors_shade = False,
                   sigma_shade=3,
                   quantiles = [0.16, 0.5, 0.84],
                   do_mcmc_fit = False,
                   mcmc_version = 'general',
                   burn_in = 5000,
                   nsteps = 15000,
                   thin = 5,
                   sigma_errors = 1.0,
                   title_text = None,
                   add_fit_legend = True,
                   plot_fit_results=True,
                   verbose=0):
    """
    Peform a fit to the radio spectrum using SY + FF model in combination with 
    free-free absorption.

    Parameters
    ----------
    freqs : array
        Frequency array in Hz.
    fluxes : array
        Flux density array in mJy.
    fluxes_err : array
        Flux density error array in mJy.
    basename_save : str
        Basename to save the output files.
    verbose : int
        Verbosity level.

    """
    
    # if sigma_shade == 1 or sigma_shade == 1.0:
    #     quantiles = [0.16, 0.5, 0.84]
    # elif sigma_shade == 2 or sigma_shade == 2.0:
    #     quantiles = [0.025, 0.5, 0.975]
    # elif sigma_shade == 3 or sigma_shade == 3.0:
    #     quantiles = [0.00015, 0.5, 0.9985]
    # else:
    #     raise ValueError("Unsupported sigma value. Use sigma = 1, 2, or 3.")
    
    
    x = freqs / 1e9
    y = fluxes
    yerr = fluxes_err
    weights = 1.0 / yerr
    if nu0 is None:
        nu0 = np.mean(x)
    print(f' ++==>> Using reference frequency of {nu0} GHz.')



        
    def min_func(params):
        A_sy = params['A_sy']
        A_ff = params['A_ff']
        alpha_nt = params['alpha_nt']
        nu_tau_t = params['nu_tau_t']
        f_cov = params['f_cov']
        model = RC_function_SY_FF_ffa_v2(x, A_sy, A_ff, alpha_nt, nu_tau_t, f_cov, nu0)
        # res = (y - model) / (np.log(yerr))
        epsilon = 1e-8
        relative_error = yerr / (np.abs(y) + epsilon)
        weights = 1 / (relative_error + epsilon)
        res = (y - model) * np.sqrt(weights)
        # res = (y - model) * np.sqrt(2 / np.log1p(yerr**2))
        # res = (y - model)/yerr
        # loss = huber_loss(res)
        
        return res.copy()


    fit_params = lmfit.Parameters()
    A_sy_init = 0.7*np.nanmean(np.log10(y))
    A_ff_init = 0.2*np.nanmean(np.log10(y))
    a_nth_init = np.polyfit(np.log10(x), 
                                    np.log10(y), 1)[0]
    
    print('Init log A_sy=',A_sy_init)
    print('Init log A_ff=',A_ff_init)
    print('Init A_sy=',10**A_sy_init)
    print('Init A_ff=',10**A_ff_init)
    print('Init a_nth=',a_nth_init)
    
    fit_params.add("A_sy", value=A_sy_init, min=-A_sy_init*20, max=A_sy_init*20)
    fit_params.add("A_ff", value=A_ff_init, min=-A_ff_init*20, max=A_ff_init*20)
    fit_params.add("alpha_nt", value=a_nth_init, min=-2.0, max=0.0)
    fit_params.add("nu_tau_t", value=1.0, min=0.1, max=10.0)
    # fit_params.add("f_cov", value=0.99, min=0.0, max=1.0)
    fit_params.add("f_cov", value=1.0, min=0.99, max=1.0)
    # fit_params.add("f_cov", value=1.0, vary=False)
    

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                             loss="cauchy", tr_solver="exact",
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                           loss="cauchy", tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))

    x_resample = np.linspace(np.min(x)*0.7, np.max(x)*1.3, 500)
    
    
    if do_mcmc_fit == True:
        nwalkers = int(5 * 25)
        
        if mcmc_version == 'lmfit':
            """ 
            This will be removed in the future.
            """
            results_emcee = mini.emcee(burn=burn_in,
                                    steps=nsteps,
                                    thin=thin, nwalkers=nwalkers,
                                    is_weighted=False,
                                    float_behavior='posterior',
                                    params=result.params, workers=-1)

            _samples_emcee = results_emcee.flatchain[burn_in:]
            _A_sy = np.asarray(_samples_emcee['A_sy'])
            _A_ff = np.asarray(_samples_emcee['A_ff'])
            _alpha_nt = np.asarray(_samples_emcee['alpha_nt'])
            _nu_tau_t = np.asarray(_samples_emcee['nu_tau_t'])
            _f_cov = np.asarray(_samples_emcee['f_cov'])
            model_samples = np.array(
                [RC_function_SY_FF_ffa_v2(x_resample, _A_sy[i], _A_ff[i], _alpha_nt[i], _nu_tau_t, _f_cov, nu0) for i in
                range(_samples_emcee.shape[0])])
            samples_emcee = np.asarray(_samples_emcee[['A_sy', 'A_ff', 'alpha_nt', 'nu_tau_t', 'f_cov']])
            param_dict = {}
            for i, label in enumerate(['A_sy', 'A_ff','alpha_nt', 'nu_tau_t', 'f_cov']):
                q = np.percentile(samples_emcee[:, i], [2.5, 50, 97.5])
                param_dict[label] = {
                    'best': q[1],
                    'lower': (q[1] - q[0]),
                    'upper': (q[2] - q[1])
                }
            print(param_dict)
        if mcmc_version == 'general':
            """
            This seems to be more robust as we have more control over
            the MCMC process.
            """
            def RC_SY_FF_FFA_v2(params,x):
                tau_nu = (x/params[3])**(-2.1)
                S_ff_abs = params[1] * ((1-np.exp(-tau_nu))/(tau_nu)) * ((x/nu0)**(-0.1))
                S_sy_abs = params[0] * (1-params[4]*(1-np.exp(-tau_nu))) * ((x/nu0)**params[2])
                S_total_abs = S_ff_abs + S_sy_abs
                return S_total_abs
                
            samples_emcee, param_dict = general_mcmc(x_data = x, 
                                                    y_data = y, 
                                                    yerr_data = yerr, 
                                                    result_1 = result, 
                                                    param_labels = ["A_sy", "A_ff", 'alpha_nt', 'nu_tau_t', 'f_cov'], 
                                                    burn_in = burn_in,
                                                    nwalkers = nwalkers,
                                                    nsteps = nsteps,
                                                    thin = thin,
                                                    sigma_errors = sigma_errors,
                                                    model_func = RC_SY_FF_FFA_v2,
                                                    quantiles=quantiles)
            _A_sy = samples_emcee.T[0]
            _A_ff = samples_emcee.T[1]
            _alpha_nt = samples_emcee.T[2]
            _nu_tau_t = samples_emcee.T[3]
            _f_cov = samples_emcee.T[4]

            model_samples = np.array(
                [RC_function_SY_FF_ffa_v2(x_resample, 
                                    _A_sy[i], _A_ff[i], _alpha_nt[i], 
                                    _nu_tau_t[i], _f_cov[i], nu0) for i in range(_A_sy.shape[0])])
            print(param_dict)

        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)

    model_resample = RC_function_SY_FF_ffa_v2(x_resample,
                                        result.params['A_sy'].value,
                                        result.params['A_ff'].value,
                                        result.params['alpha_nt'].value,
                                        result.params['nu_tau_t'].value,
                                        result.params['f_cov'].value,
                                    nu0)

    model_best = RC_function_SY_FF_ffa_v2(x,
                                        result.params['A_sy'].value,
                                        result.params['A_ff'].value,
                                        result.params['alpha_nt'].value,
                                        result.params['nu_tau_t'].value,
                                        result.params['f_cov'].value,
                                    nu0)



    Sy_term = RC_function_SY_ffa_v2(x_resample,
                            result.params['A_sy'].value,
                            result.params['alpha_nt'].value,
                            result.params['nu_tau_t'].value,
                            result.params['f_cov'].value,
                            nu0)
    
    
    ff_term = RC_function_FF_ffa_v2(x_resample,
                            result.params['A_ff'].value,
                            result.params['nu_tau_t'].value,
                            nu0)
    
    ff_term_pred = RC_function_FF_ffa_v2(x,
                            result.params['A_ff'].value,
                            result.params['nu_tau_t'].value,
                            nu0)
    
    thermal_fraction_freq = ff_term_pred / model_best

    if plot_fit_results:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 4),
                                    gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.errorbar(x, y, yerr=yerr, 
                    fmt='o',label='Data', color='k', ecolor='gray',alpha=0.5)

        ax1.plot(x_resample,model_resample,
                color='red', ls='-.', label='Fit')
        ax2.plot(x, (y-model_best)/model_best,
                color='green', ls='dotted', label='Residual')
        ax2.set_ylim(-1.0,1.0)


        ax1.plot(x_resample, Sy_term,
                '-', label='sy')
        
        ax1.plot(x_resample, ff_term,
                '-', label='ff')

        if do_mcmc_fit == True:
            ax1.plot(x_resample, model_mean,
                    color='purple', linestyle=(0, (5, 10)), label='MCMC')

        # plt.fill_between(x_resample, lower_bound, upper_bound, color='lightgray', alpha=0.5,
        #                  label='Parameter uncertainty')
        if plot_errors_shade:
            if do_mcmc_fit == True:
                ax1.fill_between(x_resample,
                                model_mean - sigma_shade*model_std,
                                model_mean + sigma_shade*model_std, color='lightgray',
                                alpha=0.7)
            else:
                # Define the number of Monte Carlo samples
                num_samples = 5000

                # Generate random samples from parameter distributions
                A_sy_samples = np.random.normal(result.params['A_sy'].value, 
                                                result.params['A_sy'].stderr,
                                                num_samples)
                A_ff_samples = np.random.normal(result.params['A_ff'].value, 
                                                result.params['A_ff'].stderr,
                                                num_samples)
                alpha_nt_samples = np.random.normal(result.params['alpha_nt'].value,
                                                result.params['alpha_nt'].stderr,
                                                num_samples)
                nu_tau_t_samples = np.random.normal(result.params['nu_tau_t'].value,
                                                result.params['nu_tau_t'].stderr,
                                                num_samples)
                f_cov_samples = np.random.normal(result.params['f_cov'].value,
                                                result.params['f_cov'].stderr,
                                                num_samples)

                # Compute model predictions for each sample
                model_predictions = np.zeros((num_samples, len(x_resample)))
                for i in range(num_samples):
                    model_predictions[i] = RC_function_SY_FF_ffa_v2(x_resample, 
                                                            A_sy_samples[i],
                                                            A_ff_samples[i],
                                                            alpha_nt_samples[i],
                                                            nu_tau_t_samples[i],
                                                            f_cov_samples[i],
                                                            nu0)

                median_prediction = np.median(model_predictions, axis=0)
                std_prediction = np.std(model_predictions, axis=0)

                ax1.fill_between(x_resample, median_prediction - sigma_shade*std_prediction,
                                median_prediction + sigma_shade*std_prediction,
                                color='lightgray', alpha=0.5,
                                # label='Uncertainty (1-sigma)'
                                )
        # plt.ylim(1e-3,1.2*np.max(y))
        if add_fit_legend == True:
            ax1.legend(loc=(0.05, 0.05),frameon=True,prop={'size': 11},ncol=2)
        if np.nanmax(y) < 5:
            ax1.set_ylim(0.1*np.nanmin(y),10.0*np.nanmax(y))
        else:
            ax1.set_ylim(0.1*np.nanmin(y),3.0*np.nanmax(y))
        ax2.set_xlabel(r'$\nu$ [GHz]')
        # plt.ylabel('Integrated Flux Density [mJy]')
        ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
        
        text_x, text_y = 0.62, 0.72
        text = (r"$\alpha_{\rm nt}"+f"= {(result.params['alpha_nt'].value):.2f}\pm "
                rf"{(result.params['alpha_nt'].stderr):.2f}$")
        text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
        text_bbox = plt.text(text_x, text_y, text,
                            # ha='center', va='center',
                            fontsize=12, color='black',
                            bbox=text_bbox_props, transform=fig.transFigure)
        
        if do_mcmc_fit == True:
            if mcmc_version == 'general':
                text_x, text_y = 0.62, 0.82
                text = (r"$\alpha_{\rm nt}^{\rm MCMC}"+f"= {(param_dict['alpha_nt']['best']):.2f}"
                        rf"_{{-{param_dict['alpha_nt']['lower']:.2f}}}^{{+{param_dict['alpha_nt']['upper']:.2f}}}$")
                text_bbox = plt.text(text_x, text_y, text,
                                    # ha='center', va='center',
                                    fontsize=12, color='black',
                                    # bbox=text_bbox_props, 
                                    transform=fig.transFigure)
        
        if title_text is not None:
            ax1.set_title(title_text)

        if log_plot == True:
            ax1.semilogx()
            ax1.semilogy()

        plt.subplots_adjust(hspace=0.05)
        ax2.legend(loc=(0.7, 0.55),prop={'size': 14},frameon=False,ncol=2)
        # legend()

        if basename_save is not None:
            if save_name_append is None:
                save_name_append = '_RC_sy_ff_ffa_v2_fit'
            else:
                save_name_append = save_name_append + '_RC_sy_ff_ffa_v2_fit'
            plt.savefig(basename_save.replace('.fits','_')+save_name_append+'.jpg', dpi=600,
                        bbox_inches='tight')
        # plt.show()
        corner_kwargs = {
            'bins': 30,
            'hist_bin_factor': 1.0,
            'color': 'purple',  # Change the color of the scatter plots and histograms
            'hist_kwargs': {
                'color': 'green',      # Histogram color
                'edgecolor': 'black',  # Edge color of the bins
                'linewidth': 1.5       # Width of the bin edges
            },
            'scatter_kwargs': {
                'alpha': 0.6,          # Transparency of points
                's': 10,               # Size of points
                'color': 'purple',     # Color of points
                'edgecolor': 'none'    # Edge color of points
            },
            'contour_kwargs': {
                'colors': 'blue',       # Color of contour lines
                'linewidths': 1.2       # Width of contour lines
            }
        }
        if do_mcmc_fit == True:
            try:
                from scipy.stats import gaussian_kde
                # fig_c = plt.figure(figsize=(2,2))
                fig_c = plt.figure()
                _ = corner.corner(samples_emcee,
                                        labels=[r'$A_{\rm sy}$',
                                                r'$A_{\rm ff}$',
                                                r'$\alpha_{\rm nth}$',
                                                r'$\nu_{\rm t}$',
                                                r'$\Xi$'
                                                ],
                                        truths=[result.params['A_sy'].value,
                                                result.params['A_ff'].value,
                                                result.params['alpha_nt'].value,
                                                result.params['nu_tau_t'].value,
                                                result.params['f_cov'].value
                                                ],
                                        show_titles=True,
                                        quantiles=quantiles,
                                        # **corner_kwargs
                                        # fig=fig_c
                                        )
                print(samples_emcee.shape)
                if basename_save is not None:
                    save_name_append_corner = save_name_append + '_corner'
                    plt.savefig(basename_save.replace('.fits','_')+save_name_append_corner+'.jpg', 
                                dpi=600,bbox_inches='tight')
                    
                print('++==>> Parameter Results (MCMC sampling).')
                print(lmfit.fit_report(results_emcee.params))
            except:
                pass
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))
    if do_mcmc_fit:
        return(mini,result,thermal_fraction_freq,samples_emcee, param_dict)
    else:
        return(mini,result,thermal_fraction_freq)


def do_fit_spec_RC_S2(freqs,fluxes,fluxes_err,nu0=None,
                      basename_save=None, log_plot=True,
                      save_name_append=None,
                      plot_errors_shade=False,
                      do_mcmc_fit=False,
                      title_text=None,
                      burn_in = 1000,steps=5000,
                      verbose=0):
    x = freqs / 1e9
    y = fluxes
    yerr = fluxes_err
    if nu0 is None:
        nu0 = np.mean(x)


    def min_func(params):
        A1 = params['A1']
        A2 = params['A2']
        alpha_nt = params['alpha_nt']
        # res = (y - RC_function_S2(x, A1, A2, alpha_nt,nu0))/yerr
        model = RC_function_S2(x, A1, A2, alpha_nt,nu0)
        # res = (y - model)/(y+yerr)
        res = (y - model) / np.log(y+yerr)
        # res = (y - model) / (np.log(abs(yerr)+1.0)) # okay 9
        # res = data - RC_function_S2(nu, A1l, alpha_nt)
        return res.copy()

    fit_params = lmfit.Parameters()
    fit_params.add("A1", value=0.5, min=0, max=5000)
    fit_params.add("A2", value=0.5, min=-5, max=5000)
    fit_params.add("alpha_nt", value=-0.9, min=-2.5, max=2.5)

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                             loss="cauchy", tr_solver="exact",
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                           loss="cauchy", tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )


    # result_1 = mini.minimize(method='nelder',
    #                          options={'maxiter': 30000, 'maxfev': 30000,
    #                                   'xatol': 1e-12, 'fatol': 1e-12,
    #                                   'return_all': True,'adaptive': True,
    #                                   'disp': True}
    #                      )
    # second_run_params = result_1.params

    # result = mini.minimize(method='nelder',params=second_run_params,
    #                        options={'maxiter': 30000, 'maxfev': 30000,
    #                                 'xatol': 1e-12, 'fatol': 1e-12,
    #                                 'return_all': True,'adaptive': True,
    #                                 'disp': True}
    #                  )
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # fig = plt.figure(figsize=(8, 4))
    x_resample = np.linspace(np.min(x)*0.25, np.max(x)*1.3, 500)
    ax1.errorbar(x, y, yerr=yerr, fmt='o',
                 label='Data', color='k', ecolor='gray',
                 alpha=0.5)

    # nwalkers = int(len(y) * 3 * 20)
    nwalkers = int(3 * 100)

    if do_mcmc_fit == True:
        results_emcee = mini.emcee(burn=burn_in,
                                   steps=steps,
                                   thin=5, nwalkers=nwalkers,
                                   is_weighted=False,
                                   float_behavior='posterior',
                                   params=result.params, workers=-1)

        samples_emcee = results_emcee.flatchain[burn_in:]
        _A1 = np.asarray(samples_emcee['A1'])
        _A2 = np.asarray(samples_emcee['A2'])
        _alpha_nt = np.asarray(samples_emcee['alpha_nt'])

        model_samples = np.array(
            [RC_function_S2(x_resample, _A1[i], _A2[i], _alpha_nt[i], nu0) for i in
             range(samples_emcee.shape[0])])

        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)

    model_resample = RC_function_S2(x_resample,
                                 result.params['A1'].value,
                                 result.params['A2'].value,
                                 result.params['alpha_nt'].value,
                                 nu0)

    model_best = RC_function_S2(x,
                                 result.params['A1'].value,
                                 result.params['A2'].value,
                                 result.params['alpha_nt'].value,
                                 nu0)

    ax1.plot(x_resample, model_resample,
             color='red', ls='-.', label='Best-fit model')
    
    
    A1term = RC_function_S2(x_resample,
                            result.params['A1'].value,
                            result.params['A2'].value*0,
                            result.params['alpha_nt'].value,nu0)
    
    
    A2term = RC_function_S2(x_resample,
                            result.params['A1'].value*0,
                            result.params['A2'].value,
                            result.params['alpha_nt'].value,nu0)
    
    Snu0_low = RC_function_S2(1.4,
                            result.params['A1'].value,
                            result.params['A2'].value,
                            result.params['alpha_nt'].value,1.4)
    Snu0_mid = RC_function_S2(10.0,
                            result.params['A1'].value,
                            result.params['A2'].value,
                            result.params['alpha_nt'].value,10.0)
    Snu0_high = RC_function_S2(33.0,
                            result.params['A1'].value,
                            result.params['A2'].value,
                            result.params['alpha_nt'].value,33.0)
    
    thermal_fraction_low = result.params['A1'].value/Snu0_low
    thermal_fraction_mid = result.params['A1'].value/Snu0_mid
    thermal_fraction_high = result.params['A1'].value/Snu0_high
    
    ax1.plot(x_resample, A1term,
             '-', label='A1 term')
    
    ax1.plot(x_resample, A2term,
             '-', label='A2 term')

    ax2.plot(x, (y-model_best)/model_best,
             color='green', ls='dotted', label='Residual')
    ax2.set_xlabel(r'$\nu$ [GHz]')
    if plot_errors_shade:
        if do_mcmc_fit == True:
            ax1.plot(x_resample, model_mean,
                     color='purple', linestyle=(0, (5, 10)), label='MCMC Mean')
            ax1.fill_between(x_resample,
                             model_mean - 3*model_std,
                             model_mean + 3*model_std, color='lightgray',
                            alpha=0.5)
        else:
            # Define the number of Monte Carlo samples
            num_samples = 1000

            # Generate random samples from parameter distributions
            A1_samples = np.random.normal(result.params['A1'].value, result.params['A1'].stderr,
                                          num_samples)
            A2_samples = np.random.normal(result.params['A2'].value,
                                          result.params['A2'].stderr,
                                          num_samples)
            alpha_samples = np.random.normal(result.params['alpha_nt'].value,
                                             result.params['alpha_nt'].stderr,
                                             num_samples)

            # Compute model predictions for each sample
            model_predictions = np.zeros((num_samples, len(x_resample)))
            for i in range(num_samples):
                model_predictions[i] = RC_function_S2(x_resample,
                                                      A1_samples[i],
                                                      A2_samples[i],
                                                      alpha_samples[i],
                                                      nu0)

            median_prediction = np.median(model_predictions, axis=0)
            std_prediction = np.std(model_predictions, axis=0)

            ax1.fill_between(x_resample,
                             median_prediction - 1*std_prediction,
                             median_prediction + 1*std_prediction,
                             color='lightgray', alpha=0.5,
                             # label='Uncertainty (1-sigma)'
                             )
    # plt.ylim(1e-3,1.2*np.max(y))
    
    ax1.legend()
    ax1.set_ylim(0.1*np.min(y),3.0*np.max(y))
    ax1.set_xlabel(r'$\nu$ [GHz]')
    # plt.ylabel('Integrated Flux Density [mJy]')
    ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
    
    text_x, text_y = 0.72, 0.75
    text = (rf"$\alpha = {(result.params['alpha_nt'].value):.2f}\pm "
            rf"{(result.params['alpha_nt'].stderr):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = plt.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        bbox=text_bbox_props, transform=fig.transFigure)
    
    
    text_x, text_y = 0.50, 0.78
    text = (rf"$fth(10.0) = {(thermal_fraction_mid):.2f}\pm "
            rf"{(result.params['A1'].stderr/Snu0_mid):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = plt.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        # bbox=text_bbox_props, 
                        transform=fig.transFigure)
    
    # text_x, text_y = 0.5, 0.35
    # text = (rf"$fth(8.0) = {(thermal_fraction_mid):.2f}\pm "
    #         rf"{(result.params['A1'].stderr/Snu0_mid):.2f}$")
    # text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    # text_bbox = plt.text(text_x, text_y, text,
    #                     # ha='center', va='center',
    #                     fontsize=12, color='black',
    #                     # bbox=text_bbox_props, 
    #                     transform=fig.transFigure)
    
    text_x, text_y = 0.50, 0.73
    text = (rf"$fth(33.0) = {(thermal_fraction_high):.2f}\pm "
            rf"{(result.params['A1'].stderr/Snu0_high):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = plt.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        # bbox=text_bbox_props, 
                        transform=fig.transFigure)
    
    if title_text is not None:
        ax1.set_title(title_text)


    if log_plot == True:
        ax1.semilogx()
        ax1.semilogy()
    plt.subplots_adjust(hspace=0.05)
    ax2.legend()

    if basename_save is not None:
        if save_name_append is None:
            save_name_append = '_RC_alpha_fit_S2'
        else:
            save_name_append = save_name_append + '_RC_alpha_fit_S2'
        plt.savefig(basename_save.replace('.fits','_')+save_name_append+'.jpg', dpi=600,
                    bbox_inches='tight')
    plt.show()
    plt.figure()

    if do_mcmc_fit == True:
        _ = corner.corner(samples_emcee[['A1','A2','alpha_nt']],
                                labels=[r'$A_1$',
                                        r'$A_2$',
                                        r'$\alpha_{\rm nt}$'],
                                truths=[result.params['A1'].value,
                                        result.params['A2'].value,
                                        result.params['alpha_nt'].value],
                                show_titles=True,
                                quantiles=[0.025, 0.5, 0.975])
        plt.show()
        print('++==>> Parameter Results (MCMC sampling).')
        print(lmfit.fit_report(results_emcee.params))
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))


    return mini,result

def RC_function_sy_ff_dust(nu, Asy, Aff, Adu, alpha_nt, alpha_du, nu0):
    dust_comp = Adu * (500 ** (alpha_du)) * ((nu / 500) ** (alpha_du))
    sy_comp = Asy * (nu0 ** (alpha_nt)) * ((nu / nu0) ** (alpha_nt))
    return Aff * ((nu / nu0) ** (-0.1)) + dust_comp + sy_comp

def do_fit_spec_RC_sy_ff_dust(freqs,fluxes,fluxes_err,nu0=None,
                      basename_save=None, log_plot=True,
                      save_name_append=None,
                      plot_errors_shade=False,
                      do_mcmc_fit=False,
                      title_text=None,
                      verbose=0):
    x = freqs / 1e9
    y = fluxes
    yerr = fluxes_err
    if nu0 is None:
        nu0 = np.mean(x)


    def min_func(params):
        Asy = params['Asy']
        Aff = params['Aff']
        Adu = params['Adu']
        alpha_nt = params['alpha_nt']
        alpha_du = params['alpha_du']
        # res = (y - RC_function_S2(x, A1, A2, alpha_nt,nu0))/yerr
        model = RC_function_sy_ff_dust(x, Asy, Aff, Adu, alpha_nt, alpha_du, nu0)
        # res = (y - model)/(y+yerr)
        res = (y - model) / np.log(y+yerr)
        # res = (y - model) / (np.log(abs(yerr)+1.0)) # okay 9
        # res = data - RC_function_S2(nu, A1l, alpha_nt)
        return res.copy()

    fit_params = lmfit.Parameters()
    fit_params.add("Asy", value=0.5, min=0, max=5000)
    fit_params.add("Aff", value=0.5, min=0, max=5000)
    fit_params.add("Adu", value=0.5, min=0, max=5000)
    fit_params.add("alpha_nt", value=-0.9, min=-2.5, max=0.5)
    fit_params.add("alpha_du", value=3.0, min=1.8, max=4.0)

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                             loss="cauchy", tr_solver="exact",
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                           loss="cauchy", tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )


    # result_1 = mini.minimize(method='nelder',
    #                          options={'maxiter': 30000, 'maxfev': 30000,
    #                                   'xatol': 1e-12, 'fatol': 1e-12,
    #                                   'return_all': True,'adaptive': True,
    #                                   'disp': True}
    #                      )
    # second_run_params = result_1.params

    # result = mini.minimize(method='nelder',params=second_run_params,
    #                        options={'maxiter': 30000, 'maxfev': 30000,
    #                                 'xatol': 1e-12, 'fatol': 1e-12,
    #                                 'return_all': True,'adaptive': True,
    #                                 'disp': True}
    #                  )
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # fig = plt.figure(figsize=(8, 4))
    x_resample = np.linspace(np.min(x)*0.25, np.max(x)*1.3, 500)
    ax1.errorbar(x, y, yerr=yerr, fmt='o',
                 label='Data', color='k', ecolor='gray',
                 alpha=0.5)

    # nwalkers = int(len(y) * 3 * 20)
    nwalkers = int(5 * 100)

    if do_mcmc_fit == True:
        burn_in = 3000
        results_emcee = mini.emcee(burn=burn_in,
                                   steps=20000,
                                   thin=5, nwalkers=nwalkers,
                                   is_weighted=False,
                                   float_behavior='posterior',
                                   params=result.params, workers=-1)

        samples_emcee = results_emcee.flatchain[burn_in:]
        _Asy = np.asarray(samples_emcee['Asy'])
        _Aff = np.asarray(samples_emcee['Aff'])
        _Adu = np.asarray(samples_emcee['Adu'])
        _alpha_nt = np.asarray(samples_emcee['alpha_nt'])
        _alpha_du = np.asarray(samples_emcee['alpha_du'])

        model_samples = np.array(
            [RC_function_sy_ff_dust(x_resample, 
                            _Asy[i], 
                            _Aff[i],
                            _Adu[i], 
                            _alpha_nt[i],
                            _alpha_du[i], 
                            nu0) for i in
             range(samples_emcee.shape[0])])

        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)

    model_resample = RC_function_sy_ff_dust(x_resample,
                                 result.params['Asy'].value,
                                 result.params['Aff'].value,
                                 result.params['Adu'].value,
                                 result.params['alpha_nt'].value,
                                 result.params['alpha_du'].value,
                                 nu0)

    model_best = RC_function_sy_ff_dust(x,
                                 result.params['Asy'].value,
                                 result.params['Aff'].value,
                                 result.params['Adu'].value,
                                 result.params['alpha_nt'].value,
                                 result.params['alpha_du'].value,
                                 nu0)

    ax1.plot(x_resample, model_resample,
             color='red', ls='-.', label='Best-fit model')
    
    
    Asyterm = RC_function_sy_ff_dust(x_resample,
                                 result.params['Asy'].value,
                                 result.params['Aff'].value*0,
                                 result.params['Adu'].value*0,
                                 result.params['alpha_nt'].value,
                                 result.params['alpha_du'].value,
                                 nu0)
    
    
    Affterm = RC_function_sy_ff_dust(x_resample,
                                 result.params['Asy'].value*0,
                                 result.params['Aff'].value,
                                 result.params['Adu'].value*0,
                                 result.params['alpha_nt'].value,
                                 result.params['alpha_du'].value,
                                 nu0)
    Aduterm = RC_function_sy_ff_dust(x_resample,
                                 result.params['Asy'].value*0,
                                 result.params['Aff'].value*0,
                                 result.params['Adu'].value,
                                 result.params['alpha_nt'].value,
                                 result.params['alpha_du'].value,
                                 nu0)

    Snu0_14 = RC_function_sy_ff_dust(1.4,
                                 result.params['Asy'].value,
                                 result.params['Aff'].value,
                                 result.params['Adu'].value,
                                 result.params['alpha_nt'].value,
                                 result.params['alpha_du'].value,
                            1.4)
    
    Snu0_6 = RC_function_sy_ff_dust(6.0,
                                 result.params['Asy'].value,
                                 result.params['Aff'].value,
                                 result.params['Adu'].value,
                                 result.params['alpha_nt'].value,
                                 result.params['alpha_du'].value,
                            6.0)
    
    thermal_fraction_14 = result.params['Aff'].value/Snu0_14
    
    thermal_fraction_6 = result.params['Aff'].value/Snu0_6
    
    ax1.plot(x_resample, Asyterm,
             '-', label='Sy term')
    
    ax1.plot(x_resample, Affterm,
             '-', label='FF term')
    ax1.plot(x_resample, Aduterm,
             '-', label='Dust term')

    ax2.plot(x, (y-model_best)/model_best,
             color='green', ls='dotted', label='Residual')
    ax2.set_xlabel(r'$\nu$ [GHz]')
    if plot_errors_shade:
        if do_mcmc_fit == True:
            ax1.plot(x_resample, model_mean,
                     color='purple', linestyle=(0, (5, 10)), label='MCMC Mean')
            ax1.fill_between(x_resample,
                             model_mean - 3*model_std,
                             model_mean + 3*model_std, color='lightgray',
                            alpha=0.5)
        else:
            # Define the number of Monte Carlo samples
            num_samples = 1000

            # Generate random samples from parameter distributions
            Asy_samples = np.random.normal(result.params['Asy'].value, 
                                          result.params['Asy'].stderr,
                                          num_samples)
            Aff_samples = np.random.normal(result.params['Aff'].value,
                                          result.params['Aff'].stderr,
                                          num_samples)
            Adu_samples = np.random.normal(result.params['Adu'].value,
                                          result.params['Adu'].stderr,
                                          num_samples)
            alpha_nt_samples = np.random.normal(result.params['alpha_nt'].value,
                                             result.params['alpha_nt'].stderr,
                                             num_samples)
            alpha_du_samples = np.random.normal(result.params['alpha_du'].value,
                                             result.params['alpha_du'].stderr,
                                             num_samples)

            # Compute model predictions for each sample
            model_predictions = np.zeros((num_samples, len(x_resample)))
            for i in range(num_samples):
                model_predictions[i] = RC_function_sy_ff_dust(x_resample,
                                                      Asy_samples[i],
                                                      Aff_samples[i],
                                                      Adu_samples[i],
                                                      alpha_nt_samples[i],
                                                      alpha_du_samples[i],
                                                      nu0)

            median_prediction = np.median(model_predictions, axis=0)
            std_prediction = np.std(model_predictions, axis=0)

            ax1.fill_between(x_resample,
                             median_prediction - 1*std_prediction,
                             median_prediction + 1*std_prediction,
                             color='lightgray', alpha=0.5,
                             # label='Uncertainty (1-sigma)'
                             )
    # plt.ylim(1e-3,1.2*np.max(y))
    
    ax1.legend()
    ax1.set_ylim(0.1*np.min(y),3.0*np.max(y))
    ax1.set_xlabel(r'$\nu$ [GHz]')
    # plt.ylabel('Integrated Flux Density [mJy]')
    ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
    
    text_x, text_y = 0.70, 0.37
    text = (rf"$\alpha = {(result.params['alpha_nt'].value):.2f}\pm "
            rf"{(result.params['alpha_nt'].stderr):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = plt.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        bbox=text_bbox_props, transform=fig.transFigure)
    
    
    text_x, text_y = 0.5, 0.40
    text = (rf"$fth(1.4) = {(thermal_fraction_14):.2f}\pm "
            rf"{(result.params['Aff'].stderr/Snu0_14):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = plt.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        # bbox=text_bbox_props, 
                        transform=fig.transFigure)
    
    text_x, text_y = 0.5, 0.35
    text = (rf"$fth(6.0) = {(thermal_fraction_6):.2f}\pm "
            rf"{(result.params['Aff'].stderr/Snu0_6):.2f}$")
    text_bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
    text_bbox = plt.text(text_x, text_y, text,
                        # ha='center', va='center',
                        fontsize=12, color='black',
                        # bbox=text_bbox_props, 
                        transform=fig.transFigure)
    
    if title_text is not None:
        ax1.set_title(title_text)


    if log_plot == True:
        ax1.semilogx()
        ax1.semilogy()
    plt.subplots_adjust(hspace=0.05)
    ax2.legend()

    if basename_save is not None:
        if save_name_append is None:
            save_name_append = '_RC_sy_ff_dust'
        else:
            save_name_append = save_name_append + '_RC_sy_ff_dust'
        plt.savefig(basename_save.replace('.fits','_')+save_name_append+'.jpg', dpi=600,
                    bbox_inches='tight')
    plt.show()
    plt.figure()

    if do_mcmc_fit == True:
        _ = corner.corner(samples_emcee[['Asy','Aff','Adu','alpha_nt','alpha_du']],
                                labels=[r'$A_{\rm sy}$',
                                        r'$A_{\rm ff}$',
                                        r'$A_{\rm du}$',
                                        r'$\alpha_{\rm nt}$',
                                        r'$\alpha_{\rm du}$'],
                                truths=[result.params['Asy'].value,
                                        result.params['Aff'].value,
                                        result.params['Adu'].value,
                                        result.params['alpha_nt'].value,
                                        result.params['alpha_du'].value],
                                show_titles=True,
                                quantiles=[0.025, 0.5, 0.975])
        plt.show()
        print('++==>> Parameter Results (MCMC sampling).')
        print(lmfit.fit_report(results_emcee.params))
    print('++==>> Parameter Results (from least-squares fit).')
    print(lmfit.fit_report(result.params))


    return mini,result



"""
#Spectral Index Maps
"""

def linear_function(x, alpha, b,x0=1.0):
    return b*((x/x0)**alpha) #+ c*x*x


def makecube(imagelist):
    temp = []
    # if isinstance(imagelist[0], str):
    for image in imagelist:
        temp.append(load_fits_data(image))
    # else:
    #     for image in imagelist:
    #         temp.append(image)
    return np.dstack(temp)



# def huber_loss(r):
#     """
#     Calculate the Huber loss for a residual array r.
    
#     Parameters:
#     - r (array): Residual array, where each element is (y_obs - y_pred)/yerr.
#     - delta (float): The threshold at which the loss function changes from quadratic to linear.
    
#     Returns:
#     - float: Sum of the Huber loss over all residuals.
#     """
#     delta = 1.0
#     # Calculate the condition for each residual
#     condition = np.abs(r) < delta
    
#     # Calculate the squared loss for residuals within the threshold
#     squared_loss = 0.5 * (r[condition] ** 2)
    
#     # Calculate the linear loss for residuals outside the threshold
#     linear_loss = delta * (np.abs(r[~condition]) - 0.5 * delta)
    
#     # Sum all the losses
#     total_loss = np.sum(squared_loss) + np.sum(linear_loss)
    
#     return total_loss



def huber_loss(residuals, delta=1.0):
    abs_res = np.abs(residuals)
    quadratic = 0.5 * residuals**2
    linear = delta * (abs_res - 0.5 * delta)
    return np.where(abs_res <= delta, quadratic, linear)

def calculate_weights(yerr, epsilon=0.1, w_min=0.01, w_max=100):
    """
    Calculate weights from error values.

    Args:
    yerr (numpy array): Array of measurement errors.
    epsilon (float): Small constant to prevent division by zero and soften the weight.
    w_min (float): Minimum allowable weight.
    w_max (float): Maximum allowable weight.

    Returns:
    numpy array: Weights for each measurement.
    """
    # Basic weights calculation with softening
    # weights = 1.0 / (yerr + epsilon)**2
    weights = 1/np.sqrt(yerr+epsilon)

    # Capping the weights
    # weights = np.clip(weights, w_min, w_max)

    return weights

def do_fit_spec_map(freqs,fluxes,fluxes_err,nu0=1.0,verbose=0):

    x = freqs
    y = fluxes
    yerr = fluxes_err
    
    epsilon = 1e-10
    delta = 1.0
    def min_func(params):
        alpha = params['alpha']
        b = params['b']
        model = linear_function(x, alpha, b,nu0)
        relative_error = yerr / (np.abs(y) + epsilon)
        weights = 1 / (relative_error + epsilon)
        weightned_residual = (y - model) * np.sqrt(weights)
        # weightned_residual = (y - model) / np.log(abs(y+yerr))
        # weightned_residual = (y - model) / np.log(abs(yerr))
        # weightned_residual = (y - model) /yerr
        return weightned_residual.copy()
        
    
    fit_params = lmfit.Parameters()
    fit_params.add("alpha", value=-0.5, min=-3, max=3)
    fit_params.add("b", value=5.0, min=-1, max=500)
    
    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                        nan_policy='omit', reduce_fcn='neglogcauchy')
    
    result_1 = mini.minimize(method='least_squares',
                           max_nfev=15000, #f_scale = 1.0,
                        #    loss="huber", 
                           loss="cauchy",
                           tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12, 
                           verbose=verbose
                           )
    second_run_params = result_1.params
    
    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=15000, #f_scale = 1.0,
                        #    loss="huber", 
                           loss="cauchy", 
                           tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12, 
                           verbose=0
                        )
    
    return result


def do_fit_spec_SY_FF_map(freqs,fluxes,fluxes_err,nu0=None,verbose=0):
    x = freqs
    y = fluxes
    yerr = fluxes_err
    if nu0 is None:
        nu0 = np.mean(x)

    epsilon = 1e-16
    def min_func(params):
        A_sy = params['A_sy']
        A_ff = params['A_ff']
        alpha_nt = params['alpha_nt']
        model = RC_function_SY_FF(x, A_sy, A_ff, alpha_nt,nu0)
        
        relative_error = yerr / (np.abs(y) + epsilon)
        weights = 1 / (relative_error + epsilon)
        weightned_residual = (y - model) * np.sqrt(weights)
        # res = (y - RC_function_S2(x, A1, A2, alpha_nt,nu0))/(yerr+1)
        # res = (y - model) / (y+np.sqrt((yerr)**2.0+0.1))#okay 
        # res = (y - model) / (y+yerr)
        # res = (y - model) / (np.log(y+yerr))
        # res = data - RC_function_S2(nu, A1l, alpha_nt)
        return weightned_residual.copy()

    fit_params = lmfit.Parameters()
    fit_params.add("A_sy", value=1.0, min=1.0e-6, max=1000)
    fit_params.add("A_ff", value=0.1, min=1.0e-6, max=100)
    # fit_params.add("A1", value=0.5, min=-10, max=500)
    # fit_params.add("A2", value=0.5, min=-10, max=5000)

    fit_params.add("alpha_nt", value=-0.85, min=-3.0, max=3.0)
    # fit_params.add("alpha_nt", value=-0.85, min=-0.9, max=-0.8)

    mini = lmfit.Minimizer(min_func, fit_params, max_nfev=15000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    result_1 = mini.minimize(method='least_squares',
                             max_nfev=200000,  # f_scale = 1.0,
                            #  loss="huber", 
                             loss="cauchy",
                             tr_solver="exact",
                             ftol=1e-12, xtol=1e-12, gtol=1e-12,
                             verbose=verbose
                             )
    second_run_params = result_1.params

    result = mini.minimize(method='least_squares',
                           params=second_run_params,
                           max_nfev=200000,  # f_scale = 1.0,
                        #    loss="huber", 
                           loss="cauchy",
                           tr_solver="exact",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           verbose=verbose
                           )
    
    return result

def specidx_map(imagelist,residuallist,
                ref_image_conv=None,
                freqs=None,
                ref_image_mask = None,
                flux_sys_error_frac = 0.05,
                nu0=10.0,
                mask=None,sigma_global_mask=6,
                iterations=1,
                dilation_size=2,
                needs_convolution=False,conv_task='fft',
                return_only_cube=False,
                n_jobs=1,
                verbose=0):
    
    if isinstance(imagelist[0], str):
        cube_image = makecube(imagelist)
        cube_residuals = makecube(residuallist)
    else:
        cube_image = imagelist
        cube_residuals = residuallist
    
    if mask is None:
        if ref_image_mask is not None:
            ref_image_mask = ref_image_mask
        else:
            ref_image_mask = imagelist[-1]
        _,mask = mask_dilation(ref_image_mask,
                               rms=mad_std(load_fits_data(residuallist[-1])),
                                     show_figure=True,
                                     PLOT=True,
                                     iterations=iterations,
                                     dilation_size=dilation_size,
                                     sigma=sigma_global_mask)
    mask_3d = mask[:, :, np.newaxis]
    inv_mask = ~mask
    inv_mask_3d = inv_mask[:, :, np.newaxis]
    
    if needs_convolution:
        if ref_image_conv is None:
            ref_image_conv = sort_list_by_beam_size(imagelist=imagelist,
                                                return_df = False)[0][::-1][0]
        else:
            ref_image_conv = ref_image_conv
    
    
    if isinstance(ref_image_conv, str):
        psf_zise = int(get_beam_size_px(ref_image_conv)[0])
        psf_image_size = int(load_fits_data(ref_image_conv).shape[0])
        print(f"PSF BEAM SIZE is --> {psf_zise} px")
        print(f"PSF IMAGE SIZE is --> {psf_image_size} px")
        larger_beam_image_data = load_fits_data(ref_image_conv)
        

        psf_name = tcreate_beam_psf(ref_image_conv,
                                        size=(psf_image_size, psf_image_size),
                                        aspect = 'elliptical')
        PSF_DATA = load_fits_data(psf_name)
        
    num_images = cube_image.shape[2]
    if needs_convolution:
        conv_cube = np.empty_like(cube_image)
        conv_cube_res = np.empty_like(cube_residuals)
        if conv_task == 'fft':
            for i in tqdm(range(num_images)):
                conv_image_i_uncor = scipy.signal.fftconvolve(cube_image[:, :, i], 
                                                            PSF_DATA, 'same')
                conv_residual_i_uncor = scipy.signal.fftconvolve(cube_residuals[:, :, i], 
                                                            PSF_DATA, 'same')
                
                larger_beam_area = beam_area2(ref_image_conv)
                beam_area_i = beam_area2(imagelist[i])
                
                # factor_conv = np.max(larger_beam_image_data)/np.max(conv_image_i_uncor)
                factor_conv_i = larger_beam_area/beam_area_i
                
                print(f"Factor Convolution is --> {factor_conv_i}")
                # factor_conv = 1.0

                conv_cube[:, :, i] = conv_image_i_uncor * factor_conv_i
                conv_image_i = conv_cube[:, :, i]
                
                conv_cube_res[:, :, i] = conv_residual_i_uncor * factor_conv_i
                conv_res_i = conv_cube_res[:, :, i]
                
                conv_name = imagelist[i].replace('-image.','-image-conv.').replace('-image.cutout','-image-conv.cutout')
                pf.writeto(conv_name,
                                conv_image_i,
                                overwrite=True)
                copy_header(ref_image_conv,conv_name,conv_name)
                
                conv_res_name = residuallist[i].replace('-residual.','-residual-conv.').replace('-residual.cutout','-residual-conv.cutout')
                pf.writeto(conv_res_name,
                                conv_res_i,
                                overwrite=True)
                copy_header(ref_image_conv,conv_res_name,conv_res_name)
                
        # if conv_task == 'imsmooth':
        #     if isinstance(imagelist[0], str):
        #         for i in tqdm(range(num_images)):
        #             conv_name = imagelist[i].replace('-image.','-image-conv.').replace('-image.cutout','-image-conv.cutout')
        #             pf.writeto(conv_name,
        #                             cube_image[:, :, i],
        #                             overwrite=True)
        #             copy_header(imagelist[0],conv_name,conv_name)
                    
        #             convolve_2D_smooth(imagelist[0],conv_name,mode='transfer',add_prefix='')
        #             conv_cube[:,:,i] = load_fits_data(conv_name)
            
    else:
        conv_cube = cube_image.copy()
        conv_cube_res = cube_residuals.copy()
    
    # conv_cube_res = cube_residuals.copy()
    masked_cube = np.where(inv_mask_3d, np.nan, conv_cube)
    masked_cube_res = np.where(inv_mask_3d, np.nan, conv_cube_res)
    
    masked_cube = masked_cube.astype(np.float32)
    masked_cube_res = masked_cube_res.astype(np.float32)
    
    if return_only_cube:
        return(None,None,conv_cube,masked_cube_res,masked_cube)
    else:
        # cube_image * mask_3d
        idx = np.column_stack(np.where(mask==True))
        if freqs is None:
            freqs = getfreqs(imagelist)
        alphaimage = np.empty_like(conv_cube[:,:,0])
        alphaimage_error = np.empty_like(conv_cube[:,:,0])
        
        alphaimage[:] = np.nan
        alphaimage_error[:] = np.nan
        
        # x = np.log10(freqs/1e9)
        x = freqs/1e9
        
        nspec = len(idx)
        tenpc = int(nspec / 10.0)
        count = 0
        pcount = 0
        
        
        def compute_pixel_spectral_index(i, j, x, masked_cube, masked_cube_res, nu0):
            y = masked_cube[i, j, :] * 1000
            yerr = np.sqrt((masked_cube_res[i, j, :])**2.0 + (flux_sys_error_frac * masked_cube[i, j, :])**2.0) * 1000
            results_fit = do_fit_spec_map(x, y, yerr, nu0)
            return (i, j, 
                    results_fit.params['alpha'].value, 
                    results_fit.params['alpha'].stderr,
                    results_fit.params['b'].value,
                    y,yerr,results_fit)
        
        
        from joblib import Parallel, delayed
        
        pixel_indices = [(i, j) for i, j in idx]
        results = Parallel(n_jobs=n_jobs)(
        delayed(compute_pixel_spectral_index)(i, j, x, masked_cube, masked_cube_res, nu0) 
        for i, j in tqdm(pixel_indices, total=len(pixel_indices))
        )
        
        
        
        # for i, j in tqdm(idx):
        # # for i, j in idx:
        #     # if count == 0:
        #     #     print(str(pcount) + '%...')
            
            
            
        #     # yerr = (masked_cube_res[i,j,:]*1000)/(masked_cube[i,j,:]*1000 * np.log(10))
        #     # y = np.log10(masked_cube[i,j,:]*1000)
        #     y = masked_cube[i,j,:]*1000
        #     yerr = np.sqrt((masked_cube_res[i,j,:])**2.0 + (flux_sys_error_frac*masked_cube[i,j,:])**2.0)*1000
            
        
        #     results_fit = do_fit_spec_map(x,y,yerr,nu0)
        for (i, j, 
            alpha_value, alpha_err,
            b_value,y,yerr,results_fit) in results:
            alphaimage[i, j] = alpha_value
            alphaimage_error[i, j] = alpha_err
            
            # alphaimage[i, j] = results_fit.params['alpha'].value
            # alphaimage_error[i, j] = results_fit.params['alpha'].stderr
            count += 1
            if count == tenpc:
                count = 0
                pcount += 10
                if verbose>0:
                    model_best = linear_function(x, 
                                                alpha_value, 
                                                b_value,
                                                nu0)
                    plt.figure()

                    plt.errorbar(x, 
                                y, 
                                yerr=abs(yerr), 
                                fmt='o',
                                label='Data', color='k', ecolor='gray',
                                alpha=0.5)

                    plt.plot(x, model_best,
                            color='red', ls='-.', label='Best-fit model')
                    
                    plt.ylim(abs(np.nanmin(y)*0.1),np.nanmax(y)*5)
                    plt.semilogy()
                    plt.semilogx()
                    
                    plt.legend()
                    plt.show()
                    print(lmfit.fit_report(results_fit.params))

        
        if isinstance(imagelist[0], str):
            """
            Only save the alpha image if the input is a list of strings.
            """
            # alphaimage_name = imagelist[0].replace('-0000-image.fits','-alpha.fits')
            # alphaimage_error_name = imagelist[0].replace('-0000-image.fits','-alphaerror.fits')
            if ref_image_conv is None:
                ref_image_conv = sort_list_by_beam_size(imagelist=imagelist,
                                                    return_df = False)[0][::-1][0]
                
            _alphaimage_name = (ref_image_conv.replace('-image.','-alpha.').
                            replace('-image.cutout','-alpha.cutout').
                            replace('-image-pb','-alpha-pb'))
            
            _alphaimage_error_name = (ref_image_conv.replace('-image.','-alphaerror.').
                                    replace('-image.cutout','-alphaerror.cutout').
                                    replace('-image-pb','-alphaerror-pb'))
            
            alphaimage_name = os.path.dirname(os.path.dirname(_alphaimage_name))+'/'+os.path.basename(_alphaimage_name)
            alphaimage_error_name = os.path.dirname(os.path.dirname(_alphaimage_error_name))+'/'+os.path.basename(_alphaimage_error_name)
            
            
            pf.writeto(alphaimage_name,alphaimage,overwrite=True)
            pf.writeto(alphaimage_error_name,alphaimage_error,overwrite=True)
            copy_header(ref_image_conv,alphaimage_name,alphaimage_name)
            copy_header(ref_image_conv,alphaimage_error_name,alphaimage_error_name)
        
        return(alphaimage, alphaimage_error,conv_cube,masked_cube_res,masked_cube)
        
        
    
    
def specidx_map_SY_FF(imagelist,residuallist,
                      ref_image_conv=None,
                      freqs=None,
                      ref_image_mask = None,
                      flux_sys_error_frac = 0.05,
                      nu0=10.0,
                      mask=None,sigma_global_mask=6,
                      iterations=1,
                      dilation_size=2,
                      sed_model='S2',
                      needs_convolution=False,conv_task='fft',
                      n_jobs=1,
                      verbose=0):
    
    if isinstance(imagelist[0], str):
        cube_image = makecube(imagelist)
        cube_residuals = makecube(residuallist)
    else:
        cube_image = imagelist
        cube_residuals = residuallist
    
    if mask is None:
        if ref_image_mask is not None:
            ref_image_mask = ref_image_mask
        else:
            ref_image_mask = imagelist[-1]
        _,mask = mask_dilation(ref_image_mask,
                               rms=mad_std(load_fits_data(residuallist[-1])),
                                     show_figure=True,
                                     PLOT=True,
                                     iterations=iterations,
                                     dilation_size=dilation_size,
                                     sigma=sigma_global_mask)
    mask_3d = mask[:, :, np.newaxis]
    inv_mask = ~mask
    inv_mask_3d = inv_mask[:, :, np.newaxis]
    
    if needs_convolution == False:
        if ref_image_conv is None:
            ref_image_conv = sort_list_by_beam_size(imagelist=imagelist,
                                                return_df = False)[0][::-1][0]
        else:
            ref_image_conv = imagelist[0]

    
    if isinstance(ref_image_conv, str):
        psf_zise = int(get_beam_size_px(ref_image_conv)[0])
        psf_image_size = int(load_fits_data(ref_image_conv).shape[0])
        print(f"PSF BEAM SIZE is --> {psf_zise} px")
        print(f"PSF IMAGE SIZE is --> {psf_image_size} px")
        larger_beam_image_data = load_fits_data(ref_image_conv)
        

        psf_name = tcreate_beam_psf(ref_image_conv,
                                        size=(psf_image_size, psf_image_size),
                                        aspect = 'elliptical')
        PSF_DATA = load_fits_data(psf_name)
        
    num_images = cube_image.shape[2]
    if needs_convolution:
        conv_cube = np.empty_like(cube_image)
        conv_cube_res = np.empty_like(cube_residuals)
        if conv_task == 'fft':
            for i in tqdm(range(num_images)):
                conv_image_i_uncor = scipy.signal.fftconvolve(cube_image[:, :, i], 
                                                            PSF_DATA, 'same')
                conv_residual_i_uncor = scipy.signal.fftconvolve(cube_residuals[:, :, i], 
                                                            PSF_DATA, 'same')
                
                larger_beam_area = beam_area2(ref_image_conv)
                beam_area_i = beam_area2(imagelist[i])
                
                # factor_conv = np.max(larger_beam_image_data)/np.max(conv_image_i_uncor)
                factor_conv_i = larger_beam_area/beam_area_i
                
                print(f"Factor Convolution is --> {factor_conv_i}")
                # factor_conv = 1.0

                conv_cube[:, :, i] = conv_image_i_uncor * factor_conv_i
                conv_image_i = conv_cube[:, :, i]
                
                conv_cube_res[:, :, i] = conv_residual_i_uncor * factor_conv_i
                conv_res_i = conv_cube_res[:, :, i]
                
                conv_name = imagelist[i].replace('-image.','-image-conv.').replace('-image.cutout','-image-conv.cutout')
                pf.writeto(conv_name,
                                conv_image_i,
                                overwrite=True)
                copy_header(ref_image_conv,conv_name,conv_name)
                
                conv_res_name = residuallist[i].replace('-residual.','-residual-conv.').replace('-residual.cutout','-residual-conv.cutout')
                pf.writeto(conv_res_name,
                                conv_res_i,
                                overwrite=True)
                copy_header(ref_image_conv,conv_res_name,conv_res_name)
        # if conv_task == 'imsmooth':
        #     if isinstance(imagelist[0], str):
        #         for i in tqdm(range(num_images)):
        #             conv_name = imagelist[i].replace('-image.','-image-conv.').replace('-image.cutout','-image-conv.cutout')
        #             pf.writeto(conv_name,
        #                             cube_image[:, :, i],
        #                             overwrite=True)
        #             copy_header(imagelist[0],conv_name,conv_name)
                    
        #             convolve_2D_smooth(imagelist[0],conv_name,mode='transfer',add_prefix='')
        #             conv_cube[:,:,i] = load_fits_data(conv_name)
            
    else:
        conv_cube = cube_image.copy()
        conv_cube_res = cube_residuals.copy()
    
    # conv_cube_res = cube_residuals.copy()
    masked_cube = np.where(inv_mask_3d, np.nan, conv_cube)
    masked_cube_res = np.where(inv_mask_3d, np.nan, conv_cube_res)
    
    # cube_image * mask_3d
    idx = np.column_stack(np.where(mask==True))
    if freqs is None:
        freqs = getfreqs(imagelist)
    alphaimage = np.empty_like(conv_cube[:,:,0])
    alphaimage_error = np.empty_like(conv_cube[:,:,0])
    A_sy_map = np.empty_like(conv_cube[:,:,0])
    A_sy_map_err = np.empty_like(conv_cube[:,:,0])
    A_ff_map = np.empty_like(conv_cube[:,:,0])
    A_ff_map_err = np.empty_like(conv_cube[:,:,0])
    Snu0 = np.empty_like(conv_cube[:,:,0])
    Snu0_err = np.empty_like(conv_cube[:,:,0])
    f_th_33 = np.empty_like(conv_cube[:,:,0])
    f_th_33_err = np.empty_like(conv_cube[:,:,0])
    S_tot_33 = np.empty_like(conv_cube[:,:,0])
    S_tot_33_err = np.empty_like(conv_cube[:,:,0])

    
    alphaimage[:] = np.nan
    alphaimage_error[:] = np.nan
    A_sy_map[:] = np.nan
    A_sy_map_err[:] = np.nan
    A_ff_map[:] = np.nan
    A_ff_map_err[:] = np.nan
    f_th_33[:] = np.nan
    f_th_33_err[:] = np.nan
    Snu0[:] = np.nan
    Snu0_err[:] = np.nan
    S_tot_33[:] = np.nan
    S_tot_33_err[:] = np.nan

    
#     x = np.log10(freqs)
    x = freqs.copy()/1e9
    
    nspec = len(idx)
    tenpc = int(nspec / 10.0)
    count = 0
    pcount = 0
    
    masked_cube = masked_cube.astype(np.float32)
    masked_cube_res = masked_cube_res.astype(np.float32)
    
    
    def compute_pixel_nth_spectral_index(i, j, x, masked_cube, masked_cube_res, nu0):
        y = masked_cube[i, j, :] * 1000
        yerr = np.sqrt((masked_cube_res[i, j, :])**2.0 + (flux_sys_error_frac * masked_cube[i, j, :])**2.0) * 1000
        results_fit = do_fit_spec_SY_FF_map(x, y, yerr, nu0)
        return (i, j, 
                results_fit.params['alpha_nt'].value, 
                results_fit.params['alpha_nt'].stderr,
                results_fit.params['A_sy'].value,
                results_fit.params['A_sy'].stderr,
                results_fit.params['A_ff'].value,
                results_fit.params['A_ff'].stderr,
                y,yerr,results_fit
                )
    
    from joblib import Parallel, delayed
    
    pixel_indices = [(i, j) for i, j in idx]
    results = Parallel(n_jobs=n_jobs)(
    delayed(compute_pixel_nth_spectral_index)(i, j, x, masked_cube, masked_cube_res, nu0) 
    for i, j in tqdm(pixel_indices, total=len(pixel_indices))
    )
    
    if sed_model == 'S2':
        # for i, j in tqdm(idx):
        # for i, j in idx:
        for (i, j, alpha_nt_value, alpha_nt_err,
             A_sy_value,A_sy_err,
             A_ff_value,A_ff_err,
             y,yerr,results_fit) in results:
            # if count == 0:
            #     print(str(pcount) + '%...')
            
            
    #         y = np.log10(masked_cube[i,j,:])
            # y = masked_cube[i,j,:]*1000
            # yerr = masked_cube_res[i,j,:]*1000
        
            # results_fit = do_fit_spec_SY_FF_map(x,y,yerr,nu0)
            
            alphaimage[i, j] = alpha_nt_value
            alphaimage_error[i, j] = alpha_nt_err
            A_sy_map[i, j] = A_sy_value
            A_sy_map_err[i, j] = A_sy_err
            A_ff_map[i, j] = A_ff_value
            A_ff_map_err[i, j] = A_ff_err
            
            Snu0[i,j] = RC_function_SY_FF(nu0, 
                                    A_sy_value, 
                                    A_ff_value, 
                                    alpha_nt_value,
                                    nu0)
            S_tot_33[i,j] = RC_function_SY_FF(33, 
                                    A_sy_value, 
                                    A_ff_value, 
                                    alpha_nt_value,
                                    nu0)
            
            S_tot_33_err = RC_function_SY_FF(33, 
                                    A_sy_err, 
                                    A_ff_err, 
                                    alpha_nt_value,
                                    nu0)
            
            S_ff_33 = RC_function_SY_FF(33, 
                                    0.0, 
                                    A_ff_value, 
                                    0.0,
                                    nu0)
            S_ff_33_err = RC_function_SY_FF(33, 
                                    0.0, 
                                    A_ff_err, 
                                    alpha_nt_value,
                                    nu0)
            
            f_th_33[i,j] = S_ff_33 / S_tot_33[i,j]
            
            f_th_33_err[i,j] = f_th_33[i,j] * np.sqrt((S_ff_33_err/S_ff_33)**2.0 + (S_tot_33_err/S_tot_33[i,j])**2.0)
            
            try:
                Snu0_err[i,j] = RC_function_SY_FF(nu0, 
                                        A_sy_err, 
                                        A_ff_err, 
                                        alpha_nt_value,
                                        nu0)
            except:
                Snu0_err[i,j] = np.nan                 
                
            count += 1
            if count == tenpc:
                count = 0
                pcount += 10
                if verbose>0:
                    model_best = RC_function_SY_FF(x, 
                                                A_sy_value, 
                                                A_ff_value,
                                                alpha_nt_value,
                                                nu0
                                                )
                    
                    model_best_sy = RC_function_SY_FF(x, 
                                                A_sy_value, 
                                                0.0,
                                                alpha_nt_value,
                                                nu0
                                                )
                    model_best_ff = RC_function_SY_FF(x, 
                                                0.0, 
                                                A_ff_value,
                                                alpha_nt_value,
                                                nu0
                                                )
                    
                    
                    plt.figure()

                    plt.errorbar(x, 
                                y, 
                                yerr=abs(yerr), 
                                fmt='o',
                                label='Data', color='k', ecolor='gray',
                                alpha=0.5)
                    plt.plot(x, model_best,
                            color='red', ls='-.', label='Best-fit model')
                    plt.plot(x, model_best_sy,
                            color='blue', ls='-.', label='SY term')
                    plt.plot(x, model_best_ff,
                            color='orange', ls='-.', label='FF term')
                    
                    
                    plt.ylim(abs(np.nanmin(y)*0.1),np.nanmax(y)*5)
                    plt.semilogy()
                    plt.semilogx()
                    
                    plt.legend()
                    plt.show()
                    print(lmfit.fit_report(results_fit.params))
        
    
    if isinstance(imagelist[0], str):
        """
        Only save the alpha image if the input is a list of strings.
        """
        # alphaimage_name = imagelist[0].replace('-0000-image.fits','-alpha.fits')
        # alphaimage_error_name = imagelist[0].replace('-0000-image.fits','-alphaerror.fits')
        _alphaimage_name = (ref_image_conv.replace('-image.','-alpha_nt.').
                        replace('-image.cutout','-alpha_nt.cutout').
                        replace('-image-pb','-alpha_nt-pb'))
        
        _alphaimage_error_name = (ref_image_conv.replace('-image.','-alpha_nt_error.').
                                replace('-image.cutout','-alpha_nt_error.cutout').
                                replace('-image-pb','-alpha_nt_error-pb'))
        
        alphaimage_name = os.path.dirname(os.path.dirname(_alphaimage_name))+'/'+os.path.basename(_alphaimage_name)
        alphaimage_error_name = os.path.dirname(os.path.dirname(_alphaimage_error_name))+'/'+os.path.basename(_alphaimage_error_name)

        pf.writeto(alphaimage_name,alphaimage,overwrite=True)
        pf.writeto(alphaimage_error_name,alphaimage_error,overwrite=True)
        copy_header(ref_image_conv,alphaimage_name,alphaimage_name)
        copy_header(ref_image_conv,alphaimage_error_name,alphaimage_error_name)
    
    return(alphaimage, alphaimage_error,Snu0,Snu0_err,
           A_ff_map, A_ff_map_err, 
           A_sy_map, A_sy_map_err, 
           f_th_33,f_th_33_err,
           conv_cube,masked_cube_res,masked_cube)
    
    
    
    


"""
 ____  _       _   _   _
|  _ \| | ___ | |_| |_(_)_ __   __ _
| |_) | |/ _ \| __| __| | '_ \ / _` |
|  __/| | (_) | |_| |_| | | | | (_| |
|_|   |_|\___/ \__|\__|_|_| |_|\__, |
                               |___/
Plotting Functions
"""

class CustomFormatter(mticker.ScalarFormatter):
    def __init__(self, factor=1, **kwargs):
        self.factor = factor
        mticker.ScalarFormatter.__init__(self, **kwargs)

    def __call__(self, x, pos=None):
        x = x * self.factor
        if x == 0:
            return "0.00"
        return "{:.2f}".format(x)


def make_scalebar(ax, left_side, length, color='w', linestyle='-', label='',
                  fontsize=12, text_offset=0.1*u.arcsec):
    axlims = ax.axis()
    lines = ax.plot(u.Quantity([left_side.ra, left_side.ra-length]),
                    u.Quantity([left_side.dec]*2),
                    color=color, linestyle=linestyle, marker=None,
                    transform=ax.get_transform('fk5'),
                   )
    txt = ax.text((left_side.ra-length/2).to(u.deg).value,
                  (left_side.dec+text_offset).to(u.deg).value,
                  label,
                  verticalalignment='bottom',
                  horizontalalignment='center',
                  transform=ax.get_transform('icrs'),
                  color=color,
                  fontsize=fontsize,
                 )
    ax.axis(axlims)
    return lines,txt


def plot_radial_profile(imagedatas, refimage=None,
                        ax=None, centre=None, labels=None,
                        line_styles=None,
                        figsize=(5, 5)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    else:
        pass

    if labels is None:
        _labels = [''] * len(imagedatas)
    else:
        _labels = labels

    if refimage != None:
        cell_size = get_cell_size(refimage)
        xaxis_units = '[arcsec]'
    else:
        cell_size = 1.0
        xaxis_units = '[px]'

    for i in range(len(imagedatas)):
        radius, intensity = get_profile(imagedatas[i], center=centre)
        try:
            ax.plot(radius * cell_size, intensity, 
                    label=_labels[i],
                    color=line_styles['color'][i],
                    linestyle=line_styles['linestyle'][i],
                    linewidth=line_styles['linewidth'][i],
                    )
        except:
            ax.plot(radius * cell_size, abs(intensity), label=_labels[i])
        

#     plt.semilogy()
    plt.xlabel(f"Radius {xaxis_units}")
    plt.ylabel(f"Radial Intensity [mJy/beam]")
    # plt.xlim(0,cell_size*radius[int(len(radius)/2)])
    # plt.semilogx()
    if labels != None:
        plt.legend()
    return (ax)


def azimuthal_average_profile_with_shaded_errors(image, rms_image, center, sigma=1, bin_size=1.0,
                                                log_scale=True,cell_size=1.0,
                                                figsize=(5, 5),
                                                ylabel='Azimuthal Average Intensity $I(R)$ [Jy/Beam]',
                                                xlabel='Radius [pixels]',
                                                title=None
                                                ):
    """
    Calculate the azimuthal average profile of an image and plot the errors using shaded regions.
    
    Parameters:
    - image: 2D numpy array, the intensity image.
    - rms_image: 2D numpy array, the rms (error) image.
    - center: tuple (x, y), the coordinates of the center (in pixels).
    - sigma: float, the number of standard deviations for the shaded error regions.
    - bin_size: float, size of the radial bins for averaging.

    Returns:
    - radii: numpy array, the radial distances.
    - profile: numpy array, the azimuthal average of intensity at each radius.
    - error_rms: numpy array, the error at each radius based on the rms image.
    - error_std: numpy array, the error at each radius based on the standard deviation.
    """
    
    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Define bin edges and centers
    r_max = np.max(r)
    bin_edges = np.arange(0, r_max + bin_size, bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create arrays to hold the azimuthal average and errors
    profile = np.zeros_like(bin_centers)
    error_rms = np.zeros_like(bin_centers)
    error_std = np.zeros_like(bin_centers)

    for i in range(len(bin_centers)):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])

        if np.any(mask):
            profile[i] = np.mean(image[mask])
            error_rms[i] = np.sqrt(np.mean(rms_image[mask]**2)) # Use RMS image for errors
            error_std[i] = np.std(image[mask])                  # Use std deviation of pixel intensities        
        # if np.any(mask):
        #     profile[i] = np.nanmean(image[mask])*np.nansum(mask)
        #     error_rms[i] = np.sqrt(np.mean(rms_image[mask]**2))*np.nansum(mask)  # Use RMS image for errors
        #     error_std[i] = np.nanstd(image[mask])*np.nansum(mask)  # Use std deviation of pixel intensities

        # if np.any(mask):
        #     profile[i] = np.nansum(image[mask])
        #     error_rms[i] = np.sqrt(np.mean(rms_image[mask]**2))  # Use RMS image for errors
        #     error_std[i] = np.nanstd(image[mask])  # Use std deviation of pixel intensities

    nans_mask = ~np.isnan(profile)
    bin_centers = bin_centers[nans_mask]
    profile = profile[nans_mask]
    error_rms = error_rms[nans_mask]
    error_std = error_std[nans_mask]
    
    
    # Plotting the results with shaded error regions
    plt.figure(figsize=figsize)
    
    # Plot the profile
    plt.plot(bin_centers*cell_size, profile, 
            #  label='Profile', 
             color='black', linestyle='-', linewidth=2)

    # Plot shaded error regions
    plt.fill_between(bin_centers*cell_size, profile - sigma * error_rms, profile + sigma * error_rms, 
                     color='gray', alpha=0.3, 
                    #  label=f'RMS Errors ($\pm{sigma}\sigma$)'
                    )
    
    # plt.fill_between(bin_centers*cell_size, profile - sigma * error_std, profile + sigma * error_std, 
    #                  color='red', alpha=0.3, 
    #                  label=f'STD Errors ($\pm{sigma}\sigma$)'
    #                 )
    if log_scale:
        plt.yscale('log')
    plt.xlabel(fr'{xlabel}', fontsize=14)
    plt.ylabel(fr'{ylabel}', fontsize=14)
    plt.title(title)
    # plt.legend(loc='upper right')
    # plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.show()

    return bin_centers, profile, error_rms, error_std


def fast_plot2(imagename, crop=False, box_size=128, center=None, with_wcs=True,vmax_factor=0.5,
               vmin_factor=1, plot_colorbar=True, figsize=(5, 5), aspect=1, ax=None):
    """
    Fast plotting of an astronomical image with/or without a wcs header.

    imagename:
        str or 2d array.
        If str (the image file name), it will attempt to read the wcs and plot the coordinates axes.

        If 2darray, will plot the data with generic axes.

        support functions:
            load_fits_data() -> casa to numpy: A function designed mainly to read CASA fits images,
                     but can be used to open any fits images.

                     However, it does not read header/wcs.
                     Note: THis function only works inside CASA environment.




    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(1,1,1)
    #     try:
    if isinstance(imagename, str) == True:
        if with_wcs == True:
            hdu = pf.open(imagename)
            #         hdu=pf.open(img)
            ww = WCS(hdu[0].header, naxis=2)
            try:
                if len(np.shape(hdu[0].data) == 2):
                    g = hdu[0].data[0][0]
                else:
                    g = hdu[0].data
            except:
                g = load_fits_data(imagename)
        if with_wcs == False:
            g = load_fits_data(imagename)

        if crop == True:
            xin, xen, yin, yen = do_cutout(imagename, box_size=box_size, center=center, return_='box')
            g = g[xin:xen, yin:yen]

    else:
        g = imagename

    if crop == True:
        max_x, max_y = np.where(g == g.max())
        xin = max_x[0] - box_size
        xen = max_x[0] + box_size
        yin = max_y[0] - box_size
        yen = max_y[0] + box_size
        g = g[xin:xen, yin:yen]

    if mad_std(g) == 0:
        std = g.std()
    else:
        std = mad_std(g)
    if ax is None:
        if with_wcs == True and isinstance(imagename, str) == True:
            ax = fig.add_subplot(projection=ww.celestial)
            ax.set_xlabel('RA')
            ax.set_ylabel('DEC')
        else:
            ax = fig.add_subplot()
            ax.set_xlabel('x pix')
            ax.set_ylabel('y pix')

    vmin = vmin_factor * std

    #     print(g)
    vmax = vmax_factor * g.max()

    norm = simple_norm(g, stretch='sqrt', asinh_a=0.02, min_cut=vmin, max_cut=vmax)

    im_plot = ax.imshow((g), cmap='magma_r', origin='lower', alpha=1.0, norm=norm,
                        aspect=aspect)  # ,vmax=vmax, vmin=vmin)#norm=norm
    #     ax.set_title('Image')
    try:
        levels_g = np.geomspace(5.0 * g.max(), 0.1 * g.max(), 7)
        #     x = np.geomspace(1.5*mad_std(g),10*mad_std(g),4)
        levels_black = np.geomspace(3 * (mad_std(g) + 0.00001), 0.1 * g.max(), 7)
    except:
        try:
            levels_g = np.geomspace(5.0 * g.max(), 3 * (mad_std(g), 7))
            levels_black = np.asarray([0])
        except:
            levels_g = np.asarray([0])
            levels_black = np.asarray([0])
    #     xneg = np.geomspace(5*mad_std(g),vmin_factor*mad_std(g),2)
    #     y = -xneg[::-1]
    #     levels_black = np.append(y,x)

    #     levels_white = np.geomspace(g.max(),10*mad_std(g),7)
    # levels_white = np.geomspace(g.max(), 0.1 * g.max(), 5)

    #     cg.show_contour(data, colors='black',levels=levels_black,linestyle='.-',linewidths=0.2,alpha=1.0)
    #     cg.show_contour(data, colors='#009E73',levels=levels_white[::-1],linewidths=0.2,alpha=1.0)
    try:
        ax.contour(g, levels=levels_black, colors='black', linewidths=0.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)
        #     ax.contour(g, levels=levels_white[::-1],colors='#009E73',linewidths=0.2,alpha=1.0)#cmap='Reds', linewidths=0.75)
        ax.contour(g, levels=levels_g[::-1], colors='white', linewidths=0.6, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    except:
        print('Not plotting contours!')
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    try:
        cb = plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91, 0.08, 0.05, 0.84]))
        cb.set_label(r"Flux [Jy/Beam]")
    except:
        pass
    # if ax==None:
    #     if plot_colorbar==True:
    #         # cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.08,0.05,0.82]))
    #         cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.08,0.05,0.32]))
    #         # cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))
    #         cb.set_label(r"Flux [Jy/Beam]")
    return (ax)


def plot_flux_petro(imagename, flux_arr, r_list,
                    savefig=True, show_figure=True,
                    add_save_name = ''):
    plt.figure(figsize=(4, 3))
    cell_size = get_cell_size(imagename)
    plt.plot(r_list * cell_size, 1000 * flux_arr / beam_area2(imagename),
             color='black', lw='3')
    idx_lim = int(np.where(flux_arr / np.max(flux_arr) > 0.95)[0][0] * 1.5)
    plt.grid()
    plt.xlabel('Aperture Radii [arcsec]')
    plt.ylabel(r'$S_\nu$ [mJy]')
    plt.title('Curve of Growth')
    try:
        plt.xlim(0, r_list[idx_lim] * cell_size)
    except:
        plt.xlim(0, r_list[-1] * cell_size)
    if savefig is True:
        plt.savefig(
            imagename.replace('.fits', '_flux_aperture_'+add_save_name+'.jpg'),
            dpi=300, bbox_inches='tight')
    if show_figure == True:
        plt.show()
    else:
        plt.close()

def make_cl(image):
    std = mad_std(image)
    levels = np.geomspace(image.max() * 5, 7 * std, 10)
    return (levels[::-1])


def plot_slices_fig(data_2D, show_figure=True, label='',color=None,FIG=None,linestyle='--.'):
    plot_slice = np.arange(0, data_2D.shape[0])

    if FIG is None:
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
    else:
        fig,ax1,ax2 = FIG
    ax1.plot(plot_slice, np.mean(data_2D, axis=0), linestyle=linestyle, color=color, ms=14,
             label=label)
    ax1.legend(fontsize=11)
    ax1.grid()
    ax1.set_ylabel('mean $x$ direction')
    ax1.set_xlim(data_2D.shape[0] / 2 - 0.25 * data_2D.shape[0],
                 data_2D.shape[0] / 2 + 0.25 * data_2D.shape[0])

    ax2.plot(plot_slice, np.mean(data_2D, axis=1), linestyle=linestyle, color=color, ms=14,
             label=label)
    ax2.set_xlabel('Image Slice [px]')
    ax2.set_ylabel('mean $y$ direction')
    ax2.set_xlim(data_2D.shape[0] / 2 - 0.25 * data_2D.shape[0],
                 data_2D.shape[0] / 2 + 0.25 * data_2D.shape[0])
    ax2.grid()
    # plt.semilogx()
    # plt.xlim(300,600)
    # if image_results_conv is not None:
    #     plt.savefig(
    #         image_results_conv.replace('.fits', 'result_lmfit_slices.pdf'),
    #         dpi=300, bbox_inches='tight')
    #     if show_figure == True:
    #         plt.show()
    #     else:
    #         plt.close()
    return(fig,ax1,ax2)

def plot_slices(data_2D, residual_2D, model_dict, image_results_conv=None,
                Rp_props=None, show_figure=True):
    plot_slice = np.arange(0, data_2D.shape[0])
    if Rp_props is not None:
        plotlim = Rp_props['c' + str(1) + '_rlast']
        # plotlim = 0
        # for i in range(Rp_props['ncomps']):
        #     plotlim = plotlim + Rp_props['c' + str(i + 1) + '_rlast']

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    norm_plot = np.max(np.mean(data_2D, axis=0))
    ax1.plot(plot_slice, np.mean(data_2D, axis=0)/norm_plot, '--.', color='purple', ms=14,
             label='DATA')
    ax1.plot(plot_slice, np.mean(model_dict['model_total_conv']/norm_plot, axis=0), '.-',
             color='limegreen', linewidth=4, label='MODEL')
    ax1.plot(plot_slice, np.mean(model_dict['best_residual_conv']/norm_plot, axis=0), '.-',
             color='black', linewidth=4, label='RESIDUAL')
    try:
        ax1.plot(plot_slice, np.mean(residual_2D, axis=0)/norm_plot, '.-', color='grey',
                linewidth=4, label='MAP RESIDUAL')
    except:
        pass
    #     ax1.set_xlabel('$x$-slice')
    #     ax1.set_xaxis('off')
    #     ax1.set_xticks([])
    ax1.legend(fontsize=11)
    ax1.grid()
    ax1.set_ylabel('fractional mean $x$ direction')
    if Rp_props is not None:
        ax1.set_xlim(Rp_props['c1_x0c'] - plotlim, Rp_props['c1_x0c'] + plotlim)
    #     ax1.set_title('asd')
    # plt.plot(np.mean(shuffled_image,axis=0),color='red')

    norm_plot = np.max(np.mean(data_2D, axis=1))
    ax2.plot(plot_slice, np.mean(data_2D, axis=1)/norm_plot, '--.', color='purple', ms=14,
             label='DATA')
    ax2.plot(plot_slice, np.mean(model_dict['model_total_conv']/norm_plot, axis=1), '.-',
             color='limegreen', linewidth=4, label='MODEL')
    ax2.plot(plot_slice, np.mean(model_dict['best_residual_conv']/norm_plot, axis=1), '.-',
             color='black', linewidth=4, label='RESIDUAL')
    try:
        ax2.plot(plot_slice, np.mean(residual_2D, axis=1)/norm_plot, '.-', color='grey',
                linewidth=4, label='MAP RESIDUAL')
    except:
        pass
    ax2.set_xlabel('Image Slice [px]')
    ax2.set_ylabel('fractional mean $y$ direction')
    if Rp_props is not None:
        ax2.set_xlim(Rp_props['c1_y0c'] - plotlim, Rp_props['c1_y0c'] + plotlim)
    ax2.grid()
    # plt.semilogx()
    # plt.xlim(300,600)
    if image_results_conv is not None:
        plt.savefig(
            image_results_conv.replace('.fits', 'result_lmfit_slices.pdf'),
            dpi=300, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()


def plot_fit_results(imagename, model_dict, image_results_conv,
                     sources_photometies,bkg_image=None,vmax_factor=0.1,data_2D_=None,
                     mask = None,
                     vmin_factor=3, obs_type = 'radio',
                     show_figure=True,crop=False,box_size=100):
    if data_2D_ is not None:
        data_2D = data_2D_
    else:
        data_2D = load_fits_data(imagename)

    plot_data_model_res(data_2D, modelname=model_dict['model_total_conv'],
               residualname=model_dict['best_residual_conv'],
               reference_image=imagename,
               NAME=image_results_conv[-2].replace('.fits',
                                                   '_data_model_res'),
               crop=crop, vmin_factor=vmin_factor,obs_type = obs_type,
               box_size=box_size)



    ncomponents = sources_photometies['ncomps']
    if sources_photometies is not None:
        plotlim =  4.0 * sources_photometies['c'+str(int(ncomponents))+'_Rp']
        if plotlim > data_2D.shape[0]/2:
            plotlim = data_2D.shape[0]/2
        # plotlim = 0
        # for i in range(ncomponents):
        #     plotlim = plotlim + sources_photometies['c' + str(i + 1) + '_rlast']

    model_name = image_results_conv[-2]
    residual_name = image_results_conv[-1]
    cell_size = get_cell_size(imagename)
    profile_data = {}
    # center = get_peak_pos(imagename)
    center = nd.maximum_position(data_2D*mask)[::-1]
    for i in range(ncomponents):
        component_name = image_results_conv[
            i]  # crop_image.replace('.fits','')+"_"+str(ncomponents)+"C_model_component_"+str(i+1)+special_name+'_IMFIT_opt.fits'
        Ir_r = get_profile(component_name,
                           center=center)
        profile_data['r' + str(i + 1)], profile_data['Ir' + str(i + 1)], \
        profile_data['c' + str(i + 1) + '_name'] = Ir_r[0], Ir_r[
            1], component_name

    r, ir = get_profile(data_2D, center=center)
    rmodel, irmodel = get_profile(model_name, center=center)
    rre, irre = get_profile(residual_name, center=center)
    if bkg_image is not None:
        r_bkg, ir_bkg = get_profile(bkg_image,center=center)

    # plt.plot(radiis[0],profiles[0])
    # plt.plot(radiis[1],profiles[1])
    # plt.plot(radiis[2],np.log(profiles[2]))
    # colors = ['black','purple','gray','red']
    colors = ['red', 'blue', 'teal', 'brown', 'cyan','orange','forestgreen',
              'pink', 'slategrey','darkseagreen','peru','royalblue','darkorange']

    plt.figure(figsize=(5, 5))
    plt.plot(r * cell_size, abs(ir), '--.', ms=10, color='purple', alpha=1.0,
             label='DATA')
    for i in range(ncomponents):
        #     try:
        #         plt.plot(profile_data['r'+str(i+1)],abs(profile_data['Ir'+str(i+1)])[0:r.shape[0]],'--',label='comp'+str(i+1),color=colors[i])
        plt.plot(profile_data['r' + str(i + 1)] * cell_size,
                 abs(profile_data['Ir' + str(i + 1)]), '--',
                 label='COMP_' + str(i + 1), color=colors[i])
    #     except:
    #         pass
    if bkg_image is not None:
        ir_model_data =  irmodel# + ir_bkg
        plt.plot(r_bkg * cell_size, abs(ir_bkg), '--', label='bkg', color='brown')
    else:
        ir_model_data = irmodel

    plt.plot(r * cell_size, abs(irre), '.-', label='RESIDUAL', color='black')
    plt.plot(r * cell_size, abs(ir_model_data), '--', color='limegreen', label='MODEL',
             linewidth=4)
    plt.semilogy()
    if obs_type == 'radio':
        plt.xlabel(r'Projected Radius $R$ [arcsec]')
        plt.ylabel(r'Radial Intensity $I(R)$ [Jy/beam]')
    else:
        plt.xlabel(r'Projected Radius $R$ [px]')
        plt.ylabel(r'Radial Intensity $I(R)$')
        # plt.semilogx()
    plt.legend(fontsize=11)
    plt.ylim(1e-7, -0.05 * np.log(ir[0]))
    # plt.xlim(0,3.0)
    plt.grid()
    if sources_photometies is not None:
        plt.xlim(0, plotlim * cell_size)
        idRp_main = int(sources_photometies['c1_Rp'])
        plt.axvline(r[idRp_main] * cell_size)
    plt.savefig(image_results_conv[-2].replace('.fits', 'result_lmfit_IR.pdf'),
                dpi=300, bbox_inches='tight')
    if show_figure == True:
        plt.show()
        # return(plt)
    else:
        plt.close()




# plt.savefig(config_file.replace('params_imfit.csv','result_lmfit_py_IR.pdf'),dpi=300, bbox_inches='tight')

def total_flux(data2D,image,mask=None,BA=None,
               sigma=6,iterations=3,dilation_size=7,PLOT=False,
               silent=True):

    if BA is None:
        try:
            BA = beam_area2(image)
        except:
            print('WARNING: Beam area not found, setting to 1.')
            BA = 1
    else:
        BA = BA
    if mask is None:
        _,mask = mask_dilation(data2D,sigma=sigma,iterations=iterations,
                               dilation_size=dilation_size,PLOT=PLOT)
    else:
        mask = mask
#     tf = np.sum((data2D> sigma*mad_std(data2D))*data2D)/beam_area2(image)
    blank_sum = np.sum(data2D)/BA
    sum3S = np.sum(data2D*(data2D> 3.0*mad_std(data2D)))/BA
    summask = np.sum(data2D*mask)/BA
    if silent==False:
        print('Blank Sum   = ',blank_sum)
        print('Sum  3sigma = ',sum3S)
        print('Sum mask    = ',summask)
    return(summask)

def total_flux_faster(data2D,mask):
#     tf = np.sum((data2D> sigma*mad_std(data2D))*data2D)/beam_area2(image)
    summask = np.sum(data2D*mask)
    return(summask)


def plot_decomp_results(imagename,compact,extended_model,data_2D_=None,
                        vmax_factor=0.5,vmin_factor=3,rms=None,
                        figsize=(13,13),nfunctions=None,
                        obs_type = 'radio',
                        special_name=''):

    decomp_results = {}
    if rms is None:
        rms = mad_std(load_fits_data(imagename))
    else:
        rms = rms

    max_factor =  load_fits_data(imagename).max()
#     compact = model_dict['model_c1_conv']
    if data_2D_ is not None:
        data_2D = data_2D_
    else:
        data_2D = load_fits_data(imagename)

    # rms_std_data = mad_std(data_2D)
    extended = data_2D  - compact
    if nfunctions == 1:
        residual_modeling = data_2D - (compact)
    else:
        residual_modeling = data_2D - (compact + extended_model)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(3, 3, 1)
    # ax.yaxis.set_ticks([])
    ax1 = eimshow(imagename,ax=ax1,fig=fig,rms=rms,plot_title='Total Emission',
                 vmax_factor=vmax_factor,vmin_factor=vmin_factor)
    # cb = plt.colorbar(mappable=plt.gca().images[0],
    #                           cax=fig.add_axes([0.9, 0.65, 0.02, 0.2]))
    # ax.yaxis.set_ticks([])
    ax1.axis('off')
    ax2 = fig.add_subplot(3, 3, 2)
    # ax.yaxis.set_ticks([])
    ax2 = eimshow(compact,ax=ax2,fig=fig,rms=rms,
                 plot_title='Compact Emission',
                 vmax_factor=vmax_factor,vmin_factor=vmin_factor)
    # ax.yaxis.set_ticks([])
    ax2.axis('off')

    ax3 = fig.add_subplot(3, 3, 3)
    # ax.yaxis.set_ticks([])
    ax3 = eimshow(extended,ax=ax3,fig=fig,rms=rms,
                 plot_title='Diffuse Emission',vmax_factor=vmax_factor,
                 vmin_factor=vmin_factor)
    # ax.yaxis.set_ticks([])
    # cb = plt.colorbar(mappable=plt.gca().images[0],
    #                           cax=fig.add_axes([0.00, 0.65, 0.02, 0.2]))
    ax3.axis('off')

    ax4 = fig.add_subplot(3, 3, 4)
    slice_ext = np.sqrt(np.mean(extended,axis=0)**2.0 + np.mean(extended,axis=1)**2.0)
    if nfunctions == 1:
        slice_ext_model = np.sqrt(
            np.mean(residual_modeling, axis=0) ** 2.0 + np.mean(residual_modeling,
                                                             axis=1) ** 2.0)
    else:
        slice_ext_model = np.sqrt(
            np.mean(extended_model, axis=0) ** 2.0 + np.mean(extended_model,
                                                             axis=1) ** 2.0)
    slice_data = np.sqrt(np.mean(data_2D,axis=0)**2.0 + np.mean(data_2D,axis=1)**2.0)
    ax4.plot(slice_ext,label='COMPACT SUB')
    ax4.plot(slice_data,label='DATA')
    ax4.plot(slice_ext_model, label='EXTENDED MODEL')
    ax4.legend(prop={'size': 11})
    xlimit = [data_2D.shape[0] / 2 - 0.15 * data_2D.shape[0],
              data_2D.shape[0] / 2 + 0.15 * data_2D.shape[0]]
    ax4.set_xlim(xlimit[0],xlimit[1])
    # ax.semilogx()

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.axis('off')

    try:
        omaj, omin, _, _, _ = beam_shape(imagename)
        dilation_size = int(
            np.sqrt(omaj * omin) / (2 * get_cell_size(imagename)))
    except:
        dilation_size = 10

    _, mask_model_rms_self_compact = mask_dilation(compact,
                                           sigma=1, dilation_size=dilation_size,
                                           iterations=2,PLOT=False)
    _, mask_data = mask_dilation(data_2D, rms=rms,
                                           sigma=6, dilation_size=dilation_size,
                                           iterations=2,PLOT=False)
    _, mask_model_rms_self_extended = mask_dilation(extended,
                                           sigma=6, dilation_size=dilation_size,
                                           iterations=2,PLOT=False)

    _, mask_model_rms_image_compact = mask_dilation(compact,
                                            rms=rms,
                                            sigma=1, dilation_size=dilation_size,
                                            iterations=2,PLOT=False)
    _, mask_model_rms_image_extended = mask_dilation(extended,
                                            rms=rms,
                                            sigma=6, dilation_size=dilation_size,
                                            iterations=2,PLOT=False)
    if nfunctions == 1:
        _, mask_model_rms_image_extended_model = mask_dilation(residual_modeling,
                                                rms=rms,
                                                sigma=1, dilation_size=dilation_size,
                                                iterations=2,PLOT=False)
    else:
        _, mask_model_rms_image_extended_model = mask_dilation(extended_model,
                                                rms=rms,
                                                sigma=1, dilation_size=dilation_size,
                                                iterations=2,PLOT=False)

    try:
        beam_area_px = beam_area2(imagename)
    except:
        beam_area_px = 1
    if obs_type == 'radio':
        flux_scale_factor = 1000
    else:
        flux_scale_factor = 1
    # print('Flux on compact (self rms) = ',
    #       1000*np.sum(compact*mask_model_rms_self_compact)/beam_area_px)
    # print('Flux on compact (data rms) = ',
    #       1000 * np.sum(compact * mask_model_rms_image_compact) / beam_area_px)
    flux_density_compact = flux_scale_factor*np.sum(
        compact*mask_model_rms_image_compact)/beam_area_px
    if nfunctions == 1:
        flux_density_extended_model = flux_scale_factor * np.sum(
            residual_modeling * mask_data) / beam_area_px
    else:
        flux_density_extended_model = flux_scale_factor * np.sum(
            extended_model * mask_data) / beam_area_px

    flux_density_ext_old = flux_scale_factor*total_flux(extended,imagename,BA=beam_area_px,
                                       mask = mask_model_rms_image_extended)
    flux_density_ext = flux_scale_factor*np.sum(
        extended*mask_data)/beam_area_px

    flux_data = flux_scale_factor*total_flux(data_2D,imagename,BA=beam_area_px,
                                       mask = mask_data)
    flux_density_ext_self_rms = flux_scale_factor*total_flux(extended,imagename,BA=beam_area_px,
                                       mask = mask_model_rms_self_extended)

    if nfunctions==1:
        flux_res = flux_data - (flux_density_compact)
    else:
        flux_res = flux_data - (
                    flux_density_extended_model + flux_density_compact)

    # print('Flux on extended (self rms) = ',flux_density_ext_self_rms)
    # print('Flux on extended (data rms) = ',flux_density_ext_old)
    # print('Flux on extended2 (data rms) = ', flux_density_ext)
    # print('Flux on extended model (data rms) = ', flux_density_extended_model)
    # print('Flux on data = ', flux_data)
    # print('Flux on residual = ', flux_res)

    decomp_results['flux_data'] = flux_data
    decomp_results['flux_density_ext_old'] = flux_density_ext_old
    decomp_results['flux_density_ext'] = flux_density_ext
    decomp_results['flux_density_ext_self_rms'] = flux_density_ext_self_rms
    decomp_results['flux_density_extended_model'] = flux_density_extended_model
    decomp_results['flux_density_compact'] = flux_density_compact
    decomp_results['flux_density_model'] = (flux_density_compact +
                                            flux_density_extended_model)
    decomp_results['flux_density_res'] = flux_res



    # print('r_half_light (old vs new) = {:0.2f} vs {:0.2f}'.format(p.r_half_light, p_copy.r_half_light))
    ax5.annotate(r"$S_\nu^{\rm core-comp}/S_\nu^{\rm total}=$"+'{:0.2f}'.format(flux_density_compact/flux_data),
                (0.33, 0.32), xycoords='figure fraction', fontsize=18)
    ax5.annotate(r"$S_\nu^{\rm ext}/S_\nu^{\rm total}\ \ \ =$"+'{:0.2f}'.format(flux_density_ext/flux_data),
                (0.33, 0.29), xycoords='figure fraction', fontsize=18)
    ax5.annotate(r"$S_\nu^{\rm ext \ model}/S_\nu^{\rm total}\ \ \ =$"+'{:0.2f}'.format(flux_density_extended_model/flux_data),
                (0.33, 0.26), xycoords='figure fraction', fontsize=18)
    ax5.annotate(r"$S_\nu^{\rm res}/S_\nu^{\rm total}\ \ \ =$"+'{:0.2f}'.format(flux_res/flux_data),
                (0.33, 0.23), xycoords='figure fraction', fontsize=18)
    
    # plt.tight_layout()
    plt.savefig(
        imagename.replace('.fits', '_ext_vs_comp'+special_name+'.jpg'),
        dpi=300,
        bbox_inches='tight')

    # save_data = True
    # if save_data == True:
    if obs_type != 'radio':
        exteded_file_name = imagename.replace('.fits', '') + \
                            special_name + '_extended.fits'
        pf.writeto(exteded_file_name,extended,overwrite=True)
        copy_header(imagename,exteded_file_name)
        compact_file_name = imagename.replace('.fits', '') + \
                            special_name + '_conv_compact.fits'
        pf.writeto(compact_file_name,compact,overwrite=True)
        copy_header(imagename,compact_file_name)
        decomp_results['compact_model_image'] = compact_file_name
        decomp_results['extended_model_image'] = exteded_file_name
    

    return(decomp_results)



def plot_interferometric_decomposition(imagename0, imagename,
                                       modelname, residualname,
                                       crop=False, box_size=512,
                                       max_percent_lowlevel=99.0,
                                       max_percent_highlevel=99.9999,
                                       NAME=None, EXT='.pdf',
                                       vmin0=None,
                                       run_phase = '1st',
                                       vmin_factor=3,vmax_factor=0.1,
                                       SPECIAL_NAME='', show_figure=True):
    """

    """
    fig = plt.figure(figsize=(16, 16))
    try:
        g = pf.getdata(imagename)
        I1 = pf.getdata(imagename0)
        if len(np.shape(g) == 4):
            g = g[0][0]
            I1 = I1[0][0]
        m = pf.getdata(modelname)
        r = pf.getdata(residualname)
    except:
        I1 = load_fits_data(imagename0)
        g = load_fits_data(imagename)
        m = load_fits_data(modelname)
        r = load_fits_data(residualname)

    dx1 = g.shape[0]/2
    dx = g.shape[0]/2
    if crop == True:
        xin, xen, yin, yen = do_cutout(imagename, box_size=box_size,
                                       center=None, return_='box')
        #         I1 = I1[int(2*xin):int(xen/2),int(2*yin):int(yen/2)]
        I1 = I1[int(xin + box_size / 1.25):int(xen - box_size / 1.25),
             int(yin + box_size / 1.25):int(yen - box_size / 1.25)]
        dx1 = I1.shape[0]/2
        # g = g[xin:xen,yin:yen]
        # m = m[xin:xen,yin:yen]
        # r = r[xin:xen,yin:yen]

    if mad_std(I1) == 0:
        std0 = I1.std()
    else:
        std0 = mad_std(I1)

    if mad_std(g) == 0:
        std = g.std()
    else:
        std = mad_std(g)

    if mad_std(r) == 0:
        std_r = r.std()
    else:
        std_r = mad_std(r)

    if mad_std(m) == 0:
        std_m = m.std()
    else:
        std_m = mad_std(m)

    #     print(I1)
    vmin0 = 3 * std  # 0.5*g.min()#
    if vmin0 is not None:
        vmin0 = vmin0
    else:
        vmin0 = 3 * std

    vmax0 = 1.0 * g.max()
    vmin = vmin_factor * std  # 0.5*g.min()#
    vmax = vmax_factor * g.max()
    vmin_r = 0.5 * r.min()  # 1*std_r
    vmax_r = 1.0 * r.max()
    vmin_m = 1 * mad_std(m)  # vmin#0.01*std_m#0.5*m.min()#
    vmax_m = m.max()  # vmax#0.5*m.max()

    # levels_I1 = np.geomspace(2*I1.max(), 1.5 * np.std(I1), 6)
    levels_I1 = np.geomspace(2 * I1.max(), vmin0, 6)
    levels_g = np.geomspace(2*g.max(), 3 * std, 6)
    levels_m = np.geomspace(2*m.max(), 20 * std_m, 6)
    levels_r = np.geomspace(2*r.max(), 3 * std_r, 6)
    levels_neg = np.asarray([-3]) * std
    script_R = "\u211B"
    script_M = "\u2133"
    if run_phase == '1st':
        title_labels = [r"$I_1^{\rm mask}[\sigma_{\mathrm{opt}}]$",
                        r"$I_2$",
                        r""+script_M+r"$_{1,2} = I_{1}^{\rm mask}[\sigma_{\mathrm{"
                                     r"opt}}] * "
                                     r"\theta_2$",
                        r""+script_R+r"$_{1,2} = I_2 -$"+script_M+r"$_{1,2}$"
                        ]

    if run_phase == '2nd':
        # title_labels = [r""+script_R+r"$_{1,2}$",
        #                 r"$I_3$",
        #                 r"$I_{1}^{\rm mask} * \theta_3 + $"+script_R+r"$_{1,2} "
        #                                                              r"* \theta_3$",
        #                 r""+script_R+r"$_{T}$"
        #                 ]
        title_labels = [r""+script_R+r"$_{1,2}$",
                        r"$I_3$",
                        r""+script_M+"$_{1,3} + $"+script_M+r"$_{2,3}$",
                        r""+script_R+r"$_{T}$"
                        ]

    if run_phase == 'compact':
        title_labels = [r""+script_R+r"$_{1,2}$",
                        r"$I_3$",
                        r"$I_{1}^{\rm mask} * \theta_3$",
                        r"$I_3 - I_{1}^{\rm mask} * \theta_3$"
                        ]

    # colors = [(0, 0, 0), (1, 1, 1)]
    # cmap_name = 'black_white'
    # import matplotlib.colors as mcolors
    # cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors,
    #                                                N=len(levels_g))
    cm = 'gray'
    #     norm = simple_norm(g,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    norm = visualization.simple_norm(g, stretch='linear',
                                     max_percent=max_percent_lowlevel)
    norm0 = simple_norm(abs(I1), min_cut=0.5 * np.std(I1), max_cut=vmax,
                        stretch='sqrt')  # , max_percent=max_percent_highlevel)
    norm2 = simple_norm(abs(g), min_cut=vmin, max_cut=vmax,
                        stretch='asinh',asinh_a=0.02)  # , max_percent=max_percent_highlevel)
    CM = 'magma_r'
    ax = fig.add_subplot(1, 4, 1)

    #     im = ax.imshow(I1, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(I1, cmap='magma_r', extent=[-dx1,dx1,-dx1,dx1],
                        origin='lower', alpha=1.0, norm=norm0)

    ax.set_title(title_labels[0])

    cmap_magma_r = plt.cm.get_cmap('magma_r')
    # contour_palette = ['#000000', '#444444', '#888888', '#DDDDDD']
    # contour_palette = ['#000000', '#222222', '#444444', '#666666', '#888888',
    #                    '#AAAAAA', '#CCCCCC', '#EEEEEE', '#FFFFFF']
    contour_palette = ['#000000', '#444444', '#666666', '#EEEEEE', '#EEEEEE',
                       '#FFFFFF']


    ax.contour(I1, levels=levels_I1[::-1], colors=contour_palette,
               extent=[-dx1, dx1, -dx1, dx1],
               linewidths=1.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)

    cell_size = get_cell_size(imagename0)

    xticks = np.linspace(-dx1, dx1, 5)
    xticklabels = np.linspace(-dx1*cell_size, +dx1*cell_size, 5)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    ax.set_xlabel(r'Offset [arcsec]')
    ax = add_beam_to_image(imagename0, ax=ax, dx=dx1,
                           cell_size=cell_size)
    # ax.set_yticks([])
    # ax.set_yticklabels([])

    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    ax = fig.add_subplot(1, 4, 2)
    #     im = ax.imshow(g, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(g, cmap='magma_r',extent=[-dx,dx,-dx,dx],
                        origin='lower', alpha=1.0, norm=norm2)

    ax.set_title(title_labels[1])


    ax.contour(g, levels=levels_g[::-1], colors=contour_palette,
               extent=[-dx,dx,-dx,dx],
               linewidths=1.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)

    # cb = plt.colorbar(mappable=plt.gca().images[0],
    #                   cax=fig.add_axes([-0.0, 0.40, 0.02,0.19]))  # ,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    cb = plt.colorbar(mappable=plt.gca().images[0],
                      cax=fig.add_axes([0.07, 0.40, 0.02,0.19]),
                      orientation='vertical',shrink=1, aspect='auto',
                      pad=1, fraction=1.0,
                      drawedges=False, ticklocation='left')
    cb.formatter = CustomFormatter(factor=1000, useMathText=True)
    cb.update_ticks()
#     print('++++++++++++++++++++++')
#     print(plt.gca().images[0])
    cb.set_label(r'Flux Density [mJy/beam]', labelpad=1)
    cb.ax.xaxis.set_tick_params(pad=1)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(1)
    # cb.dividers.set_color('none')

    cell_size = get_cell_size(imagename)
    # ax = add_beam_to_image(imagename, ax=ax, dx=dx,
    #                        cell_size=cell_size)
    xticks = np.linspace(-dx, dx, 5)
    xticklabels = np.linspace(-dx*cell_size, +dx*cell_size, 5)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    # ax.set_yticks([])
    # ax.set_yticklabels([])

    ax = plt.subplot(1, 4, 3)

    #     im_plot = ax.imshow(m, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax_m, vmin=vmin_m)#norm=norm
    im_plot = ax.imshow(m, cmap='magma_r',extent=[-dx,dx,-dx,dx],
                        origin='lower', alpha=1.0, norm=norm2)
    ax.set_title(title_labels[2])
    ax.contour(m, levels=levels_g[::-1],
               colors=contour_palette,
               extent=[-dx,dx,-dx,dx],
               linewidths=1.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.08,0.3,0.02,0.4]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    cell_size = get_cell_size(modelname)
    xticks = np.linspace(-dx, dx, 5)
    xticklabels = np.linspace(-dx*cell_size, +dx*cell_size, 5)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    # ax = add_beam_to_image(modelname, ax=ax, dx=dx,
    #                        cell_size=cell_size)
    # ax.set_yticks([])
    # ax.set_yticklabels([])



    ax = plt.subplot(1, 4, 4)
    norm_re = simple_norm(r, min_cut=vmin, max_cut=vmax, stretch='sqrt')  # , max_percent=max_percent_highlevel)
    #     norm = simple_norm(r,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    #     ax.imshow(r,origin='lower',cmap='magma_r',alpha=1.0,vmax=vmax_r, vmin=vmin)#norm=norm
    ax.imshow(r, origin='lower',extent=[-dx,dx,-dx,dx],
              cmap='magma_r', alpha=1.0, norm=norm2)
    #     ax.imshow(r, cmap='magma_r',norm=norm,alpha=0.3,origin='lower')

    ax.contour(r, levels=levels_r[::-1],
               extent=[-dx,dx,-dx,dx],
               colors=contour_palette,
               linewidths=1.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    ax.contour(r, levels=levels_neg[::-1],extent=[-dx,dx,-dx,dx],
               colors='k', linewidths=1.0,
               alpha=1.0)

    ax = add_beam_to_image(imagename, ax=ax, dx=dx,
                           cell_size=cell_size)

    cell_size = get_cell_size(residualname)
    xticks = np.linspace(-dx, dx, 5)
    xticklabels = np.linspace(-dx*cell_size, +dx*cell_size, 5)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    # ax.set_yticks([])
    # ax.set_yticklabels([])


    ax.set_title(title_labels[3])
    #     cb1=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.40,0.02,0.19]))
    #     cb1.set_label(r'Flux [Jy/beam]',labelpad=1)
    #     cb1.ax.xaxis.set_tick_params(pad=1)
    #     cb1.ax.tick_params(labelsize=12)
    #     cb1.outline.set_linewidth(1)
    if NAME is not None:
        plt.savefig(NAME.replace('.fits', '') + SPECIAL_NAME + EXT, dpi=300,
                    bbox_inches='tight')
        plt.savefig(NAME.replace('.fits', '') + SPECIAL_NAME + '.jpg', dpi=300,
                    bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()

def fast_plot(imagename0, imagename, modelname, residualname, crop=False, box_size=512,
              max_percent_lowlevel=99.0, max_percent_highlevel=99.9999,
              NAME=None, EXT='.pdf', SPECIAL_NAME='', show_figure=True):

    """
    Fast plotting of image <> model <> residual images.
    TO BE REMOVED

    CHECK >> PLOT_INT_DEC
    """
    fig = plt.figure(figsize=(16, 16))
    try:
        g = pf.getdata(imagename)
        I1 = pf.getdata(imagename0)
        if len(np.shape(g) == 4):
            g = g[0][0]
            I1 = I1[0][0]
        m = pf.getdata(modelname)
        r = pf.getdata(residualname)
    except:
        I1 = load_fits_data(imagename0)
        g = load_fits_data(imagename)
        m = load_fits_data(modelname)
        r = load_fits_data(residualname)

    if crop == True:
        xin, xen, yin, yen = do_cutout(imagename, box_size=box_size, center=None, return_='box')
        #         I1 = I1[int(2*xin):int(xen/2),int(2*yin):int(yen/2)]
        I1 = I1[int(xin + box_size / 1.25):int(xen - box_size / 1.25),
             int(yin + box_size / 1.25):int(yen - box_size / 1.25)]
        # g = g[xin:xen,yin:yen]
        # m = m[xin:xen,yin:yen]
        # r = r[xin:xen,yin:yen]

    if mad_std(I1) == 0:
        std0 = I1.std()
    else:
        std0 = mad_std(I1)

    if mad_std(g) == 0:
        std = g.std()
    else:
        std = mad_std(g)

    if mad_std(r) == 0:
        std_r = r.std()
    else:
        std_r = mad_std(r)

    if mad_std(m) == 0:
        std_m = m.std()
    else:
        std_m = mad_std(m)

    #     print(I1)
    vmin0 = 3 * std  # 0.5*g.min()#
    vmax0 = 1.0 * g.max()
    vmin = 0.5 * std  # 0.5*g.min()#
    vmax = 1.0 * g.max()
    vmin_r = 0.5 * r.min()  # 1*std_r
    vmax_r = 1.0 * r.max()
    vmin_m = 1 * mad_std(m)  # vmin#0.01*std_m#0.5*m.min()#
    vmax_m = m.max()  # vmax#0.5*m.max()

    levels_I1 = np.geomspace(I1.max(), 1.5 * np.std(I1), 7)
    levels_g = np.geomspace(g.max(), 3 * std, 7)
    levels_m = np.geomspace(m.max(), 20 * std_m, 7)
    levels_r = np.geomspace(r.max(), 3 * std_r, 7)

    #     norm = simple_norm(g,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    norm = visualization.simple_norm(g, stretch='linear', max_percent=max_percent_lowlevel)
    norm0 = simple_norm(abs(I1), min_cut=0.5 * np.std(I1), max_cut=vmax,
                        stretch='sqrt')  # , max_percent=max_percent_highlevel)
    norm2 = simple_norm(abs(g), min_cut=vmin, max_cut=vmax, stretch='sqrt')  # , max_percent=max_percent_highlevel)
    CM = 'magma_r'
    ax = fig.add_subplot(1, 4, 1)

    #     im = ax.imshow(I1, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(I1, cmap='magma_r', origin='lower', alpha=1.0, norm=norm0)

    ax.set_title(r'$I_1^{\rm mask}$')

    ax.contour(I1, levels=levels_I1[::-1], colors='#009E73', linewidths=0.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    ax = fig.add_subplot(1, 4, 2)
    #     im = ax.imshow(g, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(g, cmap='magma_r', origin='lower', alpha=1.0, norm=norm2)

    ax.set_title(r'$I_2$')

    ax.contour(g, levels=levels_g[::-1], colors='#009E73', linewidths=0.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    cb = plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0, 0.40, 0.02,
                                                                      0.19]))  # ,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    cb.set_label(r'Flux [Jy/beam]', labelpad=1)
    cb.ax.xaxis.set_tick_params(pad=1)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(1)
    # cb.dividers.set_color('none')

    ax = plt.subplot(1, 4, 3)

    #     im_plot = ax.imshow(m, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax_m, vmin=vmin_m)#norm=norm
    im_plot = ax.imshow(m, cmap='magma_r', origin='lower', alpha=1.0, norm=norm2)
    ax.set_title(r'$I_{1}^{\rm mask} * \theta_2$')
    ax.contour(m, levels=levels_g[::-1], colors='#009E73', linewidths=0.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.08,0.3,0.02,0.4]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    ax = plt.subplot(1, 4, 4)
    norm_re = simple_norm(r, min_cut=vmin, max_cut=vmax, stretch='sqrt')  # , max_percent=max_percent_highlevel)
    #     norm = simple_norm(r,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    #     ax.imshow(r,origin='lower',cmap='magma_r',alpha=1.0,vmax=vmax_r, vmin=vmin)#norm=norm
    ax.imshow(r, origin='lower', cmap='magma_r', alpha=1.0, norm=norm2)
    #     ax.imshow(r, cmap='magma_r',norm=norm,alpha=0.3,origin='lower')

    ax.contour(r, levels=levels_r[::-1], colors='#009E73', linewidths=0.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    ax.set_yticks([])
    ax.set_title(r'$R_{12}$')
    #     cb1=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.40,0.02,0.19]))
    #     cb1.set_label(r'Flux [Jy/beam]',labelpad=1)
    #     cb1.ax.xaxis.set_tick_params(pad=1)
    #     cb1.ax.tick_params(labelsize=12)
    #     cb1.outline.set_linewidth(1)
    if NAME is not None:
        plt.savefig(NAME.replace('.fits', '') + SPECIAL_NAME + EXT, dpi=300,
                    bbox_inches='tight')
        plt.savefig(NAME.replace('.fits', '') + SPECIAL_NAME + '.jpg', dpi=300,
                    bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()


def plot_data_model_res(imagename, modelname, residualname, reference_image,
                        crop=False,box_size=512, NAME=None, CM='magma_r',
                        vmin_factor=3.0,vmax_factor=0.1,
                        max_percent_lowlevel=99.0, max_percent_highlevel=99.9999,
                        obs_type = 'radio',
                        ext='.pdf', show_figure=True):
    """
    Plots fitting results: image <> model <> residual images.

    Parameters
    ----------
    imagename : str
        Path to the image.
    modelname : str
        Path to the model.
    residualname : str
        Path to the residual.
    reference_image : str
        Path to the reference image.
    crop : bool, optional
        Crop the image to the box_size. The default is False.
    box_size : int, optional
        Size of the box to crop the image. The default is 512.
    NAME : str, optional
        Name of the output file. The default is None.
    CM : str, optional
        Colormap. The default is 'magma_r'.
    vmin_factor : float, optional
        Factor to multiply the standard deviation of the image to set the
        minimum value of the colormap. The default is 3.0.
    vmax_factor : float, optional
        Factor to multiply the maximum value of the image to set the
        maximum value of the colormap. The default is 0.1.
    max_percent_lowlevel : float, optional
        Maximum percentile to set the low level of the colormap. The default is 99.0.
    max_percent_highlevel : float, optional
        Maximum percentile to set the high level of the colormap. The default is 99.9999.
    ext : str, optional
        Extension of the output file. The default is '.pdf'.
    show_figure : bool, optional
        Show the figure. The default is True.
    """
    fig = plt.figure(figsize=(12, 12))
    try:
        try:
            g = pf.getdata(imagename)
            # if len(np.shape(g)==4):
            #     g = g[0][0]
            m = pf.getdata(modelname)
            r = pf.getdata(residualname)
        except:
            g = imagename
            m = modelname
            r = residualname
    except:
        g = load_fits_data(imagename)
        m = load_fits_data(modelname)
        r = load_fits_data(residualname)

    if crop == True:
        xin, xen, yin, yen = do_cutout(reference_image, box_size=box_size,
                                       center=None, return_='box')
        #         I1 = I1[int(2*xin):int(xen/2),int(2*yin):int(yen/2)]
        g = g[xin:xen, yin:yen]
        m = m[xin:xen, yin:yen]
        r = r[xin:xen, yin:yen]

    if mad_std(g) == 0:
        std = g.std()
    else:
        std = mad_std(g)

    if residualname is not None:
        if mad_std(r) == 0:
            std_r = r.std()
        else:
            std_r = mad_std(r)
    else:
        std_r = std

    if mad_std(m) == 0:
        std_m = m.std()
    else:
        std_m = mad_std(m)

    dx = g.shape[0]/2
    try:
        cell_size = get_cell_size(reference_image)
        axis_units_label = r'Offset [arcsec]'
    except:
        print('No cell or pixel size information in the image wcs/header. '
              'Setting cell/pixel size = 1.')
        cell_size = 1
        axis_units_label = r'Offset [px]'

    #     print(I1)
    vmin = vmin_factor * std  # 0.5*g.min()#
    vmax = vmax_factor * g.max()
    vmin_r = vmin  # 1.0*r.min()#1*std_r
    vmax_r = vmax #1.0 * r.max()
    vmin_m = vmin  # 1*mad_std(m)#vmin#0.01*std_m#0.5*m.min()#
    vmax_m = vmax  # 0.5*m.max()#vmax#0.5*m.max()

    levels_g = np.geomspace(3*g.max(), 3 * std, 6)
    levels_m = np.geomspace(3*m.max(), 10 * std_m, 6)
    levels_r = np.geomspace(3*r.max(), 3 * std_r, 6)

    #     norm = simple_norm(g,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    norm = visualization.simple_norm(g, stretch='linear',
                                     max_percent=max_percent_lowlevel)
    norm2 = simple_norm(abs(g), min_cut=vmin, max_cut=vmax,
                        stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)

    ax = fig.add_subplot(1, 3, 1)

    #     im = ax.imshow(g, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(g, cmap=CM,
                        origin='lower', extent=[-dx,dx,-dx,dx],
                        alpha=1.0, norm=norm2)

    ax = add_beam_to_image(imagename=reference_image, ax=ax,
                           dx=dx,cell_size=cell_size)

    ax.set_title(r'Data')

    contour_palette = ['#000000', '#444444', '#666666', '#EEEEEE',
                       '#EEEEEE', '#FFFFFF']

    ax.contour(g, levels=levels_g[::-1], colors=contour_palette,
               linewidths=1.2,extent=[-dx, dx, -dx, dx],
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    cb1 = plt.colorbar(mappable=plt.gca().images[0],
                       cax=fig.add_axes([0.91, 0.40, 0.02, 0.19]))
    if obs_type == 'radio':
        cb1.formatter = CustomFormatter(factor=int(1000/vmax_factor), 
                                        useMathText=True)
        cb1.update_ticks()
        cb1.set_label(r'Flux Density [mJy/beam]', labelpad=1)
        cb1.ax.xaxis.set_tick_params(pad=1)
        cb1.ax.tick_params(labelsize=12)
        cb1.outline.set_linewidth(1)
    else:
        cb1.formatter = CustomFormatter(factor=int(1/vmax_factor), 
                                        useMathText=True)
        cb1.update_ticks()
        cb1.set_label(r'Pixel Intensity]', labelpad=1)
        cb1.ax.xaxis.set_tick_params(pad=1)
        cb1.ax.tick_params(labelsize=12)
        cb1.outline.set_linewidth(1)
        
    """
    # No need for this additional colorbar.
    cb = plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes(
        [-0.0, 0.40, 0.02,
         0.19]))  # ,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    cb.formatter = CustomFormatter(factor=1000, useMathText=True)
    cb.update_ticks()
    cb.set_label(r'Flux [mJy/beam]', labelpad=1)
    cb.ax.xaxis.set_tick_params(pad=1)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(1)
    # cb.dividers.set_color('none')
    """
    xticks = np.linspace(-dx, dx, 4)
    xticklabels = np.linspace(-dx*cell_size, +dx*cell_size, 4)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    ax.set_xlabel(axis_units_label)
    # ax.set_yticks([])
    # ax.set_yticklabels([])


    ax = plt.subplot(1, 3, 2)

    #     im_plot = ax.imshow(m, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax_m, vmin=vmin_m)#norm=norm
    norm_mod = simple_norm(m, min_cut=vmin, max_cut=vmax,
                           stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)

    im_plot = ax.imshow(m, cmap=CM,
                        origin='lower',extent=[-dx,dx,-dx,dx],
                        alpha=1.0,
                        norm=norm_mod)
    ax.set_title(r'Model')
    ax.contour(m, levels=levels_g[::-1],
               extent=[-dx, dx, -dx, dx],
               colors=contour_palette, linewidths=1.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.08,0.3,0.02,0.4]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    ax.set_xlabel(axis_units_label)
    # ax.set_yticks([])
    ax.set_yticklabels([])

    ax = plt.subplot(1, 3, 3)
    norm_re = simple_norm(r, min_cut=vmin, max_cut=vmax,
                          stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)
    #     norm = simple_norm(r,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    #     ax.imshow(r,origin='lower',cmap='magma_r',alpha=1.0,vmax=vmax_r, vmin=vmin)#norm=norm
    ax.imshow(r, origin='lower',extent=[-dx, dx, -dx, dx],
              cmap=CM, alpha=1.0, norm=norm_re)
    #     ax.imshow(r, cmap='magma_r',norm=norm,alpha=0.3,origin='lower')

    ax.contour(r, levels=levels_g[::-1],
               extent=[-dx, dx, -dx, dx],
               colors=contour_palette, linewidths=1.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    levels_neg = np.asarray([-6 * std])

    ax.contour(r, levels=levels_neg[::-1],
               extent=[-dx, dx, -dx, dx],
               colors='k', linewidths=1.0,
               alpha=1.0)
    ax.set_yticks(xticks, xticklabels)
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel(axis_units_label)
    # ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title(r'Residual')
    # cb1.dividers.set_color('none')
    if NAME != None:
        plt.savefig(NAME + ext, dpi=300, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()


def plot_image_model_res(imagename, modelname, residualname, reference_image, crop=False,
               box_size=512, NAME=None, CM='magma_r',
               vmin_factor=3.0,vmax_factor=0.1,
               max_percent_lowlevel=99.0, max_percent_highlevel=99.9999,
               ext='.pdf', show_figure=True):
    """
    Fast plotting of image <> model <> residual images.

    """
    fig = plt.figure(figsize=(12, 12))
    try:
        try:
            g = pf.getdata(imagename)
            # if len(np.shape(g)==4):
            #     g = g[0][0]
            m = pf.getdata(modelname)
            r = pf.getdata(residualname)
        except:
            g = imagename
            m = modelname
            r = residualname
    except:
        g = load_fits_data(imagename)
        m = load_fits_data(modelname)
        r = load_fits_data(residualname)

    if crop == True:
        xin, xen, yin, yen = do_cutout(reference_image, box_size=box_size,
                                       center=None, return_='box')
        #         I1 = I1[int(2*xin):int(xen/2),int(2*yin):int(yen/2)]
        g = g[xin:xen, yin:yen]
        m = m[xin:xen, yin:yen]
        r = r[xin:xen, yin:yen]

    if mad_std(g) == 0:
        std = g.std()
    else:
        std = mad_std(g)

    if mad_std(r) == 0:
        std_r = r.std()
    else:
        std_r = mad_std(r)

    if mad_std(m) == 0:
        std_m = m.std()
    else:
        std_m = mad_std(m)

    #     print(I1)
    vmin = vmin_factor * std  # 0.5*g.min()#
    vmax = vmax_factor * g.max()
    vmin_r = vmin  # 1.0*r.min()#1*std_r
    vmax_r = vmax #1.0 * r.max()
    vmin_m = vmin  # 1*mad_std(m)#vmin#0.01*std_m#0.5*m.min()#
    vmax_m = vmax  # 0.5*m.max()#vmax#0.5*m.max()

    levels_g = np.geomspace(g.max(), 3 * std, 7)
    levels_m = np.geomspace(m.max(), 10 * std_m, 7)
    levels_r = np.geomspace(r.max(), 3 * std_r, 7)

    #     norm = simple_norm(g,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    norm = visualization.simple_norm(g, stretch='linear',
                                     max_percent=max_percent_lowlevel)
    norm2 = simple_norm(abs(g), min_cut=vmin, max_cut=vmax,
                        stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)

    ax = fig.add_subplot(2, 3, 1)

    #     im = ax.imshow(g, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(g, cmap=CM, origin='lower', alpha=1.0, norm=norm2)

    ax.set_title(r'Image')

    ax.contour(g, levels=levels_g[::-1], colors='#009E73', linewidths=0.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    cb = plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes(
        [-0.0, 0.40, 0.02,
         0.19]))  # ,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    cb.set_label(r'Flux [Jy/beam]', labelpad=1)
    cb.ax.xaxis.set_tick_params(pad=1)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(1)
    # cb.dividers.set_color('none')

    ax = plt.subplot(2, 3, 2)

    #     im_plot = ax.imshow(m, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax_m, vmin=vmin_m)#norm=norm
    norm_mod = simple_norm(m, min_cut=vmin, max_cut=vmax,
                           stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)

    im_plot = ax.imshow(m, cmap=CM, origin='lower', alpha=1.0,
                        norm=norm_mod)
    ax.set_title(r'Model')
    ax.contour(m, levels=levels_g[::-1], colors='#009E73', linewidths=0.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.08,0.3,0.02,0.4]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    ax = plt.subplot(2, 3, 3)
    norm_re = simple_norm(r, min_cut=vmin, max_cut=vmax,
                          stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)
    #     norm = simple_norm(r,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    #     ax.imshow(r,origin='lower',cmap='magma_r',alpha=1.0,vmax=vmax_r, vmin=vmin)#norm=norm
    ax.imshow(r, origin='lower', cmap=CM, alpha=1.0, norm=norm_re)
    #     ax.imshow(r, cmap='magma_r',norm=norm,alpha=0.3,origin='lower')

    ax.contour(r, levels=levels_r[::-1], colors='#009E73', linewidths=0.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    ax.set_yticks([])
    ax.set_title(r'Residual')
    cb1 = plt.colorbar(mappable=plt.gca().images[0],
                       cax=fig.add_axes([0.91, 0.40, 0.02, 0.19]))
    cb1.set_label(r'Flux [Jy/beam]', labelpad=1)
    cb1.ax.xaxis.set_tick_params(pad=1)
    cb1.ax.tick_params(labelsize=12)
    cb1.outline.set_linewidth(1)
    return(ax,plt,fig)
    # cb1.dividers.set_color('none')
    # if NAME != None:
    #     plt.savefig(NAME + ext, dpi=300, bbox_inches='tight')
    #     if show_figure == True:
    #         plt.show()
    #     else:
    #         plt.close()


def eimshow(imagename, crop=False, box_size=128, center=None, with_wcs=True,
            vmax=None,fig=None,
            dx_shift = 0,
            dy_shift = 0,
            vmax_factor=0.5, neg_levels=np.asarray([-3]), CM='magma_r',
            cmap_cont='magma_r',
            rms=None, plot_title=None, apply_mask=False,
            add_contours=True, extent=None, projection='offset', add_beam=False,
            vmin_factor=3, plot_colorbar=False, cbar_orientation=None,pad=-0.2,
            figsize=(5, 5), aspect=None,n_pts_labels = 5,cbar_n_points=6,num_contours=6,
            show_axis='on',flux_units='mJy',add_frequency=False,freq_label=None,
            source_distance=None, scalebar_length=250 * u.pc,
            ax=None, return_extent=False,no_show=False,
            save_name=None, special_name='',
            text_annotation=None, text_position=(0.3, 0.8),
            verbose = 0):
    """
    Customised imshow function for plotting images. It includes the option to
    automatically add contour levels, custom projections, colorbars, scalebars, etc.

    Parameters
    ----------
    imagename : str, np.ndarray
        Path to the image.
    crop : bool, optional
        Crop the image to the box_size. The default is False.
    box_size : int, optional
        Size of the box to crop the image. The default is 128.
    center : tuple, optional
        Center of the image. The default is None.
    with_wcs : bool, optional
        Use the wcs information of the image. The default is True.
    vmax : float, optional
        Maximum value of the colormap. The default is None.
    vmax_factor : float, optional
        Factor to multiply the maximum value of the image to set the
        maximum value of the colormap. The default is 0.5.
    neg_levels : np.ndarray, optional
        Negative levels for the contours. The default is np.asarray([-3]).
    CM : str, optional
        Colormap. The default is 'magma_r'.
    cmap_cont : str, optional
        Colormap for the contours. The default is 'terrain'.
    rms : float, optional
        RMS of the image. The default is None.
    plot_title : str, optional
        Title of the plot. The default is None.
    apply_mask : bool, optional
        Apply a mask to the image. The default is False.
    add_contours : bool, optional
        Add contours to the image. The default is True.
    extent : list, optional
        Extent of the image. The default is None.
    projection : str, optional
        Projection of the image. The default is 'offset'.
    add_beam : bool, optional
        Add the beam to the image. The default is False.
    vmin_factor : int, optional
        Factor to multiply the standard deviation of the image to set the
        minimum value of the colormap. The default is 3.
    plot_colorbar : bool, optional
        Plot the colorbar. The default is True.
    figsize : tuple, optional
        Size of the figure. The default is (5, 5).
    aspect : str, optional
        Aspect of the image. The default is None.
    show_axis : str, optional
        Show the axis. The default is 'on'.
    flux_units : str, optional
        Units of the flux. The default is 'Jy'.
    source_distance : float, optional
        Distance to the source. The default is None.
    scalebar_length : float, optional
        Length of the scalebar. The default is 250 * u.pc.
    ax : matplotlib.pyplot.axis, optional
        Axis of the plot. The default is None.
    save_name : str, optional
        Name of the output file. The default is None.
    special_name : str, optional
        Special name for the output file. The default is ''.
    """
    try:
        import cmasher as cmr
        # print('Imported cmasher for density maps.'
        #       'If you would like to use, examples:'
        #       'CM = cmr.ember,'
        #       'CM = cmr.flamingo,'
        #       'CM = cmr.gothic'
        #       'CM = cmr.lavender')
        """
        ... lilac,rainforest,sepia,sunburst,torch.
        Diverging: copper,emergency,fusion,infinity,pride'
        """
    except:
        print('Error importing cmasher. If you want '
              'to use its colormaps, install it. '
              'Then you can use for example:'
              'CM = cmr.flamingo')
    if ax is None:
        fig = plt.figure(figsize=figsize)
        # if isinstance(box_size, int):
        #     fig = plt.figure(figsize=figsize)
        # else:
        #     scale_fig_x = box_size[0]/box_size[1]
        #     fig = plt.figure(figsize=(figsize[0]*scale_fig_x,figsize[1]))
    else:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        else:
            pass
    if isinstance(imagename, str) == True:

        if with_wcs == True:
            hdu = pf.open(imagename)
            ww = WCS(hdu[0].header, naxis=2)
            try:
                if len(np.shape(hdu[0].data) == 2):
                    g = hdu[0].data[0][0]
                else:
                    g = hdu[0].data
            except:
                g = load_fits_data(imagename)
        if with_wcs == False:
            g = load_fits_data(imagename)

        if crop == True:
            yin, yen, xin, xen = do_cutout(imagename, box_size=box_size,
                                           center=center, return_='box')
            g = g[xin:xen, yin:yen]
            # crop = False

        if apply_mask == True:
            _, mask_d = mask_dilation(imagename, cell_size=None,
                                      sigma=6, rms=None,
                                      dilation_size=None,
                                      iterations=2, dilation_type='disk',
                                      PLOT=False, show_figure=False)
            print('Masking emission....')
            g = g * mask_d[xin:xen, yin:yen]
        
        g = np.nan_to_num(g,nan=0)
    
    else:
        g = np.nan_to_num(imagename,nan=0)
        mask_d = 1
        # print('3', g)

        if crop == True:
            xin, xen, yin, yen = do_cutout(imagename, box_size=box_size,
                                           center=center, return_='box')
            g = g[xin:xen, yin:yen]
            if apply_mask == True:
                print('Masking emission....')
                g = g * mask_d[xin:xen, yin:yen]

    if rms is not None:
        std = rms
    else:
        if mad_std(g) == 0:
            """
            About std:
                mad_std is much more robust than np.std.
                But:
                    if mad_std is applied to a masked image, with zero
                    values outside the emission region, mad_std(image) is zero!
                    So, in that case, np.std is a good option.
            """
            # print('5', g)
            std = np.nanstd(g)
        else:
            std = mad_std(g)

    if isinstance(imagename, str) == True:
        try:
            cell_size = get_cell_size(imagename)
            axis_units_label = r'Offset [arcsec]'
        except:
            print(
                'No cell or pixel size information in the image wcs/header. '
                'Setting cell/pixel size = 1.')
            cell_size = 1
            axis_units_label = r'Offset [px]'
    else:
        cell_size = 1
        axis_units_label = r'Offset [px]'
    
    dx = g.shape[1] / 2
    dy = g.shape[0] / 2
    if ax is None:

        if (projection == 'celestial') and (with_wcs == True) and (isinstance(
                imagename, str) == True):
            ax = fig.add_subplot(projection=ww.celestial)
            ax.set_xlabel('RA', fontsize=14)
            ax.set_ylabel('DEC', fontsize=14)
            ax.grid()

        if isinstance(imagename, str) == False:
            projection = 'px'

        if projection == 'offset':
            ax = fig.add_subplot()
            # dx = g.shape[0] / 2
            axis_units_label = r'Offset [arcsec]'
            ax.set_xlabel(axis_units_label, fontsize=14)

        dx = g.shape[1] / 2
        dy = g.shape[0] / 2
        if projection == 'px':
            ax = fig.add_subplot()
            cell_size = 1
            # dx = g.shape[0] / 2
            ax.set_xlabel('x pix')
            ax.set_ylabel('y pix')
            axis_units_label = r'Offset [px]'
            ax.set_xlabel(axis_units_label, fontsize=14)
            ax.set_ylabel(axis_units_label, fontsize=14)

    else:
        projection = 'offset'
        if projection == 'offset':
            # ax = fig.add_subplot()
            # dx = g.shape[0] / 2
            axis_units_label = r'Offset [arcsec]'
            ax.set_xlabel(axis_units_label, fontsize=14)

    
    xticks = np.linspace(-dx+dx_shift/2, dx+dx_shift/2, n_pts_labels)
    yticks = np.linspace(-dy+dy_shift/2, dy+dy_shift/2, n_pts_labels)
    xticklabels = np.linspace(-(dx-dx_shift/2) * cell_size, +(dx+dx_shift/2) * cell_size, n_pts_labels)
    yticklabels = np.linspace(-(dy-dy_shift/2) * cell_size, +(dy+dy_shift/2) * cell_size, n_pts_labels)
    # if dx < 10:
    #     xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    # else:
    #     xticklabels = ['{:.0f}'.format(xtick) for xtick in xticklabels]
    # if dy < 10:
    #     yticklabels = ['{:.2f}'.format(ytick) for ytick in yticklabels]
    # else:
    #     yticklabels = ['{:.0f}'.format(ytick) for ytick in yticklabels]

    if (projection =='offset') or (projection == 'celestial'):
        xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
        yticklabels = ['{:.2f}'.format(ytick) for ytick in yticklabels]
    else:
        xticklabels = ['{:.0f}'.format(xtick) for xtick in xticklabels]
        yticklabels = ['{:.0f}'.format(ytick) for ytick in yticklabels]

    ax.set_yticks(yticks, yticklabels)
    ax.set_xticks(xticks, xticklabels)
    ax.set_aspect('equal')

    ax.tick_params(axis='y', which='both', labelsize=16, color='black',
                   pad=5)
    ax.tick_params(axis='x', which='both', labelsize=16, color='black',
                   pad=5)
    # ax2_x.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax3_y.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if projection != 'celestial':
        ax.grid(which='both', axis='both', color='gray', linewidth=0.6,
                alpha=0.7)
        ax.grid(which='both', axis='both', color='gray', linewidth=0.6,
                alpha=0.7)
    else:
        ax.grid()

    ax.axis(show_axis)
    # ax.axis(show_axis)

    vmin = vmin_factor * std
    if extent is None:
        extent = [-dx+dx_shift, dx+dx_shift, -dy+dy_shift, dy+dy_shift]
    #     print(g)

    if vmax is not None:
        vmax = vmax
    else:
        if vmax_factor is not None:
            vmax = vmax_factor * g.max()
        else:
            vmax = 0.95 * g.max()


    norm0 = simple_norm(g, stretch='linear', max_percent=99.0)
    norm = simple_norm(g, stretch='sqrt', asinh_a=0.02, min_cut=vmin,
                       max_cut=vmax)
    
    if no_show:
        # # plot the first normalization (low level, transparent)
        im_plot = ax.imshow(g, origin='lower',aspect=aspect,
                            cmap='gray', norm=norm0, alpha=0.0, extent=extent)

        # cm = copy.copy(plt.cm.get_cmap(CM))
        # cm.set_under((0, 0, 0, 0))
        im_plot = ax.imshow((g), cmap=CM, origin='lower', alpha=0.0, extent=extent,
                            norm=norm,
                            aspect=aspect)  # ,vmax=vmax, vmin=vmin)#norm=norm

    else:
        # # plot the first normalization (low level, transparent)
        im_plot = ax.imshow(g, origin='lower',aspect=aspect,
                            cmap='gray', norm=norm0, alpha=0.5, extent=extent)

        # cm = copy.copy(plt.cm.get_cmap(CM))
        # cm.set_under((0, 0, 0, 0))
        im_plot = ax.imshow((g), cmap=CM, origin='lower', alpha=1.0, extent=extent,
                            norm=norm,
                            aspect=aspect)  # ,vmax=vmax, vmin=vmin)#norm=norm
        
    


    if add_contours:
        try:
            levels_g = np.geomspace(2.0 * g.max(), 5 * std, num_contours)
            levels_low = np.asarray([4 * std, 3 * std])
            levels_black = np.geomspace(vmin_factor * std + 0.00001, 2.5 * g.max(), num_contours)
            levels_neg = neg_levels * std
            levels_white = np.geomspace(g.max(), 0.1 * g.max(), num_contours)

            contour_palette_ = ['#000000', '#444444', '#666666', '#EEEEEE',
                               '#EEEEEE', '#FFFFFF']
            if '_r' in CM:
                contour_palette = contour_palette_
            else:
                contour_palette = contour_palette_[::-1]

            if no_show:
                contour = ax.contour(g, levels=levels_g[::-1],
                                    #  colors=contour_palette[::-1], #for dark cmap
                                    colors=contour_palette, #for light cmap
                                    # aspect=aspect,
                                    linewidths=1.2, extent=extent,
                                    alpha=0.0)

                contour = ax.contour(g, levels=levels_low[::-1],
                                    colors='brown',
                                    # aspect=aspect,
                                    # linestyles=['dashed', 'dashdot'],
                                    linewidths=1.0, extent=extent,
                                    alpha=0.0)

            else:
                contour = ax.contour(g, levels=levels_g[::-1],
                                    #  colors=contour_palette[::-1], #for dark cmap
                                    colors=contour_palette, #for light cmap
                                    # aspect=aspect,
                                    linewidths=1.2, extent=extent,
                                    alpha=1.0)

                contour = ax.contour(g, levels=levels_low[::-1],
                                    colors='brown',
                                    # aspect=aspect,
                                    # linestyles=['dashed', 'dashdot'],
                                    linewidths=1.0, extent=extent,
                                    alpha=1.0)
            # ax.clabel(contour, inline=1, fontsize=10)
        except:
            pass
        try:
            if no_show:
                ax.contour(g, levels=levels_neg[::-1], colors='k',
                        linewidths=1.0, extent=extent,
                        alpha=0.0)

            else:
                ax.contour(g, levels=levels_neg[::-1], colors='k',
                        linewidths=1.0, extent=extent,
                        alpha=1.0)
        except:
            pass
        
    if add_frequency:
        try:
            if freq_label is None:
                if isinstance(imagename, str) == True:
                    frequency = f"{(getfreqs([imagename])[0]/1e9):.1f} GHz"
            else:
                frequency = freq_label
            ax.annotate(frequency,
                        xy=(0.70, 0.82), xycoords='axes fraction',
                        fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        color='red')
        except:
            pass
        
    
    if plot_colorbar:
        
        divider = make_axes_locatable(ax)
        
        try:
            # cb = plt.colorbar(mappable=plt.gca().images[0],
            #                   cax=fig.add_axes([0.90, 0.15, 0.05, 0.70]))

            # cb = plt.colorbar(im_plot, 
            #                   ax=ax,
            #                   cax=fig.add_axes([0.90, 0.15, 0.05, 0.70]))
            # cbar_ax = fig.add_axes([0.15, 0.85, 0.7, 0.05])  # [left, bottom, width, height]
            if cbar_orientation == 'horizontal':
                cax = divider.append_axes("top", size="7%", pad=0.05)
                cb = fig.colorbar(im_plot, 
                                #   pad=pad,
                                  cax=cax,fraction=0.046,
                                  orientation='horizontal')
            else:
                # cb = plt.colorbar(im_plot, 
                #                 ax=ax,
                #                 cax=fig.add_axes([0.90, 0.15, 0.05, 0.70])
                #                 )
                cax = divider.append_axes("right", size="7%", pad=0.05)
                cb = plt.colorbar(im_plot, 
                                ax=ax,fraction=0.046,
                                cax=cax
                                )
            # cb = plt.colorbar(im_plot, 
            #                   ax=ax,
            #                   orientation='horizontal', fraction=0.046, pad=0.04)

            if flux_units == 'Jy':
                cb.set_label(r"Flux Density [Jy/Beam]", labelpad=10, fontsize=16)
                cb.formatter = CustomFormatter(factor=1, useMathText=True)
                cb.update_ticks()
            if flux_units == 'mJy':
                cb.set_label(r"Flux Density [mJy/Beam]", labelpad=10, fontsize=16)
                cb.formatter = CustomFormatter(factor=1000, useMathText=True)
                cb.update_ticks()
            if flux_units == 'any':
                cb.set_label(r"Pixel Intensity", labelpad=10, fontsize=16)
                cb.formatter = CustomFormatter(factor=1, useMathText=True)
                cb.update_ticks()
            if flux_units == 'sfr':
                cb.set_label(r"${\rm M}_{\rm \odot} \ {\rm yr^{-1}} \ {\rm beam^{-1}}$", labelpad=10, fontsize=16)
                cb.formatter = CustomFormatter(factor=1, useMathText=True)
                cb.update_ticks()

            levels_colorbar2 = np.geomspace(1.0 * vmax, 3 * std,
                                            cbar_n_points)
            cb.set_ticks(levels_colorbar2)

            cb.ax.yaxis.set_tick_params(labelleft=True, labelright=False,
                                        tick1On=False, tick2On=False)
            cb.ax.yaxis.tick_right()
            # cb.set_ticks(levels_colorbar2)
            # cb.set_label(r'Flux [mJy/beam]', labelpad=10, fontsize=16)
            #         cbar.ax.xaxis.set_tick_params(pad=0.1,labelsize=10)
            cb.ax.tick_params(labelsize=16)
            cb.outline.set_linewidth(1)
            # cbar.dividers.set_color(None)

            # Make sure the color bar has ticks and labels at the top, since the bar is on the top as well.
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')
        except:
            pass

    if plot_title is not None:
        ax.set_title(plot_title)

    if add_beam == True:

        if isinstance(imagename, str) == True:
            try:
                from matplotlib.patches import Ellipse
                imhd = imhead(imagename)
                a = imhd['restoringbeam']['major']['value']
                b = imhd['restoringbeam']['minor']['value']
                pa = imhd['restoringbeam']['positionangle']['value']
                if projection == 'px':
                    el = Ellipse((-(dx-dx_shift) * 0.85, 
                                  -(dy-dy_shift) * 0.85), 
                                 b, a, angle=pa,
                                 facecolor='black', alpha=1.0)
                else:
                    el = Ellipse((-(dx-dx_shift) * 0.85, 
                                  -(dy-dy_shift) * 0.85), 
                                 b / cell_size,
                                 a / cell_size,
                                 angle=pa, facecolor='black', alpha=1.0)

                ax.add_artist(el, )

                Oa = '{:.2f}'.format(a)
                Ob = '{:.2f}'.format(b)

                blabel_pos_y, blabel_pos_x = g.shape
                blabel_pos_x = (blabel_pos_x + dx+dx_shift)# * (dx/dy)
                blabel_pos_y = (blabel_pos_y + dy+dy_shift) 

                #         ax.annotate(r'$' + Oa +'\\times'+Ob+'$',
                #                     xy=(blabel_pos_x* 0.77, blabel_pos_y * 0.58), xycoords='data',
                #                     fontsize=15,bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                #                     color='red')
                ax.annotate(r"$" + Oa + "''\\times" + Ob + "''$",
                            xy=(0.63, 0.08), xycoords='axes fraction',
                            fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='white',
                                      alpha=0.9),
                            color='red')

                el.set_clip_box(ax.bbox)
            except:
                print('Error adding beam.')

    if source_distance is not None:
        # try:
        ww.wcs.radesys = 'icrs'
        radesys = ww.wcs.radesys
        # distance = source_distance * u.Mpc
        distance = angular_distance_cosmo(source_distance)  # * u.Mpc
        #         scalebar_length = scalebar_length
        scalebar_loc = (0.99, 0.99)  # y, x
        left_side = coordinates.SkyCoord(
            *ww.celestial.wcs_pix2world(
                g.shape[0],
                g.shape[1],
                0) * u.deg,
            frame=radesys.lower())

        length = (scalebar_length / distance).to(u.arcsec,
                                                 u.dimensionless_angles())

        scale_bar_length_pixels = length.value / cell_size
        scale_bar_position = (-(dx-1*dx_shift) * 0.50, -(dy-1*dy_shift) * 0.9)

        ax.annotate('',
                    xy=(scale_bar_position[0] + scale_bar_length_pixels,
                        scale_bar_position[1]),
                    # xy=(0.1, 0.1), ##
                    xytext=scale_bar_position, arrowprops=dict(arrowstyle='-',
                                                               color='black', #navy
                                                               lw=5))

        ax.text(scale_bar_position[0] + scale_bar_length_pixels / 2,
                scale_bar_position[1] + scale_bar_length_pixels / 15,
                f'{scalebar_length}', fontsize=20,
                color='black', ha='center',weight='bold',
                va='bottom')
        # except:
        #     print('Error adding scalebar.')
    if text_annotation:
        ax.annotate(text_annotation, xy=text_position, xycoords='figure fraction',
                    fontsize=18, color='black', fontweight='bold')

    if save_name != None:
    #         if not os.path.exists(save_name+special_name+'.jpg'):
        plt.savefig(save_name + special_name + '.jpg', dpi=600,
                    bbox_inches='tight')
        # plt.savefig(save_name + special_name + '.pdf', dpi=600,
        #             bbox_inches='tight')
    # if no_show:
    #     plt.clf()
    #     plt.close()
    if return_extent:
        return ax,extent
    else:
        return ax

def plot_alpha_map(alphaimage,alphaimage_error,radio_map,frequencies,
                   vmin_factor = 3,neg_levels=np.asarray([-3]),
                   vmin=None,vmax=None,rms=None,figsize=(6, 6),
                   extent = None,crop=False,centre=None,
                   plot_title='',label_colorbar='',
                   cmap='magma_r',n_contours = 8,
                   projection='offset',cell_size = None,
                   box_size=None,plot_alpha_error_points=False,
                   save_name=None):

    from matplotlib.colors import ListedColormap
    
    fig = plt.figure(figsize=figsize)
    

    # fig, ax = plt.subplots(figsize=figsize)

    if isinstance(radio_map, str) == True:
        g = load_fits_data(radio_map)
    else:
        g = radio_map

    if crop:
        if centre is None:
            centre = (int(alphaimage.shape[0]/2),int(alphaimage.shape[1]/2))
        else:
            centre = centre
            
        _alphaimage = do_cutout_2D(alphaimage, box_size=box_size, 
                                          center=centre, return_='data')

        _alphaimage_error = do_cutout_2D(alphaimage_error, box_size=box_size, 
                                          center=(centre[0],centre[1]), return_='data')
        _g = do_cutout_2D(g, box_size=box_size, 
                                          center=centre, return_='data')

    else:
        _alphaimage = alphaimage.copy()
        _alphaimage_error = alphaimage_error.copy()
        _g = g


    

    # Access the 'magma' colormap
    base_cmap = plt.colormaps[cmap]

    # Create a new colormap with 3 colors from the base colormap
    new_colors = base_cmap(np.linspace(0, 1, 10))
    custom_cmap = ListedColormap(new_colors)
    # custom_cmap = 'magma_r'


    if vmin is None:
        vmin = np.nanmin(_alphaimage) - 1*abs(np.nanmedian(_alphaimage))
    else:
        vmin = vmin
    if vmax is None:
        vmax = np.nanmax(_alphaimage) + 1*abs(np.nanmedian(_alphaimage))
    else:
        vmax = vmax
    
    if projection == 'offset':
        ax,extent = eimshow(radio_map,return_extent = True,figsize=figsize,
                        add_contours=False,CM='magma',no_show=True,
                            crop=crop,center=centre,box_size=box_size)
        if cell_size is None:
            if isinstance(radio_map, str):
                cell_size = get_cell_size(radio_map)
            else:
                raise ValueError(f"Please, if selecting offset projection mode," 
                                 f"provide the cell_size of the image.")
    else:
        ax = fig.add_subplot(111)  # 111 means 1x1 grid, first subplot
        cell_size = 1
    
    img = plt.imshow(_alphaimage, origin='lower', 
                     vmin=vmin, 
                     extent=extent,
    #                  vmax=0.0,
                     vmax=vmax, 
                     cmap=custom_cmap,
    #                  cmap='plasma'
                    )
    if rms is None:
        std = mad_std(_g)
    else:
        std = rms 
    
    # print(extent)
    
    
    levels_g = np.geomspace(3.0 * _g.max(), 5 * std, n_contours)
    levels_low = np.asarray([4 * std, 3 * std])
    levels_black = np.geomspace(vmin_factor * std + 0.00001, 2.5 * _g.max(), 6)
    levels_neg = neg_levels * std
    levels_white = np.geomspace(_g.max(), 0.1 * _g.max(), 6)

    contour_palette = ['#000000', '#444444', '#666666', '#EEEEEE',
                       '#EEEEEE', '#FFFFFF']


    contour = ax.contour(_g, levels=levels_g[::-1],
                         colors=contour_palette,
                          origin='lower', 
                        #   cmap='Greys',
                         # aspect=aspect,
                         linewidths=1.2, extent=extent,
                         alpha=1.0)

    if plot_alpha_error_points:
        # Downsample the data for scatter plot
        downscale_factor = 10  # Adjust this factor as needed
        x_downsampled, y_downsampled = np.meshgrid(
            np.arange(0, _alphaimage.shape[1], downscale_factor),
            np.arange(0, _alphaimage.shape[0], downscale_factor)
        )
        
        marker_sizes = 50 * abs(_alphaimage_error[y_downsampled, x_downsampled]/_alphaimage[y_downsampled, x_downsampled])
        min_error_pos = np.where(marker_sizes==np.nanmin(marker_sizes))
        max_error_pos = np.where(marker_sizes==np.nanmax(marker_sizes))
        # print(x_downsampled*cell_size,y_downsampled*cell_size)
        sc = ax.scatter(x_downsampled*cell_size, y_downsampled*cell_size, s=marker_sizes, 
                     c='cyan', alpha=0.9)
        _ = ax.scatter(x_downsampled[min_error_pos]*cell_size, 
                        y_downsampled[min_error_pos]*cell_size, 
                        s=marker_sizes[min_error_pos], 
                         c='black', alpha=0.8,
                        label=fr'Min err $\alpha$ = {(marker_sizes[min_error_pos][0]/50):.2f}')
        _ = ax.scatter(x_downsampled[max_error_pos]*cell_size, 
                        y_downsampled[max_error_pos]*cell_size, 
                        s=marker_sizes[max_error_pos], 
                        c='black', alpha=0.8,
        #                 marker='x',
                        label=fr'Max err $\alpha$ = {(marker_sizes[max_error_pos][0]/50):.2f}')
        plt.legend()
    if label_colorbar == '':
        label_colorbar = fr"Spectral Index $\alpha$ [{(frequencies[0]/1e9):.1f} ~ {(frequencies[-1]/1e9):.1f} GHz]"
    else:
        _label_colorbar = fr"[{(frequencies[0]/1e9):.1f} ~ {(frequencies[-1]/1e9):.1f} GHz]"
        label_colorbar = label_colorbar + _label_colorbar

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%",
                              pad=0.05)  # Adjust 'pad' for space and 'size' for width
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(label=label_colorbar)
    # cbar.formatter = CustomFormatter(factor=1000, useMathText=True)
    # cbar.update_ticks()
    
    
    # cbar = plt.colorbar(img, 
    #                     label=label_colorbar)
    ax.grid(which='both', axis='both', color='gray', linewidth=0.6,
                alpha=0.7)
    
    ax.set_title(f"{plot_title} ({len(frequencies)} images)")
    # plt.savefig('alpha_map_and_errors_MCG12_S_C_X_new_fit_in_linear.pdf',dpi=300,bbox_inches='tight')
    # plt.show()
    if save_name != None:
        plt.savefig(save_name, dpi=300,
                    bbox_inches='tight')
        
    pass
    

def add_beam_to_image(imagename, ax, dx, cell_size):
    if isinstance(imagename, str) == True:
        try:
            from matplotlib.patches import Ellipse

            imhd = imhead(imagename)
            a = imhd['restoringbeam']['major']['value']
            b = imhd['restoringbeam']['minor']['value']
            pa = imhd['restoringbeam']['positionangle']['value']
            el = Ellipse((-dx * 0.85, -dx * 0.85), b / cell_size, a / cell_size,
                         angle=pa, facecolor='black', alpha=1.0)

            ax.add_artist(el, )

            Oa = '{:.2f}'.format(a)
            Ob = '{:.2f}'.format(b)

            blabel_pos_x, blabel_pos_y = 2*dx, 2*dx
            blabel_pos_x = blabel_pos_x + dx
            blabel_pos_y = blabel_pos_y + dx

            #         ax.annotate(r'$' + Oa +'\\times'+Ob+'$',
            #                     xy=(blabel_pos_x* 0.77, blabel_pos_y * 0.58), xycoords='data',
            #                     fontsize=15,bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            #                     color='red')
            ax.annotate(r'$' + Oa + '\\times' + Ob + '$',
                        xy=(0.60, 0.06), xycoords='axes fraction',
                        fontsize=15, bbox=dict(boxstyle='round', facecolor='white',
                                               alpha=0.9),
                        color='red')

            el.set_clip_box(ax.bbox)
        except:
            print('Error adding beam.')
    return ax


def plot_image(image, residual_name=None, box_size=200, box_size_inset=60,
               center=None, rms=None, add_inset='auto', add_beam=True,
               vmin_factor=3, max_percent_lowlevel=99.0,
               levels_neg_factors=np.asarray([-3]),
               max_percent_highlevel=99.9999, vmax=None,
               do_cut=False, CM='magma_r', cbar_axes=[0.03, 0.11, 0.05, 0.77],
               # cbar_axes=[0.9, 0.15, 0.04, 0.7],
               source_distance=None, save_name=None, special_name='',
               scalebar_length=1.0 * u.kpc,
               show_axis='on', plot_color_bar=True, figsize=(5, 5),
               cmap_cont='terrain',
               projection='offset'):
    import cmasher as cmr
    import astropy.io.fits as fits
    from astropy import coordinates
    import matplotlib.ticker as mticker
    from matplotlib.ticker import FuncFormatter, FormatStrFormatter

    hdu = fits.open(image)
    ww = WCS(hdu[0].header)
    #     ww.wcs.ctype = [ 'XOFFSET' , 'YOFFSET' ]
    imhd = imhead(image)
    if do_cut == True:
        if center is None:
            st = imstat(image)
            # print('  >> Center --> ', st['maxpos'])
            yin, yen, xin, xen = st['maxpos'][0] - box_size, st['maxpos'][
                0] + box_size, st['maxpos'][1] - box_size, \
                                 st['maxpos'][1] + box_size
        else:
            yin, yen, xin, xen = center[0] - box_size, center[0] + box_size, \
                                 center[1] - box_size, center[1] + box_size
    else:
        yin, yen, xin, xen = 0, -1, 0, -1

    #     cmr is a package for additional density plot maps, if you would like to use
    #        >> https://cmasher.readthedocs.io/user/sequential/rainforest.html#rainforest
    #     CM = 'cmr.horizon'
    #     cm = cmr.neon

    fontsize = 12
    tick_fontsize = 12
    fig = plt.figure(figsize=figsize)
    scalling = 1

    # ax = fig.add_subplot(projection=ww.celestial[cutout,cutout])

    pixel_scale = (ww.pixel_scale_matrix[1, 1] * 3600)

    # if projection == 'celestial':
    ax = fig.add_subplot(projection=ww.celestial)

    #     xoffset_in = (xen_cut-xin_cut) * pixel_scale / 2
    #     yoffset_in = (yen_cut-yin_cut) * pixel_scale / 2

    extent = [xin, xen, yin, yen]
    # improve image visualization with normalizations
    norm = visualization.simple_norm(
        hdu[0].data.squeeze()[xin:xen, yin:yen] * scalling, stretch='linear',
        max_percent=max_percent_lowlevel)
    # plot the first normalization (low level, transparent)
    im = ax.imshow(hdu[0].data.squeeze()[xin:xen, yin:yen], origin='lower',
                   cmap=CM, norm=norm, alpha=0.2, extent=extent)

    cm = copy.copy(plt.cm.get_cmap(CM))
    cm.set_under((0, 0, 0, 0))

    data_range = hdu[0].data.squeeze()[xin:xen, yin:yen] * scalling

    # set min and max levels
    if rms != None:  # in case you want to set the min level manually.
        vmin = vmin_factor * rms
        std = rms
    else:
        if mad_std(data_range) == 0:
            std = data_range.std()
        else:
            std = mad_std(data_range)
        vmin = vmin_factor * std
    if vmax is not None:
        vmax = vmax
    else:
        vmax = 1.0 * data_range.max()

    # set second normalization
    norm2 = simple_norm(abs(hdu[0].data.squeeze())[xin:xen, yin:yen] * scalling,
                        min_cut=vmin,
                        max_cut=vmax, stretch='asinh',
                        asinh_a=0.05)  # , max_percent=max_percent_highlevel)
    norm2.vmin = vmin

    # this is what is actually shown on the final plot, better contrast.
    im = ax.imshow(hdu[0].data.squeeze()[xin:xen, yin:yen] * scalling,
                   origin='lower',
                   norm=norm2, cmap=cm, aspect='auto', extent=extent)

    levels_colorbar = np.geomspace(2.0 * vmax, 5 * std,
                                   8)  # draw contours only until 5xstd level.
    levels_neg = np.asarray(levels_neg_factors * std)
    levels_low = np.asarray([4 * std, 3 * std])
    #     levels_neg = np.asarray([])
    #     levels_low = np.asarray([])

    #     levels_colorbar = np.append(levels_neg,levels_pos)

    levels_colorbar2 = np.geomspace(1.0 * vmax, 3 * std,
                                    5)  # draw contours only until 5xstd level.
    ax.contour(hdu[0].data.squeeze()[xin:xen, yin:yen], extent=extent,
               levels=levels_colorbar[::-1], cmap=cmap_cont,
               linewidths=1.25)  # ,alpha=0.6)

    ax.contour(hdu[0].data.squeeze()[xin:xen, yin:yen], levels=levels_low[::-1],
               colors='brown', linestyles=['dashed', 'dashdot'], extent=extent,
               linewidths=1.0, alpha=0.5)
    ax.contour(hdu[0].data.squeeze()[xin:xen, yin:yen], levels=levels_neg[::-1],
               colors='gray', extent=extent,
               linewidths=1.5, alpha=1.0)

    ww.wcs.radesys = 'icrs'
    radesys = ww.wcs.radesys
    if source_distance is not None:
        # distance = source_distance * u.Mpc
        distance = angular_distance_cosmo(source_distance)  # * u.Mpc
        img = hdu[0].data.squeeze()[xin:xen, yin:yen]
        #         scalebar_length = scalebar_length
        scalebar_loc = (0.82, 0.1)  # y, x
        left_side = coordinates.SkyCoord(
            *ww.celestial[xin:xen, yin:yen].wcs_pix2world(
                scalebar_loc[1] * img.shape[1],
                scalebar_loc[0] * img.shape[0],
                0) * u.deg,
            frame=radesys.lower())

        length = (scalebar_length / distance).to(u.arcsec,
                                                 u.dimensionless_angles())
        make_scalebar(ax, left_side, length, color='red', linestyle='-',
                      label=f'{scalebar_length:0.1f}',
                      text_offset=0.1 * u.arcsec, fontsize=24)

    #     _ = ax.set_xlabel(f"Right Ascension {radesys}")
    #     _ = ax.set_ylabel(f"Declination {radesys}")
    _ = ax.set_xlabel(f"Right Ascension")
    _ = ax.set_ylabel(f"Declination")

    if projection == 'celestial':
        ra = ax.coords['ra']
        ra.set_major_formatter('hh:mm:ss.s')
        dec = ax.coords['dec']
        #         ra.set_axislabel(f"RA ({radesys})", fontsize=fontsize)
        #         dec.set_axislabel(f"Dec ({radesys})", fontsize=fontsize, minpad=0.0)
        ra.set_axislabel(f"RA", fontsize=fontsize)
        dec.set_axislabel(f"Dec", fontsize=fontsize, minpad=0.0)

        ra.ticklabels.set_fontsize(tick_fontsize)
        ra.set_ticklabel(exclude_overlapping=True)
        dec.ticklabels.set_fontsize(tick_fontsize)
        dec.set_ticklabel(exclude_overlapping=True)
        ax.tick_params(axis="y", direction="in", pad=-25)
        ax.tick_params(axis="x", direction="in", pad=-25)
        ax.axis(show_axis)
    if projection == 'offset':
        """
        This is a workaround to set the axes in terms of offsets [arcsec] 
        relative to the image center. I found this to work because I must use 
        the wcs coordinates (projection=celestial) if the scale bar is ploted. 
        So this uses the coordinates, but then the axes are removed and on top of 
        that the relative offsets are added.   
        However, this assumes that the source is at the center of the image. 
        """
        ax.axis('off')
        xoffset = (hdu[0].data.shape[0] - xin * 2) * pixel_scale / 2
        yoffset = (hdu[0].data.shape[1] - yin * 2) * pixel_scale / 2
        ax2_x = ax.twinx()
        ax3_y = ax2_x.twiny()
        ax2_x.set_ylabel('Offset [arcsec]', fontsize=14)
        ax3_y.set_xlabel('Offset [arcsec]', fontsize=14)
        ax2_x.yaxis.set_ticks(np.linspace(-xoffset, xoffset, 7))
        ax3_y.xaxis.set_ticks(np.linspace(-yoffset, yoffset, 7))

        ax2_x.tick_params(axis='y', which='both', labelsize=16, color='black',
                          pad=-30)
        ax3_y.tick_params(axis='x', which='both', labelsize=16, color='black',
                          pad=-25)
        ax2_x.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3_y.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2_x.grid(which='both', axis='both', color='gray', linewidth=0.6,
                   alpha=0.7)
        ax3_y.grid(which='both', axis='both', color='gray', linewidth=0.6,
                   alpha=0.7)
        ax2_x.axis(show_axis)
        ax3_y.axis(show_axis)

    freq = '{:.2f}'.format(imhd['refval'][2] / 1e9)
    label_pos_x, label_pos_y = hdu[0].data.squeeze()[xin:xen, yin:yen].shape
    label_pos_x = label_pos_x + xin
    label_pos_y = label_pos_y + yin
    #     ax.annotate(r'' + freq + 'GHz', xy=(0.35, 0.05),  xycoords='figure fraction', fontsize=14,
    #                 color='red')
    ax.annotate(r'' + freq + 'GHz',
                xy=(0.55, 0.82), xycoords='axes fraction',
                fontsize=18,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                color='red')

    if plot_color_bar == True:
        def format_func(value, tick_number, scale_density='mJy'):
            # Use the custom formatter for the colorbar ticks
            mantissa = value * 1000
            return r"${:.1f}$".format(mantissa)

        cax = fig.add_axes(cbar_axes)

        cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                            shrink=1, aspect='auto', pad=10, fraction=1.0,
                            drawedges=False, ticklocation='left')
        #         cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        #         cbar.formatter = mticker.ScalarFormatter(useMathText=True)
        cbar.formatter = CustomFormatter(factor=1000, useMathText=True)
        cbar.update_ticks()

        #         cbar.update_ticks()
        # cbar_axes = [0.12, 0.95, 0.78, 0.06]
        # cax = fig.add_axes(cbar_axes)
        #
        # cbar = fig.colorbar(im, cax=cax, orientation='horizontal', format='%.0e',
        #                     shrink=1.0, aspect='auto', pad=0.1, fraction=1.0, drawedges=False
        #                     )
        cbar.ax.yaxis.set_tick_params(labelleft=True, labelright=False,
                                      tick1On=False, tick2On=False)
        cbar.ax.yaxis.tick_left()
        cbar.set_ticks(levels_colorbar2)
        cbar.set_label(r'Flux [mJy/beam]', labelpad=10, fontsize=16)
        #         cbar.ax.xaxis.set_tick_params(pad=0.1,labelsize=10)
        cbar.ax.tick_params(labelsize=16)
        cbar.outline.set_linewidth(1)
        # cbar.dividers.set_color(None)

        # Make sure the color bar has ticks and labels at the top, since the bar is on the top as well.
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

    imhd = imhead(image)
    if add_inset == 'auto':
        a = imhd['restoringbeam']['major']['value']
        if a / pixel_scale <= 15:
            add_inset = True
        else:
            add_inset = False

    # add residual inset
    if residual_name is not None:
        residual_data = pf.getdata(residual_name)
        axins_re = inset_axes(ax, width="40%", height="40%", loc='lower right',
                              bbox_to_anchor=(0.00, -0.10, 1.0, 1.0),
                              bbox_transform=ax.transAxes)
        xoffset = hdu[0].data.shape[0] * pixel_scale / 2
        yoffset = hdu[0].data.shape[1] * pixel_scale / 2
        extent = [-xoffset, xoffset, -yoffset, yoffset]
        vmin_re = np.min(residual_data)
        vmax_re = np.max(residual_data)

        norm_re = visualization.simple_norm(residual_data * scalling,
                                            stretch='linear',
                                            max_percent=max_percent_lowlevel)
        # im = ax.imshow(fh[0].data.squeeze()[cutout,cutout], cmap='gray_r',norm=norm)

        #     if projection is 'offset':
        imre = axins_re.imshow(residual_data, origin='lower',
                               cmap=CM, norm=norm_re, alpha=0.2, extent=extent)

        norm_re2 = simple_norm(residual_data * scalling, min_cut=vmin_re,
                               max_cut=vmax_re, stretch='linear',
                               asinh_a=0.05)  # , max_percent=max_percent_highlevel)

        imre2 = axins_re.imshow(residual_data * scalling, origin='lower',
                                norm=norm_re2, cmap=CM, aspect='auto',
                                extent=extent)
        # axins_re.contour(residual_data, colors='k',extent=extent,
        #               linewidths=1.0, alpha=1.0)
        #         axins_re.imshow(residual_data, cmap=CM,extent= [-xoffset, xoffset, -yoffset, yoffset],origin='lower')
        axins_re.set_ylabel('', fontsize=10)
        axins_re.set_xlabel('', fontsize=10)
        axins_re.xaxis.set_ticks(np.linspace(-xoffset, xoffset, 4))
        axins_re.yaxis.set_ticks([])
        axins_re.tick_params(axis='y', which='both', labelsize=11, color='black')
        axins_re.tick_params(axis='x', which='both', labelsize=11, color='black')
        axins_re.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axins_re.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #         cbr_re = plt.colorbar(imre2)
        axins_re.axis(show_axis)
    #         axins_re.axis('off')

    if add_inset == True:
        st = imstat(image)
        print('  >> Center --> ', st['maxpos'])
        # sub region of the original image
        yin_cut, yen_cut, xin_cut, xen_cut = st['maxpos'][0] - box_size_inset, \
                                             st['maxpos'][0] + box_size_inset, \
                                             st['maxpos'][1] - box_size_inset, \
                                             st['maxpos'][1] + box_size_inset

        axins = inset_axes(ax, width="40%", height="40%", loc='lower left',
                           bbox_to_anchor=(0.05, -0.05, 1.0, 1.0),
                           bbox_transform=ax.transAxes)

        xoffset_in = (xen_cut - xin_cut) * pixel_scale / 2
        yoffset_in = (yen_cut - yin_cut) * pixel_scale / 2

        extent_inset = [-xoffset_in, xoffset_in, -yoffset_in, yoffset_in]

        Z2 = hdu[0].data.squeeze()[xin_cut:xen_cut, yin_cut:yen_cut]

        vmax_inset = np.max(Z2)
        vmin_inset = 1 * np.std(Z2)

        norm_inset = visualization.simple_norm(Z2, stretch='linear',
                                               max_percent=max_percent_lowlevel)

        norm2_inset = simple_norm(abs(Z2), min_cut=vmin,
                                  max_cut=vmax_inset, stretch='asinh',
                                  asinh_a=0.008)  # , max_percent=max_percent_highlevel)
        norm2_inset.vmin = vmin

        axins.imshow(Z2, cmap=CM, norm=norm_inset, alpha=0.2,
                     extent=extent_inset,
                     aspect='auto', origin='lower')
        axins.imshow(Z2, norm=norm2_inset, extent=extent_inset,
                     cmap=cm,
                     origin="lower", alpha=1.0, aspect='auto')

        axins.set_ylabel('', fontsize=10)
        axins.set_xlabel('', fontsize=10)
        axins.xaxis.set_ticks(np.linspace(-xoffset_in, xoffset_in, 4))
        axins.yaxis.set_ticks(np.linspace(-yoffset_in, yoffset_in, 4))
        axins.tick_params(axis='y', which='both', labelsize=14, color='black')
        axins.tick_params(axis='x', which='both', labelsize=14, color='black')
        axins.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axins.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        axins.grid(True, alpha=0.9)
        levels_inset = np.geomspace(3.0 * Z2.max(), 5 * std,
                                    6)  # draw contours only until 5xstd level.
        levels_inset_neg = np.asarray([-3 * std])
        levels_inset_low = np.asarray([4 * std, 3 * std])

        csi = axins.contour(Z2, levels=levels_colorbar[::-1], cmap=cmap_cont,
                            extent=extent_inset,
                            linewidths=1.25, alpha=1.0)
        #         axins.clabel(csi, inline=False, fontsize=8, manual=False, zorder=99)
        axins.contour(Z2, levels=levels_inset_low[::-1], colors='brown',
                      extent=extent_inset, linestyles=['dashed', 'dashdot'],
                      linewidths=1.0, alpha=1.0)
        axins.contour(Z2, levels=levels_inset_neg[::-1], colors='gray',
                      extent=extent_inset,
                      linewidths=1.5, alpha=1.0)

        # inset zoom box is not working properly, so I am doing it manually.
        import matplotlib.patches as patches
        from matplotlib.patches import FancyArrowPatch
        rect = patches.Rectangle((yin_cut, xin_cut),
                                 xen_cut - xin_cut, yen_cut - yin_cut,
                                 linewidth=2, edgecolor='black',
                                 facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        arrow_start = (yin_cut / 2, xin_cut / 2)
        arrow_end = (yin_cut / 2 + 50, xin_cut / 2 + 50)
        arrow = FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->',
                                mutation_scale=24, linewidth=1.5, color='black')
        ax.add_patch(arrow)

        # manually add lines to connect the rectangle and arrow
        x1, y1 = rect.get_xy()
        x2, y2 = arrow_end
        ax.plot([x1 + 10, x2 + 100], [y1 + 10, y2 + 100], linestyle='--',
                color='black')
        #         ax.indicate_inset_zoom(axins)
        axins.axis(show_axis)

        if add_beam == True:
            from matplotlib.patches import Ellipse
            a_s = imhd['restoringbeam']['major']['value']
            b_s = imhd['restoringbeam']['minor']['value']
            pa_s = imhd['restoringbeam']['positionangle']['value']
            el_s = Ellipse((xoffset_in + 10, yoffset_in + 10), b_s,
                           a_s, angle=pa_s,
                           facecolor='r', alpha=0.5)
            axins.add_artist(el_s)
            el_s.set_clip_box(axins.bbox)

    #         axins.set_xlim(x1, x2)
    #         axins.set_ylim(y1, y2)
    #         axins.set_xticklabels([])
    #         axins.set_yticklabels([])
    #         ax.indicate_inset_zoom(axins, edgecolor="black")

    add_inset2 = False
    if add_inset2 == True:
        #         ax.axis(show_axis)

        st = imstat(image)
        print('  >> Center --> ', st['maxpos'])
        # sub region of the original image
        xin_cut, xen_cut, yin_cut, yen_cut = st['maxpos'][0] - box_size_inset, \
                                             st['maxpos'][0] + box_size_inset, \
                                             st['maxpos'][1] - box_size_inset, \
                                             st['maxpos'][1] + box_size_inset

        print(hdu[0].data.squeeze()[xin_cut:xen_cut, yin_cut:yen_cut].shape)
        Z2 = hdu[0].data.squeeze()[xin_cut:xen_cut, yin_cut:yen_cut]

        vmax_inset = np.max(Z2)
        vmin_inset = 1 * np.std(Z2)

        axins = inset_axes(ax, width="40%", height="40%", loc='lower left',
                           bbox_to_anchor=(0.00, -0.10, 1.0, 1.0),
                           bbox_transform=ax.transAxes)
        # axins = ax.inset_axes([0.00, 0.1, 0.35, 0.4], transform=ax.transData)
        # axins = ax.inset_axes([0.00, 0.1, 0.35, 0.4], transform=ax.transAxes)
        x1, x2, y1, y2 = xin_cut, xen_cut, yin_cut, yen_cut
        print(xen_cut - xin_cut)
        extent = np.asarray([xin_cut,
                             xen_cut,
                             yin_cut,
                             yen_cut
                             ]) * pixel_scale / 2

        xoffset_in = (xen_cut - xin_cut)  # * pixel_scale / 2
        yoffset_in = (yen_cut - yin_cut)  # * pixel_scale / 2
        #         xoffset_in = (xin_cut - xen_cut)
        #         yoffset_in = (yin_cut - yen_cut)
        # extent_arcsec = [-xoffset_in, xoffset_in, -yoffset_in, yoffset_in]
        extent_arcsec = extent
        #         extent_arcsec= [xin_cut* pixel_scale, xen_cut* pixel_scale,
        #                         yin_cut* pixel_scale, yen_cut* pixel_scale]
        norm_inset = visualization.simple_norm(Z2, stretch='linear',
                                               max_percent=max_percent_lowlevel)

        norm2_inset = simple_norm(abs(Z2), min_cut=vmin,
                                  max_cut=vmax_inset, stretch='asinh',
                                  asinh_a=0.05)  # , max_percent=max_percent_highlevel)
        norm2_inset.vmin = vmin

        axins.imshow(Z2, cmap=CM, norm=norm_inset, alpha=0.2,
                     extent=extent_arcsec,
                     aspect='auto', origin='lower')
        axins.imshow(Z2, norm=norm2_inset, extent=extent_arcsec,
                     cmap=cm,
                     origin="lower", alpha=1.0, aspect='auto')

        axins.grid(True, alpha=0.3)
        levels_inset = np.geomspace(3.0 * Z2.max(), 5 * std,
                                    8)  # draw contours only until 5xstd level.
        levels_inset_neg = np.asarray([-3 * std])

        csi = axins.contour(Z2, levels=levels_colorbar[::-1], colors='grey',
                            extent=extent_arcsec,
                            linewidths=1.0, alpha=1.0)
        axins.clabel(csi, inline=False, fontsize=8, manual=False, zorder=99)
        axins.contour(Z2, levels=levels_inset_neg[::-1], colors='k',
                      extent=extent_arcsec,
                      linewidths=1.0, alpha=1.0)

        if add_beam == True:
            from matplotlib.patches import Ellipse
            a_s = imhd['restoringbeam']['major']['value']
            b_s = imhd['restoringbeam']['minor']['value']
            pa_s = imhd['restoringbeam']['positionangle']['value']
            el_s = Ellipse((xin_cut + 10, yin_cut + 10), b_s / pixel_scale,
                           a_s / pixel_scale, angle=pa_s,
                           facecolor='r', alpha=0.5)
            axins.add_artist(el_s)
            el_s.set_clip_box(axins.bbox)

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])

        ax.indicate_inset_zoom(axins, edgecolor="black")

    if add_beam == True:
        from matplotlib.patches import Ellipse
        a = imhd['restoringbeam']['major']['value']
        b = imhd['restoringbeam']['minor']['value']
        pa = imhd['restoringbeam']['positionangle']['value']
        el = Ellipse((hdu[0].data.shape[0] - 60 - xin, 60 + xin),
                     b / pixel_scale, a / pixel_scale, angle=pa,
                     facecolor='black', alpha=1.0)
        ax.add_artist(el, )

        Oa = '{:.2f}'.format(a)
        Ob = '{:.2f}'.format(b)

        blabel_pos_x, blabel_pos_y = hdu[0].data.squeeze()[xin:xen,
                                     yin:yen].shape
        blabel_pos_x = blabel_pos_x + xin
        blabel_pos_y = blabel_pos_y + yin

        #         ax.annotate(r'$' + Oa +'\\times'+Ob+'$',
        #                     xy=(blabel_pos_x* 0.77, blabel_pos_y * 0.58), xycoords='data',
        #                     fontsize=15,bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        #                     color='red')
        ax.annotate(r'$' + Oa + '\\times' + Ob + '$',
                    xy=(0.55, 0.12), xycoords='axes fraction',
                    fontsize=15,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    color='red')

        el.set_clip_box(ax.bbox)

    #     plt.tight_layout()
    if save_name != None:
        #         if not os.path.exists(save_name+special_name+'.jpg'):
        plt.savefig(save_name + special_name + '.jpg', dpi=300,
                    bbox_inches='tight')
        plt.savefig(save_name + special_name + '.pdf', dpi=600,
                    bbox_inches='tight')
    #         else:
    #             print('Skiping save')
    #     plt.show()
    return (plt, ax)


def plot_compare_conv_deconv(conv_image, deconv_image):
    # Create figure and subplots
    fig, ax = plt.subplots(figsize=(6, 3))
    img_size_x = conv_image.shape[0]
    img_size_y = conv_image.shape[1]

    # Plot image on left side
    im1 = ax.imshow(deconv_image, cmap='viridis', extent=[-500, 0, -500, 500], aspect='auto')
    ax.contour(deconv_image, extent=[-500, 0, -500, 500],
               cmap='Greys')  # ,levels=np.geomspace(mad_std(Z1),Z1.max(),3),extent=[-5, 0, -5, 5])

    # Plot image on right side
    im2 = ax.imshow(conv_image, cmap='plasma', extent=[0, 500, -500, 500], aspect='auto')
    ax.contour(conv_image, extent=[0, 500, -500, 500],
               cmap='Greys')  # ,levels=np.geomspace(mad_std(Z1),Z1.max(),3),extent=[-5, 0, -5, 5])
    # # Draw diagonal line
    # line = plt.Line2D([0, 5], [-5, 0], color='white', linewidth=2, linestyle='--')
    # ax.add_line(line)

    # Set axis limits and labels
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # # Add colorbars
    # cbar1 = plt.colorbar(im1, ax=ax, shrink=0.7)
    # cbar1.set_label('Z1')
    # cbar2 = plt.colorbar(im2, ax=ax, shrink=0.7)
    # cbar2.set_label('Z2')

    plt.show()





def add_ellipse(ax, x0, y0, d_r, q, PA, label=None, show_center=True,
                label_offset=5, **kwargs):
    """
    Add an elliptical aperture to a plot with a distance label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to which the ellipse should be added.
    x0 : float
        The x-coordinate of the center of the ellipse.
    y0 : float
        The y-coordinate of the center of the ellipse.
    d_r : float
        The radial distance from the center of the ellipse to its edge.
    q : float
        The axis ratio of the ellipse (b/a).
    PA : float
        The position angle of the major axis of the ellipse, in degrees.
    label : str or None, optional
        The label to use for the distance. If None, no label will be added.
    label_offset : float, optional
        The distance between the label and the ellipse edge, in pixels.
    **kwargs : optional
        Additional arguments that are passed to the Ellipse constructor.
    """
    a = d_r / (1 - q ** 2) ** 0.5
    # a = d_r
    b = a * q
    theta = np.deg2rad(PA)
    ellipse = Ellipse(xy=(x0, y0), width=a * 2, height=b * 2, angle=PA,
                      linestyle='-.', **kwargs)
    ax.add_artist(ellipse)
    if show_center:
        ax.plot(x0, y0, '+', color='white', ms=10)

    # Add distance label
    if label is not None:
        dx = label_offset * np.cos(theta)
        dy = label_offset * np.sin(theta)
        label_x, label_y = x0 + (a + label_offset) * np.cos(theta), y0 + (
                    b + label_offset) * np.sin(theta)
        print(dx, dy)
        ax.annotate(label, xy=(label_x, label_y),
                    xytext=(label_x + dx, label_y + dy),
                    #                     ha='center', va='center',
                    fontsize=14
                    #                     rotation=PA,
                    #                     textcoords='offset pixels'
                    )


def eimshow_with_cgrid(image, pix_to_pc, rms=None, dx=400, dy=400, do_cut=False, data_2D=None,
                       vmin_factor=3.0, vmax_factor=0.1, add_contours=True,
                       neg_levels=np.asarray([-3]),
                       centre=None, apply_mask=False, sigma_mask=6, CM='magma_r',
                       cmap_cont='terrain',
                       dilation_size=None, iterations=2,
                       r_d=200, circles=np.asarray([1, 3, 9])):
    fig = plt.figure(figsize=(4, 4))
    fig.subplots_adjust(wspace=0, hspace=0)
    if data_2D is not None:
        image_data = data_2D
    else:
        image_data = load_fits_data(image)
    cell_size = get_cell_size(image)

    if apply_mask:
        _, mask_dilated = mask_dilation(image, cell_size=cell_size,
                                        sigma=sigma_mask,
                                        dilation_size=dilation_size,
                                        iterations=iterations, rms=rms,
                                        PLOT=False)
        image_data = image_data * mask_dilated

    pos_x, pos_y = nd.maximum_position(image_data)

    ax1 = fig.add_subplot(1, 1, 1)
    #     rms= mad_std(load_fits_data(residuallist_comb[-1]))
    if do_cut == True:
        image_data_plot = image_data[int(pos_x - dx):int(pos_x + dx),
                          int(pos_y - dy):int(pos_y + dy)]
        centre_plot = nd.maximum_position(image_data_plot)
    else:
        centre_plot = pos_x, pos_y
        image_data_plot = image_data
    # im1 = ax1.imshow(deconv_image_cut, cmap='magma', aspect='auto')
    ax1 = eimshow(image_data_plot, rms=rms, vmin_factor=vmin_factor, ax=ax1, CM=CM,
                  extent=[-dx, dx, -dx, dx],
                  neg_levels=neg_levels, cmap_cont=cmap_cont,
                  add_contours=add_contours, vmax_factor=vmax_factor)

    #     if centre:
    #     ax1.plot(centre_plot[0]-(pos_x-dx),centre_plot[1]-(pos_x-dx), marker=r'x', color='black',ms=10)
    ax1.plot(centre_plot[0] * cell_size * 0, centre_plot[1] * cell_size * 0, marker=r'x',
             color='white', ms=10)
    #     ax1.set_title('Deconvolved',size=14)
    ax1 = add_circle_grids(ax=ax1, image=image_data_plot, pix_to_pc=pix_to_pc,
                           #                            center=centre_plot[::-1],
                           center=(centre_plot[1] * cell_size, centre_plot[0] * cell_size),
                           circles=circles, r_d=r_d,  # in pc
                           add_labels=False)

    xticks = np.linspace(-dx, dx, 5)
    xticklabels = np.linspace(-dx * cell_size, +dx * cell_size, 5)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax1.set_yticks(xticks, [])
    ax1.set_xticks(xticks, xticklabels)

    ax1.set_xlabel('offset [arcsec]')

    ax1.grid(False)
    ax1.axis('on')
    return (ax1, image_data_plot)


def add_circle_grids(ax, image, pix_to_pc, r_d=200, add_labels=True,
                     extent=None, center=None,
                     circles=np.asarray([1, 3, 6, 12])):
    # Set the center of the image
    size_max = int(np.sqrt((0.5 * image.shape[0]) ** 2.0 +
                           (0.5 * image.shape[1]) ** 2.0))
    if center is None:
        center = (image.shape[0] // 2, image.shape[1] // 2)

    # Set the radial distance between circles

    r_d_pix = r_d / pix_to_pc
    # Ni = int(size_max/r_d)
    # Create a figure and axis object
    # fig, ax = plt.subplots()
    #     ax = eimshow(load_fits_data(imagelist_vla[-1]),vmin_factor=3,rms=rms)

    # Plot the image
    # ax.imshow(image)

    # Add the circular dashed lines
    prev_text = None
    angle = -90

    for i in range(len(circles)):
        radius = circles[i] * r_d_pix
        circle = Circle(center, circles[i] * r_d_pix, color='limegreen', lw=2,
                        linestyle='-.',
                        fill=False)
        ax.add_patch(circle)
        label = f'{circles[i] * r_d} pc'
        if add_labels:
            x = center[0] + radius * np.cos(np.deg2rad(angle)) + 50
            y = center[1] + radius * np.sin(np.deg2rad(angle)) + 50
            text = Text(x, y, label, ha='center', va='center', color='green')
            ax.add_artist(text)
            prev_text = text
        angle = angle + 90
    # Show the plot
    ax.axis(False)
    return (ax)


def add_contours(ax, image, levels, scale, colors='white', linewidths=1.0):
    """
    Add contours to a plot with distance labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to which the contours should be added.
    image : numpy.ndarray
        The image data.
    levels : list or array-like
        The contour levels to plot.
    scale : float
        The scale of the image, in kpc/pixel.
    colors : str or list or array-like, optional
        The color(s) to use for the contours.
    linewidths : float or list or array-like, optional
        The width(s) of the contour lines.
    """
    # Plot the contours
    cs = ax.contour(image, levels=levels, colors=colors, linewidths=linewidths)

    # Add distance labels
    for i, level in enumerate(levels):
        if len(cs.collections) <= i:
            continue
        c = cs.collections[i]
        label = f"{level * scale:.1f} kpc"
        try:
            x, y = c.get_paths()[0].vertices.mean(axis=0)
        except IndexError:
            continue
        ax.text(x, y, label, color='red', ha='center', va='center')

    # Add a colorbar
    ax.figure.colorbar(cs, ax=ax)


"""

#Printing
"""

def print_subcomponent_stats(df_int):
    total_flux_extended1 = np.mean((df_int['S_R12_py']))
    total_flux_err1 = np.std((df_int['S_R12_py']))
    total_flux_extended2 = np.mean((df_int['S_I3_ext_mask_py'] + df_int['S_I3_ext_full_py']) / 2)
    total_flux_err2 = np.std((df_int['S_I3_ext_mask_py'] + df_int['S_I3_ext_full_py']) / 2)
    total_flux_extended3 = np.mean((df_int['S_I3_ext_mask_py']))
    total_flux_err3 = np.std((df_int['S_I3_ext_mask_py']))
    total_flux_extended4 = np.mean((df_int['S_R12_py'] + df_int['S_I3_res_mask_py']))
    total_flux_err4 = np.std((df_int['S_R12_py'] + df_int['S_I3_res_mask_py']))
    print('Flux 1 on Extended Emission=', total_flux_extended1 * 1000, '+/-', total_flux_err1 * 1000, ' mJy')
    print('Flux 2 on Extended Emission=', total_flux_extended2 * 1000, '+/-', total_flux_err2 * 1000, ' mJy')
    print('Flux 3 on Extended Emission=', total_flux_extended3 * 1000, '+/-', total_flux_err3 * 1000, ' mJy')
    print('Flux 4 on Extended Emission=', total_flux_extended4 * 1000, '+/-', total_flux_err4 * 1000, ' mJy')
    avg_extended = np.mean([total_flux_extended1, total_flux_extended2, total_flux_extended3, total_flux_extended4])
    avg_extended_err = np.std([total_flux_extended1, total_flux_extended2, total_flux_extended3, total_flux_extended4])

    total_flux_extended_f = np.mean((df_int['S_I3_ext_full_frac_py'] + df_int['S_I3_ext_mask_frac_py']) / 2)
    total_flux_f_err = np.std((df_int['S_I3_ext_full_frac_py'] + df_int['S_I3_ext_mask_frac_py']) / 2)

    total_flux_source = np.mean(
        (df_int['S_I2_full_py'] + df_int['S_I2_mask_py'] + df_int['S_I3_full_py'] + df_int['S_I3_mask_py']) / 4)
    total_flux_source_err = np.std(
        (df_int['S_I2_full_py'] + df_int['S_I2_mask_py'] + df_int['S_I3_full_py'] + df_int['S_I3_mask_py']) / 4)

    total_flux_comp = np.mean((df_int['S_model_M13_py'] + df_int['S_M12_py']) / 2)
    total_flux_comp_err = np.std((df_int['S_model_M13_py'] + df_int['S_M12_py']) / 2)
    print('Flux Extended Fraction=', total_flux_extended_f, '+/-', total_flux_f_err)
    print('Flux Fraction Ext/Comp=', avg_extended / total_flux_comp, '+/-', avg_extended_err / total_flux_comp)
    print('Flux Fraction comp/total=', total_flux_comp / total_flux_source, '+/-',
          total_flux_comp_err / total_flux_source)

    total_flux_res = np.mean((df_int['S_I3_res_full_py'] + df_int['S_I3_res_mask_py']) / 2)
    total_flux_res_err = np.std((df_int['S_I3_res_full_py'] + df_int['S_I3_res_mask_py']) / 2)

    max_flux = np.mean(df_int['S_I3_full_py'] + df_int['S_I3_mask_py']) / 2
    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('Averaged Flux on Extended Emission=', avg_extended * 1000, '+/-', avg_extended_err * 1000, ' mJy')
    print('Flux on Compact Emission=', total_flux_comp * 1000, '+/-', total_flux_comp_err * 1000, ' mJy')
    print('Flux on Residual=', total_flux_res * 1000, '+/-', total_flux_res_err * 1000, ' mJy')
    print('Flux Total=', total_flux_source * 1000, '+/-', total_flux_source_err * 1000, ' mJy')
    print('Flux max_flux=', max_flux * 1000, ' mJy')



def plot_fluxes(x_data,y_data,yerr,
                add_fit_legend=True,
                title_text=None,
                log_plot=True,
                labels = [],
                ecolors = [],
                symbols = [],
                figsize=(6,3),
                fig_save_name=None
               ):

    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=figsize,
                                   # gridspec_kw={'height_ratios': [1, 1]}
                                 )
    for i in range(y_data.shape[0]):
    # fig, ax = plt.subplots()
    # x_resample = np.linspace(np.min(x_data)*0.7, np.max(y_data)*1.3, 500)
        
        if labels != []:
            label = labels[i]
        else:
            label = None
            
        if ecolors != []:
            ecolor = ecolors[i]
        else:
            ecolor='black'

        if symbols != []:
            fmt = symbols[i]
        else:
            fmt = 'd'
            
        ax1.errorbar(x_data, y_data[i], yerr=yerr[i], markeredgewidth=3.0,
                     fmt=fmt,label=label, color=ecolor, ecolor=ecolor,alpha=0.5)


    if add_fit_legend == True:
        ax1.legend(loc=(0.05, 0.05),frameon=True,prop={'size': 11})
    if np.nanmax(y_data) < 5:
        ax1.set_ylim(0.05*np.nanmin(y_data),1.2*np.nanmax(y_data))
    else:
        ax1.set_ylim(0.05*np.nanmin(y_data),1.2*np.nanmax(y_data))
    
    if title_text is not None:
        ax1.set_title(title_text)

    if log_plot == True:
        ax1.semilogx()
        ax1.semilogy()

    plt.subplots_adjust(hspace=0.05)
    # ax1.legend(loc=(0.7, 0.55),prop={'size': 14},frameon=False)
    ax1.legend(
        # loc=(0.7, 0.80),
               prop={'size': 14},ncol=2,
               frameon=True)
    ax1.set_ylabel(r'$S_{\nu}$ [mJy]')
    ax1.set_xlabel(r'$\nu$ [GHz]')
    if fig_save_name is not None:
        plt.savefig(f"{fig_save_name}.pdf", bbox_inches='tight', pad_inches = 0.01)
    # plt.show()
    return(fig,ax1)



"""
 ____              _
/ ___|  __ ___   _(_)_ __   __ _
\___ \ / _` \ \ / / | '_ \ / _` |
 ___) | (_| |\ V /| | | | | (_| |
|____/ \__,_| \_/ |_|_| |_|\__, |
                           |___/
#Saving
"""


def save_results_csv(result_mini, save_name, ext='.csv', save_corr=True,
                     save_params=True):
    values = result_mini.params.valuesdict()
    if save_corr:
        try:
            covariance = result_mini.covar
            covar_df = pd.DataFrame(covariance, index=values.keys(),
                                    columns=values.keys())
            covar_df.to_csv(save_name + '_mini_corr' + ext, index_label='parameter')
        except:
            print('Error saving covariance matrix. Skiping...')

    if save_params:
        try:
            stderr = [result_mini.params[name].stderr for name in values.keys()]
            df = pd.DataFrame({'value': list(values.values()), 'stderr': stderr},
                              index=values.keys())
            df.to_csv(save_name + '_mini_params' + ext, index_label='parameter')
        except:
            print('Errors not present in mini, saving only parameters.')
            df = pd.DataFrame(result_mini.params.valuesdict(), index=['value'])
            df.T.to_csv(save_name + '_mini_params' + ext,
                        index_label='parameter')
"""
#Utils
"""

def get_vis_amp_uvwave(vis,spwid,avg_in_time=True,wantedpol='RR,LL'):
    msmd.open(vis)
    scans = msmd.scansforintent('*TARGET*').tolist()
    msmd.close()

    ms.open(vis)
    ms.selectinit(datadescid=spwid)
    ms.select({'scan_number': scans})
    ms.selectpolarization(wantedpol=wantedpol)
    # ms.selectchannel(nchan=1)
    mydata = ms.getdata(['time', 'amplitude', 'axis_info',
                         'u', 'v',
                         'flag'], ifraxis=True)
    ms.close()

    freq_axis = mydata['axis_info']['freq_axis']['chan_freq']
    mydata['amplitude'][mydata['flag']] = np.nan
    antsel = np.ones_like(mydata['axis_info']['ifr_axis']['ifr_shortname'], dtype='bool')

    amp_avg_time = np.nanmean(mydata['amplitude'].T, axis=0)
    amp_avg_chan = np.nanmean(mydata['amplitude'].T, axis=2)
    amp_avg_chan_corr = np.nanmean(amp_avg_chan, axis=2)
    amp_avg_chan_corr_time = np.nanmean(amp_avg_chan_corr, axis=0)

    # freq_axis
    lightspeed = 299792458.0  # speed of light in m/s
    wavelen = (lightspeed / np.mean(freq_axis, axis=0)) * 1e3

    uu = mydata['u'].T / wavelen
    vv = mydata['v'].T / wavelen
    uvwave = np.sqrt(vv ** 2.0 + uu ** 2.0)

    if avg_in_time==True:
        uvwave_final = np.nanmean(uvwave,axis=0)
        amp_avg_final = amp_avg_chan_corr_time
    else:
        uvwave_final = uvwave
        amp_avg_final = amp_avg_chan_corr
    return(uvwave_final[antsel],amp_avg_final[antsel])




def read_spws(vis):
    tb.open(vis + "/DATA_DESCRIPTION")
    SPECTRAL_WINDOW_ID = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()
    # print(SPECTRAL_WINDOW_ID)
    return (SPECTRAL_WINDOW_ID)

def read_spws_v2(vis):
    spwtab = tb.open(vis+'::SPECTRAL_WINDOW')
    tb.close()
    return(spwtab)

def read_colsnames(vis):
    tb.open(vis)
    colnames = tb.colnames()
    tb.close()
    # print(colnames)
    return (colnames)


def query_ms(vis, spwid, query_args=['UVW', 'FLAG']):
    ms.open(vis)
    # select the key
    ms.selectinit(datadescid=spwid)
    query_results = ms.getdata(query_args)
    ms.selectinit(reset=True)
    ms.close()
    return (query_results)

def get_uvwave(vis, spw_id):
    from astropy.constants import c
    ms.open(vis)
    ms.selectinit(spw_id)
    d = ms.getdata(["uvw"])
    ms.done()
    msmd.open(vis)
    chan_freq = msmd.chanfreqs(spw_id)
    msmd.done()
    nchan = len(chan_freq)
    # print(nchan)
    # print(chan_freq)

    # d["uvw"] is an array of float64 with shape [3, nvis]
    u, v, w = d["uvw"]  # unpack into len nvis vectors
    broadcast = np.ones((nchan, 1))
    uu = u * broadcast
    vv = v * broadcast
    ww = w * broadcast

    wavelengths = c.value / chan_freq[:, np.newaxis]  # m
    uw = 1e-3 * uu / wavelengths  # [klambda]
    vw = 1e-3 * vv / wavelengths  # [klambda]
    return (uw, vw, wavelengths)

def get_blinfo():
    return dict([((x, y), np.where((ant1 == x) & (ant2 == y))[0]) 
                 for x in ant_uniq for y in ant_uniq if y > x])


def get_uvwave_tab(vis,index_freq=None):
    # spw info
    def get_blinfo():
        return dict([((x, y), np.where((ant1 == x) & (ant2 == y))[0]) 
                     for x in ant_uniq for y in ant_uniq if y > x])

    lightspeed = 299792458.0 # speed of light in m/s
    
    # spwtab = tb.open(vis+'::SPECTRAL_WINDOW')
    # chan_freq = tb.getcol('CHAN_FREQ').flatten()
    # tb.close()
    ms.open(vis)
    mydata = ms.getdata(['axis_info'])
    chan_freq = mydata['axis_info']['freq_axis']['chan_freq'].flatten()
    ms.close()

    #vis and antenna info
    tb.open(vis)
    uvw = tb.getcol('UVW')
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    flag = tb.getcol('FLAG')
    flag_row = tb.getcol('FLAG_ROW')
    tb.close()
    tb.open(vis+'::ANTENNA')
    station = tb.getcol('STATION')
    tb.close()
    
    ant_uniq = np.unique(np.hstack((ant1, ant2)))
    bldict = get_blinfo()
    if index_freq is not None:
        wavelen = (lightspeed/chan_freq[index_freq])*1e3 # in Klambda
    else:
        wavelen = (lightspeed/chan_freq.mean())*1e3 # in Klambda
    # flag_final = np.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])
    flag_final = np.logical_or(flag, flag_row[np.newaxis,np.newaxis,:])
    # flag_final_VLA = np.logical_or(flag_VLA, flag_row_VLA[:,np.newaxis,np.newaxis])
    # wavelen = (lightspeed/chan_freq.mean())*1e3
    uwave = uvw[0]/wavelen
    vwave = uvw[1]/wavelen
    uvwave = np.sqrt(uwave**2.0 + vwave**2.0)
    return(uvw,uvwave,wavelen)

    
    
def plot_uv_wavelength(vis, color='black', label=None,
                       chunk_size=int(4), downsample_factor=int(50),
                      fig=None, ax=None, alpha=1.0, figsize=(10,10)):
    """
    Plot UV wavelength (uwave vs vwave) for radio interferometric data.
    
    Parameters:
    -----------
    vis : str
        Path to measurement set (MS) file
    color : str, optional
        Color of the plot points
    chunk_size : int, optional
        Size of chunks for frequency averaging
    downsample_factor : int, optional
        Factor by which to downsample the data points
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    alpha : float, optional
        Transparency of plot points
    figsize : tuple, optional
        Figure size if creating new figure
        
    Returns:
    --------
    fig, ax : tuple
        Matplotlib figure and axes objects
    """
    LIGHT_SPEED = 299792458.0  # speed of light in m/s
    
    def get_baseline_info(ant1, ant2, ant_uniq):
        """Create dictionary mapping baseline pairs to data indices."""
        return {(x, y): np.where((ant1 == x) & (ant2 == y))[0]
                for x in ant_uniq for y in ant_uniq if y > x}
    
    msmd = casatools.msmetadata()
    ms = casatools.ms()
    tb = casatools.table()
    print(f"++==> Reading info of visibility: {vis}")
    ms.open(vis)
    mydata = ms.getdata(['axis_info'])
    chan_freq = mydata['axis_info']['freq_axis']['chan_freq'].flatten()
    ms.close()
    
    #vis and antenna info
    tb.open(vis)
    uvw = tb.getcol('UVW')
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    flag = tb.getcol('FLAG')
    flag_row = tb.getcol('FLAG_ROW')
    tb.close()
    tb.open(vis+'::ANTENNA')
    station = tb.getcol('STATION')
    tb.close()
    
    # Get unique antennas and create baseline dictionary
    ant_uniq = np.unique(np.hstack((ant1, ant2)))
    baseline_dict = get_baseline_info(ant1, ant2, ant_uniq)
    
    # Create station labels dictionary
    labels = {bl: f"{station[bl[0]]}-{station[bl[1]]}" 
             for bl in baseline_dict.keys()}
    
    # Setup plotting
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    # Prepare frequency chunks for averaging
    n_chunks = len(chan_freq) // chunk_size
    freq_chunks = chan_freq.reshape((n_chunks, chunk_size))
    chunk_freqs = freq_chunks.mean(axis=1)
    wavelengths = (LIGHT_SPEED / chunk_freqs.reshape(-1, 1)) * 1e3  # in Klambda

    total_baselines = len(list(baseline_dict.keys()))
    print(f"      Processing baselines for uv plot.")
    print(f"      Total number of baselines = {total_baselines}")
    # Plot each baseline
    kk = 0 
    for baseline in tqdm(list(baseline_dict.keys())):
        # Get baseline indices
        bl_indices = baseline_dict[baseline]
        
        # Extract u,v coordinates
        u = uvw[0, bl_indices]
        v = uvw[1, bl_indices]
        
        # Create conjugate points (including a NaN separator)
        # u_points = np.tile(np.hstack([u, np.nan, -u]), (n_chunks, 1)) / wavelengths
        # v_points = np.tile(np.hstack([v, np.nan, -v]), (n_chunks, 1)) / wavelengths
        uwave_points = np.tile(np.hstack([u, -u]), (n_chunks, 1)) / wavelengths
        vwave_points = np.tile(np.hstack([v, -v]), (n_chunks, 1)) / wavelengths
        
        if kk < 1:
            # n_points = 2*len(chan_freq)*uvw.shape[-1]*total_baselines
            n_points = 2*vwave_points.shape[-1]*total_baselines
            print(f"      Total number of uv-lambda points = {n_points/1e6} million")
            print(f"      Total number of downsampled uv-lambda points to plot = {n_points/(chunk_size*downsample_factor*1e6)} million.")
        # Plot with downsampling
        ax.plot(uwave_points[:, ::downsample_factor], 
                vwave_points[:, ::downsample_factor],
                '.', markersize=0.1, color=color, alpha=alpha)
        kk = kk+1
    
    # if label is not None:
        """
        Plot the last set of points to create a label.
        """
        # ax.plot(uwave_points[0, 0], 
        #         vwave_points[0, 0],
        #         '.', markersize=0.1, color=color, 
        #         label = label,
        #         alpha=alpha)
        # ax.legend(loc='upper right')
        # On the last iteration, add a custom legend entry with a larger marker size
        
        # legend_marker = Line2D([0], [0], marker='.', color='w', label=label, 
        #                     markerfacecolor=color, markersize=5, alpha=1.0)
        # ax.add_artist(ax.legend(handles=[legend_marker], loc='upper right', framealpha=0.5))
        
    # ax.set_xlabel('U (kλ)')
    # ax.set_ylabel('V (kλ)')
    # ax.set_title('UV Coverage')
    # ax.grid(True, alpha=0.3)
    # ax.set_aspect('equal')
    
    return fig, ax
    
def plot_uwave_vwave(vis,color=None,fig=None,ax=None,alpha=1.0,
                     chunk_size = int(8), #channel average
                     downsample_factor = 200,
                     figsize=(8,8),
                     uv_ranges=[],
                     plot_axes = True,
                     save_figure=False,
                     show_figure=False,
                     title_text = None,
                     figure_save_name='uvwave_coverage.jpg'
                    ):
    lightspeed = 299792458.0 # speed of light in m/s
    def get_blinfo():
        return dict([((x, y), np.where((ant1 == x) & (ant2 == y))[0]) 
                    for x in ant_uniq for y in ant_uniq if y > x])
    
#     spwtab = tb.open(vis+'::SPECTRAL_WINDOW')
#     chan_freq = tb.getcol('CHAN_FREQ').flatten()
#     tb.close()
    msmd = casatools.msmetadata()
    ms = casatools.ms()
    tb = casatools.table()
    print(f"++==> Reading info of visibility: {vis}")
    
    msmd.open(vis)
    bandwidth = msmd.bandwidths()
    nspw = len(bandwidth)
    chan_freqs_all = np.empty(nspw, dtype=object)
    spws_freq = np.zeros(nspw)

    for nch in range(nspw):
        chan_freqs_all[nch] = msmd.chanfreqs(nch)
        spws_freq[nch] = np.mean(chan_freqs_all[nch])
    msmd.done()
    chan_freq = np.concatenate(chan_freqs_all)
    
    #vis and antenna info
    tb.open(vis)
    uvw = tb.getcol('UVW')
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    flag = tb.getcol('FLAG')
    flag_row = tb.getcol('FLAG_ROW')
    tb.close()
    tb.open(vis+'::ANTENNA')
    station = tb.getcol('STATION')
    tb.close()

    ant_uniq = np.unique(np.hstack((ant1, ant2)))
    bldict = get_blinfo()
    wavelen = (lightspeed/chan_freq.mean())*1e6 # in Klambda
    # flag_final = np.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])
    flag_final = np.logical_or(flag, flag_row[np.newaxis,np.newaxis,:])
    # flag_final_VLA = np.logical_or(flag_VLA, flag_row_VLA[:,np.newaxis,np.newaxis])


    labels = {}
    for bl in bldict.keys():
        labels[bl] = station[bl[0]]+'-'+station[bl[1]]

    # plt.ion() 
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    def chunk_average(arr, chunk_size):
        # Reshape the array to have chunks of size `chunk_size`
        reshaped = arr.reshape(arr.shape[0], -1, chunk_size)
        # Average over each chunk along the last axis
        return reshaped.mean(axis=2)
    
    def downsample(arr, factor):
        return arr[:, ::factor]
    
    chunk_size_plot = 1000
    
    
    
    reshaped_size = int(len(chan_freq)/chunk_size)
    avg_chan = chan_freq.reshape((reshaped_size, chunk_size))
    chunk_chan_mean = avg_chan.mean(axis=1)
    chunk_chan_mean_broad = np.repeat(chunk_chan_mean, chunk_size)
    chunk_chan_mean_column = chunk_chan_mean.reshape(-1, 1)
    UUU = []
    VVV = []
    kk = 0
    # colors = [
    #     '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728',
    #     '#9467BD', '#8C564B', '#E377C2', '#7F7F7F',
    #     '#BCBD22', '#17BECF']
    colors = [
        '#E69F00', '#56B4E9', '#009E73', '#F0E442',
        '#0072B2', '#D55E00', '#CC79A7', '#000000']
    u_points_max = []
    v_points_max = []
    print(f"      Processing baselines for uv plot.")
    print(f"      Total number of baselines = {len(list(bldict.keys()))}")
    kk = 0 
    wavelen = (lightspeed/chunk_chan_mean_column)*1e3 # in Klambda
    for bl in tqdm(list(bldict.keys())[::1]):
        # print(len(list(bldict.keys())))
        
        u = uvw[0,bldict[bl]]
        v = uvw[1,bldict[bl]]
        uuu = np.tile(np.hstack([u, -u]), (reshaped_size, 1))/wavelen
        # print(uuu.shape[0] *uuu.shape[1])
        vvv = np.tile(np.hstack([v, -v]), (reshaped_size, 1))/wavelen
        # Reduce the datasets
        nspw = uuu.shape[0]
        # colors = plt.cm.viridis(np.linspace(0, 1, nspw))
        
        UUU_downsampled = downsample(uuu, downsample_factor)
        VVV_downsampled = downsample(vvv, downsample_factor)
        # print(UUU_downsampled)
        if color is None:
            color = colors[kk % len(colors)]
        # ax.plot(UUU_downsampled[i], VVV_downsampled[i], 'o', markersize=1.0, color=color, alpha=0.7)
        # if kk < 1:
        #     n_points = 2*len(chan_freq)*uvw.shape[-1]*len(list(bldict.keys()))
        #     print(f"      Total number of uv points = {n_points/1e9} billion")
        #     print(f"      Total number of downsampled uv points to plot = {n_points/(chunk_size*downsample_factor*1e9)} billion.")
            
        ax.plot(UUU_downsampled.T, VVV_downsampled.T, '.', markersize=0.5, color=color, alpha=alpha)
        
        max_u_point = np.nanmax(UUU_downsampled)
        max_v_point = np.nanmax(VVV_downsampled)
        u_points_max.append(max_u_point)
        v_points_max.append(max_v_point)
        kk = kk + 1
        # for i in range(nspw):
        #     color = colors[i % len(colors)]
        #     ax.plot(UUU_downsampled[i], VVV_downsampled[i], 'o', markersize=1.0, color=color, alpha=0.7)
#         UUU_reduced = chunk_average(uuu, chunk_size_plot)
#         VVV_reduced = chunk_average(vvv, chunk_size_plot)
#         ax.plot(UUU_reduced.T, VVV_reduced.T, '.', markersize=0.5, color=color, alpha=0.7)
#         # ax.clear()  # Clear previous plot
#         ax.plot(uuu, vvv, '.',markersize=0.1,color=color,alpha=alpha)#, label=labels[bl])
#     #     fig.canvas.draw()
#     #     fig.canvas.flush_events()
#     # plt.ioff()
#     # plt.show()
    
    # xy_limits = [np.nanmin([np.nanmin(VVV_downsampled), np.nanmin(UUU_downsampled)]),
    #              np.nanmax([np.nanmax(VVV_downsampled), np.nanmax(UUU_downsampled)])]
    xy_limits = [-np.nanmax([np.nanmax(v_points_max), np.nanmax(u_points_max)]),
                 np.nanmax([np.nanmax(v_points_max), np.nanmax(u_points_max)])]
    
    if plot_axes == True:
        ax.set_xlabel('$u$ [k$\lambda$]')
        ax.set_ylabel('$v$ [k$\lambda$]')
        # ax.grid()
        if title_text is not None:
            ax.set_title(title_text)
    uv_ranges.append(xy_limits)
    # ax.set_ylim(xy_limits[0]*1.1,xy_limits[1]*1.1)
    # ax.set_xlim(xy_limits[0]*1.1,xy_limits[1]*1.1)
    ax.set_ylim(np.nanmin(uv_ranges)*1.1,np.nanmax(uv_ranges)*1.1)
    ax.set_xlim(np.nanmin(uv_ranges)*1.1,np.nanmax(uv_ranges)*1.1)
    ax.axis('equal')
    if save_figure:
        plt.savefig(figure_save_name,bbox_inches='tight',dpi=300)
        plt.clf()
        plt.close()
    if show_figure:
        plt.show()
        plt.clf()
        plt.close()
    # return(fig,ax,uuu,UUU_downsampled,VVV_downsampled,uvw)
    return(fig,ax,uv_ranges)


def plot_uwave_vwave_v3(vis,color=None,fig=None,ax=None,alpha=1.0,
                     chunk_size = int(8), #channel average
                     downsample_factor = 200,
                     chunk_size_plot = 100,
                     figsize=(8,8),
                     uv_ranges=[],
                     plot_axes = True,
                     save_figure=False,
                     show_figure=False,
                     title_text = None,
                     figure_save_name='uvwave_coverage.jpg'
                    ):
    lightspeed = 299792458.0 # speed of light in m/s
    def get_blinfo():
        return dict([((x, y), np.where((ant1 == x) & (ant2 == y))[0]) 
                    for x in ant_uniq for y in ant_uniq if y > x])
    

    msmd = casatools.msmetadata()
    ms = casatools.ms()
    tb = casatools.table()
    print(f"++==> Reading info of visibility: {vis}")
    
    msmd.open(vis)
    bandwidth = msmd.bandwidths()
    nspw = len(bandwidth)
    chan_freqs_all = np.empty(nspw, dtype=object)
    spws_freq = np.zeros(nspw)

    for nch in range(nspw):
        chan_freqs_all[nch] = msmd.chanfreqs(nch)
        spws_freq[nch] = np.mean(chan_freqs_all[nch])
    msmd.done()
    chan_freq = np.concatenate(chan_freqs_all)
    
    #vis and antenna info
    tb.open(vis)
    uvw = tb.getcol('UVW')
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    flag = tb.getcol('FLAG')
    flag_row = tb.getcol('FLAG_ROW')
    tb.close()
    tb.open(vis+'::ANTENNA')
    station = tb.getcol('STATION')
    tb.close()

    ant_uniq = np.unique(np.hstack((ant1, ant2)))
    bldict = get_blinfo()
    wavelen = (lightspeed/chan_freq.mean())*1e6 # in Klambda
    # flag_final = np.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])
    flag_final = np.logical_or(flag, flag_row[np.newaxis,np.newaxis,:])
    # flag_final_VLA = np.logical_or(flag_VLA, flag_row_VLA[:,np.newaxis,np.newaxis])


    labels = {}
    for bl in bldict.keys():
        labels[bl] = station[bl[0]]+'-'+station[bl[1]]

    # plt.ion() 
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    
    def chunk_average(arr, chunk_size):
        # Reshape the array to have chunks of size `chunk_size`
        reshaped = arr[:, :arr.shape[1] - arr.shape[1] % chunk_size].reshape(arr.shape[0], -1, chunk_size)
        # Average over each chunk along the last axis
        return reshaped.mean(axis=2)


    
    def downsample(arr, factor):
        return arr[:, ::factor]
    
    
    
    
    reshaped_size = len(chan_freq) // chunk_size_plot
    avg_chan = chan_freq[:reshaped_size * chunk_size_plot].reshape(reshaped_size, chunk_size_plot)
    chunk_chan_mean = avg_chan.mean(axis=1)
    chunk_chan_mean_broad = np.repeat(chunk_chan_mean, chunk_size_plot)
    chunk_chan_mean_column = chunk_chan_mean.reshape(-1, 1)
    UUU = []
    VVV = []
    kk = 0

    colors = [
        '#E69F00', '#56B4E9', '#009E73', '#F0E442',
        '#0072B2', '#D55E00', '#CC79A7', '#000000']
    u_points_max = []
    v_points_max = []
    print(f"      Processing baselines for uv plot.")
    print(f"      Total number of baselines = {len(list(bldict.keys()))}")
    kk = 0 
    for bl in tqdm(list(bldict.keys())[::1]):
        # print(len(list(bldict.keys())))
        wavelen = (lightspeed / chunk_chan_mean_column) * 1e3  # in Klambda
        u = uvw[0, bldict[bl]]
        v = uvw[1, bldict[bl]]
        uuu = np.tile(np.hstack([u, np.nan, -u]), (reshaped_size, 1)) / wavelen
        # print(uuu.shape[0] *uuu.shape[1])
        vvv = np.tile(np.hstack([v, np.nan, -v]), (reshaped_size, 1)) / wavelen
        # Reduce the datasets
        nspw = uuu.shape[0]
        
        UUU_downsampled = chunk_average(uuu, downsample_factor)
        VVV_downsampled = chunk_average(vvv, downsample_factor)
        # print(UUU_downsampled)
        
        if color is None:
            color = colors[kk % len(colors)]
            
        if UUU_downsampled.size > 0 and UUU_downsampled.size > 0:
            # Plot the averaged points if they are not empty
            ax.plot(UUU_downsampled.T, VVV_downsampled.T, '.', markersize=0.5, color=color, alpha=alpha)
            max_u_point = np.nanmax(UUU_downsampled)
            max_v_point = np.nanmax(VVV_downsampled)
            u_points_max.append(max_u_point)
            v_points_max.append(max_v_point)
            
        kk = kk + 1
        
    xy_limits = [-np.nanmax([np.nanmax(v_points_max), np.nanmax(u_points_max)]),
                 np.nanmax([np.nanmax(v_points_max), np.nanmax(u_points_max)])]
    
    if plot_axes == True:
        ax.set_xlabel('$u$ [k$\lambda$]')
        ax.set_ylabel('$v$ [k$\lambda$]')
        # ax.grid()
        if title_text is not None:
            ax.set_title(title_text)
    uv_ranges.append(xy_limits)

    ax.set_ylim(np.nanmin(uv_ranges)*1.1,np.nanmax(uv_ranges)*1.1)
    ax.set_xlim(np.nanmin(uv_ranges)*1.1,np.nanmax(uv_ranges)*1.1)
    ax.axis('equal')
    if save_figure:
        plt.savefig(figure_save_name,bbox_inches='tight',dpi=300)
        plt.clf()
        plt.close()
    if show_figure:
        plt.show()
        plt.clf()
        plt.close()
    return(fig,ax,uv_ranges)



def plot_uwave_vwave_v4(vis, color=None, fig=None, ax=None, alpha=1.0,
                        chunk_size=int(8),  # channel average
                        downsample_factor=200,
                        chunk_size_plot=100,
                        figsize=(8, 8),
                        uv_ranges=[],
                        plot_axes=True,
                        save_figure=False,
                        show_figure=False,
                        title_text=None,
                        figure_save_name='uvwave_coverage.jpg'):
    
    # Constants
    lightspeed = 299792458.0  # speed of light in m/s

    def get_blinfo():
        # Dictionary mapping antenna pairs to the indices of their baselines
        return dict([((x, y), np.where((ant1 == x) & (ant2 == y))[0])
                    for x in ant_uniq for y in ant_uniq if y > x])
    
    # Reading metadata and visibility information
    msmd = casatools.msmetadata()
    tb = casatools.table()
    print(f"++==> Reading info of visibility: {vis}")
    
    # Get spectral information
    msmd.open(vis)
    nspw = msmd.nspw()
    chan_freqs_all = np.array([msmd.chanfreqs(spw) for spw in range(nspw)], dtype=object)
    msmd.done()
    chan_freq = np.concatenate(chan_freqs_all)
    
    # Get visibility and antenna data
    tb.open(vis)
    uvw = tb.getcol('UVW')
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    flag = tb.getcol('FLAG')
    flag_row = tb.getcol('FLAG_ROW')
    tb.close()
    
    tb.open(vis + '::ANTENNA')
    station = tb.getcol('STATION')
    tb.close()

    # Get unique antennas and baselines
    ant_uniq = np.unique(np.hstack((ant1, ant2)))
    bldict = get_blinfo()

    # Handle flags
    flag_final = np.logical_or(flag, flag_row[np.newaxis,np.newaxis,:])

    # Wavelength calculation for each channel
    wavelengths = lightspeed / chan_freq

    # Average the channels for better performance
    reshaped_size = len(chan_freq) // chunk_size_plot
    avg_chan = chan_freq[:reshaped_size * chunk_size_plot].reshape(reshaped_size, chunk_size_plot)
    chunk_chan_mean = avg_chan.mean(axis=1)
    wavelengths_mean = lightspeed / chunk_chan_mean

    # Initialize plot if not provided
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
    
    u_points_max = []
    v_points_max = []
    
    print(f"      Processing baselines for uv plot.")
    print(f"      Total number of baselines = {len(list(bldict.keys()))}")
    
    # Loop through the baselines
    for kk, bl in enumerate(tqdm(list(bldict.keys()))):
        u = uvw[0, bldict[bl]]
        v = uvw[1, bldict[bl]]
        
        # Apply flagging
        # flagged_u = np.ma.masked_array(u, flag_final[bldict[bl]])
        # flagged_v = np.ma.masked_array(v, flag_final[bldict[bl]])

        # # Skip this baseline if there's no unflagged data
        # if flagged_u.size == 0 or flagged_v.size == 0:
        #     continue
        
        # # Average the u and v points over chunks
        # valid_size = flagged_u.size // chunk_size_plot * chunk_size_plot
        # if valid_size == 0:
        #     continue
        
        # Reshape and average
        u_chunked = u.reshape(-1, chunk_size_plot).mean(axis=1)
        v_chunked = v.reshape(-1, chunk_size_plot).mean(axis=1)

        # Convert u and v to uwave and vwave using averaged wavelengths
        uwave = u_chunked / wavelengths_mean
        vwave = v_chunked / wavelengths_mean

        # Downsample the data for efficient plotting
        uwave_downsampled = uwave[::downsample_factor]
        vwave_downsampled = vwave[::downsample_factor]

        # Choose color for the baseline
        plot_color = color if color is not None else colors[kk % len(colors)]
        
        # Plot the downsampled data if non-empty
        if uwave_downsampled.size > 0 and vwave_downsampled.size > 0:
            ax.plot(uwave_downsampled, vwave_downsampled, '.', markersize=0.5, color=plot_color, alpha=alpha)
            u_points_max.append(np.nanmax(np.abs(uwave_downsampled)))
            v_points_max.append(np.nanmax(np.abs(vwave_downsampled)))

    # Set plot limits
    if u_points_max and v_points_max:
        max_limit = np.nanmax([np.nanmax(u_points_max), np.nanmax(v_points_max)])
        xy_limits = [-max_limit, max_limit]
        ax.set_xlim(xy_limits)
        ax.set_ylim(xy_limits)

    # Set plot labels and title
    if plot_axes:
        ax.set_xlabel('$u$ [kλ]')
        ax.set_ylabel('$v$ [kλ]')
        if title_text is not None:
            ax.set_title(title_text)
    ax.axis('equal')
    if save_figure:
        plt.savefig(figure_save_name, bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()
    
    if show_figure:
        plt.show()
        plt.clf()
        plt.close()
    
    return fig, ax, uv_ranges




# def plot_uwave_vwave_v2(vis,color='black',fig=None,ax=None,alpha=1.0):
#     lightspeed = 299792458.0 # speed of light in m/s
#     def get_blinfo():
#         return dict([((x, y), np.where((ant1 == x) & (ant2 == y))[0]) 
#                     for x in ant_uniq for y in ant_uniq if y > x])
#     spwtab = tb.open(vis+'::SPECTRAL_WINDOW')
#     chan_freq = tb.getcol('CHAN_FREQ').flatten()
#     tb.close()

#     #vis and antenna info
#     tb.open(vis)
#     uvw = tb.getcol('UVW')
#     ant1 = tb.getcol('ANTENNA1')
#     ant2 = tb.getcol('ANTENNA2')
#     flag = tb.getcol('FLAG')
#     flag_row = tb.getcol('FLAG_ROW')
#     tb.close()
#     tb.open(vis+'::ANTENNA')
#     station = tb.getcol('STATION')
#     tb.close()

#     ant_uniq = np.unique(np.hstack((ant1, ant2)))
#     bldict = get_blinfo()
#     wavelen = (lightspeed/chan_freq.mean())*1e6 # in Klambda
#     # flag_final = np.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])
#     flag_final = np.logical_or(flag, flag_row[np.newaxis,np.newaxis,:])
#     # flag_final_VLA = np.logical_or(flag_VLA, flag_row_VLA[:,np.newaxis,np.newaxis])


#     labels = {}
#     for bl in bldict.keys():
#         labels[bl] = station[bl[0]]+'-'+station[bl[1]]

#     # plt.ion() 
#     if fig is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
    
#     def chunk_average(arr, chunk_size):
#         # Reshape the array to have chunks of size `chunk_size`
#         reshaped = arr.reshape(arr.shape[0], -1, chunk_size)
#         # Average over each chunk along the last axis
#         return reshaped.mean(axis=2)
    
#     # Function to downsample an array along its columns (data points)
#     def downsample(arr, factor):
#         return arr[:, ::factor]

#     # Function to average an array's rows (channels)
#     def average_channels(arr, factor):
#         # Reshape the array to have rows of size `factor`
#         reshaped = arr.reshape(-1, factor, arr.shape[1])
#         # Average over each set of rows
#         return reshaped.mean(axis=1)
    
#     # Downsampling and averaging factors
#     downsample_factor = 1000  # For data points
#     downsample_channel_factor = 64  # For channels
    
#     chunk_size = int(12) #channel average
#     reshaped_size = int(len(chan_freq)/chunk_size)
#     avg_chan = chan_freq.reshape((reshaped_size, chunk_size))
#     chunk_chan_mean = avg_chan.mean(axis=1)
#     chunk_chan_mean_broad = np.repeat(chunk_chan_mean, chunk_size)
#     chunk_chan_mean_column = chunk_chan_mean.reshape(-1, 1)
#     UUU = []
#     VVV = []
#     for bl in tqdm(list(bldict.keys())[::1]):
#         wavelen = (lightspeed/chunk_chan_mean_column)*1e3 # in Klambda
#         u = uvw[0,bldict[bl]]
#         v = uvw[1,bldict[bl]]
#         uuu = np.tile(np.hstack([u, np.nan, -u]), (reshaped_size, 1))/wavelen
#         # print(uuu.shape[0] *uuu.shape[1])
#         vvv = np.tile(np.hstack([v, np.nan, -v]), (reshaped_size, 1))/wavelen
#         # Reduce the datasets
        
#         # Step 1: Downsample data points
#         UUU_downsampled = downsample(uuu, downsample_factor)
#         VVV_downsampled = downsample(vvv, downsample_factor)

#         # Step 2: Average channels
#         UUU_reduced = average_channels(UUU_downsampled, downsample_channel_factor)
#         VVV_reduced = average_channels(VVV_downsampled, downsample_channel_factor)

#         ax.plot(UUU_reduced.T, VVV_reduced.T, '.', markersize=0.5, color=color, alpha=0.7)
# #         UUU_reduced = chunk_average(uuu, chunk_size_plot)
# #         VVV_reduced = chunk_average(vvv, chunk_size_plot)
# #         ax.plot(UUU_reduced.T, VVV_reduced.T, '.', markersize=0.5, color=color, alpha=0.7)
# #         # ax.clear()  # Clear previous plot
# #         ax.plot(uuu, vvv, '.',markersize=0.1,color=color,alpha=alpha)#, label=labels[bl])
# #     #     fig.canvas.draw()
# #     #     fig.canvas.flush_events()
# #     # plt.ioff()
# #     # plt.show()
#     return(fig,ax)




def adjust_arrays(a, b):
    len_a, len_b = len(a), len(b)

    # If a is shorter, pad it with NaN values
    if len_a < len_b:
        a = np.pad(a, (0, len_b - len_a), 'constant', constant_values=np.nan)
    # If b is shorter, pad it with NaN values
    elif len_b < len_a:
        b = np.pad(b, (0, len_a - len_b), 'constant', constant_values=np.nan)

    return a, b


def get_phase_centre(vis):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    msmd = casatools.msmetadata()
    ms = casatools.ms()
    tb = casatools.table()

    msmd.open(vis)
    ra_radians = msmd.phasecenter()['m0']['value']
    dec_radians = msmd.phasecenter()['m1']['value']
    msmd.close()
    # Convert to SkyCoord object
    coord = SkyCoord(ra=ra_radians * u.radian, dec=dec_radians * u.radian, frame='icrs')

    # Format the output using 'hmsdms'
    formatted_coord = coord.to_string('hmsdms')
    formatted_ra, formatted_dec = formatted_coord.split()

    formatted_ra_hms = formatted_ra.replace('h', ':').replace('m', ':').replace('s', '')
    formatted_dec_dms = formatted_dec.replace('d', '.').replace('m', '.').replace('s', '')

    formatted_output = "J2000 {} {}".format(formatted_ra_hms, formatted_dec_dms)

    print(formatted_output)
    return (formatted_output)

"""
EXPERIMENTAL
"""

#Pytorch
def torch_fft_convolve(image_np, psf_np):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert numpy arrays to PyTorch tensors
    image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)
    psf_tensor = torch.from_numpy(psf_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # Zero-pad the PSF to match the image size
    psf_padded = torch.zeros_like(image_tensor)
    psf_padded[0, 0, :psf_np.shape[0], :psf_np.shape[1]] = psf_tensor[0, 0]

    # FFT-based convolution
    image_fft = torch.fft.fft2(image_tensor)
    psf_fft = torch.fft.fft2(psf_padded)
    result_fft = image_fft * psf_fft
    result = torch.fft.ifft2(result_fft).real

    return result.squeeze(0).squeeze(0).cpu().numpy()

def write_data_for_SED_fitter(freq, flux, flux_err, theta1, theta2, flag, filename):
    with open(filename, 'w') as file:
        # Writing the header
        file.write("# Space for commenting the references for the fluxes\n")
        file.write("#--------------------------------------------------------------------------\n")
        file.write("#Freq     Flux       Flux_err   theta1    theta2   Flag\n")
        file.write("# GHz     mJy        mJy        arcsec    arcsec   0=data,1=UL,2=fake\n")
        file.write("#--------------------------------------------------------------------------\n")
        
        # Writing the data
        for f, _flux, _flux_err, t1, t2, fl in zip(freq, flux, flux_err, theta1, theta2, flag):
            file.write(f"{f:.5f}   {_flux:.5f}   {_flux_err:.5f}     {t1:.2f}      {t2:.2f}      {fl}\n")

def write_data_for_SED_fitter_v2(freq, flux, flux_err, flux_err_3, theta1, theta2, flag, filename):
    with open(filename, 'w') as file:
        # Writing the header
        file.write("# Space for commenting the references for the fluxes\n")
        file.write("#--------------------------------------------------------------------------\n")
        file.write("#Freq     Flux        Flux_err   Flux_err_3  theta1   theta2   Flag\n")
        file.write("# GHz     mJy         mJy        mJy         arcsec   arcsec   0=data,1=UL,2=fake\n")
        file.write("#--------------------------------------------------------------------------\n")
        
        # Writing the data
        for f, _flux, _flux_err, _flux_err_3, t1, t2, fl in zip(freq, flux, flux_err, flux_err_3, theta1, theta2, flag):
            file.write(f"{f:.5f}   {_flux:.5f}    {_flux_err:.5f}    {_flux_err_3:.5f}     {t1:.2f}     {t2:.2f}     {fl}\n")

def prepare_data(root_paths,prefix_images):
    MFS_images = []
    MFS_residuals = []
    imagelist = []
    residuallist = []
    freqlist = []
    for i in range(len(root_paths)):
        root_path_i = root_paths[i]
        prefix_images_i = prefix_images[i]
        imagelist_i = glob.glob(root_path_i+prefix_images_i)
        imagelist_i, residuallist_i, freqs_i = sort_list_by_freq(imagelist_i)
        MFS_images.append(imagelist_i[0].replace('0000-image','MFS-image'))
        MFS_residuals.append(residuallist_i[0].replace('0000-residual','MFS-residual'))

        imagelist.extend(imagelist_i)
        residuallist.extend(residuallist_i)
        freqlist.extend(freqs_i)
        
    freqlist_MFS = getfreqs(MFS_images)


    imagelist_beam,residuallist_beam = \
        sort_list_by_beam_size(imagelist=MFS_images,
                                     return_df = False)
#     ax=eimshow(imagelist_beam[0],crop=True,add_beam=True)
    return(np.asarray(MFS_images),
           np.asarray(MFS_residuals),
           np.asarray(imagelist),
           np.asarray(residuallist),
           freqlist_MFS,
           np.asarray(freqlist))


def compute_image_stats_wsclean(path,
                        image_list,
                        image_statistics,
                        prefix='',
                        show_figure=True):
    """
    This function will compute statistics of a cleaned image from a wsclean run
    (provided an image prefix). It will also store
    associated model and residual images.

    Parameters
    ----------
    path : str
        Path to the image files.
    image_list : list
        List to store  the image names of each self-cal step.
    image_statistics : dict
        Dictionary to store the statistics of the images at a given self-cal step.
        It can be an existing dictionary.
    prefix : str
        Prefix of the image files.

    """
    file_list = glob.glob(f"{path}*{prefix}*MFS-image.fits")
    file_list.sort(key=os.path.getmtime, reverse=False)
    try:
        image_list[prefix] = file_list[-1]
    except:
        image_list[prefix] = file_list
    image_list[prefix+'_residual'] = image_list[prefix].replace(
        'MFS-image.fits', 'MFS-residual.fits')
    image_list[prefix+'_model'] = image_list[prefix].replace(
        'MFS-image.fits', 'MFS-model.fits')

    sigma = 6.0
    

    level_stats = level_statistics(image_list[prefix],sigma=sigma)
    image_stats = get_image_statistics(imagename=image_list[prefix],
                                             residual_name = image_list[prefix+'_residual'],
                                             dic_data=level_stats,
                                             sigma_mask=sigma)
    img_props = compute_image_properties(image_list[prefix],
                                               image_list[prefix+'_residual'],
                                               results = image_stats,
                                               sigma_mask = sigma,
                                               show_figure=show_figure)[-1]



    image_statistics[prefix] = img_props
    return(image_statistics,image_list)

def create_wsclean_mask(imagename,rms_mask,sigma_mask,mask_grow_iterations,PLOT=False):
    original_mask,dilated_mask = mask_dilation(imagename,
                                                     PLOT=PLOT,
                                                     rms=rms_mask,
                                                     sigma=sigma_mask,
                                                     iterations=mask_grow_iterations)

    mask = dilated_mask
    mask_wslclean = mask * 1.0  # mask in wsclean is inverted??? But how this works?
    mask_name = imagename.replace('.fits', '') + '_mask.fits'
    pf.writeto(mask_name, mask_wslclean, overwrite=True)
    
    #copy the fits header from the original image into the mask.
    copy_header(imagename,
                      mask_name,
                      mask_name)
    
    return(mask_name,dilated_mask)


def run_wsclean(g_name, imsize='2048', imsizey='2048',cell='0.06asec',
                opt_args = "''",
                robust=0.5,base_name=None,
                savemodel=False,shift=None,
                nsigma_automask='4.0', nsigma_autothreshold='2.0',
                datacolumn='DATA',mask=None,
                niter=1000,quiet=True,
                with_multiscale=False, scales='None',
                uvtaper=[],nc = 4,
                image_list={},image_statistics={}):


    g_vis = g_name + '.ms'
    if imsizey is None:
        imsizey = imsize
    if base_name is None:
        base_name  = 'final_image'
    else:
        base_name = base_name

    os.system("export OPENBLAS_NUM_THREADS=1 && python /mnt/ext10TB/GitHub/morphen/selfcal/imaging_with_wsclean.py --f " +
              g_name + " --sx "
              + str(imsize) + " --sy " + str(imsizey) + " --niter "
              + str(niter) + " --data " + datacolumn + " --cellsize " + cell
              + ' --nsigma_automask ' + nsigma_automask + ' --mask '+str(mask)
              + ' --nsigma_autothreshold ' + nsigma_autothreshold
              + ' --scales ' + scales + ' --nc ' + str(nc)
              +' --opt_args '+ opt_args
              +' --quiet '+ str(quiet) + ' --with_multiscale '+str(with_multiscale)
              + ' --shift ' + str(shift)
              + " --r " + str(robust) + " --t "+str(uvtaper)
              + " --update_model " + str(savemodel) + " --save_basename " + base_name)

    image_statistics,image_list = compute_image_stats_wsclean(path=os.path.dirname(g_name) + '/',
                                                    image_list=image_list,
                                                    image_statistics=image_statistics,
                                                    prefix=base_name)

    print('Using residual map as RMS estimator:',image_list[base_name+'_residual'])
    rms_mask = mad_std(load_fits_data(image_list[base_name+'_residual']))
    mask_name,dilated_mask = create_wsclean_mask(image_list[base_name],
                                         rms_mask=rms_mask,
                                         sigma_mask=6.0,
                                         mask_grow_iterations=2,
                                         PLOT=True)
    
    
#     eimshow(image_list[base_name],
#                   rms=mad_std(load_fits_data(image_list[base_name+'_residual'])),
#                   # crop=True,box_size=300,
#                   crop=True,box_size=int(4*image_statistics[base_name]['C95radii']),
#                   add_beam=True)


    try:
        centre = nd.maximum_position(load_fits_data(image_list[base_name]))[::-1]
        # centre = (int(image_statistics[prefix]['y0']),int(image_statistics[prefix]['x0']))
        fig = plt.figure(figsize=(16, 8))
        ax0 = fig.add_subplot(1, 2, 2)
        ax0 = eimshow(imagename = image_list[base_name],
                            center=centre,
                            projection = 'offset',plot_colorbar=True,
                            vmax_factor=0.8,
                            rms=mad_std(load_fits_data(image_list[base_name + '_residual'])),
                            # crop=True,box_size=300,
                            figsize=(8, 8), ax=ax0,fig=fig,
                            crop=True, box_size=int(4 * image_statistics[base_name]['C95radii']),
                            # save_name=image_list[prefix].replace('.fits', '_map'),
                            add_beam=True)
        ax0.set_title(f'Radio Map')
        ax1 = fig.add_subplot(1, 2, 1)
        ax1 = eimshow(imagename = image_list[base_name + '_residual'],
                            center=centre,
                            projection='offset',
                            vmin_factor=-3.0, vmax_factor=0.99,
                            add_contours=False,
                            figsize=(8, 8), ax=ax1,fig=fig,
                            crop=True, box_size=int(4 * image_statistics[base_name]['C95radii']),
                            save_name=image_list[base_name].replace('.fits', '_map'),
                            plot_title = f'Residual Map', plot_colorbar=False,
                            add_beam=False)

    except:
        print('--==>> Error on plotting radio map.')
        pass
    

    sub_band_images = glob.glob(image_list[base_name].replace('-MFS-image.fits','')+'-????-image.fits')
    sub_band_residuals = []
    for i in range(len(sub_band_images)):
        sub_band_residuals.append(sub_band_images[i].replace('-image.fits','-residual.fits'))

    
    
    _FLUXES = []
    _FLUXES_err = []
#     mask_MFS = load_fits_data(mask_name)
    for i in range(len(sub_band_images)):
        print('++>> Computing flux density on sub-band image ',sub_band_images[i])
        print('++>> Associated sub-band  residual image is ',sub_band_residuals[i])
        
        img_props = compute_image_properties(sub_band_images[i],
                                                sub_band_residuals[i],
                                                sigma_mask = 6.0,
                                                last_level=1.0,
                                                mask=dilated_mask,
                                                verbose=1,
                                                show_figure=True)[-1]
        
        # flux_density,flux_density_err = img_props['total_flux_levels'], img_props['flux_error_res_3']
        flux_density,flux_density_err = img_props['total_flux_mask'], img_props['flux_error_res_3']
        
        print('Flux density = ', flux_density)
        _FLUXES.append(flux_density)
        _FLUXES_err.append(flux_density_err)
    FLUXES = np.asarray(_FLUXES)
    FLUXES_err = np.asarray(_FLUXES_err)
    freqlist = getfreqs(sub_band_images)

#     plt.figure(figsize=(8, 6))
#     plt.errorbar(freqlist/1e9, FLUXES*1000, 
#                     yerr=FLUXES_err*1000,
#                     fmt='o', 
#                     # label='Observed data', 
#                     color='k', ecolor='gray', alpha=0.5)
#     plt.xlabel('Frequency [GHz]')
#     plt.ylabel('Flux Density [mJy]')
#     plt.ylim(0,)
#     plt.title('Sub-Band Images')
#     plt.savefig(image_list[base_name].replace('-MFS-image.fits','_freq_flux.jpg'),dpi=300,bbox_inches='tight')
#     plt.show()
    mini,result_1 = do_fit_spec_RC_linear(freqlist,FLUXES*1000,FLUXES_err*1000,
                                            plot_errors_shade = True, do_mcmc_fit=False,
                                            basename_save=image_list[base_name],
                                            verbose=2)
    print('---------------------------------------------')
    print('------------ RESTORING BEAM SIZE ------------')
    Omaj,Omin, _, _, _ = beam_shape(image_list[base_name])
    print(f"                {Omaj:.3f} arcec            ")
    print('---------------------------------------------')
    
    
    return(image_list,image_statistics,Omaj)


# import builtins
# import inspect

# # # Override the built-in print function
# # original_print = builtins.print

# # def custom_print(*args, **kwargs):
# #     frame = inspect.currentframe().f_back
# #     info = inspect.getframeinfo(frame)
# #     file_name = info.filename
# #     line_number = info.lineno
# #     original_print(f'[{file_name}:{line_number}]', *args, **kwargs)

# # builtins.print = custom_print


# # Save the original print function
# original_print = builtins.print

# def custom_print(*args, **kwargs):
#     frame = inspect.currentframe().f_back
#     info = inspect.getframeinfo(frame)
#     file_name = info.filename
#     line_number = info.lineno
#     # Call the original print function with the additional file and line info
#     original_print(f'[{file_name}:{line_number}]', *args, **kwargs)

# # Override the built-in print function
# builtins.print = custom_print
