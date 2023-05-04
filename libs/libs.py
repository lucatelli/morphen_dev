"""
                                                          ..___|**_
                                                  .|||||||||*+@+*__*++.
                                              _||||.           .*+;].,#_
                                         _|||*_                _    .@@@#@.
                                   _|||||_               .@##@#| _||_
   Radio Morphen              |****_                   .@.,/\..@_.
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
__version__ = 0.2
__author__  = 'Geferson Lucatelli'
__email__   = 'geferson.lucatelli@postgrad.manchester.ac.uk'
__date__    = '2023 05 05'
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
import casatasks
from casatasks import *
import casatools
# from casatools import *
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pf
from casatools import image as IA

import lmfit
from lmfit import Model
from lmfit import Parameters, fit_report, minimize

import string
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl_
import glob
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import os
from astropy.stats import mad_std
from scipy.ndimage import gaussian_filter
from astropy import visualization
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from skimage.measure import perimeter_crofton

from scipy.optimize import leastsq, fmin, curve_fit
import scipy.ndimage as nd
import scipy
from scipy.stats import circmean, circstd
from scipy.signal import savgol_filter

from astropy.cosmology import FlatLambdaCDM
import numpy as np
from astropy import units as u
import pandas as pd
import sys
import pickle
import time
import corner
import re

# import pymc3 as pm

from petrofit.photometry import make_radius_list
from petrofit.petrosian import Petrosian
from petrofit.photometry import source_photometry
from petrofit.segmentation import make_catalog, plot_segments
from petrofit.segmentation import plot_segment_residual
from petrofit.photometry import order_cat
import copy
# from copy import copy
import astropy.io.fits as fits
import matplotlib.ticker as mticker
from jax import jit
from jax.numpy.fft import fft2, ifft2, fftshift
import jax.numpy as jnp

try:
    import cupy as cp
    import cupyx.scipy.signal


    # import torch
    # import torch.nn.functional as F
except:
    print('GPU Libraries not imported.')
    pass
# sys.path.append('../../scripts/analysis_scripts/')
sys.path.append('../analysis_scripts/')

#redshift for some sources.
z_d = {'VV705': 0.04019,'UGC5101':0.03937,'UGC8696':0.03734, 'VV250':0.03106}



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

    Params:
        x0,y0: center position
        PA: position angle of the meshgrid
        x,y: meshgrid arrays
    """
    # gal_center = (x0+0.01,y0+0.01)
    x0 = x0
    y0 = y0
    # convert to radians
    t = (PA * np.pi) / 180.0
    return ((x - x0) * np.cos(t) + (y - y0) * np.sin(t),
            -(x - x0) * np.sin(t) + (y - y0) * np.cos(t))

def sersic2D(xy, x0, y0, PA, ell, n, In, Rn,cg=0.0):
    q = 1 - ell
    x, y = xy
    # x,y   = np.meshgrid(np.arange((size[1])),np.arange((size[0])))
    xx, yy = rotation(PA, x0, y0, x, y)
    # r     = (abs(xx)**(c+2.0)+((abs(yy))/(q))**(c+2.0))**(1.0/(c+2.0))
    r = np.sqrt((abs(xx) ** (cg+2.0) + ((abs(yy)) / (q)) ** (cg+2.0)))
    model = In * np.exp(-bn(n) * ((r / (Rn)) ** (1.0 / n) - 1.))
    return (model)

def FlatSky(data_level, a):
    return (a * data_level)

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


@jit
def sersic2D_GPU(xy, x0, y0, PA, ell, n, In, Rn, cg=0.0):
    """
    Using Jax >> 10x faster.
    """
    q = 1 - ell
    x, y = xy

    xx, yy = rotation_GPU(PA, x0, y0, x, y)
    # r     = (abs(xx)**(c+2.0)+((abs(yy))/(q))**(c+2.0))**(1.0/(c+2.0))
    r = jnp.sqrt((abs(xx) ** (cg + 2.0) + ((abs(yy)) / (q)) ** (cg + 2.0)))
    model = In * jnp.exp(-bn(n) * ((r / (Rn)) ** (1.0 / n) - 1.))
    return (model)

@jit
def rotation_GPU(PA, x0, y0, x, y):
    """
    Rotate an input image array. It can be used to modify
    the position angle (PA).

    Using Jax >> 10x faster.

    Params:
        x0,y0: center position
        PA: position angle of the meshgrid
        x,y: meshgrid arrays
    """
    # gal_center = (x0+0.01,y0+0.01)
    x0 = x0
    y0 = y0
    # convert to radians
    t = (PA * jnp.pi) / 180.0
    return ((x - x0) * jnp.cos(t) + (y - y0) * jnp.sin(t),
            -(x - x0) * jnp.sin(t) + (y - y0) * jnp.cos(t))


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
    h1 = 0.669  # +/- 0.006 >> Plank Collaboration XLVI 2016
    h2 = 0.732  # +/- 0.017 >> Riess et al. 2016
    # h = 100 * (h1 + h2) / 2
    h = 67.8  # * (h1 + h2) / 2
    cosmo = FlatLambdaCDM(H0=h, Om0=Om0)

    D_L = cosmo.luminosity_distance(z=z).value
    print('D_l = ', D_L)  # 946.9318492873492 Mpc
    return(D_L)

def angular_distance_cosmo(z, Om0=0.308):
    h1 = 0.669  # +/- 0.006 >> Plank Collaboration XLVI 2016
    h2 = 0.732  # +/- 0.017 >> Riess et al. 2016
    h = 67.8# * (h1 + h2) / 2
    cosmo = FlatLambdaCDM(H0=h, Om0=Om0)

    d_A = cosmo.angular_diameter_distance(z=z)
    print('D_a = ', d_A)  # 946.9318492873492 Mpc
    return(d_A)

def arcsec_to_pc(z, cell_size, Om0=0.308):
    h1 = 0.669  # +/- 0.006 >> Plank Collaboration XLVI 2016
    h2 = 0.732  # +/- 0.017 >> Riess et al. 2016
    h = 67.8# * (h1 + h2) / 2
    cosmo = FlatLambdaCDM(H0=h, Om0=Om0)

    d_A = cosmo.angular_diameter_distance(z=z)
    print('D_a = ', d_A)  # 946.9318492873492 Mpc

    theta = 1 * u.arcsec
    distance_pc = (theta * d_A).to(u.pc, u.dimensionless_angles())
    # unit is Mpc only now

    print('Linear Distance = ', distance_pc)  # 3.384745689510495 Mpc
    return (distance_pc)

def pixsize_to_pc(z, cell_size, Om0=0.308):
    h1 = 0.669  # +/- 0.006 >> Plank Collaboration XLVI 2016
    h2 = 0.732  # +/- 0.017 >> Riess et al. 2016
    # h = 100 * (h1 + h2) / 2
    h = 67.8  # * (h1 + h2) / 2
    cosmo = FlatLambdaCDM(H0=h, Om0=Om0)

    d_A = cosmo.angular_diameter_distance(z=z)
    print('D_a = ', d_A)  # 946.9318492873492 Mpc

    theta = cell_size * u.arcsec
    distance_pc = (theta * d_A).to(u.pc, u.dimensionless_angles())  # unit is Mpc only now

    print('Linear Distance = ', distance_pc)  # 3.384745689510495 Mpc
    return (distance_pc.value)



def cosmo_stats(imagename,z,results=None):
    """
    Get beam shape info in physical units.
    """
    if results == None:
        results = {}
        results['#imagename'] = os.path.basename(imagename)
    pc_scale, bmaj, bmin, BA_pc = beam_physical_area(imagename, z=z)
    results['arcsec_to_pc'] = pc_scale.value
    results['bmaj_pc'] = bmaj.value
    results['bmin_pc'] = bmin.value
    results['BA_pc'] = BA_pc.value
    return(results)


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


def ctn(image):
    '''
        ctn > casa to numpy
        FUnction that read fits files inside CASA environment.
        Read a CASA format image file and return as a numpy array.
        Also works with wsclean images!
        Note: For some reason, casa returns a rotated mirroed array, so we need
        to undo it by a rotation.
        '''
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
            data_image = pf.getdata(image)
            return (data_image)
        except:
            print('Error loading fits file')
            return(ValueError)




def normalise_in_log(profile):
    def normalise(x):
        return (x- x.min())/(x.max() - x.min())
    y = profile.copy()
    nu_0 = normalise(np.log(y))
    return nu_0
def shuffle_2D(image):
#     image = ctn(crop_residual)
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
        g = ctn(image)
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

def beam_shape(image):
    '''
    Return the beam shape (bmin,bmaj,pa) given an image.
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




def get_beam_size_px(imagename):
    aO,bO,_,_,_ = beam_shape(imagename)
    cs = get_cell_size(imagename)
    aO_px = aO/cs
    bO_px = bO/cs
    beam_size_px = np.sqrt(aO_px * bO_px)
    return(beam_size_px,aO,bO)

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
    if center == None:
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
    if cutout_filename == None:
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
    # if cutout_filename == None:
    cutout_filename_res = img.replace('-image.fits', '-residual_cutout' +
                                      special_name + '.fits')
    hdu2.writeto(cutout_filename_res, overwrite=True)
    return(cutout_filename,cutout_filename_res)


def do_cutout(image, box_size=300, center=None, return_='data'):
    """
    Fast cutout of a numpy array.

    Returs: numpy data array or a box for that cutout, if asked.
    """

    if center is None:
        try:
            # imhd = imhead(image)
            st = imstat(image)
            print('  >> Center --> ', st['maxpos'])
            xin, xen, yin, yen = st['maxpos'][0] - box_size, st['maxpos'][
                0] + box_size, \
                                 st['maxpos'][1] - box_size, st['maxpos'][
                                     1] + box_size
        except:
            max_x, max_y = np.where(ctn(image) == ctn(image).max())
            xin = max_x[0] - box_size
            xen = max_x[0] + box_size
            yin = max_y[0] - box_size
            yen = max_y[0] + box_size
    else:
        xin, xen, yin, yen = center[0] - box_size, center[0] + box_size, center[1] - box_size, center[1] + box_size
    if return_ == 'data':
        data_cutout = ctn(image)[xin:xen, yin:yen]
        return (data_cutout)
    if return_ == 'box':
        box = xin, xen, yin, yen  # [xin:xen,yin:yen]
        return (box)


def do_cutout_2D(image_data, box_size=300, center=None, return_='data'):
    """
    Fast cutout of a numpy array.

    Returs: numpy data array or a box for that cutout, if asked.
    """

    if center is None:
        x0, y0= nd.maximum_position(image_data)
        print('  >> Center --> ', x0, y0)
        if x0-box_size>1:
            xin, xen, yin, yen = x0 - box_size, x0 + box_size, \
                                 y0 - box_size, y0 + box_size
        else:
            print('Box size is larger than image!')
            return ValueError
    else:
        xin, xen, yin, yen = center[0] - box_size, center[0] + box_size, \
            center[1] - box_size, center[1] + box_size
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
    if file_to_save == None:
        file_to_save = image_to_copy.replace('.fits', 'header.fits')

    from astropy.io import fits
    with fits.open(image) as hdul1:
        with fits.open(image_to_copy, mode='update') as hdul2:
            hdul2[0].header = hdul1[0].header
            hdul2.flush()
    pass


def tcreate_beam_psf(imname, cellsize=None,size=(128,128),app_name='',
                     aspect=None):
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
    if aspect=='equal':
        print('WARNING: Using circular Gaussian for Gaussian beam convolution.')
        minoraxis = str(imhd['restoringbeam']['minor']['value']) + str(
            imhd['restoringbeam']['minor']['unit'])
        majoraxis = str(imhd['restoringbeam']['minor']['value']) + str(
            imhd['restoringbeam']['major']['unit'])

    else:
        minoraxis = str(imhd['restoringbeam']['minor']['value']) + str(
            imhd['restoringbeam']['minor']['unit'])
        majoraxis = str(imhd['restoringbeam']['major']['value']) + str(
            imhd['restoringbeam']['major']['unit'])


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
    if fractions == None:
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

"""


def cvi(imname):
    try:
        os.system('casaviewer ' + imname)
    except:
        try:
            os.system('~/casaviewer ' + imname)
        except:
            pass


def sort_list_by_beam_size(imagelist, residuallist, return_df=False):
    """
    Sort a list of images by beam size.
    """
    beam_sizes_list = []
    for i in tqdm(range(len(imagelist))):
        beam_sizes = {}
        beam_size_px, _, _ = get_beam_size_px(imagelist[i])
        beam_sizes['imagename'] = imagelist[i]
        beam_sizes['residualname'] = residuallist[i]
        beam_sizes['id'] = i
        beam_sizes['B_size_px'] = beam_size_px
        beam_sizes_list.append(beam_sizes)

    df_beam_sizes = pd.DataFrame(beam_sizes_list)
    df_beam_sizes_sorted = df_beam_sizes.sort_values('B_size_px')
    imagelist_sort = np.asarray(df_beam_sizes_sorted['imagename'])
    residuallist_sort = np.asarray(df_beam_sizes_sorted['residualname'])
    i = 0
    for image in imagelist_sort:
        print(i, '>>', os.path.basename(image))
        i = i + 1
    if return_df == True:
        return (imagelist_sort, residuallist_sort, df_beam_sizes_sorted)
    if return_df == False:
        return (imagelist_sort, residuallist_sort)


"""
 ____  _        _
/ ___|| |_ __ _| |_ ___
\___ \| __/ _` | __/ __|
 ___) | || (_| | |_\__ \
|____/ \__\__,_|\__|___/

"""

def rms_estimate(imagedata):
    mean_value = np.mean(imagedata)
    square_difference = (imagedata - mean_value) ** 2
    mean_squared_diff = np.mean(square_difference)
    RMS = np.sqrt(mean_squared_diff)
    return(RMS)
