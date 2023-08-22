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

Using short library file.
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
from tqdm import tqdm

import numpy as np
from scipy import ndimage
from sklearn.neighbors import KNeighborsClassifier


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


def arrea(r):
    return (np.pi * r * r)


def area_to_radii(A):
    return (np.sqrt(A / np.pi))

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

def mask_dilation(image, cell_size=None, sigma=5,rms=None,
                  dilation_size=None,iterations=3, dilation_type='disk',
                  PLOT=True,show_figure=True):


    from scipy import ndimage
    from scipy.ndimage import morphology
    from skimage.morphology import disk, square
    from skimage.morphology import dilation

    if isinstance(image, str) == True:
        data = ctn(image)
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
            if dilation_size == None:
                dilation_size = 7
                # dilation_size = 5

    mask = (data >= sigma * std)
    mask3 = (data >= 3 * std)
    data_mask = mask * data

    if dilation_type == 'disk':
        data_mask_d = ndimage.binary_dilation(mask,
                                            structure=disk(dilation_size),
                                            iterations=iterations).\
                                                astype(mask.dtype)

    if dilation_type == 'square':
        data_mask_d = ndimage.binary_dilation(mask,
                                            structure=square(dilation_size),
                                            iterations=iterations).\
                                                astype(mask.dtype)

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
    #         plt.savefig(image.replace('.fits','_masks.jpg'),dpi=300, bbox_inches='tight')

    if cell_size is not None:
        if isinstance(image, str) == True:
            try:
                print((data * data_mask_d).sum() / beam_area2(image, cell_size))
                print((data * data_mask).sum() / beam_area2(image, cell_size))
                print((data).sum() / beam_area2(image, cell_size))
            except:
                print('Provide a cell size of the image.')
    return (mask, data_mask_d)

def mask_dilation_from_mask(image, mask_init, cell_size=None, sigma=3,rms=None,
                  dilation_size=3,iterations=5, dilation_type='disk',
                  PLOT=True,show_figure=True):
    from scipy import ndimage
    from scipy.ndimage import morphology
    from skimage.morphology import disk, square
    from skimage.morphology import dilation

    if isinstance(image, str) == True:
        data = ctn(image)
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
            if dilation_size == None:
                dilation_size = 7
                # dilation_size = 5

    data_init = data * mask_init
    mask3 = (data >= 3 * std)
    # std = mad_std(data[mask_init])
    mask = (data_init >= sigma * std)
    data_mask = mask * data

    if dilation_type == 'disk':
        data_mask_d = ndimage.binary_dilation(mask,
                                            structure=disk(dilation_size),
                                            iterations=iterations).\
                                                astype(mask.dtype)

    if dilation_type == 'square':
        data_mask_d = ndimage.binary_dilation(mask,
                                            structure=square(dilation_size),
                                            iterations=iterations).\
                                                astype(mask.dtype)

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


import numpy as np

import numpy as np



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
    median_distance = np.median(distances[mask])
    mean_distance = np.mean(distances[mask])
    std_distance = np.std(distances[mask])
    # mad_std_distance = mad_std(distances[mask])

    return median_distance, mean_distance, std_distance

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


def plot_many_IRs(imagelist, labels_profiles=None, save_fig=None):
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

    #     for k in range(len(profiles)):
    #         plt.plot(radiis[k],profiles[k],label=str(labels_profiles[k]))
    #         # if labels_profiles:

    #         plt.xlabel('Radial distance [arcsec]',fontsize=18)
    #         plt.ylabel('PSF intensity',fontsize=18)
    #         plt.xticks(fontsize=18)
    #         plt.yticks(fontsize=18)
    #         plt.grid()
    #         plt.legend(prop={'size': 16})
    #         plt.xlim(0.0,0.2)
    #         if save_fig:
    #             plt.savefig(save_fig)
    # plt.show()
    return (profiles, radiis)

def get_peak_pos(imagename):
    st = imstat(imagename=imagename)
    maxpos = st['maxpos'][0:2]
    print('Peak Pos=', maxpos)
    return (maxpos)


def get_profile(imagename, center=None,binsize=1):
    if isinstance(imagename, str) == True:
        data_2D = ctn(imagename)
    else:
        data_2D = imagename

    if center is None:
        nr, radius, profile = azimuthalAverage(data_2D,return_nr = True,
                                               binsize=binsize)
    else:
        nr, radius, profile = azimuthalAverage(data_2D, return_nr=True,
                                               binsize=binsize,center=center)
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
    nbins = int(np.round(r.max() / binsize) + 1)
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
        radial_prof = np.array([image.flat[mask.flat * (whichbin == b)].std() for b in range(1, nbins + 1)])
    else:
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
        return bin_centers, radial_prof
    elif return_nr:
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


def get_image_statistics(imagename,cell_size,
                         mask_component=None,mask=None,
                         residual_name=None,region='', dic_data=None,
                         sigma_mask=5,apply_mask=True,
                         fracX=0.1, fracY=0.1):
    """
    Get some basic image statistics.



    """
    if dic_data == None:
        dic_data = {}
        dic_data['#imagename'] = os.path.basename(imagename)

    image_data = ctn(imagename)
    if mask is not None:
        image_data = image_data * mask
        apply_mask = False
    #     dic_data['imagename'] = imagename
    if apply_mask == True:
        omask, mask = mask_dilation(imagename, sigma=sigma_mask)
        image_data = image_data*mask
    if mask_component is not None:
        image_data = image_data * mask_component
        if mask is None:
            mask = mask_component
        # mask = mask_component
    if (mask_component is None) and (apply_mask == False) and (mask is None):
        unity_mask = np.ones(ctn(imagename).shape)
        omask, mask = unity_mask, unity_mask
        image_data = ctn(imagename) * mask

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



    if residual_name is not None:
        data_res = ctn(residual_name)
        flux_res_error = 3 * np.sum(data_res * mask) \
                         / beam_area2(imagename, cell_size)
        # rms_res =imstat(residual_name)['flux'][0]
        flux_res = np.sum(ctn(residual_name)) / beam_area2(imagename, cell_size)

        res_error_rms =np.sqrt(
            np.sum((abs(data_res * mask -
                        np.mean(data_res * mask))) ** 2 * np.sum(mask))) / \
                       beam_area2(imagename,cell_size)

        try:
            total_flux_tmp = dic_data['total_flux_mask']
        except:
            total_flux_tmp = flux_im
            total_flux_tmp = total_flux(image_data,imagename,mask=mask)

        print('Estimate #1 of flux error (based on sum of residual map): ')
        print('Flux = ', total_flux_tmp * 1000, '+/-',
              abs(flux_res_error) * 1000, 'mJy')
        print('Fractional error flux = ', flux_res_error / total_flux_tmp)
        print('-----------------------------------------------------------------')
        print('Estimate #2 of flux error (based on rms of '
              'residual x area): ')
        print('Flux = ', total_flux_tmp * 1000, '+/-',
              abs(res_error_rms) * 1000, 'mJy')
        print('Fractional error flux = ', res_error_rms / total_flux_tmp)

        dic_data['max_residual'] = np.max(data_res * mask)
        dic_data['min_residual'] = np.min(data_res * mask)
        dic_data['flux_residual'] = flux_res
        dic_data['flux_error_res'] = abs(flux_res_error)
        dic_data['flux_error_res_2'] = abs(res_error_rms)
        dic_data['mad_std_residual'] = mad_std(data_res)
        dic_data['rms_residual'] = rms_estimate(data_res)

    #     print(' Flux=%.5f Jy/Beam' % flux_im)
    #     print(' Flux peak (image)=%.5f Jy' % flux_peak_im, 'Flux peak (residual)=%.5f Jy' % flux_peak_re)
    #     print(' flux_im/sigma_im=%.5f' % snr_im, 'flux_im/sigma_re=%.5f' % snr)
    #     print(' rms_im=%.5f' % rms_im, 'rms_re=%.5f' % rms_re)
    #     print(' flux_peak_im/rms_im=%.5f' % peak_im_rms, 'flux_peak_re/rms_re=%.5f' % peak_re_rms)
    #     print(' sumsq_im/sumsq_re=%.5f' % q)
    return (dic_data)


def level_statistics(img, cell_size=None, mask_component=None,
                    sigma=6, do_PLOT=True, crop=False,data_2D = None,
                    box_size=256, bkg_to_sub=None, apply_mask=True,
                    mask=None,rms=None,
                    results=None, dilation_size=None, iterations=2,
                    add_save_name='', SAVE=True, show_figure=False, ext='.jpg'):
    """
    Function old name: plot_values_std

    Estimate information for multiple bin levels of the emission.
    It splits the range of image intensity values in four distinct regions:

        1. Inner region: peak intensity -> 0.1 * peak intensity
        2. Mid region: 0.1 * peak intensity -> 10 * rms
        3. Low region: 10 * rms -> 5 * rms
        4. Uncertain region: 5 * rms -> 3 * rms

    """
    if cell_size is None:
        cell_size = get_cell_size(img)
    if data_2D is not None:
        g_ = data_2D
    else:
        g_ = ctn(img)
    g = g_.copy()
    if rms is None:
        std = mad_std(g_)
    else:
        std = rms

    if bkg_to_sub is not None:
        g = g - bkg_to_sub
    if mask_component is not None:
        g = g * mask_component

    print('Mad    >  ', std)
    print('std    >  ', np.std(g_))
    print('median >  ', np.median(g_))
    print('mean   >  ', np.mean(g_))

    ###########################################
    ############## CASA UTILITY  ##############
    ###########################################
    g_hd = imhead(img)
    omaj = g_hd['restoringbeam']['major']['value']
    omin = g_hd['restoringbeam']['minor']['value']
    beam_area_ = beam_area(omaj, omin, cellsize=cell_size)

    if mask is not None:
        g = g * mask
        apply_mask = False  # do not calculate the mask again, in case is True.
        g = g * mask
        g2 = g[mask]
    if apply_mask == True:
        _, mask_dilated = mask_dilation(img, cell_size=cell_size, sigma=sigma,
                                        dilation_size=dilation_size,rms=rms,
                                        iterations=iterations,
                                        PLOT=False)
        g = g * mask_dilated
        if mask is not None:
            g2 = g2[mask_dilated]
        else:
            g2 = g[mask]

    # if (mask_component is None) and (apply_mask == False) and (mask is None):
    #     g = g_
    #     g2 = g_

    # std2 = mad_std(g2)

    dl = 1e-6
    if mask_component is not None:
        levels = np.geomspace(g.max(), (1.0 * std + dl), 5)
        levels_top = np.geomspace(g.max(), g.max() * 0.1, 3)
        try:
            levels_mid = np.geomspace(g.max() * 0.1, (10 * std + dl), 5)
        except:
            levels_mid = np.asarray([0])
        try:
            levels_low = np.geomspace(10 * std, (5.0  * std + dl), 2)
            levels_uncertain = np.geomspace(5.0 * std, (3.0 * std + dl), 3)
        except:
            levels_low = np.asarray([0])
            levels_uncertain = np.asarray([0])

    else:
        if apply_mask is not False:
            # print('asdasd', g.max(), std)
            levels = np.geomspace(g.max(), (1 * std + dl), 5)
            levels_top = np.geomspace(g.max(), g.max() * 0.1, 3)
            try:
                levels_mid = np.geomspace(g.max() * 0.1, (10 * std + dl), 5)
            except:
                levels_mid = np.asarray([0])
            try:
                levels_low = np.geomspace(10 * std, (5.0 * std + dl), 2)
                levels_uncertain = np.geomspace(3 * std, (1.0 * std + dl), 3)
            except:
                levels_low = np.asarray([0])
                levels_uncertain = np.asarray([0])
        else:
            levels = np.geomspace(g.max(), (3 * std + dl), 5)
            levels_top = np.geomspace(g.max(), g.max() * 0.1, 3)
            # levels_mid = np.geomspace(g.max() * 0.1, (10 * std + dl), 5)
            try:
                levels_mid = np.geomspace(g.max() * 0.1, (10 * std + dl), 5)
            except:
                levels_mid = np.asarray([0])
            try:
                levels_low = np.geomspace(10 * std, (5.0 * std + dl), 2)
                levels_uncertain = np.geomspace(3 * std, (1.0 * std + dl), 3)
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
    if results == None:
        results = {}
        results['#imagename'] = os.path.basename(img)

    print('Low Flux (extended) Jy                    > ', low_flux, ' >> ratio=',
          low_flux / total_flux)
    print('Mid Flux (outer core + inner extended) Jy > ', mid_flux, ' >> ratio=',
          mid_flux / total_flux)
    print('Inner Flux (core) Jy                      > ', inner_flux,
          ' >> ratio=', inner_flux / total_flux)
    print('Uncertain Flux (<5std)                    > ', uncertain_flux,
          ' >> ratio=', uncertain_flux / total_flux)
    print('Total Flux Jy                             > ', total_flux)
    print('Total area (in # ob beams)                > ', number_of_beams)
    print('Total inner area (in # ob beams)          > ', n_beams_inner)
    print('Total mid area (in # ob beams)            > ', n_beams_mid)
    print('Total low area (in # ob beams)            > ', n_beams_low)
    print('Total uncertain area (in # ob beams)      > ', n_beams_uncertain)
    print('Inner Flux (core) fraction                > ',
          inner_flux / total_flux)
    print('Outer Flux (ext)  fraction                > ', ext_flux / total_flux)

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
    if results == None:
        results = {}
        results['#imagename'] = os.path.basename(imagename)

    if isinstance(imagename, str) == True:
        image_data = ctn(imagename)
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
        data_2D = ctn(imagename)

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

    results_final['error_petro'] = error_petro
    if mask_component is not None:
        r, ir = get_profile(ctn(imagename) * mask_component)
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


def measures(imagename, residualname, z, mask_component=None, sigma_mask=6,
             last_level=3.0, vmin_factor=1.0, plot_catalog=False,data_2D=None,
             npixels=128, fwhm=81, kernel_size=21, dilation_size=None,
             main_feature_index=0, results_final={}, iterations=2,
             fracX=0.10, fracY=0.10, deblend=False, bkg_sub=False,
             bkg_to_sub=None, rms=None,do_petro=True,
             apply_mask=True, do_PLOT=False, SAVE=True, show_figure=True,
             mask=None,do_measurements='all',compute_A=False,
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
        print('     >> INFO: Using provided mask.')

    if apply_mask == True:
        print('     >> CALC: Performing mask dilation.')
        _, mask_dilated = mask_dilation(imagename, cell_size=cell_size,
                                        sigma=sigma_mask,
                                        dilation_size=dilation_size,
                                        iterations=iterations, rms=rms,
                                        PLOT=True)
        mask = mask_dilated
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
        data_2D = ctn(imagename)

    results_final = level_statistics(img=imagename, cell_size=cell_size,
                                    mask_component=mask_component,
                                    mask=mask, apply_mask=False,
                                    data_2D=data_2D,
                                    sigma=sigma_mask, do_PLOT=do_PLOT,
                                    results=results_final, bkg_to_sub=bkg_to_sub,
                                    show_figure=show_figure,
                                    add_save_name=add_save_name,
                                    SAVE=SAVE, ext='.jpg')
    levels, fluxes, agrow, plt, \
        omask2, mask2, results_final = make_flux_vs_std(imagename,
                                                        cell_size=cell_size,
                                                        residual=residualname,
                                                        mask_component=mask_component,
                                                        last_level=last_level,
                                                        sigma_mask=sigma_mask,
                                                        apply_mask=False,
                                                        mask=mask,
                                                        rms=rms,
                                                        data_2D=data_2D,
                                                        vmin_factor=vmin_factor,
                                                        results=results_final,
                                                        show_figure=show_figure,
                                                        bkg_to_sub=bkg_to_sub,
                                                        add_save_name=add_save_name,
                                                        SAVE=SAVE)
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
    error_petro = False
    if do_petro == True:
        try:
            r_list, area_arr, area_beam, p, flux_arr, error_arr, results_final, cat, \
                segm, segm_deblend, sorted_idx_list = \
                compute_petrosian_properties(data_2D, imagename,
                                             mask_component=mask_component,
                                             global_mask=mask,
                                             source_props=results_final,
                                             apply_mask=False,
                                             sigma_level=sigma_mask,
                                             bkg_sub=bkg_sub, bkg_to_sub=bkg_to_sub,
                                             vmin=vmin_factor, plot=do_PLOT,
                                             deblend=deblend,
                                             fwhm=fwhm, kernel_size=kernel_size,
                                             show_figure=show_figure,
                                             add_save_name=add_save_name,
                                             npixels=npixels)
        except:
            error_petro = True
    else:
        error_petro = True

    results_final['error_petro'] = error_petro
    if mask_component is not None:
        r, ir = get_profile(ctn(imagename) * mask_component)
    else:
        r, ir = get_profile(imagename)
    #     rpix = r / cell_size
    rpix = r.copy()

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
        x0c, y0c = results_final['x0'], results_final['y0']
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

    return (results_final, mask)


def make_flux_vs_std(img, cell_size, residual, mask_component=None,
                     aspect=1, last_level=3.0,mask=None,data_2D=None,
                     dilation_size=2,iterations=None, dilation_type='disk',
                     sigma_mask=5, rms=None, results=None, bkg_to_sub=None,
                     apply_mask=True, vmin_factor=3, vmax_factor=0.5,
                     crop=False, box_size=256,
                     SAVE=True,add_save_name='',show_figure=True, ext='.jpg'):
    """
    Params
    ------

    mask_component: 2D np array
        for a multi component source, this is the mask for a specific component.

    """
    if results == None:
        results = {}
        results['#imagename'] = os.path.basename(img)

    from skimage.draw import disk
    if data_2D is not None:
        g_ = data_2D
    else:
        g_ = ctn(img)
    # res_ = ctn(residual)
    g = g_.copy()
    # res = res_.copy()

    if bkg_to_sub is not None:
        g = g - bkg_to_sub
    ###########################################
    ############## CASA UTILITY  ##############
    ###########################################
    ####################################################################
    ############## CASA COMPONENT  #####################################
    ####################################################################
    g_hd = imhead(img)                                      ############
    freq = g_hd['refval'][2] / 1e9                          ############
    print(freq)                                             ############
    omaj = g_hd['restoringbeam']['major']['value']          ############
    omin = g_hd['restoringbeam']['minor']['value']          ############
    beam_area_ = beam_area(omaj, omin, cellsize=cell_size)  ############
    ##############                 #####################################
    ####################################################################
    if rms is not None:
        print('Using rms provided...')
        std = rms
    else:
        std = mad_std(g_)
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
        omask, mask = mask_dilation(img, sigma=sigma_mask,
                                    dilation_size=dilation_size,
                                    iterations=iterations,
                                    dilation_type=dilation_type,
                                    show_figure=show_figure)
        # eimshow(mask)
        g = g * mask  # *(g_>3*mad_std(g_)) + 0.1*mad_std(g_)
        # res = res * mask
        # total_flux = np.sum(g_ * (g_ > 3 * std)) / beam_area_
        total_flux = np.sum(g) / beam_area_
        total_flux_nomask = np.sum(g_) / beam_area_
        # g = g_

    if mask_component is not None:
        #
        g = g * mask_component
        # res = res * mask_component

        total_flux_nomask = np.sum(g) / beam_area_
        total_flux = np.sum(g * (g > 3 * std)) / beam_area_
        mask = mask_component
        omask = mask_component


    # if (apply_mask is None) and  (mask is None):
    #     mask = mask_component
        # g = g_
    if (mask_component is None) and (apply_mask ==False) and (mask is None):
        # g = g_
        total_flux = np.sum(g * (g > 3 * std)) / beam_area_
        total_flux_nomask = np.sum(g) / beam_area_

        unity_mask = np.ones(g.shape) == 1
        omask, mask = unity_mask, unity_mask

    # total_flux_nomask = np.sum(g_) / beam_area_
    # if (mask is None) and (mask_component is not None):
    #     mask = mask_component


    bins = 128  # resolution of levels
    results['total_flux_nomask'] = total_flux_nomask
    results['total_flux_mask'] = total_flux
    # flux_mc, flux_error_m = mc_flux_error(img, g,
    #               res,
    #               num_threads=6, n_samples=1000)
    #
    # results['flux_mc'] = flux_mc
    # results['flux_error_mc'] = flux_error_m


    try:
        # this should not be linspace, should be spaced in a logarithmic sense!!
        levels = np.geomspace(g.max(), last_level * std,
                              bins)
        levels2 = np.geomspace(g.max(), last_level * std,
                               100)
    except:
        levels = np.geomspace(g.max(), last_level * np.std(g),
                              bins)
        levels2 = np.geomspace(g.max(), last_level * np.std(g),
                               100)

    fluxes = []
    areas = []
    for i in range(len(levels)):
        if i == 0:
            # condition = (g >= levels[i])
            # flux = g[np.where(condition)].sum() / beam_area_
            condition = (g >= levels[i])
            flux = (g * (condition)).sum() / beam_area_

            area = np.sum(condition)
            fluxes.append(flux)
            areas.append(area)
        else:
            # condition = (g < levels[i - 1]) & (g >= levels[i])
            # flux = g[np.where(condition)].sum() / beam_area_
            condition = ((g < levels[i - 1]) & (g >= levels[i]))
            flux = (g * (condition)).sum() / beam_area_
            area = np.sum(condition)
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
    agrow = areas
    results['total_flux_levels'] = np.sum(fluxes)


    agrow_beam = agrow / beam_area_
    Lgrow = np.cumsum(fluxes)
    Lgrow_norm = Lgrow / fluxes.sum()
    # print(Lgrow_norm)
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

    try:
        sigma_20 = levels[mask_L20_idx[-1]]
    except:
        sigma_20 = levels[mask_L50_idx[-1]]
    sigma_50 = levels[mask_L50_idx[-1]]
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
    print('Inner Perimeter (%50):', (inner_perimeter))
    print('Outer Perimeter (%90):', (outer_perimeter90))
    print('Outer Perimeter (%99):', (outer_perimeter))


    # Geometry
    x0max, y0max = peak_center(g * mask)
    results['x0'], results['y0'] = x0max, y0max
    # determine momentum centres.
    x0m, y0m, _, _ = momenta(g * mask, PArad_0=None, q_0=None)
    results['x0m'], results['y0m'] = x0m, y0m

    # some geometrical measures
    # calculate PA and axis-ratio
    region_split = [i for i, x in enumerate(levels > sigma_50) if x][-1]
    PA, q, x0col, y0col, PAm, qm, \
        PAmi, qmi, PAmo, qmo, \
        x0median, y0median, \
        x0median_i, y0median_i, \
        x0median_o, y0median_o = cal_PA_q(g * mask, Isequence=levels,
                                          region_split=region_split,
                                          SAVENAME=img.replace('.fits','_ellipsefit') + ext)

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
    L50_norm = Lgrow_norm[mask_L50_idx[-1]]  # ~ 0.5
    L50 = Lgrow[mask_L50_idx[-1]]
    L80_norm = Lgrow_norm[mask_L80_idx[-1]]  # ~ 0.8
    L80 = Lgrow[mask_L80_idx[-1]]
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

    A20, C20radii, npix20 = estimate_area((g > sigma_20) * mask, cell_size, omaj,
                                          omin)
    A50, C50radii, npix50 = estimate_area((g > sigma_50) * mask, cell_size, omaj,
                                          omin)
    A80, C80radii, npix80 = estimate_area((g > sigma_80) * mask, cell_size, omaj,
                                          omin)
    A90, C90radii, npix90 = estimate_area((g > sigma_90) * mask, cell_size, omaj,
                                          omin)
    A95, C95radii, npix95 = estimate_area((g > sigma_95) * mask, cell_size, omaj,
                                          omin)

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

    print(C20radii, C50radii, C80radii, C90radii)
    C1 = np.log10(C80radii / C20radii)
    C2 = np.log10(C90radii / C50radii)

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

    results['max'] = g.max()
    results['std_image'] = std
    try:
        results['std_residual'] = mad_std(ctn(residual))
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
    results['AC1'] = AC1
    results['AC2'] = AC2
    results['CAC1'] = CAC1
    results['CAC2'] = CAC2

    results['L20'] = L20
    results['A20'] = A20
    results['C20'] = sigma_20 / std
    results['C20radii'] = C20radii
    results['npix20'] = npix20

    results['L50'] = L50
    results['A50'] = A50
    results['C50'] = sigma_50 / std
    results['C50radii'] = C50radii
    results['npix50'] = npix50

    results['L80'] = L80
    results['A80'] = A80
    results['C80'] = sigma_80 / std
    results['C80radii'] = C80radii
    results['npix80'] = npix80

    results['L90'] = L90
    results['A90'] = A90
    results['C90'] = sigma_90 / std
    results['C90radii'] = C90radii
    results['npix90'] = npix90
    results['flag9095'] = flag9095
    results['flagL9095'] = flagL9095


    results['L95'] = L95
    results['A95'] = A95
    results['C95'] = sigma_95 / std
    results['C95radii'] = C95radii
    results['npix95'] = npix95


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

    #     results['nbeams50'] = nbeams50
    #     results['nbeams50_100'] = nbeams50_100
    #     results['nbeams50_full'] = nbeams50_full

    print('R50/R50_100 >> ', area_ratio)
    print('R50 >> ', C50radii)
    print('R50_100 >> ', C50_100radii)
    print('Gaussianity >> ', gaussianity)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    # total_flux = 0.013
    # ax1.scatter(levels[:],np.cumsum(fluxes)/total_flux)
    ax1.scatter(levels[:], np.cumsum(fluxes) / np.sum(fluxes),
                label='Levels Norm Flux')
    ax1.scatter(levels[:], np.cumsum(fluxes) / results['total_flux_mask'],
                label='Mask Norm Flux')
    print('Sum of fluxes = ', np.sum(fluxes))

    #     print(np.sum(fluxes))
    ax1.axhline(0, ls='-.', color='black')
    # ax1.axvline(g.max()*0.5,label=r'$0.5\times \max$',color='purple')
    # ax1.axvline(g.max()*0.1,label=r'$0.1\times \max$',color='#E69F00')
    ax1.axvline(sigma_50,
                label=r'Half-Flux $\sim$ {:0.2f}"'.format(C50radii*cell_size),
                ls='-.', color='lime')
    ax1.axhline(L50_norm, ls='-.', color='lime')
    ax1.axvline(sigma_95,
                label=r'95$\%$-Flux $\sim$ {:0.2f}"'.format(C95radii*cell_size),
                ls='dashdot', color='#56B4E9')
    ax1.axvline(std * 6, label=r'6.0$\times$ std', color='black')
    if last_level<3:
        ax1.axvline(std * 3, label=r'3.0$\times$ std', color='brown')

    ax1.set_title('Total Flux '
                  '($\sigma$ levels) = {:0.2f} mJy'.format(1000*np.sum(fluxes)))
    #     ax1.axvline(mad_std(g)*1,label=r'$1.0\times$ std',color='gray')
    ax1.axvline(levels[-1], label=r'' + str(last_level) + '$\\times$ std',
                color='cyan')
    #     plt.legend()
    ax1.set_xlabel('Levels [Jy/Beam]')
    # plt.ylabel('Integrated Flux per level [Jy]')
    ax1.set_ylabel('Fraction of Integrated Flux per level')
    ax1.semilogx()
    ax1.grid(alpha=0.5)
    # plt.xlim(1e-6,)
    ax1.legend(loc='lower left')

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

    im_plot = ax2.imshow((g), cmap='magma_r', origin='lower', alpha=1.0,
                         norm=norm,
                         aspect=aspect)  # ,vmax=vmax, vmin=vmin)#norm=norm

    # levels_50 = np.asarray([-3*sigma_50,sigma_50,3*sigma_50])

    try:
        ax2.contour(g, levels=levels_50, colors='lime', linewidths=2.5,
                    alpha=1.0)  # cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=levels_90, colors='white', linewidths=2.0,
                    linestyles='dashdot',
                    alpha=1.0)  # cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=levels_95, colors='#56B4E9', linewidths=2.0,
                    linestyles='dashdot',
                    alpha=1.0)  # cmap='Reds', linewidths=0.75)
        #         ax2.contour(g, levels=levels_3sigma,colors='#D55E00',linewidths=1.5,alpha=1.0)#cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=[last_level * std], colors='cyan', linewidths=0.6,
                    alpha=1.0,
                    linestyles='dashed')  # cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=[6.0 * std], colors='black', linewidths=1.5,
                    alpha=0.9,
                    linestyles='dashed')  # cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=[4.0 * std], colors='brown', linewidths=1.2,
                    alpha=0.9,
                    linestyles='dashed')  # cmap='Reds', linewidths=0.75)
        ax2.contour(g, levels=[5.0 * std], colors='brown', linewidths=0.6,
                    alpha=0.3,
                    linestyles='dashed')  # cmap='Reds', linewidths=0.75)

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
    return (levels, fluxes, agrow, plt, omask, mask, results)


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
    x0, y0 = pos
    A1img = np.abs(img - rot180(img, x0, y0)) / (np.sum(np.abs(img)))
    if use_mask==True:
        return np.sum(mask * A1img)
    else:
        AsySigma = 3.00
        A1mask = A1img > np.median(A1img) + AsySigma * mad_std(A1img)
        return np.sum(mask * A1mask * A1img)


def geo_mom(p, q, I, centered=True, normed=True, complex=False, verbose=False):
    """return the central moment M_{p,q} of image I
    http://en.wikipedia.org/wiki/Image_moment
    F.Ferrari 2012, prior to 4th JPAS
    """

    M, N = I.shape
    x, y = np.meshgrid(np.arange(N), np.arange(M))

    M_00 = I.sum()

    if centered:
        # centroids
        x_c = (1 / M_00) * np.sum(x * I)
        y_c = (1 / M_00) * np.sum(y * I)

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

    M_pq = (1 / NORM) * np.sum(XX ** p * YY ** q * I)

    return M_pq


def q_PA(image):
    """
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
    lam1 = np.sqrt(abs((1 / 2.) * (mu20 + mu02 + np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2))) / m00)
    lam2 = np.sqrt(abs((1 / 2.) * (mu20 + mu02 - np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2))) / m00)
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
    profiles of elliptical and spiral galaxies, which may not change to much for the
    former while it may does for the later.
    The script that helps in this task is called mfmtk_isophote.py
    """

    if PArad_0 == None:
        PArad = PA  # + np.pi/2
    else:
        PArad = PArad_0

    PAdeg = np.rad2deg(PArad)

    if q_0 == None:
        q = b / a
    else:
        q = q_0
    return (x0col, y0col, q, PAdeg)


def cal_PA_q(gal_image_0,Isequence = None,region_split=None,SAVENAME=None):
    '''
    Estimates inner and outer PA nad q=(b/a)
    '''
    from fitEllipse2018 import main_test2
    # mean Inner q,  mean outer q,  mean Inner PA,  mean Outer PA
    qmi, qmo, PAmi, PAmo, qm, PAm,\
        x0median,y0median,x0median_i,y0median_i,\
        x0median_o,y0median_o = main_test2(gal_image_0,
                                           Isequence = Isequence,region_split=region_split,
                                           SAVENAME=SAVENAME)

    # global PA,  global q
    PA, q, x0col, y0col = q_PA(gal_image_0)

    print("Initial PA and q = ", PA, q)
    print("Median PA and q = ", PAm, qm)
    print("Inner-Mean PA and q = ", PAmi, qmi)
    print("Outer-Mean PA and q = ", PAmo, qmo)
    return (PA, q, x0col, y0col, PAm, qm, PAmi, qmi, PAmo, qmo,
            x0median,y0median,x0median_i,y0median_i,x0median_o,y0median_o)



def savitzky_golay_2d(z, window_size, order, derivative=None):
    """
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
    if derivative == None:
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
    """ make a standard galaxy, id est, PA=0, q=1
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
    """Reprojects a 2D numpy array ("image") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image.
    http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
    refactored by FF, 2013-2014 (see transpolar.py)
    """

    if origin == None:
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
    Gradient Index
    Calculates an index based on the image gradient magnitude and orientation

    SGwindow and SGorder are Savitsky-Golay filter parameters
    F. Ferrari, 2014
    """

    def sigma_func(params):
        ''' calculates the sigma psi with different parameters
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

        if Rp == None:
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
        ''' calculates the sigma psi with different parameters
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
        if Rp == None:
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

    #     image_data=ctn(imagelist[1])

    PA, q, x0col, y0col, PAm, qm, PAmi, qmi, PAmo, qmo = cal_PA_q(image_data)

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


import numpy as np
from sympy import *


def D_Cfit(z):
    #     a = 1.0/(1+z)
    h1 = 0.669  # +/- 0.006 >> Plank Collaboration XLVI 2016
    h2 = 0.732  # +/- 0.017 >> Riess et al. 2016
    #     h = (h1 + h2)/2
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
    D_L = luminosity_distance_cosmo(z=z)
    D_Lnu = D_L * (1 + z) ** (-(alpha + 1) / 2) * dist_conversion_factor

    # flux is in Jy
    L_nu = 4.0 * np.pi * (D_Lnu ** 2.0) * flux * lum_conversion_factor
    # Ld = 4.0 * np.pi * D_L**2.0 * flux
    L_nu_error = 0.0
    return (L_nu, L_nu_error)


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
        L_NT = Lumform / (1 + 1 / 13. * ((1 + z) * nu) ** (-0.1 - al))
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
    Hubble = 70  # km/s/Mpc
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


def compute_SFR_NT(flux, frequency, z, alpha, alpha_NT=-0.85, flux_error=None,
                   calibration_kind='Murphy12', return_with_error=False):
    '''
        To do:
            [ ] - Implement error estimates
            [ ] - Check the thermal contribution
            [ ] - Use spectral index from the image
            [ ] - Explore different Te's (electron temperature)
    '''

    if calibration_kind == 'Murphy11':
        Lnu_NT, Lnu_NT_error = compute_Lnu(flux, z,
                                           alpha)  # 0.0014270422727500343
        SFR = 6.64 * (1e-29) * ((frequency) ** (-alpha_NT)) * Lnu_NT

    if calibration_kind == 'Tabatabaei2017':
        '''
        There is something wrong for this kind!
        '''
        Lnu_NT, Lnu_NT_error = compute_Lnu(flux, z,
                                           alpha)  # 0.0014270422727500343
        SFR = 1.11 * 1e-37 * 1e9 * frequency * Lnu_NT
        if flux_error is not None:
            SFR_error = 1.11 * 1e-37 * 1e9 * frequency * Lnu_NT_error
        else:
            SFR_error = 0.0

    if calibration_kind == 'Murphy12':
        Te = 1e4
        Lnu_NT, Lnu_NT_error = compute_Lnu(flux, z, alpha)
        SFR = 1e-27 * ( \
                    (2.18 * ((Te / (1e4)) ** 0.45) * (frequency ** (-0.1)) + \
                     15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
            ) * Lnu_NT

        if flux_error is not None:
            Lnu_NT_error, Lnu_NT_error2 = compute_Lnu(flux_error, z, alpha)
            SFR_error = 1e-27 * ( \
                        (2.18 * ((Te / (1e4)) ** 0.45) * (frequency ** (-0.1)) + \
                         15.1 * (frequency ** (alpha_NT))) ** (-1.00) \
                ) * Lnu_NT_error
        else:
            SFR_error = 0.0
    print('SFR =', SFR, '+/-', SFR_error, 'Mo/yr')
    if return_with_error:
        return (SFR, SFR_error)
    else:
        return (SFR)

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
 

"""


def do_petrofit(image, cell_size, mask_component=None, fwhm=8, kernel_size=5, npixels=32,
                main_feature_index=0, sigma_mask=7, dilation_size=10,
                apply_mask=True, PLOT=True, show_figure = True, results=None):
    from petrofit.photometry import order_cat
    from petrofit.photometry import make_radius_list

    from petrofit.photometry import source_photometry
    from petrofit.segmentation import make_catalog, plot_segments
    from petrofit.segmentation import plot_segment_residual
    from petrofit.photometry import order_cat

    if results == None:
        results = {}
        results['#imagename'] = os.path.basename(image)

    # sigma = fwhm * gaussian_fwhm_to_sigma
    # kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size)
    data_2D_ = ctn(image)
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
        deblend=False,
        kernel_size=kernel_size,
        fwhm=fwhm,
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
        bkg_sub=False,  # Subtract background
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
    p_copy.eta = 0.15
    p_copy.epsilon = 2

    print('eta =', p_copy.eta)
    print('epsilon =', p_copy.epsilon)
    print('r_half_light (old vs new) = {:0.2f} vs {:0.2f}'.format(p.r_half_light, p_copy.r_half_light))
    print('r_total_flux (old vs new) = {:0.2f} vs {:0.2f}'.format(p.r_total_flux, p_copy.r_total_flux))



    print('R50 = ', p_copy.r_half_light)
    print('Rp=', p_copy.r_petrosian)

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
                 i='1', petro_properties={},sigma_type='clip',eta_value=0.15,
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
                                                      bkg_sub=bkg_sub, sigma=sigma,
                                                      sigma_type=sigma_type,
                                                      plot=plot, vmax=0.3 * data_2D.max(),
                                                      vmin=vmin * mad_std(data_2D)
                                                      )
    #     fast_plot2(mask_source * data_2D)
    p = Petrosian(r_list, area_arr, flux_arr)
    from copy import copy
    p_015 = copy(p)
    if eta_value is None:
        p_015.eta = 0.15
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
                                 fwhm=121, kernel_size=81, npixels=None,
                                 add_save_name=''):
    # if mask:
    if source_props == None:
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
                                               kernel_size=kernel_size,
                                               fwhm=fwhm,
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
                                                   kernel_size=kernel_size,
                                                   fwhm=fwhm,
                                                   npixels=npixels,
                                                   # because we already deblended it!
                                                   plot=plot_catalog,
                                                   vmax=vmax*data_component.max(),
                                                   vmin=vmin * std)
        except:
            cat, segm, segm_deblend = make_catalog(image=data_component,
                                                   threshold=0.01 * std,
                                                   deblend=deblend,
                                                   kernel_size=kernel_size,
                                                   fwhm=fwhm,
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
            (p.r_total_flux is np.nan):
        print('WARNING: Number of pixels for petro region is to small. Looping '
              'over until good condition is satisfied.')
        print('Rlast     >> ', source_props['rlast'])
        print('Rp        >> ', source_props['Rp'])
        print('Rtotal    >> ', p.r_total_flux)

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
                Rlast_new = 2 * source_props['Rp'] + 10

        source_props, flux_arr, area_arr, error_arr, p, r_list = \
            petrosian_metrics(source=source,data_2D=data_component,
                              segm=segm_mask,global_mask=global_mask,
                              mask_source=mask_component,eta_value=0.20,
                              i=ii,petro_properties=source_props,
                              rlast=Rlast_new,sigma=sigma_level,
                              vmin=vmin,bkg_sub=bkg_sub,error=error,
                              plot=plot)

    if (source_props['rlast'] < 2 * source_props['Rp']) or \
            (p.r_total_flux is np.nan):
        print('WARNING: Number of pixels for petro region is to small. Looping '
              'over until good condition is satisfied.')
        print('Rlast     >> ', source_props['rlast'])
        print('Rp        >> ', source_props['Rp'])
        print('Rtotal    >> ', p.r_total_flux)
        Rlast_new = 2 * source_props['Rp'] + 3
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

    print('eta =', p.eta)
    print('epsilon =', p.epsilon)
    print('r_half_light (old vs new) = {:0.2f}'.
          format(p.r_half_light))
    print('r_total_flux (old vs new) = {:0.2f}'.
          format(p.r_total_flux))


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

    print('R50 = ', p.r_half_light)
    print('Rp=', p.r_petrosian)

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

    # if (global_mask is not None) and (imagename is not None):
    #     data_comp_mask = data_component * global_mask
    #     total_comp_flux = np.sum(data_comp_mask) / beam_area_
    #     source_props['c' + ii + '_total_flux'] = total_comp_flux
    return(r_list, area_arr, area_beam, p, flux_arr, error_arr, source_props,
           cat, sorted_idx_list, segm, segm_deblend)

def compute_petro_source(data_2D, mask_component=None, global_mask=None,
                         imagename=None, i=0, source_props={},
                         sigma_level=3, bkg_sub=False,
                         vmin=1, plot=False, deblend=False, ):
    # if mask:
    if imagename is not None:
        beam_area_ = beam_area2(imagename, cellsize=None)

    ii = str(i + 1)
    std = mad_std(data_2D)
    if mask_component is not None:
        data_component = data_2D * mask_component
    else:
        mask_component = np.ones(data_2D.shape)
        data_component = data_2D
    cat, segm, segm_deblend = make_catalog(image=data_component,
                                           threshold=sigma_level * std,
                                           deblend=deblend,
                                           # because we already deblended it!
                                           plot=plot, vmax=data_component.max(),
                                           vmin=vmin * std)

    sorted_idx_list = order_cat(cat, key='area', reverse=True)
    source = cat[sorted_idx_list[0]]

    source_props['c' + ii + '_PA'] = source.orientation.value
    source_props['c' + ii + '_q'] = 1 - source.ellipticity.value
    source_props['c' + ii + '_area'] = source.area.value
    source_props['c' + ii + '_Re'] = source.equivalent_radius.value
    source_props['c' + ii + '_x0c'] = source.xcentroid
    source_props['c' + ii + '_y0c'] = source.ycentroid
    source_props['c' + ii + '_label'] = source.label

    # help function to be used if iteration required.
    source_props, p = petro_params(source=source, data_2D=data_component, segm=segm,
                                mask_source=mask_component,
                                i=ii, petro_properties=source_props,
                                rlast=None, sigma=sigma_level,
                                vmin=vmin, bkg_sub=bkg_sub,
                                plot=plot)

    """
    Check if Rp is larger than last element (rlast) of the R_list. If yes,
    we need to run petro_params again, with a larger rlast, at least r_last>=Rp.
    If not, R50 will be np.nan as well Snu.
    """
    if (source_props['c' + ii + '_rlast'] < 2 * source_props['c' + ii + '_Rp']) \
            or (p.r_total_flux is np.nan):
        print('WARNING: Number of pixels for petro region is to small. '
              'Looping over until good condition is satisfied.')
        # Rlast_new = 2 * source_props['c' + ii + '_Rp'] + 3

        _, area_convex_mask = convex_shape(mask_component)
        # rlast = int(2 * np.sqrt((np.sum(mask_source) / np.pi)))
        Rlast_new = int(3.0 * area_to_radii(area_convex_mask))
        source_props, p = petro_params(source=source, data_2D=data_component,
                                    segm=segm, mask_source=mask_component,
                                    i=ii, petro_properties=source_props,
                                    rlast=Rlast_new, sigma=sigma_level,
                                    vmin=vmin,
                                    bkg_sub=bkg_sub, plot=plot)

        if (source_props['c' + ii + '_rlast'] < 2 * source_props['c' + ii + '_Rp']) \
                or (p.r_total_flux is np.nan):
            print('WARNING: Number of pixels for petro region is to small. '
                  'Looping over until good condition is satisfied.')
            # Rlast_new = 2 * source_props['Rp'] + 3
            Rlast_new = 2 * source_props['c' + ii + '_Rp'] + 3
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

def petro_cat(data_2D, fwhm=24, npixels=None, kernel_size=15,
              nlevels=30, contrast=0.001,bkg_sub=False,
              sigma_level=20, vmin=5,
              deblend=True, plot=False):
    """
    Use PetroFit class to create catalogues.
    """
    cat, segm, segm_deblend = make_catalog(
        image=data_2D,
        threshold=sigma_level * mad_std(data_2D),
        kernel_size=kernel_size, fwhm=fwhm, nlevels=nlevels,
        deblend=deblend,
        npixels=npixels,contrast=contrast,
        plot=plot, vmax=data_2D.max(), vmin=vmin * mad_std(data_2D)
    )

    sorted_idx_list = order_cat(cat, key='area', reverse=True)
    #     idx = sorted_idx_list[main_feature_index]  # index 0 is largest
    #     source = cat[idx]  # get source from the catalog
    return (cat, segm, sorted_idx_list)


def petro_params(source, data_2D, segm, mask_source,
                 i='1', petro_properties={},sigma_type='clip',eta_value=None,
                 rlast=None, sigma=3, vmin=3, bkg_sub=True, plot=False):
    if rlast is None:
        rlast = int(2 * np.sqrt((np.sum(mask_source) / np.pi)))
    else:
        rlast = rlast

    r_list = make_radius_list(max_pix=rlast,  # Max pixel to go up to
                              n=int(rlast)  # the number of radii to produce
                             )
    cutout_size = 2 * max(r_list)
    flux_arr, area_arr, error_arr = source_photometry(source, data_2D, segm,
                                                      r_list, cutout_size=cutout_size,
                                                      bkg_sub=bkg_sub, sigma=sigma,
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
        #     print('    R50 =', R50)
        #     print('     Rp =', Rp)


    if eta_value is not None:
        from copy import copy
        p_new = copy(p)
        p_new.eta = 0.25
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
                 fwhm=24, npixels=None, kernel_size=15, nlevels=30,
                 contrast=0.001,sigma_level=20, vmin=5,bkg_sub=False,
                 deblend=True,PLOT=False,apply_mask=False):
    '''
    From a 2D image array, perform simple source extraction, and calculate basic petrosian
    properties.
    '''
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
                or (p.r_total_flux is np.nan):
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
                or (p.r_total_flux is np.nan):
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


def sep_background(imagename,mask=None,apply_mask=False,
                   bw=64, bh=64, fw=5, fh=5):
    import sep
    import fitsio
    '''
    If using astropy.io.fits, you get an error (see bug on sep`s page).
    '''
    data_2D = fitsio.read(imagename)

    if (mask is None) and (apply_mask==True):
        _, mask = mask_dilation(imagename, PLOT=False,
                                sigma=3, iterations=2, dilation_size=10)
        bkg = sep.Background(data_2D, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh)

    else:
        bkg = sep.Background(data_2D, mask=mask,
                             bw=bw, bh=bh, fw=fw, fh=fh)
    bkg_rms = bkg.rms()
    bkg_image = bkg.back()
    return(bkg)



def sep_source_ext(imagename, sigma=5.0, iterations=5, dilation_size=3,
                   deblend_nthresh = 100, deblend_cont=0.005,maskthresh=0.0,
                   gain=1,filter_kernel=None,mask=None,
                   segmentation_map=False,clean_param=1.0, clean=True,
                   minarea=20,filter_type='matched',sort_by='flux',
                   bw=64, bh=64, fw=3, fh=3, ell_size_factor=4, apply_mask=False):
    import sep
    import fitsio
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from matplotlib import rcParams

    data_2D = fitsio.read(imagename)
    m, s = np.mean(data_2D), mad_std(data_2D)
    plt.imshow(data_2D, interpolation='nearest', cmap='gray', vmin=m - s,
               vmax=m + s, origin='lower')
    plt.colorbar()
    bkg = sep.Background(data_2D)
    if apply_mask:
        _, mask = mask_dilation(data_2D, sigma=sigma, iterations=iterations,
                                dilation_size=dilation_size)
    # else:
    #     mask = None
    if mask is not None:
        data_2D = data_2D * mask
    bkg = sep.Background(data_2D, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh)
    print(bkg.globalback)
    print(bkg.globalrms)
    bkg_image = bkg.back()
    plt.imshow(bkg_image)
    bkg_rms = bkg.rms()
    # fast_plot2(bkg_rms)
    data_sub = data_2D - bkg
    if segmentation_map == True:
        objects, seg_maps = sep.extract(data_sub, thresh=sigma,
                                        minarea=minarea, filter_type=filter_type,
                                        deblend_nthresh=deblend_nthresh,
                                        deblend_cont=deblend_cont,filter_kernel=filter_kernel,
                                        maskthresh=maskthresh, gain=gain,
                                        clean=clean, clean_param=clean_param,
                                        segmentation_map=segmentation_map,
                                        err=bkg.globalrms, mask=mask)
    else:
        objects = sep.extract(data_sub, thresh=sigma,
                              minarea=minarea, filter_type=filter_type,
                              deblend_nthresh=deblend_nthresh,
                              deblend_cont=deblend_cont,filter_kernel=filter_kernel,
                              maskthresh=maskthresh,gain=gain,
                              clean=clean, clean_param=clean_param,
                              segmentation_map=segmentation_map,
                              err=bkg.globalrms, mask=mask)


    # len(objects)
    from matplotlib.patches import Ellipse
    from skimage.draw import ellipse

    fig, ax = plt.subplots()
    m, s = np.mean(data_sub), np.std(data_sub)
    im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                   vmin=m - s, vmax=m + s, origin='lower')

    masks_regions = []

    y, x = np.indices(data_2D.shape[:2])
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width= 2*ell_size_factor * objects['a'][i],
                    height= 2*ell_size_factor * objects['b'][i],
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
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
        label = str('ID'+str(i))
        text = Text(xc+10, yc+10, label+10, ha='center', va='center', color='black')
        ax.add_artist(text)
        masks_regions.append(mask_ell)
    flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
                                         3.0, err=bkg.globalrms, gain=1.0)
    for i in range(len(objects)):
        print("object {:d}: flux = {:f} +/- {:f}".format(i, flux[i], fluxerr[i]))
    objects['b'] / objects['a'], np.rad2deg(objects['theta'])

    #sort regions from largest size to smallest size.
    mask_areas = []
    mask_fluxes = []
    for mask_comp in masks_regions:
        area_mask = np.sum(mask_comp)
        sum_mask = np.sum(mask_comp*data_2D)
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

    if segmentation_map == True:
        return (masks_regions,sorted_indices_desc, seg_maps)
    else:
        return (masks_regions,sorted_indices_desc)


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



def construct_model_parameters(n_components=None, params_values_init=None,
                               init_constraints=None,
                               constrained=True, fix_n=False, fix_value_n=False,
                               fix_x0_y0=False,dr_fix = None,fix_geometry=True,
                               init_params=0.25, final_params=4.0):
    """
    This function creates a single or multi-component Sersic model to be fitted
    onto an astronomical image.

    Note:

    Parameters
    ----------
    n_components:
    params_values_init: np.array or None; optional
        np.array containing initial values for paremeters. These values
        are generated using
            params_values_init = read_imfit_params(imfit_config_file).


    """

    if n_components is None:
        n_components = len(params_values_init) - 1

    smodel2D = setup_model_components(n_components=n_components)
    # print(smodel2D)
    model_temp = Model(sersic2D)
    dr = 10


    # params_values_init = [] #grid of parameter values, each row is the
    # parameter values of a individual component

    if params_values_init is not None:
        """This takes the values from an IMFIT config file as init 
        params and set number of components.
        """
        for i in range(0, n_components):
            # x0, y0, PA, ell, n, In, Rn = params_values_init[i]
            x0, y0, PA, ell, n, In, Rn = params_values_init[i]
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
                            print('Fixing sersic index of component',i,' to 0.5')
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

        smodel2D.set_param_hint('s_a', value=0.5, min=0.01, max=1.0)
    else:
        if init_constraints is not None:
            if constrained == True:
                for j in range(init_constraints['ncomps']):
                    if fix_n is not False:
                        fix_n_j = fix_n[j]
                        fix_value_n_j = fix_value_n[j]
                    if fix_x0_y0 is not False:
                        fix_x0_y0_j = fix_x0_y0[j]
                        dr_fix_j = dr_fix[j]
                    else:
                        fix_x0_y0_j = False
                        dr_fix_j = False
                    jj = str(j + 1)
                    for param in model_temp.param_names:
                        #                         smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                        #                                                 value=eval(param), min=0.000001)
                        if (param == 'n'):
                            if (fix_n_j == True):
                                print('Fixing Sersic Index of component',j,' to 0.5.')
                                dn = 0.01
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=fix_value_n_j,
                                    min=fix_value_n_j-dn, max=fix_value_n_j+dn)
                            else:
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=0.5, min=0.45, max=8.0)

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
                            dell = 0.2
                            ell = 1 - init_constraints['c' + jj + '_q']
                            ell_min = ell * 0.2
                            #                         if ell + dell <= 1.0:
                            if ell * 2.0 <= 0.6:
                                ell_max = ell * 2.0
                                if ell_max <= 0.5:
                                    ell_max = 0.8
                            else:
                                ell_max = 0.8


                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=ell, min=ell_min, max=ell_max)

                        if param == 'cg':
                            if fix_geometry == True:
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
                            convolved model. The PSF convolution atenuates a lot 
                            the signal, specially for radio images.
                            """
                            I50_max = I50 * 20
                            I50_min = I50 * 0.1
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=I50, min=I50_min, max=I50_max)
                        if param == 'Rn':
                            R50 = init_constraints['c' + jj + '_R50']
                            dR = R50 * 0.5
                            # R50_max = R50 * 4.0
                            # R50_max = init_constraints['c' + jj + '_Rp']
                            R50_max = 2.0*init_constraints['c' + jj + '_R50']
                            R50_min = R50 * 0.1 #should be small.
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
                                print('Limiting ', param)
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
                                print('Limiting ', param)
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
                                print('Limiting ', param)
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
                                print('Limiting ', param)
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

            smodel2D.set_param_hint('s_a', value=0.5, min=0.01, max=1.0)
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
                smodel2D.set_param_hint('s_a', value=0.5, min=0.01, max=1.0)
            except:
                print('Please, if not providing initial parameters file,')
                print('provide basic information for the source.')
                return (ValueError)

    params = smodel2D.make_params()
    print(smodel2D.param_hints)
    return (smodel2D, params)


def constrain_nelder_mead_params(params,
                                 max_factor = 1.03,
                                 min_factor = 0.97):
    """
    Constrain Nelder-Mead optimised parameters.
    Since Nelder-Mead is robust, we can feed these values
    into Least-Squares.
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

    petro_properties_copy = petro_properties.copy()
    new_comp_id = petro_properties['ncomps'] + 1
    for k in range(len(unique_list)):
        #         print(unique_list[k])
        # do not change anything for other parameters.
        petro_properties_copy['c' + str(new_comp_id) + '_' + unique_list[k]] = \
        petro_properties_copy['c' + str(copy_from_id) + '_' + unique_list[k]]
        if unique_list[k] == 'R50':
            # multiply the R50 value by a factor, e.g., 2.0
            factor = 3
            petro_properties_copy[
                'c' + str(new_comp_id) + '_' + unique_list[k]] = \
            petro_properties_copy[
                'c' + str(copy_from_id) + '_' + unique_list[k]] * factor
        if unique_list[k] == 'I50':
            # divide the I50 value by a factor, e.g., 1
            factor = 0.05
            petro_properties_copy[
                'c' + str(new_comp_id) + '_' + unique_list[k]] = \
            petro_properties_copy[
                'c' + str(copy_from_id) + '_' + unique_list[k]] * factor
    # update number of components
    petro_properties_copy['ncomps'] = petro_properties_copy['ncomps'] + 1
    return (petro_properties_copy)


def do_fit2D_GPU(imagename, params_values_init=None, ncomponents=None,
             init_constraints=None, data_2D_=None, residualname=None,
             init_params=0.25, final_params=4.0, constrained=True,
             fix_n=True, fix_value_n=False, dr_fix=2,
             fix_x0_y0=False, psf_name=None, convolution_mode='CPU',
             convolve_cutout=False, cut_size=512, self_bkg=False, rms_map=None,
             fix_geometry=True,
             special_name='', method1='least_squares', method2='least_squares',
             save_name_append=''):
    startTime = time.time()

    if data_2D_ is None:
        data_2D = cp.asarray(pf.getdata(imagename))
    else:
        data_2D = cp.asarray(data_2D_)

    if residualname is not None:
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
        residual_2D = pf.getdata(residualname)
        residual_2D_shuffled = shuffle_2D(residual_2D)
        print('Using clean background for optmization...')
        #         background = residual_2D #residual_2D_shuffled
        background = cp.asarray(residual_2D_shuffled)

    else:
        if self_bkg == True:
            if rms_map is not None:
                print('Using provided RMS map.')
            else:
                print('No residual/background provided. Using image bkg map...')
                background_map = sep_background(imagename)
                background = cp.asarray(background_map.back())
        else:
            background = 0
            print('Using only flat sky for rms bkg.')

    if psf_name is not None:
        PSF_CONV = True
        try:
            PSF_BEAM_raw = pf.getdata(psf_name)
            if len(PSF_BEAM_raw.shape) == 4:
                PSF_BEAM_raw = PSF_BEAM_raw[0][0]
        except:
            PSF_BEAM_raw = ctn(psf_name)


        PSF_BEAM = cp.asarray(PSF_BEAM_raw)

    else:
        PSF_CONV = False
        PSF_BEAM = None

    size = data_2D.shape
    xy = cp.meshgrid(cp.arange((size[1])), cp.arange((size[0])))

    #     FlatSky_level = background#mad_std(data_2D)
    FlatSky_level = mad_std(data_2D.get())
    nfunctions = ncomponents

    def residual_2D(params):
        dict_model = {}
        model = 0
        for i in range(1, nfunctions + 1):
            model = model + sersic2D_GPU(xy, params['f' + str(i) + '_x0'],
                                     params['f' + str(i) + '_y0'],
                                     params['f' + str(i) + '_PA'],
                                     params['f' + str(i) + '_ell'],
                                     params['f' + str(i) + '_n'],
                                     params['f' + str(i) + '_In'],
                                     params['f' + str(i) + '_Rn'],
                                     params['f' + str(i) + '_cg'], )
        # print(model.shape)
        model = cp.asarray(model) + cp.asarray(FlatSky(FlatSky_level, params['s_a'])) + background
        """
        Experimental convolution with GPU. Faster???
        """
        # model_gpu = cp.asarray(model)
        model_gpu = model
        # psf_gpu = cp.asarray(PSF_BEAM)
        model_conv_gpu = cupyx.scipy.signal.fftconvolve(model_gpu,
                                                        PSF_BEAM,
                                                        mode='same')
        cp.cuda.Stream.null.synchronize()
        MODEL_2D_conv = model_conv_gpu
        # MODEL_2D_conv = cp.asnumpy(model_conv_gpu)
        # MODEL_2D_conv = model_conv_gpu.get()
        residual = data_2D - MODEL_2D_conv
        return (residual.get())

    smodel2D, params = construct_model_parameters(
        params_values_init=params_values_init, n_components=nfunctions,
        init_constraints=init_constraints,
        fix_n=fix_n, fix_value_n=fix_value_n,
        fix_x0_y0=fix_x0_y0, dr_fix=dr_fix, fix_geometry=fix_geometry,
        init_params=init_params, final_params=final_params,
        constrained=constrained)

    mini = lmfit.Minimizer(residual_2D, params, max_nfev=10000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    # initial minimization.

    print(' >> Using', method1, ' solver for first optimisation run... ')
    # take parameters from previous run, and re-optimize them.
    #     method2 = 'ampgo'#'least_squares'
    #     method2 = 'least_squares'
    result_extra = None
    if method1 == 'nelder':
        # very robust, but takes time....
        #         print(' >> Using', method1,' solver for first optimisation run... ')
        result_1 = mini.minimize(method='nelder',
                                 #                                  xatol = 1e-12, fatol = 1e-12, disp=True,
                                 #                                  adaptive = True,max_nfev = 30000,
                                 options={'maxiter': 30000, 'maxfev': 30000,
                                          'xatol': 1e-12, 'fatol': 1e-12,
                                          'return_all': True,
                                          'disp': True}
                                 )

    if method1 == 'least_squares':
        # faster, but usually not good for first run.
        result_1 = mini.minimize(method='least_squares',
                                 max_nfev=30000, x_scale='jac',  # f_scale=0.5,
                                 tr_solver="exact",
                                 tr_options={'regularize': True},
                                 ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=2,
                                 loss="cauchy")  # ,f_scale=0.5, max_nfev=5000, verbose=2)

    if method1 == 'differential_evolution':
        # de is giving some issues, I do not know why.
        result_1 = mini.minimize(method='differential_evolution', popsize=600,
                                 disp=True,  # init = 'random',
                                 # mutation=(0.5, 1.5), recombination=[0.2, 0.9],
                                 max_nfev=20000,
                                 workers=1, updating='deferred', vectorized=True)

    print(' >> Using', method2, ' solver for second optimisation run... ')

    if method2 == 'nelder':
        result = mini.minimize(method='nelder', params=result_1.params,
                               options={'maxiter': 30000, 'maxfev': 30000,
                                        'xatol': 1e-13, 'fatol': 1e-13,
                                        'disp': True})

    if method2 == 'ampgo':
        # ampgo is not workin well/ takes so long ???
        result = mini.minimize(method='ampgo', params=result_1.params,
                               maxfunevals=10000, totaliter=30, disp=True,
                               maxiter=5, glbtol=1e-8)

    if method2 == 'least_squares':
        # faster, usually converges and provide errors.
        # Very robust if used in second opt from first opt parameters.
        result = mini.minimize(method='least_squares', params=result_1.params,
                               max_nfev=30000,
                               tr_solver="exact",
                               tr_options={'regularize': True},
                               x_scale='jac',  # f_scale=0.5,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=2,
                               loss="cauchy")  # ,f_scale=0.5, max_nfev=5000, verbose=2)

    if method2 == 'differential_evolution':
        result = mini.minimize(method='differential_evolution',
                               params=result_1.params,
                               options={'maxiter': 30000, 'workers': -1,
                                        'tol': 0.001, 'vectorized': True,
                                        'strategy': 'randtobest1bin',
                                        'updating': 'deferred', 'disp': True,
                                        'seed': 1}
                               )

    params = result.params

    model_temp = Model(sersic2D_GPU)
    model = 0
    # size = data_2D.shape
    # xy = np.meshgrid(np.arange((size[0])), np.arange((size[1])))
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
                              params['f' + str(i) + '_cg']) + \
                     background / ncomponents + cp.asarray(FlatSky(
            FlatSky_level, params['s_a'])) / ncomponents
        #                                  params['f'+str(i)+'_Rn'])+FlatSky(FlatSky_level, params['s_a'])/ncomponents
        # print(model_temp[0])
        model = model + model_temp
        # print(model)
        model_dict['model_c' + str(i)] = model_temp.get()


        model_dict['model_c' + str(i) + '_conv'] = cupyx.scipy.signal.fftconvolve(
            model_temp, PSF_BEAM,
            'same').get()  # + FlatSky(FlatSky_level, params['s_a'])/ncomponents

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
        image_results_deconv.append(imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_dec_model_component_" + str(
            i) + special_name + save_name_append + '.fits')

    #     model = model
    model_dict['model_total'] = model.get()  # + FlatSky(FlatSky_level, params['s_a'])

    model_dict['model_total_conv'] = cupyx.scipy.signal.fftconvolve(model,
                                                              PSF_BEAM_raw,
                                                              'same').get()  # + FlatSky(FlatSky_level, params['s_a'])


    model_dict['best_residual'] = data_2D.get() - model_dict['model_total']
    model_dict['best_residual_conv'] = data_2D.get() - model_dict['model_total_conv']

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
    # # initial minimization.
    # method1 = 'differential_evolution'
    # print(' >> Using', method1, ' solver for first optimisation run... ')
    # # take parameters from previous run, and re-optimize them.
    # #     method2 = 'ampgo'#'least_squares'
    # method2 = 'least_squares'

    image_results_conv.append(imagename.replace('.fits', '') + "_" + str(
        ncomponents) + "C_model" + special_name + save_name_append + '.fits')
    image_results_deconv.append(imagename.replace('.fits', '') + "_" + str(
        ncomponents) + "C_dec_model" + special_name + save_name_append + '.fits')
    image_results_conv.append(imagename.replace('.fits', '') + "_" + str(
        ncomponents) + "C_residual" + special_name + save_name_append + ".fits")

    # save mini results (full) to a pickle file.
    with open(imagename.replace('.fits', '_' + str(
            ncomponents) + 'C_fit' + special_name + save_name_append + '.pickle'),
              "wb") as f:
        pickle.dump(result, f)
    exec_time = time.time() - startTime
    print('Exec time fitting=', exec_time, 's')

    # save results to csv file.
    try:
        save_results_csv(result_mini=result,
                         save_name=image_results_conv[-2].replace('.fits', ''),
                         ext='.csv',
                         save_corr=True, save_params=True)
    except:
        pass

    return (result, mini, result_1, result_extra, model_dict, image_results_conv,
            image_results_deconv)


def return_and_save_model(mini_results, imagename, ncomponents, background=0.0,
                          save_results=False,save_name_append=''):
    params = mini_results.params
    data_2D = ctn(imagename)
    model_temp = Model(sersic2D)
    model = 0
    PSF_CONV = True
    size = ctn(imagename).shape
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
                     background / ncomponents + FlatSky(FlatSky_level,
                                                        arams['s_a']) / ncomponents
        # print(model_temp[0])
        model = model + model_temp
        # print(model)
        model_dict['model_c' + str(i)] = model_temp

        if PSF_CONV == True:
            model_dict['model_c' + str(i) + '_conv'] = scipy.signal.fftconvolve(
                model_temp, PSF_BEAM,
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
                                                                  PSF_BEAM,
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

def fast_plot2(imagename, crop=False, box_size=128, center=None, with_wcs=True,vmax_factor=0.5,
               vmin_factor=1, plot_colorbar=True, figsize=(5, 5), aspect=1, ax=None):
    """
    Fast plotting of an astronomical image with/or without a wcs header.

    imagename:
        str or 2d array.
        If str (the image file name), it will attempt to read the wcs and plot the coordinates axes.

        If 2darray, will plot the data with generic axes.

        support functions:
            ctn() -> casa to numpy: A function designed mainly to read CASA fits images,
                     but can be used to open any fits images.

                     However, it does not read header/wcs.
                     Note: THis function only works inside CASA environment.




    """
    if ax == None:
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
                g = ctn(imagename)
        if with_wcs == False:
            g = ctn(imagename)

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
    if ax == None:
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

    ax1.plot(plot_slice, np.mean(data_2D, axis=0), '--.', color='purple', ms=14,
             label='DATA')
    ax1.plot(plot_slice, np.mean(model_dict['model_total_conv'], axis=0), '.-',
             color='limegreen', linewidth=4, label='MODEL')
    ax1.plot(plot_slice, np.mean(model_dict['best_residual_conv'], axis=0), '.-',
             color='black', linewidth=4, label='RESIDUAL')
    ax1.plot(plot_slice, np.mean(residual_2D, axis=0), '.-', color='grey',
             linewidth=4, label='MAP RESIDUAL')
    #     ax1.set_xlabel('$x$-slice')
    #     ax1.set_xaxis('off')
    #     ax1.set_xticks([])
    ax1.legend(fontsize=11)
    ax1.grid()
    ax1.set_ylabel('mean $x$ direction')
    if Rp_props is not None:
        ax1.set_xlim(Rp_props['c1_x0c'] - plotlim, Rp_props['c1_x0c'] + plotlim)
    #     ax1.set_title('asd')
    # plt.plot(np.mean(shuffled_image,axis=0),color='red')

    ax2.plot(plot_slice, np.mean(data_2D, axis=1), '--.', color='purple', ms=14,
             label='DATA')
    ax2.plot(plot_slice, np.mean(model_dict['model_total_conv'], axis=1), '.-',
             color='limegreen', linewidth=4, label='MODEL')
    ax2.plot(plot_slice, np.mean(model_dict['best_residual_conv'], axis=1), '.-',
             color='black', linewidth=4, label='RESIDUAL')
    ax2.plot(plot_slice, np.mean(residual_2D, axis=1), '.-', color='grey',
             linewidth=4, label='MAP RESIDUAL')
    ax2.set_xlabel('Image Slice [px]')
    ax2.set_ylabel('mean $y$ direction')
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
                     sources_photometies,vmax_factor=0.1,data_2D_=None,
                     vmin_factor=3, show_figure=True,crop=False,box_size=100):
    if data_2D_ is not None:
        data_2D = data_2D_
    else:
        data_2D = ctn(imagename)

    fast_plot3(data_2D, modelname=model_dict['model_total_conv'],
               residualname=model_dict['best_residual_conv'],
               reference_image=imagename,
               NAME=image_results_conv[-2].replace('.fits',
                                                   'result_image_conv.pdf'),
               crop=crop, vmin_factor=vmin_factor,
               box_size=box_size)



    ncomponents = sources_photometies['ncomps']
    if sources_photometies is not None:
        plotlim =  3 * sources_photometies['c1_rlast']
        # plotlim = 0
        # for i in range(ncomponents):
        #     plotlim = plotlim + sources_photometies['c' + str(i + 1) + '_rlast']

    model_name = image_results_conv[-2]
    residual_name = image_results_conv[-1]
    cell_size = get_cell_size(imagename)
    profile_data = {}
    # center = get_peak_pos(imagename)
    center = nd.maximum_position(data_2D)[::-1]
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

    # plt.plot(radiis[0],profiles[0])
    # plt.plot(radiis[1],profiles[1])
    # plt.plot(radiis[2],np.log(profiles[2]))
    # colors = ['black','purple','gray','red']
    colors = ['red', 'blue', 'teal', 'brown', 'cyan','orange','forestgreen','pink']
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

    plt.plot(r * cell_size, abs(irre), '.-', label='RESIDUAL', color='black')
    plt.plot(r * cell_size, abs(irmodel), '--', color='limegreen', label='MODEL',
             linewidth=4)
    plt.semilogy()
    plt.xlabel(r'$r$ [arcsec]')
    plt.ylabel(r'$I(r)$ [Jy/beam]')
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

def total_flux(data2D,image,mask=None,
               sigma=6,iterations=3,dilation_size=7,PLOT=False,
               silent=True):

    BA = beam_area2(image)
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
                        figsize=(13,13),
                        special_name=''):

    decomp_results = {}
    if rms == None:
        rms = mad_std(ctn(imagename))
    else:
        rms = rms

    max_factor =  ctn(imagename).max()
#     compact = model_dict['model_c1_conv']
    if data_2D_ is not None:
        data_2D = data_2D_
    else:
        data_2D = ctn(imagename)

    # rms_std_data = mad_std(data_2D)
    extended = data_2D  - compact
    residual_modeling = data_2D - (compact + extended_model)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(3, 3, 1)
    # ax.yaxis.set_ticks([])
    ax = eimshow(imagename,ax=ax,rms=rms,plot_title='Total Emission',
                 vmax_factor=vmax_factor,vmin_factor=vmin_factor)
    # cb = plt.colorbar(mappable=plt.gca().images[0],
    #                           cax=fig.add_axes([0.9, 0.65, 0.02, 0.2]))
    # ax.yaxis.set_ticks([])
    ax.axis('off')
    ax = fig.add_subplot(3, 3, 2)
    # ax.yaxis.set_ticks([])
    ax = eimshow(compact,ax=ax,rms=rms,
                 plot_title='Compact Emission',max_factor=data_2D.max(),
                 vmax_factor=vmax_factor,vmin_factor=vmin_factor)
    # ax.yaxis.set_ticks([])
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 3)
    # ax.yaxis.set_ticks([])
    ax = eimshow(extended,ax=ax,rms=rms,max_factor=data_2D.max(),
                 plot_title='Diffuse Emission',vmax_factor=vmax_factor,
                 vmin_factor=vmin_factor)
    # ax.yaxis.set_ticks([])
    # cb = plt.colorbar(mappable=plt.gca().images[0],
    #                           cax=fig.add_axes([0.00, 0.65, 0.02, 0.2]))
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 4)
    slice_ext = np.sqrt(np.mean(extended,axis=0)**2.0 + np.mean(extended,axis=1)**2.0)
    slice_ext_model = np.sqrt(
        np.mean(extended_model, axis=0) ** 2.0 + np.mean(extended_model, axis=1) ** 2.0)
    slice_data = np.sqrt(np.mean(data_2D,axis=0)**2.0 + np.mean(data_2D,axis=1)**2.0)
    ax.plot(slice_ext,label='COMPACT SUB')
    ax.plot(slice_data,label='DATA')
    ax.plot(slice_ext_model, label='EXTENDED MODEL')
    plt.legend()
    xlimit = [data_2D.shape[0] / 2 - 0.15 * data_2D.shape[0],
              data_2D.shape[0] / 2 + 0.15 * data_2D.shape[0]]
    ax.set_xlim(xlimit[0],xlimit[1])
    # ax.semilogx()

    ax = fig.add_subplot(3, 3, 5)
    ax.axis('off')

    omaj, omin, _, _, _ = beam_shape(imagename)
    dilation_size = int(
        np.sqrt(omaj * omin) / (2 * get_cell_size(imagename)))

    _, mask_model_rms_self_compact = mask_dilation(compact,
                                           sigma=1, dilation_size=dilation_size,
                                           iterations=2,PLOT=False)
    _, mask_data = mask_dilation(data_2D,
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
    _, mask_model_rms_image_extended_model = mask_dilation(extended_model,
                                            rms=rms,
                                            sigma=1, dilation_size=dilation_size,
                                            iterations=2,PLOT=False)

    print('Flux on compact (self rms) = ',
          1000*np.sum(compact*mask_model_rms_self_compact)/beam_area2(imagename))
    print('Flux on compact (data rms) = ',
          1000 * np.sum(compact * mask_model_rms_image_compact) / beam_area2(imagename))
    flux_density_compact = 1000*np.sum(
        compact*mask_model_rms_image_compact)/beam_area2(imagename)
    flux_density_extended_model = 1000 * np.sum(
        extended_model * mask_data) / beam_area2(imagename)

    flux_density_ext = 1000*total_flux(extended,imagename,
                                       mask = mask_model_rms_image_extended)
    flux_density_ext2 = 1000*np.sum(
        extended*mask_data)/beam_area2(imagename)

    flux_data = 1000*total_flux(data_2D,imagename,
                                       mask = mask_data)
    flux_density_ext_self_rms = 1000*total_flux(extended,imagename,
                                       mask = mask_model_rms_self_extended)

    flux_res = flux_data - (flux_density_extended_model + flux_density_compact)

    print('Flux on extended (self rms) = ',flux_density_ext_self_rms)
    print('Flux on extended (data rms) = ',flux_density_ext)
    print('Flux on extended2 (data rms) = ', flux_density_ext2)
    print('Flux on extended model (data rms) = ', flux_density_extended_model)
    print('Flux on data = ', flux_data)
    print('Flux on residual = ', flux_res)

    decomp_results['flux_data'] = flux_data
    decomp_results['flux_density_ext'] = flux_density_ext
    decomp_results['flux_density_ext2'] = flux_density_ext2
    decomp_results['flux_density_extended_model'] = flux_density_extended_model
    decomp_results['flux_density_compact'] = flux_density_compact
    decomp_results['flux_res'] = flux_res


    # print('r_half_light (old vs new) = {:0.2f} vs {:0.2f}'.format(p.r_half_light, p_copy.r_half_light))
    ax.annotate(r"$S_\nu^{\rm comp}=$"+'{:0.2f}'.format(flux_density_compact)+' mJy',
                (0.33, 0.32), xycoords='figure fraction', fontsize=18)
    ax.annotate(r"$S_\nu^{\rm ext}\ \ \ =$"+'{:0.2f}'.format(flux_density_ext2)+' mJy',
                (0.33, 0.29), xycoords='figure fraction', fontsize=18)
    ax.annotate(r"$S_\nu^{\rm ext \ model}\ \ \ =$"+'{:0.2f}'.format(flux_density_extended_model)+' mJy',
                (0.33, 0.26), xycoords='figure fraction', fontsize=18)
    plt.savefig(
        imagename.replace('.fits', '_extended'+special_name+'.jpg'),
        dpi=300,
        bbox_inches='tight')

    save_data = True
    if save_data == True:
        exteded_file_name = imagename.replace('.fits', '') + \
                            special_name + '_extended.fits'
        pf.writeto(exteded_file_name,extended,overwrite=True)
        copy_header(imagename,exteded_file_name)
        compact_file_name = imagename.replace('.fits', '') + \
                            special_name + '_compact.fits'
        pf.writeto(compact_file_name,compact,overwrite=True)
        copy_header(imagename,compact_file_name)


    return(decomp_results)



def plot_interferometric_decomposition(imagename0, imagename,
                                       modelname, residualname,
                                       crop=False, box_size=512,
                                       max_percent_lowlevel=99.0,
                                       max_percent_highlevel=99.9999,
                                       NAME=None, EXT='.pdf',
                                       run_phase = '1st',
                                       vmin_factor=3,vmax_factor=0.1,
                                       SPECIAL_NAME='', show_figure=True):
    """
    Fast plotting of image <> model <> residual images.

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
        I1 = ctn(imagename0)
        g = ctn(imagename)
        m = ctn(modelname)
        r = ctn(residualname)

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
    vmax0 = 1.0 * g.max()
    vmin = vmin_factor * std  # 0.5*g.min()#
    vmax = vmax_factor * g.max()
    vmin_r = 0.5 * r.min()  # 1*std_r
    vmax_r = 1.0 * r.max()
    vmin_m = 1 * mad_std(m)  # vmin#0.01*std_m#0.5*m.min()#
    vmax_m = m.max()  # vmax#0.5*m.max()

    levels_I1 = np.geomspace(2*I1.max(), 1.5 * np.std(I1), 7)
    levels_g = np.geomspace(2*g.max(), 3 * std, 7)
    levels_m = np.geomspace(2*m.max(), 20 * std_m, 7)
    levels_r = np.geomspace(2*r.max(), 3 * std_r, 7)
    levels_neg = np.asarray([-3]) * std
    if run_phase == '1st':
        title_labels = [r'$I_1^{\rm mask}$',
                        r'$I_2$',
                        r'$I_{1}^{\rm mask} * \theta_2$',
                        r'$R_{12} = I_2 - I_{1}^{\rm mask} * \theta_2 $'
                        ]

    if run_phase == '2nd':
        title_labels = [r'$R_{12}$',
                        r'$I_3$',
                        r'$I_{1}^{\rm mask} * \theta_3 + R_{12} * \theta_3$',
                        r'$R_{T}$'
                        ]

    if run_phase == 'compact':
        title_labels = [r'$R_{12}$',
                        r'$I_3$',
                        r'$I_{1}^{\rm mask} * \theta_3$',
                        r'$I_3 - I_{1}^{\rm mask} * \theta_3$'
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

    ax.contour(I1, levels=levels_I1[::-1], colors=cm,
               extent=[-dx1, dx1, -dx1, dx1],
               linewidths=0.8, alpha=1.0)  # cmap='Reds', linewidths=0.75)

    cell_size = get_cell_size(imagename0)

    xticks = np.linspace(-dx1, dx1, 5)
    xticklabels = np.linspace(-dx1*cell_size, +dx1*cell_size, 5)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    ax.set_xlabel(r'Offset [arcsec]')
    # ax.set_yticks([])
    ax.set_yticklabels([])

    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    ax = fig.add_subplot(1, 4, 2)
    #     im = ax.imshow(g, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(g, cmap='magma_r',extent=[-dx,dx,-dx,dx],
                        origin='lower', alpha=1.0, norm=norm2)

    ax.set_title(title_labels[1])

    ax.contour(g, levels=levels_g[::-1], colors=cm,extent=[-dx,dx,-dx,dx],
               linewidths=0.8, alpha=1.0)  # cmap='Reds', linewidths=0.75)

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
    cb.set_label(r'Flux [mJy/beam]', labelpad=1)
    cb.ax.xaxis.set_tick_params(pad=1)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(1)
    # cb.dividers.set_color('none')

    cell_size = get_cell_size(imagename)
    xticks = np.linspace(-dx, dx, 5)
    xticklabels = np.linspace(-dx*cell_size, +dx*cell_size, 5)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    # ax.set_yticks([])
    ax.set_yticklabels([])

    ax = plt.subplot(1, 4, 3)

    #     im_plot = ax.imshow(m, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax_m, vmin=vmin_m)#norm=norm
    im_plot = ax.imshow(m, cmap='magma_r',extent=[-dx,dx,-dx,dx],
                        origin='lower', alpha=1.0, norm=norm2)
    ax.set_title(title_labels[2])
    ax.contour(m, levels=levels_g[::-1], colors=cm,extent=[-dx,dx,-dx,dx],
               linewidths=0.8, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.08,0.3,0.02,0.4]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    cell_size = get_cell_size(modelname)
    xticks = np.linspace(-dx, dx, 5)
    xticklabels = np.linspace(-dx*cell_size, +dx*cell_size, 5)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    # ax.set_yticks([])
    ax.set_yticklabels([])



    ax = plt.subplot(1, 4, 4)
    norm_re = simple_norm(r, min_cut=vmin, max_cut=vmax, stretch='sqrt')  # , max_percent=max_percent_highlevel)
    #     norm = simple_norm(r,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    #     ax.imshow(r,origin='lower',cmap='magma_r',alpha=1.0,vmax=vmax_r, vmin=vmin)#norm=norm
    ax.imshow(r, origin='lower',extent=[-dx,dx,-dx,dx],
              cmap='magma_r', alpha=1.0, norm=norm2)
    #     ax.imshow(r, cmap='magma_r',norm=norm,alpha=0.3,origin='lower')

    ax.contour(r, levels=levels_r[::-1],extent=[-dx,dx,-dx,dx],
               colors=cm,#colors='grey',
               linewidths=0.8, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    ax.contour(r, levels=levels_neg[::-1],extent=[-dx,dx,-dx,dx],
               colors='k', linewidths=1.0,
               alpha=1.0)

    cell_size = get_cell_size(residualname)
    xticks = np.linspace(-dx, dx, 5)
    xticklabels = np.linspace(-dx*cell_size, +dx*cell_size, 5)
    xticklabels = ['{:.2f}'.format(xtick) for xtick in xticklabels]
    ax.set_yticks(xticks,xticklabels)
    ax.set_xticks(xticks,xticklabels)
    # ax.set_yticks([])
    ax.set_yticklabels([])


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
        I1 = ctn(imagename0)
        g = ctn(imagename)
        m = ctn(modelname)
        r = ctn(residualname)

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
        g = ctn(imagename)
        m = ctn(modelname)
        r = ctn(residualname)

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
            vmax_factor=0.5, neg_levels=np.asarray([-3]), CM='magma_r',
            rms=None, max_factor=None,plot_title=None,apply_mask=False,
            add_contours=True,extent=None,
            vmin_factor=3, plot_colorbar=True, figsize=(5, 5), aspect=1,
            ax=None):
    """
    Fast plotting of an astronomical image with/or without a wcs header.
    neg_levels=np.asarray([-3])
    imagename:
        str or 2d array.
        If str (the image file name), it will attempt to read the wcs and plot the coordinates axes.

        If 2darray, will plot the data with generic axes.

        support functions:
            ctn() -> casa to numpy: A function designed mainly to read CASA fits images,
                     but can be used to open any fits images.

                     However, it does not read header/wcs.
                     Note: THis function only works inside CASA environment.

    """
    try:
        import cmasher as cmr
        print('Imported cmasher for density maps.'
              'If you would like to use, examples:'
              'CM = cmr.ember,'
              'CM = cmr.flamingo,'
              'CM = cmr.gothic'
              'CM = cmr.lavender')
        """
        ... lilac,rainforest,sepia,sunburst,torch.
        Diverging: copper,emergency,fusion,infinity,pride'
        """
    except:
        print('Error importing cmasher. If you want '
              'to use its colormaps, install it. '
              'Then you can use for example:'
              'CM = cmr.flamingo')
    if ax == None:
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
                g = ctn(imagename)
        if with_wcs == False:
            g = ctn(imagename)
            # print('1', g)

        if crop == True:
            xin, xen, yin, yen = do_cutout(imagename, box_size=box_size,
                                           center=center, return_='box')
            g = g[xin:xen, yin:yen]
            # print('2', g)
            crop = False

    else:
        g = imagename
        # print('3', g)

    if crop == True:
        xin, xen, yin, yen = do_cutout(imagename, box_size=box_size,
                                       center=center, return_='box')
        g = g[xin:xen, yin:yen]
        # print('4', g)
    #         max_x, max_y = np.where(g == g.max())
    #         xin = max_x[0] - box_size
    #         xen = max_x[0] + box_size
    #         yin = max_y[0] - box_size
    #         yen = max_y[0] + box_size
    #         g = g[xin:xen, yin:yen]
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
            std = g.std()
        else:
            std = mad_std(g)

    if ax == None:
        if with_wcs == True and isinstance(imagename, str) == True:
            ax = fig.add_subplot(projection=ww.celestial)
            ax.set_xlabel('RA')
            ax.set_ylabel('DEC')
        else:
            ax = fig.add_subplot()
            ax.set_xlabel('x pix')
            ax.set_ylabel('y pix')

    if apply_mask == True:
        _, mask_d = mask_dilation(imagename, cell_size=None,
                                  sigma=6, rms=None,
                                  dilation_size=None,
                                  iterations=3, dilation_type='disk',
                                  PLOT=False, show_figure=False)
        g = g * mask_d

    vmin = vmin_factor * std

    #     print(g)
    if max_factor is not None:
        vmax = vmax_factor * max_factor
    else:
        vmax = vmax_factor * g.max()

    norm = simple_norm(g, stretch='sqrt', asinh_a=0.02, min_cut=vmin,
                       max_cut=vmax)

    im_plot = ax.imshow((g), cmap=CM, origin='lower', alpha=1.0,extent=extent,
                        norm=norm,
                        aspect=aspect)  # ,vmax=vmax, vmin=vmin)#norm=norm
    if plot_title is not None:
        ax.set_title(plot_title)

    levels_g = np.geomspace(2.0 * g.max(), vmin_factor * std, 9)

    #     x = np.geomspace(1.5*mad_std(g),10*mad_std(g),4)
    levels_black = np.geomspace(vmin_factor * std + 0.00001, 2.5 * g.max(), 7)
    #     xneg = np.geomspace(5*mad_std(g),vmin_factor*mad_std(g),2)
    #     y = -xneg[::-1]
    #     levels_black = np.append(y,x)
    levels_neg = neg_levels * std
    #     levels_white = np.geomspace(g.max(),10*mad_std(g),7)
    levels_white = np.geomspace(g.max(), 0.1 * g.max(), 5)

    #     cg.show_contour(data, colors='black',levels=levels_black,linestyle='.-',linewidths=0.2,alpha=1.0)
    #     cg.show_contour(data, colors='#009E73',levels=levels_white[::-1],linewidths=0.2,alpha=1.0)
    # try:
    #     ax.contour(g, levels=levels_black, colors='grey', linewidths=0.2,
    #                alpha=1.0)  # cmap='Reds', linewidths=0.75)
    # except:
    #     pass
    if add_contours:
        try:
            ax.contour(g, levels=levels_g[::-1], colors='grey', linewidths=1.0,extent=extent,
                       alpha=1.0)  # cmap='Reds', linewidths=0.75)
        except:
            pass
        try:
            ax.contour(g, levels=levels_neg[::-1], colors='k', linewidths=1.0,extent=extent,
                       alpha=1.0)
        except:
            pass
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    if plot_colorbar:
        try:
            cb = plt.colorbar(mappable=plt.gca().images[0],
                              cax=fig.add_axes([0.91, 0.08, 0.05, 0.84]))
            cb.set_label(r"Flux Density [Jy/Beam]")
        except:
            pass
    # if ax==None:
    #     if plot_colorbar==True:
    #         # cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.08,0.05,0.82]))
    #         cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.08,0.05,0.32]))
    #         # cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))
    #         cb.set_label(r"Flux [Jy/Beam]")
    return (ax)


def plot_image(image, residual_name=None, box_size=200, box_size_inset=60,
               center=None, rms=None, add_inset='auto', add_beam=True,
               vmin_factor=3, max_percent_lowlevel=99.0,
               max_percent_highlevel=99.9999,
               do_cut=False, CM='magma_r', cbar_axes=[0.03, 0.11, 0.05, 0.77],
               # cbar_axes=[0.9, 0.15, 0.04, 0.7],
               source_distance=None, save_name=None, special_name='',
               show_axis='on', plot_color_bar=True, figsize=(5, 5),
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
        if center == None:
            st = imstat(image)
            print('  >> Center --> ', st['maxpos'])
            xin, xen, yin, yen = st['maxpos'][0] - box_size, st['maxpos'][
                0] + box_size, st['maxpos'][1] - box_size, \
                                 st['maxpos'][1] + box_size
        else:
            xin, xen, yin, yen = center[0] - box_size, center[0] + box_size, \
                                 center[1] - box_size, center[1] + box_size
    else:
        xin, xen, yin, yen = 0, -1, 0, -1

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
    extent = None

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
    vmax = 1.0 * data_range.max()

    # set second normalization
    norm2 = simple_norm(abs(hdu[0].data.squeeze())[xin:xen, yin:yen] * scalling,
                        min_cut=vmin,
                        max_cut=vmax, stretch='asinh',
                        asinh_a=0.01)  # , max_percent=max_percent_highlevel)
    norm2.vmin = vmin

    # this is what is actually shown on the final plot, better contrast.
    im = ax.imshow(hdu[0].data.squeeze()[xin:xen, yin:yen] * scalling,
                   origin='lower',
                   norm=norm2, cmap=cm, aspect='auto', extent=extent)

    levels_colorbar = np.geomspace(1.0 * data_range.max(), 5 * std,
                                   8)  # draw contours only until 5xstd level.
    levels_neg = np.asarray([-3 * std])
    levels_low = np.asarray([3 * std])
    #     levels_neg = np.asarray([])
    #     levels_low = np.asarray([])

    #     levels_colorbar = np.append(levels_neg,levels_pos)

    levels_colorbar2 = np.geomspace(1.0 * data_range.max(), 3 * std,
                                    5)  # draw contours only until 5xstd level.
    ax.contour(hdu[0].data.squeeze()[xin:xen, yin:yen],
               levels=levels_colorbar[::-1], colors='grey',
               linewidths=1.0)  # ,alpha=0.6)

    ax.contour(hdu[0].data.squeeze()[xin:xen, yin:yen], levels=levels_low[::-1],
               colors='brown', linestyle='dashdot',
               linewidths=1.0, alpha=0.6)
    ax.contour(hdu[0].data.squeeze()[xin:xen, yin:yen], levels=levels_neg[::-1],
               colors='k',
               linewidths=1.0, alpha=0.5)

    ww.wcs.radesys = 'icrs'
    radesys = ww.wcs.radesys
    if source_distance is not None:
        # distance = source_distance * u.Mpc
        distance = angular_distance_cosmo(source_distance)  # * u.Mpc
        img = hdu[0].data.squeeze()[xin:xen, yin:yen]
        scalebar_length = 1.0 * u.kpc
        scalebar_loc = (0.80, 0.1)  # y, x
        left_side = coordinates.SkyCoord(
            *ww.celestial[xin:xen, yin:yen].wcs_pix2world(
                scalebar_loc[1] * img.shape[1],
                scalebar_loc[0] * img.shape[0],
                0) * u.deg,
            frame=radesys.lower())

        length = (scalebar_length / distance).to(u.arcsec,
                                                 u.dimensionless_angles())
        make_scalebar(ax, left_side, length, color='purple', linestyle='-',
                      label=f'{scalebar_length:0.1f}',
                      text_offset=0.1 * u.arcsec, fontsize=20)

    #     _ = ax.set_xlabel(f"Right Ascension {radesys}")
    #     _ = ax.set_ylabel(f"Declination {radesys}")
    _ = ax.set_xlabel(f"Right Ascension")
    _ = ax.set_ylabel(f"Declination")

    freq = '{:.2f}'.format(imhd['refval'][2] / 1e9)
    label_pos_x, label_pos_y = hdu[0].data.squeeze()[xin:xen, yin:yen].shape
    #     ax.annotate(r'' + freq + 'GHz', xy=(0.35, 0.05),  xycoords='figure fraction', fontsize=14,
    #                 color='red')
    ax.annotate(r'' + freq + 'GHz',
                xy=(label_pos_x * 0.20, label_pos_y * (0.02)), xycoords='data',
                fontsize=14,
                color='red')

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
        xoffset = hdu[0].data.shape[0] * pixel_scale / 2
        yoffset = hdu[0].data.shape[1] * pixel_scale / 2
        ax2_x = ax.twinx()
        ax3_y = ax2_x.twiny()
        ax2_x.set_ylabel('Offset [arcsec]', fontsize=14)
        ax3_y.set_xlabel('Offset [arcsec]', fontsize=14)
        ax2_x.yaxis.set_ticks(np.linspace(-xoffset, xoffset, 6))
        ax3_y.xaxis.set_ticks(np.linspace(-yoffset, yoffset, 6))


        ax2_x.tick_params(axis='y', which='both', labelsize=16, color='black',
                          pad=-30)
        ax3_y.tick_params(axis='x', which='both', labelsize=16, color='black',
                          pad=-25)
        ax2_x.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3_y.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2_x.grid(which='both', axis='both', color='gray', linewidth=0.6,
                   alpha=0.5)
        ax3_y.grid(which='both', axis='both', color='gray', linewidth=0.6,
                   alpha=0.5)
        ax2_x.axis(show_axis)
        ax3_y.axis(show_axis)

    if plot_color_bar == True:

        def format_func(value, tick_number, scale_density='mJy'):
            # Use the custom formatter for the colorbar ticks
            mantissa = value * 1000
            return r"${:.1f}$".format(mantissa)

        cax = fig.add_axes(cbar_axes)


        cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                            shrink=1, aspect='auto', pad=10, fraction=1.0,
                            drawedges=False, ticklocation='left')

        cbar.formatter = CustomFormatter(factor=1000, useMathText=True)
        cbar.update_ticks()

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

        x1, x2, y1, y2 = xin_cut, xen_cut, yin_cut, yen_cut
        print(xen_cut - xin_cut)
        extent = [xin_cut,#*pixel_scale/2,
                  xen_cut,#*pixel_scale/2,
                  yin_cut,#*pixel_scale/2,
                  yen_cut#*pixel_scale/2
                  ]
        # xoffset_in = (xin_cut - xen_cut) * pixel_scale / 2
        # yoffset_in = (yin_cut - yen_cut) * pixel_scale / 2
        xoffset_in = (xin_cut - xen_cut) * pixel_scale / 2
        yoffset_in = (yin_cut - yen_cut) * pixel_scale / 2
        # extent_arcsec = [-xoffset_in, xoffset_in, -yoffset_in, yoffset_in]
        extent_arcsec = extent
        #         extent_arcsec= [xin_cut* pixel_scale, xen_cut* pixel_scale,
        #                         yin_cut* pixel_scale, yen_cut* pixel_scale]
        norm_inset = visualization.simple_norm(Z2 , stretch='linear',
                                               max_percent=max_percent_lowlevel)

        norm2_inset = simple_norm(abs(Z2) , min_cut=vmin,
                                  max_cut=vmax_inset, stretch='asinh',
                                  asinh_a=0.05)  # , max_percent=max_percent_highlevel)
        norm2_inset.vmin = vmin

        axins.imshow(Z2, cmap=CM, norm=norm_inset, alpha=0.2,
                     extent=extent_arcsec,
                     aspect='auto', origin='lower')
        axins.imshow(Z2, norm=norm2_inset, extent=extent_arcsec,
                     cmap=cm,
                     origin="lower", alpha=1.0, aspect='auto')

        levels_inset = np.geomspace(3.0 * Z2.max(), 5 * std,
                                       8)  # draw contours only until 5xstd level.
        levels_inset_neg = np.asarray([-3 * std])

        csi = axins.contour(Z2, levels=levels_inset[::-1], colors='grey',
                            extent=extent_arcsec,
                      linewidths=1.0, alpha=1.0)
        axins.contour(Z2, levels=levels_inset_neg[::-1], colors='k',
                      extent=extent_arcsec,
                      linewidths=1.0, alpha=1.0)

        axins.clabel(csi, inline=False, fontsize=8, manual=False)


        axins.xaxis.set_ticks(np.linspace(-xoffset_in, xoffset_in, 4))
        axins.yaxis.set_ticks(np.linspace(-yoffset_in, yoffset_in, 4))


        axins.axis('on')
        ax.indicate_inset_zoom(axins)
        axins.grid(True)

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
        el = Ellipse((40, 40), b / pixel_scale, a / pixel_scale, angle=pa,
                     facecolor='r', alpha=0.5)
        ax.add_artist(el)
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