import matplotlib.pyplot as plt
import numpy as np
import casatasks
from casatasks import *
import casatools
# from casatools import *
from scipy.ndimage import rotate
# import analysisUtils as au
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pf
from casatools import image as IA
# import lmfit
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

from scipy.optimize import leastsq, fmin, curve_fit
import scipy.ndimage as nd
import scipy
from scipy.stats import circmean, circstd
from scipy.signal import savgol_filter

import sys

# sys.path.append('../../scripts/analysis_scripts/')
sys.path.append('analysis_scripts/')
sys.path.append('/opt/casa-6.5.1-23-py3.8/')
import analysisUtils as au
from analysisUtils import *

# from polarTransform.imageTransform import ImageTransform
# from polarTransform.pointsConversion import *


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

    Params:
        x0,y0: center position
        a: amplitude
        fwhm: full width at half max
        q: axis ratio, q = b/a; e = 1 -q
        c: geometric parameter that controls how boxy the ellipse is
        PA: position angle of the meshgrid
        size: size of the 2D image data array
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
    return ((x - x0) * np.cos(t) + (y - y0) * np.sin(t), -(x - x0) * np.sin(t) + \
            (y - y0) * np.cos(t))


def bn(n):
    """
    bn function from Cioti .... (1997);
    Used to define the relation between Rn (half-light radii) and total luminosity

    Parameters:
        n: sersic index
    """

    return 2. * n - 1. / 3. + 0 * ((4. / 405.) * n) + ((46. / 25515.) * n ** 2.0)


def sersic2D(xy, x0, y0, PA, ell, n, In, Rn):
    q = 1 - ell
    x, y = xy
    # x,y   = np.meshgrid(np.arange((size[1])),np.arange((size[0])))
    xx, yy = rotation(PA, x0, y0, x, y)
    # r     = (abs(xx)**(c+2.0)+((abs(yy))/(q))**(c+2.0))**(1.0/(c+2.0))
    r = np.sqrt((abs(xx) ** (2.0) + ((abs(yy)) / (q)) ** (2.0)))
    model = In * np.exp(-bn(n) * ((r / (Rn)) ** (1.0 / n) - 1.))
    return (model)


def FlatSky(data_level, a):
    return (a * data_level)


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
        Read a CASA format image file and return as a numpy array.
        Also works with wsclean images!
        Note: For some reason, casa returns a rotated mirroed array, so we need to undo it by a rotation. 
        '''
    ia = IA()
    ia.open(image)
    try:
        try:
            numpy_array = ia.getchunk()[:, :, 0, 0]
        except:
            numpy_array = ia.getchunk()[:, :]
    except:
        pass
    ia.close()
    # casa gives a mirroed and 90-degree rotated image :(
    return (np.rot90(numpy_array)[::-1, ::])


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
    Computes the estimated projected beam area (theroetical), given the semi-major and minor axis
    and the cell size used during cleaning. 
    Return the beam area in pixels. 
    '''
    BArea = ((np.pi * Omaj * Omin) / (4 * np.log(2))) / (cellsize ** 2.0)
    return (BArea)


def beam_area2(image, cellsize):
    '''
    Computes the estimated projected beam area (theroetical), given the semi-major and minor axis
    and the cell size used during cleaning. 
    Return the beam area in pixels. 
    '''
    imhd = imhead(image)
    Omaj = imhd['restoringbeam']['major']['value']
    Omin = imhd['restoringbeam']['minor']['value']
    BArea = ((np.pi * Omaj * Omin) / (4 * np.log(2))) / (cellsize ** 2.0)
    return (BArea)


def cut_image(img, center=None, size=(1024, 1024), cutout_filename=None, special_name=''):
    """
    Cut images keeping updated header/wcs.
    This function is a helper to cut both image and its associated residual (from casa or wsclean).
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
        cutout_filename = img.replace('-image.fits', '-image_cutout' + special_name + '.fits')
    hdu.writeto(cutout_filename, overwrite=True)

    # do the same for the residual image
    hdu2 = pf.open(img)[0]
    wcs2 = WCS(hdu2.header, naxis=2)

    hdu_res = pf.open(img.replace('-image.fits', '-residual.fits'))[0]
    # plt.imshow(hdu_res.data[0][0])
    wcs_res = WCS(hdu_res.header, naxis=2)
    cutout_res = Cutout2D(hdu_res.data[0][0], position=position, size=size, wcs=wcs2)
    hdu2.data = cutout_res.data
    hdu2.header.update(cutout_res.wcs.to_header())
    # if cutout_filename == None:
    cutout_filename_res = img.replace('-image.fits', '-residual_cutout' + special_name + '.fits')
    hdu2.writeto(cutout_filename_res, overwrite=True)
    return(cutout_filename,cutout_filename_res)


def do_cutout(image, box_size=300, center=None, return_='data'):
    """
    Fast cutout of a numpy array.
    
    Returs: numpy data array or a box for that cutout, if asked.
    """
    imhd = imhead(image)
    if center == None:
        st = imstat(image)
        print('  >> Center --> ', st['maxpos'])
        xin, xen, yin, yen = st['maxpos'][0] - box_size, st['maxpos'][0] + box_size, st['maxpos'][1] - box_size, \
                             st['maxpos'][1] + box_size
    else:
        xin, xen, yin, yen = center[0] - box_size, center[0] + box_size, center[1] - box_size, center[1] + box_size
    if return_ == 'data':
        data_cutout = ctn(image)[xin:xen, yin:yen]
        return (data_cutout)
    if return_ == 'box':
        box = xin, xen, yin, yen  # [xin:xen,yin:yen]
        return (box)


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
    hdu = pf.open(image)[0]
    wcs = WCS(hdu.header, naxis=2)
    image_to_copy_data = pf.getdata(image_to_copy)
    hdu.data = image_to_copy_data
    if file_to_save == None:
        """
        This is just a safe fallback, you would want to overwrite the
        existing file, therefore set image_to_copy=file_to_save 
        when calling this function.
        """
        file_to_save = image_to_copy.replace('.fits', 'header.fits')
    hdu.writeto(file_to_save, overwrite=True)

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
    """
    ihl = imhead(imagename, mode='list')
    ih = imhead(imagename)

    M = ihl['shape'][0]
    N = ihl['shape'][1]
    #     frac_X = int(frac*M)
    #     frac_Y = int(frac*N)
    #     slice_pos_X = 0.15 * M
    #     slice_pos_Y = 0.85 * N
    frac_X = int(fracX * M)
    frac_Y = int(fracY * N)
    slice_pos_X = int(0.02 * M)
    slice_pos_Y = int(0.98 * N)

    #     cut = [[slice_pos_X,slice_pos_X+frac_X],
    #            [slice_pos_Y-frac_Y,slice_pos_Y]
    #           ]
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
    """
    ihl = imhead(imagename, mode='list')
    ih = imhead(imagename)

    M = ihl['shape'][0]
    N = ihl['shape'][1]
    #     frac_X = int(fracX*M)
    #     frac_Y = int(fracY*N)
    #     slice_pos_X = int(0.20 * M)
    #     slice_pos_Y = int(0.80 * N)

    #     cut = [[slice_pos_X - frac_X,slice_pos_X+frac_X],
    #            [slice_pos_Y - frac_Y,slice_pos_Y+frac_Y]
    #           ]
    frac_X = int(fracX * M)
    frac_Y = int(fracY * N)
    slice_pos_X = int(0.02 * M)
    slice_pos_Y = int(0.98 * N)

    cut = [[slice_pos_X, slice_pos_X + frac_X],
           [slice_pos_Y - frac_Y, slice_pos_Y]
           ]

    return (cut)
