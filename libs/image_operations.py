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


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from astropy import units as u
import astropy.io.fits as pf
from casatools import image as IA
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import scipy.ndimage as nd

from read_data import *

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


def do_cutout_2D(image_data, box_size=(300,300), center=None, return_='data'):
    """
    Fast cutout of a numpy array.

    Returs: numpy data array or a box for that cutout, if asked.
    """

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
    if file_to_save == None:
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