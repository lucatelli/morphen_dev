"""
                                                          ..___|**_
                                                  .|||||||||*+@+*__*++.
                                              _||||.           .*+;].,#_
                                         _|||*_                _    .@@@#@.
                                   _|||||_               .@##@#| _||_
         Morphen              |****_                   .@.,/\..@_.
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
__version__ = '0.0.1-beta'
__author__  = 'Geferson Lucatelli'
__email__   = ''
__date__    = '2023 02'

print(__doc__)

import numpy as np
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
import scipy
from scipy import signal
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import time
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.stats import mad_std
import lmfit
from astropy import visualization
from astropy.visualization import simple_norm
import scipy.ndimage as nd
import pickle
import pandas as pd

from petrofit.photometry import make_radius_list
from petrofit.petrosian import Petrosian
from petrofit.photometry import source_photometry
from petrofit.segmentation import make_catalog, plot_segments
from petrofit.segmentation import plot_segment_residual
from petrofit.photometry import order_cat

import sep
import fitsio

"""
             ____            _   _               _ 
            / ___|__ _ _   _| |_(_) ___  _ __   | |
           | |   / _` | | | | __| |/ _ \| '_ \  | |
           | |__| (_| | |_| | |_| | (_) | | | | |_|
            \____\__,_|\__,_|\__|_|\___/|_| |_| (_)

 _____                      _                      _        _ 
| ____|_  ___ __   ___ _ __(_)_ __ ___   ___ _ __ | |_ __ _| |
|  _| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ _` | |
| |___ >  <| |_) |  __/ |  | | | | | | |  __/ | | | || (_| | |
|_____/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__\__,_|_|
           |_|                                                
 ____                 _                                  _   
|  _ \  _____   _____| | ___  _ __  _ __ ___   ___ _ __ | |_ 
| | | |/ _ \ \ / / _ \ |/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __|
| |_| |  __/\ V /  __/ | (_) | |_) | | | | | |  __/ | | | |_ 
|____/ \___| \_/ \___|_|\___/| .__/|_| |_| |_|\___|_| |_|\__|
                             |_|                             
To do:
    - Needs lots of documenting, 
    - importing issues (I am using this code on a 
        larger collection of scripts that I run on jupyter).
    -  good model and fast convergence if config input file is well constructed. 
        So, make this code more general, establish good initial values 
        automatically
                             
"""


"""
 _   _ _   _ _     
| | | | |_(_) |___ 
| | | | __| | / __|
| |_| | |_| | \__ \
 \___/ \__|_|_|___/

"""
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
            try:
                numpy_array = ia.getchunk()[:, :, 0, 0]
            except:
                numpy_array = ia.getchunk()[:, :]
        except:
            pass
        ia.close()
        # casa gives a mirroed and 90-degree rotated image :(
        data_image = np.rot90(numpy_array)[::-1, ::]
    except:
        try:
            data_image = pf.getdata(image)
        except:
            print('Error loading fits file')
            return(ValueError)
    return(data_image)


def save_results_csv(result_mini, save_name, ext='.csv', save_corr=True,
                     save_params=True):
    values = result_mini.params.valuesdict()
    if save_corr:
        covariance = result_mini.covar
        covar_df = pd.DataFrame(covariance, index=values.keys(),
                                columns=values.keys())
        covar_df.to_csv(save_name + '_mini_corr' + ext, index_label='parameter')

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
        
        Assume that we have a blob surrounded by a disky emission (though detected 
        as being one source). Both are placed on the same region, on top of each other 
        (e.g. from optical, we call a bulge and 
        a disk). We need two functions to model this region. 
        
        So, if component i=1 is the blob (or the bulge) we copy the parameters from it and 
        create a second component. We just have to ajust some of the parameters. 
        E.g. the effective radius of this new component, is in principle, larger than the original component. 
        As well, the effective intensity will be smaller because we are adding a component 
        further away from the centre. Other quantities, however, are uncertain, such as the Sersic index, position angle 
        etc, but may be close to those of component i. 
        
    """
    
    from collections import OrderedDict
    dict_keys = list(petro_properties.keys())
    unique_list = list(OrderedDict.fromkeys([elem.split('_')[1] for \
        elem in dict_keys if '_' in elem]))
#     print(unique_list)

    petro_properties_copy = petro_properties.copy()
    new_comp_id = sources_photometies['ncomps'] + 1
    for k in range(len(unique_list)):
#         print(unique_list[k])
        # do not change anything for other parameters. 
        petro_properties_copy['c'+str(new_comp_id)+'_'+unique_list[k]] = \
            petro_properties_copy['c'+str(copy_from_id)+'_'+unique_list[k]]
        if unique_list[k] == 'R50':
            # multiply the R50 value by a factor, e.g., 1.5
            factor = 2
            petro_properties_copy['c'+str(new_comp_id)+'_'+unique_list[k]] = \
                petro_properties_copy['c'+str(copy_from_id)+'_'+unique_list[k]]*factor
        if unique_list[k] == 'I50':
            # divide the I50 value by a factor, e.g., 2
            factor = 0.5
            petro_properties_copy['c'+str(new_comp_id)+'_'+unique_list[k]] = \
                petro_properties_copy['c'+str(copy_from_id)+'_'+unique_list[k]]*factor
    #update number of components
    petro_properties_copy['ncomps'] = petro_properties_copy['ncomps'] + 1
    return(petro_properties_copy)

"""
 __  __       _   _         
|  \/  | __ _| |_| |__  ___ 
| |\/| |/ _` | __| '_ \/ __|
| |  | | (_| | |_| | | \__ \
|_|  |_|\__,_|\__|_| |_|___/
"""

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
 __  __       _ _   _      ____                _      
|  \/  |_   _| | |_(_)    / ___|  ___ _ __ ___(_) ___ 
| |\/| | | | | | __| |____\___ \ / _ \ '__/ __| |/ __|
| |  | | |_| | | |_| |_____|__) |  __/ |  \__ \ | (__ 
|_|  |_|\__,_|_|\__|_|    |____/ \___|_|  |___/_|\___|

"""


def setup_model_components(n_components=2):
    """
        Set up a single sersic component or a composition of sersic components.

        Uses the LMFIT objects to easilly create model components.

        fi_ is just a prefix to distinguish parameter names.

    """
    if n_components == 1:
        smodel2D = Model(sersic2D, prefix='f1_') + Model(FlatSky, prefix='s_')
    if n_components > 1:
        smodel2D = Model(sersic2D, prefix='f1_')
        for i in range(2, n_components + 1):
            smodel2D = smodel2D + Model(sersic2D, prefix='f' + str(i) + '_')
        smodel2D = smodel2D + Model(FlatSky, prefix='s_')
    return (smodel2D)


def construct_model_parameters(n_components=None, params_values_init=None,
                               init_constraints=None,
                               constrained=True, fix_n=False, fix_x0_y0=False,
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
    dr_fix = 2

    # params_values_init = [] #grid of parameter values, each row is the
    # parameter values of a individual component

    if params_values_init is not None:
        """This takes the values from config file as init params and set 
        number of components.
        """
        for i in range(0, n_components):
            x0, y0, PA, ell, n, In, Rn = params_values_init[i]
            if fix_x0_y0 is not False:
                fix_x0_y0_i = fix_x0_y0[i]
            else:
                fix_x0_y0_i = False
            
            if fix_n is not False:
                fix_n_i = fix_n[i]
            
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
                                max=4.0)
                    if param == 'x0':
                        if fix_x0_y0_i is not False:
                            """
                            Fix centre position by no more than dr_fix.
                            """
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=eval(param),
                                min=eval(param) - dr_fix,
                                max=eval(param) + dr_fix)
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
                            Fix centre position by no more than dr_fix.
                            """
                            smodel2D.set_param_hint(
                                'f' + str(i + 1) + '_' + param,
                                value=eval(param),
                                min=eval(param) - dr_fix,
                                max=eval(param) + dr_fix)
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
                                                value=0.5, min=0.3, max=5)
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

        smodel2D.set_param_hint('s_a', value=10.0, min=0.1, max=50.0)
    else:
        if init_constraints is not None:
            if constrained == True:
                for j in range(init_constraints['ncomps']):
                    if fix_n is not False:
                        fix_n_j = fix_n[j]
                    jj = str(j + 1)
                    for param in model_temp.param_names:

                        #                         smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                        #                                                 value=eval(param), min=0.000001)
                        if (param == 'n'):
                            if (fix_n_j == True):
                                print('Fixing Sersic Index of component',j,' to 0.5.')
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=0.5, min=0.49, max=0.51)
                            else:
                                smodel2D.set_param_hint(
                                    'f' + str(j + 1) + '_' + param,
                                    value=0.5, min=0.3, max=4.0)

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
                            dO = 60
                            _PA = init_constraints['c' + jj + '_PA']
                            PA_max = _PA + dO
                            PA_min = _PA - dO
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=_PA, min=PA_min, max=PA_max)
                        if param == 'ell':
                            dell = 0.2
                            ell = 1 - init_constraints['c' + jj + '_q']
                            ell_min = ell * 0.5
                            #                         if ell + dell <= 1.0:
                            if ell * 2.0 <= 1.0:
                                ell_max = ell * 2.0
                            else:
                                ell_max = 1.0
                            #                         if ell - dell >= 0.0:

                            #                         else:
                            #                             ell_min = 0.0

                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=ell, min=ell_min, max=ell_max)

                        if param == 'In':
                            I50 = init_constraints['c' + jj + '_I50']
                            I50_max = I50 * 100
                            I50_min = I50 * 0.1
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=I50_max, min=I50_min, max=I50_max)
                        if param == 'Rn':
                            R50 = init_constraints['c' + jj + '_R50']
                            dR = R50 * 0.75
                            R50_max = R50 * 1.2
                            R50_min = R50 * 0.1
                            smodel2D.set_param_hint(
                                'f' + str(j + 1) + '_' + param,
                                value=R50_min, min=R50_min, max=R50_max)

                        if param == 'x0':
                            ddxx = 5
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
                            ddyy = 5
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
                                value=0.5, min=0.3, max=5)
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

            smodel2D.set_param_hint('s_a', value=10.0, min=0.1, max=50.0)
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
                smodel2D.set_param_hint('s_a', value=10.0, min=0.1, max=50.0)
            except:
                print('Please, if not providing initial parameters file,')
                print('provide basic information for the source.')
                return (ValueError)

    params = smodel2D.make_params()
    print(smodel2D.param_hints)
    return (smodel2D, params)


def do_fit2D(imagename, params_values_init=None, ncomponents=None,
             init_constraints=None, data_2D_=None, residualname=None,
             init_params=0.25, final_params=4.0, constrained=True, fix_n=True,
             fix_x0_y0=False,
             special_name='', method1='nelder', method2='least_squares',
             save_name_append=''):
    startTime = time.time()

    if data_2D_ is None:
        data_2D = pf.getdata(imagename)
    else:
        data_2D = data_2D_

    if residualname is not None:
        residual_2D = pf.getdata(residualname)
        residual_2D_shuffled = shuffle_2D(residual_2D)
        print('Using clean background for optmization...')
        #         background = residual_2D #residual_2D_shuffled
        background = residual_2D_shuffled

    else:
        print('No residual/background provided. Using image bkg map...')
        background_map = sep_background(imagename)
        background = background_map.back()

    PSF_CONV = True
    PSF_BEAM = pf.getdata(
        imagename.replace('-image.cutout.fits', '-beampsf.cutout.fits'))
    size = data_2D.shape
    xy = np.meshgrid(np.arange((size[1])), np.arange((size[0])))
    #     FlatSky_level = background#mad_std(data_2D)
    FlatSky_level = mad_std(data_2D)
    nfunctions = ncomponents

    def residual_2D(params):
        dict_model = {}
        model = 0
        for i in range(1, nfunctions + 1):
            model = model + sersic2D(xy, params['f' + str(i) + '_x0'],
                                     params['f' + str(i) + '_y0'],
                                     params['f' + str(i) + '_PA'],
                                     params['f' + str(i) + '_ell'],
                                     params['f' + str(i) + '_n'],
                                     params['f' + str(i) + '_In'],
                                     params['f' + str(i) + '_Rn'])
        # print(model.shape)
        model = model + FlatSky(FlatSky_level, params['s_a']) + background
        if PSF_CONV == True:
            MODEL_2D_conv = scipy.signal.fftconvolve(model, PSF_BEAM, 'same')
        else:
            MODEL_2D_conv = model  # + background

        residual = data_2D - MODEL_2D_conv  # - FlatSky(FlatSky_level, params['s_a'])
        return (residual)

    smodel2D, params = construct_model_parameters(
        params_values_init=params_values_init, n_components=nfunctions,
        init_constraints=init_constraints,
        fix_n=fix_n, fix_x0_y0=fix_x0_y0,
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
                                 mutation=(0.5, 1.5), recombination=[0.2, 0.9],
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

    model_temp = Model(sersic2D)
    model = 0
    size = ctn(crop_image).shape
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
                              params['f' + str(
                                  i) + '_Rn']) + \
                     background / ncomponents + \
                     FlatSky(FlatSky_level, params['s_a']) / ncomponents
        #                                  params['f'+str(i)+'_Rn'])+FlatSky(FlatSky_level, params['s_a'])/ncomponents
        # print(model_temp[0])
        model = model + model_temp
        # print(model)
        model_dict['model_c' + str(i)] = model_temp

        if PSF_CONV == True:
            model_dict['model_c' + str(i) + '_conv'] = scipy.signal.fftconvolve(
                model_temp, PSF_BEAM,'same')
        else:
            model_dict['model_c' + str(i) + '_conv'] = model_temp

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
    model_dict['model_total'] = model  # + FlatSky(FlatSky_level, params['s_a'])

    if PSF_CONV == True:
        model_dict['model_total_conv'] = scipy.signal.fftconvolve(model,
                                                                  PSF_BEAM,
                                                                  'same')  # + FlatSky(FlatSky_level, params['s_a'])
    else:
        model_dict['model_total_conv'] = model

    model_dict['best_residual'] = data_2D - model_dict['model_total']
    model_dict['best_residual_conv'] = data_2D - model_dict['model_total_conv']

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
    method1 = 'differential_evolution'
    print(' >> Using', method1, ' solver for first optimisation run... ')
    # take parameters from previous run, and re-optimize them.
    #     method2 = 'ampgo'#'least_squares'
    method2 = 'least_squares'

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
    save_results_csv(result_mini=result,
                     save_name=image_results_conv[-2].replace('.fits', ''),
                     ext='.csv',
                     save_corr=True, save_params=True)

    return (result, mini, result_1, result_extra, model_dict, image_results_conv,
            image_results_deconv)

def do_cutout_2D(image_data, box_size=300, center=None, return_='data'):
    """
    Fast cutout of a numpy array.

    Returs: numpy data array or a box for that cutout, if asked.
    """

    if center is None:
        x0, y0= nd.maximum_position(ctn(image_data))
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

def sep_background(imagename,mask=None,apply_mask=True,
                   bw=64, bh=64, fw=5, fh=5):
    '''
    If using astropy.io.fits, you get an error (see bug on sep`s page).
    '''
    data_2D = fitsio.read(imagename)

    if (mask is None) and (apply_mask==True):
        _, mask = mask_dilation(imagename, PLOT=False,
                                sigma=3, iterations=2, dilation_size=10)
        bkg = sep.Background(data_2D, mask=mask, bw=12, bh=12, fw=5, fh=5)

    else:
        bkg = sep.Background(data_2D, mask=mask,
                             bw=bw, bh=bh, fw=fw, fh=fh)
    bkg_rms = bkg.rms()
    bkg_image = bkg.back()
    return(bkg)


"""
 ____      _                 _             
|  _ \ ___| |_ _ __ ___  ___(_) __ _ _ __  
| |_) / _ \ __| '__/ _ \/ __| |/ _` | '_ \ 
|  __/  __/ |_| | | (_) \__ \ | (_| | | | |
|_|   \___|\__|_|  \___/|___/_|\__,_|_| |_|

"""
def petro_cat(data_2D, fwhm=24, npixels=None, kernel_size=15,
              nlevels=30, contrast=0.001,
              sigma_level=20, vmin=5,
              deblend=True, plot=True):
    """
    Use PetroFit class to create catalogues.
    """
    cat, segm, segm_deblend = make_catalog(
        image=data_2D,
        threshold=20.0 * mad_std(data_2D),
        deblend=False,
        kernel_size=kernel_size,
        fwhm=fwhm,
        npixels=npixels,
        plot=plot, vmax=data_2D.max(), vmin=vmin * mad_std(data_2D)
    )

    sorted_idx_list = order_cat(cat, key='area', reverse=True)
    #     idx = sorted_idx_list[main_feature_index]  # index 0 is largest
    #     source = cat[idx]  # get source from the catalog
    return (cat, sorted_idx_list)


def petro_params(source, data_2D, segm, i='1', petro_properties={},
                 rlast=None, sigma=3, vmin=3, bkg_sub=False, plot=True):
    if rlast is None:
        rlast = int(2 * np.sqrt((np.sum(mask_source) / np.pi)))
    else:
        rlast = rlast

    r_list = make_radius_list(
        max_pix=rlast,  # Max pixel to go up to
        n=int(rlast)  # the number of radii to produce
    )
    cutout_size = 2 * max(r_list)
    flux_arr, area_arr, error_arr = source_photometry(source, data_2D, segm,
                                                      r_list, cutout_size=cutout_size,
                                                      bkg_sub=bkg_sub, sigma=3, sigma_type='clip',
                                                      plot=plot, vmax=0.3 * data_2D.max(),
                                                      vmin=vmin * mad_std(data_2D)
                                                      )
    #     fast_plot2(mask_source * data_2D)
    p = Petrosian(r_list, area_arr, flux_arr)
    R50 = p.r_half_light
    Snu = p.total_flux
    Rp = p.r_petrosian
    Rpidx = int(2 * Rp)
    petro_properties['c' + i + '_R50'] = R50
    petro_properties['c' + i + '_Snu'] = Snu
    petro_properties['c' + i + '_Rp'] = Rp
    petro_properties['c' + i + '_Rpidx'] = Rpidx
    petro_properties['c' + i + '_rlast'] = rlast
    plt.figure()
    p.plot(plot_r=True)
    #     print('    R50 =', R50)
    #     print('     Rp =', Rp)
    return (petro_properties)


def source_props(data_2D, source_props={}):
    '''
    From a 2D image array, perform simple source extraction, and calculate basic petrosian
    properties.
    '''
    cat, sorted_idx_list = petro_cat(data_2D, fwhm=24, npixels=None, kernel_size=15,
                                     nlevels=30, contrast=0.001,
                                     sigma_level=20, vmin=5,
                                     deblend=True, plot=True)
    #     i = 0
    for i in range(len(sorted_idx_list)):
        ii = str(i + 1)
        seg_image = cat[sorted_idx_list[i]]._segment_img.data
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

        source_props = petro_params(source=source, data_2D=data_2D, segm=segm,
                                    i=ii, petro_properties=source_props,
                                    rlast=None, sigma=3, vmin=3, bkg_sub=False,
                                    plot=False)

        #         print(Rp_props['rlast'],2*Rp_props['Rp'])
        if source_props['c' + ii + '_rlast'] < 2 * source_props['c' + ii + '_Rp']:
            Rlast_new = 2 * source_props['c' + ii + '_Rp'] + 3
            source_props = petro_params(source=source, data_2D=data_2D,
                                        segm=segm, i=ii,
                                        petro_properties=source_props,
                                        rlast=Rlast_new, sigma=3, vmin=3,
                                        bkg_sub=False, plot=False)
        r, ir = get_profile(data_2D * mask_source, binsize=1.0)
        I50 = ir[int(source_props['c' + ii + '_R50'])]
        source_props['c' + ii + '_I50'] = I50

    source_props['ncomps'] = len(sorted_idx_list)
    return (source_props)

def fast_plot3(imagename, modelname, residualname, reference_image, crop=False,
               box_size=512, NAME=None,
               max_percent_lowlevel=99.0, max_percent_highlevel=99.9999,
               ext='.pdf'):
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
        xin, xen, yin, yen = do_cutout_2D(reference_image, box_size=box_size,
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
    vmin = 0.01 * std  # 0.5*g.min()#
    vmax = 1.0 * g.max()
    vmin_r = -0.5 * vmin  # 1.0*r.min()#1*std_r
    vmax_r = 1.0 * r.max()
    vmin_m = vmin  # 1*mad_std(m)#vmin#0.01*std_m#0.5*m.min()#
    vmax_m = vmax  # 0.5*m.max()#vmax#0.5*m.max()

    levels_g = np.geomspace(g.max(), 3 * std, 7)
    levels_colorbar2 = np.geomspace(g.max(), 3 * std, 5)
    levels_m = np.geomspace(m.max(), 10 * std_m, 7)
    levels_r = np.geomspace(r.max(), 3 * std_r, 7)

    #     norm = simple_norm(g,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    norm = visualization.simple_norm(g, stretch='linear',
                                     max_percent=max_percent_lowlevel)
    norm2 = simple_norm(abs(g), min_cut=vmin, max_cut=vmax,
                        stretch='asinh', asinh_a=0.005)  # , max_percent=max_percent_highlevel)
    CM = 'magma_r'
    ax = fig.add_subplot(1, 3, 1)

    #     im = ax.imshow(g, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(g, cmap=CM, origin='lower', alpha=1.0, norm=norm2)

    ax.set_title(r'Image')

    ax.contour(g, levels=levels_g[::-1], colors='#009E73', linewidths=0.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    cb = plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes(
        [-0.0, 0.40, 0.02,0.19]))
#     cax = fig.add_axes([-0.0, 0.40, 0.02,0.19])

# #         cbar = fig.colorbar(im, cax=cax, orientation='vertical', format='%.2e',
# #                             shrink=1, aspect='auto', pad=10, fraction=1.0,
# #                             drawedges=False, ticklocation='left')
#     cb = fig.colorbar(g, cax=cax, orientation='vertical',
#                         shrink=1, aspect='auto', pad=10, fraction=1.0,
#                         drawedges=False, ticklocation='left')
#     cb.formatter = CustomFormatter(factor=1000, useMathText=True)
    cb.update_ticks()


#         cbar.update_ticks()
    # cbar_axes = [0.12, 0.95, 0.78, 0.06]
    # cax = fig.add_axes(cbar_axes)
    #
    # cbar = fig.colorbar(im, cax=cax, orientation='horizontal', format='%.0e',
    #                     shrink=1.0, aspect='auto', pad=0.1, fraction=1.0, drawedges=False
    #                     )
    cb.ax.yaxis.set_tick_params(labelleft=True, labelright=False, tick1On=False, tick2On=False)
    cb.ax.yaxis.tick_left()
    cb.set_ticks(levels_colorbar2)
    cb.set_label(r'Flux [Count/pixel]', labelpad=1)
    cb.ax.xaxis.set_tick_params(pad=1)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(1)
    # cb.dividers.set_color('none')

    ax = plt.subplot(1, 3, 2)

    #     im_plot = ax.imshow(m, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax_m, vmin=vmin_m)#norm=norm
    norm_mod = simple_norm(m, min_cut=vmin_m, max_cut=vmax_m,
                           stretch='asinh', asinh_a=0.005)  # , max_percent=max_percent_highlevel)

    im_plot = ax.imshow(m, cmap=CM, origin='lower', alpha=1.0,
                        norm=norm_mod)
    ax.set_title(r'Model')
    ax.contour(m, levels=levels_g[::-1], colors='#009E73', linewidths=0.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.08,0.3,0.02,0.4]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    ax = plt.subplot(1, 3, 3)
    norm_re = simple_norm(r, min_cut=vmin_r, max_cut=vmax_r,
                          stretch='asinh', asinh_a=0.005)  # , max_percent=max_percent_highlevel)
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
    cb1.set_label(r'Intensity [count/pixel]', labelpad=1)
    cb1.ax.xaxis.set_tick_params(pad=1)
    cb1.ax.tick_params(labelsize=12)
    cb1.outline.set_linewidth(1)
    # cb1.dividers.set_color('none')
    if NAME != None:
        plt.savefig(NAME + ext, dpi=300, bbox_inches='tight')


def get_peak_pos(imagename):
    image_data = pf.getdata(imagename)
    maxpos = np.where(image_data == image_data.max())
    print('Peak Pos=', maxpos)
    return (maxpos)

def get_profile(imagename, center=None):
    if center is None:
        nr, radius, profile = azimuthalAverage(ctn(imagename),return_nr = True,binsize=1)
    else:
        nr, radius, profile = azimuthalAverage(ctn(imagename), return_nr=True, binsize=1,center=center)
    return (radius, profile)

def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False,
                     binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None,
                     mask=None):
    """
    *** THIS FUNCTION WAS TAKEN FROM HERE
    https://github.com/keflavich/image_tools/blob/master/image_tools/radialprofile.py
    ***

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
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

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


"""
This function accepts IMFIT config input files.


Usage:

    crop_image = # name of image data
    psf_name = # name of psf image data

    data_2D = pf.getdata(crop_image)
    config_file = # name of imfit config file 
    
    PSF_BEAM = pf.getdata(psf_name)
    
    imfit_conf_values = read_imfit_params(config_file)
 
    n_components = len(imfit_conf_values)-1
    construct_model_parameters(imfit_conf_values[0:-1],n_components=n_components,
        init_params = 0.25,final_params = 4.0,constrained=True)
    
    result_mini, mini,model_dict = do_fit2D(imagename=crop_image,
                                        params_values_init = imfit_conf_values[0:-1],
                                        ncomponents=n_components,constrained=True,
                                        init_params = 0.25,final_params = 4.0)
"""


"""
## Experiments with Bayesian Inference
## DO NOT USE FOR NOW!
import numpy as np

# try:
#     np.__config__.blas_opt_info = np.__config__.blas_ilp64_opt_info
# except Exception:
#     pass

import pymc3 as pm
import theano.tensor as tt



def two_component_sersic(params, x, y):
    n1, Re1, Ie1, xc1, yc1, q1, PA1, n2, Re2, Ie2, xc2, yc2, q2, PA2 = params
    bn1 = 2 * n1 - 1 / 3
    bn2 = 2 * n2 - 1 / 3
    xx1, yy1 = rotation(PA1, xc1, yc1, x, y)
    r1 = np.sqrt((xx1) ** 2 + ((yy1) / q1) ** 2)
    I1 = Ie1 * np.exp(-bn1 * (np.power(r1 / Re1, 1. / n1)) - 1)
    xx2, yy2 = rotation(PA2, xc2, yc2, x, y)
    r2 = np.sqrt((xx2) ** 2 + ((yy2) / q2) ** 2)
    I2 = Ie2 * np.exp(-bn2 * (np.power(r2 / Re2, 1. / n2)) - 1)
    return I1 + I2


def fit_two_component_sersic(image, x, y):
    with pm.Model() as model:
        n1 = pm.Uniform('n1', lower=0.3, upper=2.0)
        Re1 = pm.Uniform('Re1', lower=2.1, upper=20)
        Ie1 = pm.Uniform('Ie1', lower=0.0001, upper=0.5)
        xc1 = pm.Uniform('xc1', lower=80, upper=120)
        yc1 = pm.Uniform('yc1', lower=80, upper=120)
        q1 = pm.Uniform('q1', lower=0.001, upper=0.99)
        PA1 = pm.Uniform('PA1', lower=-0.001, upper=359.99)
        n2 = pm.Uniform('n2', lower=0.3, upper=2)
        Re2 = pm.Uniform('Re2', lower=15.1, upper=90)
        Ie2 = pm.Uniform('Ie2', lower=0.00001, upper=0.1)
        xc2 = pm.Uniform('xc2', lower=80, upper=120)
        yc2 = pm.Uniform('yc2', lower=80, upper=120)
        q2 = pm.Uniform('q2', lower=0.001, upper=0.99)
        PA2 = pm.Uniform('PA2', lower=-0.001, upper=359.99)

        params = pm.math.stack(
            [n1, Re1, Ie1, xc1, yc1, q1, PA1, n2, Re2, Ie2, xc2, yc2, q2, PA2])
        model_image = two_component_sersic(params, x, y)
        sigma = pm.HalfCauchy('sigma', beta=10)
        likelihood = pm.Normal('likelihood', mu=model_image, sigma=sigma,
                               observed=image)

        trace = pm.sample(400, tune=300, chains=8)

    best_params = pm.summary(trace)['mean']
    best_params_error = pm.summary(trace)['sd']

    return best_params, best_params_error, trace


def plot_posteriors(trace):
    pm.plot_posterior(trace,
                      varnames=['n1', 'Re1', 'Ie1', 'xc1', 'yc1', 'q1', 'PA1',
                                'n2', 'Re2', 'Ie2', 'xc2', 'yc2', 'q2', 'PA2'])
    plt.show()

image = pf.getdata(imagename)# 2D image data
y, x = np.indices(image.shape)
best_params, best_params_error, trace = fit_two_component_sersic(image, x, y)
# plot_posteriors(trace)

"""