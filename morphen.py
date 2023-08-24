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
__version__ = '0.3.1b'
__codiname__ = 'Pelicoto'
__author__ = 'Geferson Lucatelli'
__email__ = 'geferson.lucatelli@postgrad.manchester.ac.uk'
__date__ = '2023 08 31'
# print(__doc__)


import argparse
import os
import sys
import matplotlib as mpl
sys.path.append('libs/')
import libs as mlibs
import coloredlogs
import logging
from matplotlib import use as mpluse

class config():
    """
    Configuration Class to specify basic parameters.
    """

    def reset_rc_params():
        mpl.rcParams.update({'font.size': 16,
                             "text.usetex": False,  #
                             "font.family": "sans-serif",
                             'mathtext.fontset': 'stix',
                             "font.family": "sans",
                             'font.weight': 'medium',  # medium, semibold, light, 500
                             'font.family': 'STIXGeneral',
                             'xtick.labelsize': 16,
                             'figure.figsize': (6, 4),
                             'ytick.labelsize': 16,
                             'axes.labelsize': 16,
                             'xtick.major.width': 1,
                             'ytick.major.width': 1,
                             'axes.linewidth': 1.5,
                             'axes.edgecolor': 'black',
                             'lines.linewidth': 2,
                             'legend.fontsize': 14,
                             'grid.linestyle': '--',
                             'axes.grid.which': 'major',  # set the grid to appear only on major ticks
                             'axes.grid.axis': 'both',  # set the grid to appear on both the x and y axis
                             'axes.spines.right': False,
                             'axes.grid': True,
                             })
        pass

    reset_rc_params()
    sigma=3
    mask_iterations = 2
    show_plots = True
    ext = '.jpg'
    log_file_name = 'logfile.log'

    if "--noshow" in sys.argv:
        mpluse('Agg')

    def __init__(self):
        print("Initializing Morphen")

# class _logging_():
#     def __init__(self,log_file_name):
#         self.log_file_name = log_file_name.replace('.fits','.log')
#         self.start_log()
#
#
#     def start_log(self):
#         self.logger = logging.getLogger(__name__)
#         # Set the log level
#         self.logger.setLevel(logging.DEBUG)
#         # Fancy format
#         log_format = "%(asctime)s - %(levelname)s - %(message)s"
#         # Use colored logs to add color to the log messages
#         coloredlogs.install(level='DEBUG', logger=self.logger, fmt=log_format)
#         # coloredlogs.install(level='CALC', logger=logger, fmt=log_format)
#         # config.log_file_name = config.file_name('.fits', '.log')
#         file_handler = logging.FileHandler(self.log_file_name)
#         file_handler.setLevel(logging.DEBUG)
#         file_handler.setFormatter(logging.Formatter(log_format))
#         self.logger.addHandler(file_handler)
#         self.logger.info("Initializing Logging!")
#         config.loger_file = True

class _logging_():
    logger = logging.getLogger(__name__)
    # Set the log level
    logger.setLevel(logging.DEBUG)
    # Fancy format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    # Use colored logs to add color to the log messages
    coloredlogs.install(level='DEBUG', logger=logger, fmt=log_format)
    # coloredlogs.install(level='CALC', logger=logger, fmt=log_format)

    try:
        print("# Removing previous log file.")
        os.system(f"rm -r {config.log_file_name}")
    except:
        pass
    file_handler = logging.FileHandler(config.log_file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)


    def __init__(self):
        logger.info("Initializing Logging!")

class read_data():
    """
    Read Input Data
    """
    def __init__(self, filename=None,residualname=None,psfname=None,
                 imagelist = [],residuallist=[],):

        self.filename = filename
        self.residualname = residualname
        self.psfname = psfname
        # self.normalise_in_log()
        self.print_names()

    def print_names(self):
        if self.filename != None:
            print('Image File:', os.path.basename(self.filename))
        if self.residualname != None:
            print('Residual File:', os.path.basename(self.residualname))
        if self.psfname != None:
            print('PSF File:', os.path.basename(self.psfname))


class radio_image_analysis():
    def __init__(self, input_data,z = None,
                 # logger=None,
                 crop=False,box_size=256,
                 apply_mask=True,mask=None,dilation_size = None,
                 sigma_level=3, sigma_mask=6,vmin_factor=3,last_level=1,
                 results=None,mask_component=None,
                 npixels=128,kernel_size=21,fwhm=81,
                 SAVE=True, show_figure=True):
        self.input_data = input_data
        # self.logger = logger
        self.crop = crop
        self.box_size = box_size
        self.apply_mask = apply_mask
        self.dilation_size = dilation_size
        self.sigma_level = sigma_level
        self.sigma_mask = sigma_mask
        self.mask = mask
        self.mask_component = mask_component
        self.vmin_factor = vmin_factor
        self.last_level = last_level
        self.npixels = npixels
        self.kernel_size = kernel_size
        self.fwhm = fwhm
        self.z = z
        self.results = results
        self.SAVE = SAVE
        self.show_figure = show_figure


        self.data_array()
        self.image_properties()

    def data_array(self):
        if self.input_data.filename != None:
            self.data_2D = mlibs.ctn(self.input_data.filename)
        else:
            self.data_2D = None

        if self.input_data.residualname != None:
            self.residual_2D = mlibs.ctn(self.input_data.residualname)
        else:
            self.residual_2D = None

        if self.input_data.psfname != None:
            self.psf_2D = mlibs.ctn(self.input_data.psfname)
        else:
            self.psf_2D = None

    def image_properties(self):
        try:
            self.cell_size = mlibs.get_cell_size(self.input_data.filename)
        except:
            # print('!! WARNING !! Setting cellsize/pixelsize to unity.')
            _logging_.logger.warning("Setting cellsize/pixelsize to unity.")
            self.cell_size = 1

        _logging_.logger.info("Computing image level statistics.")
        # _logging_.file_handler.info("Computing image level statistics.")

        self.image_level_statistics = \
            mlibs.level_statistics(img=self.input_data.filename,
                                   cell_size=self.cell_size, crop=self.crop,
                                   sigma = self.sigma_level,
                                   apply_mask=self.apply_mask,
                                   results=self.results, SAVE=self.SAVE,
                                   ext=config.ext,
                                   show_figure=config.show_plots)

        _logging_.logger.info("Computing image properties.")
        self.levels, self.fluxes, self.agrow, self.plt_image, \
            self.omask, self.mask, self.results_im_props = \
            mlibs.compute_image_properties(img=self.input_data.filename,
                                           cell_size=self.cell_size,
                                           residual=self.input_data.residualname,
                                           sigma_mask=self.sigma_mask,
                                           dilation_size=self.dilation_size,
                                           crop=self.crop,
                                           iterations=config.mask_iterations,
                                           box_size=self.box_size,
                                           last_level=self.last_level,
                                           mask=self.mask,
                                           apply_mask=self.apply_mask,
                                           vmin_factor=self.vmin_factor,
                                           results=self.results,
                                           show_figure=self.show_figure,
                                           logger=_logging_.logger)

        self.img_stats = \
            mlibs.get_image_statistics(imagename=self.input_data.filename,
                                       residual_name=self.input_data.residualname,
                                       cell_size=self.cell_size,
                                       mask_component=None,
                                       mask=self.mask,
                                       region='', dic_data=None,
                                       sigma_mask=self.sigma_mask,
                                       apply_mask=self.apply_mask,
                                       fracX=0.15, fracY=0.15)

        self.image_measures, _ = \
            mlibs.measures(imagename=self.input_data.filename,
                           residualname=self.input_data.residualname,
                           z=self.z,
                           mask_component=self.mask_component,
                           sigma_mask=self.sigma_mask,
                           last_level=self.last_level,
                           vmin_factor=self.vmin_factor,
                           plot_catalog=True, data_2D=self.data_2D,
                           npixels=self.npixels,fwhm=self.fwhm,
                           kernel_size=self.kernel_size,
                           dilation_size=self.dilation_size,
                           main_feature_index=0,
                           results_final={},
                           iterations=config.mask_iterations,
                           fracX=0.15, fracY=0.15,
                           deblend=False, bkg_sub=False,
                           bkg_to_sub=None, rms=None,
                           do_petro=False,
                           apply_mask=self.apply_mask,
                           do_PLOT=True, SAVE=self.SAVE,
                           show_figure=self.show_figure,
                           mask=self.mask,
                           do_measurements='all',
                           compute_A=True,
                           add_save_name='')


class source_extraction():
    """
    Source extraction.
    """
    def __init__(self, input_data,
                 crop=False, box_size=256,
                 apply_mask=False, mask=None, dilation_size = None,
                 sigma_level=3, sigma_mask=6, vmin_factor=3, mask_component=None,
                 bw=128, bh=21, fw=81, fh=81,
                 segmentation_map = True, filter_type='matched',
                 deblend_nthresh=50, deblend_cont=1e-6,
                 clean_param=0.5, clean=True,
                 sort_by='flux',  # sort detected source by flux
                 sigma=15,  # min rms to search for sources
                 ell_size_factor=3.0,  # unstable, please inspect!
                 SAVE=True, show_figure=True):

        self.input_data = input_data
        self.crop = crop
        self.box_size = box_size
        self.apply_mask = apply_mask
        self.mask = mask
        self.dilation_size = dilation_size
        self.sigma_level = sigma_level
        self.sigma_mask = sigma_mask
        self.sigma = sigma
        self.ell_size_factor ell_size_factor
        self.vmin_factor = vmin_factor
        self.mask_component = mask_component
        self.bw = bw
        self.bh = bh
        self.fw = fw
        self.fh = fh
        if self.bw==self.bh==self.fw==self.fh==None:
            self.bspx, self.aO, self.bO = \
                mlibs.get_beam_size_px(self.input_data.filename)
            self.bw = self.bspx / 3
            self.bh = self.bspx / 3
            self.fw = self.bspx / 6
            self.fh = self.bspx / 6
        else:
            self.bw = 128
            self.bh = 21
            self.fw = 81
            self.fh = 81

        try:
            self.minarea = mlibs.beam_area2(self.input_data.filename)
        except:
            self.minarea = 50
        self.bw, self.bh, self.fw, self.fh = bw, bh, fw, fh
        self.segmentation_map = segmentation_map
        self.filter_type = filter_type
        self.deblend_nthresh = deblend_nthresh
        self.deblend_cont = deblend_cont
        self.clean_param = clean_param
        self.clean = clean
        self.sort_by = sort_by
        self.SAVE = SAVE
        self.show_figure = show_figure

    def get_sources(self):
        self.masks, self.indices, self.seg_maps = \
            mlibs.sep_source_ext(self.input_data.filename,
                           bw=self.bw, bh=self.bh, fw=self.fw, fh=self.fh,
                           # filtering options for source detection
                           minarea=self.minarea,
                           segmentation_map=self.segmentation_map,
                           filter_type=self.filter_type, mask=self.mask,
                           deblend_nthresh=self.deblend_nthresh,
                           deblend_cont=self.deblend_cont,
                           clean_param=self.clean_param,
                           clean=self.clean,
                           sort_by=self.sort_by,
                           sigma=self.sigma,
                           ell_size_factor=self.ell_size_factor,
                           apply_mask=self.apply_mask)

class evaluate_source_structure():
    """
    """
    pass


class sersic_multifit():
    """
    Multi-Sersic Fitting Decomposition.

    Perform a semi-automated and robust multi-sersic image decomposition.
    It supports GPU-acceleration using Jax.

    Basic principles:
        - run a source extraction, to identify relevant emission.
        - compute basic properties for each identified region, such as
          size, intensity, shape and orientation
        - uses that information to construct an object and prepare the
          settings to start the fit
        - compute statistics of the fit
        - run an MCMC on model parameters (optional, takes time)
        - if asked, calculates the relative fluxes of each component.

    To-do:
        - automated evaluation of which components is compact (unresolved)
        and which components is extended
        - automated evaluation of which structure cannot be modelled by a
        single function. For example, a spiral galaxy is reconized as a single
        source, but it can not be modelled by a single function: we require to
        model the bulge/bar/disk, for example.

    """
    def __init__(self, names):
        self.names = names
        self.print_test()

    def print_test(self):
        print('Init Sersic')



class morphometry():
    """
    """
    pass


class radio_star_formation():
    """
    Compute star-formation estimates from radio emission, given the converstion
    law.
    """

class make_plots():
    """
    """
    pass

class save_results():
    """
    """
    pass


class wsclean_imaging():
    """
    """
    pass


class casa_imaging():
    """
    """
    pass

class selfcalibration():
    """
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates the curvature\
                        of a given galaxy image.')
    parser.add_argument('-filename',  '--filename',  required=False, \
                        help='The input array of the light profile of the galaxy.')
    parser.add_argument('-residualname',  '--residualname',  required=False, \
                        help='The input array of the light profile of the galaxy.')

    parser.add_argument('-noshow',  '--noshow', required=False,  nargs='?',
                        default=False,  const=True,
                        help='Do not show plots.')
    # parser.add_argument('-filename',  '--filename',  required=False, \
    #                     help='The input array of the light profile of the galaxy.')

    args = parser.parse_args()

    if args.filename != None:
        input_data = read_data(filename=args.filename,
                               residualname=args.residualname,)
        radio_image_analysis(input_data)
        # sersic_multifit(names)
