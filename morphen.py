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
sys.path.append('analysis_scripts/')
import libs as mlibs
import coloredlogs
import logging
from matplotlib import use as mpluse
import analysisUtils as au
from analysisUtils import *

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
                             'font.weight': 'medium',  
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
                             'axes.grid.which': 'major',  
                             'axes.grid.axis': 'both', 
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
        self.get_data()

    def print_names(self):
        if self.filename != None:
            print('Image File:', os.path.basename(self.filename))
        if self.residualname != None:
            print('Residual File:', os.path.basename(self.residualname))
        if self.psfname != None:
            print('PSF File:', os.path.basename(self.psfname))

    def get_data(self):
        self.image_data_2D = None
        self.residual_data_2D = None
        self.psf_data_2D = None
        if self.filename != None:
            self.image_data_2D = mlibs.ctn(self.filename)
        if self.residualname != None:
            self.residual_data_2D = mlibs.ctn(self.residualname)
        if self.psfname != None:
            self.psf_data_2D = mlibs.ctn(self.psfname)


class radio_image_analysis():
    def __init__(self, input_data,z = None,do_petro=False,
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
        self.do_petro = do_petro
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

        self.image_properties()


    def image_properties(self):
        try:
            self.cell_size = mlibs.get_cell_size(self.input_data.filename)
        except:
            # print('!! WARNING !! Setting cellsize/pixelsize to unity.')
            _logging_.logger.warning("Setting cellsize/pixelsize to unity.")
            self.cell_size = 1

        _logging_.logger.info("Computing image level statistics.")
        # _logging_.file_handler.info("Computing image level statistics.")

        # self.image_level_statistics = \
        #     mlibs.level_statistics(img=self.input_data.filename,
        #                            cell_size=self.cell_size, crop=self.crop,
        #                            sigma = self.sigma_level,
        #                            apply_mask=self.apply_mask,
        #                            results=self.results, SAVE=self.SAVE,
        #                            ext=config.ext,
        #                            show_figure=config.show_plots)
        #
        # _logging_.logger.info("Computing image properties.")
        # self.levels, self.fluxes, self.agrow, self.plt_image, \
        #     self.omask, self.mask, self.results_im_props = \
        #     mlibs.compute_image_properties(img=self.input_data.filename,
        #                                    cell_size=self.cell_size,
        #                                    residual=self.input_data.residualname,
        #                                    sigma_mask=self.sigma_mask,
        #                                    dilation_size=self.dilation_size,
        #                                    crop=self.crop,
        #                                    iterations=config.mask_iterations,
        #                                    box_size=self.box_size,
        #                                    last_level=self.last_level,
        #                                    mask=self.mask,
        #                                    apply_mask=self.apply_mask,
        #                                    vmin_factor=self.vmin_factor,
        #                                    results=self.results,
        #                                    show_figure=self.show_figure,
        #                                    logger=_logging_.logger)
        #
        # self.img_stats = \
        #     mlibs.get_image_statistics(imagename=self.input_data.filename,
        #                                residual_name=self.input_data.residualname,
        #                                cell_size=self.cell_size,
        #                                mask_component=None,
        #                                mask=self.mask,
        #                                region='', dic_data=None,
        #                                sigma_mask=self.sigma_mask,
        #                                apply_mask=self.apply_mask,
        #                                fracX=0.15, fracY=0.15)

        self.image_measures, self.mask, self.omask = \
            mlibs.measures(imagename=self.input_data.filename,
                           residualname=self.input_data.residualname,
                           z=self.z,
                           mask_component=self.mask_component,
                           sigma_mask=self.sigma_mask,
                           last_level=self.last_level,
                           vmin_factor=self.vmin_factor,
                           plot_catalog=True, 
                           data_2D=self.input_data.image_data_2D,
                           npixels=self.npixels,fwhm=self.fwhm,
                           kernel_size=self.kernel_size,
                           dilation_size=self.dilation_size,
                           main_feature_index=0,
                           results_final={},
                           iterations=config.mask_iterations,
                           fracX=0.15, fracY=0.15,
                           deblend=False, bkg_sub=False,
                           bkg_to_sub=None, rms=None,
                           do_petro=self.do_petro,
                           apply_mask=self.apply_mask,
                           do_PLOT=True, SAVE=self.SAVE,
                           show_figure=self.show_figure,
                           mask=self.mask,
                           do_measurements='all',
                           compute_A=True,
                           add_save_name='',logger=_logging_.logger)


class source_extraction():
    """
    Source extraction class, responsible to find relevant regions of emission.

    For now, only SEP is implemented. Soon, other algorithms will be added
    in this class as alternatives for source extraction. These are:
        - PyBDSF
        - AstroDendro
        - Photutils
    """
    def __init__(self, input_data,z=0.05,ids_to_add=[1,2],
                 crop=False, box_size=256,
                 apply_mask=False, mask=None, dilation_size = None,
                 sigma_level=3, sigma_mask=6, vmin_factor=3, mask_component=None,
                 bwf=2, bhf=2, fwf=10, fhf=10,
                 segmentation_map = True, filter_type='matched',
                 deblend_nthresh=3, deblend_cont=1e-8,
                 clean_param=0.5, clean=True,
                 sort_by='flux',  # sort detected source by flux
                 sigma=12,  # min rms to search for sources
                 ell_size_factor=2.0,  # unstable, please inspect!
                 show_detection=False,show_petro_plots=False,
                 SAVE=True, show_figure=True,dry_run = False):

        self.input_data = input_data
        self.z = z
        self.ids_to_add = ids_to_add
        self.crop = crop
        self.box_size = box_size
        self.apply_mask = apply_mask
        self.mask = mask
        self.dilation_size = dilation_size
        self.sigma_level = sigma_level
        self.sigma_mask = sigma_mask
        self.sigma = sigma
        self.ell_size_factor = ell_size_factor
        self.vmin_factor = vmin_factor
        self.mask_component = mask_component
        # self.bw = bw
        # self.bh = bh
        # self.fw = fw
        # self.fh = fh
        # if (bw == None) & (bw == None) & (bw == None) & (bw == None):
        try:
            self.bspx, self.aO, self.bO = \
                mlibs.get_beam_size_px(self.input_data.filename)
            print(self.bspx)
            self.bw = self.bspx / bwf
            self.bh = self.bspx / bhf
            self.fw = self.bspx / fwf
            self.fh = self.bspx / fhf
        except:
            self.bw = 128
            self.bh = 21
            self.fw = 81
            self.fh = 81

        try:
            self.minarea = mlibs.beam_area2(self.input_data.filename)
        except:
            self.minarea = self.input_data.data_2D.shape[0]/25
        # self.bw, self.bh, self.fw, self.fh = bw, bh, fw, fh
        self.segmentation_map = segmentation_map
        self.filter_type = filter_type
        self.deblend_nthresh = deblend_nthresh
        self.deblend_cont = deblend_cont
        self.clean_param = clean_param
        self.clean = clean
        self.sort_by = sort_by
        self.SAVE = SAVE
        self.show_figure = show_figure
        self.show_detection = show_detection
        self.show_petro_plots = show_petro_plots
        
        if dry_run is True:
            self.show_detection = True

        self.get_sources()
        if dry_run is not True:
            self.contruct_source_properties()

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
                           apply_mask=self.apply_mask,
                           show_detection=self.show_detection)

    def contruct_source_properties(self):
        (self.sources_photometries, self.n_components,
         self.psf_name, self.mask) = \
            mlibs.prepare_fit(self.input_data.filename,
                              self.input_data.residualname,
                              z=self.z,ids_to_add = self.ids_to_add,
                              bw=self.bw, bh=self.bh, fw=self.fw, fh=self.fh,
                              sigma=self.sigma,
                              deblend_nthresh=self.deblend_nthresh,
                              ell_size_factor=self.ell_size_factor,
                              deblend_cont=self.deblend_cont,
                              clean_param=self.clean_param,
                              show_petro_plots=self.show_petro_plots)

class evaluate_source_structure():
    """
    This will be designed to evaluate the souce structure
    in order to check its complexity and compute how many model
    components will be required to perfor the multi-sersic fitting. 
    
    Also, this will compute basic source morphology, in order to 
    quantify which component represents a compact or a extended 
    structure. 
    """
    pass
    

class sersic_multifit_radio():
    """
    Multi-Sersic Fitting Decomposition.

    Perform a semi-automated and robust multi-sersic image decomposition.
    It supports GPU-acceleration using Jax. If no GPU is present, Jax still 
    will benefit from CPU parallel processing. Do not worry, you do not have 
    to change anything, Jax will automatically detect wheter you are runnin on
    CPU or GPU. 

    Basic principles:
        - run a source extraction, to identify relevant emission.
        - compute basic properties for each identified region, such as
          size, intensity, shape and orientation
        - uses that information to construct an object and prepare the
          settings to start the fit
        - compute statistics of the fit
        - if asked, calculates the relative fluxes of each component.
    To improve:
        - run an MCMC on model parameters (optional, takes time)
    To-do:
        - automated evaluation of which components is compact (unresolved)
        and which components is extended
        - automated evaluation of which structure cannot be modelled by a
        single function. For example, a spiral galaxy is reconized as a single
        source, but it can not be modelled by a single function: we require to
        model the bulge/bar/disk, for example.
    """
    def __init__(self, input_data, SE, aspect=None, 
                 fix_geometry = True,
                 comp_ids = ['1'],
                 fix_n = None, 
                 fix_value_n = None, dr_fix = None,
                 convolution_mode='GPU',method1='least_squares',
                 method2='least_squares'):
        """
            Examples: 
                fix_n = [True, True, True,True, True, True,True, True, True]
                fix_value_n=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                dr_fix = [3, 3, 100, 100, 10, 10, 10, 10, 10]
        """
        self.input_data = input_data
        self.SE = SE
        self.aspect = aspect
        self.comp_ids = comp_ids
        self.fix_geometry = fix_geometry
        self.convolution_mode = convolution_mode
        self.method1 = method1
        self.method2 = method2
        
        
        if fix_n==None:
            self.fix_n = [True] * self.SE.n_components
        else:
            self.fix_n = fix_n
            
        if fix_value_n==None:
            self.fix_value_n = [0.5] * self.SE.n_components
        else:
            self.fix_value_n = fix_value_n
            
        if dr_fix==None:
            self.dr_fix = [10] * self.SE.n_components
        else:
            self.dr_fix = dr_fix
        
        self.__sersic_radio()

    def __sersic_radio(self):
        (self.results_fit, self.lmfit_results, self.lmfit_results_1st_pass,
         self.errors_fit, self.models, self.results_compact_conv_morpho,
         self.results_compact_deconv_morpho, self.results_ext_conv_morpho,
         self.results_ext_deconv_morpho) = \
            mlibs.run_image_fitting(imagelist=[self.input_data.filename],
                                    residuallist=[self.input_data.residualname],
                                    aspect=self.aspect,
                                    comp_ids=self.comp_ids,# which IDs refers to compact components?
                                    sources_photometries=self.SE.sources_photometries,
                                    n_components=self.SE.n_components,
                                    z=self.SE.z,
                                    convolution_mode=self.convolution_mode,
                                    method1=self.method1,
                                    method2=self.method2,
                                    mask=self.SE.mask,
                                    save_name_append='',
                                    fix_n=self.fix_n,
                                    fix_value_n=self.fix_value_n,
                                    fix_geometry=self.fix_geometry,  # unstable if  False
                                    dr_fix=self.dr_fix,
                                    logger=_logging_.logger)


class morphometry():
    """
    Core functionalities from Morfometryka. 
    Morfometryka is not publically available yet, 
    so these functions will be added in a later stage. 
    """
    def __init__(self, input_data, aspect=None):
        self.input_data = input_data
        
        
    def _concentration(self):
        pass
    def _asymetry(self):
        pass
    def _momentum(self):
        pass
    def _sigma_psi(self):
        pass
    def _entropy(self):
        """
        This will be provided here soon, since it was my Master Thesis subject. 
        """
        pass
    def _kurvature(self):
        """
        This will be provided here soon, since it was my Master Thesis subject. 
        """
        pass

    
    
        
    pass


class radio_star_formation():
    """
    Compute star-formation estimates from radio emission, given the converstion
    law.
    """

    def __init__(self, input_data, SMFR,decompose = True,z=0.01,
                 calibration_kind='Murphy12',
                 alpha = -0.85, alpha_NT = -0.85, frequency = None):
        self.input_data = input_data
        self.SMFR = SMFR
        self.decompose = decompose
        self.z = z
        self.frequency = frequency #in GHz
        self.alpha = alpha
        self.alpha_NT = alpha_NT
        self.return_with_error = True
        if frequency is None:
            try:
                imhd = mlibs.imhead(self.input_data.filename)
                frequency = mlibs.imhd['refval'][2] / 1e9 # GHz
                _logging_.logger.info(f"Using frequency of {frequency:.2f} GHz for "
                                      f"star formation estimate.")
            except:
                try:
                    # GHz
                    frequency = mlibs.get_frequency(self.input_data.filename)
                    _logging_.logger.info(f"Using frequency of {frequency:.2f} GHz for "
                                          f"star formation estimate.")
                except:
                    _logging_.logger.warning('Frequency may be wrong. Please, '
                                             'provide the frequency of the observation.')
                    frequency = 5.0  # GHz

        else:
            frequency = frequency
        self.frequency = frequency
        self.calibration_kind = calibration_kind
        self.compute_SFR_extended()

    def compute_flux_compact(self):
        """
        Computes the total flux Density of radio compact components.
        """
        pass

    def compute_SFR_extended(self):
        """
        Computes star-formation for radio extended components.
        """

        self.SFR_ext, self.SFR_ext_err = \
            mlibs.compute_SFR_NT(flux=self.SMFR.results_fit['flux_density_extended_model'][0] /
                                      1000,
                                 flux_error=self.SMFR.results_fit['flux_res'][0] / 1000,
                                 frequency=self.frequency, z=self.z,
                                 alpha=self.alpha, alpha_NT=self.alpha_NT,
                                 calibration_kind=self.calibration_kind,
                                 return_with_error=self.return_with_error)
        mlibs.print_logger_header(title="SFR Estimates",
                            logger=_logging_.logger)
        _logging_.logger.info(f" ==> SFR ={self.SFR_ext:.2f} +/- {self.SFR_ext_err:.2f} "
                              f"Mo/yr")



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
    parser = argparse.ArgumentParser(description='Morphen.')
    parser.add_argument('-filename',  '--filename',  required=False, \
                        help='Image data.')
    parser.add_argument('-residualname',  '--residualname',  required=False, \
                        help='Associated Residual from Image Data')
    parser.add_argument('-psfname',  '--psfname',  required=False, \
                        help='PSF name.')

    parser.add_argument('-image_stats',  '--image_stats',
                        required=False,  nargs='?',  default=False,
                        const=True,
                        help='Compute basic image statistics.')

    parser.add_argument('-find_sources',  '--find_sources',
                        required=False,  nargs='?',  default=False,
                        const=True,
                        # action='store_true',
                        help='Perform source extraction and basic photometry.')

    parser.add_argument('--sigma', type=float, default=10,
                        help='Sigma value for source extraction.')
    parser.add_argument('--ell_size_factor', type=float, default=2,
                        help='Factor of ellipse size for source plot artistics/statistics.')

    parser.add_argument('-sersic_radio',  '--sersic_radio',
                        required=False,  nargs='?',  default=False,
                        const=True,
                        help='Perform Sersic Image Fitting for radio data.')

    parser.add_argument('--solver2', type=str, default='least_squares',
                        help='2nd run solver method (nelder or least_squares).')

    parser.add_argument('-SFR-do',  '--SFR-do',
                        required=False,  nargs='?',  default=False,
                        const=True,
                        help='Compute SFR Estimates.')

    parser.add_argument('--redshift', type=float, default=0.01,
                        help='Redshift of the source.')

    # parser.add_argument('-sersic_optical',  '--sersic_optical',
    #                     required=False,  nargs='?',  default=False,
    #                     const=True,
    #                     help='Perform Sersoc Image Fitting for optical data.')

    parser.add_argument('-noshow',  '--noshow', required=False,  nargs='?',
                        default=False,  const=True,
                        help='Do not show plots.')
    # parser.add_argument('-filename',  '--filename',  required=False, \
    #                     help='The input array of the light profile of the galaxy.')

    args = parser.parse_args()

    if args.filename != None:
        input_data = read_data(filename=args.filename,
                               residualname=args.residualname,)
        if "--image_stats" in sys.argv:
            radio_image_analysis(input_data)
        if "--find_sources" in sys.argv:
            SE = source_extraction(input_data,sigma=args.sigma,
                              ell_size_factor=args.ell_size_factor)
            if "--sersic_radio" in sys.argv:
                if args.residualname != None:
                    SMFR = sersic_multifit_radio(input_data, SE,
                                                 method2 = args.solver2)
                else:
                    print("Error: Please, provide a residual image (e.g. the one "
                          "generate during interferometric deconvolution).")
                if '--SFR-do' in sys.argv:
                    SFR = radio_star_formation(input_data,
                                               SMFR,
                                               z=args.redshift)
