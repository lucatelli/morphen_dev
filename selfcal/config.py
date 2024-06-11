"""
Configuration of different template parameters for self-calibration and imaging.
This is intended to be used as a first trial of self-calibration.
"""

FIELD = ''
SPWS = ''
ANTENNAS = ''
refantmode = 'strict'
refant = ''
minblperant=3
solnorm = False # True or False (globally used). If '', will use False for phases and True for
             # amplitudes.
combine = ''
outlierfile = ''


quiet = True
do_additional_images = False
run_mode = 'terminal'


# path = ('/path/to/vis/')
# vis_list = ['example_vis'] #do not use .ms extension

#VLA
receiver = 'L'
instrument = 'EVLA' # 'EVLA' or 'eM'

#number of channels-out for wsclean's MFS imager.
if instrument == 'EVLA':
    if receiver == 'L' or receiver == 'S' or receiver == 'C':
        nc = 6
    else:
        nc = 4
if instrument == 'eM':
    nc = 3
# negative_arg='no-negative'
negative_arg='negative'
steps = [
    'startup',  # create directory structure, start variables and clear visibilities.
    'save_init_flags',  # save (or restore) the initial flags and run statwt
    # 'fov_image', # create a FOV image
    'test_image',#create a test image
    # 'run_autoflag_init', # run rflag on the initial data (rarely used)
    'select_refant', #select reference antenna
    'p0',#initial test  of selfcal, phase only (p)
    'p1',#redo phase-only selfcal (if enough flux density); ignores p0
    # 'p2',#continue phase-only selfcal (incremental)
    # 'ap1',#amp-selfcal (ap); uses p0 or (p1 and p2)
    # 'split_trial_1',#split the data after first trial (and run wsclean)
    # 'report_results',#report results of first trial
    # 'run_autoflag_final',#run rflag on the final data
]


cell_sizes_JVLA = {'L':'0.2arcsec',
                   'S':'0.1arcsec',
                   'C':'0.06arcsec',
                   'X':'0.04arcsec',
                   'Ku':'0.02arcsec',
                   'K':'0.01arcsec',
                   'Ka':'0.06arcsec'}

cell_sizes_eMERLIN = {'L':'0.03arcsec',
                      'C':'0.008arcsec'}



taper_sizes_eMERLIN = {'L':'0.2arcsec',
                       'C':'0.05arcsec'}

taper_sizes_JVLA = {'L':'1.0arcsec',
                    'S':'0.8arcsec',
                    'C':'0.6arcsec',
                    'X':'0.6arcsec',
                    'Ku':'0.2arcsec',
                    'K':'0.03arcsec',
                    'Ka':'0.07arcsec'}

if instrument == 'eM':
    cell_size = cell_sizes_eMERLIN[receiver]
    taper_size = taper_sizes_eMERLIN[receiver]
if instrument == 'EVLA':
    cell_size = cell_sizes_JVLA[receiver]
    taper_size = taper_sizes_JVLA[receiver]


init_parameters = {'fov_image': {'imsize': 1024*8,
                                'cell': '0.5arcsec',
                                'basename': 'FOV_phasecal_image',
                                'niter': 100,
                                'robust': 0.5},
                  'test_image': {'imsize': int(1024*6),
                                 'imsizey': int(1024*6),
                                 # 'FIELD_SHIFT':"'15:34:55.658  +23.29.43.026'",
                                 'FIELD_SHIFT':None,
                                 'cell': cell_size,
                                 'prefix': 'test_image',
                                 'uvtaper': [''],
                                 'niter': 10000,
                                 'robust': 0.0}
                  }

global_parameters = {'imsize': init_parameters['test_image']['imsize'],
                     'imsizey': init_parameters['test_image']['imsizey'],
                     'FIELD_SHIFT': init_parameters['test_image']['FIELD_SHIFT'],
                     'cell': init_parameters['test_image']['cell'],
                     'nsigma_automask' : '4.0',
                     'nsigma_autothreshold' : '1.5',
                     'uvtaper' : [''],
                     'with_multiscale' : True,
                     'scales' : 'None',
                     'niter':100000}

general_settings = {'timebin_statw': '24s',#timebin for statwt
                    'calwt': False,#calibrate/update the weights during applycal?
                    }


"""
Selfcal parameters to be used for very faint sources, 
with a total integrated flux density lower than 10 mJy.
"""
params_very_faint = {'name': 'very_faint',
                     'p0': {'robust': 0.5,
                            'solint' : '96s' if receiver in ('K', 'Ka', 'Ku') or instrument ==
                                               'eM' else '60s',
                            'sigma_mask': 6.0 if instrument == 'eM' else 15.0,
                            'combine': 'spw',
                            'gaintype': 'T',
                            'calmode': 'p',
                            'minsnr': 0.75 if instrument == 'eM' else 1.0,
                            'spwmap': [], #leavy empty here. It will be filled later if combine='spw'
                            'nsigma_automask' : '3.0',
                            'nsigma_autothreshold' : '1.5',
                            'uvtaper' : [''],
                            'with_multiscale' : False,
                            'scales': 'None',
                            'compare_solints' : False},
                     'ap1': {'robust': 1.0,
                             'solint' : '96s' if receiver in ('K', 'Ka', 'Ku') or instrument ==
                                               'eM' else '60s',
                             'sigma_mask': 6,
                             'combine': 'spw',
                             'gaintype': 'T',
                             'calmode': 'ap',
                             'minsnr': 0.75 if instrument == 'eM' else 1.0,
                             'spwmap': [], #leavy empty here. It will be filled later if combine='spw'
                             'nsigma_automask' : '3.0',
                             'nsigma_autothreshold' : '1.5',
                             'uvtaper' : [''],
                             'with_multiscale' : False if receiver in ('K', 'Ka', 'Ku') or
                                                          instrument == 'eM' else True,
                             'scales': 'None',
                             'compare_solints' : False},
                     }


"""
Selfcal parameters to be used for faint sources, 
with a total integrated flux density between 10 and 20 mJy.
"""
params_faint = {'name': 'faint',
                'p0': {'robust': 0.0,
                       'solint' : '120s' if receiver in ('K', 'Ka', 'Ku') or instrument == 'eM' else '60s',
                       'sigma_mask': 12.0 if instrument == 'eM' else 20,
                       'combine': 'spw',
                       'gaintype': 'T',
                       'calmode': 'p',
                       'minsnr': 1.0 if instrument == 'eM' else 1.5,
                       'spwmap': [],
                       'nsigma_automask' : '5.0',
                       'nsigma_autothreshold' : '2.5',
                       'uvtaper' : [''],
                       'with_multiscale' : False,
                       'scales' : 'None',
                       'compare_solints' : False},
                'p1': {'robust': 0.5,
                       'solint' : '120s' if receiver in ('K', 'Ka', 'Ku') or instrument == 'eM' else '60s',
                       'sigma_mask': 12,
                       'combine': 'spw' if receiver in ('X', 'K', 'Ka', 'Ku') or instrument == 'eM' else '',
                       'gaintype': 'T',
                       'calmode': 'p',
                        'minsnr': 1.0,
                       'spwmap': [],
                       'nsigma_automask' : '3.0',
                       'nsigma_autothreshold' : '1.5',
                       # 'uvtaper' : [''],
                       'uvtaper': [taper_size] if receiver in ('X', 'Ku', 'K', 'Ka') or
                                                  instrument == 'eM' else [''],
                       'with_multiscale': True if receiver in ('K', 'Ka', 'Ku') or instrument ==
                                                   'eM' else True,
                       # 'scales' : '0,5,20',
                       'scales': 'None',
                       'compare_solints' : False},
                'p2': {'robust': 1.0,
                       'solint' : '72s' if receiver in ('K', 'Ka', 'Ku') or instrument == 'eM' else '36s',
                       'sigma_mask': 8,
                       'combine': 'spw' if receiver in ('K', 'Ka', 'Ku') or instrument == 'eM' else '',
                       'gaintype': 'T',
                       'calmode': 'p',
                       'minsnr': 1.0,
                       'spwmap': [],
                       'nsigma_automask': '3.0',
                       'nsigma_autothreshold': '1.5',
                       'uvtaper': [''],
                       'with_multiscale': True if receiver in ('K', 'Ka', 'Ku') or instrument ==
                                                   'eM' else True,
                       # 'scales': '0,5,10,20,40',
                       'scales': 'None',
                       'compare_solints': False},
                'ap1': {'robust': 0.5,
                        'solint': 'inf' if instrument == 'eM' else '120s',
                        'sigma_mask': 6,
                        'combine': 'spw' if receiver in ('X', 'K', 'Ka', 'Ku') or instrument == 'eM' else '',
                        'gaintype': 'T',
                        'calmode': 'ap',
                        'minsnr': 1.0,
                        'spwmap': [],
                        'nsigma_automask' : '3.0',
                        'nsigma_autothreshold' : '1.5',
                        'uvtaper': [''],
                        # 'uvtaper' : [taper_size] if receiver in ('X', 'Ku', 'K', 'Ka') or
                        #                             instrument == 'eM' else [''],
                        'with_multiscale': True if receiver in ('K', 'Ka', 'Ku') or instrument ==
                                                    'eM' else True,
                        # 'scales': '0,5,20,40',
                        'scales': 'None',
                        'compare_solints' : False},
                }



"""
Selfcal parameters to be used for standard sources, 
with a total integrated flux density between 20 and 50 mJy.
"""
params_standard_1 = {'name': 'standard_1',
                   'p0': {
                          'robust': -0.5 if receiver in ('K', 'Ka') or instrument == 'eM' else -1.0,
                          'solint' : '96s' if instrument == 'eM' else '96s',
                          'sigma_mask': 15.0 if instrument == 'eM' else 50.0,
                          'combine': 'spw' if receiver in ('K', 'Ka', 'Ku') or instrument == 'eM' else '',
                          'gaintype': 'T',
                          'calmode': 'p',
                          'minsnr': 1.0 if receiver in ('K', 'Ka') or instrument == 'eM' else 2.0,
                          'spwmap': [],
                          'nsigma_automask' : '5.0',
                          'nsigma_autothreshold' : '2.5',
                          'uvtaper' : [''],
                          'with_multiscale' : False,
                          # 'scales' : '0,5,20',
                          'scales': 'None',
                          'compare_solints' : False},
                   'p1': {'robust': 0.5 if receiver in ('K', 'Ka') or instrument == 'eM' else -0.5,
                          'solint' : '120s' if instrument == 'eM' else '60s',
                          'sigma_mask': 15.0 if instrument == 'eM' else 50.0,
                          'combine': 'spw' if instrument == 'eM' else '',
                          'gaintype': 'T' if receiver in ('K', 'Ka') else 'G',
                          'calmode': 'p',
                          'minsnr': 1.0 if receiver in ('K', 'Ka') else 1.5,
                          'spwmap': [],
                          'nsigma_automask' : '4.0',
                          'nsigma_autothreshold' : '1.0',
                          'uvtaper' : [''],
                          'with_multiscale' : False,
                          # 'scales' : '0,5,20',
                          'scales': 'None',
                          'compare_solints' : False},
                   'p2': {'robust': 0.5,
                          'solint': '48s',
                          'sigma_mask': 12.0 if instrument == 'eM' else 25.0,
                          'combine': '',
                          'gaintype': 'T',
                          'calmode': 'p',
                          'minsnr': 1.5,
                          'spwmap': [],
                          'nsigma_automask' : '3.0',
                          'nsigma_autothreshold' : '1.0',
                          'uvtaper' : [taper_size] if receiver in ('X', 'Ku', 'K', 'Ka') else [''],
                          'with_multiscale': False if receiver in ('K', 'Ka', 'Ku') or instrument
                                                      == 'eM' else True,
                          # 'scales': '0,5,10,20,40',
                          'scales': 'None',
                          'compare_solints' : False},
                   'ap1': {'robust': 0.5,
                           'solint': '60s' if instrument == 'eM' else '60s',
                           'sigma_mask': 8,
                           'combine': 'spw' if instrument == 'eM' else '',
                           'gaintype': 'G',
                           'calmode': 'ap',
                           'minsnr': 1.5,
                           'spwmap': [],
                           'nsigma_automask' : '3.0',
                           'nsigma_autothreshold' : '1.0',
                           'uvtaper' : [''],
                           'with_multiscale': False if receiver in ('K', 'Ka', 'Ku') or
                                                       instrument == 'eM' else True,
                           # 'scales': '0,5,10,20,40',
                           'scales': 'None',
                           'compare_solints' : False},
                 }


"""
Selfcal parameters to be used for standard sources, 
with a total integrated flux density between 50 and 100 mJy.
"""
"""
Selfcal parameters to be used for standard sources, 
with a total integrated flux density between 50 and 100 mJy.
Note that some values may change if using e-MERLIN or VLA.
"""
params_standard_2 = {'name': 'standard_2',
                   'p0': {'robust': 0.0 if receiver in ('K', 'Ku', 'Ka') or instrument == 'eM' else -0.5,
                          'solint' : '96s' if instrument == 'eM' else '96s',
                          'sigma_mask': 25.0 if instrument == 'eM' else 50.0,
                          'combine': 'spw' if instrument == 'eM' else '',
                          'gaintype': 'T',
                          'calmode': 'p',
                          'minsnr': 1.0 if receiver in ('K', 'Ku', 'Ka') or instrument == 'eM' else 3.0,
                          'spwmap': [],
                          'nsigma_automask' : '6.0',
                          'nsigma_autothreshold' : '3.0',
                          'uvtaper' : [''],
                          'with_multiscale' : False,
                          # 'scales': '0,5,10',
                          'scales': 'None',
                          'compare_solints' : False},
                   'p1': {'robust': 0.0 if receiver in ('K', 'Ku', 'Ka') or instrument == 'eM' else -0.5,
                          'solint' : '60s' if instrument == 'eM' else '60s',
                          'sigma_mask': 18 if instrument == 'eM' else 35,
                          'combine': 'spw' if instrument == 'eM' else '', #needs to be tested
                          'gaintype': 'T' if instrument == 'eM' else 'G',
                          'calmode': 'p',
                          'minsnr': 1.0 if instrument == 'eM' else 3.0,
                          'spwmap': [],
                          'nsigma_automask' : '3.0',
                          'nsigma_autothreshold' : '1.5',
                          'uvtaper' : [taper_size] if instrument == 'eM' else [''],
                          'with_multiscale' : True if instrument == 'eM' else True,
                          # 'scales': '0,5,10,20',
                          'scales': 'None',
                          'compare_solints' : False},
                   'p2': {'robust': 0.0 if instrument == 'eM' else 1.0,
                          'solint': '60s' if instrument == 'eM' else '36s',
                          'sigma_mask': 12,
                          'combine': 'spw' if instrument == 'eM' else '',
                          'gaintype': 'T',
                          'calmode': 'p',
                          'minsnr': 1.0 if instrument == 'eM' else 3.0,
                          'spwmap': [],
                          'nsigma_automask' : '3.0',
                          'nsigma_autothreshold' : '1.5',
                          'uvtaper' : [''],
                          'with_multiscale' : True,
                          # 'scales': '0,5,10,20,40',
                          'scales': 'None',
                          'compare_solints' : False},
                   'ap1': {'robust': 0.5,
                           'solint': '120s' if instrument == 'eM' else '36s',
                           'sigma_mask': 6,
                           'combine': 'spw' if instrument == 'eM' else '',
                           'gaintype': 'G',
                           'calmode': 'ap',
                           'minsnr': 1.0 if instrument == 'eM' else 3.0,
                           'spwmap': [],
                           'nsigma_automask' : '3.0',
                           'nsigma_autothreshold' : '1.5',
                           'uvtaper' : [''],
                           'with_multiscale' : True if instrument == 'eM' else True,
                           # 'scales': '0,5,10,20,40',
                           'scales': 'None',
                           'compare_solints' : False},
                 }

"""
Selfcal parameters to be used for bright sources, 
with a total integrated flux density above 0.1 Jy.
"""
params_bright = {'name': 'bright',
                 'p0': {'robust': -0.5 if receiver in ('K', 'Ka') else -1.0,
                        'solint': '96s' if instrument == 'eM' else '48s',
                        'sigma_mask': 60,
                        'combine': '',
                        'gaintype': 'G',
                        'calmode': 'p',
                        'minsnr': 3.0,
                        'spwmap': [],
                        'nsigma_automask': '6.0',
                        'nsigma_autothreshold': '3.0',
                        'uvtaper' : [''],
                        'with_multiscale' : False,
                        # 'scales': '0,5,10',
                        'scales': 'None',
                        'compare_solints' : False},
                 'p1': {'robust': 0.0 if receiver in ('K', 'Ka') or instrument == 'eM' else -0.5,
                        'solint' : '60s' if instrument == 'eM' else '24s',
                        'sigma_mask': 15.0 if instrument == 'eM' else 60.0,
                        'combine': 'spw' if instrument == 'eM' else '',
                        'gaintype': 'T' if instrument == 'eM' else 'G',
                        'calmode': 'p',
                        'minsnr': 1.0 if instrument == 'eM' else 3.0,
                        'spwmap': [],
                        'nsigma_automask': '4.0' if instrument == 'eM' else '6.0',
                        'nsigma_autothreshold': '1.5' if instrument == 'eM' else '3.0',
                        'uvtaper' : [taper_size] if instrument == 'eM' else [''],
                        'with_multiscale': True,
                        # 'scales': '0,5,10,20',
                        'scales': 'None',
                        'compare_solints': False},
                 'p2': {'robust': 0.0,
                        'solint': '24s',
                        'sigma_mask': 12.0 if instrument == 'eM' else 25.0,
                        'combine': '',
                        'gaintype': 'G',
                        'calmode': 'p',
                        'minsnr': 3.0,
                        'spwmap': [],
                        'uvtaper' : [''],
                        'nsigma_automask': '4.0',
                        'nsigma_autothreshold': '1.5',
                        'with_multiscale': True,
                        # 'scales': '0,5,10,20,40',
                        'scales': 'None',
                        'compare_solints': False},
                 'ap1': {'robust': 0.5,
                         'solint': '96s' if instrument == 'eM' else '36s',
                         'sigma_mask': 8.0 if instrument == 'eM' else 15.0,
                         'combine': 'spw' if instrument == 'eM' else '',
                         'gaintype': 'T' if instrument == 'eM' else 'G',
                         'calmode': 'ap',
                         'minsnr': 1.0 if instrument == 'eM' else 3.0,
                         'spwmap': [],
                         'uvtaper' : [''],
                         'nsigma_automask': '3.0',
                         'nsigma_autothreshold': '1.5',
                         'with_multiscale': True if instrument == 'eM' else True,
                         # 'scales': '0,5,10,20,40',
                         'scales': 'None',
                         'compare_solints': False},
                 }


params_trial_2 = None # comment this and uncomment the following lines
                      # if this is the second pass of self-calibration.


# params_trial_2 = {'name': 'trial_2',
#                  'p0': {'robust': 0.0,
#                         'solint' : '36s',
#                         'sigma_mask': 12,
#                         'combine': '',
#                         'gaintype': 'G',
#                         'calmode': 'p',
#                         'minsnr': 3.0,
#                         'spwmap': [],
#                         'nsigma_automask': '6.0',
#                         'nsigma_autothreshold': '3.0',
#                         'uvtaper' : [''],
#                         'with_multiscale' : True,
#                         'scales': '0,5,20,50',
#                         'compare_solints' : False},
#                  'p1': {'robust': 0.5,
#                         'solint' : '12s',
#                         'sigma_mask': 8,#set to 15 if e-MERLIN
#                         'combine': '',
#                         'gaintype': 'G',
#                         'calmode': 'p',
#                         'minsnr': 3.0,
#                         'spwmap': [],
#                         'nsigma_automask': '6.0',
#                         'nsigma_autothreshold': '3.0',
#                         'uvtaper' : [''],
#                         'with_multiscale': True,
#                         'scales': '0,5,20,50',
#                         'compare_solints': False},
#                  'ap1': {'robust': 0.5,
#                          'solint': '60s',
#                          'sigma_mask': 8,
#                          'combine': '',
#                          'gaintype': 'G',
#                          'calmode': 'ap',
#                          'minsnr': 3.0,
#                          'spwmap': [],
#                          'uvtaper' : [''],
#                          'nsigma_automask': '3.0',
#                          'nsigma_autothreshold': '1.5',
#                          'with_multiscale': True,
#                          'scales': '0,5,20,50',
#                          'compare_solints': False},
#                  }