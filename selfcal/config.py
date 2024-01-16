"""
Configuration of different template parameters for self-calibration and imaging.
"""

FIELD = ''
SPWS = ''
ANTENNAS = ''
# refant = ''
minblperant=3
solnorm = False # True will always be used for amp-selfcal!.
combine = ''
outlierfile = ''


quiet = False
run_mode = 'terminal'

path = ('/media/sagauga/galnet/LIRGI_Sample/VLA-Archive/A_config/23A-324/X_band/MCG12'
        '/autoselfcal/')
vis_list = ['MCG12-02-001.calibrated.avg12s']  # do not use the .ms extension

steps = [
    'startup',  # create directory structure, start variables and clear visibilities.
    'save_init_flags',
    # 'fov_image',
    # # # 'run_rflag_init',
    'test_image',
    'select_refant',
    'p0',#initial test selfcal step
    'p1',#start of the first trial of selfcal, phase only (p)
    'p2',#continue first trial, can be p or ap, but uses gain table from step 1;
    'ap1',
    'split_trial_1',#split the data after first trial
    'report_results',#report results of first trial
    # 'run_rflag_final',
]


cell_sizes_JVLA = {'L':'0.2arcsec',
                   'S':'0.1arcsec',
                   'C':'0.06arcsec',
                   'X':'0.04arcsec',
                   'Ku':'0.02arcsec',
                   'K':'0.014arcsec',
                   'Ka':'0.01arcsec'}

cell_sizes_eMERLIN = {'L':'0.05arcsec',
                      'C':'0.008arcsec'}



taper_sizes_eMERLIN = {'L':'0.2arcsec',
                       'C':'0.04arcsec'}

taper_sizes_JVLA = {'L':'1.0arcsec',
                    'S':'0.5arcsec',
                    'C':'0.3arcsec',
                    'X':'0.2arcsec',
                    'Ku':'0.1arcsec',
                    'K':'0.03arcsec',
                    'Ka':'0.04arcsec'}



init_parameters = {'fov_image': {'imsize': 1024*8,
                                'cell': '0.2arcsec',
                                'basename': 'FOV_phasecal_image',
                                'niter': 100,
                                'robust': 0.5},
                  'test_image': {'imsize': int(1024*2),
                                 'imsizey': int(1024*2),
                                 'FIELD_SHIFT':None,
                                 'cell': cell_sizes_JVLA['X'],
                                 'prefix': 'test_image',
                                 'niter': 10000,
                                 'robust': 0.0}
                  }

global_parameters = {'imsize': init_parameters['test_image']['imsize'],
                     'imsizey': init_parameters['test_image']['imsizey'],
                     'FIELD_SHIFT': init_parameters['test_image']['FIELD_SHIFT'],
                     'cell': init_parameters['test_image']['cell'],
                     'nsigma_automask' : '3.0',
                     'nsigma_autothreshold' : '1.5',
                     'uvtaper' : [''],
                     'niter':100000}


"""
===================

SELFCAL STRATEGY

===================

calmode        solint(sec)    niter (for CLEAN)

------------  --------------  ------------------
p                 120                200       
p                 120                300
ap                inf                300
p                 120                400
p                  64                500
p                  64                750
ap               3600                750
p                  64               1000
p                  32               1250
p                  32               1500
ap                120               1500
p                  32               2000
p                  16               3000
p                  16               5000
ap                 32              10000
------------------------------------------------


"""

params_very_faint = {'name': 'very_faint',
                     'p0': {'robust': 0.5,
                            'solint' : '96s',
                            'sigma_mask': 12,
                            'combine': 'spw',
                            'gaintype': 'T',
                            'calmode': 'p',
                            'minsnr': 1.5,
                            'spwmap': [],#leavy empty here. It will be filled later
                            'nsigma_automask' : '3.0',
                            'nsigma_autothreshold' : '1.5',
                            'uvtaper' : [''],
                            'with_multiscale' : False},
                     'ap1': {'robust': 1.0,
                             'solint': '96s',
                             'sigma_mask': 6,
                             'combine': 'spw',
                             'gaintype': 'T',
                             'calmode': 'ap',
                             'minsnr': 1.5,
                             'spwmap': [],#leavy empty here. It will be filled later
                             'nsigma_automask' : '3.0',
                             'nsigma_autothreshold' : '1.5',
                             'uvtaper' : [''],
                             'with_multiscale' : True},
                     }


params_faint = {'name': 'faint',
                'p0': {'robust': 0.0,
                       'solint' : '96s',
                       'sigma_mask': 12,
                       'combine': 'spw',
                       'gaintype': 'T',
                       'calmode': 'p',
                       'minsnr': 1.5,
                       'spwmap': [],
                       'nsigma_automask' : '5.0',
                       'nsigma_autothreshold' : '2.5',
                       'uvtaper' : [''],
                       'with_multiscale' : False,
                       'compare_solints' : False},
                'p1': {'robust': 0.5,
                       'solint' : '120s',
                       'sigma_mask': 8,
                       'combine': '',
                       'gaintype': 'T',
                       'calmode': 'p',
                        'minsnr': 1.0,
                       'spwmap': [],
                       'nsigma_automask' : '3.0',
                       'nsigma_autothreshold' : '1.5',
                       'uvtaper' : [''],
                       'with_multiscale' : True,
                       'scales' : '0,5,20',
                       'compare_solints' : False},
                'ap1': {'robust': 1.0,
                        'solint': '120s',
                        'sigma_mask': 6,
                        'combine': '',
                        'gaintype': 'T',
                        'calmode': 'ap',
                        'minsnr': 1.0,
                        'spwmap': [],
                        'nsigma_automask' : '3.0',
                        'nsigma_autothreshold' : '1.5',
                        'uvtaper' : [''],
                        'with_multiscale' : True,
                        'scales': '0,5,20',
                        'compare_solints' : False},
                }


#Selfcal parameters to be used for standard sources, with a total integrated
# flux density between ~20 mJy and 0.1 Jy.
params_standard_1 = {'name': 'standard_1',
                   'p0': {'robust': -0.5,
                          'solint' : '240s',
                          'sigma_mask': 15,
                          'combine': '',
                          'gaintype': 'T',
                          'calmode': 'p',
                          'minsnr': 1.5,
                          'spwmap': [],
                          'nsigma_automask' : '5.0',
                          'nsigma_autothreshold' : '2.5',
                          'uvtaper' : [''],
                          'with_multiscale' : False,
                          'scales' : '0,5,20',
                          'compare_solints' : False},
                   'p1': {'robust': 0.0,
                          'solint' : '180s',
                          'sigma_mask': 12,
                          'combine': '',
                          'gaintype': 'G',
                          'calmode': 'p',
                          'minsnr': 1.5,
                          'spwmap': [],
                          'nsigma_automask' : '3.0',
                          'nsigma_autothreshold' : '1.5',
                          'uvtaper' : [''],
                          'with_multiscale' : True,
                          'scales' : '0,5,20',
                          'compare_solints' : False},
                   'p2': {'robust': 0.5,
                          'solint': '120s',
                          'sigma_mask': 7,
                          'combine': '',
                          'gaintype': 'T',
                          'calmode': 'p',
                          'minsnr': 1.5,
                          'spwmap': [],
                          'nsigma_automask' : '3.0',
                          'nsigma_autothreshold' : '1.5',
                          'uvtaper' : [''],
                          'with_multiscale' : True,
                          'scales': '0,5,10,20,40',
                          'compare_solints' : False},
                   'ap1': {'robust': 1.0,
                           'solint': '120s',
                           'sigma_mask': 6,
                           'combine': '',
                           'gaintype': 'T',
                           'calmode': 'ap',
                           'minsnr': 1.5,
                           'spwmap': [],
                           'nsigma_automask' : '3.0',
                           'nsigma_autothreshold' : '1.5',
                           'uvtaper' : [''],
                           'with_multiscale' : True,
                           'scales': '0,5,10,20,40',
                           'compare_solints' : False},
                 }



params_standard_2 = {'name': 'standard_2',
                   'p0': {'robust': -0.5,
                          'solint' : '96s',
                          'sigma_mask': 25, #set to 15 if e-MERLIN
                          'combine': '',
                          'gaintype': 'T',
                          'calmode': 'p',
                          'minsnr': 2.0,
                          'spwmap': [],
                          'nsigma_automask' : '6.0',
                          'nsigma_autothreshold' : '3.0',
                          'uvtaper' : [''],
                          'with_multiscale' : False,
                          'scales': '0,5,10',
                          'compare_solints' : False},
                   'p1': {'robust': -0.5,
                          'solint' : '60s',
                          'sigma_mask': 15,
                          'combine': '',
                          'gaintype': 'G',
                          'calmode': 'p',
                          'minsnr': 2.0,
                          'spwmap': [],
                          'nsigma_automask' : '3.0',
                          'nsigma_autothreshold' : '1.5',
                          'uvtaper' : [''],
                          'with_multiscale' : False,
                          'scales': '0,5,10,20',
                          'compare_solints' : False},
                   'p2': {'robust': 0.5,
                          'solint': '36s',
                          'sigma_mask': 10,
                          'combine': '',
                          'gaintype': 'T',
                          'calmode': 'p',
                          'minsnr': 2.0,
                          'spwmap': [],
                          'nsigma_automask' : '3.0',
                          'nsigma_autothreshold' : '1.5',
                          'uvtaper' : [''],
                          'with_multiscale' : True,
                          'scales': '0,5,10,20,40',
                          'compare_solints' : False},
                   'ap1': {'robust': 0.5,
                           'solint': '36s',
                           'sigma_mask': 8,
                           'combine': '',
                           'gaintype': 'G',
                           'calmode': 'ap',
                           'minsnr': 2.0,
                           'spwmap': [],
                           'nsigma_automask' : '3.0',
                           'nsigma_autothreshold' : '1.5',
                           'uvtaper' : [''],
                           'with_multiscale' : True,
                           'scales': '0,5,10,20,40',
                           'compare_solints' : False},
                 }

#Selfcal parameters to be used for bright sources, with a total integrated
# flux density above 0.1 Jy
params_bright = {'name': 'bright',
                 'p0': {'robust': -1.0,
                        'solint' : '48s',
                        'sigma_mask': 50,
                        'combine': '',
                        'gaintype': 'G',
                        'calmode': 'p',
                        'minsnr': 3.0,
                        'spwmap': [],
                        'nsigma_automask': '6.0',
                        'nsigma_autothreshold': '3.0',
                        'uvtaper' : [''],
                        'with_multiscale' : False,
                        'scales': '0,5,10',
                        'compare_solints' : False},
                 'p1': {'robust': -0.5,
                        'solint' : '48s',
                        'sigma_mask': 25,#set to 15 if e-MERLIN
                        'combine': '',
                        'gaintype': 'G',
                        'calmode': 'p',
                        'minsnr': 3.0,
                        'spwmap': [],
                        'nsigma_automask': '6.0',
                        'nsigma_autothreshold': '3.0',
                        'uvtaper' : [''],
                        'with_multiscale': False,
                        'scales': '0,5,10,20',
                        'compare_solints': False},
                 'p2': {'robust': 0.5,
                        'solint': '24s',
                        'sigma_mask': 18,
                        'combine': '',
                        'gaintype': 'G',
                        'calmode': 'p',
                        'minsnr': 3.0,
                        'spwmap': [],
                        'uvtaper' : [''],
                        'nsigma_automask': '3.0',
                        'nsigma_autothreshold': '1.5',
                        'with_multiscale': True,
                        'scales': '0,5,10,20,40',
                        'compare_solints': False},
                 'ap1': {'robust': 0.5,
                         'solint': '24s',
                         'sigma_mask': 12,
                         'combine': '',
                         'gaintype': 'G',
                         'calmode': 'ap',
                         'minsnr': 3.0,
                         'spwmap': [],
                         'uvtaper' : [''],
                         'nsigma_automask': '3.0',
                         'nsigma_autothreshold': '1.5',
                         'with_multiscale': True,
                         'scales': '0,5,10,20,40',
                         'compare_solints': False},
                 }

