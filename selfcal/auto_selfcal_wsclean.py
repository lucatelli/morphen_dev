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

Selfcal module for JVLA data reduction.
Note that this module can also be used for manual self-calibration.
The automated self-calibration is still experimental, but showed to be good
in some cases.



This module consists of performing interferometric imaging with tclean and runing
gain gal for self-calibration. There are multiple steps to be aware of:
    1. initial trials of  imaging > gaincal (p) > imaging > gaincal (p).
    These solution tables will be ignored later. These use
    conservative thresholds for the (auto)masking in combination with a low robust
    parameter. This ensures that only the brightest emission is imaged and
    self-calibrated. Also, a combination of longer solution intervals is used.
    The idea is to get a good starting model for the selfcalibration,
    and correct the initial majors shifts in phase.
    Be aware that using a low robust parameter
    will fail if your source is not bright enough
    (< ~ 10 mJy).  However, please
    check your data carefully. Even if the source is not that bright,
    using a long solution interval for the first trial may also fail.
    2. second trial (and sometimes the final) of
    imaging > gaincal p > imaging > gaincal p (or ap).
    We that after applying the first trial, our model is okay, but not good. We also
    have to think that our previous gain tables are not perferct. So, we head to the
    second trial of selfcal, and we can ignore our previous table(s).
    This time, we use a more agressive threshold for the (auto)masking in combination
    with a higher robust parameter. Again, this will depend on the nature of your data.
    A good initial hint is to start with a robust 0.5 - 1.0. If your source is bright
    enough, you can use a robust 0.0. If your source is not that bright, you can use
    a robust 1.0 - 2.0.




Additional features:
    - For the automation process, imaging is performed with the CASA's auto-masking
    featureenabled by setting `usemask='auto-multithresh' and `interactive=False`.
    Tuning the parameters for the auto-maskin is a very tedious and tricky task.
    The values used here are the ones that worked well for VLA-A configuration at C band.
    - If your science source have other nearby sources in the FOV, you may want
    to use them for self-cal. Nearly at the end of the code, you can set
    `use_outlier_fields = True` and provide the coordinates list of each outlier
    and the corresponding image sizes. For example:
        phasecenters = ["J2000 10:59:32.005 24.29.39.442"]
        imagesizes = [[512, 512]]
    The code will automatically create the outlier file with each outlier field.


Notes: The CASA Team appear to have fixed a bug when using auto-masking in combination
with outlier fields.


"""
__version__ = 0.3
__author__ = 'Geferson Lucatelli'
__email__ = 'geferson.lucatelli@postgrad.manchester.ac.uk'
__date__ = '2023 05 10'
print(__doc__)

import os
from casatasks import *

try:
    import casatools
    from casatasks import *
except:
    print('Not importing casatools because you are inside CASA.')
    pass

from casaplotms import plotms
from casaviewer.imview import imview

msmd = casatools.msmetadata()
ms = casatools.ms()
tb = casatools.table()

'''
tclean parameters
'''

parallel = True
calcpsf = True
threshold = '20.0e-6Jy'
FIELD = ''
SPWS = ''
ANTENNAS = ''
refant = 'ea18'
minblperant=3

imsize = 4096
cell = '0.07arcsec'

smallscalebias = 0.75
robust = 0.0
weighting = 'briggs'
gain = 0.1
pblimit = -0.1
nterms = 3
ext = ''

#
usemask = 'auto-multithresh'
interactive = False
# usemask='user'
sidelobethreshold = 3.5
noisethreshold = 10.0
lownoisethreshold = 5.0
minbeamfrac = 0.06
growiterations = 75
negativethreshold = 0.0
"""
sidelobethreshold = 3.5
noisethreshold = 10.0
lownoisethreshold = 4.0
minbeamfrac = 0.06
growiterations = 50
negativethreshold = 15.0
"""
"""
This works well for C and L bands
sidelobethreshold=3.5
noisethreshold=15.0
lownoisethreshold=5.0
minbeamfrac=0.06
growiterations=50
negativethreshold=15.0
"""

os.environ['SAVE_ALL_AUTOMASKS'] = "true"

# gain settings
solint_short = '24s'
solint_mid = '60s'
solint_mid2 = '48s'
solint_long = '96s'
solint_long2 = '192s'
solint_inf = 'inf'

solnorm = False
combine = ''

# plotting config
data_range = [-5.55876e-06, 0.00450872]

outlierfile = ''


# proj_name = '.calibrated'


def report_flag(summary, axis):
    for id, stats in summary[axis].items():
        print('%s %s: %5.1f percent flagged' % (
        axis, id, 100. * stats['flagged'] / stats['total']))
    pass


def eview(imagename, contour=None,
          data_range=None,  # [-2.52704e-05, 0.0159025],
          colormap='Rainbow 2', scaling=-2.0, zoom=1, out=None):
    if data_range == None:
        st = imstat(imagename)
        min_data = -1.0 * st['rms'][0]
        max_data = 1 * st['max'][0]
        data_range = [min_data, max_data]
    if contour == None:
        contour = imagename
    # if out==None:
    #     out = imagename + '_drawing.png'
    imview(raster={
        'file': imagename,
        'range': data_range,
        'colormap': colormap, 'scaling': scaling, 'colorwedge': True},
        contour={'file': contour,
                 'levels': [-0.1, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.8]},
        # include negative cont
        # axes={'x':'Declination'} ,
        # zoom={'blc': [3,3], 'trc': [3,3], 'coord': 'pixel'},
        zoom=zoom,
        out=out,
        # scale=scale,
        # dpi=dpi,
        # orient=orient
    )


# .replace('/','')


def get_image_statistics(imagename, residualname=None):
    if residualname == None:
        try:
            residualname = imagename.replace('.image', '.residual')
        except:
            print('Please, provide the residual image name')
    stats_im = imstat(imagename=imagename)
    stats_re = imstat(imagename=residualname)

    # determine the flux flux peak of image and residual
    flux_peak_im = stats_im['max'][0]
    flux_peak_re = stats_re['max'][0]

    # determine the rms and std of residual and of image
    rms_re = stats_re['rms'][0]
    rms_im = stats_im['rms'][0]
    sigma_re = stats_re['sigma'][0]
    sigma_im = stats_im['sigma'][0]

    # determine the image and residual flux
    flux_im = stats_im['flux'][0]
    # flux_re = stats_re['flux']

    sumsq_im = stats_im['sumsq'][0]
    sumsq_re = stats_re['sumsq'][0]

    q = sumsq_im / sumsq_re
    # flux_ratio = flux_re/flux_im

    snr_im = flux_im / sigma_im
    snr = flux_im / sigma_re

    peak_im_rms = flux_peak_im / rms_im
    peak_re_rms = flux_peak_re / rms_re

    print(' Flux=%.5f Jy/Beam' % flux_im)
    print(' Flux peak (image)=%.5f Jy' % flux_peak_im,
          'Flux peak (residual)=%.5f Jy' % flux_peak_re)
    print(' flux_im/sigma_im=%.5f' % snr_im, 'flux_im/sigma_re=%.5f' % snr)
    print(' rms_im=%.5f' % rms_im, 'rms_re=%.5f' % rms_re)
    print(' flux_peak_im/rms_im=%.5f' % peak_im_rms,
          'flux_peak_re/rms_re=%.5f' % peak_re_rms)
    print(' sumsq_im/sumsq_re=%.5f' % q)


def plot_visibilities(g_vis, name, with_DATA=False, with_MODEL=False,
                      with_CORRECTED=False):
    plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
           antenna=ANTENNAS, spw=SPWS, coloraxis='baseline', avgantenna=True,
           ydatacolumn='corrected-model', avgchannel='64', avgtime='360',
           width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
           plotfile=os.path.dirname(
               g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_corrected-model.jpg')

    # plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',
    #        antenna=ANTENNAS, spw=SPWS, coloraxis='baseline', avgantenna=True,
    #        ydatacolumn='corrected-model', avgchannel='64', avgtime='360',
    #        width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
    #        plotfile=os.path.dirname(
    #            g_vis) + '/selfcal/plots/' + name + '_uvdist_amp_corrected-model.jpg')

    plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
           antenna=ANTENNAS, spw=SPWS, coloraxis='baseline', avgantenna=True,
           ydatacolumn='corrected/model', avgchannel='64', avgtime='360',
           width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
           plotrange=[-1, -1, 0, 5],
           plotfile=os.path.dirname(
               g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_corrected_div_model.jpg')

    # plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',
    #        antenna=ANTENNAS, spw=SPWS, coloraxis='baseline', avgantenna=True,
    #        ydatacolumn='corrected/model', avgchannel='64', avgtime='360',
    #        width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
    #        plotrange=[-1, -1, 0, 5],
    #        plotfile=os.path.dirname(
    #            g_vis) + '/selfcal/plots/' + name + '_uvdist_amp_corrected_div_model.jpg')

    if with_MODEL == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='baseline',
               ydatacolumn='model', avgchannel='64', avgtime='360',
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_model.jpg')

        # plotms(vis=g_vis, xaxis='uvdist', yaxis='amp', avgantenna=True,
        #        antenna=ANTENNAS, spw=SPWS, coloraxis='baseline',
        #        ydatacolumn='model', avgchannel='64', avgtime='360',
        #        width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
        #        plotfile=os.path.dirname(
        #            g_vis) + '/selfcal/plots/' + name + '_uvdist_amp_model.jpg')

        plotms(vis=g_vis, xaxis='freq', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='scan',
               ydatacolumn='model', avgchannel='', avgtime='360',
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_freq_amp_model.jpg')

    if with_DATA == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='baseline',
               ydatacolumn='data', avgchannel='64', avgtime='360',
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               # plotrange=[-1,-1,-1,0.3],
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_data.jpg')
        # plotms(vis=g_vis, xaxis='uvdist', yaxis='amp', avgantenna=True,
        #        antenna=ANTENNAS, spw=SPWS, coloraxis='baseline',
        #        ydatacolumn='data', avgchannel='64', avgtime='360',
        #        width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
        #        # plotrange=[-1,-1,0,0.3],
        #        plotfile=os.path.dirname(
        #            g_vis) + '/selfcal/plots/' + name + '_uvdist_amp_data.jpg')
        plotms(vis=g_vis, xaxis='freq', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='scan',
               ydatacolumn='data', avgchannel='', avgtime='360',
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_freq_amp_data.jpg')

    if with_CORRECTED == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS,
               # plotrange=[-1,-1,0,0.3],
               ydatacolumn='corrected', avgchannel='64', avgtime='360',
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_corrected.jpg')
        # plotms(vis=g_vis, xaxis='uvdist', yaxis='amp', avgantenna=True,
        #        antenna=ANTENNAS, spw=SPWS,
        #        # plotrange=[-1,-1,0,0.3],
        #        ydatacolumn='corrected', avgchannel='64', avgtime='360',
        #        width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
        #        plotfile=os.path.dirname(
        #            g_vis) + '/selfcal/plots/' + name + '_uvdist_amp_corrected.jpg')
        plotms(vis=g_vis, xaxis='freq', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='scan',
               ydatacolumn='corrected', avgchannel='', avgtime='360',
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_freq_amp_corrected.jpg')

    pass


def make_dirty(g_name, field, n_interaction, mask=''):
    '''
    Help function to create the dirty beam image.
    '''
    g_vis = g_name + '.ms'
    niter = 0
    image_dirty = str(n_interaction) + '_dirty_' + os.path.basename(g_name) + '_' + str(
        imsize) + '_' + cell + '_' + str(
        niter) + '.' + weighting + '.' + specmode + '.' + deconvolver + '.' + gridder
    tclean(vis=g_vis,
           imagename=os.path.dirname(g_vis) + '/selfcal/' + image_dirty,
           field=FIELD, gain=gain,
           specmode='mfs', deconvolver=deconvolver, gridder=gridder,
           scales=scales, smallscalebias=smallscalebias,
           imsize=imsize, cell=cell,
           weighting=weighting, robust=robust,
           niter=niter, interactive=interactive,
           pblimit=pblimit,
           mask=mask,
           savemodel='none',
           usepointing=False)

    pass



def start_image(g_name, n_interaction, imsize='2048', cell='0.05asec',
                robust=0.0,
                base_name=None,
                nsigma_automask = '7.0',nsigma_autothreshold='0.1',
                delmodel=True, niter=600,
                opt_args = '',quiet=True,shift=None,
                PLOT=False, datacolumn='DATA',mask=None,
                savemodel=True, uvtaper=[""]):
    '''
    Wsclean wrapper function. It calls wslcean from the command line with some
    predifined arguments. This initial step runs on the DATA column and creates
    the initial model which is used to calculate the initial complex self-gains.
    '''
    g_vis = g_name + '.ms'
    if base_name is None:
        base_name = str(n_interaction)+'_start_image_'
    else:
        base_name = base_name
    if delmodel == True:
        delmod(g_vis)
        clearcal(g_vis)

    os.system("export OPENBLAS_NUM_THREADS=1 && python imaging_with_wsclean.py --f " +
              g_name + " --sx "
              + str(imsize) + " --sy " + str(imsize) + " --niter "
              + str(niter) + " --data " + datacolumn + " --cellsize " + cell
              + ' --nsigma_automask ' + nsigma_automask + ' --mask '+str(mask)
              + ' --nsigma_autothreshold ' + nsigma_autothreshold
              # +' --opt_args '+ opt_args
              + ' --quiet ' + str(quiet)
              + ' --shift ' + str(shift)
              + " --r " + str(robust) + " --t "+str(uvtaper)
              + " --update_model " + str(savemodel) + " --save_basename " + base_name)

    if PLOT == True:
        plot_visibilities(g_vis=g_vis, name=base_name,
                          with_MODEL=True, with_DATA=True)

    pass


def get_tb_data(table, param):
    tb.open(table)
    param_data = tb.getcol(param).ravel()
    tb.close()
    return (param_data)


def make_plot_snr(caltable, cut_off, plot_snr=True, bins=50, density=True,
                  save_fig=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    snr = get_tb_data(caltable, 'SNR')
    plt.figure(figsize=(3, 3))
    if plot_snr:
        plt.hist(snr, bins=bins, density=density, histtype='step')
        # plt.legend( loc='upper right' )
        plt.xlabel('SNR')
        # plt.semilogy()
        plt.semilogx()
        plt.axvline(x=3, color='k', linestyle='--')
        plt.axvline(x=cut_off, color='r', linestyle='--')
        plt.grid()
        if save_fig == True:
            plt.savefig(caltable.replace('.tb', '.jpg'), dpi=300, bbox_inches='tight')
        plt.show()
        # plt.clf()
        # plt.close()

    print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
        stats.percentileofscore(snr, cut_off), ''))


def calibration_table_plot(table, stage='selfcal',
                           table_type='gain_phase', kind='',
                           xaxis='time', yaxis='phase',
                           fields=['']):
    if not os.path.exists(os.path.dirname(table) + '/plots/' + stage):
        os.makedirs(os.path.dirname(table) + '/plots/' + stage)

    if yaxis == 'phase':
        plotrange = [-1, -1, -180, 180]
    else:
        plotrange = [-1, -1, -1, -1]

    if fields == '':
        plotms(vis=table, xaxis=xaxis, yaxis=yaxis, field='',
               gridcols=1, gridrows=1, coloraxis='spw', antenna='', plotrange=plotrange,
               width=800, height=540, dpi=600, overwrite=True, showgui=True,
               plotfile=os.path.dirname(
                   table) + '/plots/' + stage + '/' + table_type + '_' + xaxis + '_' + yaxis + '_field_' + str(
                   'all') + '.jpg')
    else:

        for FIELD in fields:
            plotms(vis=table, xaxis=xaxis, yaxis=yaxis, field=FIELD,
                   # gridcols=4,gridrows=4,coloraxis='spw',antenna='',iteraxis='antenna',
                   # width=2048,height=1280,dpi=256,overwrite=True,showgui=False,
                   gridcols=1, gridrows=1, coloraxis='spw', antenna='',
                   plotrange=plotrange,
                   width=800, height=540, dpi=600, overwrite=True, showgui=False,
                   plotfile=os.path.dirname(
                       table) + '/plots/' + stage + '/' + table_type + '_' + xaxis + '_' + yaxis + '_field_' + str(
                       FIELD) + '.jpg')

    pass

def check_solutions(g_name, field, cut_off=3.0, minsnr=0.01, n_interaction=0, uvrange='',
                    solnorm=solnorm, combine='', calmode='p', gaintype='G',
                    gain_tables_selfcal=[''], special_name='',
                    return_solution_stats=False):
    g_vis = g_name + '.ms'
    minsnr = minsnr
    caltable_int = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_int_minsnr_' + str(
        minsnr) + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'
    # caltable_5 = 'selfcal/selfcal_'+str(n_interaction)+'_'+g_name+'_solint_5.tb'
    caltable_20 = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_48_minsnr_' + str(
        minsnr) + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'
    caltable_40 = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_96_minsnr_' + str(
        minsnr) + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'
    caltable_60 = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_192_minsnr_' + str(
        minsnr) + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'
    caltable_120 = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_384_minsnr_' + str(
        minsnr) + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'
    caltable_inf = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_inf_minsnr_' + str(
        minsnr) + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'

    if not os.path.exists(caltable_int):
        print('>> Performing test-gaincal for solint=int...')
        gaincal(vis=g_vis, caltable=caltable_int, solint='int', refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    # if not os.path.exists(caltable_5):
    # print(  '>> Performing test-gaincal for solint=5s...')
    # gaincal(vis=g_vis,caltable=caltable_5,solint='5s',refant=refant,
    #     calmode=calmode,gaintype=gaintype, minsnr=1,gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_20):
        print('>> Performing test-gaincal for solint=48s...')
        gaincal(vis=g_vis, caltable=caltable_20, solint='48s', refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_40):
        print('>> Performing test-gaincal for solint=96s...')
        gaincal(vis=g_vis, caltable=caltable_40, solint='96s', refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_60):
        print('>> Performing test-gaincal for solint=192s...')
        gaincal(vis=g_vis, caltable=caltable_60, solint='192s', refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_120):
        print('>> Performing test-gaincal for solint=384s...')
        gaincal(vis=g_vis, caltable=caltable_120, solint='384s', refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_inf):
        print('>> Performing test-gaincal for solint=inf...')
        gaincal(vis=g_vis, caltable=caltable_inf, solint='inf', refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)

    def make_plot_check(cut_off=cut_off, return_solution_stats=False):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        snr_int = get_tb_data(caltable_int, 'SNR')
        # snr_5 = get_tb_data(caltable_5,'SNR')
        snr_20 = get_tb_data(caltable_20, 'SNR')
        snr_40 = get_tb_data(caltable_40, 'SNR')
        snr_60 = get_tb_data(caltable_60, 'SNR')
        snr_120 = get_tb_data(caltable_120, 'SNR')
        snr_inf = get_tb_data(caltable_inf, 'SNR')

        plt.figure()
        plt.hist(snr_int, bins=50, density=True, histtype='step', label='int')
        # plt.hist( snr_5, bins=50, density=True, histtype='step', label='5 seconds' )
        plt.hist(snr_20, bins=50, density=True, histtype='step', label='48 seconds')
        plt.hist(snr_40, bins=50, density=True, histtype='step', label='96 seconds')
        plt.hist(snr_60, bins=50, density=True, histtype='step', label='192 seconds')
        plt.hist(snr_120, bins=50, density=True, histtype='step', label='384 seconds')
        plt.hist(snr_inf, bins=50, density=True, histtype='step', label='inf')
        plt.legend(loc='upper right')
        plt.xlabel('SNR')
        plt.semilogx()
        plt.savefig(os.path.dirname(g_name) + '/selfcal/plots/' + str(n_interaction) +
                    '_' + os.path.basename(
            g_name) + '_combine' + combine + '_gtype_' + gaintype
                    + special_name + '_gain_solutions_comparisons_norm.pdf')
        # plt.clf()
        # plt.close()
        plt.show()

        plt.figure()
        plt.figure()
        plt.hist(snr_int, bins=50, density=False, histtype='step', label='int')
        # plt.hist( snr_5, bins=50, density=False, histtype='step', label='5 seconds' )
        plt.hist(snr_20, bins=50, density=False, histtype='step', label='48 seconds')
        plt.hist(snr_40, bins=50, density=False, histtype='step', label='96 seconds')
        plt.hist(snr_60, bins=50, density=False, histtype='step', label='192 seconds')
        plt.hist(snr_120, bins=50, density=False, histtype='step',
                 label='384 seconds')
        plt.hist(snr_inf, bins=50, density=False, histtype='step', label='inf')
        plt.legend(loc='upper right')
        plt.xlabel('SNR')
        plt.semilogx()
        plt.savefig(os.path.dirname(g_name) + '/selfcal/plots/' + str(n_interaction) +
                    '_' + os.path.basename(g_name) + '_combine' + combine +
                    '_gtype_' + gaintype + special_name +
                    '_gain_solutions_comparisons.pdf')

        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_int, cut_off), 'int'))
        # print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format(
        #     stats.percentileofscore( snr_5, cut_off ), '5s' ) )
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_20, cut_off), '48s'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_40, cut_off), '96s'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_60, cut_off), '192s'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_120, cut_off), '384s'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_inf, cut_off), 'inf'))

        plt.show()
        # print('################################')
        # print(np.mean(snr_int))
        # print('################################')
        # print(stats.percentileofscore(snr_int, cut_off))
        SNRs = [np.array(snr_int),
                           np.array(snr_20),
                           np.array(snr_40),
                           np.array(snr_60),
                           np.array(snr_120),
                           np.array(snr_inf)]
        percentiles_SNRs = np.asarray([stats.percentileofscore(snr_int, cut_off),
                                       stats.percentileofscore(snr_20, cut_off),
                                       stats.percentileofscore(snr_40, cut_off),
                                       stats.percentileofscore(snr_60, cut_off),
                                       stats.percentileofscore(snr_120, cut_off),
                                       stats.percentileofscore(snr_inf, cut_off)])
        if return_solution_stats:
            return (SNRs, percentiles_SNRs)
        else:
            pass
        # plt.clf()
        # plt.close()

    def compare_phase_variation():
        plotms(caltable_int, antenna='Mk2', scan='', yaxis='phase')

        plotms(caltable_40, antenna='', scan='', yaxis='phase', plotindex=1,
               clearplots=False, customsymbol=True, symbolsize=20,
               symbolcolor='ff0000', symbolshape='circle')

        plotms(caltable_20, antenna='', scan='', yaxis='phase', plotindex=2,
               clearplots=False, customsymbol=True, symbolsize=12,
               symbolcolor='green', symbolshape='square')

        plotms(caltable_inf, antenna='', scan='', yaxis='phase', plotindex=3,
               clearplots=False, customsymbol=True, symbolsize=8,
               symbolcolor='yellow', symbolshape='square')

        plotms(caltable_60, antenna='', scan='', yaxis='phase', plotindex=4,
               clearplots=False, customsymbol=True, symbolsize=4,
               symbolcolor='purple', symbolshape='square',
               width=1600, height=1080, showgui=True, overwrite=True,
               plotfile=os.path.dirname(g_name) + '/selfcal/plots/' + str(
                   n_interaction) +
                        '_' + os.path.basename(g_name) + '_combine' + combine +
                        '_gtype_' + gaintype + special_name +
                        '_phase_variation_intervals.jpg')

    def compare_amp_variation():
        plotms(caltable_int, antenna='', scan='', yaxis='amp')

        plotms(caltable_40, antenna='', scan='', yaxis='amp', plotindex=1,
               clearplots=False, customsymbol=True, symbolsize=20,
               symbolcolor='ff0000', symbolshape='circle')

        plotms(caltable_20, antenna='', scan='', yaxis='amp', plotindex=2,
               clearplots=False, customsymbol=True, symbolsize=12,
               symbolcolor='green', symbolshape='square')

        plotms(caltable_inf, antenna='', scan='', yaxis='amp', plotindex=3,
               clearplots=False, customsymbol=True, symbolsize=8,
               symbolcolor='yellow', symbolshape='square')

        plotms(caltable_60, antenna='', scan='', yaxis='amp', plotindex=4,
               clearplots=False, customsymbol=True, symbolsize=4,
               symbolcolor='purple', symbolshape='square',
               width=1600, height=1080, showgui=True, overwrite=True,
               plotfile=os.path.dirname(g_name) + '/selfcal/plots/' + str(
                   n_interaction) +
                        '_' + os.path.basename(
                   g_name) + special_name + '_amp_variation_intervals.jpg')

    #
    # def plot_gains():
    #     plotms(caltable_int,antenna='ea01',scan='',yaxis='phase',
    #         gridrows=5,gridcols=5,iteraxis='antenna',coloraxis='spw')

    if return_solution_stats == True:
        SNRs, percentiles_SNRs = make_plot_check(cut_off=cut_off,
                                                 return_solution_stats=return_solution_stats)
    else:
        make_plot_check(cut_off=cut_off)
    compare_phase_variation()
    if calmode == 'ap':
        compare_amp_variation()

    if return_solution_stats == True:
        return (SNRs, percentiles_SNRs)
    else:
        pass


def run_wsclean(g_name, n_interaction, imsize='2048', cell='0.05asec',
                robust=0.5,base_name=None,
                savemodel=True,shift=None,
                nsigma_automask='8.0', nsigma_autothreshold='1.0',
                datacolumn='CORRECTED',mask=None,
                niter=1000,quiet=True,with_multiscale=False,
                uvtaper=[], PLOT=False):


    g_vis = g_name + '.ms'

    if base_name is None:
        base_name  = str(n_interaction)+'_update_model_image_'
    else:
        base_name = base_name



    os.system("export OPENBLAS_NUM_THREADS=1 && python imaging_with_wsclean.py --f " +
              g_name + " --sx "
              + str(imsize) + " --sy " + str(imsize) + " --niter "
              + str(niter) + " --data " + datacolumn + " --cellsize " + cell
              + ' --nsigma_automask ' + nsigma_automask + ' --mask '+str(mask)
              + ' --nsigma_autothreshold ' + nsigma_autothreshold
              # +' --opt_args '+ opt_args
              +' --quiet '+ str(quiet) + ' --with_multiscale '+str(with_multiscale)
              + ' --shift ' + str(shift)
              + " --r " + str(robust) + " --t "+str(uvtaper)
              + " --update_model " + str(savemodel) + " --save_basename " + base_name)


    if PLOT == True:
        plot_visibilities(g_vis=g_vis, name=base_name,
                          with_MODEL=True, with_CORRECTED=True)

    pass


def self_gain_cal(g_name, n_interaction, gain_tables=[],
                  combine=combine, solnorm=False,
                  spwmap=[],uvrange='',
                  minsnr=5.0, solint='inf', gaintype='G', calmode='p',
                  action='apply', flagbackup=True, PLOT=False, special_name=''):
    g_vis = g_name + '.ms'

    cal_basename = '_selfcal_'
    base_name =  str(n_interaction)+'_update_model_image_'+cal_basename

    if calmode == 'p':
        cal_basename = cal_basename + 'phase_'
        base_name = base_name + 'phase_'
    if calmode == 'ap' or calmode == 'a':
        cal_basename = cal_basename + 'ampphase_'
        base_name = base_name + 'ampphase_'
    if gain_tables != []:
        cal_basename = cal_basename + 'incremental_'

    caltable = (os.path.dirname(g_name) + '/selfcal/' + str(n_interaction) \
                + cal_basename + os.path.basename(g_name) \
                + '_' + '_solint_' + solint + '_minsnr_' + str(minsnr) +
                '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb')
    if not os.path.exists(caltable):
        if calmode == 'ap' or calmode == 'a':
            solonrm = True
        else:
            solnorm = False
        gaincal(vis=g_vis, field=FIELD, caltable=caltable, spwmap=spwmap,
                solint=solint, gaintable=gain_tables, combine=combine,
                refant=refant, calmode=calmode, gaintype=gaintype,
                uvrange=uvrange,
                minsnr=minsnr, solnorm=solnorm,minblperant=minblperant)
    else:
        print(' => Using existing caltable with same parameters asked.')
        print(' => Not computing again...')

    calibration_table_plot(table=caltable,
                           fields='', yaxis='phase',
                           table_type=str(
                               n_interaction) + '_selfcal_phase_' + os.path.basename(
                               g_name) +
                                      '_solint_' + solint + '_minsnr_' + str(
                               minsnr) + '_combine' + combine +
                                      '_gtype_' + gaintype + special_name)

    if calmode == 'ap' or calmode == 'a':
        calibration_table_plot(table=caltable,
                               fields='', yaxis='amp',
                               table_type=str(n_interaction) + '_selfcal_ampphase_' +
                                          os.path.basename(g_name) + '_solint_' + solint +
                                          '_minsnr_' + str(
                                   minsnr) + '_combine' + combine +
                                          '_gtype_' + gaintype + special_name)

    make_plot_snr(caltable=caltable, cut_off=minsnr,
                  plot_snr=True, bins=50, density=True, save_fig=True)

    if action == 'apply':
        if flagbackup == True:
            print('     => Creating new flagbackup file before mode ',
                  calmode, ' selfcal ...')
            flagmanager(vis=g_vis, mode='save',
                        versionname='before_selfcal_mode_' + calmode,
                        comment='Before selfcal apply.')

        gain_tables.append(caltable)
        print('     => Reporting data flagged before selfcal '
              'apply interaction', n_interaction, '...')
        summary_bef = flagdata(vis=g_vis, field=FIELD, mode='summary')
        report_flag(summary_bef, 'field')

        applycal(vis=g_vis, gaintable=gain_tables, spwmap=spwmap,
                 flagbackup=False, calwt=True)

        print('     => Reporting data flagged after selfcal '
              'apply interaction', n_interaction, '...')
        summary_aft = flagdata(vis=g_vis, field=FIELD, mode='summary')
        report_flag(summary_aft, 'field')

        if PLOT == True:
            plot_visibilities(g_vis=g_vis, name=base_name,
                              with_CORRECTED=True, with_MODEL=False, with_DATA=False)

    return (gain_tables)



def self_gain_blcal(g_name, field, n_interaction, gain_tables=[],
                  combine=combine, solnorm=False, uvtaper=[], uvrange='',
                  niter=500, spwmap=[],
                  specmode='mfs', deconvolver='mtmfs', ext='', gridder='standard',
                  minsnr=5.0, solint='inf', gaintype='G', calmode='p',
                  action='apply', flagbackup=True, PLOT=False, special_name=''):
    g_vis = g_name + '.ms'

    cal_basename = '_selfcal_blcal_'
    base_name = '_update_model_image_'

    if calmode == 'p':
        cal_basename = cal_basename + 'phase_'
        base_name = base_name + 'phase_'
    if calmode == 'ap':
        cal_basename = cal_basename + 'ampphase_'
        base_name = base_name + 'ampphase_'
    if gain_tables != []:
        cal_basename = cal_basename + 'incremental_'

    if interactive == True:
        base_name = base_name + 'interactive_'
        cal_basename = cal_basename + 'interactive_'
    if usemask == 'auto-multithresh':
        base_name = base_name + 'auto'
        cal_basename = cal_basename + 'auto'

    image_update_model = str(n_interaction) + base_name + os.path.basename(g_name) \
                         + '_' + str(imsize) + '_' + cell + '_' + str(
        niter) + '.' + weighting \
                         + '.' + str(
        robust) + '.' + specmode + '.' + deconvolver + '.' + gridder

    caltable = (os.path.dirname(g_name) + '/selfcal/' + str(n_interaction) \
                + cal_basename + os.path.basename(g_name) \
                + '_' + '_solint_' + solint + '_minsnr_' + str(minsnr) +
                '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb')
    if not os.path.exists(caltable):
        if calmode == 'ap' or calmode == 'a':
            solonrm = True
        else:
            solnorm = False
        blcal(vis=g_vis, field=FIELD, caltable=caltable, spwmap=spwmap,
                solint=solint, gaintable=gain_tables, combine=combine,
                calmode=calmode,
                solnorm=solnorm)
    else:
        print(' => Using existing caltable with same parameters asked.')
        print(' => Not computing again...')

    calibration_table_plot(table=caltable,
                           fields='', yaxis='phase',
                           table_type=str(
                               n_interaction) + '_selfcal_phase_' + os.path.basename(
                               g_name) +
                                      '_solint_' + solint + '_minsnr_' + str(
                               minsnr) + '_combine' + combine +
                                      '_gtype_' + gaintype + special_name)

    if calmode == 'ap' or calmode == 'a':
        calibration_table_plot(table=caltable,
                               fields='', yaxis='amp',
                               table_type=str(n_interaction) + '_selfcal_ampphase_' +
                                          os.path.basename(g_name) + '_solint_' + solint +
                                          '_minsnr_' + str(
                                   minsnr) + '_combine' + combine +
                                          '_gtype_' + gaintype + special_name)

    make_plot_snr(caltable=caltable, cut_off=minsnr,
                  plot_snr=True, bins=50, density=True, save_fig=True)

    if action == 'apply':
        if flagbackup == True:
            print('     => Creating new flagbackup file before mode ',
                  calmode, ' selfcal ...')
            flagmanager(vis=g_vis, mode='save',
                        versionname='before_selfcal_mode_' + calmode,
                        comment='Before selfcal apply.')

        gain_tables.append(caltable)
        print('     => Reporting data flagged before selfcal '
              'apply interaction', n_interaction, '...')
        summary_bef = flagdata(vis=g_vis, field=FIELD, mode='summary')
        report_flag(summary_bef, 'field')

        applycal(vis=g_vis, gaintable=gain_tables, spwmap=spwmap,
                 flagbackup=False, calwt=True)

        print('     => Reporting data flagged after selfcal '
              'apply interaction', n_interaction, '...')
        summary_aft = flagdata(vis=g_vis, field=FIELD, mode='summary')
        report_flag(summary_aft, 'field')

        if PLOT == True:
            plot_visibilities(g_vis=g_vis, name=image_update_model,
                              with_CORRECTED=True, with_MODEL=False, with_DATA=False)

    return (gain_tables)



run_mode = 'jupyter'
# run_mode = 'terminal'


if run_mode == 'jupyter':
    print('selfcal script is not doing anything, you can use it on a '
          'jupyter notebook. For that, you have to manually '
          'set your variable names and steps taken to selfcalibrate your data.')
