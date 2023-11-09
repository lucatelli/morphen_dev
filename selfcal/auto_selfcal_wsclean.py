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
import sys
sys.path.append('../libs/')
import libs as mlibs
import glob
from casatasks import *
import numpy as np
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
# refant = ''
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
           correlation='LL,RR',
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
           correlation='LL,RR',
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
               correlation='LL,RR',
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
               correlation='LL,RR',
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_freq_amp_model.jpg')

    if with_DATA == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='baseline',
               ydatacolumn='data', avgchannel='64', avgtime='360',
               correlation='LL,RR',
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
               correlation='LL,RR',
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_freq_amp_data.jpg')

    if with_CORRECTED == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS,
               # plotrange=[-1,-1,0,0.3],
               ydatacolumn='corrected', avgchannel='64', avgtime='360',
               correlation='LL,RR',
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
               correlation='LL,RR',
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
               # correlation='LL,RR',
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
                   # correlation='LL,RR',
                   plotfile=os.path.dirname(
                       table) + '/plots/' + stage + '/' + table_type + '_' + xaxis + '_' + yaxis + '_field_' + str(
                       FIELD) + '.jpg')

    pass

def check_solutions(g_name, field, cut_off=3.0, minsnr=0.01, n_interaction=0, uvrange='',
                    solnorm=solnorm, combine='', calmode='p', gaintype='G',solint_factor=1.0,
                    gain_tables_selfcal=[''], special_name='',refant = '', minblperant=4,
                    return_solution_stats=False):
    g_vis = g_name + '.ms'
    minsnr = minsnr
    solint_template = np.asarray([24,48,96,192,384])
    solints = solint_template * solint_factor

    caltable_int = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_int_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'

    caltable_1 = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+str(int(solints[0]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'

    caltable_2 = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+str(int(solints[1]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'

    caltable_3 = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+str(int(solints[2]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'

    caltable_4 = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+str(int(solints[3]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'

    caltable_5 = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+str(int(solints[4]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'

    caltable_inf = os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_inf_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' + gaintype + special_name + '.tb'


    if not os.path.exists(caltable_int):
        print('>> Performing test-gaincal for solint=int...')
        gaincal(vis=g_vis, caltable=caltable_int, solint='int', refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_1):
        print('>> Performing test-gaincal for solint='+str(solints[0])+'s...')
        gaincal(vis=g_vis, caltable=caltable_1, solint=str(solints[0])+'s',
                refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_2):
        print('>> Performing test-gaincal for solint='+str(solints[1])+'s...')
        gaincal(vis=g_vis, caltable=caltable_2, solint=str(solints[1])+'s',
                refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_3):
        print('>> Performing test-gaincal for solint='+str(solints[2])+'s...')
        gaincal(vis=g_vis, caltable=caltable_3, solint=str(solints[2])+'s',
                refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_4):
        print('>> Performing test-gaincal for solint='+str(solints[3])+'s...')
        gaincal(vis=g_vis, caltable=caltable_4, solint=str(solints[3])+'s',
                refant=refant,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_5):
        print('>> Performing test-gaincal for solint='+str(solints[4])+'s...')
        gaincal(vis=g_vis, caltable=caltable_5, solint=str(solints[4])+'s',
                refant=refant,
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
        snr_1 = get_tb_data(caltable_1, 'SNR')
        snr_2 = get_tb_data(caltable_2, 'SNR')
        snr_3 = get_tb_data(caltable_3, 'SNR')
        snr_4 = get_tb_data(caltable_4, 'SNR')
        snr_5 = get_tb_data(caltable_5, 'SNR')
        snr_inf = get_tb_data(caltable_inf, 'SNR')

        plt.figure()
        plt.hist(snr_int, bins=50, density=True, histtype='step',
                 label='int')
        plt.hist(snr_1, bins=50, density=True, histtype='step',
                 label=str(solints[0])+' seconds')
        plt.hist(snr_2, bins=50, density=True, histtype='step',
                 label=str(solints[1])+' seconds')
        plt.hist(snr_3, bins=50, density=True, histtype='step',
                 label=str(solints[2])+' seconds')
        plt.hist(snr_4, bins=50, density=True, histtype='step',
                 label=str(solints[3])+' seconds')
        plt.hist(snr_5, bins=50, density=True, histtype='step',
                 label=str(solints[4])+' seconds')
        plt.hist(snr_inf, bins=50, density=True, histtype='step',
                 label='inf')
        plt.legend(loc='upper right')
        plt.xlabel('SNR')
        plt.semilogx()
        plt.savefig(os.path.dirname(g_name) + '/selfcal/plots/' + str(n_interaction) +
                    '_' + os.path.basename(
            g_name) + '_calmode' + calmode + '_combine' + combine + '_gtype_' + gaintype
                    + special_name + '_gain_solutions_comparisons_norm.pdf')
        # plt.clf()
        # plt.close()
        plt.show()

        plt.figure()
        plt.figure()
        plt.hist(snr_int, bins=50, density=False, histtype='step',
                 label='int')
        plt.hist(snr_1, bins=50, density=False, histtype='step',
                 label=str(solints[0])+' seconds')
        plt.hist(snr_2, bins=50, density=False, histtype='step',
                 label=str(solints[1])+' seconds')
        plt.hist(snr_3, bins=50, density=False, histtype='step',
                 label=str(solints[2])+' seconds')
        plt.hist(snr_4, bins=50, density=False, histtype='step',
                 label=str(solints[3])+' seconds')
        plt.hist(snr_5, bins=50, density=False, histtype='step',
                 label=str(solints[4])+' seconds')
        plt.hist(snr_inf, bins=50, density=False, histtype='step',
                 label='inf')
        plt.legend(loc='upper right')
        plt.xlabel('SNR')
        plt.semilogx()
        plt.savefig(os.path.dirname(g_name) + '/selfcal/plots/' + str(n_interaction) +
                    '_' + os.path.basename(g_name) + '_calmode' + calmode + '_combine' + combine +
                    '_gtype_' + gaintype + special_name +
                    '_gain_solutions_comparisons.pdf')

        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_int, cut_off), 'int'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_1, cut_off), str(solints[0])+' s'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_2, cut_off), str(solints[1])+' s'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_3, cut_off), str(solints[2])+' s'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_4, cut_off), str(solints[3])+' s'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_5, cut_off), str(solints[4])+' s'))
        print('P(<=' + str(cut_off) + ') = {0}  ({1})'.format(
            stats.percentileofscore(snr_inf, cut_off), 'inf'))

        plt.show()
        # print('################################')
        # print(np.mean(snr_int))
        # print('################################')
        # print(stats.percentileofscore(snr_int, cut_off))
        SNRs = [np.array(snr_int),
                           np.array(snr_1),
                           np.array(snr_2),
                           np.array(snr_3),
                           np.array(snr_4),
                           np.array(snr_5),
                           np.array(snr_inf)]
        percentiles_SNRs = np.asarray([stats.percentileofscore(snr_int, cut_off),
                                       stats.percentileofscore(snr_1, cut_off),
                                       stats.percentileofscore(snr_2, cut_off),
                                       stats.percentileofscore(snr_3, cut_off),
                                       stats.percentileofscore(snr_4, cut_off),
                                       stats.percentileofscore(snr_5, cut_off),
                                       stats.percentileofscore(snr_inf, cut_off)])
        if return_solution_stats:
            return (SNRs, percentiles_SNRs)
        else:
            pass
        # plt.clf()
        # plt.close()

    def compare_phase_variation():
        plotms(caltable_int, antenna='Mk2', scan='', yaxis='phase')

        plotms(caltable_3, antenna='', scan='', yaxis='phase', plotindex=1,
               clearplots=False, customsymbol=True, symbolsize=20,
               symbolcolor='ff0000', symbolshape='circle')

        plotms(caltable_2, antenna='', scan='', yaxis='phase', plotindex=2,
               clearplots=False, customsymbol=True, symbolsize=12,
               symbolcolor='green', symbolshape='square')

        plotms(caltable_inf, antenna='', scan='', yaxis='phase', plotindex=3,
               clearplots=False, customsymbol=True, symbolsize=8,
               symbolcolor='yellow', symbolshape='square')

        plotms(caltable_4, antenna='', scan='', yaxis='phase', plotindex=4,
               clearplots=False, customsymbol=True, symbolsize=4,
               symbolcolor='purple', symbolshape='square',
               width=1600, height=1080, showgui=True, overwrite=True,
               plotfile=os.path.dirname(g_name) + '/selfcal/plots/' + str(
                   n_interaction) +
                        '_' + os.path.basename(g_name) + '_combine' + '_calmode' + calmode + combine +
                        '_gtype_' + gaintype + special_name +
                        '_phase_variation_intervals.jpg')

    def compare_amp_variation():
        plotms(caltable_int, antenna='', scan='', yaxis='amp')

        plotms(caltable_3, antenna='', scan='', yaxis='amp', plotindex=1,
               clearplots=False, customsymbol=True, symbolsize=20,
               symbolcolor='ff0000', symbolshape='circle')

        plotms(caltable_2, antenna='', scan='', yaxis='amp', plotindex=2,
               clearplots=False, customsymbol=True, symbolsize=12,
               symbolcolor='green', symbolshape='square')

        plotms(caltable_inf, antenna='', scan='', yaxis='amp', plotindex=3,
               clearplots=False, customsymbol=True, symbolsize=8,
               symbolcolor='yellow', symbolshape='square')

        plotms(caltable_4, antenna='', scan='', yaxis='amp', plotindex=4,
               clearplots=False, customsymbol=True, symbolsize=4,
               symbolcolor='purple', symbolshape='square',
               width=1600, height=1080, showgui=True, overwrite=True,
               plotfile=os.path.dirname(g_name) + '/selfcal/plots/' + str(
                   n_interaction) +
                        '_' + os.path.basename(g_name) +
                        '_combine' + '_calmode' + calmode + combine +
                        '_gtype_' + gaintype + special_name +
                        '_amp_variation_intervals.jpg')

    #
    # def plot_gains():
    #     plotms(caltable_int,antenna='ea01',scan='',yaxis='phase',
    #         gridrows=5,gridcols=5,iteraxis='antenna',coloraxis='spw')

    if return_solution_stats == True:
        SNRs, percentiles_SNRs = \
            make_plot_check(cut_off=cut_off,
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
                  spwmap=[],uvrange='',append=False,solmode='',#L1R
                  minsnr=5.0, solint='inf', gaintype='G', calmode='p',
                  interp = '',refant = '', minblperant = 4,
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
                uvrange=uvrange,append=append,solmode=solmode,interp = interp,
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
                 interp = interp,
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


def run_rflag(g_vis, display='report', action='calculate',
              timedevscale=4.0, freqdevscale=4.0, winsize=7, datacolumn='corrected'):
    if action == 'apply':
        print('Flag statistics before rflag:')
        summary_before = flagdata(vis=g_vis, field='', mode='summary')
        report_flag(summary_before, 'field')
        flagmanager(vis=g_name + '.ms', mode='save', versionname='seflcal_before_rflag',
                    comment='Before rflag at selfcal step.')
    flagdata(vis=g_vis, mode='rflag', field='', spw='', display=display,
             datacolumn=datacolumn, ntime='scan', combinescans=True,
             extendflags=False,
             winsize=winsize,
             timedevscale=timedevscale, freqdevscale=freqdevscale,
             flagnearfreq=False, flagneartime=False, growaround=True,
             action=action, flagbackup=False, savepars=True
             )

    if action == 'apply':
        flagdata(vis=g_vis, field='', spw='',
                 datacolumn='data',
                 mode='extend', action=action, display='report',
                 flagbackup=False, growtime=75.0,
                 growfreq=75.0, extendpols=False)

    if action == 'apply':
        flagmanager(vis=g_name + '.ms', mode='save', versionname='seflcal_after_rflag',
                    comment='After rflag at selfcal step.')
        try:
            statwt(vis=g_vis, statalg='chauvenet', timebin='60s', datacolumn='corrected')
        except:
            statwt(vis=g_vis, statalg='chauvenet', timebin='60s', datacolumn='data')

        print('Flag statistics after rflag:')
        summary_after = flagdata(vis=g_vis, field='', mode='summary')
        report_flag(summary_after, 'field')

def find_refant(msfile, field,tablename):
    """
    This function comes from the e-MERLIN CASA Pipeline.
    """
    # Find phase solutions per scan:
    # tablename = calib_dir +
    gaincal(vis=msfile,
            caltable=tablename,
            field=field,
            refantmode='flex',
            solint = 'inf',
            minblperant = 2,
            gaintype = 'G',
            calmode = 'p')
    # find_casa_problems()
    # Read solutions (phases):
    tb.open(tablename+'/ANTENNA')
    antenna_names = tb.getcol('NAME')
    tb.close()
    tb.open(tablename)
    antenna_ids = tb.getcol('ANTENNA1')
    #times  = tb.getcol('TIME')
    flags = tb.getcol('FLAG')
    phases = np.angle(tb.getcol('CPARAM'))
    snrs = tb.getcol('SNR')
    tb.close()
    # Analyse number of good solutions:
    good_frac = []
    good_snrs = []
    for i, ant_id in enumerate(np.unique(antenna_ids)):
        cond = antenna_ids == ant_id
        #t = times[cond]
        f = flags[0,0,:][cond]
        p = phases[0,0,:][cond]
        snr = snrs[0,0,:][cond]
        frac =  1.0*np.count_nonzero(~f)/len(f)*100.
        snr_mean = np.nanmean(snr[~f])
        good_frac.append(frac)
        good_snrs.append(snr_mean)
    sort_idx = np.argsort(good_frac)[::-1]
    print('Antennas sorted by % of good solutions:')
    for i in sort_idx:
        print('{0:3}: {1:4.1f}, <SNR> = {2:4.1f}'.format(antenna_names[i],
                                                               good_frac[i],
                                                               good_snrs[i]))
    if good_frac[sort_idx[0]] < 90:
        print('Small fraction of good solutions with selected refant!')
        print('Please inspect antennas to select optimal refant')
        print('You may want to use refantmode= flex" in default_params')
    pref_ant = antenna_names[sort_idx]
    # if 'Lo' in antenna_names:
    #     priorities = ['Pi','Da','Kn','De','Cm']
    # else:
    #     priorities = ['Mk2','Pi','Da','Kn', 'Cm', 'De']
    # refant = ','.join([a for a in pref_ant if a in priorities])
    pref_ant_list = ','.join(list(pref_ant))
    return pref_ant_list


# run_mode = 'jupyter'
run_mode = 'terminal'
if run_mode == 'terminal':

    """
    If running this in a terminal, you can safely set quiet=False. This is 
    refers to the wsclean quiet parameter. If running this code in a Jupyter Notebook, 
    please set quiet=True. Otherwise, jupyter can crash due to the very long 
    output cells. 
    """
    quiet = False

    steps = [
        'startup',
        'save_init_flags',
        # 'fov_image',
        # # 'run_rflag_init',
        'test_image',
        'select_refant',
        '0',#initial test selfcal step
        # '1',#start of the first trial of selfcal, phase only (p)
        # '2',#continue first trial, use gain table from step 1; can be p or ap.
        # # 'run_rflag_final',
        # # '3',
        # '4',
    ]

    path = ('/media/sagauga/galnet/LIRGI_Sample/VLA-Archive/A_config/23A-324/C_band'
            '/MCG12/selfcalibration/')
    vis_list = ['MCG12-02-001.avg12s.calibrated']
    proj_name = ''
    for field in vis_list:
        g_name = path + field + proj_name
        g_vis = g_name + '.ms'
        # refant = ''# 'ea18,ea09'

        try:
            steps_performed
        except NameError:
            steps_performed = []

        if 'startup' in steps and 'startup' not in steps_performed:
            """
            Create basic directory structure for saving stuff.
            """
            print('==> Creating basic directory structure.')
            if not os.path.exists(path + 'selfcal/'):
                os.makedirs(path + 'selfcal/')
            if not os.path.exists(path + 'selfcal/plots'):
                os.makedirs(path + 'selfcal/plots')
            image_list = {}
            residual_list = {}
            model_list = {}
            image_statistics = {}
            gain_tables_applied = {}
            steps_performed = []
            # start the CASA logger (open the window).
            import casalogger.__main__
            steps_performed.append('startup')

        if 'save_init_flags' in steps and 'save_init_flags' not in steps_performed:
            """
            Create a backup file of the flags; run statwt.
            """
            if not os.path.exists(g_name + '.ms.flagversions/flags.Original/'):
                print("     ==> Creating backup flags file 'Original'...")
                flagmanager(vis=g_name + '.ms', mode='save', versionname='Original',
                            comment='Original flags.')

                print("     ==> Running statwt.")

                if not os.path.exists(g_name + '.ms.flagversions/flags.statwt_1/'):
                    statwt(vis=g_vis, statalg='chauvenet', timebin='60s', datacolumn='data')
            else:
                print("     ==> Skipping flagging backup init (exists).")
                print("     ==> Restoring flags to original...")
                flagmanager(vis=g_name + '.ms', mode='restore', versionname='Original')
                # if not os.path.exists(g_name + '.ms.flagversions/flags.statwt_1/'):
                #     print("     ==> Running statwt.")
                #     statwt(vis=g_vis, statalg='chauvenet', timebin='60s', datacolumn='data')
            print(" ==> Amount of data flagged at the start of selfcal.")
            summary = flagdata(vis=g_name + '.ms', field='', mode='summary')
            report_flag(summary, 'field')
            steps_performed.append('save_init_flags')


        if 'startup' not in steps_performed:
            """
            In case you rerun the code without restarting the kernel or re-starting 
            the selfcal without running the startup step.
            """
            image_list = {}
            residual_list = {}
            model_list = {}
            image_statistics = {}
            gain_tables_applied = {}
            import casalogger.__main__

            steps_performed = []

        if 'fov_image' in steps and 'fov_image' not in steps_performed:
            """
            Create a FOV image.
            """
            niter = 10000
            robust = 0.5  # or 0.5 if lots of extended emission.
            run_wsclean(g_name, imsize=1024 * 7, cell='0.2asec',
                        robust=robust, base_name='FOV_phasecal_image',
                        nsigma_automask='8.0', nsigma_autothreshold='3.0',
                        n_interaction='0', savemodel=False, quiet=False,
                        datacolumn='DATA', shift=None,
                        # shift="'18:34:46.454 +059.47.32.191'",
                        # uvtaper=['0.05arcsec'],
                        niter=niter,
                        PLOT=False)
            file_list = glob.glob(f"{path}*MFS-image.fits")
            file_list.sort(key=os.path.getmtime, reverse=False)
            image_list['FOV_image'] = file_list[-1]
            file_list = glob.glob(f"{path}*MFS-image.fits")
            file_list.sort(key=os.path.getmtime, reverse=False)
            image_list['FOV_residual'] = file_list[-1]
            file_list = glob.glob(f"{path}*MFS-model.fits")
            file_list.sort(key=os.path.getmtime, reverse=False)
            image_list['FOV_model'] = file_list[-1]
            steps_performed.append('fov_image')

        if 'run_rflag_init' in steps:
            """
            Run automatic rflag on the data before selfcalibration.
            """
            run_rflag(g_vis, display='report', action='apply',
                      timedevscale=4.0, freqdevscale=4.0, winsize=7, datacolumn='data')
            steps_performed.append('run_rflag_init')


        """
        This is the moment we define global image/cleaning properties. 
        These will be used in the subsequent steps of selfcalibration.
        """
        ########################################
        imsize = 1024*3
        cell = '0.066asec'
        FIELD_SHIFT = None
        ########################################

        if 'test_image' in steps and 'test_image' not in steps_performed:
        # if 'test_image' in steps and 'test_image':
            """
            After creating a FOV image, or checking info about other sources in the 
            field (e.g. NVSS, FIRST, etc), you may want to create a basic 
            image to that center (or None) to see how the image (size) 
            will accomodate the source(s). This setting will be used in all the
            subsequent steps of selfcalibration. 
            
            Note also that masks are not used in this step, but the image will be used 
            to create a mask for the next step, which is the first step of 
            selfcalibration. 
            """
            niter = 10000
            robust = 0.5
            run_wsclean(g_name, imsize=imsize, cell=cell,
                        robust=robust, base_name='phasecal_image',
                        nsigma_automask='10.0', nsigma_autothreshold='6.0',
                        n_interaction='0', savemodel=False, quiet=quiet,
                        datacolumn='DATA', shift=FIELD_SHIFT,
                        # uvtaper=['0.05arcsec'],
                        niter=niter,
                        PLOT=False)

            file_list = glob.glob(f"{path}*MFS-image.fits")
            file_list.sort(key=os.path.getmtime, reverse=False)
            print(file_list)
            try:
                image_list['test_image'] = file_list[-1]
            except:
                image_list['test_image'] = file_list
            # file_list = glob.glob(f"{path}*MFS-image.fits")
            # file_list.sort(key=os.path.getmtime, reverse=False)
            image_list['test_residual'] = image_list['test_image'].replace(
                'MFS-image.fits','MFS-residual.fits')
            # file_list = glob.glob(f"{path}*MFS-model.fits")
            # file_list.sort(key=os.path.getmtime, reverse=False)
            image_list['test_model'] = image_list['test_image'].replace(
                'MFS-image.fits','MFS-model.fits')


            level_stats = mlibs.level_statistics(image_list['test_image'])
            image_stats = mlibs.get_image_statistics(imagename=image_list['test_image'],
                                                     dic_data=level_stats)

            image_statistics['test_image'] = image_stats
            if 'test_image' not in steps_performed:
                steps_performed.append('test_image')

        if 'select_refant' in steps and 'select_refant' not in steps_performed:
            print(' ==> Estimating order of best referent antennas...')
            tablename_refant = os.path.dirname(g_name) + '/selfcal/find_refant.phase'
            refant = find_refant(msfile=g_vis, field='',
                                 tablename=tablename_refant)
            print(' ==> Preferential reference antenna order = ', refant)
            steps_performed.append('select_refant')


        if '0' in steps:
            iteration = '0'
            ############################################################################
            #### 0. Zero interaction. Use a small/negative robust parameter,        ####
            ####    to find the bright/compact emission(s).                         ####
            ############################################################################
            robust = 0.0  # decrease more if lots of failed solutions.
            niter = 10000

            if 'start_image' not in steps_performed:

                mask = mlibs.mask_dilation(image_list['test_image'],
                                           PLOT=False,
                                           sigma=12,
                                           iterations=3)[1]
                mask_wslclean = mask * 1.0  # mask in wsclean is inverted
                mask_name = image_list['test_image'].replace('.fits', '') + '_mask.fits'
                mlibs.pf.writeto(mask_name, mask_wslclean, overwrite=True)
                print(' ==> Using mask ', mask_name, ' in wsclean for deconvolution.')

                start_image(g_name, n_interaction=iteration,
                            imsize=imsize, cell=cell,
                            # uvtaper=['0.05arcsec'],
                            delmodel=True,
                            # opt_args=' -multiscale -multiscale-scales 0 ',
                            nsigma_automask='12.0',
                            nsigma_autothreshold='6.0',
                            # next time probably needs to use 7.0 instead of 3.0
                            niter=niter, shift=FIELD_SHIFT,quiet=quiet,
                            # uvtaper='0.04asec',
                            savemodel=True, mask=mask_name,
                            robust=robust, datacolumn='DATA')


                file_list = glob.glob(f"{path}*MFS-image.fits")
                file_list.sort(key=os.path.getmtime, reverse=False)
                image_list['start_image'] = file_list[-1]
                image_list['start_residual'] = image_list['start_image'].replace(
                    'MFS-image.fits','MFS-residual.fits')
                # file_list = glob.glob(f"{path}*MFS-model.fits")
                # file_list.sort(key=os.path.getmtime, reverse=False)
                image_list['start_model'] = image_list['start_image'].replace(
                    'MFS-image.fits','MFS-model.fits')


                level_stats = mlibs.level_statistics(image_list['start_image'])
                image_stats = mlibs.get_image_statistics(imagename=image_list['start_image'],
                                                         dic_data=level_stats)

                image_statistics['start_image'] = image_stats
                if 'start_image' not in steps_performed:
                    steps_performed.append('start_image')


            gaintype = 'G'
            calmode = 'p'
            combine = ''
            cut_off = 1.5
            SNRs, percentiles_SNRs = check_solutions(g_name, field, cut_off=cut_off,
                                                     n_interaction=iteration,
                                                     solnorm=solnorm, combine=combine,
                                                     calmode=calmode,
                                                     gaintype=gaintype,
                                                     gain_tables_selfcal=[],
                                                     return_solution_stats=True)


        if 'run_rflag_final' in steps:
            run_rflag(g_vis, display='report', action='apply',
                      timedevscale=4.0, freqdevscale=4.0, winsize=7,
                      datacolumn='corrected')
            steps_performed.append('run_rflag_final')


if run_mode == 'jupyter':
    print('selfcal script is not doing anything, you can use it on a '
          'jupyter notebook. For that, you have to manually '
          'set your variable names and steps taken to selfcalibrate your data.')
