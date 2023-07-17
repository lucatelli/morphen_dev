# Self-Calibration

#wsclean -name UGC08387_50K_autothr_3_automask_5_mgain_0.9.briggs_robust_0.5.selfcal.multiscale -size 1024 1024 -scale 0.03arcsec -mgain 0.9 -niter 50000 -weight briggs 0.5 -auto-threshold 3 -auto-mask 5 -multiscale -data-column CORRECTED_DATA UGC08387_AL746.calibrated.ms
# wsclean -name 0_corrected_IRASF17132+531_100K_thr_auto3_automask_5_mgain_0.99.briggs_0.5 -size 1024 1024 -scale 0.03arcsec -mgain 0.99 -niter 100000 -weight briggs 0.5 -auto-threshold 3 -auto-mask 5 -data-column CORRECTED_DATA -no-update-model-required IRASF17132+531_AL746.calibrated.ms/
# source /opt/lofarsoft/lofarinit.sh

# Self-Calibration
#imaging settings
## Using VLA Calculator

import os
from casatasks import *
# from casatools import *
import casatools
from casaplotms import plotms
from casaviewer.imview import imview

'''
tclean parameters
'''
imsize = 3072
unit = 'mJy'
rms_std_factor = 3

parallel = False
#rms_level = 0.037 #Natural
# rms_level = 0.044#0.005 #Robust-briggs
# threshold = str(rms_level/rms_std_factor)+unit
threshold = '10.0e-6Jy'
# rms_std = '1.0e-5Jy'
FIELD=''
SPWS = ''
ANTENNAS = ''
weighting = 'briggs'
refant=''
cell = '0.05arcsec'
gridder = 'standard'
specmode = 'mfs'
deconvolver = 'mtmfs'
scales=[0,1,2,3,4,5,8,12,20,30]
smallscalebias=0.7
robust = 0.0
gain = 0.05
pblimit=-0.1
nterms = 3
ext = ''
if deconvolver=='mtmfs':
    ext = ext + '.tt0'

# usemask='auto-multithresh'
usemask='user'
sidelobethreshold=2.0
noisethreshold=12.0
lownoisethreshold=4.0
minbeamfrac=0.05
growiterations=75
negativethreshold=12.0
# os.environ['SAVE_ALL_AUTOMASKS']="true"
interactive = True

# gain settings
solint_inf = 'inf'
solint_long = '120s'
solint_mid = '60s'
solint_short = '30s'
solnorm = False
combine = ''

#plotting config
data_range=[-5.55876e-06, 0.00450872]

if not os.path.exists('selfcal/'):
    os.makedirs('selfcal/')
if not os.path.exists('selfcal/plots'):
    os.makedirs('selfcal/plots')


# proj_name = '.calibrated'
proj_name = '_combined_w_0.75'
image_list = ['UGC8696']#,'VV250a','VV705']


def report_flag(summary,axis):
    for id, stats in summary[ axis ].items():
        print('%s %s: %5.1f percent flagged' % ( axis, id, 100. * stats[ 'flagged' ] / stats[ 'total' ] ))
    pass


def eview(imagename,contour=None,
    data_range=None,#[-2.52704e-05, 0.0159025],
    colormap='Rainbow 2',scaling=-2.0,zoom=1,out=None):
    if data_range==None:
        st = imstat(imagename)
        min_data = -1.0*st['rms'][0]
        max_data = 1*st['max'][0]
        data_range = [min_data,max_data]
    if contour==None:
        contour = imagename
    # if out==None:
    #     out = imagename + '_drawing.png'
    imview(raster={
        'file': imagename,
        'range': data_range,
        'colormap': colormap,'scaling': scaling,'colorwedge' : True},
        contour={'file': contour,'levels': [-0.1, 0.01,0.05,0.1,0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.8]},#include negative cont
        # axes={'x':'Declination'} ,
        # zoom={'blc': [3,3], 'trc': [3,3], 'coord': 'pixel'},
        zoom=zoom,
        out=out,
        # scale=scale,
        # dpi=dpi,
        # orient=orient
     )
# .replace('/','')


def get_image_statistics(imagename,residualname=None):
    if residualname==None:
        try:
            residualname = imagename.replace('.image','.residual')
        except:
            print('Please, provide the residual image name')
    stats_im = imstat(imagename=imagename)
    stats_re = imstat(imagename=residualname)

    #determine the flux flux peak of image and residual
    flux_peak_im = stats_im['max'][0]
    flux_peak_re = stats_re['max'][0]

    #determine the rms and std of residual and of image
    rms_re = stats_re['rms'][0]
    rms_im = stats_im['rms'][0]
    sigma_re = stats_re['sigma'][0]
    sigma_im = stats_im['sigma'][0]

    #determine the image and residual flux
    flux_im = stats_im['flux'][0]
    # flux_re = stats_re['flux']

    sumsq_im = stats_im['sumsq'][0]
    sumsq_re = stats_re['sumsq'][0]

    q = sumsq_im/sumsq_re
    # flux_ratio = flux_re/flux_im

    snr_im = flux_im/sigma_im
    snr = flux_im/sigma_re

    peak_im_rms = flux_peak_im/rms_im
    peak_re_rms = flux_peak_re/rms_re

    print(' Flux=%.5f Jy/Beam' % flux_im)
    print(' Flux peak (image)=%.5f Jy' % flux_peak_im, 'Flux peak (residual)=%.5f Jy' % flux_peak_re)
    print(' flux_im/sigma_im=%.5f' % snr_im, 'flux_im/sigma_re=%.5f' % snr)
    print(' rms_im=%.5f' % rms_im, 'rms_re=%.5f' % rms_re)
    print(' flux_peak_im/rms_im=%.5f' % peak_im_rms, 'flux_peak_re/rms_re=%.5f' % peak_re_rms)
    print(' sumsq_im/sumsq_re=%.5f' % q)

def plot_visibilities(g_vis,name,with_DATA=False,with_MODEL=False,with_CORRECTED=False):


    plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
        antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
        ydatacolumn='corrected-model', avgchannel='64', avgtime='60',
        width=800,height=540,showgui=False,overwrite=True,
        plotfile='selfcal/plots/'+name+'_uvwave_amp_corrected-model.jpg')

    plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',
        antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
        ydatacolumn='corrected-model', avgchannel='64', avgtime='60',
        width=800,height=540,showgui=False,overwrite=True,
        plotfile='selfcal/plots/'+name+'_uvdist_amp_corrected-model.jpg')

    plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
        antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
        ydatacolumn='corrected/model', avgchannel='64', avgtime='60',
        width=800,height=540,showgui=False,overwrite=True,
        plotrange=[-1,-1,0,5],
        plotfile='selfcal/plots/'+name+'_uvwave_amp_corrected_div_model.jpg')

    plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',
        antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
        ydatacolumn='corrected/model', avgchannel='64', avgtime='60',
        width=800,height=540,showgui=False,overwrite=True,
        plotrange=[-1,-1,0,5],
        plotfile='selfcal/plots/'+name+'_uvdist_amp_corrected_div_model.jpg')


    if with_MODEL == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
            antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
            ydatacolumn='model', avgchannel='64', avgtime='30',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile='selfcal/plots/'+name+'_uvwave_amp_model.jpg')

        plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',
            antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
            ydatacolumn='model', avgchannel='64', avgtime='30',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile='selfcal/plots/'+name+'_uvdist_amp_model.jpg')

        plotms(vis=g_vis, xaxis='freq', yaxis='amp',
            antenna=ANTENNAS,spw=SPWS,coloraxis='scan',
            ydatacolumn='model', avgchannel='', avgtime='60',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile='selfcal/plots/'+name+'_freq_amp_model.jpg')


    if with_DATA ==True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
            antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
            ydatacolumn='data', avgchannel='64', avgtime='30',
            width=800,height=540,showgui=False,overwrite=True,
            # plotrange=[-1,-1,-1,0.3],
            plotfile='selfcal/plots/'+name+'_uvwave_amp_data.jpg')
        plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',
            antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
            ydatacolumn='data', avgchannel='64', avgtime='30',
            width=800,height=540,showgui=False,overwrite=True,
            # plotrange=[-1,-1,0,0.3],
            plotfile='selfcal/plots/'+name+'_uvdist_amp_data.jpg')
        plotms(vis=g_vis, xaxis='freq', yaxis='amp',
            antenna=ANTENNAS,spw=SPWS,coloraxis='scan',
            ydatacolumn='data', avgchannel='', avgtime='60',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile='selfcal/plots/'+name+'_freq_amp_data.jpg')

    if with_CORRECTED ==True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
            antenna=ANTENNAS,spw=SPWS,
            # plotrange=[-1,-1,0,0.3],
            ydatacolumn='corrected', avgchannel='64', avgtime='30',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile='selfcal/plots/'+name+'_uvwave_amp_corrected.jpg')
        plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',
            antenna=ANTENNAS,spw=SPWS,
            # plotrange=[-1,-1,0,0.3],
            ydatacolumn='corrected', avgchannel='64', avgtime='30',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile='selfcal/plots/'+name+'_uvdist_amp_corrected.jpg')
        plotms(vis=g_vis, xaxis='freq', yaxis='amp',
            antenna=ANTENNAS,spw=SPWS,coloraxis='scan',
            ydatacolumn='corrected', avgchannel='', avgtime='60',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile='selfcal/plots/'+name+'_freq_amp_corrected.jpg')

    pass

def make_dirty(g_name,field,n_interaction,mask=''):
    '''
    Help function to create the dirty beam image.
    '''
    g_vis = g_name + '.ms'
    niter = 0
    image_dirty = str(n_interaction)+'_dirty_'+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+specmode+'.'+deconvolver+'.'+gridder
    tclean(vis=g_vis,
           imagename='selfcal/'+image_dirty,
           field=FIELD,
           specmode='mfs',deconvolver=deconvolver,gridder=gridder,
           scales=scales, smallscalebias=smallscalebias,
           imsize=imsize,cell= cell,
           weighting=weighting,robust=robust,
           niter=niter,interactive=interactive,
           pblimit=pblimit,
           mask=mask,
           savemodel='none',
           usepointing=False)

    pass

def initial_test_image(g_name,field,n_interaction='test',niter=250,
    usemask=usemask,interactive=interactive,mask=''):
    '''
    Help function to create a initial test image (few interactions) in order
    to unvel some basic parameters that need to be used in the automask.
    '''
    g_vis = g_name + '.ms'
    # niter = 0
    image_test = str(n_interaction)+'_image_'+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+specmode+'.'+deconvolver+'.'+gridder
    print(image_test)
    tclean(vis=g_vis,
        imagename='selfcal/'+image_test,
        field=FIELD,spw=SPWS,
        specmode=specmode,deconvolver=deconvolver,gridder=gridder,
        scales=scales, smallscalebias=smallscalebias,
        imsize=imsize,cell= cell,
        weighting=weighting,robust=robust,
        niter=niter,interactive=interactive,
        usemask=usemask,threshold=threshold,
        sidelobethreshold=sidelobethreshold,
        noisethreshold=noisethreshold,
        lownoisethreshold=lownoisethreshold,
        minbeamfrac=minbeamfrac,
        growiterations=growiterations,
        datacolumn='data',
        mask=mask,
        negativethreshold=negativethreshold,
        pblimit=pblimit,nterms=nterms,pbcor=True,
        savemodel='none')
    try:
        eview('selfcal/'+image_test+'.image'+ext,data_range=data_range,
            scaling=-2.0,out='selfcal/'+image_test+'.image'+ext+'.png')
    except:
        print('Error in ploting image with contours....')
        pass

    pass

def start_image(g_name,field,n_interaction,delmodel=False,interactive=False,
    niter=600,
    usemask=usemask,PLOT=True,datacolumn='corrected',mask='',
    savemodel='modelcolumn',uvtaper=[],uvrange='',startmodel=''):
    '''
    Help function to create the initial image/model for self calibration.
    Parameters can be simply changed outside the function.

    It creates an automatic output name according the parameter values.
    Additionally, it performs some visibility plots for the data.
    
    Note that this start operates on the data column, not the corrected. 
    '''
    g_vis = g_name + '.ms'
    base_name = '_start_image_'
    if interactive:
        base_name=base_name+'interactive_'
    if usemask=='auto-multithresh':
        base_name = base_name+'automask_multithresh_'


    image_start_model = str(n_interaction)+base_name+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+str(robust)+'.'+specmode+'.'+deconvolver+'.'+gridder
    if delmodel==True:
        delmod(g_vis)
        clearcal(g_vis)
    print('selfcal/'+image_start_model+'.image')
    tclean(vis=g_vis,
        imagename='selfcal/'+image_start_model,
        field=FIELD,spw=SPWS,
        specmode=specmode,deconvolver=deconvolver,gridder=gridder,
        scales=scales, smallscalebias=smallscalebias,
        imsize=imsize,cell= cell,
        weighting=weighting,robust=robust,
        niter=niter,interactive=interactive,
        pblimit=pblimit,
        usemask=usemask,
        mask=mask,
        sidelobethreshold=sidelobethreshold,
        noisethreshold=noisethreshold,
        lownoisethreshold=lownoisethreshold,
        minbeamfrac=minbeamfrac,
        growiterations=growiterations,
        negativethreshold=negativethreshold,
        uvtaper=uvtaper,startmodel=startmodel,uvrange=uvrange,
        threshold=threshold,nterms=nterms,parallel=parallel,
        datacolumn=datacolumn,cycleniter=25,
        savemodel=savemodel)

    try:
        eview('selfcal/'+image_start_model+'.image'+ext,data_range=data_range,
            scaling=-2.0,out='selfcal/'+image_start_model+'.image'+ext+'.png')
    except:
        pass

    if PLOT==True:
        plot_visibilities(g_vis=g_vis,name=image_start_model,
            with_MODEL=True,with_DATA=True)

    print(' Start Image Statistics:')
    get_image_statistics('selfcal/'+image_start_model+'.image'+ext)
    pass

def get_tb_data(table,param):
    tb.open(table)
    param_data = tb.getcol(param).ravel()
    tb.close()
    return(param_data)

def make_plot_snr(caltable,cut_off,plot_snr=True,bins=50,density=True,
    save_fig=False):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    snr = get_tb_data(caltable,'SNR')

    if plot_snr:
        plt.hist( snr, bins=bins, density=density, histtype='step')
        plt.legend( loc='upper right' )
        plt.xlabel( 'SNR' )
        if save_fig==True:
            plt.savefig(caltable.replace('.tb','.jpg'),dpi=300)
            plt.clf()
            plt.close()
        plt.show()

    print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format( stats.percentileofscore( snr, cut_off ), '' ) )

def calibration_table_plot(table,stage='selfcal',
     table_type='gain_phase',kind='',
     xaxis='time',yaxis='phase',
     fields=['']):

     if not os.path.exists('selfcal/plots/'+stage):
         os.makedirs('selfcal/plots/'+stage)

     if yaxis == 'phase':
         plotrange=[-1,-1,-180,180]
     else:
         plotrange=[-1,-1,-1,-1]

     if fields=='':
         plotms(vis=table,xaxis=xaxis,yaxis=yaxis,field='',
             gridcols=1,gridrows=1,coloraxis='spw',antenna='',plotrange=plotrange,
             width=800,height=540,dpi=600,overwrite=True,showgui=False,
             plotfile='selfcal/plots/'+stage+'/'+table_type+'_'+xaxis+'_'+yaxis+'_field_'+str('all')+'.jpg')
     else:

         for FIELD in fields:
             plotms(vis=table,xaxis=xaxis,yaxis=yaxis,field=FIELD,
                 # gridcols=4,gridrows=4,coloraxis='spw',antenna='',iteraxis='antenna',
                 # width=2048,height=1280,dpi=256,overwrite=True,showgui=False,
                 gridcols=1,gridrows=1,coloraxis='spw',antenna='',plotrange=plotrange,
                 width=800,height=540,dpi=600,overwrite=True,showgui=False,
                 plotfile='selfcal/plots/'+stage+'/'+table_type+'_'+xaxis+'_'+yaxis+'_field_'+str(FIELD)+'.jpg')

     pass


def check_solutions(g_name,field,cut_off=3.0,n_interaction=0,
                    solnorm=solnorm,combine=combine,calmode='p',gaintype='G',
                    gain_tables_selfcal=['']):

    g_vis = g_name + '.ms'

    caltable_int = 'selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_int.tb'
    # caltable_5 = 'selfcal/selfcal_'+str(n_interaction)+'_'+g_name+'_solint_5.tb'
    caltable_20 = 'selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_20.tb'
    caltable_40 = 'selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_40.tb'
    caltable_60 = 'selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_60.tb'
    caltable_120 = 'selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_120.tb'
    caltable_inf = 'selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_inf.tb'


    if not os.path.exists(caltable_int):
        print(  '>> Performing test-gaincal for solint=int...')
        gaincal(vis=g_vis,caltable=caltable_int,solint='int',refant=refant,
                solnorm=solnorm,combine=combine,
                calmode=calmode,gaintype=gaintype, minsnr=1,gaintable=gain_tables_selfcal)
    # if not os.path.exists(caltable_5):
        # print(  '>> Performing test-gaincal for solint=5s...')
        # gaincal(vis=g_vis,caltable=caltable_5,solint='5s',refant=refant,
        #     calmode=calmode,gaintype=gaintype, minsnr=1,gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_20):
        print(  '>> Performing test-gaincal for solint=20s...')
        gaincal(vis=g_vis,caltable=caltable_20,solint='20s',refant=refant,
                solnorm=solnorm,combine=combine,
                calmode=calmode,gaintype=gaintype, minsnr=1,gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_40):
        print(  '>> Performing test-gaincal for solint=40s...')
        gaincal(vis=g_vis,caltable=caltable_40,solint='40s',refant=refant,
                solnorm=solnorm,combine=combine,
                calmode=calmode,gaintype=gaintype, minsnr=1,gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_60):
        print(  '>> Performing test-gaincal for solint=60s...')
        gaincal(vis=g_vis,caltable=caltable_60,solint='60s',refant=refant,
                solnorm=solnorm,combine=combine,
                calmode=calmode,gaintype=gaintype, minsnr=1,gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_120):
        print(  '>> Performing test-gaincal for solint=120s...')
        gaincal(vis=g_vis,caltable=caltable_120,solint='120s',refant=refant,
                solnorm=solnorm,combine=combine,
                calmode=calmode,gaintype=gaintype, minsnr=1,gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_inf):
        print(  '>> Performing test-gaincal for solint=inf...')
        gaincal(vis=g_vis,caltable=caltable_inf,solint='inf',refant=refant,
                solnorm=solnorm,combine=combine,
                calmode=calmode,gaintype=gaintype, minsnr=1,gaintable=gain_tables_selfcal)

    def make_plot_check(cut_off = cut_off):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        snr_int = get_tb_data(caltable_int,'SNR')
        # snr_5 = get_tb_data(caltable_5,'SNR')
        snr_20 = get_tb_data(caltable_20,'SNR')
        snr_40 = get_tb_data(caltable_40,'SNR')
        snr_60 = get_tb_data(caltable_60,'SNR')
        snr_120 = get_tb_data(caltable_120,'SNR')
        snr_inf = get_tb_data(caltable_inf,'SNR')


        plt.hist( snr_int, bins=50, density=True, histtype='step', label='int' )
        # plt.hist( snr_5, bins=50, density=True, histtype='step', label='5 seconds' )
        plt.hist( snr_20, bins=50, density=True, histtype='step', label='20 seconds' )
        plt.hist( snr_40, bins=50, density=True, histtype='step', label='40 seconds' )
        plt.hist( snr_60, bins=50, density=True, histtype='step', label='60 seconds' )
        plt.hist( snr_120, bins=50, density=True, histtype='step', label='120 seconds' )
        plt.hist( snr_inf, bins=50, density=True, histtype='step', label='inf' )
        plt.legend( loc='upper right' )
        plt.xlabel( 'SNR' )
        plt.semilogx()
        plt.savefig('selfcal/plots/'+str(n_interaction)+'_'+g_name+'_gain_solutions_comparisons_norm.pdf')
        plt.clf()
        plt.close()

        plt.figure()
        plt.hist( snr_int, bins=50, density=False, histtype='step', label='int' )
        # plt.hist( snr_5, bins=50, density=False, histtype='step', label='5 seconds' )
        plt.hist( snr_20, bins=50, density=False, histtype='step', label='20 seconds' )
        plt.hist( snr_40, bins=50, density=False, histtype='step', label='40 seconds' )
        plt.hist( snr_60, bins=50, density=False, histtype='step', label='60 seconds' )
        plt.hist( snr_120, bins=50, density=False, histtype='step', label='120 seconds' )
        plt.hist( snr_inf, bins=50, density=False, histtype='step', label='inf' )
        plt.legend( loc='upper right' )
        plt.xlabel( 'SNR' )
        plt.semilogx()
        plt.savefig('selfcal/plots/'+str(n_interaction)+'_'+os.path.basename(g_name)+'_gain_solutions_comparisons.pdf')

        print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format(
            stats.percentileofscore( snr_int, cut_off ), 'int' ) )
        # print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format(
        #     stats.percentileofscore( snr_5, cut_off ), '5s' ) )
        print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format(
            stats.percentileofscore( snr_20, cut_off ), '20s' ) )
        print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format(
            stats.percentileofscore( snr_40, cut_off ), '40s' ) )
        print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format(
            stats.percentileofscore( snr_60, cut_off ), '60s' ) )
        print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format(
            stats.percentileofscore( snr_120, cut_off ), '120s' ) )
        print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format(
            stats.percentileofscore( snr_inf, cut_off ), 'inf' ) )
        plt.clf()
        plt.close()

    def compare_phase_variation():
        plotms(caltable_int,antenna='ea01',scan='',yaxis='phase')

        plotms(caltable_40,antenna='',scan='',yaxis='phase',plotindex=1,
            clearplots=False,customsymbol=True,symbolsize=20,
            symbolcolor='ff0000',symbolshape='circle')

        plotms(caltable_20,antenna='',scan='',yaxis='phase',plotindex=2,
            clearplots=False,customsymbol=True,symbolsize=12,
            symbolcolor='green',symbolshape='square')

        plotms(caltable_inf,antenna='',scan='',yaxis='phase',plotindex=3,
            clearplots=False,customsymbol=True,symbolsize=8,
            symbolcolor='yellow',symbolshape='square')


        plotms(caltable_60,antenna='',scan='',yaxis='phase',plotindex=4,
            clearplots=False,customsymbol=True,symbolsize=4,
            symbolcolor='purple',symbolshape='square',
            width=1600,height=1080,showgui=True,overwrite=True,
            plotfile='selfcal/plots/'+str(n_interaction)+'_'+os.path.basename(g_name)+'_phase_variation_intervals.jpg')

    def compare_amp_variation():
        plotms(caltable_int,antenna='',scan='',yaxis='amp')

        plotms(caltable_40,antenna='',scan='',yaxis='amp',plotindex=1,
            clearplots=False,customsymbol=True,symbolsize=20,
            symbolcolor='ff0000',symbolshape='circle')

        plotms(caltable_20,antenna='',scan='',yaxis='amp',plotindex=2,
            clearplots=False,customsymbol=True,symbolsize=12,
            symbolcolor='green',symbolshape='square')

        plotms(caltable_inf,antenna='',scan='',yaxis='amp',plotindex=3,
            clearplots=False,customsymbol=True,symbolsize=8,
            symbolcolor='yellow',symbolshape='square')


        plotms(caltable_60,antenna='',scan='',yaxis='amp',plotindex=4,
            clearplots=False,customsymbol=True,symbolsize=4,
            symbolcolor='purple',symbolshape='square',
            width=1600,height=1080,showgui=True,overwrite=True,
            plotfile='selfcal/plots/'+str(n_interaction)+'_'+os.path.basename(g_name)+'_amp_variation_intervals.jpg')

    #
    # def plot_gains():
    #     plotms(caltable_int,antenna='ea01',scan='',yaxis='phase',
    #         gridrows=5,gridcols=5,iteraxis='antenna',coloraxis='spw')

    make_plot_check(cut_off=cut_off)
    compare_phase_variation()
    if calmode=='ap':
        compare_amp_variation()

    pass


def update_model_image(g_name,field,n_interaction,
    interactive=interactive,datacolumn='corrected',
    usemask=usemask,niter = 1000,mask='',
    uvtaper=[],uvrange='',PLOT=False):

    g_vis = g_name + '.ms'


    base_name = '_update_model_image_'


    if interactive == True:
        base_name = base_name+'interactive_'
    if usemask=='auto-multithresh':
        base_name = base_name+'automask_multithresh_'



    image_update_model = str(n_interaction)+base_name+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+str(robust)+'.'+specmode+'.'+deconvolver+'.'+gridder

    tclean(vis=g_vis,
        imagename='selfcal/'+image_update_model,
        field=FIELD,
        specmode=specmode,deconvolver=deconvolver,gridder=gridder,
        scales=scales, smallscalebias=smallscalebias,
        imsize=imsize,cell= cell,
        weighting=weighting,robust=robust,
        niter=niter,interactive=interactive,
        pblimit=pblimit,nterms=nterms,
        usemask=usemask,threshold=threshold,
        mask=mask,
        sidelobethreshold=sidelobethreshold,
        noisethreshold=noisethreshold,
        lownoisethreshold=lownoisethreshold,
        minbeamfrac=minbeamfrac,
        spw=SPWS,
        datacolumn=datacolumn,cycleniter=25,
        growiterations=growiterations,
        negativethreshold=negativethreshold,parallel=parallel,
        verbose=True,uvtaper=uvtaper,uvrange=uvrange,
        savemodel='modelcolumn',
        usepointing=False)


    if PLOT==True:
        plot_visibilities(g_vis=g_vis,name=image_update_model,
            with_MODEL=True,with_CORRECTED=True)

    try:
        eview('selfcal/'+image_update_model+'.image'+ext,data_range=data_range,
            scaling=-2.0,out='selfcal/'+image_update_model+'.image'+ext+'.png')
    except:
        print('Error in ploting image with contours....')
        pass

    print(' Image (update) Statistics:')
    get_image_statistics('selfcal/'+image_update_model+'.image'+ext)

    pass

def self_gain_cal(g_name,field,n_interaction,gain_tables=[],
    combine=combine,solnorm=False,uvtaper=[],uvrange='',
    niter=500,
    minsnr=5.0,solint='inf',gaintype='G',calmode='p',
    action='apply',flagbackup=True,PLOT=False):

    g_vis = g_name + '.ms'

    cal_basename = '_selfcal_'
    base_name = '_update_model_image_'

    if calmode=='p':
        cal_basename = cal_basename + 'phase_'
        base_name =  base_name + 'phase_'
    if calmode=='ap':
        cal_basename = cal_basename + 'ampphase_'
        base_name =  base_name + 'ampphase_'
    if gain_tables != []:
        cal_basename = cal_basename + 'incremental_'


    if interactive == True:
        base_name = base_name+'interactive_'
        cal_basename = cal_basename+'interactive_'
    if usemask=='auto-multithresh':
        base_name = base_name+'automask_multithresh_'
        cal_basename = cal_basename+'automask_multithresh_'



    image_update_model = str(n_interaction)+base_name+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+str(robust)+'.'+specmode+'.'+deconvolver+'.'+gridder


    caltable = 'selfcal/'+str(n_interaction)+cal_basename+os.path.basename(g_name)+'_'+'_solint_'+solint+'_minsnr_'+str(minsnr)+'.tb'
    if not os.path.exists(caltable):
        if calmode=='ap':
            solonrm=True
        else:
            solnorm=False
        gaincal(vis=g_vis,field=FIELD,caltable=caltable,solint=solint,gaintable=gain_tables,combine=combine,
            refant=refant,calmode=calmode,gaintype=gaintype, minsnr=minsnr,solnorm=solnorm)
    else:
        print(' => Using existing caltable with same parameters asked.')
        print(' => Not computing again...')

    calibration_table_plot(table=caltable,
        fields='',yaxis='phase',
        table_type='selfcal_phase_'+os.path.basename(g_name)+'_'+str(n_interaction)+'_solint_'+solint+'_minsnr_'+str(minsnr))

    if calmode=='ap':
        calibration_table_plot(table=caltable,
            fields='',yaxis='amp',
            table_type='selfcal_ampphase_'+os.path.basename(g_name)+'_'+str(n_interaction)+'_solint_'+solint+'_minsnr_'+str(minsnr))


    make_plot_snr(caltable=caltable,cut_off=minsnr,
        plot_snr=True,bins=50,density=True,save_fig=True)

    if action=='apply':
        if flagbackup == True:
            print('     => Creating new flagbackup file before mode ',calmode,' selfcal ...')
            flagmanager(vis=g_vis,mode='save',versionname='before_selfcal_mode_'+calmode,
                comment='Before selfcal apply.')

        gain_tables.append(caltable)
        print('     => Reporting data flagged before selfcal apply interaction',n_interaction,'...')
        summary_bef = flagdata(vis=g_vis, field=FIELD,mode='summary')
        report_flag(summary_bef,'field')

        applycal(vis=g_vis,gaintable=gain_tables,flagbackup=False,calwt=False)

        print('     => Reporting data flagged after selfcal apply interaction',n_interaction,'...')
        summary_aft = flagdata(vis=g_vis, field=FIELD,mode='summary')
        report_flag(summary_aft,'field')

        if PLOT==True:
            plot_visibilities(g_vis=g_vis,name=image_update_model,
            with_CORRECTED=True,with_MODEL=False,with_DATA=False)

    return(gain_tables)


def image_selfcal(g_name,field,n_interaction,
    niter = 4000,interactive = False,usemask=usemask,
    mask='',
    base_name = '_selfcal_image_'):
    g_vis = g_name + '.ms'
    if interactive == True:
        base_name = base_name+'interactive_'
    if usemask=='auto-multithresh':
        base_name = base_name+'automask_multithresh_'

    image_deep_selfcal = str(n_interaction)+base_name+'_'+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+str(robust)+'.'+specmode+'.'+deconvolver+'.'+gridder

    print('selfcal/'+image_deep_selfcal+'.image'+ext)

    tclean(vis=g_vis,
        imagename='selfcal/'+image_deep_selfcal,
        field=FIELD,spw=SPWS,antenna=ANTENNAS,
        specmode=specmode,deconvolver=deconvolver,gridder=gridder,
        imsize=imsize,cell=cell,uvrange=uvrange,
        weighting=weighting,robust=robust,
        scales=scales, smallscalebias=smallscalebias,
        niter=niter,interactive=interactive,threshold=threshold,
        pblimit=pblimit,pbcor=True,
        mask=mask,
        usemask=usemask,nterms=nterms,
        sidelobethreshold=sidelobethreshold,
        noisethreshold=noisethreshold,
        lownoisethreshold=lownoisethreshold,
        minbeamfrac=minbeamfrac,
        growiterations=growiterations,
        negativethreshold=negativethreshold,
        verbose=True,parallel=parallel,cycleniter=25,
        savemodel='none',datacolumn='corrected'
        )
    # if PLOT==True:
        # plot_visibilities(g_vis=g_vis,name=image_deep_selfcal,with_DATA=True)
    # pass
    #usepointing=False
    try:
        eview('selfcal/'+image_deep_selfcal+'.image'+ext,data_range=data_range,
            scaling=-2.0,out='selfcal/'+image_deep_selfcal+'.image'+ext+'.png')
    except:
        pass

    print(' Image Statistics:')
    get_image_statistics('selfcal/'+image_deep_selfcal+'.image'+ext)
    pass

def mask_params(imagename,residualname=None,sidelobelevel=0.2):
    '''
    This is completely experimental, because there are parameters that may be
    guessed initially.
    '''
    if residualname==None:
        try:
            residualname = imagename.replace('.image','') + '.residual'
        except:
            print('Please, provide the residual image name')

    st_I = imstat(imagename=imagename)
    st_box_I = imstat(imagename=imagename,box='25,25,400,400')

    st_R = imstat(imagename=residualname)
    st_box_R = imstat(imagename=residualname,box='25,25,400,400')


    fluxpeak_residual = st_R['max'][0]
    rms_residual = st_box_R['rms'][0]

    fluxpeak_image = st_I['max'][0]
    rms_image = st_box_I['rms'][0]

    print('rms of image box=',rms_image)
    print('rms of residual box=',rms_residual)
    print('flux peak image=',fluxpeak_image)
    print('flux peak residual=',fluxpeak_residual)

    svr = sidelobelevel * fluxpeak_residual
    ntr = 3 * rms_residual

    sidelobethreshold_ = 2.0

    sidelobethreshold_value_ = svr * sidelobethreshold_

    threshold_init = max(sidelobethreshold_value_,ntr)

    sidelobethreshold = threshold_init /(sidelobelevel*fluxpeak_residual)
    noisethreshold = threshold_init/(rms_residual)

    print('Initial guess for automask parameters are:')
    print(sidelobethreshold,noisethreshold)

    return(sidelobethreshold,noisethreshold)

steps=[
	   'save_init_flags',
	   '0',
	   #'2',
       	#    '3'
      ]

os.environ['SAVE_ALL_AUTOMASKS']="false"
usemask='user'#'auto-multithresh'
interactive=True

# for field in image_list:
#     g_name = field + proj_name

#     if 'save_init_flags' in steps:
#         if not os.path.exists(g_name+'.ms.flagversions/flags.Original/'):
#             flagmanager(vis=g_name+'.ms',mode='save',versionname='Original',
#                 comment='Original flags.')
#         else:
#             print('     ==> Skipping flagging backup init (exists).')

#     if 'restore_init_flags' in steps:
#         try:
#             print('     ==> Restoring flags to original...')
#             flagmanager(vis=g_name+'.ms',mode='restore',versionname='Original')
#             delmod(vis=g_name + '.ms')
#             clearcal(vis=g_name + '.ms')
#             print('     ==> Renaming selfcal/ folder to selfcal.old/ ')
#             os.system('mv selfcal/ selfcal.old/')
#         except:
#             pass

#     os.environ['SAVE_ALL_AUTOMASKS']="false"#need to set to false if using the same image twice.

#     if 'make_dirty' in steps:
#         make_dirty(g_name=g_name,field=field,n_interaction=0)

#     if 'initial_test_image' in steps:
#         initial_test_image(g_name,field,n_interaction='test',niter=1500,
#             usemask=usemask,interactive=interactive)

#     if '0' in steps:
#         ############################################################################
#         #### 0. Zero interaction. Use a small/negative robust parameter,        ####
#         ####    to find the bright/compact emission(s).                         ####
#         ############################################################################
#         robust = 0.0 #decrease more if lots of failed solutions.
#         niter = 500
#         threshold = '20.0e-6Jy'

#         start_image(g_name,field,0,delmodel=True,PLOT=True,niter=niter,
#             interactive=interactive,usemask=usemask,datacolumn='data')

#         # check_solutions(g_name,field,calmode='p',combine='scan')
#         gain_tables_selfcal_temp=self_gain_cal(g_name,field,n_interaction=0,
#             niter=niter,
#             minsnr = 1.0,solint = '240s',flagbackup=True,combine='',
#             calmode='p',action='calculate',PLOT=True)

#         # robust = 0.0
#         # image_selfcal(g_name,field,n_interaction=0,interactive=False,usemask=usemask)
#     if '1' in steps:
#         ############################################################################
#         #### 1. First interaction. Increase a little the robust parameter,      ####
#         ####    start to consider more extended emission.                       ####
#         ############################################################################
#         threshold = '5.0e-6Jy'
#         niter = 800
#         robust = 2.0 #or 0.5 if lots of extended emission.
#         #
#         # update_model_image(g_name,field,n_interaction=1,interactive=interactive,
#         #     uvtaper=[],niter=niter,usemask=usemask,PLOT=True)
#         #
#         # # do not consider incremental gains yet (last table is not good)
#         # check_solutions(g_name,field,n_interaction=1,gain_tables_selfcal=[])

#         gain_tables_selfcal_temp=self_gain_cal(g_name,field,n_interaction=1,
#             minsnr = 1.5,solint = '240s',flagbackup=True,
#             gain_tables=[],calmode='p',gaintype='G',
#             action='apply',PLOT=True)


#         os.environ['SAVE_ALL_AUTOMASKS']="true"
#         # mask = 'selfcal/1_update_model_image_interactive_VV705_AL746_fix_combined_1024_0.05arcsec_500.briggs.mfs.mtmfs.standard.mask'
#         # image_selfcal(g_name,field,n_interaction=1,interactive=True,
#             # usemask=usemask,mask=mask)
#         # image_selfcal(g_name,field,n_interaction='1',niter = 3000,interactive=True,usemask=usemask)

#     if '2' in steps:
#         ############################################################################
#         #### 2. Second interaction. Increase more the robust parameter, or use  ####
#         ####    uvtapering. Consider even more extended emission (if there is). ####
#         ############################################################################
#         os.environ['SAVE_ALL_AUTOMASKS']="false"
#         robust = 1.0
#         threshold = '3.0e-6Jy'
#         niter = 1000
#         #update_model_image(g_name,field,n_interaction=2,interactive=interactive,
#         #    uvtaper=[],niter=niter,usemask=usemask,PLOT=True)


#         # #if your solutions are good in the previous step, try to add them bellow,
#         # #e.g. gain_tables_selfcal=gain_tables_selfcal, so the next ones will be
#         # #incremental.
#         #check_solutions(g_name,field,n_interaction=2,
#         #    gain_tables_selfcal=[],calmode='p')
#         #
#         #
#         # inspect the previous solutions (percentage of flagged data), and
#         # try to decrease the solution interval (e.g. 30s, in this case).
#         # again, if previous solutions are good, consider a incremental run,
#         # e.g. gain_tables=gain_tables_selfcal in the call bellow.
#         gain_tables_selfcal=self_gain_cal(g_name,field,n_interaction=2,
#             minsnr = 2.91,solint = solint_long,flagbackup=True,
#             gain_tables=[],calmode='p',gaintype='G',
#             action='apply',PLOT=True)
#         #
#         # os.environ['SAVE_ALL_AUTOMASKS']="true"
#         # perform imaging to check.
#         # robust = 1.0
#         #image_selfcal(g_name,field,n_interaction='2',interactive=interactive,usemask=usemask)
#     #
#     #
#     #
#     # ############################################################################
#     # #### 3. Third interaction.If you see that further improvements can be   ####
#     # ####    obtained, do one more interaction, now amp selfcal.             ####
#     # ####    Be sure that the previous phase gains are ok, because you       ####
#     # ####    need them for the amp gain. If they are not, consider           ####
#     # ####    to iterate as many times you see fit in phases again.           ####
#     # ############################################################################
#     if '3' in steps:
#         ############################################################################
#         #### 2. Second interaction. Increase more the robust parameter, or use  ####
#         ####    uvtapering. Consider even more extended emission (if there is). ####
#         ############################################################################
#         os.environ['SAVE_ALL_AUTOMASKS']="false"
#         robust = -0.1
#         threshold = '2e-6Jy'
#         niter = 1000
#         #update_model_image(g_name,field,n_interaction=3,interactive=interactive,
#         #   uvtaper=[],niter=niter,usemask=usemask,PLOT=True)

#         # #if your solutions are good in the previous step, try to add them bellow,
#         # #e.g. gain_tables_selfcal=gain_tables_selfcal, so the next ones will be
#         # #incremental.
#         #gain_tables_selfcal = ['selfcal/1_selfcal_phase_interactive_0935+6121_LE1014_C_L14_002_20191012_avg_2ndtry__solint_120s_minsnr_1.5.tb']
#         # check_solutions(g_name,field,n_interaction=3,
#         #     gain_tables_selfcal=gain_tables_selfcal,calmode='ap',solnorm=True)
#         #
#         gain_tables_selfcal = ['selfcal/2_selfcal_phase_interactive_UGC5101_combined_w_0.05_RR_LL_newshift__solint_120s_minsnr_2.91.tb']
#         # # inspect the previous solutions (percentage of flagged data), and
#         # # try to decrease the solution interval (e.g. 30s, in this case).
#         # # again, if previous solutions are good, consider a incremental run,
#         # # e.g. gain_tables=gain_tables_selfcal in the call bellow.
#         #gain_tables_selfcal=self_gain_cal(g_name,field,n_interaction=3,
#         #    minsnr = 3.0,solint = 'inf',flagbackup=True,solnorm=True,
#         #    gain_tables=gain_tables_selfcal,calmode='ap',gaintype='G',combine='scan',
#         #    action='apply',PLOT=True)
#         #
#         # os.environ['SAVE_ALL_AUTOMASKS']="true"
#         # perform imaging to check.
#         #image_selfcal(g_name,field,n_interaction='3',interactive=True,usemask='user',niter=5000)
#         image_selfcal(g_name,field,n_interaction='3',interactive=False,usemask='auto-multithresh',niter=2000)



    #
    # robust = 1.5
    # update_model_image(g_name,field,n_interaction=3,interactive=interactive,
    #     uvtaper=[],niter=500,usemask=usemask,PLOT=True)
    # # update_model_image(g_name,field,n_interaction=1,interactive=interactive,
    # #     uvtaper=['300klambda'],niter=1000,usemask=usemask)
    #
    #
    # check_solutions(g_name,field,calmode='ap',n_interaction=3,
    #     gain_tables_selfcal=gain_tables_selfcal)
    # # os.environ['SAVE_ALL_AUTOMASKS']="false"
    # gain_tables_selfcal=self_gain_cal(g_name,field,n_interaction=3,
    #     minsnr = 3.0,solint = '30s',flagbackup=False,
    #     interactive=True,combine='scan',calmode='ap',solnorm=True,gaintype='T',
    #     update_model=False,delmodel=False,gain_tables=gain_tables_selfcal,
    #     usemask=usemask,niter=1000,
    #     action='apply',PLOT=True)
    #
    # # os.environ['SAVE_ALL_AUTOMASKS']="true"
    #
    # # mask = 'selfcal/1_update_model_image_interactive_VV705_AL746_fix_combined_1024_0.05arcsec_500.briggs.mfs.mtmfs.standard.mask'
    # # image_selfcal(g_name,field,n_interaction=1,interactive=True,
    # #     usemask=usemask,mask=mask)
    #
    # robust = 1.5
    # image_selfcal(g_name,field,n_interaction='3',interactive=False,usemask=usemask)