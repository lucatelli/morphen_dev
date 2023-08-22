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

Selfcal.
"""
__version__ = 0.3
__author__  = 'Geferson Lucatelli'
__email__   = 'geferson.lucatelli@postgrad.manchester.ac.uk'
__date__    = '2023 05 05'
print(__doc__)

import os
from casatasks import *
try:
    import casatools
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
imsize = 3072
unit = 'mJy'
rms_std_factor = 3

parallel = True
calcpsf = True
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
cell = '0.008arcsec'

smallscalebias=0.7
robust = 0.0
gain = 0.05
pblimit=-0.1
nterms = 3
ext = ''

#
# usemask='auto-multithresh'
interactive = True
usemask='user'
sidelobethreshold=3.5
noisethreshold=15.0
lownoisethreshold=5.0
minbeamfrac=0.06
growiterations=50
negativethreshold=15.0
# os.environ['SAVE_ALL_AUTOMASKS']="true"


# gain settings
solint_inf = 'inf'
solint_long = '120s'
solint_mid = '60s'
solint_short = '30s'
solnorm = False
combine = ''

#plotting config
data_range=[-5.55876e-06, 0.00450872]


outlierfile=''

# proj_name = '.calibrated'


def create_external_file(g_vis,iteration, phasecenters, imagesizes,
                         output_file='outlier_fields_file.py'):
    """
    Creates an outliers field file based on provided parameters and iteration
    during self-calibration. So we do not have to manually change the information
    of the file EVERY iteration of selfcal.

    Parameters:
    - iteration (int): Current iteration.
    - phasecenters (list): List of phase centers.
    - imagesizes (list): List of image sizes. Each entry
        should be a list of two integers.
    - output_file (str): Name of the file to save the data.
        Defaults to 'outlier_fields_file.py'.

    Returns:
    None. Writes to the output file.
    """

    if len(phasecenters) != len(imagesizes):
        raise ValueError("Length of phasecenters and imagesizes lists must be equal.")

    with open(output_file, 'w') as f:
        for i, (phasecenter, imsize) in enumerate(zip(phasecenters, imagesizes)):
            f.write(f"#Outlier in field {i + 1}\n")
            f.write(f"imagename={os.path.dirname(g_vis)}/selfcal/{iteration}_VV250_field_outlier_{i + 1}\n")
            f.write(f"imsize=[{imsize[0]},{imsize[1]}]\n")
            f.write(f"phasecenter = {phasecenter}\n")
            f.write("\n")  # Separate sections with an empty line

    print(f"External file '{output_file}' created!")
    return(output_file)


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
        antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',avgantenna=True,
        ydatacolumn='corrected-model', avgchannel='64', avgtime='360',
        width=800,height=540,showgui=False,overwrite=True,
        plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvwave_amp_corrected-model.jpg')

    plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',
        antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',avgantenna=True,
        ydatacolumn='corrected-model', avgchannel='64', avgtime='360',
        width=800,height=540,showgui=False,overwrite=True,
        plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvdist_amp_corrected-model.jpg')

    plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
        antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',avgantenna=True,
        ydatacolumn='corrected/model', avgchannel='64', avgtime='360',
        width=800,height=540,showgui=False,overwrite=True,
        plotrange=[-1,-1,0,5],
        plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvwave_amp_corrected_div_model.jpg')

    plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',
        antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',avgantenna=True,
        ydatacolumn='corrected/model', avgchannel='64', avgtime='360',
        width=800,height=540,showgui=False,overwrite=True,
        plotrange=[-1,-1,0,5],
        plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvdist_amp_corrected_div_model.jpg')


    if with_MODEL == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',avgantenna=True,
            antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
            ydatacolumn='model', avgchannel='64', avgtime='360',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvwave_amp_model.jpg')

        plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',avgantenna=True,
            antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
            ydatacolumn='model', avgchannel='64', avgtime='360',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvdist_amp_model.jpg')

        plotms(vis=g_vis, xaxis='freq', yaxis='amp',avgantenna=True,
            antenna=ANTENNAS,spw=SPWS,coloraxis='scan',
            ydatacolumn='model', avgchannel='', avgtime='360',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_freq_amp_model.jpg')


    if with_DATA ==True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',avgantenna=True,
            antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
            ydatacolumn='data', avgchannel='64', avgtime='360',
            width=800,height=540,showgui=False,overwrite=True,
            # plotrange=[-1,-1,-1,0.3],
            plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvwave_amp_data.jpg')
        plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',avgantenna=True,
            antenna=ANTENNAS,spw=SPWS,coloraxis='baseline',
            ydatacolumn='data', avgchannel='64', avgtime='360',
            width=800,height=540,showgui=False,overwrite=True,
            # plotrange=[-1,-1,0,0.3],
            plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvdist_amp_data.jpg')
        plotms(vis=g_vis, xaxis='freq', yaxis='amp',avgantenna=True,
            antenna=ANTENNAS,spw=SPWS,coloraxis='scan',
            ydatacolumn='data', avgchannel='', avgtime='360',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_freq_amp_data.jpg')

    if with_CORRECTED ==True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',avgantenna=True,
            antenna=ANTENNAS,spw=SPWS,
            # plotrange=[-1,-1,0,0.3],
            ydatacolumn='corrected', avgchannel='64', avgtime='360',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvwave_amp_corrected.jpg')
        plotms(vis=g_vis, xaxis='uvdist', yaxis='amp',avgantenna=True,
            antenna=ANTENNAS,spw=SPWS,
            # plotrange=[-1,-1,0,0.3],
            ydatacolumn='corrected', avgchannel='64', avgtime='360',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_uvdist_amp_corrected.jpg')
        plotms(vis=g_vis, xaxis='freq', yaxis='amp',avgantenna=True,
            antenna=ANTENNAS,spw=SPWS,coloraxis='scan',
            ydatacolumn='corrected', avgchannel='', avgtime='360',
            width=800,height=540,showgui=False,overwrite=True,
            plotfile=os.path.dirname(g_vis)+'/selfcal/plots/'+name+'_freq_amp_corrected.jpg')

    pass

def make_dirty(g_name,field,n_interaction,mask=''):
    '''
    Help function to create the dirty beam image.
    '''
    g_vis = g_name + '.ms'
    niter = 0
    image_dirty = str(n_interaction)+'_dirty_'+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+specmode+'.'+deconvolver+'.'+gridder
    tclean(vis=g_vis,
           imagename=os.path.dirname(g_vis)+'/selfcal/'+image_dirty,
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

# def initial_test_image(g_name,field,n_interaction='test',niter=250,
#     usemask=usemask,interactive=interactive,mask=''):
#     '''
#     Help function to create a initial test image (few interactions) in order
#     to unvel some basic parameters that need to be used in the automask.
#     '''
#     g_vis = g_name + '.ms'
#     # niter = 0
#     image_test = str(n_interaction)+'_image_'+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+specmode+'.'+deconvolver+'.'+gridder
#     print(image_test)
#     tclean(vis=g_vis,
#         imagename=os.path.dirname(g_vis)+'/selfcal/'+image_test,
#         field=FIELD,spw=SPWS,outlierfile=outlierfile,
#         specmode=specmode,deconvolver=deconvolver,gridder=gridder,
#         scales=scales, smallscalebias=smallscalebias,
#         imsize=imsize,cell= cell,
#         weighting=weighting,robust=robust,
#         niter=niter,interactive=interactive,
#         usemask=usemask,threshold=threshold,
#         sidelobethreshold=sidelobethreshold,
#         noisethreshold=noisethreshold,
#         lownoisethreshold=lownoisethreshold,
#         minbeamfrac=minbeamfrac,
#         growiterations=growiterations,
#         datacolumn='data',
#         mask=mask,
#         negativethreshold=negativethreshold,
#         pblimit=pblimit,nterms=nterms,pbcor=True,
#         savemodel='none')
#     try:
#         eview(os.path.dirname(g_vis)+'/selfcal/'+image_test+'.image'+ext,data_range=data_range,
#             scaling=-2.0,out=os.path.dirname(g_vis)+'/selfcal/'+image_test+'.image'+ext+'.png')
#     except:
#         print('Error in ploting image with contours....')
#         pass
#
#     pass

def start_image(g_name,field,n_interaction,robust=0.0,cycleniter=25,
                specmode = 'mfs',deconvolver = 'mtmfs',ext = '', scales=[0, 8, 24], gridder = 'standard',
                # gridder='standard',  # gridder = 'wproject',# specmode = 'mfs'
                delmodel=False,interactive=False,niter=600,
                usemask=usemask,PLOT=True,datacolumn='corrected',mask='',
                savemodel='modelcolumn',uvtaper=[],uvrange='',startmodel=''):
    '''
    Help function to create the initial image/model for self calibration.
    Parameters can be simply changed outside the function.

    It creates an automatic output name according the parameter values.
    Additionally, it performs some visibility plots for the data.

    Note that this start operates on the data column, not the corrected.
    '''
    if deconvolver == 'mtmfs':
        ext = ext + '.tt0'
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
        imagename=os.path.dirname(g_vis)+'/selfcal/'+image_start_model,
        field=FIELD,spw=SPWS,outlierfile=outlierfile,
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
        datacolumn=datacolumn,cycleniter=cycleniter,
        savemodel=savemodel)

    try:
        eview(os.path.dirname(g_vis)+'/selfcal/'+image_start_model+'.image'+ext,data_range=data_range,
            scaling=-2.0,out=os.path.dirname(g_vis)+'/selfcal/'+image_start_model+'.image'+ext+'.png')
    except:
        pass

    if PLOT==True:
        plot_visibilities(g_vis=g_vis,name=image_start_model,
            with_MODEL=True,with_DATA=True)

    print(' Start Image Statistics:')
    get_image_statistics(os.path.dirname(g_vis)+'/selfcal/'+image_start_model+'.image'+ext)
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
        plt.semilogy()
        plt.semilogx()
        plt.grid()
        if save_fig==True:
            plt.savefig(caltable.replace('.tb','.jpg'),dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()
        plt.show()

    print( 'P(<='+str(cut_off)+') = {0}  ({1})'.format( stats.percentileofscore( snr, cut_off ), '' ) )

def calibration_table_plot(table,stage='selfcal',
     table_type='gain_phase',kind='',
     xaxis='time',yaxis='phase',
     fields=['']):

     if not os.path.exists(os.path.dirname(table)+'/plots/'+stage):
         os.makedirs(os.path.dirname(table)+'/plots/'+stage)

     if yaxis == 'phase':
         plotrange=[-1,-1,-180,180]
     else:
         plotrange=[-1,-1,-1,-1]

     if fields=='':
         plotms(vis=table,xaxis=xaxis,yaxis=yaxis,field='',
             gridcols=1,gridrows=1,coloraxis='spw',antenna='',plotrange=plotrange,
             width=800,height=540,dpi=600,overwrite=True,showgui=True,
             plotfile=os.path.dirname(table)+'/plots/'+stage+'/'+table_type+'_'+xaxis+'_'+yaxis+'_field_'+str('all')+'.jpg')
     else:

         for FIELD in fields:
             plotms(vis=table,xaxis=xaxis,yaxis=yaxis,field=FIELD,
                 # gridcols=4,gridrows=4,coloraxis='spw',antenna='',iteraxis='antenna',
                 # width=2048,height=1280,dpi=256,overwrite=True,showgui=False,
                 gridcols=1,gridrows=1,coloraxis='spw',antenna='',plotrange=plotrange,
                 width=800,height=540,dpi=600,overwrite=True,showgui=False,
                 plotfile=os.path.dirname(table)+'/plots/'+stage+'/'+table_type+'_'+xaxis+'_'+yaxis+'_field_'+str(FIELD)+'.jpg')

     pass


def check_solutions(g_name,field,cut_off=3.0,n_interaction=0,
                    solnorm=solnorm,combine=combine,calmode='p',gaintype='G',
                    gain_tables_selfcal=['']):

    g_vis = g_name + '.ms'

    caltable_int = os.path.dirname(g_name)+'/selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_int.tb'
    # caltable_5 = 'selfcal/selfcal_'+str(n_interaction)+'_'+g_name+'_solint_5.tb'
    caltable_20 = os.path.dirname(g_name)+'/selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_20.tb'
    caltable_40 = os.path.dirname(g_name)+'/selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_40.tb'
    caltable_60 = os.path.dirname(g_name)+'/selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_60.tb'
    caltable_120 = os.path.dirname(g_name)+'/selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_120.tb'
    caltable_inf = os.path.dirname(g_name)+'/selfcal/selfcal_'+str(n_interaction)+'_'+os.path.basename(g_name)+'_solint_inf.tb'


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
        plt.savefig(os.path.dirname(g_name)+'/selfcal/plots/'+str(n_interaction)+
                    '_'+g_name+'_gain_solutions_comparisons_norm.pdf')
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
        plt.savefig(os.path.dirname(g_name)+'/selfcal/plots/'+str(n_interaction)+
                    '_'+os.path.basename(g_name)+'_gain_solutions_comparisons.pdf')

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
            plotfile=os.path.dirname(g_name)+'/selfcal/plots/'+str(n_interaction)+
                     '_'+os.path.basename(g_name)+'_phase_variation_intervals.jpg')

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
            plotfile=os.path.dirname(g_name)+'/selfcal/plots/'+str(n_interaction)+
                     '_'+os.path.basename(g_name)+'_amp_variation_intervals.jpg')

    #
    # def plot_gains():
    #     plotms(caltable_int,antenna='ea01',scan='',yaxis='phase',
    #         gridrows=5,gridcols=5,iteraxis='antenna',coloraxis='spw')

    make_plot_check(cut_off=cut_off)
    compare_phase_variation()
    if calmode=='ap':
        compare_amp_variation()

    pass


def update_model_image(g_name,field,n_interaction,robust=0.5,
                       specmode = 'mfs',deconvolver = 'mtmfs',ext = '',gridder = 'standard',
                       # gridder = 'wproject',# specmode = 'mfs'
                       scales=[0,8,16,32,64],
                       interactive=interactive,datacolumn='corrected',
                       usemask=usemask,niter = 1000,mask='',cycleniter=100,
                       usepointing=True,psfphasecenter='',phasecenter='',
                       uvtaper=[],uvrange='',PLOT=False):

    if deconvolver == 'mtmfs':
        ext = ext + '.tt0'

    g_vis = g_name + '.ms'


    base_name = '_update_model_image_'


    if interactive == True:
        base_name = base_name+'interactive_'
    if usemask=='auto-multithresh':
        base_name = base_name+'automask_multithresh_'



    image_update_model = str(n_interaction)+base_name+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+str(robust)+'.'+specmode+'.'+deconvolver+'.'+gridder

    tclean(vis=g_vis,
        imagename=os.path.dirname(g_name)+'/selfcal/'+image_update_model,
        spw=SPWS,field=FIELD,outlierfile=outlierfile,
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
        usepointing=usepointing,psfphasecenter=psfphasecenter,phasecenter=phasecenter,
        datacolumn=datacolumn,cycleniter=cycleniter,calcpsf=calcpsf,
        growiterations=growiterations,
        negativethreshold=negativethreshold,parallel=parallel,
        verbose=True,uvtaper=uvtaper,uvrange=uvrange,
        savemodel='modelcolumn')


    if PLOT==True:
        plot_visibilities(g_vis=g_vis,name=image_update_model,
            with_MODEL=True,with_CORRECTED=True)

    try:
        eview(os.path.dirname(g_name)+'/selfcal/'+image_update_model+'.image'+ext,data_range=data_range,
            scaling=-2.0,out=os.path.dirname(g_name)+'/selfcal/'+image_update_model+'.image'+ext+'.png')
    except:
        print('Error in ploting image with contours....')
        pass

    print(' Image (update) Statistics:')
    get_image_statistics(os.path.dirname(g_name)+'/selfcal/'+image_update_model+'.image'+ext)

    pass

def self_gain_cal(g_name,field,n_interaction,gain_tables=[],
    combine=combine,solnorm=False,uvtaper=[],uvrange='',
    niter=500,spwmap=[],
    specmode = 'mfs',deconvolver = 'mtmfs',ext = '',gridder = 'standard',
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



    image_update_model = str(n_interaction)+base_name+os.path.basename(g_name)\
                         +'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting\
                         +'.'+str(robust)+'.'+specmode+'.'+deconvolver+'.'+gridder


    caltable = os.path.dirname(g_name)+'/selfcal/'+str(n_interaction)\
               +cal_basename+os.path.basename(g_name)\
               +'_'+'_solint_'+solint+'_minsnr_'+str(minsnr)+'_combine'+combine+'_gtype_'+gaintype+'.tb'
    if not os.path.exists(caltable):
        if calmode=='ap':
            solonrm=True
        else:
            solnorm=False
        gaincal(vis=g_vis,field=FIELD,caltable=caltable,spwmap=spwmap,
                solint=solint,gaintable=gain_tables,combine=combine,
            refant=refant,calmode=calmode,gaintype=gaintype, minsnr=minsnr,solnorm=solnorm)
    else:
        print(' => Using existing caltable with same parameters asked.')
        print(' => Not computing again...')

    calibration_table_plot(table=caltable,
        fields='',yaxis='phase',
        table_type=str(n_interaction)+'_selfcal_phase_'+os.path.basename(g_name)+'_solint_'+solint+'_minsnr_'+str(minsnr)+'_combine'+combine+'_gtype_'+gaintype)

    if calmode=='ap':
        calibration_table_plot(table=caltable,
            fields='',yaxis='amp',
            table_type=str(n_interaction)+'_selfcal_ampphase_'+os.path.basename(g_name)+'_solint_'+solint+'_minsnr_'+str(minsnr)+'_combine'+combine+'_gtype_'+gaintype)


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

        applycal(vis=g_vis,gaintable=gain_tables,spwmap=spwmap,
                 flagbackup=False,calwt=False)

        print('     => Reporting data flagged after selfcal apply interaction',n_interaction,'...')
        summary_aft = flagdata(vis=g_vis, field=FIELD,mode='summary')
        report_flag(summary_aft,'field')

        if PLOT==True:
            plot_visibilities(g_vis=g_vis,name=image_update_model,
            with_CORRECTED=True,with_MODEL=False,with_DATA=False)

    return(gain_tables)

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

# os.environ['SAVE_ALL_AUTOMASKS']="false"
# # usemask='auto-multithresh'
# # interactive=False
# usemask='user'
# interactive=True


steps=[
       'startup',
       'save_init_flags',
       '0',
       # '1',
       # '2',
       # '3',
       # '4',
       ]

# run_mode = 'jupyter'
run_mode = 'terminal'


phasecenters = ["J2000 13:15:30.698 62.07.45.34",
                "J2000 13:15:23.162 62.06.44.84",
                "J2000 13:16:42.847 61.55.30.09",
                "J2000 13:14:24.69 62.19.45.78",
                "J2000 13:14:42.941 62.09.10.86",
                "J2000 13:18:27.001 62.00.36.27"]

imagesizes = [[3072, 3072],
              [3072, 3072],
              [1024, 1024],
              [1024, 1024],
              [1024, 1024],
              [1024, 1024]]
use_outlier_fields = True

if run_mode == 'terminal':
    path = '/run/media/sagauga/xfs_evo/lirgi_sample/combined_data/C_band/VV250/'
    image_list = ['VV250']
    proj_name = '_EVLA_eM.avg12s'

    for field in image_list:
        g_name = path + field + proj_name
        g_vis = g_name + '.ms'

        if 'startup' in steps:
            if not os.path.exists(path+'selfcal/'):
                os.makedirs(path+'selfcal/')
            else:
                print('>> Skiping create directory structure...')
            if not os.path.exists(path+'selfcal/plots'):
                os.makedirs(path+'selfcal/plots')
            else:
                print('>> Skiping create directory structure...')

        if 'save_init_flags' in steps:
            if not os.path.exists(g_name+'.ms.flagversions/flags.Original/'):
                flagmanager(vis=g_name+'.ms',mode='save',versionname='Original',
                    comment='Original flags.')
            else:
                print('     ==> Skipping flagging backup init (exists).')
                flagmanager(vis=g_name+'.ms',mode='restore',versionname='Original',
                    comment='Original flags.')
                print('     ==> Restoring Flags Instead...')

        if '0' in steps:
            iteration = '0'
            ############################################################################
            #### 0. Zero interaction. Use a small/negative robust parameter,        ####
            ####    to find the bright/compact emission(s).                         ####
            ############################################################################
            robust = 0.0 #decrease more if lots of failed solutions.
            niter = 300
            threshold = '40.0e-6Jy'

            if use_outlier_fields == True:
                outlierfile = create_external_file(g_vis=g_vis, iteration=iteration,
                                     phasecenters=phasecenters,
                                     imagesizes=imagesizes,
                                     output_file=os.path.dirname(g_vis)+'/'+iteration+'_outlier_fields_file.py')
            else:
                outlierfile = ''

            # os.environ['SAVE_ALL_AUTOMASKS'] = "true"
            start_image(g_name,field,n_interaction=0,
                        # uvtaper=['0.05arcsec'],
                        delmodel=True,PLOT=False,niter=niter,
                        robust=robust,interactive=interactive,cycleniter=25,
                        usemask=usemask,datacolumn='data')

            gain_tables_selfcal_temp=self_gain_cal(g_name,field,
                                                   n_interaction=iteration,
                                                   niter=niter,
                                                   minsnr = 0.5,
                                                   solint = 'inf',
                                                   flagbackup=True,
                                                   gaintype='G',combine='scan',
                # spwmap = [[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]],
                # spwmap=[[0, 0, 0, 0, 4, 4, 4, 4]],
                # spwmap=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                                   calmode='p',
                                                   action='calculate',
                                                   PLOT=False)

        os.environ['SAVE_ALL_AUTOMASKS'] = "false"
        if '1' in steps:
            iteration = '1'
            ############################################################################
            #### 1. First interaction. Increase a little the robust parameter,      ####
            ####    start to consider more extended emission.                       ####
            ############################################################################
            threshold = '20.0e-6Jy'
            niter = 300
            robust = 1.5  # or 0.5 if lots of extended emission.

            if use_outlier_fields == True:
                outlierfile = create_external_file(g_vis=g_vis, iteration=iteration,
                                     phasecenters=phasecenters,
                                     imagesizes=imagesizes,
                                     output_file=os.path.dirname(g_vis)+'/'+iteration+'_outlier_fields_file.py')
            else:
                outlierfile = ''

            update_model_image(g_name, field, robust = robust,
                               n_interaction=iteration,
                               scales=[0,8,16,32],
                               interactive=interactive,uvtaper=[], niter=niter,
                               usemask=usemask, cycleniter = 25, PLOT=False)

            gain_tables_selfcal_temp = self_gain_cal(g_name, field,
                                                     n_interaction=iteration,
                                                     minsnr=0.5, solint='3600s',
                                                     flagbackup=True,
                                                    #  spwmap=[[0, 0, 0, 0, 4, 4, 4, 4]],
                                                    #  spwmap=[
                                                    #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                    #       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                                                     gain_tables=[], calmode='p',
                                                     gaintype='T',combine='scan',
                                                     action='apply', PLOT=False)

        if '2' in steps:
            iteration = 2
            ############################################################################
            #### 2. Second interaction. Increase more the robust parameter, or use  ####
            ####    uvtapering. Consider even more extended emission (if there is). ####
            ############################################################################
            robust = 2.0
            threshold = '3.0e-6Jy'
            niter = 1000

            if use_outlier_fields == True:
                outlierfile = create_external_file(g_vis=g_vis, iteration=iteration,
                                     phasecenters=phasecenters,
                                     imagesizes=imagesizes,
                                     output_file=os.path.dirname(g_vis)+'/'+str(iteration)+'_outlier_fields_file.py')
            else:
                outlierfile = ''

            update_model_image(g_name,field,n_interaction=iteration,cycleniter=50,
                               interactive=interactive,robust = robust,
                               scales=[0, 8, 20,40,80],
                               # psfphasecenter='J2000 11h28m33.35  +58d33m49.10',
                               # phasecenter='J2000 11h28m33.35  +58d33m49.10',
                               # usepointing=True,
                               uvtaper=[],niter=niter,usemask=usemask,PLOT=False)

            gain_tables_selfcal = self_gain_cal(g_name, field, n_interaction=iteration,
                                                minsnr=0.5, solint='5000s',
                                                flagbackup=True,
                                                # spwmap=[[0, 0, 0, 0, 4, 4, 4, 4]],
                                                gain_tables=[],combine='scan',
                                                calmode='ap', gaintype='T',
                                                action='apply', PLOT=False)

        if '3' in steps:
            iteration = 3
            ############################################################################
            #### 2. Third interaction. Increase more the robust parameter, or use  ####
            ####    uvtapering. Consider even more extended emission (if there is). ####
            ############################################################################
            robust = 2.0
            threshold = '1.0e-6Jy'
            niter = 1000

            if use_outlier_fields == True:
                outlierfile = create_external_file(g_vis=g_vis, iteration=iteration,
                                     phasecenters=phasecenters,
                                     imagesizes=imagesizes,
                                     output_file=os.path.dirname(g_vis)+'/{iteration}_outlier_fields_file.py')
            else:
                outlierfile = ''

            update_model_image(g_name,field,n_interaction=iteration,cycleniter=75,
                               interactive=interactive,robust = robust,
                               # uvtaper=['0.3arcsec'],
                               # psfphasecenter='J2000 11h28m33.35  +58d33m49.10',
                               # phasecenter='J2000 11h28m33.35  +58d33m49.10',
                               # usepointing=True,
                               niter=niter,
                               usemask=usemask,PLOT=True)

            gain_tables_selfcal = self_gain_cal(g_name, field, n_interaction=iteration,
                                                minsnr=0.5, solint='60s',
                                                flagbackup=True,combine='',
                                                gain_tables=gain_tables_selfcal,
                                                calmode='p', gaintype='T',
                                                # spwmap=[[0, 0, 0, 0, 4, 4, 4, 4],[]],
                                                # spwmap = [[],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                                                action='apply', PLOT=True)

        if '4' in steps:
            iteration = '4'
            # ############################################################################
            # #### 3. Fourth interaction.If you see that further improvements can be   ####
            # ####    obtained, do one more interaction, now amp selfcal.             ####
            # ####    Be sure that the previous phase gains are ok, because you       ####
            # ####    need them for the amp gain. If they are not, consider           ####
            # ####    to iterate as many times you see fit in phases again.           ####
            # ############################################################################
            robust = 2.0
            threshold = '3e-6Jy'
            niter = 2000

            if use_outlier_fields == True:
                outlierfile = create_external_file(g_vis=g_vis, iteration=iteration,
                                     phasecenters=phasecenters,
                                     imagesizes=imagesizes,
                                     output_file=os.path.dirname(g_vis)+'/'+iteration+'_outlier_fields_file.py')
            else:
                outlierfile = ''

            update_model_image(g_name,field,n_interaction=iteration,
                               cycleniter = 100,
                               interactive=interactive,robust = robust,
                               # uvtaper=['0.6arcsec'],
                               niter=niter,usemask=usemask,PLOT=False)

            gain_tables_selfcal = self_gain_cal(g_name, field,
                                                n_interaction=iteration,
                                                minsnr=1.0, solint='60s',
                                                flagbackup=True, solnorm=True,
                                                gain_tables=gain_tables_selfcal,
                                                calmode='ap', gaintype='G',
                                                combine='',
                                                action='apply', PLOT=False)
        #careful here
        if '5' in steps:
            iteration = '5'
            robust = 2.0
            threshold = '4e-6Jy'
            niter = 2000

            if use_outlier_fields == True:
                outlierfile = create_external_file(g_vis=g_vis, iteration=iteration,
                                     phasecenters=phasecenters,
                                     imagesizes=imagesizes,
                                     output_file=os.path.dirname(g_vis)+'/'+iteration+'_outlier_fields_file.py')
            else:
                outlierfile = ''

            update_model_image(g_name,field,n_interaction=iteration,
                               cycleniter = 100,
                               interactive=interactive,robust = robust,
                               uvtaper=[],niter=niter,usemask=usemask,PLOT=True)

            gain_tables_selfcal = self_gain_cal(g_name, field,
                                                n_interaction=iteration,
                                                minsnr=2.0, solint='32s',
                                                flagbackup=True, solnorm=True,
                                                gain_tables=gain_tables_selfcal,
                                                calmode='ap', gaintype='G',
                                                combine='',
                                                action='apply', PLOT=True)


if run_mode == 'jupyter':
    print('selfcal script is not doing anything, you can use it on a '
          'jupyter notebook. For that, you have to manually '
          'set your variable names. ')
