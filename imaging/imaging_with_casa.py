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

Utility to easy imaging with CASA.
"""
__version__ = 0.3
__author__  = 'Geferson Lucatelli'
__email__   = 'geferson.lucatelli@postgrad.manchester.ac.uk'
__date__    = '2023 21 08'
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
imsize = 2048
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
cell = '0.08arcsec'

smallscalebias=0.7
robust = 0.0
gain = 0.05
pblimit=-0.1
nterms = 1
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


outlierfile='/run/media/sagauga/xfs_evo/lirgi_sample/vla/C_band/VV250/outliers_VV250.txt'

# proj_name = '.calibrated'



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


def get_tb_data(table,param):
    tb.open(table)
    param_data = tb.getcol(param).ravel()
    tb.close()
    return(param_data)

def do_imaging(g_name,field,n_interaction,robust=0.5,
                       specmode = 'mfs',deconvolver = 'mtmfs',ext = '',gridder = 'standard',
                       # gridder = 'wproject',# specmode = 'mfs'
                       scales=[0,8,16,32,64],
                       interactive=interactive,datacolumn='data',
                       usemask=usemask,niter = 5000,mask='',cycleniter=100,
                       usepointing=True,psfphasecenter='',phasecenter='',
                       uvtaper=[],uvrange='',PLOT=False):

    if deconvolver == 'mtmfs':
        ext = ext + '.tt0'

    g_vis = g_name + '.ms'


    base_name = 'clean_image_'


    if interactive == True:
        base_name = base_name+'interactive_'
    if usemask=='auto-multithresh':
        base_name = base_name+'automask_multithresh_'



    image_name = str(n_interaction)+base_name+os.path.basename(g_name)+'_'+str(imsize)+'_'+cell+'_'+str(niter)+'.'+weighting+'.'+str(robust)+'.'+specmode+'.'+deconvolver+'.'+gridder

    tclean(vis=g_vis,
        imagename=os.path.dirname(g_name)+'/imaging/'+image_name,
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
        savemodel='none')

    try:
        eview(os.path.dirname(g_name)+'/imaging/'+image_name+'.image'+ext,data_range=data_range,
            scaling=-2.0,out=os.path.dirname(g_name)+'/imaging/'+image_name+'.image'+ext+'.png')
    except:
        print('Error in ploting image with contours....')
        pass

    print(' Image (update) Statistics:')
    get_image_statistics(os.path.dirname(g_name)+'/imaging/'+image_name+'.image'+ext)
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

# os.environ['SAVE_ALL_AUTOMASKS']="false"
# # usemask='auto-multithresh'
# # interactive=False
# usemask='user'
# interactive=True


steps=[
       'imaging'
       ]

# run_mode = 'jupyter'
run_mode = 'terminal'

if run_mode == 'terminal':
    path = '/run/media/sagauga/xfs_evo/lirgi_sample/vla/C_band/VV250/'
    image_list = ['VV250a']  # ,'VV250a','VV705']
    proj_name = '_AL746_selfcalibrated.avg8s'
    for field in image_list:
        g_name = path + field + proj_name
        g_vis = g_name + '.ms'

        os.environ['SAVE_ALL_AUTOMASKS'] = "false"
        if 'imaging' in steps:
            if not os.path.exists(path+'imaging/'):
                os.makedirs(path+'imaging/')
            else:
                print('>> Skiping create directory structure...')
            threshold = '10.0e-6Jy'
            niter = 1000
            robust = 1.0  # or 0.5 if lots of extended emission.

            do_imaging(g_name, field, robust = robust, n_interaction='',
                               scales=[0, 8, 20,40],
                               interactive=interactive,uvtaper=[], niter=niter,
                               usemask=usemask, cycleniter = 100, PLOT=False)


if run_mode == 'jupyter':
    print('selfcal script is not doing anything, you can use it on a '
          'jupyter notebook. For that, you have to manually '
          'set your variable names. ')
