import pandas as pd
import numpy as np
import os

running_container = 'singularity' # or 'native'

root_dir_sys = '/nvme1/scratch/lucatelli/lirgi/combined_data/UGC5101_C/'

if running_container == 'singularity':
    mount_dir = root_dir_sys+':/mnt'
    root_dir = '/mnt/'
    wsclean_dir = '/raid1/scratch/lucatelli/apps/wsclean-gpu.simg'



## Setting image and deconvolution noptions.
### Cleaning arguments
auto_mask = '-auto-mask 3.0'
auto_threshold = '-auto-threshold 0.001'
#auto_threshold = '-threshold 1.0e-6Jy'
# threshold = '-threshold 5.0e-6Jy'


### Selecting the deconvolver
deconvolver = '-multiscale'
# deconvolver_options = '-multiscale-max-scales 15'#'-multiscale-scales 0,3,6,9,15,20,30,60,120,240'
deconvolver_options = '-multiscale-scales 0,4,8,16,32,64,128,256'
# deconvolver_args = '-channels-out 5 -join-channels'# -nmiter 100 -local-rms -local-rms -local-rms-method rms-with-min
deconvolver_args = '-channels-out 4 -join-channels -weighting-rank-filter 3 ' \
                   '-weighting-rank-filter-size 64 -no-mf-weighting ' \
                   '-use-wgridder'#'  # -nmiter 100 -local-rms -circular-beam

#image parameters
weighting = 'briggs'
# robust = '0.5'
imsize= '3072'
cell = '0.008arcsec'
niter= '10000'


#taper options (this is a to-do)
# uvtaper_mode = '-taper-tukey'
# uvtaper_args = '900000'
# uvtaper_addmode = '-maxuv-l'
# uvtaper_addargs = '800000'
# taper_mode='-taper-gaussian '
# uvtaper_mode = '-taper-gaussian'
# uvtaper_args = '0.05asec'
uvtaper_addmode = ''
uvtaper_addargs = ''
# uvtaper = uvtaper_mode + ' '+ uvtaper_args + ' ' +uvtaper_addmode + ' ' + uvtaper_addargs

#data to run deconvolution
data_column = '-data-column DATA'

#general arguments
gain_args = '-mgain 0.3 -multiscale-gain 0.05 -gain 0.05'
shift_options = ''#'-shift 13:15:30.6915 +062.07.45.3489 '
opt_args = '-super-weight 15.0 -mem 90 -j 64 -parallel-gridding 64 ' \
           '-parallel-reordering 1 -deconvolution-threads 64 ' \
           '-log-time -no-update-model-required -field all'
opt_args = opt_args + shift_options

data_range = [1e-06, 0.001]


def eview(imagename, contour=None,
          data_range=[-2.52704e-05, 0.0159025],
          colormap='Rainbow 2', scaling=-2.0, zoom=4, out=None):
    if contour == None:
        contour = imagename
    # if out==None:
    #     out = imagename + '_drawing.png'
    imview(raster={
        'file': imagename,
        # 'range': data_range,
        'colormap': colormap, 'scaling': scaling, 'colorwedge': True},
        contour={'file': contour,
                 'levels': [-0.1, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4,
                            0.6, 0.8]},
        # axes={'x':'Declination'} ,
        # zoom={'blc': [3,3], 'trc': [3,3], 'coord': 'pixel'},
        zoom=zoom,
        out=out,
        # scale=scale,
        # dpi=dpi,
        # orient=orient
    )


def get_image_statistics(imagename, residualname=None, region=''):
    if residualname == None:
        try:
            residualname = imagename.replace('.image', '.residual')
        except:
            print('Please, provide the residual image name')

    dic_data = {}
    dic_data['imagename'] = imagename

    stats_im = imstat(imagename=imagename, region=region)
    stats_re = imstat(imagename=residualname)

    box_edge, imhd = create_box(imagename)
    stats_box = imstat(imagename=imagename, box=box_edge)

    # determine the flux flux peak of image and residual
    flux_peak_im = stats_im['max'][0]
    flux_peak_re = stats_re['max'][0]
    dic_data['max_im'] = flux_peak_im
    dic_data['max_re'] = flux_peak_re

    # determine the rms and std of residual and of image
    rms_re = stats_re['rms'][0]
    rms_im = stats_im['rms'][0]
    rms_box = stats_box['rms'][0]

    sigma_re = stats_re['sigma'][0]
    sigma_im = stats_im['sigma'][0]
    sigma_box = stats_box['sigma'][0]

    dic_data['rms_im'] = rms_im
    dic_data['rms_re'] = rms_re
    dic_data['rms_box'] = rms_box

    dic_data['DR'] = rms_re / flux_peak_im

    # determine the image and residual flux
    flux_im = stats_im['flux'][0]
    flux_box = stats_box['flux'][0]
    # flux_re = stats_re['flux']
    dic_data['flux_im'] = flux_im
    dic_data['flux_box'] = flux_box

    sumsq_im = stats_im['sumsq'][0]
    sumsq_re = stats_re['sumsq'][0]

    q = sumsq_im / sumsq_re
    # flux_ratio = flux_re/flux_im

    snr_im = flux_im / rms_im
    snr = flux_im / rms_re
    snr_box = flux_im / rms_box

    dic_data['snr'] = snr
    dic_data['snr_box'] = snr
    dic_data['snr_im'] = snr_im

    peak_im_rms = flux_peak_im / rms_im
    peak_re_rms = flux_peak_re / rms_re

    dic_data['bmajor'] = imhd['restoringbeam']['major']['value']
    dic_data['bminor'] = imhd['restoringbeam']['minor']['value']
    dic_data['positionangle'] = imhd['restoringbeam']['positionangle']['value']

    print(' Flux=%.5f Jy/Beam' % flux_im)
    print(' Flux peak (image)=%.5f Jy' % flux_peak_im,
          'Flux peak (residual)=%.5f Jy' % flux_peak_re)
    print(' flux_im/sigma_im=%.5f' % snr_im, 'flux_im/sigma_re=%.5f' % snr)
    print(' rms_im=%.5f' % rms_im, 'rms_re=%.5f' % rms_re)
    print(' flux_peak_im/rms_im=%.5f' % peak_im_rms,
          'flux_peak_re/rms_re=%.5f' % peak_re_rms)
    print(' sumsq_im/sumsq_re=%.5f' % q)
    return (dic_data)


def create_box(imagename):
    """
    Create a box with 20% of the image
    at an edge (upper left) of the image.
    """
    ihl = imhead(imagename, mode='list')
    ih = imhead(imagename)

    M = ihl['shape'][0]
    N = ihl['shape'][1]
    frac_X = int(0.1 * M)
    frac_Y = int(0.1 * N)
    slice_pos_X = 0.15 * M
    slice_pos_Y = 0.85 * N

    box_edge = np.asarray([slice_pos_X - frac_X,
                           slice_pos_Y - frac_Y,
                           slice_pos_X + frac_X,
                           slice_pos_Y + frac_Y]).astype(int)

    box_edge_str = str(box_edge[0]) + ',' + str(box_edge[1]) + ',' + \
                   str(box_edge[2]) + ',' + str(box_edge[3])

    return (box_edge_str, ih)


def imaging(g_name, field, uvtaper, robust, base_name='clean_image_'):
    g_vis = g_name + '.ms'
    """
    # uvtaper_mode+uvtaper_args+'.'+uvtaper_addmode+uvtaper_addargs+
    """
    print(uvtaper_addmode, uvtaper_addargs, robust)
    if uvtaper is not '':
        taper = 'taper_'
    else:
        taper = ''

    image_deepclean_name = base_name + '_' + g_name + '_' + imsize + '_' + \
                           cell + '_' + niter + '.' + weighting + '.' + \
                           deconvolver[1:] + '.' + taper + \
                           uvtaper + '.' + str(robust)

    ext = ''
    if '-join-channels' in deconvolver_args:
        print('Using mtmfs method.')
        ext = ext + '-MFS'
    ext = ext + '-image.fits'

    print(image_deepclean_name)

    if not os.path.exists(image_deepclean_name + ext):
        if running_container == 'native':
            os.system(
                'wsclean -name ' + root_dir_sys + image_deepclean_name +
                ' -size ' + imsize + ' ' + imsize + ' -scale ' + cell +
                ' ' + gain_args + ' -niter ' + niter + ' -weight ' + weighting +
                ' ' + robust + ' ' + auto_mask + ' ' + auto_threshold +
                ' ' + deconvolver + ' ' + deconvolver_options +
                ' ' + deconvolver_args + ' ' + taper_mode + uvtaper +
                ' ' + opt_args + ' ' + data_column + ' ' + root_dir_sys + g_vis)
        if running_container == 'singularity':
            os.system(
                'singularity exec --bind ' + mount_dir + ' ' + wsclean_dir +
                ' ' + 'wsclean -name ' + root_dir + image_deepclean_name +
                ' -size ' + imsize + ' ' + imsize + ' -scale ' + cell +
                ' ' + gain_args + ' -niter ' + niter + ' -weight ' + weighting +
                ' ' + robust + ' ' + auto_mask + ' ' + auto_threshold +
                ' ' + deconvolver + ' ' + deconvolver_options +
                ' ' + deconvolver_args + ' ' + taper_mode + uvtaper +
                ' ' + opt_args + ' ' + data_column + ' ' + root_dir + g_vis)

        print(' Image Statistics:')
        image_stats = {
            "#basename": image_deepclean_name + ext}  # get_image_statistics(image_deep_selfcal  + ext)
        image_stats['imagename'] = image_deepclean_name + ext
        '''
        save dictionary to file
        '''
        return (image_stats)
    else:
        print('Skipping imaging; already done.')
        return (None)

    # pass


# proj_name = 'VV705_L26_emerlin'
# image_list = ['1518+4244']#,'VV250a','VV705']
image_list = ['UGC5101_combined_w_0.55_RR_LL']
robusts = [-1.0,-0.5
    # -2.0,-1.8,-1.4,-1.0,-0.8,-0.6,-0.2,0.0,0.5,1.0
    ]
tapers = ['','0.25asec']
# for fixed visibility weights during concat.
for field in image_list:
    g_name = field
    for robust in robusts:
        for uvtaper in tapers:
            if uvtaper == '':
                taper_mode = ''
            else:
                taper_mode = '-taper-gaussian '
            image_statistics = imaging(g_name=g_name,
                                       base_name='clean_image',
                                       field=field, robust=str(robust),
                                       uvtaper=uvtaper)

            if image_statistics is not None:
                image_statistics['robust'] = robust
                # image_statistics['vwt'] = vwt
                image_statistics['uvtaper'] = uvtaper
                df = pd.DataFrame.from_dict(image_statistics, orient='index').T
                df.to_csv(root_dir_sys+image_statistics['imagename'].replace('.fits','_data.csv'), header=True,
                          index=False)
            else:
                pass