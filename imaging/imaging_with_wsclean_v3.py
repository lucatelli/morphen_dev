import pandas as pd
import numpy as np
import os
os.system('/usr/local/cuda/extras/demo_suite/bandwidthTest &> /dev/null')
os.system('export CUDA_DEVICE=0')
running_container = 'singularity' # or 'native'

## Setting image and deconvolution noptions.
### Cleaning arguments
auto_mask = '-auto-mask 5.0'
auto_threshold = '-auto-threshold 0.001'
#auto_threshold = '-threshold 1.0e-6Jy'
# threshold = '-threshold 5.0e-6Jy'


### Selecting the deconvolver
deconvolver = '-multiscale'
deconvolver_options = '-multiscale-scales 0,4,12,20,32,64'
deconvolver_args = ('-channels-out 4 -join-channels -weighting-rank-filter 5 '
                    '-weighting-rank-filter-size 64 -no-mf-weighting '
                    '-gridder wgridder -apply-primary-beam -local-rms '
                    '-local-rms-window 25 -local-rms-method rms '
                    '-save-source-list ')
                   #' -gridder idg -idg-mode hybrid -grid-with-beam '#'  # -no-negative -nmiter 100 -local-rms

#image parameters
weighting = 'briggs'
# robust = '0.5'
imsize= '2048'
cell = '0.008arcsec'
niter= '2000'


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
# data_column = '-data-column DATA'
data_column = '-data-column CORRECTED_DATA'

#general arguments
gain_args = '-mgain 0.3 -multiscale-gain 0.05 -gain 0.05'
lirgi_coordinates = {'VV250b' : 'J2000 13:15:30.698 62.07.45.34'}

shift_options = ''#' -shift 13:15:30.698 62.07.45.34 '
opt_args = '-super-weight 20.0 -mem 90 -j 10 ' \
           '-parallel-reordering 10 -deconvolution-threads 10 ' \
           '-save-first-residual -save-weights -save-uv ' \
           '-log-time -no-update-model-required -field all ' #-no-update-model-required
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

    if not os.path.exists(root_dir_sys+image_deepclean_name + ext):
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
                'singularity exec --nv --bind ' + mount_dir + ' ' + wsclean_dir +
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


root_dir_sys_list = [
                     '/run/media/sagauga/xfs_evo/lirgi_sample/combined_data/C_band/VV250/',
                     ]
image_list = [
              'VV250_EVLA_eM.avg12s'
             ]

robusts = [-1.0,0.0,1.0]
tapers = ['']
for i in range(len(image_list)):
    field = image_list[i]
    root_dir_sys = root_dir_sys_list[i]
    if running_container == 'singularity':
        mount_dir = root_dir_sys + ':/mnt'
        root_dir = '/mnt/'
        wsclean_dir = '/home/sagauga/apps/wsclean_wg_eb.simg'
        # wsclean_dir = '/raid1/scratch/lucatelli/apps/wsclean_wg_eb.simg'
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