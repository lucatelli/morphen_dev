import pandas as pd
import numpy as np
import argparse
import os

data_range = [20e-06, 0.005]


def eview(imagename, contour=None,
          data_range=[20e-06, 0.005],
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


def imaging(g_name, field, uvtaper, robust, base_name='clean_image'):
    g_vis = g_name + '.ms'
    """
    # uvtaper_mode+uvtaper_args+'.'+uvtaper_addmode+uvtaper_addargs+
    """
    print(uvtaper_addmode, uvtaper_addargs, robust)
    if uvtaper is not '':
        taper = 'taper_'
    else:
        taper = ''

    image_deepclean_name = (base_name + '_' + g_name + '_' +
                            imsizex + 'x' + imsizey + '_' + \
                            cell + '_' + niter + '.' + weighting + '.' + \
                            deconvolver[1:] + '.' + taper + \
                            uvtaper + '.' + str(robust))

    ext = ''
    if '-join-channels' in deconvolver_args:
        print('Using mtmfs method.')
        ext = ext + '-MFS'
    ext = ext + '-image.fits'

    print(image_deepclean_name)

    if not os.path.exists(root_dir_sys + image_deepclean_name + ext):
        if running_container == 'native':
            os.system(
                'mpirun -np 4 wsclean-mp -name ' + root_dir + image_deepclean_name +
                ' -size ' + imsizex + ' ' + imsizey + ' -scale ' + cell +
                ' ' + gain_args + ' -niter ' + niter + ' -weight ' + weighting +
                ' ' + robust + ' ' + auto_mask + ' ' + auto_threshold + mask_file +
                ' ' + deconvolver + ' ' + deconvolver_options +
                ' ' + deconvolver_args + ' ' + taper_mode + uvtaper +
                ' ' + opt_args + ' ' + data_column + ' ' + root_dir + g_vis)
        if running_container == 'singularity':
            os.system(
                'singularity exec --nv --bind ' + mount_dir + ' ' + wsclean_dir +
                ' ' + 'mpirun -np 4 wsclean-mp -name ' + root_dir +
                image_deepclean_name +
                ' -size ' + imsizex + ' ' + imsizey + ' -scale ' + cell +
                ' ' + gain_args + ' -niter ' + niter + ' -weight ' + weighting +
                ' ' + robust + ' ' + auto_mask + ' ' + auto_threshold + mask_file +
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


def parse_float_list(str_values):
    return [float(val.strip()) for val in str_values.strip('[]').split(',')]


def parse_str_list(str_values):
    return [s.strip() for s in str_values.strip('[]').split(',')]


if __name__ == "__main__":
    # Define and parse the command-line arguments

    parser = argparse.ArgumentParser(description="Helper for wsclean imaging.")
    parser.add_argument("--p", type=str, help="The path to the MS file.")
    parser.add_argument("--f", nargs='?', default=False,
                        const=True, help="The name of the ms file")
    parser.add_argument("--r",
                        type=parse_float_list, nargs='?',
                        const=True, default=[0.5],
                        help="List of robust values")
    parser.add_argument("--t", type=parse_str_list, nargs='?',
                        const=True, default=[''],
                        help="List of sky-tapers values")

    parser.add_argument("--mask", nargs='?', default=None,
                        const=True, help="A fits-file mask to be used.")

    parser.add_argument("--data", type=str, nargs='?', default='DATA',  # 'CORRECTED_DATA'
                        help="Which data column to use")

    parser.add_argument("--wsclean_install", type=str, nargs='?', default='native',
                        help="How wsclean was installed (singularity or native)?")

    # To do: add option for wsclean singularity image path.

    parser.add_argument("--update_model", type=str, nargs='?', default='False',
                        help="Update model after cleaning?")

    parser.add_argument("--with_multiscale", type=str, nargs='?', default='False',
                        help="Use multiscale deconvolver?")

    parser.add_argument("--shift", type=str, nargs='?', default=None,
                        help="New phase center to shift for imaging."
                             "Eg. --shift 13:15:30.68 +62.07.45.357")

    parser.add_argument("--opt_args", type=str, nargs='?', default='',
                        help="Optional/additional arguments to be passed to "
                             "wsclean.")

    parser.add_argument("--sx", type=str, nargs='?', default='2048',
                        help="Image Size x-axis")
    parser.add_argument("--sy", type=str, nargs='?', default='2048',
                        help="Image Size y-axis")
    parser.add_argument("--cellsize", type=str, nargs='?', default='0.05asec',
                        help="Cell size")
    parser.add_argument("--niter", type=str, nargs='?', default='5000',
                        help="Number of iterations during cleaning.")

    parser.add_argument("--maxuv_l", type=str, nargs='?', default=None,
                        help="Max uv distance in lambda.")

    parser.add_argument("--minuv_l", type=str, nargs='?', default=None,
                        help="Min uv distance in lambda.")

    parser.add_argument("--nsigma_automask", type=str, nargs='?', default='10.0',
                        help="Sigma level for automasking in wsclean.")

    parser.add_argument("--nsigma_autothreshold", type=str, nargs='?', default='0.5',
                        help="Sigma level for autothreshold in wsclean.")

    parser.add_argument("--quiet", type=str, nargs='?', default='False',
                        help="Print wsclean output?")

    # parser.add_argument("--opt_args", nargs=argparse.REMAINDER,
    #                     default=['-multiscale -multiscale-scales 0,8,16,32 '
    #                             '-multiscale-scale-bias 0.75 '],
    #                     help="Optional arguments passed to wsclean.")

    # parser.add_argument("--opt_args", type=str, nargs='*',
    #                     default=' -multiscale -multiscale-scales 0,8,16,'
    #                             '32 -multiscale-scale-bias 0.75 ',
    #                     help="Optional arguments passed to wsclean.")

    # parser.add_argument("--opt_args", type=str, nargs='*',
    #                     default=' -multiscale -multiscale-scales 0,8,16,'
    #                             '32 -multiscale-scale-bias 0.75 ',
    #                     help="Optional arguments passed to wsclean as a single string.")


    parser.add_argument("--save_basename", type=str, nargs='?', default='image',
                        help="optional basename for saving image files.")

    args = parser.parse_args()
    # args, extra_args = parser.parse_known_args()
    # opt_args = args.opt_args
    # opt_args_list = opt_args.split()

    if args.update_model == 'True':
        update_model_option = ' -update-model-required '
    else:
        update_model_option = ' -no-update-model-required '

    running_container = args.wsclean_install

    if running_container == 'native':
        os.system('export OPENBLAS_NUM_THREADS=1')

    # for i in range(len(image_list)):
    field = os.path.basename(args.f).replace('.ms', '')
    g_name = field
    root_dir_sys = os.path.dirname(args.f) + '/'
    robusts = args.r
    tapers = args.t

    if running_container == 'singularity':
        mount_dir = root_dir_sys + ':/mnt'
        root_dir = '/mnt/'
        wsclean_dir = '/home/sagauga/apps/wsclean_wg_eb.simg'
        # wsclean_dir = '/home/sagauga/apps/wsclean_nvidia470_gpu.simg'
        # wsclean_dir = '/raid1/scratch/lucatelli/apps/wsclean_wg_eb.simg'
    if running_container == 'native':
        mount_dir = ''
        root_dir = root_dir_sys
        os.system('export OPENBLAS_NUM_THREADS=1')

    base_name = args.save_basename

    ## Setting image and deconvolution noptions.
    ### Cleaning arguments
    auto_mask = ' -auto-mask ' + args.nsigma_automask
    # auto_mask = ' '
    auto_threshold = ' -auto-threshold ' + args.nsigma_autothreshold
    if args.mask == 'None' or args.mask == None:
        mask_file = ' '
    else:
        # if args.mask != 'None' or args.mask != None:
        if running_container == 'native':
            mask_file = ' -fits-mask ' + args.mask + ' '
        if running_container == 'singularity':
            mask_file = ' -fits-mask ' + root_dir+os.path.basename(args.mask) + ' '

    # auto_threshold = '-threshold 1.0e-6Jy'
    # threshold = '-threshold 5.0e-6Jy'
    # base_name = '1_update_model'
    # base_name = '1_selfcal_image'


    # data to run deconvolution
    data_column = ' -data-column ' + args.data
    with_multiscale = args.with_multiscale
    ### Selecting the deconvolver
    deconvolution_mode = 'robust'
    if deconvolution_mode == 'robust':
        if with_multiscale == True or with_multiscale == 'True':
            deconvolver = '-multiscale'
            deconvolver_options = ( ' -multiscale-scales 0,5,20,40'
                                    ' -multiscale-scale-bias '
                                   '0.8 -multiscale-gain 0.05 ')
            # deconvolver = ''
            # deconvolver_options = opt_args_list
            print(' ++>> ', deconvolver_options)
            if deconvolver_options is not '' or []:
                if 'multiscale' in deconvolver_options:
                    print(' ++>> Using Multiscale deconvolver.')
                # else:
                #     print(' ++>> Using Hogbom deconvolver.')
                #     deconvolver = 'multiscale'


            # deconvolver_options = ('-multiscale-max-scales 5 -multiscale-scale-bias 0.5 ')

        else:
            deconvolver = ''
            deconvolver_options = ('')

        deconvolver_args = (' '
                            '-channels-out 4 -join-channels '
                            # '-channel-division-frequencies 4.0e9,4.5e9,5.0e9,5.5e9,'
                            # '29e9,31e9,33e9,35e9 ' #-gap-channel-division
                            # '-deconvolution-threads 24 -j 24 -parallel-reordering 24 '
                            '-weighting-rank-filter 3 -weighting-rank-filter-size 64 '
                            '-gridder wgridder  ' #-wstack-nwlayers-factor 12  -wstack-nwlayers-factor 6
                            '-parallel-deconvolution 3072 '  # -local-rms -local-rms-window 100
                            '-no-mf-weighting -circular-beam ' # -beam-size 0.1arcsec     -beam-size 
                            # 0.05arcsec
                            #-circular-beam 
                            # '-apply-primary-beam  -circular-beam '
                            # '-gridder idg -idg-mode hybrid -apply-primary-beam ' 
                            # '-local-rms -local-rms-window 25 -parallel-deconvolution 1024 '
                            # '-local-rms-method rms '
                            '-save-source-list '
                            '-fit-spectral-pol 3 '
                            '')
    if deconvolution_mode == 'FOV':
        deconvolver = ' '
        deconvolver_options = (' ')
        deconvolver_args = ('-gridder idg -idg-mode hybrid -save-source-list '
                            '-deconvolution-threads 1 -parallel-deconvolution 1024 ')

    # image parameters
    weighting = 'briggs'
    # robust = '0.5'
    imsizex = args.sx
    imsizey = args.sy
    cell = args.cellsize
    niter = args.niter

    # taper options (this is a to-do)
    # uvtaper_mode = '-taper-tukey'
    # uvtaper_args = '900000'
    # uvtaper_addmode = '-maxuv-l'
    # uvtaper_addargs = '800000'
    # taper_mode='-taper-gaussian '
    # uvtaper_mode = '-taper-gaussian'
    # uvtaper_args = '0.05asec'
    uvtaper_addmode = ''
    uvtaper_addargs = ''
    # uvtaper = uvtaper_mode + ' '+ uvtaper_args + ' ' +uvtaper_addmode + ' ' +
    # uvtaper_addargs
    uvselection = ''
    if args.maxuv_l is not None:
        uvselection = ' -maxuv-l ' + args.maxuv_l + ' '
    if args.minuv_l is not None:
        uvselection = uvselection + ' -minuv-l ' + args.minuv_l + ' '
    # general arguments
    gain_args = ' -mgain 0.4 -gain 0.05 -nmiter 200'

    if args.shift == 'None' or args.shift == None:
        # if args.shift != ' ':
        shift_options = ' '
    else:
        shift_options = ' -shift ' + args.shift + ' '
    # shift_options = ' '  # -shift 13:15:30.68  +62.07.45.357 '#' -shift 13:15:28.903
    # +62.07.11.886 '
    if args.quiet == 'True':
        quiet = ' -quiet '
    else:
        quiet = ' '
    opt_args = (' -super-weight 3.0 -mem 80 -abs-mem 35 '
                # '-pol RL,LR -no-negative '
                # ' -save-first-residual -save-weights -save-uv '-maxuv-l 3150000
                ' '+uvselection+args.opt_args+' '
                ' -log-time -field all ' + quiet + update_model_option + ' ')
    opt_args = opt_args + shift_options



        # wsclean_dir = '/home/sagauga/apps/wsclean_nvidia470_gpu.simg'
        # wsclean_dir = '/raid1/scratch/lucatelli/apps/wsclean_wg_eb.simg'
    for robust in robusts:
        for uvtaper in tapers:
            if uvtaper == '':
                taper_mode = ''
            else:
                taper_mode = '-taper-gaussian '
            image_statistics = imaging(g_name=g_name,
                                       # base_name='2_selfcal_update_model',
                                       # base_name='image',
                                       base_name=base_name,
                                       field=field, robust=str(robust),
                                       uvtaper=uvtaper)

            if image_statistics is not None:
                image_statistics['robust'] = robust
                # image_statistics['vwt'] = vwt
                image_statistics['uvtaper'] = uvtaper
                df = pd.DataFrame.from_dict(image_statistics, orient='index').T
                df.to_csv(root_dir_sys + image_statistics['imagename'].replace('.fits',
                                                                               '_data.csv'),
                          header=True,
                          index=False)
            else:
                pass