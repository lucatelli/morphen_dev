# Image deconvolution Using `wsclean`

Some features includes: 
 - automatically save images named with parameters used during deconvolution
 - loop over different robust parameters or over a list of measurement sets. 
 - can be used with `wsclean` installed on singularity containers.

Note: This was tested with wsclean verions > 3.0 only. Some arguments/options are not
available for wsclean <3.0

## Setting up arguments
The first point is to know where `wsclean` is installed, natively or in singularity.

### Native mode
If you have wsclean installed locally, you only need to set the path to the data.

    root_dir_sys = '/media/user/data/Arp220/'

### Singularity Mode
If you are using singularity,in addition of setting the variable `root_dir_sys` 
you must specify the mounting directory where your data is stored. For example:

    mount_dir = root_dir_sys+':/mnt'

Then, the root_dir where your data will be stored becomes:

    root_dir = '/mnt/'

In case of using singularity, we need also to specify where `wsclean` image is 
located:

    wsclean_dir = '/home/user/apps/wsclean-gpu.simg

## Setting image and deconvolution options.
### Cleaning arguments

    auto_mask = '-auto-mask 3.0'
    auto_threshold = '-auto-threshold 0.001'

This controls how deep deconvolution is performed. 
Following multiple instructions from wsclean manual and private comunications 
a good starting point to obtain good image quality is to set an auto mask of 3 
(which is standard) but using a very low threshold level (deep clean inside the mask).
You must experiment with your data, but usually what can be done, start with `--auto_threshold 0.1` 
or `-auto_threshold 0.01`, and check.

You can also specify manually the threshold level, for example:

    auto_threshold = '-threshold 1.0e-6Jy'

### Selecting the deconvolver
With the deconvolver argument, please select any deconvolver you want to use. 
Just check if the options are consistent and implemented. 
Please, contac me about any issue using a different deconvolver/options.

For example, one can use the multiscale deconvolver and add options to it, e.g. 

    deconvolver = '-multiscale'
    deconvolver_options = '-multiscale-scales 0,4,8,16,32,64'

You can eaasily pass other arguments, such as

    deconvolver_args = '-channels-out 4 -join-channels -weighting-rank-filter 3 ' \
                       '-weighting-rank-filter-size 64 -no-mf-weighting ' \
                       '-no-negative -use-wgridder'

### Image parameters
Set basic image parameters, such as:

    weighting = 'briggs'
    imsize= '3072'
    cell = '0.008arcsec'
    niter= '10000'

Note that the robust parameter is specified later. 
### Taper options
This is a to-do.... <br>
For now, there is a loop at the end of the code that uses the `-taper-gaussian` mode, where one can 
loop over different sky-tapers. 

### Data to run deconvolution
    data_column = '-data-column DATA'

### General arguments

    gain_args = '-mgain 0.3 -multiscale-gain 0.05 -gain 0.05'
    shift_options = ''
    opt_args = '-super-weight 15.0 -mem 90 -j 32 -parallel-gridding 32 ' \
               '-parallel-reordering 1 -deconvolution-threads 32 ' \
               '-log-time -no-update-model-required -field all'
    opt_args = opt_args + shift_options

Note that the phase shift can be enabled by just setting a shift centre value, 
for example
    
    shift_options = '-shift 13:15:32.689094 +062.07.37.563039 '

## Loop over robust or measurement sets and tapers. 
The last block of the code allows to run imaging over a list of images and/or a list of robust values and/or multiple tapers. 
```python
image_list = ['UGC5101_eM_0935_6121_avg_8s_RR_LL']
robusts = [0.0,0.5
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
                image_statistics['uvtaper'] = uvtaper
                df = pd.DataFrame.from_dict(image_statistics, orient='index').T
                df.to_csv(root_dir_sys+image_statistics['imagename'].replace('.fits','_data.csv'), header=True,
                          index=False)
            else:
                pass
```
## Know issues
Some issues related with wsclean that currently can be a limitation is imaging visibilities 
containing non-homogeneous polarizations (e.g. RL, LR, or all four together RR,LL, RL,LR), see 
(https://gitlab.com/aroffringa/wsclean/-/merge_requests/478)
So, you may want to split your visibility to have only RR,LL polarizations. In that case, deconvolution may work.

## Example usage
To excecute the code, it is recommended to use it inside casa, for example, 
on the same directory that the script is, do :
    
    $ casa
    CASA <1>: exec(open('./imaging_with_wsclean_v3.py').read())

