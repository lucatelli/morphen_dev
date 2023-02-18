from petrofit.photometry import make_radius_list
from petrofit.petrosian import Petrosian
from petrofit.photometry import source_photometry
from petrofit.segmentation import make_catalog, plot_segments
from petrofit.segmentation import plot_segment_residual
from petrofit.photometry import order_cat


def petro_cat(data_2D, fwhm=24, npixels=None, kernel_size=15,
              nlevels=30, contrast=0.001,
              sigma_level=20, vmin=5,
              deblend=True, plot=True):
    """
    Use PetroFit class to create catalogues.
    """
    cat, segm, segm_deblend = make_catalog(
        image=data_2D,
        threshold=20.0 * mad_std(data_2D),
        deblend=False,
        kernel_size=kernel_size,
        fwhm=fwhm,
        npixels=npixels,
        plot=plot, vmax=data_2D.max(), vmin=vmin * mad_std(data_2D)
    )

    sorted_idx_list = order_cat(cat, key='area', reverse=True)
    #     idx = sorted_idx_list[main_feature_index]  # index 0 is largest
    #     source = cat[idx]  # get source from the catalog
    return (cat, sorted_idx_list)


def petro_params(source, data_2D, segm, i='1', petro_properties={},
                 rlast=None, sigma=3, vmin=3, bkg_sub=False, plot=True):
    if rlast is None:
        rlast = int(2 * np.sqrt((np.sum(mask_source) / np.pi)))
    else:
        rlast = rlast

    r_list = make_radius_list(
        max_pix=rlast,  # Max pixel to go up to
        n=int(rlast)  # the number of radii to produce
    )
    cutout_size = 2 * max(r_list)
    flux_arr, area_arr, error_arr = source_photometry(source, data_2D, segm,
                                                      r_list, cutout_size=cutout_size,
                                                      bkg_sub=bkg_sub, sigma=3, sigma_type='clip',
                                                      plot=plot, vmax=0.3 * data_2D.max(),
                                                      vmin=vmin * mad_std(data_2D)
                                                      )
    #     fast_plot2(mask_source * data_2D)
    p = Petrosian(r_list, area_arr, flux_arr)
    R50 = p.r_half_light
    Snu = p.total_flux
    Rp = p.r_petrosian
    Rpidx = int(2 * Rp)
    petro_properties['c' + i + '_R50'] = R50
    petro_properties['c' + i + '_Snu'] = Snu
    petro_properties['c' + i + '_Rp'] = Rp
    petro_properties['c' + i + '_Rpidx'] = Rpidx
    petro_properties['c' + i + '_rlast'] = rlast
    plt.figure()
    p.plot(plot_r=True)
    #     print('    R50 =', R50)
    #     print('     Rp =', Rp)
    return (petro_properties)


def source_props(data_2D, source_props={}):
    '''
    From a 2D image array, perform simple source extraction, and calculate basic petrosian
    properties.
    '''
    cat, sorted_idx_list = petro_cat(data_2D, fwhm=24, npixels=None, kernel_size=15,
                                     nlevels=30, contrast=0.001,
                                     sigma_level=20, vmin=5,
                                     deblend=True, plot=True)
    #     i = 0
    for i in range(len(sorted_idx_list)):
        ii = str(i + 1)
        seg_image = cat[sorted_idx_list[i]]._segment_img.data
        source = cat[sorted_idx_list[i]]
        source_props['c' + ii + '_PA'] = source.orientation.value
        source_props['c' + ii + '_q'] = 1 - source.ellipticity.value
        source_props['c' + ii + '_area'] = source.area.value
        source_props['c' + ii + '_Re'] = source.equivalent_radius.value
        source_props['c' + ii + '_x0c'] = source.xcentroid
        source_props['c' + ii + '_y0c'] = source.ycentroid
        source_props['c' + ii + '_label'] = source.label

        label_source = source.label
        # plt.imshow(seg_image==label_source)
        mask_source = seg_image == label_source

        source_props = petro_params(source=source, data_2D=data_2D, segm=segm, i=ii, petro_properties=source_props,
                                    rlast=None, sigma=3, vmin=3, bkg_sub=False, plot=False)

        #         print(Rp_props['rlast'],2*Rp_props['Rp'])
        if source_props['c' + ii + '_rlast'] < 2 * source_props['c' + ii + '_Rp']:
            Rlast_new = 2 * source_props['c' + ii + '_Rp'] + 3
            source_props = petro_params(source=source, data_2D=data_2D, segm=segm, i=ii, petro_properties=source_props,
                                        rlast=Rlast_new, sigma=3, vmin=3, bkg_sub=False, plot=False)
        r, ir = get_profile(data_2D * mask_source, binsize=1.0)
        I50 = ir[int(source_props['c' + ii + '_R50'])]
        source_props['c' + ii + '_I50'] = I50

    source_props['ncomps'] = len(sorted_idx_list)
    return (source_props)

imagename = 'image.fits'
data_2D = pf.getdata(imagename)
Rp_props = source_props(data_2D)

# r,ir = get_profile(ctn(crop_image)*mask_source,binsize=1.0)
# plt.figure()
# plt.plot(r[0:int(2*Rp)],ir[0:int(2*Rp)])
# # plt.semilogy()