"""
 ____  _       _   _   _
|  _ \| | ___ | |_| |_(_)_ __   __ _
| |_) | |/ _ \| __| __| | '_ \ / _` |
|  __/| | (_) | |_| |_| | | | | (_| |
|_|   |_|\___/ \__|\__|_|_| |_|\__, |
                               |___/
Plotting Functions
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from astropy import units as u
import astropy.io.fits as pf
from astropy.stats import mad_std
from casatools import image as IA
from astropy.visualization import simple_norm
import scipy.ndimage as nd

from image_operations import *
from read_data import *

class CustomFormatter(mticker.ScalarFormatter):
    def __init__(self, factor=1, **kwargs):
        self.factor = factor
        mticker.ScalarFormatter.__init__(self, **kwargs)

    def __call__(self, x, pos=None):
        x = x * self.factor
        if x == 0:
            return "0.00"
        return "{:.2f}".format(x)


def make_scalebar(ax, left_side, length, color='w', linestyle='-', label='',
                  fontsize=12, text_offset=0.1*u.arcsec):
    axlims = ax.axis()
    lines = ax.plot(u.Quantity([left_side.ra, left_side.ra-length]),
                    u.Quantity([left_side.dec]*2),
                    color=color, linestyle=linestyle, marker=None,
                    transform=ax.get_transform('fk5'),
                   )
    txt = ax.text((left_side.ra-length/2).to(u.deg).value,
                  (left_side.dec+text_offset).to(u.deg).value,
                  label,
                  verticalalignment='bottom',
                  horizontalalignment='center',
                  transform=ax.get_transform('icrs'),
                  color=color,
                  fontsize=fontsize,
                 )
    ax.axis(axlims)
    return lines,txt

def plot_flux_petro(imagename, flux_arr, r_list,
                    savefig=True, show_figure=True,
                    add_save_name = ''):
    plt.figure(figsize=(4, 3))
    cell_size = get_cell_size(imagename)
    plt.plot(r_list * cell_size, 1000 * flux_arr / beam_area2(imagename),
             color='black', lw='3')
    idx_lim = int(np.where(flux_arr / np.max(flux_arr) > 0.95)[0][0] * 1.5)
    plt.grid()
    plt.xlabel('Aperture Radii [arcsec]')
    plt.ylabel(r'$S_\nu$ [mJy]')
    plt.title('Curve of Growth')
    try:
        plt.xlim(0, r_list[idx_lim] * cell_size)
    except:
        plt.xlim(0, r_list[-1] * cell_size)
    if savefig is True:
        plt.savefig(
            imagename.replace('.fits', '_flux_aperture_'+add_save_name+'.jpg'),
            dpi=300, bbox_inches='tight')
    if show_figure == True:
        plt.show()
    else:
        plt.close()



def make_cl(image):
    std = mad_std(image)
    levels = np.geomspace(image.max() * 5, 7 * std, 10)
    return (levels[::-1])

def eimshow(imagename, crop=False, box_size=128, center=None, with_wcs=True,
            vmax_factor=0.5, neg_levels=np.asarray([-3]), CM='magma_r',cmap_cont='terrain',
            rms=None, max_factor=None,plot_title=None,apply_mask=False,
            add_contours=True,extent=None,
            vmin_factor=3, plot_colorbar=True, figsize=(5, 5), aspect=1,
            ax=None):
    """
    Fast plotting of an astronomical image with/or without a wcs header.
    neg_levels=np.asarray([-3])
    imagename:
        str or 2d array.
        If str (the image file name), it will attempt to read the wcs and plot the coordinates axes.

        If 2darray, will plot the data with generic axes.

        support functions:
            ctn() -> casa to numpy: A function designed mainly to read CASA fits images,
                     but can be used to open any fits images.

                     However, it does not read header/wcs.
                     Note: THis function only works inside CASA environment.

    """
    try:
        import cmasher as cmr
        print('Imported cmasher for density maps.'
              'If you would like to use, examples:'
              'CM = cmr.ember,'
              'CM = cmr.flamingo,'
              'CM = cmr.gothic'
              'CM = cmr.lavender')
        """
        ... lilac,rainforest,sepia,sunburst,torch.
        Diverging: copper,emergency,fusion,infinity,pride'
        """
    except:
        print('Error importing cmasher. If you want '
              'to use its colormaps, install it. '
              'Then you can use for example:'
              'CM = cmr.flamingo')
    if ax == None:
        fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(1,1,1)
    #     try:
    if isinstance(imagename, str) == True:
        if with_wcs == True:
            hdu = pf.open(imagename)
            #         hdu=pf.open(img)
            ww = WCS(hdu[0].header, naxis=2)
            try:
                if len(np.shape(hdu[0].data) == 2):
                    g = hdu[0].data[0][0]
                else:
                    g = hdu[0].data
            except:
                g = ctn(imagename)
        if with_wcs == False:
            g = ctn(imagename)
            # print('1', g)

        if crop == True:
            xin, xen, yin, yen = do_cutout(imagename, box_size=box_size,
                                           center=center, return_='box')
            g = g[yin:yen,xin:xen]
            # print('2', g)
            crop = False

    else:
        g = imagename
        # print('3', g)

    if crop == True:
        xin, xen, yin, yen = do_cutout(imagename, box_size=box_size,
                                       center=center, return_='box')
        g = g[yin:yen,xin:xen]
        # print('4', g)
    #         max_x, max_y = np.where(g == g.max())
    #         xin = max_x[0] - box_size
    #         xen = max_x[0] + box_size
    #         yin = max_y[0] - box_size
    #         yen = max_y[0] + box_size
    #         g = g[xin:xen, yin:yen]
    if rms is not None:
        std = rms
    else:
        if mad_std(g) == 0:
            """
            About std:
                mad_std is much more robust than np.std.
                But:
                    if mad_std is applied to a masked image, with zero
                    values outside the emission region, mad_std(image) is zero!
                    So, in that case, np.std is a good option.
            """
            # print('5', g)
            std = g.std()
        else:
            std = mad_std(g)

    if ax == None:
        if with_wcs == True and isinstance(imagename, str) == True:
            ax = fig.add_subplot(projection=ww.celestial)
            ax.set_xlabel('RA')
            ax.set_ylabel('DEC')
        else:
            ax = fig.add_subplot()
            ax.set_xlabel('x pix')
            ax.set_ylabel('y pix')

    if apply_mask == True:
        _, mask_d = mask_dilation(imagename, cell_size=None,
                                  sigma=6, rms=None,
                                  dilation_size=None,
                                  iterations=3, dilation_type='disk',
                                  PLOT=False, show_figure=False)
        g = g * mask_d

    vmin = vmin_factor * std

    #     print(g)
    if max_factor is not None:
        vmax = vmax_factor * max_factor
    else:
        vmax = vmax_factor * g.max()

    norm = simple_norm(g, stretch='sqrt', asinh_a=0.02, min_cut=vmin,
                       max_cut=vmax)

    im_plot = ax.imshow((g), cmap=CM, origin='lower', alpha=1.0,extent=extent,
                        norm=norm,
                        aspect=aspect)  # ,vmax=vmax, vmin=vmin)#norm=norm
    if plot_title is not None:
        ax.set_title(plot_title)

    levels_g = np.geomspace(2.0 * g.max(), vmin_factor * std, 9)

    #     x = np.geomspace(1.5*mad_std(g),10*mad_std(g),4)
    levels_black = np.geomspace(vmin_factor * std + 0.00001, 2.5 * g.max(), 7)
    #     xneg = np.geomspace(5*mad_std(g),vmin_factor*mad_std(g),2)
    #     y = -xneg[::-1]
    #     levels_black = np.append(y,x)
    levels_neg = neg_levels * std
    #     levels_white = np.geomspace(g.max(),10*mad_std(g),7)
    levels_white = np.geomspace(g.max(), 0.1 * g.max(), 5)

    #     cg.show_contour(data, colors='black',levels=levels_black,linestyle='.-',linewidths=0.2,alpha=1.0)
    #     cg.show_contour(data, colors='#009E73',levels=levels_white[::-1],linewidths=0.2,alpha=1.0)
    # try:
    #     ax.contour(g, levels=levels_black, colors='grey', linewidths=0.2,
    #                alpha=1.0)  # cmap='Reds', linewidths=0.75)
    # except:
    #     pass
    if add_contours:
        try:
            ax.contour(g, levels=levels_g[::-1], cmap=cmap_cont, linewidths=1.0,extent=extent,
                       alpha=1.0)  # cmap='Reds', linewidths=0.75)
        except:
            pass
        try:
            ax.contour(g, levels=levels_neg[::-1], colors='k', linewidths=1.0,extent=extent,
                       alpha=1.0)
        except:
            pass
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    if plot_colorbar:
        try:
            cb = plt.colorbar(mappable=plt.gca().images[0],
                              cax=fig.add_axes([0.91, 0.08, 0.05, 0.84]))
            cb.set_label(r"Flux Density [Jy/Beam]")
        except:
            pass
    # if ax==None:
    #     if plot_colorbar==True:
    #         # cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.08,0.05,0.82]))
    #         cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.08,0.05,0.32]))
    #         # cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))
    #         cb.set_label(r"Flux [Jy/Beam]")
    return (ax)




def plot_slices_fig(data_2D, show_figure=True, label='',color=None,FIG=None,linestyle='--.'):
    plot_slice = np.arange(0, data_2D.shape[0])

    if FIG is None:
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
    else:
        fig,ax1,ax2 = FIG
    ax1.plot(plot_slice, np.mean(data_2D, axis=0), linestyle=linestyle, color=color, ms=14,
             label=label)
    ax1.legend(fontsize=11)
    ax1.grid()
    ax1.set_ylabel('mean $x$ direction')
    ax1.set_xlim(data_2D.shape[0] / 2 - 0.25 * data_2D.shape[0],
                 data_2D.shape[0] / 2 + 0.25 * data_2D.shape[0])

    ax2.plot(plot_slice, np.mean(data_2D, axis=1), linestyle=linestyle, color=color, ms=14,
             label=label)
    ax2.set_xlabel('Image Slice [px]')
    ax2.set_ylabel('mean $y$ direction')
    ax2.set_xlim(data_2D.shape[0] / 2 - 0.25 * data_2D.shape[0],
                 data_2D.shape[0] / 2 + 0.25 * data_2D.shape[0])
    ax2.grid()
    # plt.semilogx()
    # plt.xlim(300,600)
    # if image_results_conv is not None:
    #     plt.savefig(
    #         image_results_conv.replace('.fits', 'result_lmfit_slices.pdf'),
    #         dpi=300, bbox_inches='tight')
    #     if show_figure == True:
    #         plt.show()
    #     else:
    #         plt.close()
    return(fig,ax1,ax2)

def plot_slices(data_2D, residual_2D, model_dict, image_results_conv=None,
                Rp_props=None, show_figure=True):
    plot_slice = np.arange(0, data_2D.shape[0])
    if Rp_props is not None:
        plotlim = Rp_props['c' + str(1) + '_rlast']
        # plotlim = 0
        # for i in range(Rp_props['ncomps']):
        #     plotlim = plotlim + Rp_props['c' + str(i + 1) + '_rlast']

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(plot_slice, np.mean(data_2D, axis=0), '--.', color='purple', ms=14,
             label='DATA')
    ax1.plot(plot_slice, np.mean(model_dict['model_total_conv'], axis=0), '.-',
             color='limegreen', linewidth=4, label='MODEL')
    ax1.plot(plot_slice, np.mean(model_dict['best_residual_conv'], axis=0), '.-',
             color='black', linewidth=4, label='RESIDUAL')
    try:
        ax1.plot(plot_slice, np.mean(residual_2D, axis=0), '.-', color='grey',
                linewidth=4, label='MAP RESIDUAL')
    except:
        pass
    #     ax1.set_xlabel('$x$-slice')
    #     ax1.set_xaxis('off')
    #     ax1.set_xticks([])
    ax1.legend(fontsize=11)
    ax1.grid()
    ax1.set_ylabel('mean $x$ direction')
    if Rp_props is not None:
        ax1.set_xlim(Rp_props['c1_x0c'] - plotlim, Rp_props['c1_x0c'] + plotlim)
    #     ax1.set_title('asd')
    # plt.plot(np.mean(shuffled_image,axis=0),color='red')

    ax2.plot(plot_slice, np.mean(data_2D, axis=1), '--.', color='purple', ms=14,
             label='DATA')
    ax2.plot(plot_slice, np.mean(model_dict['model_total_conv'], axis=1), '.-',
             color='limegreen', linewidth=4, label='MODEL')
    ax2.plot(plot_slice, np.mean(model_dict['best_residual_conv'], axis=1), '.-',
             color='black', linewidth=4, label='RESIDUAL')
    try:
        ax2.plot(plot_slice, np.mean(residual_2D, axis=1), '.-', color='grey',
                linewidth=4, label='MAP RESIDUAL')
    except:
        pass
    ax2.set_xlabel('Image Slice [px]')
    ax2.set_ylabel('mean $y$ direction')
    if Rp_props is not None:
        ax2.set_xlim(Rp_props['c1_y0c'] - plotlim, Rp_props['c1_y0c'] + plotlim)
    ax2.grid()
    # plt.semilogx()
    # plt.xlim(300,600)
    if image_results_conv is not None:
        plt.savefig(
            image_results_conv.replace('.fits', 'result_lmfit_slices.pdf'),
            dpi=300, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()


def plot_fit_results(imagename, model_dict, image_results_conv,
                     sources_photometies,vmax_factor=0.1,data_2D_=None,
                     vmin_factor=3, show_figure=True,crop=False,box_size=100):
    if data_2D_ is not None:
        data_2D = data_2D_
    else:
        data_2D = ctn(imagename)

    fast_plot3(data_2D, modelname=model_dict['model_total_conv'],
               residualname=model_dict['best_residual_conv'],
               reference_image=imagename,
               NAME=image_results_conv[-2].replace('.fits',
                                                   'result_image_conv.pdf'),
               crop=crop, vmin_factor=vmin_factor,
               box_size=box_size)



    ncomponents = sources_photometies['ncomps']
    if sources_photometies is not None:
        plotlim =  4.0 * sources_photometies['c'+str(int(ncomponents))+'_rlast']
        # plotlim = 0
        # for i in range(ncomponents):
        #     plotlim = plotlim + sources_photometies['c' + str(i + 1) + '_rlast']

    model_name = image_results_conv[-2]
    residual_name = image_results_conv[-1]
    cell_size = get_cell_size(imagename)
    profile_data = {}
    # center = get_peak_pos(imagename)
    center = nd.maximum_position(data_2D)[::-1]
    for i in range(ncomponents):
        component_name = image_results_conv[
            i]  # crop_image.replace('.fits','')+"_"+str(ncomponents)+"C_model_component_"+str(i+1)+special_name+'_IMFIT_opt.fits'
        Ir_r = get_profile(component_name,
                           center=center)
        profile_data['r' + str(i + 1)], profile_data['Ir' + str(i + 1)], \
        profile_data['c' + str(i + 1) + '_name'] = Ir_r[0], Ir_r[
            1], component_name

    r, ir = get_profile(data_2D, center=center)
    rmodel, irmodel = get_profile(model_name, center=center)
    rre, irre = get_profile(residual_name, center=center)

    # plt.plot(radiis[0],profiles[0])
    # plt.plot(radiis[1],profiles[1])
    # plt.plot(radiis[2],np.log(profiles[2]))
    # colors = ['black','purple','gray','red']
    colors = ['red', 'blue', 'teal', 'brown', 'cyan','orange','forestgreen','pink']
    plt.figure(figsize=(5, 5))
    plt.plot(r * cell_size, abs(ir), '--.', ms=10, color='purple', alpha=1.0,
             label='DATA')
    for i in range(ncomponents):
        #     try:
        #         plt.plot(profile_data['r'+str(i+1)],abs(profile_data['Ir'+str(i+1)])[0:r.shape[0]],'--',label='comp'+str(i+1),color=colors[i])
        plt.plot(profile_data['r' + str(i + 1)] * cell_size,
                 abs(profile_data['Ir' + str(i + 1)]), '--',
                 label='COMP_' + str(i + 1), color=colors[i])
    #     except:
    #         pass

    plt.plot(r * cell_size, abs(irre), '.-', label='RESIDUAL', color='black')
    plt.plot(r * cell_size, abs(irmodel), '--', color='limegreen', label='MODEL',
             linewidth=4)
    plt.semilogy()
    plt.xlabel(r'$r$ [arcsec]')
    plt.ylabel(r'$I(r)$ [Jy/beam]')
    plt.legend(fontsize=11)
    plt.ylim(1e-7, -0.05 * np.log(ir[0]))
    # plt.xlim(0,3.0)
    plt.grid()
    if sources_photometies is not None:
        plt.xlim(0, plotlim * cell_size)
        idRp_main = int(sources_photometies['c1_Rp'])
        plt.axvline(r[idRp_main] * cell_size)
    plt.savefig(image_results_conv[-2].replace('.fits', 'result_lmfit_IR.pdf'),
                dpi=300, bbox_inches='tight')
    if show_figure == True:
        plt.show()
        # return(plt)
    else:
        plt.close()



# plt.savefig(config_file.replace('params_imfit.csv','result_lmfit_py_IR.pdf'),dpi=300, bbox_inches='tight')

def total_flux(data2D,image,mask=None,BA=None,
               sigma=6,iterations=3,dilation_size=7,PLOT=False,
               silent=True):

    if BA is None:
        BA = beam_area2(image)
    else:
        BA = BA
    if mask is None:
        _,mask = mask_dilation(data2D,sigma=sigma,iterations=iterations,
                               dilation_size=dilation_size,PLOT=PLOT)
    else:
        mask = mask
#     tf = np.sum((data2D> sigma*mad_std(data2D))*data2D)/beam_area2(image)
    blank_sum = np.sum(data2D)/BA
    sum3S = np.sum(data2D*(data2D> 3.0*mad_std(data2D)))/BA
    summask = np.sum(data2D*mask)/BA
    if silent==False:
        print('Blank Sum   = ',blank_sum)
        print('Sum  3sigma = ',sum3S)
        print('Sum mask    = ',summask)
    return(summask)

def total_flux_faster(data2D,mask):
#     tf = np.sum((data2D> sigma*mad_std(data2D))*data2D)/beam_area2(image)
    summask = np.sum(data2D*mask)
    return(summask)


def plot_decomp_results(imagename,compact,extended_model,data_2D_=None,
                        vmax_factor=0.5,vmin_factor=3,rms=None,
                        figsize=(13,13),nfunctions=None,
                        special_name=''):

    decomp_results = {}
    if rms == None:
        rms = mad_std(ctn(imagename))
    else:
        rms = rms

    max_factor =  ctn(imagename).max()
#     compact = model_dict['model_c1_conv']
    if data_2D_ is not None:
        data_2D = data_2D_
    else:
        data_2D = ctn(imagename)

    # rms_std_data = mad_std(data_2D)
    extended = data_2D  - compact
    if nfunctions == 1:
        residual_modeling = data_2D - (compact)
    else:
        residual_modeling = data_2D - (compact + extended_model)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(3, 3, 1)
    # ax.yaxis.set_ticks([])
    ax = eimshow(imagename,ax=ax,rms=rms,plot_title='Total Emission',
                 vmax_factor=vmax_factor,vmin_factor=vmin_factor)
    # cb = plt.colorbar(mappable=plt.gca().images[0],
    #                           cax=fig.add_axes([0.9, 0.65, 0.02, 0.2]))
    # ax.yaxis.set_ticks([])
    ax.axis('off')
    ax = fig.add_subplot(3, 3, 2)
    # ax.yaxis.set_ticks([])
    ax = eimshow(compact,ax=ax,rms=rms,
                 plot_title='Compact Emission',max_factor=data_2D.max(),
                 vmax_factor=vmax_factor,vmin_factor=vmin_factor)
    # ax.yaxis.set_ticks([])
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 3)
    # ax.yaxis.set_ticks([])
    ax = eimshow(extended,ax=ax,rms=rms,max_factor=data_2D.max(),
                 plot_title='Diffuse Emission',vmax_factor=vmax_factor,
                 vmin_factor=vmin_factor)
    # ax.yaxis.set_ticks([])
    # cb = plt.colorbar(mappable=plt.gca().images[0],
    #                           cax=fig.add_axes([0.00, 0.65, 0.02, 0.2]))
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 4)
    slice_ext = np.sqrt(np.mean(extended,axis=0)**2.0 + np.mean(extended,axis=1)**2.0)
    if nfunctions == 1:
        slice_ext_model = np.sqrt(
            np.mean(residual_modeling, axis=0) ** 2.0 + np.mean(residual_modeling,
                                                             axis=1) ** 2.0)
    else:
        slice_ext_model = np.sqrt(
            np.mean(extended_model, axis=0) ** 2.0 + np.mean(extended_model,
                                                             axis=1) ** 2.0)
    slice_data = np.sqrt(np.mean(data_2D,axis=0)**2.0 + np.mean(data_2D,axis=1)**2.0)
    ax.plot(slice_ext,label='COMPACT SUB')
    ax.plot(slice_data,label='DATA')
    ax.plot(slice_ext_model, label='EXTENDED MODEL')
    plt.legend()
    xlimit = [data_2D.shape[0] / 2 - 0.15 * data_2D.shape[0],
              data_2D.shape[0] / 2 + 0.15 * data_2D.shape[0]]
    ax.set_xlim(xlimit[0],xlimit[1])
    # ax.semilogx()

    ax = fig.add_subplot(3, 3, 5)
    ax.axis('off')

    try:
        omaj, omin, _, _, _ = beam_shape(imagename)
        dilation_size = int(
            np.sqrt(omaj * omin) / (2 * get_cell_size(imagename)))
    except:
        dilation_size = 10

    _, mask_model_rms_self_compact = mask_dilation(compact,
                                           sigma=1, dilation_size=dilation_size,
                                           iterations=2,PLOT=False)
    _, mask_data = mask_dilation(data_2D,
                                           sigma=6, dilation_size=dilation_size,
                                           iterations=2,PLOT=False)
    _, mask_model_rms_self_extended = mask_dilation(extended,
                                           sigma=6, dilation_size=dilation_size,
                                           iterations=2,PLOT=False)

    _, mask_model_rms_image_compact = mask_dilation(compact,
                                            rms=rms,
                                            sigma=1, dilation_size=dilation_size,
                                            iterations=2,PLOT=False)
    _, mask_model_rms_image_extended = mask_dilation(extended,
                                            rms=rms,
                                            sigma=6, dilation_size=dilation_size,
                                            iterations=2,PLOT=False)
    if nfunctions == 1:
        _, mask_model_rms_image_extended_model = mask_dilation(residual_modeling,
                                                rms=rms,
                                                sigma=1, dilation_size=dilation_size,
                                                iterations=2,PLOT=False)
    else:
        _, mask_model_rms_image_extended_model = mask_dilation(extended_model,
                                                rms=rms,
                                                sigma=1, dilation_size=dilation_size,
                                                iterations=2,PLOT=False)

    try:
        beam_area_px = beam_area2(imagename)
    except:
        beam_area_px = 1
    print('Flux on compact (self rms) = ',
          1000*np.sum(compact*mask_model_rms_self_compact)/beam_area_px)
    print('Flux on compact (data rms) = ',
          1000 * np.sum(compact * mask_model_rms_image_compact) / beam_area_px)
    flux_density_compact = 1000*np.sum(
        compact*mask_model_rms_image_compact)/beam_area_px
    if nfunctions == 1:
        flux_density_extended_model = 1000 * np.sum(
            residual_modeling * mask_data) / beam_area_px
    else:
        flux_density_extended_model = 1000 * np.sum(
            extended_model * mask_data) / beam_area_px

    flux_density_ext = 1000*total_flux(extended,imagename,BA=beam_area_px,
                                       mask = mask_model_rms_image_extended)
    flux_density_ext2 = 1000*np.sum(
        extended*mask_data)/beam_area_px

    flux_data = 1000*total_flux(data_2D,imagename,BA=beam_area_px,
                                       mask = mask_data)
    flux_density_ext_self_rms = 1000*total_flux(extended,imagename,BA=beam_area_px,
                                       mask = mask_model_rms_self_extended)

    if nfunctions==1:
        flux_res = flux_data - (flux_density_compact)
    else:
        flux_res = flux_data - (
                    flux_density_extended_model + flux_density_compact)

    print('Flux on extended (self rms) = ',flux_density_ext_self_rms)
    print('Flux on extended (data rms) = ',flux_density_ext)
    print('Flux on extended2 (data rms) = ', flux_density_ext2)
    print('Flux on extended model (data rms) = ', flux_density_extended_model)
    print('Flux on data = ', flux_data)
    print('Flux on residual = ', flux_res)

    decomp_results['flux_data'] = flux_data
    decomp_results['flux_density_ext'] = flux_density_ext
    decomp_results['flux_density_ext2'] = flux_density_ext2
    decomp_results['flux_density_extended_model'] = flux_density_extended_model
    decomp_results['flux_density_compact'] = flux_density_compact
    decomp_results['flux_res'] = flux_res


    # print('r_half_light (old vs new) = {:0.2f} vs {:0.2f}'.format(p.r_half_light, p_copy.r_half_light))
    ax.annotate(r"$S_\nu^{\rm comp}=$"+'{:0.2f}'.format(flux_density_compact)+' mJy',
                (0.33, 0.32), xycoords='figure fraction', fontsize=18)
    ax.annotate(r"$S_\nu^{\rm ext}\ \ \ =$"+'{:0.2f}'.format(flux_density_ext2)+' mJy',
                (0.33, 0.29), xycoords='figure fraction', fontsize=18)
    ax.annotate(r"$S_\nu^{\rm ext \ model}\ \ \ =$"+'{:0.2f}'.format(flux_density_extended_model)+' mJy',
                (0.33, 0.26), xycoords='figure fraction', fontsize=18)
    plt.savefig(
        imagename.replace('.fits', '_extended'+special_name+'.jpg'),
        dpi=300,
        bbox_inches='tight')

    save_data = True
    if save_data == True:
        exteded_file_name = imagename.replace('.fits', '') + \
                            special_name + '_extended.fits'
        pf.writeto(exteded_file_name,extended,overwrite=True)
        copy_header(imagename,exteded_file_name)
        compact_file_name = imagename.replace('.fits', '') + \
                            special_name + '_compact.fits'
        pf.writeto(compact_file_name,compact,overwrite=True)
        copy_header(imagename,compact_file_name)

    return(decomp_results)

def plot_fit_results_old(data_2D):
    colors = ['black', 'purple', 'yellow', 'red', 'green']
    axis = 1
    plt.figure(figsize=(6, 6))
    plt.plot(abs(np.mean(data_2D, axis=axis)), '.', label='data')
    plt.plot(abs(np.mean(imfit_residual, axis=axis)), label='residual')
    # plt.plot(abs(np.mean(MODEL_CORE_CONV,axis=1)),'--',label='CORE')
    # plt.plot(abs(np.mean(MODEL_DISK_CONV,axis=1)),'--',label='DISK')
    for i in range(n_comps):
        plt.plot(abs(np.mean(comps_data['c' + str(i + 1)], axis=axis)),
                 label='comp' + str(i + 1), color=colors[i])
    # plt.plot(abs(np.mean(imfit_disk,axis=0)),label='comp2',color='purple')
    # plt.plot(abs(np.mean(imfit_comp3,axis=0)),label='comp3',color='yellow')

    plt.plot(abs(np.mean(imfit_model, axis=axis)), '.k', label='model')

    plt.semilogy()
    # plt.xlim(200,400)
    plt.ylim(0.0000001, data_2D.max() / 10)
    plt.legend()
    plt.xlabel('Pixel Unit')
    plt.ylabel('Mean Averaged Intensity (axis=0)')
    plt.savefig(param_name.replace('_params_imfit.csv', '_slice.pdf'), dpi=300,
                bbox_inches='tight')

    # plt.plot(radiis[0],profiles[0])
    # plt.plot(radiis[1],profiles[1])
    # plt.plot(radiis[2],np.log(profiles[2]))
    colors = ['black', 'purple', 'gray', 'red']
    plt.figure(figsize=(5, 5))
    plt.plot(r, abs(ir), label='data', lw=5, color='red')
    plt.plot(r, abs(irmodel)[0:r.shape[0]], '--', label='model', color='blue',
             lw=5)

    for i in range(n_comps):
        try:
            plt.plot(profile_data['r' + str(i + 1)],
                     abs(profile_data['Ir' + str(i + 1)])[0:r.shape[0]], '--',
                     label='comp' + str(i + 1), color=colors[i])
        except:
            pass

    plt.axhline(values[-1], ls='--', label='FlatSky', color='green',
                animated=True)
    plt.plot(r, abs(irre)[0:r.shape[0]], '.-', label='residual')
    plt.semilogy()
    plt.xlabel(r'$r$ [arcsec]')
    plt.ylabel(r'$I(r)$ [Jy/beam]')
    plt.legend()
    plt.ylim(1e-7, -2.5)
    plt.xlim(0, 2.5)
    # mplcyberpunk.add_glow_effects()
    plt.savefig(param_name.replace('params_imfit.csv', 'result_imfit_IR.pdf'),
                dpi=300, bbox_inches='tight')
    #
    fast_plot3(crop_image, modelname=imfit_model_name,
               residualname=imfit_residual_name, reference_image=crop_image,
               NAME=param_name.replace('params_imfit.csv',
                                       'result_imfit_im_mo_re'), crop=False,
               box_size=512)
    model_no_conv = comps_data['c1'] + comps_data['c2'] + comps_data['c3']
    residual_no_conv = ctn(crop_image) - model_no_conv
    fast_plot3(ctn(crop_image), modelname=model_no_conv,
               residualname=residual_no_conv, reference_image=crop_image,
               NAME=param_name.replace('params_imfit.csv',
                                       'result_imfit_no_conv_im_mo_re'),
               crop=False, box_size=70)
    fast_plot3(ctn(crop_image), modelname=model_dict['model_total_conv'],
               residualname=model_dict['best_residual_conv'],
               reference_image=crop_image,
               NAME=config_file.replace('.csv',
                                        '_img'), crop=False,
               box_size=512)

    fast_plot2(model_dict['best_residual_conv'])
    fast_plot2(model_dict['model_total_conv'])

    ncomponents = 3
    model_name = crop_image.replace('.fits', '') + "_" + str(
        ncomponents) + "C_model" + special_name + '.fits'
    residual_name = crop_image.replace('.fits', '') + "_" + str(
        ncomponents) + "C_residual" + special_name + '.fits'

    profile_data = {}
    for i in range(ncomponents):
        component_name = crop_image.replace('.fits', '') + "_" + str(
            ncomponents) + "C_model_component_" + str(
            i + 1) + special_name + '.fits'
        Ir_r = get_profile(component_name,
                           center=get_peak_pos(crop_image))
        profile_data['r' + str(i + 1)], profile_data['Ir' + str(i + 1)], \
        profile_data['c' + str(i + 1) + '_name'] = Ir_r[0], Ir_r[
            1], component_name
    r, ir = get_profile(crop_image)
    rmodel, irmodel = get_profile(model_name)
    rre, irre = get_profile(residual_name)

    # plt.plot(radiis[0],profiles[0])
    # plt.plot(radiis[1],profiles[1])
    # plt.plot(radiis[2],np.log(profiles[2]))
    colors = ['black', 'purple', 'gray', 'red']
    plt.figure(figsize=(5, 5))
    plt.plot(r, abs(ir), label='data', lw=5, color='red')
    plt.plot(r, abs(irmodel)[0:r.shape[0]], '--', label='model', color='blue')
    for i in range(ncomponents):
        try:
            plt.plot(profile_data['r' + str(i + 1)],
                     abs(profile_data['Ir' + str(i + 1)])[0:r.shape[0]], '--',
                     label='comp' + str(i + 1), color=colors[i])
        except:
            pass

    plt.plot(r, abs(irre)[0:r.shape[0]], '.-', label='residual')
    plt.semilogy()
    plt.xlabel(r'$r$ [arcsec]')
    plt.ylabel(r'$I(r)$ [Jy/beam]')
    plt.legend()
    plt.ylim(1e-7, -2.5)
    plt.xlim(0, 90.0)
    # plt.savefig(param_name.replace('params_imfit.csv','result_imfit_IR.pdf'),dpi=300, bbox_inches='tight')
    # plt.savefig(config_file.replace('imfit.conf','result_lmfit_py_IR.pdf'),dpi=300, bbox_inches='tight')

def plot_interferometric_decomposition(imagename0, imagename,
                                       modelname, residualname,
                                       crop=False, box_size=512,
                                       max_percent_lowlevel=99.0,
                                       max_percent_highlevel=99.9999,
                                       NAME=None, EXT='.pdf',
                                       run_phase = '1st',
                                       vmin_factor=3,vmax_factor=0.1,
                                       SPECIAL_NAME='', show_figure=True):
    """
    Fast plotting of image <> model <> residual images.

    """
    fig = plt.figure(figsize=(16, 16))
    try:
        g = pf.getdata(imagename)
        I1 = pf.getdata(imagename0)
        if len(np.shape(g) == 4):
            g = g[0][0]
            I1 = I1[0][0]
        m = pf.getdata(modelname)
        r = pf.getdata(residualname)
    except:
        I1 = ctn(imagename0)
        g = ctn(imagename)
        m = ctn(modelname)
        r = ctn(residualname)

    if crop == True:
        xin, xen, yin, yen = do_cutout(imagename, box_size=box_size,
                                       center=None, return_='box')
        #         I1 = I1[int(2*xin):int(xen/2),int(2*yin):int(yen/2)]
        I1 = I1[int(xin + box_size / 1.25):int(xen - box_size / 1.25),
             int(yin + box_size / 1.25):int(yen - box_size / 1.25)]
        # g = g[xin:xen,yin:yen]
        # m = m[xin:xen,yin:yen]
        # r = r[xin:xen,yin:yen]

    if mad_std(I1) == 0:
        std0 = I1.std()
    else:
        std0 = mad_std(I1)

    if mad_std(g) == 0:
        std = g.std()
    else:
        std = mad_std(g)

    if mad_std(r) == 0:
        std_r = r.std()
    else:
        std_r = mad_std(r)

    if mad_std(m) == 0:
        std_m = m.std()
    else:
        std_m = mad_std(m)

    #     print(I1)
    vmin0 = 3 * std  # 0.5*g.min()#
    vmax0 = 1.0 * g.max()
    vmin = vmin_factor * std  # 0.5*g.min()#
    vmax = vmax_factor * g.max()
    vmin_r = 0.5 * r.min()  # 1*std_r
    vmax_r = 1.0 * r.max()
    vmin_m = 1 * mad_std(m)  # vmin#0.01*std_m#0.5*m.min()#
    vmax_m = m.max()  # vmax#0.5*m.max()

    levels_I1 = np.geomspace(2*I1.max(), 1.5 * np.std(I1), 7)
    levels_g = np.geomspace(2*g.max(), 3 * std, 7)
    levels_m = np.geomspace(2*m.max(), 20 * std_m, 7)
    levels_r = np.geomspace(2*r.max(), 3 * std_r, 7)
    levels_neg = np.asarray([-3]) * std
    if run_phase == '1st':
        title_labels = [r'$I_1^{\rm mask}$',
                        r'$I_2$',
                        r'$I_{1}^{\rm mask} * \theta_2$',
                        r'$R_{12} = I_2 - I_{1}^{\rm mask} * \theta_2 $'
                        ]

    if run_phase == '2nd':
        title_labels = [r'$R_{12}$',
                        r'$I_3$',
                        r'$I_{1}^{\rm mask} * \theta_3 + R_{12} * \theta_3$',
                        r'$R_{T}$'
                        ]

    if run_phase == 'compact':
        title_labels = [r'$R_{12}$',
                        r'$I_3$',
                        r'$I_{1}^{\rm mask} * \theta_3$',
                        r'$I_3 - I_{1}^{\rm mask} * \theta_3$'
                        ]

    # colors = [(0, 0, 0), (1, 1, 1)]
    # cmap_name = 'black_white'
    # import matplotlib.colors as mcolors
    # cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors,
    #                                                N=len(levels_g))
    cm = 'gray'
    #     norm = simple_norm(g,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    norm = visualization.simple_norm(g, stretch='linear',
                                     max_percent=max_percent_lowlevel)
    norm0 = simple_norm(abs(I1), min_cut=0.5 * np.std(I1), max_cut=vmax,
                        stretch='sqrt')  # , max_percent=max_percent_highlevel)
    norm2 = simple_norm(abs(g), min_cut=vmin, max_cut=vmax,
                        stretch='asinh',asinh_a=0.02)  # , max_percent=max_percent_highlevel)
    CM = 'magma_r'
    ax = fig.add_subplot(1, 4, 1)

    #     im = ax.imshow(I1, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(I1, cmap='magma_r', origin='lower', alpha=1.0, norm=norm0)

    ax.set_title(title_labels[0])

    ax.contour(I1, levels=levels_I1[::-1], colors=cm,
               linewidths=0.8, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    ax = fig.add_subplot(1, 4, 2)
    #     im = ax.imshow(g, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(g, cmap='magma_r', origin='lower', alpha=1.0, norm=norm2)

    ax.set_title(title_labels[1])

    ax.contour(g, levels=levels_g[::-1], colors=cm,
               linewidths=0.8, alpha=1.0)  # cmap='Reds', linewidths=0.75)

    # cb = plt.colorbar(mappable=plt.gca().images[0],
    #                   cax=fig.add_axes([-0.0, 0.40, 0.02,0.19]))  # ,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    cb = plt.colorbar(mappable=plt.gca().images[0],
                      cax=fig.add_axes([0.07, 0.40, 0.02,0.19]),
                      orientation='vertical',shrink=1, aspect='auto',
                      pad=1, fraction=1.0,
                      drawedges=False, ticklocation='left')
    cb.formatter = CustomFormatter(factor=1000, useMathText=True)
    cb.update_ticks()
#     print('++++++++++++++++++++++')
#     print(plt.gca().images[0])
    cb.set_label(r'Flux [mJy/beam]', labelpad=1)
    cb.ax.xaxis.set_tick_params(pad=1)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(1)
    # cb.dividers.set_color('none')
    ax.set_yticks([])
    ax = plt.subplot(1, 4, 3)

    #     im_plot = ax.imshow(m, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax_m, vmin=vmin_m)#norm=norm
    im_plot = ax.imshow(m, cmap='magma_r', origin='lower', alpha=1.0, norm=norm2)
    ax.set_title(title_labels[2])
    ax.contour(m, levels=levels_g[::-1], colors=cm,
               linewidths=0.8, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.08,0.3,0.02,0.4]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    ax.set_yticks([])
    ax = plt.subplot(1, 4, 4)
    norm_re = simple_norm(r, min_cut=vmin, max_cut=vmax, stretch='sqrt')  # , max_percent=max_percent_highlevel)
    #     norm = simple_norm(r,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    #     ax.imshow(r,origin='lower',cmap='magma_r',alpha=1.0,vmax=vmax_r, vmin=vmin)#norm=norm
    ax.imshow(r, origin='lower', cmap='magma_r', alpha=1.0, norm=norm2)
    #     ax.imshow(r, cmap='magma_r',norm=norm,alpha=0.3,origin='lower')

    ax.contour(r, levels=levels_r[::-1], colors=cm,#colors='grey',
               linewidths=0.8, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    ax.contour(r, levels=levels_neg[::-1], colors='k', linewidths=1.0,
               alpha=1.0)

    ax.set_yticks([])
    ax.set_title(title_labels[3])
    #     cb1=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.40,0.02,0.19]))
    #     cb1.set_label(r'Flux [Jy/beam]',labelpad=1)
    #     cb1.ax.xaxis.set_tick_params(pad=1)
    #     cb1.ax.tick_params(labelsize=12)
    #     cb1.outline.set_linewidth(1)
    if NAME is not None:
        plt.savefig(NAME.replace('.fits', '') + SPECIAL_NAME + EXT, dpi=300,
                    bbox_inches='tight')
        plt.savefig(NAME.replace('.fits', '') + SPECIAL_NAME + '.jpg', dpi=300,
                    bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()

def plot_image_model_res(imagename, modelname, residualname, reference_image, crop=False,
               box_size=512, NAME=None, CM='magma_r',
               vmin_factor=3.0,vmax_factor=0.1,
               max_percent_lowlevel=99.0, max_percent_highlevel=99.9999,
               ext='.pdf', show_figure=True):
    """
    Fast plotting of image <> model <> residual images.

    """
    fig = plt.figure(figsize=(12, 12))
    try:
        try:
            g = pf.getdata(imagename)
            # if len(np.shape(g)==4):
            #     g = g[0][0]
            m = pf.getdata(modelname)
            r = pf.getdata(residualname)
        except:
            g = imagename
            m = modelname
            r = residualname
    except:
        g = ctn(imagename)
        m = ctn(modelname)
        r = ctn(residualname)

    if crop == True:
        xin, xen, yin, yen = do_cutout(reference_image, box_size=box_size,
                                       center=None, return_='box')
        #         I1 = I1[int(2*xin):int(xen/2),int(2*yin):int(yen/2)]
        g = g[xin:xen, yin:yen]
        m = m[xin:xen, yin:yen]
        r = r[xin:xen, yin:yen]

    if mad_std(g) == 0:
        std = g.std()
    else:
        std = mad_std(g)

    if mad_std(r) == 0:
        std_r = r.std()
    else:
        std_r = mad_std(r)

    if mad_std(m) == 0:
        std_m = m.std()
    else:
        std_m = mad_std(m)

    #     print(I1)
    vmin = vmin_factor * std  # 0.5*g.min()#
    vmax = vmax_factor * g.max()
    vmin_r = vmin  # 1.0*r.min()#1*std_r
    vmax_r = vmax #1.0 * r.max()
    vmin_m = vmin  # 1*mad_std(m)#vmin#0.01*std_m#0.5*m.min()#
    vmax_m = vmax  # 0.5*m.max()#vmax#0.5*m.max()

    levels_g = np.geomspace(g.max(), 3 * std, 7)
    levels_m = np.geomspace(m.max(), 10 * std_m, 7)
    levels_r = np.geomspace(r.max(), 3 * std_r, 7)

    #     norm = simple_norm(g,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    norm = visualization.simple_norm(g, stretch='linear',
                                     max_percent=max_percent_lowlevel)
    norm2 = simple_norm(abs(g), min_cut=vmin, max_cut=vmax,
                        stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)

    ax = fig.add_subplot(2, 3, 1)

    #     im = ax.imshow(g, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(g, cmap=CM, origin='lower', alpha=1.0, norm=norm2)

    """
    cb = plt.colorbar(mappable=plt.gca().images[0],
                      cax=fig.add_axes([0.07, 0.40, 0.02,0.19]),
                      orientation='vertical',shrink=1, aspect='auto',
                      pad=1, fraction=1.0,
                      drawedges=False, ticklocation='left')
    cb.formatter = CustomFormatter(factor=1000, useMathText=True)
    cb.update_ticks()
    cb.set_label(r'Flux [mJy/beam]', labelpad=1)
    """
    ax.set_title(r'Image')

    ax.contour(g, levels=levels_g[::-1], colors='#009E73', linewidths=0.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    cb = plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes(
        [-0.0, 0.40, 0.02,
         0.19]))  # ,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    cb.formatter = CustomFormatter(factor=1000, useMathText=True)
    cb.update_ticks()
    cb.set_label(r'Flux [mJy/beam]', labelpad=1)
    cb.ax.xaxis.set_tick_params(pad=1)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(1)
    # cb.dividers.set_color('none')

    ax = plt.subplot(2, 3, 2)

    #     im_plot = ax.imshow(m, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax_m, vmin=vmin_m)#norm=norm
    norm_mod = simple_norm(m, min_cut=vmin, max_cut=vmax,
                           stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)

    im_plot = ax.imshow(m, cmap=CM, origin='lower', alpha=1.0,
                        norm=norm_mod)
    ax.set_title(r'Model')
    ax.contour(m, levels=levels_g[::-1], colors='#009E73', linewidths=0.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.08,0.3,0.02,0.4]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    ax = plt.subplot(2, 3, 3)
    norm_re = simple_norm(r, min_cut=vmin, max_cut=vmax,
                          stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)
    #     norm = simple_norm(r,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    #     ax.imshow(r,origin='lower',cmap='magma_r',alpha=1.0,vmax=vmax_r, vmin=vmin)#norm=norm
    ax.imshow(r, origin='lower', cmap=CM, alpha=1.0, norm=norm_re)
    #     ax.imshow(r, cmap='magma_r',norm=norm,alpha=0.3,origin='lower')

    ax.contour(r, levels=levels_r[::-1], colors='#009E73', linewidths=0.2,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)

    levels_neg = np.asarray([-3 * std])

    ax.contour(r, levels=levels_neg[::-1], colors='k', linewidths=1.0,
               alpha=1.0)

    ax.set_yticks([])
    ax.set_title(r'Residual')
    cb1 = plt.colorbar(mappable=plt.gca().images[0],
                       cax=fig.add_axes([0.91, 0.40, 0.02, 0.19]))
    cb1.formatter = CustomFormatter(factor=1000, useMathText=True)
    cb1.update_ticks()
    cb1.set_label(r'Flux [Jy/beam]', labelpad=1)
    cb1.ax.xaxis.set_tick_params(pad=1)
    cb1.ax.tick_params(labelsize=12)
    cb1.outline.set_linewidth(1)
    return(ax,plt,fig)
    # cb1.dividers.set_color('none')
    # if NAME != None:
    #     plt.savefig(NAME + ext, dpi=300, bbox_inches='tight')
    #     if show_figure == True:
    #         plt.show()
    #     else:
    #         plt.close()


def fast_plot3(imagename, modelname, residualname, reference_image, crop=False,
               box_size=512, NAME=None, CM='magma_r',
               vmin_factor=3.0,vmax_factor=0.1,
               max_percent_lowlevel=99.0, max_percent_highlevel=99.9999,
               ext='.pdf', show_figure=True):
    """
    Fast plotting of image <> model <> residual images.

    """
    fig = plt.figure(figsize=(12, 12))
    try:
        try:
            g = pf.getdata(imagename)
            # if len(np.shape(g)==4):
            #     g = g[0][0]
            m = pf.getdata(modelname)
            r = pf.getdata(residualname)
        except:
            g = imagename
            m = modelname
            r = residualname
    except:
        g = ctn(imagename)
        m = ctn(modelname)
        r = ctn(residualname)

    if crop == True:
        xin, xen, yin, yen = do_cutout(reference_image, box_size=box_size,
                                       center=None, return_='box')
        #         I1 = I1[int(2*xin):int(xen/2),int(2*yin):int(yen/2)]
        g = g[xin:xen, yin:yen]
        m = m[xin:xen, yin:yen]
        r = r[xin:xen, yin:yen]

    if mad_std(g) == 0:
        std = g.std()
    else:
        std = mad_std(g)

    if mad_std(r) == 0:
        std_r = r.std()
    else:
        std_r = mad_std(r)

    if mad_std(m) == 0:
        std_m = m.std()
    else:
        std_m = mad_std(m)

    #     print(I1)
    vmin = vmin_factor * std  # 0.5*g.min()#
    vmax = vmax_factor * g.max()
    vmin_r = vmin  # 1.0*r.min()#1*std_r
    vmax_r = vmax #1.0 * r.max()
    vmin_m = vmin  # 1*mad_std(m)#vmin#0.01*std_m#0.5*m.min()#
    vmax_m = vmax  # 0.5*m.max()#vmax#0.5*m.max()

    levels_g = np.geomspace(3*g.max(), 3 * std, 7)
    levels_m = np.geomspace(3*m.max(), 10 * std_m, 7)
    levels_r = np.geomspace(3*r.max(), 3 * std_r, 7)

    #     norm = simple_norm(g,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    norm = visualization.simple_norm(g, stretch='linear',
                                     max_percent=max_percent_lowlevel)
    norm2 = simple_norm(abs(g), min_cut=vmin, max_cut=vmax,
                        stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)

    ax = fig.add_subplot(1, 3, 1)

    #     im = ax.imshow(g, cmap='gray_r',norm=norm,alpha=0.2)

    #     im_plot = ax.imshow(g, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax, vmin=vmin)#norm=norm
    im_plot = ax.imshow(g, cmap=CM, origin='lower', alpha=1.0, norm=norm2)

    ax.set_title(r'Image')

    ax.contour(g, levels=levels_g[::-1], colors='grey', linewidths=1.0,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    cb = plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes(
        [-0.0, 0.40, 0.02,
         0.19]))  # ,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    cb.formatter = CustomFormatter(factor=1000, useMathText=True)
    cb.update_ticks()
    cb.set_label(r'Flux [mJy/beam]', labelpad=1)
    cb.ax.xaxis.set_tick_params(pad=1)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(1)
    # cb.dividers.set_color('none')

    ax = plt.subplot(1, 3, 2)

    #     im_plot = ax.imshow(m, cmap='magma_r',origin='lower',alpha=1.0,vmax=vmax_m, vmin=vmin_m)#norm=norm
    norm_mod = simple_norm(m, min_cut=vmin, max_cut=vmax,
                           stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)

    im_plot = ax.imshow(m, cmap=CM, origin='lower', alpha=1.0,
                        norm=norm_mod)
    ax.set_title(r'Model')
    ax.contour(m, levels=levels_g[::-1], colors='grey', linewidths=1.0,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.08,0.3,0.02,0.4]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')

    ax = plt.subplot(1, 3, 3)
    norm_re = simple_norm(r, min_cut=vmin, max_cut=vmax,
                          stretch='asinh',asinh_a=0.05)  # , max_percent=max_percent_highlevel)
    #     norm = simple_norm(r,stretch='asinh',asinh_a=0.01)#,vmin=vmin,vmax=vmax)
    #     ax.imshow(r,origin='lower',cmap='magma_r',alpha=1.0,vmax=vmax_r, vmin=vmin)#norm=norm
    ax.imshow(r, origin='lower', cmap=CM, alpha=1.0, norm=norm_re)
    #     ax.imshow(r, cmap='magma_r',norm=norm,alpha=0.3,origin='lower')

    ax.contour(r, levels=levels_g[::-1], colors='grey', linewidths=1.0,
               alpha=1.0)  # cmap='Reds', linewidths=0.75)
    levels_neg = np.asarray([-3 * std])

    ax.contour(r, levels=levels_neg[::-1], colors='k', linewidths=1.0,
               alpha=1.0)
    ax.set_yticks([])
    ax.set_title(r'Residual')
    cb1 = plt.colorbar(mappable=plt.gca().images[0],
                       cax=fig.add_axes([0.91, 0.40, 0.02, 0.19]))
    cb1.formatter = CustomFormatter(factor=1000, useMathText=True)
    cb1.update_ticks()
    cb1.set_label(r'Flux [mJy/beam]', labelpad=1)
    cb1.ax.xaxis.set_tick_params(pad=1)
    cb1.ax.tick_params(labelsize=12)
    cb1.outline.set_linewidth(1)
    # cb1.dividers.set_color('none')
    if NAME != None:
        plt.savefig(NAME + ext, dpi=300, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()


def fast_plot2(imagename, crop=False, box_size=128, center=None, with_wcs=True,vmax_factor=0.5,
               vmin_factor=1, plot_colorbar=True, figsize=(5, 5), aspect=1, ax=None):
    """
    Fast plotting of an astronomical image with/or without a wcs header.

    imagename:
        str or 2d array.
        If str (the image file name), it will attempt to read the wcs and plot the coordinates axes.

        If 2darray, will plot the data with generic axes.

        support functions:
            ctn() -> casa to numpy: A function designed mainly to read CASA fits images,
                     but can be used to open any fits images.

                     However, it does not read header/wcs.
                     Note: THis function only works inside CASA environment.




    """
    if ax == None:
        fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(1,1,1)
    #     try:
    if isinstance(imagename, str) == True:
        if with_wcs == True:
            hdu = pf.open(imagename)
            #         hdu=pf.open(img)
            ww = WCS(hdu[0].header, naxis=2)
            try:
                if len(np.shape(hdu[0].data) == 2):
                    g = hdu[0].data[0][0]
                else:
                    g = hdu[0].data
            except:
                g = ctn(imagename)
        if with_wcs == False:
            g = ctn(imagename)

        if crop == True:
            xin, xen, yin, yen = do_cutout(imagename, box_size=box_size, center=center, return_='box')
            g = g[xin:xen, yin:yen]

    else:
        g = imagename

    if crop == True:
        max_x, max_y = np.where(g == g.max())
        xin = max_x[0] - box_size
        xen = max_x[0] + box_size
        yin = max_y[0] - box_size
        yen = max_y[0] + box_size
        g = g[xin:xen, yin:yen]

    if mad_std(g) == 0:
        std = g.std()
    else:
        std = mad_std(g)
    if ax == None:
        if with_wcs == True and isinstance(imagename, str) == True:
            ax = fig.add_subplot(projection=ww.celestial)
            ax.set_xlabel('RA')
            ax.set_ylabel('DEC')
        else:
            ax = fig.add_subplot()
            ax.set_xlabel('x pix')
            ax.set_ylabel('y pix')

    vmin = vmin_factor * std

    #     print(g)
    vmax = vmax_factor * g.max()

    norm = simple_norm(g, stretch='sqrt', asinh_a=0.02, min_cut=vmin, max_cut=vmax)

    im_plot = ax.imshow((g), cmap='magma_r', origin='lower', alpha=1.0, norm=norm,
                        aspect=aspect)  # ,vmax=vmax, vmin=vmin)#norm=norm
    #     ax.set_title('Image')
    try:
        levels_g = np.geomspace(5.0 * g.max(), 0.1 * g.max(), 7)
        #     x = np.geomspace(1.5*mad_std(g),10*mad_std(g),4)
        levels_black = np.geomspace(3 * (mad_std(g) + 0.00001), 0.1 * g.max(), 7)
    except:
        try:
            levels_g = np.geomspace(5.0 * g.max(), 3 * (mad_std(g), 7))
            levels_black = np.asarray([0])
        except:
            levels_g = np.asarray([0])
            levels_black = np.asarray([0])
    #     xneg = np.geomspace(5*mad_std(g),vmin_factor*mad_std(g),2)
    #     y = -xneg[::-1]
    #     levels_black = np.append(y,x)

    #     levels_white = np.geomspace(g.max(),10*mad_std(g),7)
    # levels_white = np.geomspace(g.max(), 0.1 * g.max(), 5)

    #     cg.show_contour(data, colors='black',levels=levels_black,linestyle='.-',linewidths=0.2,alpha=1.0)
    #     cg.show_contour(data, colors='#009E73',levels=levels_white[::-1],linewidths=0.2,alpha=1.0)
    try:
        ax.contour(g, levels=levels_black, colors='black', linewidths=0.2, alpha=1.0)  # cmap='Reds', linewidths=0.75)
        #     ax.contour(g, levels=levels_white[::-1],colors='#009E73',linewidths=0.2,alpha=1.0)#cmap='Reds', linewidths=0.75)
        ax.contour(g, levels=levels_g[::-1], colors='white', linewidths=0.6, alpha=1.0)  # cmap='Reds', linewidths=0.75)
    except:
        print('Not plotting contours!')
    #     cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))#,format=ticker.FuncFormatter(fmt))#cax=fig.add_axes([0.01,0.7,0.5,0.05]))#, orientation='horizontal')
    try:
        cb = plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91, 0.08, 0.05, 0.84]))
        cb.set_label(r"Flux [Jy/Beam]")
    except:
        pass
    # if ax==None:
    #     if plot_colorbar==True:
    #         # cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.08,0.05,0.82]))
    #         cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([0.91,0.08,0.05,0.32]))
    #         # cb=plt.colorbar(mappable=plt.gca().images[0], cax=fig.add_axes([-0.0,0.38,0.02,0.23]))
    #         cb.set_label(r"Flux [Jy/Beam]")
    return (ax)


#
# In [6]: import numpy as np
#    ...: import matplotlib.pyplot as plt
#    ...: from astropy.io import fits
#    ...: from astropy.wcs import WCS
#    ...: from astropy.visualization import ZScaleInterval
#    ...: from reproject import reproject_interp
#    ...:
#    ...: # Load the optical FITS image (VV705_r.fits)
#    ...: optical_fits = fits.open('VV705_r.fits')
#    ...: optical_data = optical_fits[0].data
#    ...: optical_header = optical_fits[0].header
#    ...:
#    ...: # Create a WCS object for the optical image
#    ...: optical_wcs = WCS(optical_header)
#    ...:
#    ...: # Load the radio contour data (vv705_radio.fits)
#    ...: radio_fits = fits.open('vv705_radio.fits')
#    ...: radio_data = radio_fits[0].data
#    ...: radio_header = radio_fits[0].header
#    ...:
#    ...: # Extract the 2D slice from the 4D radio data (assuming the desired slice is at index 0 for each extra dimension)
#    ...: radio_data_2d = radio_data
#    ...:
#    ...: # Create a WCS object for the 2D radio data
#    ...: radio_wcs_2d = WCS(radio_header, naxis=[1, 2])
#    ...:
#    ...: # Reproject the 2D radio data to match the optical WCS and shape
#    ...: radio_data_reprojected, _ = reproject_interp((radio_data_2d, radio_wcs_2d), optical_wcs, shape_out=optical_data.shape)
#    ...:
#    ...: # Calculate log-spaced contour levels based on the radio data
#    ...: radio_peak = np.nanmax(radio_data_reprojected)
#    ...: radio_std = np.nanstd(radio_data_reprojected)
#    ...: contour_levels = np.geomspace(0.1 * radio_peak, 0.1 * radio_std, 10)
#    ...:
#    ...: # Create a figure and axes with WCS projection
#    ...: fig = plt.figure(figsize=(8, 8))
#    ...: ax = fig.add_subplot(111, projection=optical_wcs)
#    ...:
#    ...: # Plot the optical image
#    ...: interval = ZScaleInterval()
#    ...: vmin, vmax = interval.get_limits(optical_data)
#    ...: ax.imshow(optical_data, cmap='magma', origin='lower')
#    ...:
#    ...: # Overlay radio contours on the optical image using log-spaced levels
#    ...: ax.contour(radio_data_reprojected, levels=contour_levels[::-1], colors='red', linewidths=1, alpha=0.7)
#    ...:
#    ...: # Set axis labels and title
#    ...: ax.set_xlabel('RA (J2000)')
#    ...: ax.set_ylabel('Dec (J2000)')
#    ...: ax.set_title('Optical Image with Log-Spaced Radio Contours (J2000)')
#    ...:
#    ...: # Show the plot
#    ...: plt.grid(color='white', linestyle='--', linewidth=0.5)
#    ...: plt.show()
