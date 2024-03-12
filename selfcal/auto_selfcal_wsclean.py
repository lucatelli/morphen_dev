"""
                                                          ..___|**_
                                                  .|||||||||*+@+*__*++.
                                              _||||.           .*+;].,#_
          Morphen                        _|||*_                _    .@@@#@.
                                   _|||||_               .@##@#| _||_
   Radio Self-Calibration     |****_                   .@.,/\..@_.
          Module             #///#+++*|    .       .@@@;#.,.\@.
                              .||__|**|||||*||*+@#];_.  ;,;_
     Geferson Lucatelli                        +\*_.__|**#
                                              |..      .]]
                                               ;@       @.*.
                                                #|       _;]];|.
                                                 ]_          _+;]@.
                                                 _/_             |]\|    .  _
                                              ...._@* __ .....     ]]+ ..   _
                                                  .. .       . .. .|.|_ ..


This module consists of performing interferometric imaging with wsclean and
running CASA's task gaincal for self-calibration.
It was tested for VLA (L,S,C,X,Ku) and eMERLIN (C band) observations.

Faint sources or higher-frequency observations (e.g. < 10 mJy)
may not work well. So, more experiments are required for
K and Ka VLA bands and eMERLIN fainter sources.

The user is advised to run the code in an interactive session (ipython),
step-by-step, and check the results of each step.
Check the config.py file at:
https://github.com/lucatelli/morphen/blob/main/selfcal/config.py

Note that the pure automated self-calibration is still experimental,
but showed to be good in most cases.

Check https://github.com/lucatelli/morphen/blob/main/selfcal/README.md for more information.

"""
__version__ = 0.3
__author__ = 'Geferson Lucatelli'
__email__ = 'geferson.lucatelli@postgrad.manchester.ac.uk'
__date__ = '2024 16 01'
print(__doc__)

import os
import sys
sys.path.append('../libs/')
import libs as mlibs
import glob
import pandas as pd
from casatasks import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
try:
    import casatools
    from casatasks import *
except:
    print('Not importing casatools. '
          'Maybe you are inside inside CASA? '
          'Or check your modular installation?')
    pass

from casaplotms import plotms
from casaviewer.imview import imview


msmd = casatools.msmetadata()
ms = casatools.ms()
tb = casatools.table()

# import config_combined as cf
import config as cf
from importlib import reload
reload(cf)

FIELD = cf.FIELD
ANTENNAS = cf.ANTENNAS
refantmode = cf.refantmode
SPWS = cf.SPWS
minblperant = cf.minblperant
cell_sizes_JVLA = cf.cell_sizes_JVLA
cell_sizes_eMERLIN = cf.cell_sizes_eMERLIN
taper_sizes_eMERLIN = cf.taper_sizes_eMERLIN
taper_sizes_JVLA = cf.taper_sizes_JVLA
receiver = cf.receiver
cell_size = cf.cell_size
taper_size = cf.taper_size

solnorm = cf.solnorm
combine = cf.combine
outlierfile = cf.outlierfile #deprecated, only used in CASA
quiet = cf.quiet
run_mode = cf.run_mode

path = cf.path
vis_list = cf.vis_list
steps = cf.steps



init_parameters = cf.init_parameters
global_parameters = cf.global_parameters
params_very_faint = cf.params_very_faint
params_faint = cf.params_faint
params_standard_1 = cf.params_standard_1
params_standard_2 = cf.params_standard_2
params_bright = cf.params_bright
params_trial_2 = cf.params_trial_2

def select_parameters(total_flux,snr=None):
    if total_flux < 10:
        params = params_very_faint.copy()
    elif 10 <= total_flux < 20:
        params = params_faint.copy()
    elif 20 <= total_flux < 50:
        params = params_standard_1.copy()
    elif 50 <= total_flux < 100:
        params = params_standard_2.copy()
    else:  # total_flux >= 100
        params = params_bright.copy()

    return params

def get_spwids(vis):
    lobs = listobs(vis=vis)
    extract_spwids = {key: lobs[key] for key in lobs if 'scan_' in key}

    unique_spwids = set()

    for key in extract_spwids:
        nested_dict = extract_spwids[key]
        for inner_key in nested_dict:
            spwids = nested_dict[inner_key]['SpwIds']
            # Convert the array to a sorted tuple and add to the set
            unique_spwids.add(tuple(sorted(spwids)))

    # Convert the set of tuples back to a sorted list of lists
    unique_spwids_lists = sorted([list(t) for t in unique_spwids])
    # Flatten the list and then convert to a set to get unique elements
    unique_elements = set(element for sublist in unique_spwids_lists for element in sublist)

    # Convert the set back to a list and sort it
    unique_elements_sorted = sorted(list(unique_elements))
    return unique_elements_sorted
def get_spwmap(vis):
    lobs = listobs(vis=vis)
    extract_spwids = {key: lobs[key] for key in lobs if 'scan_' in key}

    unique_spwids = set()

    for key in extract_spwids:
        nested_dict = extract_spwids[key]
        for inner_key in nested_dict:
            spwids = nested_dict[inner_key]['SpwIds']
            # Convert the array to a sorted tuple and add to the set
            unique_spwids.add(tuple(sorted(spwids)))

    # Convert the set of tuples back to a sorted list of lists
    unique_spwids_lists = sorted([list(t) for t in unique_spwids])

    counts = {}
    for lst in unique_spwids_lists:
        if lst[0] not in counts:
            counts[lst[0]] = len(lst)

    # Construct the spwmap
    spwmap_i = [item for item, count in counts.items() for _ in range(count)]
    spwmap = [spwmap_i[:len(get_spwids(g_vis))]]
    return spwmap



def print_table(data):
    """
    Print a simple dictionary as a user-friendly readable table to the terminal.
    """
    import tableprint
    rows = []
    for key, value in data.items():
        if isinstance(value, list):
            # Convert list to string
            value = ', '.join(map(str, value))
        rows.append((key, value))


    headers = ["Parameter", "Value"]
    tableprint.table(rows, headers)
    pass

def report_flag(summary, axis):
    for id, stats in summary[axis].items():
        print('%s %s: %5.1f percent flagged' % (
        axis, id, 100. * stats['flagged'] / stats['total']))
    pass


def compute_flux_density(imagename, residualname, mask=None):
    beam_area = mlibs.beam_area2(imagename)
    image_data = mlibs.ctn(imagename)
    if mask is None:
        rms = mlibs.mad_std(mlibs.ctn(residualname))
        _, mask = mlibs.mask_dilation(imagename, show_figure=False, PLOT=True, rms=rms, sigma=6,
                                      iterations=2)
    total_flux_density = mlibs.np.nansum(image_data * mask) / beam_area

    # compute error in flux density
    data_res = mlibs.ctn(residualname)
    res_error_rms = mlibs.np.sqrt(mlibs.np.nansum(
        (abs(data_res * mask - mlibs.np.nanmean(data_res * mask))) ** 2 * mlibs.np.nansum(
            mask))) / beam_area

    # res_error_rms = 3 * mlibs.np.nansum(data_res * mask)/beam_area

    print('-----------------------------------------------------------------')
    print('Estimate of flux error (based on rms of '
          'residual x area): ')
    print('Flux Density = ', total_flux_density * 1000, '+/-',
          res_error_rms * 1000, 'mJy')
    print('Fractional error flux = ', abs(res_error_rms) / total_flux_density)
    print('-----------------------------------------------------------------')
    # print(f"Flux Density = {total_flux_density*1000:.2f} mJy")
    return (total_flux_density, abs(res_error_rms))
def compute_image_stats(path,
                        image_list,
                        image_statistics,
                        prefix='',
                        sigma=None,
                        selfcal_step=None):
    """
    This function will compute statistics of a cleaned image from a wsclean run
    at a given self-cal step (provided an image prefix). It will also store
    associated model and residual images.

    Parameters
    ----------
    path : str
        Path to the image files.
    image_list : list
        List to store  the image names of each self-cal step.
    image_statistics : dict
        Dictionary to store the statistics of the images at a given self-cal step.
        It can be an existing dictionary.
    prefix : str
        Prefix of the image files.

    """
    file_list = glob.glob(f"{path}*{prefix}*MFS-image.fits")
    file_list.sort(key=os.path.getmtime, reverse=False)
    try:
        image_list[prefix] = file_list[-1]
    except:
        image_list[prefix] = file_list
    image_list[prefix+'_residual'] = image_list[prefix].replace(
        'MFS-image.fits', 'MFS-residual.fits')
    image_list[prefix+'_model'] = image_list[prefix].replace(
        'MFS-image.fits', 'MFS-model.fits')

    if sigma is None:
        if (selfcal_step == 'test_image') or (selfcal_step == 'p0'):
            """
            We must be more conservative when creating masks to compute the total flux density 
            before self-calibration. The image may contain artifacts above the default sigma 
            threshold of 6.0, and may lead to overestimation of the total flux density.
            An alternative sigma is 8. Note that the mask dilation is a very powerful approach and 
            very sensitive to the sigma threshold. A sigma of 8 results large differences in 
            relation to a sigma of 6. 
            """
            sigma = 8.0
        else:
            sigma = 6.0


    level_stats = mlibs.level_statistics(image_list[prefix],sigma=sigma)
    image_stats = mlibs.get_image_statistics(imagename=image_list[prefix],
                                             dic_data=level_stats,
                                             sigma_mask=sigma)
    img_props = mlibs.compute_image_properties(image_list[prefix],
                                               image_list[prefix+'_residual'],
                                               results = image_stats,
                                               sigma_mask = sigma,
                                               show_figure=False)[-1]
    image_statistics[prefix] = img_props

    sub_band_images = glob.glob(
        image_list[prefix].replace('-MFS-image.fits', '') + '-????-image.fits')
    sub_band_residuals = glob.glob(
        image_list[prefix + '_residual'].replace('-MFS-residual.fits',
                                                    '') + '-????-residual.fits')

    _FLUXES = []
    _FLUXES_err = []
    for i in range(len(sub_band_images)):
        flux_density, flux_density_err = compute_flux_density(sub_band_images[i],
                                                              sub_band_residuals[i],
                                                              mask=None)
        print('Flux density = ', flux_density)
        _FLUXES.append(flux_density)
        _FLUXES_err.append(flux_density_err)
    FLUXES = mlibs.np.asarray(_FLUXES)
    FLUXES_err = mlibs.np.asarray(_FLUXES_err)
    freqlist = mlibs.getfreqs(sub_band_images)

    plt.figure(figsize=(8, 6))
    plt.errorbar(freqlist / 1e9, FLUXES * 1000,
                 yerr=FLUXES_err * 1000,
                 fmt='o',
                 # label='Observed data',
                 color='k', ecolor='gray', alpha=0.5)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Flux Density [mJy]')
    plt.ylim(0,)
    plt.title('Sub-Band Images')
    plt.savefig(image_list[prefix].replace('-MFS-image.fits', '_freq_flux.jpg'), dpi=300,
                bbox_inches='tight')

    return(image_statistics,image_list)

def create_mask(imagename,rms_mask,sigma_mask,mask_grow_iterations,PLOT=False):

    valid_sigma_mask = sigma_mask
    while True:
        mask_valid = mlibs.mask_dilation(imagename,
                                         PLOT=PLOT,
                                         rms=rms_mask,
                                         dilation_size = None,
                                         sigma=valid_sigma_mask,
                                         iterations=1)[1]
        if mask_valid.sum() > 0:
            break
        print(' ++>> No mask found with sigma_mask:',valid_sigma_mask)
        print(' ++>> Reducing sigma_mask by 2 until valid mask is found...')
        valid_sigma_mask = valid_sigma_mask - 2.0

        if valid_sigma_mask <= 6:
            print("Reached minimum sigma threshold without finding a valid mask.")
            break

    mask_valid = mlibs.mask_dilation(imagename,
                                     PLOT=PLOT,
                                     rms=rms_mask,
                                     dilation_size=1,
                                     sigma=valid_sigma_mask-1,
                                     iterations=1)[1]

    mask = mask_valid
    mask_wslclean = mask * 1.0  # mask in wsclean is inverted
    mask_name = imagename.replace('.fits', '') + '_mask.fits'
    mlibs.pf.writeto(mask_name, mask_wslclean, overwrite=True)
    return(mask_name)

def sinusoidal_function(t, A, omega, phi, offset):
    """
    ###############
    *** TESTING ***
    ###############
    """
    return A * np.sin(omega * t + phi) + offset


def fit_function_and_estimate_coherence(time, phase,PLOT=False):
    """
    ###############
    *** TESTING ***
    ###############
    """
    # Initial guess for the parameters: amplitude, angular frequency, phase shift, and offset
    guess_amplitude = (np.max(phase) - np.min(phase)) / 2
    # guess_omega = 2 * np.pi / np.ptp(time)  # Assume a period equal to the range of time
    # guess_amplitude = np.nanmean(phase)
    guess_omega = 2 * np.pi / 120.  # Assume a period equal to the range of time

    guess_phi = phase[0]
    guess_offset = np.mean(phase)

    p0 = [guess_amplitude, guess_omega, guess_phi, guess_offset]

    # Curve fitting
    try:
        params, covariance = curve_fit(sinusoidal_function, time, phase, p0=p0)
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        return None, None

    # Calculate the fitted curve
    fitted_phase = sinusoidal_function(time, *params)
    if PLOT is True:
        # Plot the original data and the fitted curve
        plt.figure(figsize=(10, 5))
        plt.scatter(time, phase, label='Original Data')
        plt.plot(time, fitted_phase, label='Fitted Curve', color='red')
        plt.legend()
        plt.xlabel('Time (seconds)')
        plt.ylabel('Phase (degrees)')
        plt.title('Phase Data and Fitted Curve')
        plt.show()

    # Analyze the residuals
    residuals = phase - fitted_phase
    residual_std = np.std(residuals)

    # Use the period of the fitted sinusoid as an estimate for coherence time
    estimated_coherence_time = 2 * np.pi / params[1]

    return estimated_coherence_time, residual_std


def estimate_local_coherence_time(time, phase,PLOT=False):
    """
    ###############
    *** TESTING ***
    ###############
    """
    # Normalize phase
    phase_normalized = phase - np.mean(phase)

    # Calculate the autocorrelation of the phase signal
    autocorr = np.correlate(phase_normalized, phase_normalized, mode='full')
    autocorr = autocorr[
               autocorr.size // 2:]  # Take the second half of the autocorrelation
    autocorr /= np.max(autocorr)  # Normalize the autocorrelation

    # Find the first point where the autocorrelation drops below 0.5 after the peak
    half_max_index = np.where(autocorr < 0.5)[0][0] if np.any(autocorr < 0.5) else len(
        autocorr) - 1

    # The coherence time is the time corresponding to the half-max index
    coherence_time = time[half_max_index] - time[0] if half_max_index < len(time) else \
    time[-1] - time[0]
    if PLOT is True:
        # Plot the autocorrelation
        plt.figure(figsize=(10, 5))
        time_delays = time[:autocorr.size] - time[0]
        plt.plot(time_delays, autocorr)
        plt.axvline(x=time_delays[half_max_index], color='r', linestyle='--',
                    label='Local Coherence Time')
        plt.title('Autocorrelation of Phase Signal')
        plt.xlabel('Time Delay (seconds)')
        plt.ylabel('Autocorrelation')
        plt.legend()
        plt.grid(True)
        plt.show()

    return coherence_time

def table_phase_time0(vis,gain_table,spw,intent='*TARGET*',ant = None):
    """
    ###############
    *** TESTING ***
    ###############
    """
    # msmd = casatools.msmetadata()
    # ms = casatools.ms()
    # tb = casatools.table()
    msmd.open(vis)
    scans = msmd.scansforintent(intent).tolist()
    msmd.close()

    tb.open(gain_table)
    time, spwid, gain, antennae = tb.getcol('TIME'), tb.getcol(
        'SPECTRAL_WINDOW_ID'), tb.getcol('CPARAM'), tb.getcol('ANTENNA1')
    tb.close()
    unique_spwids = np.unique(spwid)
    msmd.open(vis)
    scans = msmd.scansforintent(intent).tolist()
    msmd.close()

    # mstool = casatools.ms
    # myms = mstool()
    # myms.open(vis)
    #
    # # if spw is None:
    #
    # myms.selectinit(datadescid=unique_spwids[0])
    # myms.close()

    # compute co-time per spw for all scans (use all antennas)
    co_times_spw = np.zeros(len(unique_spwids))
    co_times_spw[:] = np.nan
    for i in range(len(unique_spwids)):
        spw = unique_spwids[i]
        match = spwid == spw
        if ant is not None:
            match &= antennae == int(ant)

        # fig = plt.gcf()
        # ax1 = plt.subplot(3, 1, 1)
        timee = time[match] - time[match][0]
        phase = np.mean((np.angle(gain[:, 0, match].T) * 180 / np.pi), axis=1)
        # ax1.scatter(timee, phase, marker='.', s=1)
        coherence_time, residual_std = fit_function_and_estimate_coherence(timee, phase)
        co_times_spw[i] = coherence_time
        # coherence_time = estimate_local_coherence_time(timee, phase)
        if coherence_time == 0.0:
            co_times_spw[i] = np.nan
        else:
            co_times_spw[i] = coherence_time
        print(f"Estimated Coherence Time for spw {spw} : {coherence_time} seconds")
        # print(f"Residual Standard Deviation: {residual_std} degrees")
    if np.isnan(np.nanmean(co_times_spw)) == True:
        solint = np.inf
    else:
        solint = int(np.nanmean(co_times_spw))
    return(solint)
def table_phase_time(vis,gain_table,spw,intent='*TARGET*',ant = None):
    """
    ###############
    *** TESTING ***
    ###############
    """
    # msmd = casatools.msmetadata()
    # ms = casatools.ms()
    # tb = casatools.table()
    msmd.open(vis)
    scans = msmd.scansforintent(intent).tolist()
    msmd.close()

    tb.open(gain_table)
    time, spwid, gain, antennae = tb.getcol('TIME'), tb.getcol(
        'SPECTRAL_WINDOW_ID'), tb.getcol('CPARAM'), tb.getcol('ANTENNA1')
    tb.close()
    unique_spwids = np.unique(spwid)
    msmd.open(vis)
    scans = msmd.scansforintent(intent).tolist()
    msmd.close()

    mstool = casatools.ms
    myms = mstool()
    myms.open(vis)

    # if spw is None:

    myms.selectinit(datadescid=unique_spwids[0])
    myms.close()

    # compute co-time per spw for all scans (use all antennas)
    co_times_spw = np.zeros(len(unique_spwids))
    co_times_spw[:] = np.nan
    for i in range(len(unique_spwids)):
        spw = unique_spwids[i]
        match = spwid == spw
        if ant is not None:
            match &= antennae == int(ant)

        # fig = plt.gcf()
        # ax1 = plt.subplot(3, 1, 1)
        timee = time[match] - time[match][0]
        phase = np.mean((np.angle(gain[:, 0, match].T) * 180 / np.pi), axis=1)
        # ax1.scatter(timee, phase, marker='.', s=1)
        # coherence_time, residual_std = fit_function_and_estimate_coherence(timee, phase)
        # co_times_spw[i] = coherence_time
        coherence_time = estimate_local_coherence_time(timee, phase)
        if coherence_time == 0.0:
            co_times_spw[i] = np.nan
        else:
            co_times_spw[i] = coherence_time
        print(f"Estimated Coherence Time for spw {spw} : {coherence_time} seconds")
        # print(f"Residual Standard Deviation: {residual_std} degrees")

    if np.isnan(np.nanmean(co_times_spw)) == True:
        solint = np.inf
    else:
        solint = int(np.nanmean(co_times_spw))
    return(solint)

def plot_visibilities(g_vis, name, with_DATA=False, with_MODEL=False,
                      with_CORRECTED=False):

    if with_MODEL == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='baseline',
               ydatacolumn='model', avgchannel='64', avgtime='360',
               correlation='LL,RR',plotrange=[0,0,0,0],
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_model.jpg')


        plotms(vis=g_vis, xaxis='freq', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='scan',
               ydatacolumn='model', avgchannel='', avgtime='360',
               correlation='LL,RR',plotrange=[0,0,0,0],
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_freq_amp_model.jpg')

    if with_DATA == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='baseline',
               ydatacolumn='data', avgchannel='64', avgtime='360',
               correlation='LL,RR',plotrange=[0,0,0,0],
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               # plotrange=[-1,-1,-1,0.3],
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_data.jpg')

        plotms(vis=g_vis, xaxis='freq', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='scan',
               ydatacolumn='data', avgchannel='', avgtime='360',
               correlation='LL,RR',plotrange=[0,0,0,0],
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_freq_amp_data.jpg')

    if with_CORRECTED == True:
        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
               antenna=ANTENNAS, spw=SPWS, coloraxis='baseline', avgantenna=True,
               ydatacolumn='corrected-model', avgchannel='64', avgtime='360',
               correlation='LL,RR', plotrange=[0, 0, 0, 0],
               width=1000, height=440, showgui=False, overwrite=True, dpi=1200, highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_corrected-model.jpg')

        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp',
               antenna=ANTENNAS, spw=SPWS, coloraxis='baseline', avgantenna=True,
               ydatacolumn='corrected/model', avgchannel='64', avgtime='360',
               correlation='LL,RR',
               width=1000, height=440, showgui=False, overwrite=True, dpi=1200, highres=True,
               plotrange=[0, 0, 0, 5],
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_corrected_div_model.jpg')

        plotms(vis=g_vis, xaxis='UVwave', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS,
               # plotrange=[-1,-1,0,0.3],
               ydatacolumn='corrected', avgchannel='64', avgtime='360',
               correlation='LL,RR',plotrange=[0,0,0,0],
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_uvwave_amp_corrected.jpg')

        plotms(vis=g_vis, xaxis='freq', yaxis='amp', avgantenna=True,
               antenna=ANTENNAS, spw=SPWS, coloraxis='scan',
               ydatacolumn='corrected', avgchannel='', avgtime='360',
               correlation='LL,RR',plotrange=[0,0,0,0],
               width=1000, height=440, showgui=False, overwrite=True,dpi=1200,highres=True,
               plotfile=os.path.dirname(
                   g_vis) + '/selfcal/plots/' + name + '_freq_amp_corrected.jpg')

    pass


def start_image(g_name, n_interaction, imsize='2048', imsizey=None, cell='0.05asec',
                robust=0.0,
                base_name=None,
                nsigma_automask = '7.0',nsigma_autothreshold='0.1',
                delmodel=False, niter=600,
                opt_args = '',quiet=True,shift=None,
                PLOT=False, datacolumn='DATA',mask=None,
                savemodel=True, uvtaper=[""]):
    '''
    Wsclean wrapper function. It calls wslcean from the command line with some
    predifined arguments. This initial step runs on the DATA column and creates
    the initial model which is used to calculate the initial complex self-gains.
    '''
    g_vis = g_name + '.ms'
    if imsizey is None:
        imsizey = imsize
    if base_name is None:
        base_name = str(n_interaction)+'_start_image_'
    else:
        base_name = base_name



    os.system("export OPENBLAS_NUM_THREADS=1 && python imaging_with_wsclean.py --f " +
              g_name + " --sx "
              + str(imsize) + " --sy " + str(imsizey) + " --niter "
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
                          with_MODEL=True, with_DATA=True, with_CORRECTED=True)

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
        # plt.semilogx()
        plt.axvline(x=3, color='k', linestyle='--')
        plt.axvline(x=cut_off, color='r', linestyle='--')
        plt.grid()
        if save_fig == True:
            plt.savefig(caltable.replace('.tb', '.jpg'), dpi=300, bbox_inches='tight')
        # plt.show()
        plt.clf()
        plt.close()

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
               width=1000, height=400, dpi=600, overwrite=True, showgui=True,
               # correlation='LL,RR',
               plotfile=os.path.dirname(
                   table) + '/plots/' + stage + '/' + table_type + '_' + xaxis + '_' + yaxis + '_field_' + str(
                   'all') + '.jpg')


        plotms(vis=table, xaxis=xaxis, yaxis=yaxis, field='',avgbaseline=True,
               gridcols=1, gridrows=1, coloraxis='spw', antenna='', plotrange=plotrange,
               width=1000, height=400, dpi=600, overwrite=True, showgui=True,
               # correlation='LL,RR',
               plotfile=os.path.dirname(
                   table) + '/plots/' + stage + '/' + table_type + '_' + xaxis + '_' +
                        yaxis + '_avgbaseline_field_' + str(
                   'all') + '.jpg')

    else:

        for FIELD in fields:
            plotms(vis=table, xaxis=xaxis, yaxis=yaxis, field=FIELD,
                   # gridcols=4,gridrows=4,coloraxis='spw',antenna='',iteraxis='antenna',
                   # width=2048,height=1280,dpi=256,overwrite=True,showgui=False,
                   gridcols=1, gridrows=1, coloraxis='spw', antenna='',
                   plotrange=plotrange,
                   width=1000, height=400, dpi=600, overwrite=True, showgui=False,
                   # correlation='LL,RR',
                   plotfile=os.path.dirname(
                       table) + '/plots/' + stage + '/' + table_type + '_' + xaxis + '_' + yaxis + '_field_' + str(
                       FIELD) + '.jpg')

    pass

def check_solutions(g_name, field, cut_off=2.0, minsnr=2.0, n_interaction=0, uvrange='',
                    solnorm=solnorm, combine='', calmode='p', gaintype='G',solint_factor=1.0,
                    interp = '',spwmap = [],
                    gain_tables_selfcal=[''], special_name='',refant = '', minblperant=4,
                    return_solution_stats=False):
    g_vis = g_name + '.ms'
    minsnr = minsnr
    solint_template = np.asarray([24,48,96,192,384])
    solints = solint_template * solint_factor

    caltable_int = (os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_int_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' +
                    gaintype + special_name + '.tb')

    caltable_1 = (os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+
                  str(int(solints[0]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' +
                  gaintype + special_name + '.tb')

    caltable_2 = (os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+
                  str(int(solints[1]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' +
                  gaintype + special_name + '.tb')

    caltable_3 = (os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+
                  str(int(solints[2]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' +
                  gaintype + special_name + '.tb')

    caltable_4 = (os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+
                  str(int(solints[3]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' +
                  gaintype + special_name + '.tb')

    caltable_5 = (os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_'+
                  str(int(solints[4]))+'_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' +
                  gaintype + special_name + '.tb')

    caltable_inf = (os.path.dirname(g_name) + '/selfcal/selfcal_test_' + str(
        n_interaction) + '_' + os.path.basename(g_name) + '_solint_inf_minsnr_' + str(
        minsnr) + '_calmode' + calmode + '_combine' + combine + '_gtype_' +
                    gaintype + special_name + '.tb')


    if not os.path.exists(caltable_int):
        print('>> Performing test-gaincal for solint=int...')
        gaincal(vis=g_vis, caltable=caltable_int, solint='int', refant=refant,
                interp=interp,spwmap = spwmap,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_1):
        print('>> Performing test-gaincal for solint='+str(solints[0])+'s...')
        gaincal(vis=g_vis, caltable=caltable_1, solint=str(solints[0])+'s',
                refant=refant,interp=interp,spwmap = spwmap,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_2):
        print('>> Performing test-gaincal for solint='+str(solints[1])+'s...')
        gaincal(vis=g_vis, caltable=caltable_2, solint=str(solints[1])+'s',
                refant=refant,interp=interp,spwmap = spwmap,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_3):
        print('>> Performing test-gaincal for solint='+str(solints[2])+'s...')
        gaincal(vis=g_vis, caltable=caltable_3, solint=str(solints[2])+'s',
                refant=refant,interp=interp,spwmap = spwmap,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_4):
        print('>> Performing test-gaincal for solint='+str(solints[3])+'s...')
        gaincal(vis=g_vis, caltable=caltable_4, solint=str(solints[3])+'s',
                refant=refant,interp=interp,spwmap = spwmap,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_5):
        print('>> Performing test-gaincal for solint='+str(solints[4])+'s...')
        gaincal(vis=g_vis, caltable=caltable_5, solint=str(solints[4])+'s',
                refant=refant,interp=interp,spwmap = spwmap,
                solnorm=solnorm, combine=combine, minblperant=minblperant,
                calmode=calmode, gaintype=gaintype, minsnr=minsnr, uvrange=uvrange,
                gaintable=gain_tables_selfcal)
    if not os.path.exists(caltable_inf):
        print('>> Performing test-gaincal for solint=inf...')
        gaincal(vis=g_vis, caltable=caltable_inf, solint='inf', refant=refant,
                interp=interp,spwmap = spwmap,
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
        # plt.semilogx()
        plt.savefig(os.path.dirname(g_name) + '/selfcal/plots/' + str(n_interaction) +
                    '_' + os.path.basename(
            g_name) + '_calmode' + calmode + '_combine' + combine + '_gtype_' + gaintype
                    + special_name + '_gain_solutions_comparisons_norm.pdf')
        # plt.clf()
        # plt.close()
        # plt.show()

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
        # plt.semilogx()
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

        # plt.show()
        # print('################################')
        # print(np.mean(snr_int))
        # print('################################')
        # print(stats.percentileofscore(snr_int, cut_off))
        SNRs = [
                np.array(snr_int),
                np.array(snr_1),
                np.array(snr_2),
                np.array(snr_3),
                np.array(snr_4),
                np.array(snr_5),
                np.array(snr_inf)]
        percentiles_SNRs = np.asarray([
                                       stats.percentileofscore(snr_int, cut_off),
                                       stats.percentileofscore(snr_1, cut_off),
                                       stats.percentileofscore(snr_2, cut_off),
                                       stats.percentileofscore(snr_3, cut_off),
                                       stats.percentileofscore(snr_4, cut_off),
                                       stats.percentileofscore(snr_5, cut_off),
                                       stats.percentileofscore(snr_inf, cut_off)])

        snr_data = {
            'int': SNRs[0],
            '24s': SNRs[1],
            '48s': SNRs[2],
            '96s': SNRs[3],
            '192s': SNRs[4],
            '384s': SNRs[5],
            'inf': SNRs[6]
        }

        if return_solution_stats:
            return (snr_data, percentiles_SNRs)
        else:
            pass
        plt.clf()
        plt.close()

    def compare_phase_variation():
        plotms(caltable_1, antenna='', scan='', yaxis='phase', avgbaseline=True)

        plotms(caltable_3, antenna='', scan='', yaxis='phase', plotindex=1,
               clearplots=False, customsymbol=True, symbolsize=20, avgbaseline=True,
               symbolcolor='ff0000', symbolshape='circle')

        plotms(caltable_2, antenna='', scan='', yaxis='phase', plotindex=2,
               clearplots=False, customsymbol=True, symbolsize=12, avgbaseline=True,
               symbolcolor='green', symbolshape='square')

        plotms(caltable_inf, antenna='', scan='', yaxis='phase', plotindex=3,
               clearplots=False, customsymbol=True, symbolsize=8, avgbaseline=True,
               symbolcolor='yellow', symbolshape='square')

        plotms(caltable_4, antenna='', scan='', yaxis='phase', plotindex=4,
               clearplots=False, customsymbol=True, symbolsize=4, avgbaseline=True,
               symbolcolor='purple', symbolshape='square',
               width=1300, height=400, showgui=True, overwrite=True,
               plotfile=os.path.dirname(g_name) + '/selfcal/plots/' + str(
                   n_interaction) +
                        '_' + os.path.basename(g_name) + '_combine' + '_calmode' + calmode + combine +
                        '_gtype_' + gaintype + special_name +
                        '_phase_variation_intervals.jpg')

    def compare_amp_variation():
        plotms(caltable_1, antenna='', scan='', yaxis='amp',
               plotrange=[0, 0, 0, 0],
               avgbaseline=True)

        plotms(caltable_3, antenna='', scan='', yaxis='amp', plotindex=1,
               plotrange=[0, 0, 0, 0],
               clearplots=False, customsymbol=True, symbolsize=20, avgbaseline=True,
               symbolcolor='ff0000', symbolshape='circle')

        plotms(caltable_2, antenna='', scan='', yaxis='amp', plotindex=2,
               plotrange=[0, 0, 0, 0],
               clearplots=False, customsymbol=True, symbolsize=12, avgbaseline=True,
               symbolcolor='green', symbolshape='square')

        plotms(caltable_inf, antenna='', scan='', yaxis='amp', plotindex=3,
               plotrange=[0, 0, 0, 0],
               clearplots=False, customsymbol=True, symbolsize=8, avgbaseline=True,
               symbolcolor='yellow', symbolshape='square')

        plotms(caltable_4, antenna='', scan='', yaxis='amp', plotindex=4,
               plotrange=[0, 0, 0, 0],
               clearplots=False, customsymbol=True, symbolsize=4, avgbaseline=True,
               symbolcolor='purple', symbolshape='square',
               width=1300, height=400, showgui=True, overwrite=True,
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
        return (SNRs, percentiles_SNRs, caltable_int, caltable_3, caltable_inf)
    else:
        pass


def run_wsclean(g_name, n_interaction, imsize='2048', imsizey=None,cell='0.05asec',
                robust=0.5,base_name=None,
                savemodel=True,shift=None,
                nsigma_automask='8.0', nsigma_autothreshold='1.0',
                datacolumn='CORRECTED',mask=None,
                niter=1000,quiet=True,
                with_multiscale=False, scales="'0,5,20,40'",
                uvtaper=[], PLOT=False):


    g_vis = g_name + '.ms'
    if imsizey is None:
        imsizey = imsize
    if base_name is None:
        base_name  = str(n_interaction)+'_update_model_image_'
    else:
        base_name = base_name



    os.system("export OPENBLAS_NUM_THREADS=1 && python imaging_with_wsclean.py --f " +
              g_name + " --sx "
              + str(imsize) + " --sy " + str(imsizey) + " --niter "
              + str(niter) + " --data " + datacolumn + " --cellsize " + cell
              + ' --nsigma_automask ' + nsigma_automask + ' --mask '+str(mask)
              + ' --nsigma_autothreshold ' + nsigma_autothreshold
              + ' --scales ' + scales
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
                  combine=combine, solnorm=False,refantmode = 'strict',
                  spwmap=[],uvrange='',append=False,solmode='',#L1R
                  minsnr=5.0, solint='inf', gaintype='G', calmode='p',
                  interp = '',refant = '', minblperant = 4,
                  action='apply', flagbackup=True,
                  PLOT=False, with_CORRECTED=True, with_MODEL=True, with_DATA=False,
                  special_name=''):
    g_vis = g_name + '.ms'
    # refantmode = 'flex' if refantmode == 'flex' else 'strict'
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
            print(' ++==> Using normalised solutions for amplitude self-calibration.')
            solnorm = True
        else:
            solnorm = False
        gaincal(vis=g_vis, field=FIELD, caltable=caltable, spwmap=spwmap,
                solint=solint, gaintable=gain_tables, combine=combine,
                refant=refant, calmode=calmode, gaintype=gaintype,
                refantmode = refantmode,
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
                              with_CORRECTED=with_CORRECTED,
                              with_MODEL=with_MODEL,
                              with_DATA=with_DATA)

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
             datacolumn=datacolumn, ntime='scan', combinescans=False,
             extendflags=False,
             winsize=winsize,
             channelavg=True,chanbin=4, timeavg=True, timebin='24s',
             timedevscale=timedevscale, freqdevscale=freqdevscale,
             flagnearfreq=False, flagneartime=False, growaround=True,
             action=action, flagbackup=False, savepars=True
             )

    if action == 'apply':
        flagdata(vis=g_vis, field='', spw='',
                 datacolumn=datacolumn,
                 mode='extend', action=action, display='report',
                 flagbackup=False, growtime=75.0,
                 growfreq=75.0, extendpols=False)

    if action == 'apply':
        flagmanager(vis=g_name + '.ms', mode='save', versionname='seflcal_after_rflag',
                    comment='After rflag at selfcal step.')
        try:
            print(' ++==> Running statwt on split data pos rflag...')
            statwt(vis=g_vis, statalg='chauvenet', timebin='24s',
                   datacolumn='corrected',minsamp = 3)
        except:
            print(' ++==> Running statwt on split data pos rflag...')
            statwt(vis=g_vis, statalg='chauvenet', timebin='24s',
                   datacolumn='data', minsamp = 3)

        print(' ++==> Flag statistics after rflag:')
        summary_after = flagdata(vis=g_vis, field='', mode='summary')
        report_flag(summary_after, 'field')

def find_refant(msfile, field,tablename):
    """
    This function comes from the e-MERLIN CASA Pipeline.
    https://github.com/e-merlin/eMERLIN_CASA_pipeline/blob/master/functions/eMCP_functions.py#L1501
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

def find_optimal_parameters(snr_data, max_frac_flag_current=0.15):
    """
    ###############
    *** TESTING ***
    ###############
    """
    snr_thresholds = np.linspace(1.2, 3.0, int((3.0-1.2)*10+13))
    best_time_interval = None
    best_snr_threshold = None
    best_balance = float('inf')  # Initialize with a large number

    for time_interval in snr_data.keys():
        for snr_threshold in snr_thresholds:
            current_snr_array = snr_data[time_interval]
            flagged_data_percentage = np.mean(current_snr_array < snr_threshold)

            # Check if the flagged data percentage is within the acceptable limit
            if flagged_data_percentage <= max_frac_flag_current:
                unflagged_data_variance = np.var(current_snr_array[current_snr_array >= snr_threshold])

                # Balance criterion: Consider data variance
                balance = flagged_data_percentage + unflagged_data_variance  # This can be adjusted

                if balance < best_balance:
                    best_balance = balance
                    best_time_interval = time_interval
                    best_snr_threshold = snr_threshold

    return best_time_interval, best_snr_threshold

def find_multiple_solint(opt_solint,
                         solint_template = [384, 192, 96, 48, 24, 12]):
    """
    ###############
    *** TESTING ***
    ###############
    """
    for t in solint_template:
        if t <= opt_solint:
            return t
    return solint_template[-1]



def estimate_solint(g_name, SNRs, cutoff=1.5):
    """
    ###############
    *** TESTING ***
    ###############
    """
    # Solution interval template.
    time_bins = ['24s', '48s', '96s', '192s', '384s', 'inf']

    # Initial cutoff value
    # cutoff = 2.0

    summary = flagdata(vis=g_name + '.ms', field='', mode='summary')
    init_flags = 100 * summary['flagged'] / summary['total']

    init_fraction = mlibs.scipy.stats.percentileofscore(SNRs[0], 0.5)
    # max_flagged = 25.0  + init_fraction # Maximum fraction of flagged data (10%)
    max_flagged = init_flags - init_fraction
    print(f"Initial fraction of flagged data: {init_flags}%")
    print(f"Estimated fraction of total flagged data from gaintable: {max_flagged}%")
    found_optimal = False

    while not found_optimal and cutoff >= 0.5:
        #     SNR, flagged_data = analyze_SNR(time_bins)

        for idx, snr_val in enumerate(SNRs):
            percentile = mlibs.scipy.stats.percentileofscore(SNRs[idx], cutoff)

            if percentile <= max_flagged:
                print(f"Optimal SNR: {snr_val} for time bin {time_bins[idx]} seconds.")
                found_optimal = True
                break

        if not found_optimal:
            cutoff -= 0.5
            print(
                f"No SNR meets the condition for less than {max_flagged}% flagged data. "
                f"Reducing cutoff to {cutoff} and re-running analysis.")

    if cutoff < 0.5:
        print(
            "Unable to find an optimal SNR within the specified cutoff limits.")
    return (time_bins[idx], cutoff)

# run_mode = 'jupyter'


# run_mode = 'terminal'
if run_mode == 'terminal':

    """
    If running this in a terminal, you can safely set quiet=False. This 
    refers to the wsclean quiet parameter. If running this code in a Jupyter Notebook, 
    please set quiet=True. Otherwise, jupyter can crash due to the very long 
    output cells. 
    """
    for field in vis_list:
        g_name = path + field
        g_vis = g_name + '.ms'

        try:
            steps_performed
        except NameError:
            steps_performed = []

        if 'startup' in steps and 'startup' not in steps_performed:
            """
            Create basic directory structure for saving tables and plots.
            """
            print(f"++==> Preparing to selfcalibrate {g_vis}.")
            print('++==> Creating basic directory structure.')
            if not os.path.exists(path + 'selfcal/'):
                os.makedirs(path + 'selfcal/')
            if not os.path.exists(path + 'selfcal/plots'):
                os.makedirs(path + 'selfcal/plots')
            image_list = {}
            residual_list = {}
            model_list = {}
            image_statistics = {}
            gain_tables_applied = {}
            parameter_selection = {}
            trial_gain_tables = []
            final_gain_tables = []
            steps_performed = []
            # if delmodel == True:
            print('--==> Clearing model and cal visibilities.')
            delmod(g_vis,otf=True,scr=False)
            clearcal(g_vis)

            # start the CASA logger (open the window).
            import casalogger.__main__
            # Q. How do I close it after???
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
                    statwt(vis=g_vis, statalg='chauvenet', timebin='24s', datacolumn='data')
            else:
                print("     --==> Skipping flagging backup init (exists).")
                print("     --==> Restoring flags to original...")
                flagmanager(vis=g_name + '.ms', mode='restore', versionname='Original')
                if not os.path.exists(g_name + '.ms.flagversions/flags.statwt_1/'):
                    print("     ++==> Running statwt.")
                    statwt(vis=g_vis, statalg='chauvenet', timebin='24s', datacolumn='data')
            print(" ++==> Amount of data flagged at the start of selfcal.")
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
            parameter_selection = {}
            trial_gain_tables = []
            final_gain_tables = []
            import casalogger.__main__

            steps_performed = []

        if 'fov_image' in steps and 'fov_image' not in steps_performed:
            """
            Create a FOV dirty image.
            """
            # niter = 50#knowing the dirty image is enough.
            # robust = 0.5  # or 0.5 if lots of extended emission.
            run_wsclean(g_name, imsize=init_parameters['fov_image']['imsize'],
                        # cell=cell_sizes_JVLA['C'],
                        cell=init_parameters['fov_image']['cell'],
                        robust=init_parameters['fov_image']['robust'],
                        base_name=init_parameters['fov_image']['basename'],
                        nsigma_automask='8.0', nsigma_autothreshold='3.0',
                        n_interaction='0', savemodel=False, quiet=quiet,
                        datacolumn='DATA', shift=None,
                        # shift="'18:34:46.454 +059.47.32.191'",
                        # uvtaper=['0.05arcsec'],
                        niter=init_parameters['fov_image']['niter'],
                        PLOT=False)
            file_list = glob.glob(f"{path}*MFS-image.fits")
            file_list.sort(key=os.path.getmtime, reverse=False)

            try:
                image_list['FOV_image'] = file_list[-1]
            except:
                image_list['FOV_image'] = file_list
            image_list['FOV_residual'] = image_list['FOV_image'].replace(
                'MFS-image.fits','MFS-residual.fits')
            image_list['FOV_model'] = image_list['FOV_image'].replace(
                'MFS-image.fits','MFS-model.fits')
            steps_performed.append('fov_image')

        if 'run_rflag_init' in steps:
            """
            Run automatic rflag on the data before selfcalibration.
            """
            run_rflag(g_vis, display='report', action='apply',
                      timedevscale=4.0, freqdevscale=4.0, winsize=7, datacolumn='data')
            steps_performed.append('run_rflag_init')


        """
        This is the moment we define global image/cleaning pro\perties. 
        These will be used in the subsequent steps of selfcalibration.
        """
        ########################################
        imsize = global_parameters['imsize']
        imsizey = global_parameters['imsizey']
        cell = global_parameters['cell']
        FIELD_SHIFT = global_parameters['FIELD_SHIFT']
        niter = global_parameters['niter']
        # e.g.
        # FIELD_SHIFT = "'13:37:30.309  +48.17.42.830'" #NGC5256
        ########################################

        # if 'test_image' in steps and 'test_image' not in steps_performed:
        if 'test_image' in steps and 'test_image' not in steps_performed:
            """
            After creating a FOV image or checking info about other sources in the 
            field (e.g. NVSS, FIRST, etc), you may want to create a basic 
            image to that center (or None) to see how the image (size) 
            will accomodate the source(s). This setting will be used in all the
            subsequent steps of selfcalibration. 
            
            Note also that masks are not used in this step, but the image will be used 
            to create a mask for the next step, which will be the first step of 
            selfcalibration. 
            """
            niter_test = init_parameters['test_image']['niter']
            robust = init_parameters['test_image']['robust']
            prefix = init_parameters['test_image']['prefix']
            run_wsclean(g_name, imsize=imsize, imsizey = imsizey, cell=cell,
                        robust=robust, base_name=prefix,
                        nsigma_automask='5.0', nsigma_autothreshold='3.0',
                        n_interaction='0', savemodel=False, quiet=quiet,
                        datacolumn='DATA', shift=FIELD_SHIFT,
                        with_multiscale=False, scales='0,5,20,40',
                        uvtaper=init_parameters['test_image']['uvtaper'],
                        niter=niter_test,
                        PLOT=False)

            image_statistics,image_list = compute_image_stats(path=path,
                                                              image_list=image_list,
                                                              image_statistics=image_statistics,
                                                              prefix=prefix,
                                                              selfcal_step='test_image')

            current_total_flux = image_statistics['test_image']['total_flux_mask'] * 1000
            if current_total_flux > 50.0:
                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix=prefix,
                                                                   sigma=25,
                                                                   selfcal_step='test_image')

            modified_robust = None
            if current_total_flux < 5.0:
                """
                Sometimes, a lower robust parameter (e.g. 0.0) may result in an image 
                with a lower flux density in relation to an image recovered with a
                higher robust parameter (e.g. 0.5 or 1.0), depending of the structure 
                of the source. In such cases, we attempt an image with a higher value, 
                and check if that is actually true.
                """
                if init_parameters['test_image']['uvtaper'] != ['']:
                    uvtaper = init_parameters['test_image']['uvtaper']
                    run_wsclean(g_name, imsize=imsize, imsizey=imsizey, cell=cell,
                                robust=robust, base_name=prefix,
                                nsigma_automask=global_parameters['nsigma_automask'],
                                nsigma_autothreshold=global_parameters['nsigma_autothreshold'],
                                n_interaction='0', savemodel=False, quiet=quiet,
                                datacolumn='DATA', shift=FIELD_SHIFT,
                                with_multiscale=False, scales='0,5,20,40',
                                uvtaper=uvtaper,
                                niter=niter,
                                PLOT=False)
                else:
                    modified_robust = robust + 0.5
                    run_wsclean(g_name, imsize=imsize, imsizey=imsizey, cell=cell,
                                robust=modified_robust, base_name=prefix,
                                nsigma_automask=global_parameters['nsigma_automask'],
                                nsigma_autothreshold=global_parameters['nsigma_autothreshold'],
                                n_interaction='0', savemodel=False, quiet=quiet,
                                datacolumn='DATA', shift=FIELD_SHIFT,
                                with_multiscale=False, scales='0,5,20,40',
                                uvtaper=[''],
                                niter=niter,
                                PLOT=False)

                image_statistics,image_list = compute_image_stats(path=path,
                                                                  image_list=image_list,
                                                                  image_statistics=image_statistics,
                                                                  prefix=prefix,
                                                                  selfcal_step='test_image')


            if 'test_image' not in steps_performed:
                steps_performed.append('test_image')


        if params_trial_2 is not None:
            p0_params = params_trial_2['p0']
        else:
            try:
                # current_total_flux = image_statistics['test_image']['total_flux_mask'] * 1000
                selfcal_params = select_parameters(image_statistics['test_image']['total_flux_mask'] * 1000)
                parameter_selection['test_image'] = selfcal_params
                print('Initial Template of Parameters:',parameter_selection['test_image']['name'])
                p0_params = parameter_selection['test_image']['p0']
                print_table(p0_params)
            except:
                print('No test image found. Have you run the test_image step?')
            #

        if 'p0' in steps and 'p0' not in steps_performed:
            iteration = '0'
            ############################################################################
            #### 0. Zero interaction. Use a small/negative robust parameter,        ####
            ####    to find the bright/compact emission(s).                         ####
            ############################################################################

            if modified_robust is not None:
                p0_params['robust'] = modified_robust

            minblperant = 3
            # combine='spw'
            if p0_params['combine'] == 'spw':
                p0_params['spwmap'] = get_spwmap(g_vis)

            print('Params that are currently being used:',parameter_selection['test_image']['name'])
            print_table(p0_params)

            if 'start_image' not in steps_performed:

                if image_statistics['test_image']['inner_flux_f'] > 0.5:
                    mask_grow_iterations = 2
                if image_statistics['test_image']['inner_flux_f'] < 0.5:
                    mask_grow_iterations = 3

                rms_mask = None#1 * image_statistics['test_image']['rms_box']

                # if image_statistics['test_image']['total_flux'] * 1000 > 100.0:
                """
                If the source is too bright, it may contains lots of artifacts for a robust 
                r = 0.5 (the initial test image), and those artifacts can be printed in the mask below. 
                So, we create a new test image with a lower robust parameter. 
                """
                prefix = 'test_image_0'
                run_wsclean(g_name, imsize=imsize, imsizey=imsizey, cell=cell,
                            robust=p0_params['robust'], base_name=prefix,
                            nsigma_automask=p0_params['nsigma_automask'], 
                            nsigma_autothreshold=p0_params['nsigma_autothreshold'],
                            n_interaction=iteration, savemodel=False, quiet=quiet,
                            datacolumn='DATA', shift=FIELD_SHIFT,
                            uvtaper=p0_params['uvtaper'],
                            scales = p0_params['scales'],
                            niter=niter,
                            PLOT=False)

                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix=prefix,
                                                                   selfcal_step='p0')

                mask_name = create_mask(image_list['test_image_0'],
                                        rms_mask=rms_mask,
                                        sigma_mask=p0_params['sigma_mask'],
                                        mask_grow_iterations=mask_grow_iterations)


                start_image(g_name, n_interaction=iteration,
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            # uvtaper=['0.05arcsec'],
                            delmodel=True,
                            # opt_args=' -multiscale -multiscale-scales 0 ',
                            nsigma_automask='5.0',
                            nsigma_autothreshold='2.0',
                            # next time probably needs to use 7.0 instead of 3.0
                            niter=niter, shift=FIELD_SHIFT,quiet=quiet,
                            uvtaper=p0_params['uvtaper'],
                            savemodel=True, mask=mask_name,PLOT=True,
                            robust=p0_params['robust'], datacolumn='DATA')


                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='start_image',
                                                                   selfcal_step='test_image')

                if 'start_image' not in steps_performed:
                    steps_performed.append('start_image')

            if 'select_refant' in steps and 'select_refant' not in steps_performed:
                print(' ++==> Estimating order of best referent antennas...')
                tablename_refant = os.path.dirname(g_name) + '/selfcal/find_refant.phase'
                refant = find_refant(msfile=g_vis, field='',
                                     tablename=tablename_refant)
                print(' ++==> Preferential reference antenna order = ', refant)
                steps_performed.append('select_refant')
            else:
                refant = ''


            # SNRs, percentiles_SNRs, caltable_int, caltable_3, caltable_inf = \
            #     check_solutions(g_name,
            #                     field, cut_off=p0_params['minsnr'],
            #                     n_interaction=iteration,
            #                     solnorm=solnorm,
            #                     combine=p0_params['combine'], spwmap=p0_params['spwmap'],
            #                     calmode=p0_params['calmode'], refant=refant,
            #                     gaintype=p0_params['gaintype'],
            #                     # interp='cubic,cubic',
            #                     gain_tables_selfcal=[],
            #                     return_solution_stats=True)

            if 'p0' not in steps_performed:
                gain_tables_selfcal_temp = self_gain_cal(g_name,
                                                         n_interaction=iteration,
                                                         minsnr=p0_params['minsnr'],
                                                         solint=p0_params['solint'],
                                                         flagbackup=True,
                                                         gaintype=p0_params['gaintype'],
                                                         combine=p0_params['combine'],
                                                         refant=refant,
                                                         refantmode = refantmode,
                                                         calmode=p0_params['calmode'],
                                                         spwmap = p0_params['spwmap'],
                                                        #  interp = 'cubicPD,'
                                                        #           'cubicPD',
                                                        #  interp='cubic,cubic',
                                                         # interp='linearPD,'
                                                         #        'linearflagrel',
                                                         action='apply',
                                                         PLOT=True,
                                                         gain_tables=[]
                                                         )

                run_wsclean(g_name, robust=p0_params['robust'],
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            base_name='selfcal_test_0',
                            nsigma_automask=p0_params['nsigma_automask'],
                            nsigma_autothreshold=p0_params['nsigma_autothreshold'],
                            n_interaction='', savemodel=False, quiet=quiet,
                            with_multiscale=p0_params['with_multiscale'],
                            datacolumn='CORRECTED_DATA',
                            uvtaper=p0_params['uvtaper'],
                            scales=p0_params['scales'],
                            niter=niter, shift=FIELD_SHIFT,
                            PLOT=False)
                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='selfcal_test_0',
                                                                   selfcal_step='p0')

                if params_trial_2 is None:
                    if image_statistics['selfcal_test_0']['total_flux_mask'] * 1000 < 10.0:
                        """
                        Check if using a taper will increase the flux density above 10 mJy.
                        If yes, the source will not considered as `very faint`, and we may attempt a
                        second phase-selfcal run with the template `faint` (i.e. `p1` will be executed).
                        If not, we will continue with the `very faint` template and will proceed to
                        `ap1`.
                        """
                        # modified_robust = robust + 0.5
                        print('Deconvolving image with a taper.')
                        run_wsclean(g_name, imsize=imsize, imsizey=imsizey, cell=cell,
                                    robust=0.5, base_name='selfcal_test_0',
                                    nsigma_automask=p0_params['nsigma_automask'],
                                    nsigma_autothreshold=p0_params['nsigma_autothreshold'],
                                    n_interaction='0', savemodel=False, quiet=quiet,
                                    datacolumn='CORRECTED_DATA', shift=FIELD_SHIFT,
                                    with_multiscale=p0_params['with_multiscale'],
                                    scales=p0_params['scales'],
                                    uvtaper=p0_params['uvtaper'],
                                    niter=niter,
                                    PLOT=False)
                        image_statistics, image_list = compute_image_stats(path=path,
                                                                           image_list=image_list,
                                                                           image_statistics=image_statistics,
                                                                           prefix='selfcal_test_0')



                # parameter_selection['p0_pos']['p0']['spwmap'] = p0_params['spwmap']


                trial_gain_tables.append(gain_tables_selfcal_temp)
                gain_tables_applied['p0'] = gain_tables_selfcal_temp
                steps_performed.append('p0')

        if params_trial_2 is not None:
            parameter_selection['p0_pos'] = params_trial_2
        else:
            try:
                selfcal_params = select_parameters(
                    image_statistics['selfcal_test_0']['total_flux_mask'] * 1000)
                parameter_selection['p0_pos'] = selfcal_params
                print(' ++++>> Template of Parameters to be used for now on:',
                      parameter_selection['p0_pos']['name'])
                if parameter_selection['p0_pos']['p0']['combine'] == 'spw':
                    parameter_selection['p0_pos']['p0']['spwmap'] = get_spwmap(g_vis)
            except:
                pass


        if (('p1' in steps) and ('p1' not in steps_performed) and
                ('p1' in parameter_selection['p0_pos'])):
            iteration = '1'
            # current_total_flux = image_statistics['selfcal_test_0']['total_flux_mask'] * 1000

            ############################################################################
            #### 1. First interaction. Increase a little the robust parameter,      ####
            ####    start to consider more extended emission.                       ####
            ############################################################################
            p1_params = parameter_selection['p0_pos']['p1']
            print('Params that are currently being used:', parameter_selection['p0_pos']['name'])
            print_table(p1_params)
            if 'update_model_1' not in steps_performed:
                # if params_trial_2 is not None:
                run_wsclean(g_name, robust=p1_params['robust'],
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            base_name='selfcal_test_0',
                            nsigma_automask=p1_params['nsigma_automask'],
                            nsigma_autothreshold=p1_params['nsigma_autothreshold'],
                            n_interaction='', savemodel=False, quiet=quiet,
                            with_multiscale=p1_params['with_multiscale'],
                            scales = p1_params['scales'],
                            datacolumn='CORRECTED_DATA',
                            uvtaper=p1_params['uvtaper'],
                            niter=niter, shift=FIELD_SHIFT,
                            PLOT=False)

                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='selfcal_test_0')

                mask_name = create_mask(image_list['selfcal_test_0'],
                                        rms_mask=rms_mask,
                                        sigma_mask=p1_params['sigma_mask'],
                                        mask_grow_iterations=mask_grow_iterations)

                run_wsclean(g_name, robust=p1_params['robust'],
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            nsigma_automask=p1_params['nsigma_automask'], 
                            nsigma_autothreshold=p1_params['nsigma_autothreshold'],
                            n_interaction=iteration, savemodel=True, quiet=quiet,
                            with_multiscale=p1_params['with_multiscale'],
                            scales=p1_params['scales'],
                            datacolumn='CORRECTED_DATA', mask=mask_name,
                            shift=FIELD_SHIFT,
                            uvtaper=p1_params['uvtaper'],
                            niter=niter,
                            PLOT=False)

                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='1_update_model_image')

                steps_performed.append('update_model_1')

            # current_snr = image_statistics['1_update_model_image']['snr']
            # if p1_params['compare_solints'] == True:
            #     SNRs, percentiles_SNRs, caltable_int, caltable_3, caltable_inf =  (
            #         check_solutions(g_name, field, cut_off=1.5,
            #                         n_interaction=iteration,
            #                         solnorm=solnorm,
            #                         combine=p1_params['combine'],
            #                         calmode=p1_params['calmode'], refant=refant,
            #                         gaintype=p1_params['gaintype'],
            #                         # interp='cubic,cubic',
            #                         gain_tables_selfcal=[],
            #                         # gain_tables_selfcal=gain_tables_applied['p0'],
            #                         return_solution_stats=True))
            minblperant = 3
            if p1_params['combine'] == 'spw':
                p1_params['spwmap'] = get_spwmap(g_vis)

            if params_trial_2 is not None:
                phase_tables = gain_tables_applied['p0'].copy()
            else:
                phase_tables = []
            if 'p1' not in steps_performed:
                gain_tables_selfcal_p1 = self_gain_cal(g_name,
                                                       n_interaction=iteration,
                                                       minsnr=p1_params['minsnr'],
                                                       solint=p1_params['solint'],
                                                       flagbackup=True,
                                                       gaintype=p1_params['gaintype'],
                                                       combine=p1_params['combine'],
                                                       refant=refant,
                                                       spwmap=p1_params['spwmap'],
                                                       calmode=p1_params['calmode'],
                                                       # interp='cubic,cubic',
                                                       action='apply',
                                                       PLOT=False,
                                                       gain_tables = phase_tables
                                                       # gain_tables=gain_tables_applied[
                                                       #     'p0'].copy()
                                                       )

                run_wsclean(g_name, robust=p1_params['robust'],
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            base_name='selfcal_test_1',
                            nsigma_automask=p1_params['nsigma_automask'],
                            nsigma_autothreshold=p1_params['nsigma_autothreshold'],
                            n_interaction='', savemodel=False, quiet=quiet,
                            with_multiscale=p1_params['with_multiscale'],
                            scales=p1_params['scales'],
                            datacolumn='CORRECTED_DATA',
                            uvtaper=p1_params['uvtaper'],
                            niter=niter, shift=FIELD_SHIFT,
                            PLOT=False)
                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='selfcal_test_1')

                trial_gain_tables.append(gain_tables_selfcal_p1)
                gain_tables_applied['p1'] = gain_tables_selfcal_p1
                steps_performed.append('p1')

        if (('p2' in steps) and ('p2' not in steps_performed) and
                ('p2' in parameter_selection['p0_pos'])):
            iteration = '2'
            ############################################################################
            #### 2. Second interaction. Increase more the robust parameter, or use  ####
            ####    uvtapering. Consider even more extended emission (if there is). ####
            ############################################################################
            # current_total_flux = image_statistics['1_update_model_image']['total_flux_mask'] * 1000
            # selfcal_params = select_parameters(current_total_flux)
            selfcal_params = parameter_selection['p0_pos']
            p2_params = selfcal_params['p2']

            print('Params that are currently being used:',parameter_selection['p0_pos']['name'])
            print_table(p2_params)


            if image_statistics['1_update_model_image']['inner_flux_f'] > 0.5:
                mask_grow_iterations = 2
            if image_statistics['1_update_model_image']['inner_flux_f'] < 0.5: # sign of
                # diffuse emission
                mask_grow_iterations = 3
            if 'update_model_2' not in steps_performed:
                run_wsclean(g_name, robust=p2_params['robust'],
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            base_name='selfcal_test_1',
                            nsigma_automask=p2_params['nsigma_automask'], 
                            nsigma_autothreshold=p2_params['nsigma_autothreshold'],
                            n_interaction='', savemodel=False, quiet=quiet,
                            with_multiscale=p2_params['with_multiscale'],
                            scales=p2_params['scales'],
                            datacolumn='CORRECTED_DATA',
                            uvtaper=p2_params['uvtaper'],
                            niter=niter, shift=FIELD_SHIFT,
                            PLOT=False)

                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='selfcal_test_1')
                mask_name = create_mask(image_list['selfcal_test_1'],
                                        rms_mask=rms_mask,
                                        sigma_mask=p2_params['sigma_mask'],
                                        mask_grow_iterations=mask_grow_iterations)

                run_wsclean(g_name, robust=p2_params['robust'],
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            nsigma_automask=p2_params['nsigma_automask'], 
                            nsigma_autothreshold=p2_params['nsigma_autothreshold'],
                            n_interaction=iteration, savemodel=True, quiet=quiet,
                            with_multiscale=p2_params['with_multiscale'],
                            scales=p2_params['scales'],
                            datacolumn='CORRECTED_DATA', mask=mask_name,
                            shift=FIELD_SHIFT,
                            uvtaper=p2_params['uvtaper'],
                            niter=niter,
                            PLOT=False)

                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='2_update_model_image')

                steps_performed.append('update_model_2')
            #
            # current_snr = image_statistics['2_update_model_image']['snr']
            # SNRs, percentiles_SNRs, caltable_int, caltable_3, caltable_inf =  (
            #     check_solutions(g_name, field, cut_off=cut_off,
            #                                          n_interaction=iteration,
            #                                          solnorm=solnorm, combine=combine,
            #                                          calmode=calmode,refant=refant,
            #                                          gaintype=gaintype,
            #                                          interp='cubic,cubic',
            #                                          gain_tables_selfcal=gain_tables_applied['p1'],
            #                                          return_solution_stats=True))
            minblperant = 3
            if 'p2' not in steps_performed:
                gain_tables_selfcal_p2 = self_gain_cal(g_name,
                                                       n_interaction=iteration,
                                                       minsnr=p2_params['minsnr'],
                                                       solint=p2_params['solint'],
                                                       flagbackup=True,
                                                       gaintype=p2_params['gaintype'],
                                                       combine=p2_params['combine'],
                                                       refant=refant,
                                                       # interp = 'cubic,cubic',
                                                       calmode=p2_params['calmode'],
                                                       action='apply',
                                                       PLOT=False,
                                                       gain_tables=gain_tables_applied[
                                                           'p1'].copy()
                                                       )
                trial_gain_tables.append(gain_tables_selfcal_p2)
                gain_tables_applied['p2'] = gain_tables_selfcal_p2
                steps_performed.append('p2')



        if (('ap1' in steps) and
                ('ap1' not in steps_performed) and
                ('ap1' in parameter_selection['p0_pos'].keys())):
            iteration = '3'
            ############################################################################
            #### 3. Third interaction. Increase more the robust parameter, or use  ####
            ####    uvtapering. Consider even more extended emission (if there is). ####
            ############################################################################
            # niter = 50000
            # robust = 1.0
            # try:
            #     current_total_flux = image_statistics['2_update_model_image']['total_flux_mask'] * 1000
            # except:
            #     current_total_flux = image_statistics[list(image_statistics.keys())[-1]]['total_flux_mask'] * 1000
            # selfcal_params = select_parameters(current_total_flux)
            
            vis_split_name_p = g_name + '_p_trial_1.ms'
            if not os.path.exists(vis_split_name_p):
                print(' ++==> Splitting data after phase-selfcal first trial...')
                split(vis=g_name + '.ms', outputvis=vis_split_name_p,
                      datacolumn='corrected', keepflags=True)


            if ('p1' in parameter_selection['p0_pos'].keys()) and ('p1' in steps):
                ref_step_spwmap = 'p1'
            else:
                ref_step_spwmap = 'p0'

            if (parameter_selection['p0_pos']['ap1']['combine'] == 'spw') and (
                    parameter_selection['p0_pos'][ref_step_spwmap]['spwmap'] != []):
                """
                If `combine = 'spw'` is set to be used with `ap1`, we need to check if it was also 
                used during `p0` or `p1`. If it was used, we need to reuse the `spwmap` from 
                `p0` or `p1` and append to the one to be used in `ap1`. 
                
                Note that if `p1` was ran, solutions from `p0` are ignored. We just need the 
                spwmap for one of them.
                """
                parameter_selection['p0_pos']['ap1']['spwmap'] = (
                    parameter_selection['p0_pos'][ref_step_spwmap]['spwmap'].copy())
                parameter_selection['p0_pos']['ap1']['spwmap'].append(get_spwmap(g_vis)[0])

            if (parameter_selection['p0_pos']['ap1']['combine'] == '') and (
                    parameter_selection['p0_pos'][ref_step_spwmap]['spwmap'] != []):
                parameter_selection['p0_pos']['ap1']['spwmap'] = (
                    parameter_selection['p0_pos'][ref_step_spwmap]['spwmap'].copy())
                parameter_selection['p0_pos']['ap1']['spwmap'].append([])


            selfcal_params = parameter_selection['p0_pos']
            ap1_params = selfcal_params['ap1']
            print('Params that are currently being used:',parameter_selection['p0_pos']['name'])
            print_table(ap1_params)

            if 'update_model_3' not in steps_performed:
                run_wsclean(g_name, robust=ap1_params['robust'],
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            base_name='selfcal_test_2',
                            nsigma_automask=ap1_params['nsigma_automask'], 
                            nsigma_autothreshold=ap1_params['nsigma_autothreshold'],
                            n_interaction='', savemodel=False, quiet=quiet,
                            with_multiscale=ap1_params['with_multiscale'],
                            scales=ap1_params['scales'],
                            datacolumn='CORRECTED_DATA',
                            uvtaper=ap1_params['uvtaper'],
                            niter=niter, shift=FIELD_SHIFT,
                            PLOT=False)

                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='selfcal_test_2')
                mask_name = create_mask(image_list['selfcal_test_2'],
                                        rms_mask=rms_mask,
                                        sigma_mask=ap1_params['sigma_mask'],
                                        mask_grow_iterations=mask_grow_iterations)


                if image_statistics['selfcal_test_2']['inner_flux_f'] > 0.5:
                    mask_grow_iterations = 2
                if image_statistics['selfcal_test_2']['inner_flux_f'] < 0.5:
                    mask_grow_iterations = 3

                run_wsclean(g_name, robust=ap1_params['robust'],
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            nsigma_automask=ap1_params['nsigma_automask'], 
                            nsigma_autothreshold=ap1_params['nsigma_autothreshold'],
                            n_interaction=iteration, savemodel=True, quiet=quiet,
                            with_multiscale=ap1_params['with_multiscale'],
                            scales=ap1_params['scales'],
                            datacolumn='CORRECTED_DATA', mask=mask_name,
                            shift=FIELD_SHIFT,
                            uvtaper=ap1_params['uvtaper'],
                            niter=niter,
                            PLOT=True)

                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='3_update_model_image')

                steps_performed.append('update_model_3')

            print(' ++==>> Preaparing for amp-gains....')
            # SNRs, percentiles_SNRs, caltable_int, caltable_3, caltable_inf  =  (
            #     check_solutions(g_name, field, cut_off=cut_off,
            #                                          n_interaction=iteration,
            #                                          solnorm=solnorm,
            #                                          combine=combine, spwmap = spwmap,
            #                                          calmode=calmode,refant=refant,
            #                                          gaintype=gaintype,
            #                                          interp='cubic,cubic',
            #                                          gain_tables_selfcal=phase_tables,
            #                                          return_solution_stats=True))
            minblperant = 4

            if params_trial_2 is not None:
                phase_tables = gain_tables_applied['p1'].copy()
            else:
                if 'p2' not in gain_tables_applied:
                    if 'p1' not in gain_tables_applied:
                        phase_tables = gain_tables_applied['p0']
                    else:
                        phase_tables = gain_tables_applied['p1']
                else:
                    phase_tables = gain_tables_applied['p2']

            if 'ap1' not in steps_performed:
                gain_tables_selfcal_ap1 = self_gain_cal(g_name,
                                                       n_interaction=iteration,
                                                       minsnr=ap1_params['minsnr'],
                                                       solint=ap1_params['solint'],
                                                       flagbackup=True,
                                                       gaintype=ap1_params['gaintype'],
                                                       combine=ap1_params['combine'],
                                                       refant=refant,
                                                       spwmap=ap1_params['spwmap'],
                                                       # interp='cubicPD,'
                                                       #        'cubicPD',
                                                       # interp = 'cubic,cubic',
                                                       calmode=ap1_params['calmode'],
                                                       action='apply',
                                                       PLOT=True,
                                                       gain_tables=phase_tables.copy()
                                                       )

                do_ap_inf = False
                if do_ap_inf == True:
                    ap1spwmap_new = ap1_params['spwmap']
                    ap1spwmap_new.append(ap1_params['spwmap'][-1])
                    gain_tables_selfcal_ap1 = self_gain_cal(g_name,
                                                        n_interaction=iteration,
                                                        minsnr=ap1_params['minsnr'],
                                                        solint='inf',
                                                        flagbackup=True,
                                                        gaintype=ap1_params['gaintype'],
                                                        combine=ap1_params['combine'],
                                                        refant=refant,
                                                        solnorm=False,
                                                        spwmap=ap1spwmap_new,
                                                        calmode=ap1_params['calmode'],
                                                        action='calculate',
                                                        PLOT=False,
                                                        gain_tables=gain_tables_selfcal_ap1.copy()
                                                        )

                trial_gain_tables.append(gain_tables_selfcal_ap1)
                gain_tables_applied['ap1'] = gain_tables_selfcal_ap1
                steps_performed.append('ap1')


        if 'split_trial_1' in steps and 'split_trial_1' not in steps_performed:
            vis_split_name_1 = g_name + '_trial_1.ms'
            if not os.path.exists(vis_split_name_1):
                print(' ++==> Splitting data after selfcal...')
                split(vis=g_name + '.ms', outputvis=vis_split_name_1,
                      datacolumn='corrected', keepflags=True)
                print(' ++==> Running statw on split data...')
                statwt(vis=vis_split_name_1, statalg='chauvenet', timebin='24s',
                       datacolumn='data')
            niter = 150000
            ROBUSTS = [-0.5,0.5] #[-2.0,0.0,0.5,1.0]
            print(' ++==> Imaging visibilities after first trial of selfcal...')
            for robust in ROBUSTS:
                run_wsclean(g_name + '_trial_1', robust=robust,
                            imsize=imsize, imsizey=imsizey, cell=cell,
                            base_name='selfcal_image',
                            nsigma_automask = global_parameters['nsigma_automask'],
                            nsigma_autothreshold = global_parameters['nsigma_autothreshold'],
                            n_interaction='', savemodel=False, quiet=quiet,
                            with_multiscale=global_parameters['with_multiscale'],
                            scales=global_parameters['scales'],
                            datacolumn='DATA', shift=FIELD_SHIFT,
                            uvtaper=global_parameters['uvtaper'],
                            niter=niter,
                            PLOT=False)

                image_statistics, image_list = compute_image_stats(path=path,
                                                                   image_list=image_list,
                                                                   image_statistics=image_statistics,
                                                                   prefix='selfcal_image')

            steps_performed.append('split_trial_1')

        if 'report_results' in steps:
            """
            To do: save and plot the results of the selfcalibration.
            """
            snr = [image_statistics[image]['snr'] for image in image_statistics]

            df = pd.DataFrame.from_dict(image_statistics, orient='index')



            df.to_csv(g_name + '_selfcal_statistics.csv',
                      header = True,
                      index = False)
            df_gt = pd.DataFrame.from_dict(gain_tables_applied, orient='index')
            df_gt.to_csv(g_name + '_tables_applied.csv',
                      header = True,
                      index = False)

            # try:
            # df_selfcal[['DR_SNR_E', 'DR_pk_rmsbox', 'snr', 'peak_of_flux', 'total_flux']]
            try:
                df_new = df[
                    ['snr', 'peak_of_flux', 'total_flux', 'rms_box', 'DR_pk_rmsim']]
                df_new = df_new.T
                df_new = df_new[
                    ['start_image', '1_update_model_image', '3_update_model_image', 'selfcal_image']]
                for col in df_new.columns:
                    df_new[col] = df_new[col].map("{:.2e}".format)
            except:
                df_new = df[
                    ['snr', 'peak_of_flux', 'total_flux', 'rms_box', 'DR_pk_rmsim']]
                df_new = df_new.T
                df_new = df_new[
                    ['start_image', 'selfcal_test_0', '3_update_model_image',
                     'selfcal_image']]
                for col in df_new.columns:
                    df_new[col] = df_new[col].map("{:.2e}".format)
            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(5, 6)

            ax_snr = plt.subplot(gs[0, 0:2])
            ax_snr.scatter(df['snr'], df['peak_of_flux'], label='SNR vs Sp')
            ax_snr.set_title('SNR vs Sp')

            ax_flux = plt.subplot(gs[1, 0:2])
            ax_flux.scatter(df['total_flux'] * 1000, df['peak_of_flux'], label='Flux vs Sp')
            ax_flux.set_title('Flux vs Sp')

            ax_dr_rms = plt.subplot(gs[2, 0:2])
            ax_dr_rms.scatter(df['DR_pk_rmsbox'], df['rms_box'] * 1000, label='DR vs RMS')
            ax_dr_rms.set_title('DR vs RMS')

            ax_table = plt.subplot(gs[0:3, 2:6])
            ax_table.axis('off')  # Hide the axis
            # table_data = df_.set_index('Statistics').T

            table = ax_table.table(
                cellText=df_new.values,
                colLabels=df_new.columns,
                rowLabels=df_new.index,
                loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(0.5, 0.9)


            def plot_stage_images(image, model, residual, axes_index, fig, gs, phase, rms,
                                  centre,
                                  crop=True):
                ax_image = plt.subplot(gs[3:5, axes_index:axes_index + 2])
                # phase = 'AP'
                ax_image = mlibs.eimshow(image,
                                         rms=None, add_contours=True, center=centre,
                                         ax=ax_image, crop=crop, box_size=400)
                ax_image = ax_image
                # ax_image.imshow(image_placeholder)
                ax_image.set_title(f'{phase}_IMAGE')
                # ax_image.axis('off')


            rms = mlibs.mad_std(mlibs.ctn(image_list['selfcal_image_residual']))
            centre = mlibs.nd.maximum_position(
                mlibs.ctn(image_list['selfcal_image']))
            centre = (centre[1], centre[0])

            plot_stage_images(image=image_list['selfcal_test_0'],
                              model=image_list['selfcal_test_0_model'],
                              residual=image_list['selfcal_test_0_residual'],
                              axes_index=0,
                              fig=fig,
                              gs=gs,
                              phase='O',
                              rms=rms,
                              centre=centre
                              )
            try:
                plot_stage_images(image=image_list['2_update_model_image'],
                                  model=image_list['2_update_model_image_model'],
                                  residual=image_list['2_update_model_image_residual'],
                                  axes_index=2,
                                  fig=fig,
                                  gs=gs,
                                  phase='P',
                                  rms=rms,
                                  centre=centre
                                  )
            except:
                plot_stage_images(image=image_list['3_update_model_image'],
                                  model=image_list['3_update_model_image_model'],
                                  residual=image_list['3_update_model_image_residual'],
                                  axes_index=2,
                                  fig=fig,
                                  gs=gs,
                                  phase='P',
                                  rms=rms,
                                  centre=centre
                                  )
            plot_stage_images(image=image_list['selfcal_image'],
                              model=image_list['selfcal_image_model'],
                              residual=image_list['selfcal_image_residual'],
                              axes_index=4,
                              fig=fig,
                              gs=gs,
                              phase='AP',
                              rms=rms,
                              centre=centre
                              )

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.tight_layout()
            plt.savefig(g_name + '_selfcal_results.pdf', dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()
            # plt.show()
            # except:
            #     pass

            pass


        if 'run_rflag_final' in steps:
            run_rflag(g_vis, display='report', action='apply',
                      timedevscale=2.5, freqdevscale=2.5, winsize=5,
                      datacolumn='corrected')

            vis_split_name_2 = g_name + '_trial_1_pos_rflag.ms'
            if not os.path.exists(vis_split_name_2):
                print(' ++==> Splitting data after rflag first trial...')
                split(vis=g_name + '.ms', outputvis=vis_split_name_2,
                      datacolumn='corrected', keepflags=True)
            statwt(vis=vis_split_name_2, statalg='chauvenet', timebin='24s',
                   datacolumn='data')

            niter = 150000
            robust = 0.5
            run_wsclean(vis_split_name_2, robust=robust,
                        imsize=global_parameters['imsize'],
                        imsizey=global_parameters['imsizey'],
                        cell=global_parameters['cell'],
                        base_name='selfcal_image_pos_rflag',
                        nsigma_automask='4.0',
                        nsigma_autothreshold='2.0',
                        n_interaction='', savemodel=False, quiet=True,
                        with_multiscale=True,
                        scales='None',
                        datacolumn='DATA', shift=FIELD_SHIFT,
                        # uvtaper=['0.1arcsec'],
                        niter=niter,
                        PLOT=False)

            image_statistics, image_list = compute_image_stats(path=path,
                                                               image_list=image_list,
                                                               image_statistics=image_statistics,
                                                               prefix='selfcal_image_pos_rflag')

            plot_visibilities(g_vis=vis_split_name_2, name='selfcal_image_pos_rflag',
                              with_MODEL=False, with_CORRECTED=True)

            steps_performed.append('run_rflag_final')



if run_mode == 'jupyter':
    print('selfcal script is not doing anything, you can use it on a jupyter notebook. '
          'For that, you have to manually set your variable names and steps taken '
          'to selfcalibrate your data.')
