"""
Methods for high level analysis of fingerprints - feature detection, classification, etc
author: DP, Gaussian fitting module from SD
date: 10/10/2017
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import os
import math
import peakutils
import pickle
import tkinter
from tkinter import filedialog
import scipy.signal


class Gaussian(object):
    """
    Container for fitted gaussian information. Holds fit parameters and any associated metadata
    """
    def __init__(self, baseline, amplitude, centroid, width, collision_voltage):
        self.baseline = baseline
        self.amplitude = amplitude
        self.centroid = centroid
        self.width = width
        self.cv = collision_voltage
        self.fwhm = 2*(math.sqrt(2*math.log(2)))*self.width
        self.resolution = self.centroid/self.fwhm

    def __str__(self):
        return 'Gaussian: x0={:.2f} A={:.1f} w={:.1f} cv={}'.format(self.centroid,
                                                                    self.amplitude,
                                                                    self.width,
                                                                    self.cv)
    # set repr = str for printing in lists
    __repr__ = __str__

    def print_info(self):
        """
        Method for generating strings to save to output files with all info
        :return: string
        """
        return '{},{:.2f},{:.2f},{:.1f},{:.1f},{:.1f},{:.1f}'.format(self.cv,
                                                                     self.centroid,
                                                                     self.amplitude,
                                                                     self.width,
                                                                     self.baseline,
                                                                     self.fwhm,
                                                                     self.resolution)


def multi_gauss_func(x, *params):
    """
    Attempt at basic multi-gaussian fitting by passing multiple parameter sets and generating a sum
    of gaussians
    :param x: data to fit
    :param params: List of [amplitdue, centroid x, width] parameters, one set for each Gaussian to fit (in order)
    :return: y = f(x), where f(x) describes a gaussian function
    """
    y = np.zeros_like(x)
    # make a gaussian function for each set of parameters in the input list
    for i in range(0, len(params), 4):
        y = y + gaussfunc(x, params[i], params[i+1], params[i+2], params[i+3])
    return y


def estimate_multi_params(ciu_col, dt_axis, width_frac, peak_int_threshold=0.1, min_spacing_bins=10):
    """
    Make initial guesses for a sum of gaussians fitting
    :param ciu_col: 1D numpy array representing the DT spectrum in a given column (CV)
    :param dt_axis: drift time data (x axis to the fitted gaussian's y) for peak indexing
    :param width_frac: estimation of peak width (DT * fraction), typically 10% has been found to work well
    :param peak_int_threshold: Minimum intensity threshold to detect a peak for fitting
    :param min_spacing_bins: Minimum distance between peaks IN BINS - should be about instrument resolution
    :return: list of [centroid, width, amplitude] initial guesses
    """
    # estimate the number of components by doing a simple peak finding using PeakUtils
    peak_indices = peakutils.indexes(ciu_col, thres=peak_int_threshold, min_dist=min_spacing_bins)

    params_lists = []
    # for each estimated peak/component, compute initial guess parameters for gaussian fitting
    for peak_index in peak_indices:
        centroid_guess = dt_axis[peak_index]    # centroid is the DT at the index of the peak
        amp_guess = ciu_col[peak_index]         # amplitude is the value at the index of the peak
        width_guess = peak_index * width_frac
        params_lists.extend([0, amp_guess, centroid_guess, width_guess])
        # params_lists.append([0, centroid_guess, amp_guess, width_guess])
    return params_lists


def estimate_multi_params_all(ciu_col, dt_axis, width_frac):
    """
    Make initial guesses for peak locations, but ensure overestimation. SciPy find_peaks_cwt tends to
    way overestimate peaks (at least if a small width range is provided), but the fitting algorithm works
    by passing increasing numbers of peaks until the fit converges, so we need to have a large number of
    peaks to provide.
    :param ciu_col: 1D numpy array representing the DT spectrum in a given column (CV)
    :param dt_axis: drift time data (x axis to the fitted gaussian's y) for peak indexing
    :param width_frac: estimation of peak width (DT * fraction), typically 10% has been found to work well
    :return: list of [centroid, width, amplitude] initial guesses
    """
    # estimate the number of components by doing a simple peak finding using CWT - since it tends to give lots of peaks
    peak_ind_scipy = scipy.signal.find_peaks_cwt(ciu_col, np.arange(1, 5))

    params_lists = []
    # for each estimated peak/component, compute initial guess parameters for gaussian fitting
    for peak_index in peak_ind_scipy:
        centroid_guess = dt_axis[peak_index]    # centroid is the DT at the index of the peak
        amp_guess = ciu_col[peak_index]         # amplitude is the value at the index of the peak
        width_guess = peak_index * width_frac
        params_lists.append([0.001, amp_guess, centroid_guess, width_guess])

    # sort guesses by amplitude (index 1 in each sublist) in order from largest to smallest
    params_lists = sorted(params_lists, key=lambda x: x[1], reverse=True)
    return params_lists


def gaussfunc(x, y0, a, xc, w):
    """
    Gaussian function with constraints applied for CIU data
    :param x: x
    :param y0: baseline (set to 0)
    :param a: gaussian amplitude (constrained to be positive)
    :param xc: gaussian centroid
    :param w: gaussian width
    :return: y = f(x)
    """
    # y0 = 0
    # a = abs(a)
    # w = abs(w)
    # xc = abs(xc)
    rxc = ((x-xc)**2)/(2*(w**2))
    # y = y0 + a*(np.exp(-rxc))
    y = a*(np.exp(-rxc))
    return y


def resandfwhm(centroids, widths):
    """
    Compute FWHM (full width at half max) and resolution for peak centroid/width combinations.
    Requires inputs to be lists (can have a single entry) to allow for multiple peak fittings
    :param centroids: LIST of centroids
    :param widths: LIST of widths (must be same length as list of centroids)
    :return: list of FWHMs, list of resolutions (both same length as input lists)
    """
    fwhm_list = []
    res_list = []
    for xc, w in zip(centroids, widths):
        fwhm = 2*(math.sqrt(2*math.log(2)))*w
        fwhm_list.append(fwhm)
        res_list.append(xc/fwhm)
    return fwhm_list, res_list


def adjrsquared(r2, num):
    y = 1 - (((1-r2)*(num-1))/(num-4-1))
    return y


def filter_fits(params_list, peak_width_cutoff, intensity_cutoff, centroid_bounds=None):
    """
    Simple filter to remove any peaks with a width above a specified cutoff. Intended to separate
    noise 'peaks' from protein peaks as they differ in observed width
    :param params_list: list of optimized parameters from curve fit
    :param peak_width_cutoff: maximum allowed width for a peak to remain in the list
    :param intensity_cutoff: minimum relative intensity to remain in the list
    :param centroid_bounds: list of [lower bound, upper bound] for peak centroid (in ms)
    :return: filtered params_list, with peaks above the width cutoff removed
    """
    index = 0
    filtered_list = []
    while index < len(params_list):
        # test if the peak meets all conditions for inclusion
        include_peak = False

        # ensure peak width is below the cutoff and above 0
        if 0 < params_list[index + 3] < peak_width_cutoff:
            # also remove amplitdues below the intensity cutoff
            if params_list[index + 1] > intensity_cutoff:
                if centroid_bounds is not None:
                    # centroid bounds provided - if matched, include the peak
                    if centroid_bounds[0] < params_list[index + 2] < centroid_bounds[1]:
                        include_peak = True
                elif params_list[index + 2] > 0:
                    # If no bounds provided lso remove centroids < 0
                    include_peak = True

        if include_peak:
            filtered_list.extend(params_list[index:index + 4])
        index += 4
    return filtered_list


def gaussian_fit_ciu(analysis_obj, params_obj):
    """
    Gaussian fitting module for single-gaussian analysis of CIU-type data. Determines estimated
    initial parameters and fits a single Gaussian distribution to each column of the input ciu_data
    matrix. Saves all output into a subfolder of filepath titled 'gausfitoutput'
    :param analysis_obj: ciu analysis object in which to save all data. Uses obj.ciu_data, axes, etc
     (ciu_data is 2D numpy, axis 0 = DT, axis 1 = CV) for all data handling
     Smoothing, interpolation, cropping, etc should be done prior to running this method.
    :param params_obj: Parameters object with gaussian parameter information
    :return: saves all gaussian outputs into the ciu_obj and returns it
    """
    ciu_data = analysis_obj.ciu_data
    dt_axis = analysis_obj.axes[0]    # drift time (DT) - x axis for fitting, y axis for final CIU plot
    cv_axis = analysis_obj.axes[1]
    filename = analysis_obj.filename

    widthfrac = params_obj.gaussian_width_fraction
    # min_spacing = params_obj.gaussian_min_spacing
    filter_width_max = params_obj.gaussian_width_max
    intensity_thr = params_obj.gaussian_int_threshold
    centroid_bounds = params_obj.gaussian_centroid_bound_filter
    centroid_plot_bounds = params_obj.gaussian_centroid_plot_bounds

    outputpathdir = filename.rstrip('.ciu')
    outputpath = os.path.join(os.path.dirname(analysis_obj.filename), outputpathdir)
    if not os.path.isdir(outputpath):
        os.makedirs(outputpath)

    intarray = np.swapaxes(ciu_data, 0, 1)
    popt_arr, pcov_arr, fwhm_arr, res_arr, arrivtime_centroid, stats,  = [], [], [], [], [], []
    filtered_params, arrivtime_gausfit, width_gausfit, rsq_arr, adjrsq_arr = [], [], [], [], []

    gaussians = []
    filtered_gaussians = []
    # for each CV column in the file, perform a multi-gaussian fitting and save information generated
    print('\nFile ' + str(filename))
    for cv_index, cv_col_intensities in enumerate(intarray):
        print(cv_index + 1)
        # use peak detection to estimate initial 'guess' parameters for fitting
        all_peak_guesses = estimate_multi_params_all(cv_col_intensities, dt_axis, widthfrac)

        param_guesses_multiple = []
        yfit = 0
        slope, intercept, rvalue, pvalue, stderr = 0, 0, 0, 0, 0
        adjrsq = 0
        i = 0
        # set bounds for fitting: keep baseline and centroid on DT axis, amplitude 0 to 1.5, width 0 to len(dt_axis)
        max_dt = dt_axis[len(dt_axis) - 1]
        min_dt = dt_axis[0]
        fit_bounds_lower, fit_bounds_upper = [], []
        fit_bounds_lower_append = [0, 0, min_dt, 0]
        fit_bounds_upper_append = [max_dt, 1.1, max_dt, len(dt_axis)]

        # Iterate through peak detection until convergence criterion is met, adding one additional peak each iteration
        while adjrsq < 0.995:
            try:
                param_guesses_multiple.extend(all_peak_guesses[i])
                # ensure bounds arrays maintain same shape as parameter guesses
                fit_bounds_lower.extend(fit_bounds_lower_append)
                fit_bounds_upper.extend(fit_bounds_upper_append)
            except IndexError:
                # No converge with all estimated peaks. Continue with final estimate
                print('Included all {} peaks found, but r^2 still less than convergence criterion. '
                      'Poor fitting possible'.format(i+1))
                break
            try:
                popt, pcov = curve_fit(multi_gauss_func, dt_axis, cv_col_intensities, method='trf',
                                       p0=param_guesses_multiple, maxfev=5000,
                                       bounds=(fit_bounds_lower, fit_bounds_upper))
            except RuntimeError:
                popt = []
                pcov = []
            yfit = multi_gauss_func(dt_axis, *popt)
            slope, intercept, rvalue, pvalue, stderr = linregress(cv_col_intensities, yfit)
            adjrsq = adjrsquared(rvalue ** 2, len(cv_col_intensities))
            i += 1
            # if i > 10:
            #     break
        # filter peaks by width if desired
        print('performed {} iterations'.format(i))
        filt_popt = popt
        if filter_width_max is not None:
            filt_popt = filter_fits(popt, filter_width_max, intensity_thr, centroid_bounds)
        filtered_params.append(filt_popt)

        # save Gaussian information to container objects
        index = 0
        gaussians_at_this_cv = []
        while index < len(popt):
            gaussians_at_this_cv.append(Gaussian(popt[index], popt[index+1], popt[index+2], popt[index+3], cv_axis[cv_index]))
            index += 4
        gaussians.append(gaussians_at_this_cv)
        index = 0
        filt_gaussians_at_cv = []
        while index < len(filt_popt):
            filt_gaussians_at_cv.append(Gaussian(filt_popt[index], filt_popt[index + 1], filt_popt[index + 2], filt_popt[index + 3], cv_axis[cv_index]))
            index += 4
        filtered_gaussians.append(filt_gaussians_at_cv)

        rsq_arr.append(rvalue ** 2)
        adjrsq_arr.append(adjrsq)
        arrivtime_gausfit.append(yfit)

        stats.append([slope, intercept, rvalue ** 2, adjrsq, pvalue, stderr])
        popt_arr.append(popt)
        pcov_arr.append(pcov)

    # save fit information to the analysis object and return it
    analysis_obj.gaussians = gaussians
    analysis_obj.filtered_gaussians = filtered_gaussians
    analysis_obj.gauss_adj_r2s = adjrsq_arr
    analysis_obj.gauss_fits = arrivtime_gausfit
    analysis_obj.gauss_r2s = rsq_arr
    analysis_obj.gauss_covariances = pcov_arr
    analysis_obj.gauss_fit_stats = stats

    if params_obj.gaussian_save_diagnostics:
        analysis_obj.save_gaussfits_pdf(outputpath)
        analysis_obj.plot_centroids(outputpath, centroid_plot_bounds)
        analysis_obj.plot_fwhms(outputpath)
        analysis_obj.save_gauss_params(outputpath)

    return analysis_obj


if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
    for file in files:
        with open(file, 'rb') as analysis_file:
            current_analysis_obj = pickle.load(analysis_file)
        gaussian_fit_ciu(current_analysis_obj, current_analysis_obj.params)
