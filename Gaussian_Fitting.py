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
import matplotlib.pyplot as plt


def multi_gauss_func(x, *params):
    """
    Attempt at basic multi-gaussian fitting by passing multiple parameter sets and generating a sum
    of gaussians
    :param x: data to fit
    :param params: List of [amplitdue, centroid x, width] parameters, one set for each Gaussian to fit (in order)
    :return: y = f(x), where f(x) describes a gaussian function
    """
    # y = np.zeros_like(x)
    # for init_guess_params in params_lists:
    #     y = y + gaussfunc(x, *init_guess_params)
    # return y

    y = np.zeros_like(x)
    # make a gaussian function for each set of parameters in the input list
    for i in range(0, len(params), 4):
        y = y + gaussfunc(x, params[i], params[i+1], params[i+2], params[i+3])
        # centroid = params[i]
        # amplitude = params[i + 1]
        # width = params[i + 2]
        # rxc = ((x - centroid) ** 2) / (2 * (width ** 2))
        # y = y + amplitude * (np.exp(-rxc))

        # y = y + amplitude * np.exp(-((x - centroid) / width) ** 2)
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


def gaussfunc(x, y0, A, xc, w):
    A = abs(A)
    rxc = ((x-xc)**2)/(2*(w**2))
    y = y0 + A*(np.exp(-rxc))
    return y


# def gaussfunc2(x, A, xc, w):
#     rxc = ((x - xc) ** 2) / (2 * (w ** 2))
#     y = A * (np.exp(-rxc))
#     return y


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


# def estimateparam(array):
#     ymax = np.max(array)
#     maxindex = np.nonzero(array == ymax)[0]
#     binsum = np.nonzero(array)
#     widthbin = len(binsum[0])
#     return maxindex, widthbin, ymax
#
#
# def estimateparam2(array):
#     ymax = np.max(array)
#     maxindex = np.nonzero(array == ymax)[0]
#     binsum = np.nonzero(array)
#     widthbin = len(binsum[0])
#     return [0, maxindex, widthbin, ymax]


def gaussian_fit_ciu(ciu_obj, widthfrac=0.01, intensity_thr=0.1, min_spacing=10):
    """
    Gaussian fitting module for single-gaussian analysis of CIU-type data. Determines estimated
    initial parameters and fits a single Gaussian distribution to each column of the input ciu_data
    matrix. Saves all output into a subfolder of filepath titled 'gausfitoutput'
    :param ciu_obj: ciu analysis object in which to save all data. Uses obj.ciu_data, axes, etc
     (ciu_data is 2D numpy, axis 0 = DT, axis 1 = CV) for all data handling
     Smoothing, interpolation, cropping, etc should be done prior to running this method.
    :param widthfrac: width fraction estimation for fitting, default 0.01
    :param intensity_thr: minimum intensity threshold for peak picking, default 10% (0.1)
    :param min_spacing: minimum spacing between fitted peaks IN DRIFT BINS. This should be adjusted
    to approximately the instrument resolution for a given max DT. (To be implemented in future)
    :return: saves all gaussian outputs into the ciu_obj and returns it
    """
    ciu_data = ciu_obj.ciu_data
    dt_axis = ciu_obj.axes[0]    # drift time (DT) - x axis for fitting, y axis for final CIU plot
    filename = ciu_obj.raw_obj.filename

    outputpathdir = filename.rstrip('_raw.csv')
    outputpath = os.path.join(os.path.dirname(ciu_obj.raw_obj.filepath), outputpathdir)
    if not os.path.isdir(outputpath): os.makedirs(outputpath)

    intarray = np.swapaxes(ciu_data, 0, 1)
    popt_arr = []
    pcov_arr = []
    fwhm_arr = []
    res_arr = []
    arrivtime_centroid = []
    stats = []
    arrivtime_gausfit = []
    width_gausfit = []
    rsq_arr = []
    adjrsq_arr = []
    print('\nFile ' + str(filename))
    for i, array in enumerate(intarray):
        print(i + 1)
        # param_guesses_multiple = estimateparam2(array)
        param_guesses_multiple = estimate_multi_params(array, dt_axis, widthfrac, peak_int_threshold=intensity_thr,
                                                       min_spacing_bins=min_spacing)
        # param_guesses_multiple = [[]]
        # peak_index = 0
        # while peak_index < len(param_guesses_multiple):
        #     param_guesses = param_guesses_multiple[peak_index:peak_index + 4]
        #     peak_index += 4
        #     popt, pcov = curve_fit(multi_gauss_func, dt_axis, array, method='lm',
        #                            p0=param_guesses, maxfev=5000)
        try:
            popt, pcov = curve_fit(multi_gauss_func, dt_axis, array, method='lm',
                                   p0=param_guesses_multiple, maxfev=5000)
        except RuntimeError:
            # no convergence within specified max iterations
            popt = []
            pcov = []
        # extract individual gaussian function parameters
        baselines = popt[::4]
        amplitudes = popt[1::4]
        centroids = popt[2::4]
        widths = popt[3::4]
        index = 0
        # plt.plot(dt_axis, array)
        while index < len(baselines):
            peakfit = gaussfunc(dt_axis, baselines[index], amplitudes[index], centroids[index], widths[index])
            index += 1
            # plt.plot(dt_axis, peakfit)
        # td_estimate_index, w_estimate_indexlen, amp = estimateparam(array)
        # popt, pcov = curve_fit(gaussfunc, dt_axis, array, method='lm',
        #                        p0=[0, amp, dt_axis[td_estimate_index], widthfrac * w_estimate_indexlen],
        #                        maxfev=5000)

        # fit = multi_gauss_func(dt_axis, *popt)
        # plt.plot(dt_axis, array)
        # plt.plot(dt_axis, fit, 'r-')
        # plt.show()

        # out_params = np.split(popt, len(param_guesses_multiple))

        yfit = multi_gauss_func(dt_axis, *popt)
        fwhm, res = resandfwhm(centroids, widths)
        print('Arrival time | Width')
        print(centroids, ' | ', widths)
        print('FWHM | Resolution')
        print(fwhm, ' | ', res)
        residual = array - yfit
        slope, intercept, rvalue, pvalue, stderr = linregress(array, yfit)
        adjrsq = adjrsquared(rvalue ** 2, len(array))
        print('Slope | intercept | r^2 | adjr^2 | pvalue | stderr')
        print('%.4f | %.4f | %.4f | %.4f | %.4e | %.4f' % (slope, intercept, rvalue ** 2, adjrsq, pvalue, stderr))
        rsq_arr.append(rvalue ** 2)
        adjrsq_arr.append(adjrsq)
        arrivtime_gausfit.append(yfit)
        arrivtime_centroid.append(centroids)
        width_gausfit.append(widths)
        fwhm_arr.append(fwhm)
        res_arr.append(res)
        stats.append([slope, intercept, rvalue ** 2, adjrsq, pvalue, stderr])
        popt_arr.append(popt)
        pcov_arr.append(pcov)

    # save fit information to the analysis object and return it
    ciu_obj.init_gauss_lists(popt_arr)
    ciu_obj.gauss_adj_r2s = adjrsq_arr
    ciu_obj.gauss_fwhms = fwhm_arr
    ciu_obj.gauss_fits = arrivtime_gausfit
    ciu_obj.gauss_resolutions = res_arr
    ciu_obj.gauss_r2s = rsq_arr
    ciu_obj.gauss_covariances = pcov_arr
    ciu_obj.gauss_fit_stats = stats

    # save output
    ciu_obj.save_gaussfits_pdf(outputpath)
    ciu_obj.plot_centroids(outputpath)
    ciu_obj.plot_fwhms(outputpath)
    ciu_obj.save_gauss_params(outputpath)
    print('Job completed')
    return ciu_obj
