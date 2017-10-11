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


def gaussfunc(x, y0, A, xc, w):
    rxc = ((x-xc)**2)/(2*(w**2))
    y = y0 + A*(np.exp(-rxc))
    return y


def resandfwhm(xc, w):
    fwhm = 2*(math.sqrt(2*math.log(2)))*w
    res = xc/fwhm
    return fwhm, res


def adjrsquared(r2, num):
    y = 1 - (((1-r2)*(num-1))/(num-4-1))
    return y


def estimateparam(array):
    ymax = np.max(array)
    maxindex = np.nonzero(array == ymax)[0]
    binsum = np.nonzero(array)
    widthbin = len(binsum[0])
    return maxindex, widthbin, ymax


def gaussian_fit_ciu(ciu_obj, widthfrac=0.01):
    """
    Gaussian fitting module for single-gaussian analysis of CIU-type data. Determines estimated
    initial parameters and fits a single Gaussian distribution to each column of the input ciu_data
    matrix. Saves all output into a subfolder of filepath titled 'gausfitoutput'
    :param ciu_obj: ciu analysis object in which to save all data. Uses obj.ciu_data, axes, etc
     (ciu_data is 2D numpy, axis 0 = DT, axis 1 = CV) for all data handling
     Smoothing, interpolation, cropping, etc should be done prior to running this method.
    :param widthfrac: width fraction estimation for fitting, default 0.01
    :return: saves all gaussian outputs into the ciu_obj and returns it
    """
    ciu_data = ciu_obj.ciu_data
    y_axis = ciu_obj.axes[0]    # "y-axis" refers to the CIU plot, where DT (axis 1) is plotted on the y
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
        td_estimate_index, w_estimate_indexlen, amp = estimateparam(array)
        popt, pcov = curve_fit(gaussfunc, y_axis, array, method='lm',
                               p0=[0, amp, y_axis[td_estimate_index], widthfrac * w_estimate_indexlen],
                               maxfev=5000)
        yfit = gaussfunc(y_axis, *popt)
        fwhm, res = resandfwhm(popt[2], popt[3])
        print('Arrival time | Width')
        print(popt[2], ' | ', popt[3])
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
        arrivtime_centroid.append(popt[2])
        width_gausfit.append(popt[3])
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
    # ciu_obj.save_gaussfits_pdf(outputpath)
    ciu_obj.plot_centroids(outputpath)
    ciu_obj.plot_fwhms(outputpath)
    ciu_obj.save_gauss_params(outputpath)
    print('Job completed')
    return ciu_obj
