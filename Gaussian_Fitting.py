"""
Methods for high level analysis of fingerprints - feature detection, classification, etc
author: DP, Gaussian fitting module from SD
date: 10/10/2017
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import scipy.integrate
import scipy.interpolate
import os
import math
import peakutils
import pickle
import tkinter
from tkinter import filedialog
import scipy.signal
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import lmfit
import time
import multiprocessing

from CIU_raw import CIURaw
from Raw_Processing import normalize_by_col
from CIU_analysis_obj import CIUAnalysisObj

# imports for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CIU_Params import Parameters

protein_prefix = 'p'
nonprotein_prefix = 'np'


class Gaussian(object):
    """
    Container for fitted gaussian information. Holds fit parameters and any associated metadata
    """
    def __init__(self, amplitude, centroid, width, collision_voltage, pcov, protein_bool):
        """
        Initialize a new Gaussian container
        :param amplitude: peak amplitude
        :param centroid: peak centroid
        :param width: peak width
        :param collision_voltage: Activation axis value at which this Gaussian was generated
        :param pcov: (optional, set to None if not needed) covariance matrix from curve fitting
        :param protein_bool: True if this is a protein (signal) peak, False if a non-protein (noise) peak
        """
        self.amplitude = amplitude
        self.centroid = centroid
        self.width = width
        self.cv = collision_voltage
        self.fwhm = 2*(math.sqrt(2*math.log(2)))*self.width
        self.resolution = self.centroid/self.fwhm
        self.fit_covariances = pcov
        self.is_protein = protein_bool
        if pcov is not None:
            self.fit_errors = np.sqrt(np.diag(pcov))

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
        return '{},{:.2f},{:.2f},{:.2f}'.format(self.cv, self.amplitude, self.centroid, self.width)

    def return_popt(self):
        """
        Re-generate Gaussian function parameter list (e.g. popt style from curve_fit) from
        gaussian object
        :return: [baseline, amplitude, centroid, width]
        """
        return [self.amplitude, self.centroid, self.width]

# todo: deprecated
# class FitDiagnostics(object):
#     """
#     Container for diagnostics and information from Gaussian fitting. Intended to enable rapid diagnostics
#     and hold information from multi-round fitting and ease plotting/saving various outputs.
#     Organization: A single FitDiagnostics object for a CIUAnalysis object containing all diagnostic
#     information in lists organized by collision voltage.
#     """
#
#     def __init__(self, cv_axis):
#         """
#         Initialize an empty diagnostics object over the collision voltages of a CIU fingerprint
#         :param cv_axis: collision voltage axis of the associated CIU analysis
#         """
#         self.cvs = cv_axis
#
#         # First round fitting diagnostics
#         self.num_peaks_list = []
#         self.popt_lists = []
#         self.gauss_lists = []
#         self.filt_gauss_lists = []
#         self.fit_stats_list = []
#         self.adjrsq_list = []
#
#         self.all_fits_lists = []    # list of (lists-by-CV) containing ALL fits performed at that CV (SingleFitStats)
#
#     def append_best_fit_info(self, fit_stats_obj):
#         """
#         Add the best result from a particular collision voltage to the primary lists stored
#         in this object
#         :param fit_stats_obj: container with *best* fit information (out of all fits at this voltage)
#         :type fit_stats_obj: SingleFitStats
#         :return: void
#         """
#         self.fit_stats_list.append(fit_stats_obj)
#         self.popt_lists.append(fit_stats_obj.get_popt())
#         self.gauss_lists.append(fit_stats_obj.gaussians)
#         self.adjrsq_list.append(fit_stats_obj.adjrsq)
#         self.num_peaks_list.append(len(fit_stats_obj.gaussians))


class SingleFitStats(object):
    """
    Container for holding fit information for a single multi-Gaussian fitting (one collision voltage).
    Includes r2, fit data, error estimates, etc (all output of linregress)
    Intended to use called when initializing a fit.
    *updated to include output from LMFit and original (curve_fit) in same container. Must have
    one of popt OR lmfit_output, and will generate Gaussians and r2 from both for output
    """
    def __init__(self, x_data, y_data, cv, amp_cutoff, lmfit_output=None, popt=None):
        """
        Initialize a new fit between the provided x/y data and optimized Gaussian parameters
        :param x_data: x (DT) raw data being fit by popt
        :param y_data: y (intensity) raw data being fit by popt
        :param popt: optimized parameters returned from curve_fit
        :param amp_cutoff: minimum amplitude for peak to be allowed
        :param cv: collision voltage at which this fit occurred
        :param lmfit_output: output container from LMFit (from model.fit(...))
        :type lmfit_output:
        """
        self.x_data = x_data
        self.y_data = y_data

        if lmfit_output is not None:
            protein_popt, nonprotein_popt = get_popt_from_lmoutput(lmfit_output, amp_cutoff)
            popt = [x for x in protein_popt]
            popt.extend(nonprotein_popt)
        else:
            protein_popt = popt
            nonprotein_popt = []

        self.y_fit = multi_gauss_func(x_data, *popt)
        self.slope, self.intercept, self.rvalue, self.pvalue, self.stderr = linregress(self.y_data, self.y_fit)
        self.adjrsq = adjrsquared(self.rvalue ** 2, len(y_data))
        # can't save LMFit output because it contains temp classes that are not pickle-able
        # self.lmfit_output = lmfit_output

        # Gaussian lists specific to protein and non-protein components fitted by LMFit
        self.gaussians_protein = generate_gaussians_from_popt(protein_popt, protein_bool=True, cv=cv, pcov=None)
        self.gaussians_nonprotein = generate_gaussians_from_popt(nonprotein_popt, protein_bool=False, cv=cv, pcov=None)

        self.gaussians = [x for x in self.gaussians_protein]
        self.gaussians.extend(self.gaussians_nonprotein)

        # additional information that may be present
        self.p0 = None      # initial guess array used to generate this popt
        self.pcov = None    # output covariance matrix

        self.score = None   # score from second round fitting (r2 - penalties)
        self.peak_penalties = None      # list of penalties for each peak in the Gaussian list

    def compute_fit_score(self, params_obj, penalty_scaling):
        """
        Uses a penalty function to attempt to regularize the fitting and score peak fits optimally.
        Penalty function is designed to penalize:
            - peaks whose widths deviate from expected protein peak width
            - large numbers of peaks
            - peaks that are too close together
            - large movement compared to previous CV
        :param params_obj: parameter container
        :type params_obj: Parameters
        :param penalty_scaling: how much to scale penalty (to reduce contribution relative to rsq)
        :return: score (float between 0, 1), penalties by individual peaks
        """
        # compute penalties by peak to allow removal of poorly fit peaks
        peak_penalties = []
        # todo: finalize values and/or make accessible (advanced?) for width, shared area, and min protein amp penalties
        for gaussian in self.gaussians_protein:
            # antibody settings: exp=0.45, tol=0.2, steep=1
            # membrane settings: exp=0.45, tol=0.4, steep=0.2
            current_penalty = compute_width_penalty(gaussian.width, expected_width=params_obj.gaussian_72_prot_peak_width, tolerance=params_obj.gaussian_73_prot_width_tol, steepness=1)
            if len(self.gaussians_protein) > 1:
                current_penalty += compute_area_penalty(gaussian, self.gaussians_protein, self.x_data)
            peak_penalties.append(current_penalty)

        # add up penalties and subtract from the fit adjusted r2 to obtain final score
        total_penalty = np.sum(peak_penalties)

        # add penalty for low amplitude protein peak - max protein peak shouldn't be too low
        if len(self.gaussians_nonprotein) > 0:
            if len(self.gaussians_protein) > 0:
                max_protein_amp = max([x.amplitude for x in self.gaussians_protein])
            else:
                max_protein_amp = 0
            if max_protein_amp < params_obj.gaussian_9_min_protein_amp:
                total_penalty += (params_obj.gaussian_9_min_protein_amp - max_protein_amp)

        scaled_penalty = total_penalty * penalty_scaling
        score = self.adjrsq - scaled_penalty

        # save information to the fit container
        self.score = score
        self.peak_penalties = peak_penalties
        # return score, peak_penalties

    def get_popt(self):
        """
        Return a single parameters list for all Gaussians from this fit in curve_fit compatible
        format
        :return: list of optimized params (popt)
        """
        popt = []
        for gaussian in self.gaussians:
            popt.extend(gaussian.return_popt())
        return popt

    def plot_fit(self):
        """
        plotting method for diagnostics
        :return: void
        """
        plt.clf()
        plt.scatter(self.x_data, self.y_data)
        plt.plot(self.x_data, self.y_fit, ls='--', color='black')
        for gaussian in self.gaussians:
            plt.plot(self.x_data, gaussfunc(self.x_data, *gaussian.return_popt()))
        plt.show()
        plt.close()


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
    for i in range(0, len(params), 3):
        y = y + gaussfunc(x, params[i], params[i+1], params[i+2])
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
        params_lists.extend([amp_guess, centroid_guess, width_guess])
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
        params_lists.append([amp_guess, centroid_guess, width_guess])

    # sort guesses by amplitude (index 1 in each sublist) in order from largest to smallest
    params_lists = sorted(params_lists, key=lambda x: x[0], reverse=True)
    return params_lists


def gaussfunc(x, amplitude, centroid, sigma):
    """
    Gaussian function with constraints applied for CIU data
    :param x: x
    :param amplitude: gaussian amplitude (constrained to be positive)
    :param centroid: gaussian centroid
    :param sigma: gaussian width
    :return: y = f(x)
    """
    exponent = ((x - centroid)**2) / (2 * (sigma**2))
    y = amplitude * (np.exp(-exponent))         # using this function since our data is always normalized
    # y = amplitude/(np.sqrt(2*np.pi) * sigma) * (np.exp(-exponent))     # use this for non-normalized data

    # if amplitude < 0.25:
    #     y = amplitude / sigma * (np.exp(-exponent))
    # else:
    #     y = amplitude * (np.exp(-exponent))

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
    """
    Compute adjusted r2 given the number of degrees of freedom in an analysis
    :param r2: original r2 value (float)
    :param num: degrees of freedom (int)
    :return: adjusted r2
    """
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

        # ensure peak width (FWHM) is below the cutoff and above 0
        fwhm = 2 * math.sqrt(2 * math.log(2)) * params_list[index + 2]
        if 0 < fwhm < peak_width_cutoff:
            # also remove amplitdues below the intensity cutoff
            if params_list[index] > intensity_cutoff:
                if centroid_bounds is not None:
                    # centroid bounds provided - if matched, include the peak
                    if centroid_bounds[0] < params_list[index + 1] < centroid_bounds[1]:
                        include_peak = True
                elif params_list[index + 1] > 0:
                    # If no bounds provided lso remove centroids < 0
                    include_peak = True

        if include_peak:
            filtered_list.extend(params_list[index:index + 3])
        index += 3
    return filtered_list


def reconstruct_from_fits(gaussian_lists_by_cv, axes, new_filename, params_obj):
    """
    Construct a new analysis object using the filtered Gaussian fits of the provided analysis object
    as the raw data. Must have previously performed Gaussian feature detection on the provided analysis_obj
    :param gaussian_lists_by_cv: list of lists of Gaussian objects at each CV
    :param axes: [DT_axis, CV_axis]: two numpy arrays with drift and CV axes to use
    :param new_filename:
    :param params_obj: Parameters container with parameters to save into the new CIUAnalsis obj
    :return: new CIUAnalysisObj with reconstructed raw data
    :rtype: CIUAnalysisObj
    """
    ciu_data_by_cols = []
    dt_axis = axes[0]
    # construct the raw data at each collision voltage to stitch together into a CIU matrix
    for cv_gauss_list in gaussian_lists_by_cv:
        # assemble all the parameters for Gaussians at this CV
        all_params = []
        for gaussian in cv_gauss_list:
            all_params.extend(gaussian.return_popt())

        # Use the Gaussian function to construct intensity data at each DT
        intensities = multi_gauss_func(dt_axis, *all_params)

        ciu_data_by_cols.append(intensities)

    # todo: normalize intensities

    # finally, transpose the CIU data to match the typical format, normalize, and return the object
    final_data = np.asarray(ciu_data_by_cols).T
    final_data = normalize_by_col(final_data)

    raw_obj = CIURaw(final_data, dt_axis, axes[1], new_filename)
    new_analysis_obj = CIUAnalysisObj(raw_obj, final_data, axes, params_obj)
    new_analysis_obj.short_filename = new_analysis_obj.short_filename + '_gauss-recon'

    new_analysis_obj.protein_gaussians = gaussian_lists_by_cv
    new_analysis_obj.gaussians = gaussian_lists_by_cv
    return new_analysis_obj


def parse_gaussian_list_from_file(filepath):
    """
    Read in a list of Gaussians from file and return a list of Gaussian objects.
    File format: (comma delimited text file, headers (#) ignored)
    CV1, gauss1_amp, gauss1_cent, gauss1_width, gauss2 A, c, w, ..., gaussN a, c, w
    CV2, gauss1_amp, (etc)
    CV3
    ...
    :param filepath: full path to file to read
    :return: list of lists of Gaussian objects sorted by collision voltage, list of collision voltages
    """
    gaussian_list_by_cv = []
    cvs = []
    dt_axis = []
    with open(filepath, 'r') as inputfile:
        for line in list(inputfile):
            # skip headers
            if line.startswith('#'):
                continue
            splits = line.rstrip('\n').split(',')
            splits = [x for x in splits if x is not '']

            # read DT axis
            if line.lower().startswith('drift'):
                try:
                    dt_axis = np.asarray([float(x) for x in splits[1:]])
                except ValueError:
                    print('DT axis could not be read. Line was: {}'.format(line))
                    dt_axis = []
                continue

            # get CVs from first column only to avoid duplicates
            try:
                cv = float(splits[0])
                cvs.append(cv)
            except ValueError:
                print('Invalid CV in line: {}; value must be a number. Skipping this line'.format(line))
                continue

            # read remaining Gaussian information
            index = 0
            gaussians = []
            while index < len(splits) - 1:
                try:
                    cv = float(splits[index])
                    amp = float(splits[index + 1])
                    cent = float(splits[index + 2])
                    width = float(splits[index + 3])
                    gaussians.append(Gaussian(amp, cent, width, cv, pcov=None, protein_bool=True))
                except (IndexError, ValueError):
                    print('Invalid values for Gaussian. Values were: {}. Gaussian could not be parsed and was skipped'.format(splits[index:index + 3]))
                index += 4
            gaussian_list_by_cv.append(gaussians)

    return gaussian_list_by_cv, [dt_axis, np.asarray(cvs)]


#     todo: DEPRECATED
# def check_peak_dist(popt_list, current_guess_list, min_distance_dt, max_peak_width):
#     """
#
#
#     Determine whether the centroid of the current guess is too close to an existing (already fit) peak.
#     Note: excludes peaks above the width cutoff, as these are not used for feature detection/etc anyway
#     and may overlap substantially with signal peaks (removing them may negatively impact fitting)
#     :param popt_list: Current optimized parameter list (flat, as returned from curve_fit)
#     :param current_guess_list: list of parameters for current guess [y0, amplitude, centroid, width]
#     :param min_distance_dt: minimum distance between peaks in drift axis units
#     :param max_peak_width: max width for filtering from Parameters object
#     :return: boolean; True if distance is greater than minimum
#     """
#     # automatically allow any peaks that are too wide, as these will not impact feature detection/etc
#     if current_guess_list[2] > max_peak_width:
#         return True
#
#     guess_centroid = current_guess_list[1]
#     existing_centroids = popt_list[1::3]
#     # return false if any existing centroid is too close to the current guess, excluding noise peaks
#     for existing_centroid in existing_centroids:
#         if abs(existing_centroid - guess_centroid) < min_distance_dt:
#             return False
#     return True


# todo: deprecated
# def check_peak_dists(popt_list, params_obj):
#     """
#     Look through all fitted peak parameters and determine if any peaks are too close to each other.
#     If so, return True and the parameters list with the lower intensity of the overlapping peaks removed.
#     WILL ONLY REMOVE 1 PEAK - since it is intended to function in the loop as peaks are added one
#     at a time, if a peak is added too close to another, it is removed and the iteration is stopped.
#     :param popt_list: list of output/fitted gaussian parameters
#     :param params_obj: Parameters object
#     :type params_obj: Parameters
#     :return: boolean (True if peaks are too close), updated popt_list with 'bad' peak removed
#     """
#     # convert popt_list to Gaussian objects for easier handling
#     index = 0
#     gaussians = []
#     while index < len(popt_list):
#         gaussian = Gaussian(popt_list[index], popt_list[index + 1], popt_list[index + 2], None, None)
#         # ignore peaks that are above width max - they are allowed to be close to others (noise can be anywhere)
#         if not gaussian.width > params_obj.gaussian_3_width_max:
#             gaussians.append(gaussian)
#         index += 3
#
#     # examine all Gaussians and determine if any are too close together
#     for gaussian_combo in itertools.combinations(gaussians, 2):
#         gaussian1 = gaussian_combo[0]
#         gaussian2 = gaussian_combo[1]
#         if abs(gaussian1.centroid - gaussian2.centroid) < params_obj.gaussian_6_min_peak_dist:
#             print('dist too close, was: {:.2f}'.format(abs(gaussian1.centroid - gaussian2.centroid)))
#             # these peaks are too close, return
#             if gaussian1.amplitude > gaussian2.amplitude:
#                 gaussians.remove(gaussian2)
#             else:
#                 gaussians.remove(gaussian1)
#
#             # reassemble popt_list
#             final_popt = []
#             for gaussian in gaussians:
#                 final_popt.extend(gaussian.return_popt())
#             return True, final_popt
#
#     # if no peaks are too close, return False and the original popt list
#     return False, popt_list


def gaussian_lmfit_main(analysis_obj, params_obj):
    """
    Alternative Gaussian fitting method using LMFit for composite modeling of peaks. Estimates initial peak
    parameters using helper methods, then fits optimized Gaussian distributions and saves results. Intended
    for direct call from buttons in GUI.
    :param analysis_obj: analysis container
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: parameter information container
    :type params_obj: Parameters
    :return: updated analysis object
    :rtype: CIUAnalysisObj
    """
    start_time = time.time()

    cv_col_data = np.swapaxes(analysis_obj.ciu_data, 0, 1)
    outputpath = os.path.join(os.path.dirname(analysis_obj.filename), analysis_obj.short_filename)
    if not os.path.isdir(outputpath):
        os.makedirs(outputpath)

    best_fits_by_cv = []
    scores_by_cv = []

    pool = multiprocessing.Pool(processes=params_obj.gaussian_61_num_cores)
    results = []

    for cv_index, cv_col_intensities in enumerate(cv_col_data):
        # prepare initial guesses in the form of a list of Gaussian objects, sorted by amplitude
        cv = analysis_obj.axes[1][cv_index]
        gaussian_guess_list = guess_gauss_init(cv_col_intensities, analysis_obj.axes[0], params_obj.gaussian_5_width_fraction, cv, params_obj.gaussian_1_convergence, amp_cutoff=params_obj.gaussian_2_int_threshold)

        # Run fitting and scoring across the provided range of peak options
        # all_fits = iterate_lmfitting(analysis_obj.axes[0], cv_col_intensities, gaussian_guess_list, params_obj, outputpath)

        argslist = [analysis_obj.axes[0], cv_col_intensities, gaussian_guess_list, params_obj, outputpath]
        pool_result = pool.apply_async(iterate_lmfitting, args=argslist)
        results.append(pool_result)

    for cv_index, cv_results in enumerate(results):
        all_fits = cv_results.get()

        # save the fit with the highest score out of all fits collected
        # todo: add handling for nonprotein peaks (include in fit, but filter for outputs?)
        best_fit = max(all_fits, key=lambda x: x.score)
        best_fits_by_cv.append(best_fit)
        scores_by_cv.append([fit.score for fit in all_fits])

    # output final results
    fit_time = time.time() - start_time
    print('fitting done in {:.2f}'.format(fit_time))

    # save output
    print(scores_by_cv)
    prot_gaussians = [fit.gaussians_protein for fit in best_fits_by_cv]
    nonprot_gaussians = [fit.gaussians_nonprotein for fit in best_fits_by_cv]
    all_gaussians = [fit.gaussians for fit in best_fits_by_cv]
    # rsqs = [fit.adjrsq for fit in best_fits_by_cv]
    # save_gaussfits_pdf(analysis_obj, prot_gaussians, outputpath, rsq_list=rsqs, filename_append='lmfit')
    save_fits_pdf_new(analysis_obj, best_fits_by_cv, outputpath)

    best_centroids = []
    for gauss_list in prot_gaussians:
        best_centroids.append([x.centroid for x in gauss_list])
    nonprot_centroids = []
    for gauss_list in nonprot_gaussians:
        nonprot_centroids.append([x.centroid for x in gauss_list])
    plot_centroids(best_centroids, analysis_obj, outputpath, nonprotein_centroids=nonprot_centroids)

    plot_time = time.time() - start_time - fit_time
    print('plotting done in {:.2f}'.format(plot_time))

    # save results to analysis obj
    analysis_obj.gaussians = all_gaussians
    analysis_obj.protein_gaussians = prot_gaussians
    analysis_obj.nonprotein_gaussians = nonprot_gaussians
    analysis_obj.gauss_fits_by_cv = best_fits_by_cv

    #  OLD ##############################################################################
    # basic setup example
    # single_model = lmfit.Model(gaussfunc)
    # params = single_model.make_params(a=1, xc=10, w=1)
    # # fitting example
    # result = single_model.fit(y_data, params, x=x_data)

    # dt_axis = analysis_obj.axes[0]  # drift time (DT) - x axis for fitting, y axis for final CIU plot
    # cv_axis = analysis_obj.axes[1]
    # cv_col_data = np.swapaxes(analysis_obj.ciu_data, 0, 1)
    # outputpath = os.path.join(os.path.dirname(analysis_obj.filename), analysis_obj.short_filename)
    # if not os.path.isdir(outputpath):
    #     os.makedirs(outputpath)
    #
    # for cv_index, cv_col_intensities in enumerate(cv_col_data):
    #
    #     first_model = lmfit.models.GaussianModel(prefix='g1')   # use prefixes to prevent models from having same param names
    #     models = first_model
    #     gauss_index = 2
    #     guess_params = first_model.guess(data=cv_col_intensities, x=dt_axis)
    #
    #     result = first_model.fit(cv_col_intensities, guess_params, x=dt_axis)
    #     rsq = 1 - result.residual.var() / np.var(cv_col_intensities)
    #     print(rsq)
    #
    #     while rsq < params_obj.gaussian_1_convergence:
    #         models += lmfit.models.GaussianModel(prefix='g{}'.format(gauss_index))
    #         guess_params += models.right.guess(data=cv_col_intensities, x=dt_axis)   # models.right is the Gaussian we just added, so we're using its 'guess' method
    #         # this fails on second time - rsq is 0. Maybe a bad (or same) initial guess or something? should plot to understand what's happening
    #         gauss_index += 1
    #         result = models.fit(cv_col_intensities, guess_params, x=dt_axis)
    #         rsq = 1 - result.residual.var() / np.var(cv_col_intensities)
    #         print(rsq)
    #
    #     print(result.fit_report(min_correl=0.5))
    #
    #     plt.clf()
    #     plt.plot(dt_axis, cv_col_intensities, 'b')
    #     plt.plot(dt_axis, result.best_fit, 'r-')
    #     outputname = os.path.join(outputpath, str(cv_index) + '.png')
    #     plt.savefig(outputname)

    return analysis_obj


def guess_gauss_init(ciu_col, dt_axis, width_frac, cv, rsq_cutoff, amp_cutoff):
    """
    Generate initial guesses for Gaussians. Currently using just the estimate_multi_params_all method
    with output formatted as Gaussian objects, but will likely try to include initial first round of
    fitting from curve_fit as well.
    :param ciu_col:
    :param dt_axis:
    :param width_frac:
    :param cv:
    :param amp_cutoff: minimum amplitude for peak to be allowed
    :return:
    """
    gaussians = []

    # estimate a (rather inaccurate) list of possible peaks to use as guesses for fitting
    guess_list = estimate_multi_params_all(ciu_col, dt_axis, width_frac)

    # run the initial (first round) fitting with curve_fit to generate high quality guesses
    popt, pcov, allfits = sequential_fit_rsq(guess_list, dt_axis, ciu_col, cv=cv, convergence_rsq=rsq_cutoff, amp_cutoff=amp_cutoff)

    # convert all guesses to Gaussians and sort in decreasing quality order to provide for future rounds
    r1_guesses = generate_gaussians_from_popt(opt_params_list=popt, protein_bool=True, cv=cv, pcov=pcov)
    gaussians.extend(sorted(r1_guesses, key=lambda x: x.amplitude, reverse=True))

    # todo: skip peaks that are too close in starting location to the high quality first guesses?
    for param_guess in guess_list:
        # catch 0 amplitude and just make it very small
        if abs(param_guess[0]) < 1e-5:
            param_guess[0] = 1e-5
        gaussians.extend(generate_gaussians_from_popt(param_guess, protein_bool=True, cv=cv, pcov=None))

    return gaussians


def sequential_fit_rsq(all_peak_guesses, dt_axis, cv_col_intensities, cv, convergence_rsq, amp_cutoff):
    """
    Gaussian fitting 1.0 method - adds peak components from a list of initial guesses (provided)
    until r2 value reaches a user specified convergence criterion. Abstracted for use as a way to
    provide high quality initial guesses for more advanced methods, but can also be used in isolation.
    **This method is for a single CV value** and returns the final popt, pcov, and a list of fits from all rounds.
    :param all_peak_guesses: list of lists of initial guess values for parameters. Typically from estimate_multi_params_all
    :param dt_axis: x-axis for fitting (DT axis values)
    :param cv_col_intensities: y-data for fitting (intensity data along the DT axis)
    :param cv: the collision voltage (CV) at which this fitting takes place
    :param convergence_rsq: the minimum rsq at which to stop adding peak components
    :param amp_cutoff: minimum amplitude for peak to be allowed
    :return: popt, pcov, list of SingleFitStats containers for each round of fitting.
    """
    param_guesses_multiple = []
    all_fit_rounds = []
    popt, pcov = [], []

    # set bounds for fitting: keep baseline and centroid on DT axis, amplitude 0 to 1.5, width 0 to len(dt_axis)
    max_dt = dt_axis[len(dt_axis) - 1]
    min_dt = dt_axis[0]
    fit_bounds_lower, fit_bounds_upper = [], []
    fit_bounds_lower_append = [0, min_dt, 0]
    fit_bounds_upper_append = [1, max_dt, len(dt_axis)]

    i = 0
    iterate_gaussian_flag = True
    # Iterate through peak detection until convergence criterion is met, adding one additional peak each iteration
    while iterate_gaussian_flag:
        # Set up initial guesses
        try:
            param_guesses_multiple.extend(all_peak_guesses[i])
            # ensure bounds arrays maintain same shape as parameter guesses
            fit_bounds_lower.extend(fit_bounds_lower_append)
            fit_bounds_upper.extend(fit_bounds_upper_append)
        except IndexError:
            # No converge with all estimated peaks. Continue with final estimate
            print('Included all {} peaks found, but r^2 still less than convergence criterion. '
                  'Poor fitting possible'.format(i + 1))
            break

        # Run fitting (round 1)
        try:
            popt, pcov = scipy.optimize.curve_fit(f=multi_gauss_func, xdata=dt_axis, ydata=cv_col_intensities,
                                                  method='trf',
                                                  p0=param_guesses_multiple,
                                                  bounds=(fit_bounds_lower, fit_bounds_upper))
            # perr = np.sqrt(np.diag(pcov))
        except (RuntimeError, ValueError):
            popt, pcov = [], []

        current_fit = SingleFitStats(dt_axis, cv_col_intensities, amp_cutoff=amp_cutoff, popt=popt, cv=cv)
        all_fit_rounds.append(current_fit)

        # stop iterating once convergence criteria have been reached
        if not current_fit.adjrsq < convergence_rsq:
            iterate_gaussian_flag = False
        i += 1

    return popt, pcov, all_fit_rounds


def iterate_lmfitting(x_data, y_data, guesses_list, params_obj, outputpath):
    """
    Primary fitting method. Iterates over combinations of protein and non-protein peaks using
    models generated with LMFit based on the initial peak guesses in the guesses_list. Fits are
    evaulated with r2 and scoring functions to determine which number of components gave the
    best fit, which is returned as a MinimizerResult/ModelFitResult from LMFit
    :param x_data: x_axis data for fitting (DT axis)
    :param y_data: y data to fit (intensity values along the DT axis)
    :param guesses_list: list of Gaussian objects in decreasing amplitude order for initial guesses
    :type guesses_list: list[Gaussian]
    :param params_obj: Parameters container with various parameter information
    :type params_obj: Parameters
    :param outputpath: directory in which to save outputs
    :return: best fit result as a MinimizerResult/ModelFitResult from LMFit
    """
    # determine the number of components over which to iterate fitting
    max_num_prot_pks = params_obj.gaussian_71_max_prot_components
    if params_obj.gaussian_3_mode == 'No Selection':
        max_num_nonprot_pks = params_obj.gaussian_82_max_nonprot_comps  # params_obj/advanced for more options?
    else:
        max_num_nonprot_pks = 0

    cv = guesses_list[0].cv
    output_fits = []
    # iterate over all peak combinations
    for num_prot_pks in range(1, max_num_prot_pks + 1):
        for num_nonprot_pks in range(params_obj.gaussian_81_min_nonprot_comps, max_num_nonprot_pks + 1):
            # assemble the models and fit parameters for this number of protein/non-protein peaks
            models_list, fit_params = assemble_models(num_prot_pks, num_nonprot_pks, params_obj, guesses_list, dt_axis=x_data)

            # combine all model parameters and perform the actual fitting
            final_model = models_list[0]
            for model in models_list[1:]:
                final_model += model

            output = final_model.fit(y_data, fit_params, x=x_data, method=params_obj.gaussian_6_fit_method, nan_policy='omit',
                                     scale_covar=False)

            # compute fits and score
            current_fit = SingleFitStats(x_data=x_data, y_data=y_data, cv=cv, lmfit_output=output, amp_cutoff=params_obj.gaussian_2_int_threshold)
            # only score protein peaks, as non-protein peaks can overlap and have differing widths (may add different score func eventually if needed)
            current_fit.compute_fit_score(params_obj, penalty_scaling=1)
            output_fits.append(current_fit)

            plt.clf()
            model_components = output.eval_components(x=x_data)
            output.plot_fit()
            for component_name, comp_value in model_components.items():
                plt.plot(x_data, comp_value, '--', label=component_name)
            plt.legend(loc='best')
            if not output.success:
                plt.title('{}V, fitting failed'.format(cv))
            else:
                penalty_string = ['{:.2f}'.format(x) for x in current_fit.peak_penalties]
                plt.title('{}V, r2: {:.3f}, score: {:.4f}, peak pens: {}'.format(cv, current_fit.adjrsq, current_fit.score, ','.join(penalty_string)))
            outputname = os.path.join(outputpath, '{}_p{}_np{}_fits.png'.format(cv, num_prot_pks, num_nonprot_pks))
            plt.savefig(outputname)

    return output_fits


def assemble_models(num_prot_pks, num_nonprot_pks, params_obj, guesses_list, dt_axis):
    """
    Assign the peaks in the list of guesses to protein and non-protein components of the final model.
    Guess list is assumed to be in decreasing order of amplitude. Guesses are assigned to non-protein peaks
    if their width is larger than the expected protein width, and to protein peaks otherwise.
    :param num_prot_pks: number of protein components to be fit in this iteration
    :param num_nonprot_pks: number of nonprotein components to be fit in this iteration
    :param params_obj: parameter container
    :type params_obj: Parameters
    :param guesses_list: list of Gaussian objects containing guess information, in descending order of amplitude
    :param dt_axis: x-axis for the fitting
    :return: list of LMFit Models, LMFit Parameters() dictionary
    """
    fit_params = lmfit.Parameters()

    # assemble models for this number of peaks
    guess_index = 0
    total_num_components = num_nonprot_pks + num_prot_pks
    models_list = []
    # counters for numbers of each peak type left to be fitted
    nonprots_remaining = num_nonprot_pks
    prots_remaining = num_prot_pks

    # todo: add check for out of guesses (maybe just use default make_params in that case?)

    if num_nonprot_pks > 0:
        # non-protein peak(s) present, assign peaks wider than width max for protein to them
        for comp_index in range(0, total_num_components):
            try:
                next_guess = guesses_list[guess_index]
            except IndexError:
                # out of guesses - make a generic non-protein guess (50% amplitude, centered in the middle of the dt axis, non-protein minimum width)
                dt_middle = (dt_axis[-1] - dt_axis[0]) / 2.0
                next_guess = Gaussian(amplitude=0.5, centroid=dt_middle, width=params_obj.gaussian_83_nonprot_width_min, collision_voltage=None, pcov=None, protein_bool=False)

            # todo: add FWHM conversion
            if next_guess.width > (params_obj.gaussian_72_prot_peak_width + params_obj.gaussian_73_prot_width_tol):
                # the width of this guess is wider than protein - try fitting a nonprotein peak here
                if nonprots_remaining > 0:
                    model, params = make_nonprotein_model(
                        prefix='{}{}'.format(nonprotein_prefix, guess_index + 1),
                        guess_gaussian=next_guess,
                        params_obj=params_obj,
                        dt_axis=dt_axis)
                    models_list.append(model)
                    fit_params.update(params)

                    guess_index += 1
                    nonprots_remaining -= 1
                else:
                    # no more non-protein peaks left, so add a protein peak
                    model, params = make_protein_model(prefix='{}{}'.format(protein_prefix, guess_index + 1),
                                                       guess_gaussian=next_guess,
                                                       params_obj=params_obj,
                                                       dt_axis=dt_axis)
                    models_list.append(model)
                    fit_params.update(params)
                    guess_index += 1
                    prots_remaining -= 1
            else:
                # guess peak width is narrow enough to be protein - guess it first
                if prots_remaining > 0:
                    model, params = make_protein_model(prefix='{}{}'.format(protein_prefix, guess_index + 1),
                                                       guess_gaussian=next_guess,
                                                       params_obj=params_obj,
                                                       dt_axis=dt_axis)
                    models_list.append(model)
                    fit_params.update(params)
                    guess_index += 1
                    prots_remaining -= 1
                else:
                    # no protein peaks left, so guess non-protein
                    model, params = make_nonprotein_model(
                        prefix='{}{}'.format(nonprotein_prefix, guess_index + 1),
                        guess_gaussian=next_guess,
                        params_obj=params_obj,
                        dt_axis=dt_axis)
                    models_list.append(model)
                    fit_params.update(params)

                    guess_index += 1
                    nonprots_remaining -= 1

    else:
        # protein peaks only - simply go through the guess list (descending order of amplitude)
        for prot_pk_index in range(0, num_prot_pks):
            try:
                next_guess = guesses_list[guess_index]
            except IndexError:
                # out of guesses - make a generic protein guess (50% amplitude, centered in the middle of the dt axis, estimated protein width)
                dt_middle = (dt_axis[-1] - dt_axis[0]) / 2.0
                next_guess = Gaussian(amplitude=0.5, centroid=dt_middle, width=params_obj.gaussian_72_prot_peak_width, collision_voltage=None, pcov=None, protein_bool=True)

            model, params = make_protein_model(prefix='{}{}'.format(protein_prefix, guess_index + 1),
                                               guess_gaussian=next_guess,
                                               params_obj=params_obj,
                                               dt_axis=dt_axis)
            models_list.append(model)
            fit_params.update(params)
            guess_index += 1

    return models_list, fit_params


def make_protein_model(prefix, guess_gaussian, params_obj, dt_axis):
    """
    Generate an LMFit model object from initial parameters in the guess_gaussian container and
    parameters.
    :param prefix: string prefix for this model to prevent params from having same names
    :param guess_gaussian: Gaussian object with initial guess parameters
    :type guess_gaussian: Gaussian
    :param params_obj: parameters container
    :type params_obj: Parameters
    :param dt_axis: dt_axis information for determining boundaries
    :return: LMFit model object with initialized parameters, bounds, and constraints
    """
    max_dt = dt_axis[-1]
    min_dt = dt_axis[0]

    # model = lmfit.models.GaussianModel(prefix=prefix)
    # model = lmfit.models.VoigtModel(prefix=prefix)
    model = lmfit.Model(gaussfunc, prefix=prefix)

    model_params = model.make_params()
    # model_params[prefix + 'gamma'].set(value=0.7, vary=True)

    # todo: convert from FWHM
    min_width = params_obj.gaussian_72_prot_peak_width - params_obj.gaussian_73_prot_width_tol
    max_width = params_obj.gaussian_72_prot_peak_width + params_obj.gaussian_73_prot_width_tol

    # set initial guesses and boundaries
    model_params[prefix + 'centroid'].set(guess_gaussian.centroid, min=min_dt, max=max_dt)
    model_params[prefix + 'sigma'].set(guess_gaussian.width, min=min_width, max=max_width)
    model_params[prefix + 'amplitude'].set(guess_gaussian.amplitude, min=0, max=1.5)

    # todo: apply constraints - ** might actually need to be out in the main body IF doing any relative to other peaks. If just width, can be in here

    # return the model
    return model, model_params


def make_nonprotein_model(prefix, guess_gaussian, params_obj, dt_axis):
    """
    Generate an LMFit model object from initial parameters in the guess_gaussian container and
    parameters.
    :param prefix: string prefix for this model to prevent params from having same names
    :param guess_gaussian: Gaussian object with initial guess parameters
    :type guess_gaussian: Gaussian
    :param params_obj: parameters container
    :type params_obj: Parameters
    :param dt_axis: dt_axis information for determining boundaries
    :return: LMFit model object with initialized parameters, bounds, and constraints
    """
    max_dt = dt_axis[-1]
    min_dt = dt_axis[0]

    # model = lmfit.models.GaussianModel(prefix=prefix)
    model = lmfit.Model(gaussfunc, prefix=prefix)

    model_params = model.make_params()

    # set initial guesses and boundaries
    # todo: change to FWHM instead of sigma (add conversion)
    min_width = params_obj.gaussian_83_nonprot_width_min
    # min_width = params_obj.gaussian_72_prot_peak_width + 3 * params_obj.gaussian_73_prot_width_tol
    model_params[prefix + 'centroid'].set(guess_gaussian.centroid, min=min_dt, max=max_dt)
    model_params[prefix + 'sigma'].set(guess_gaussian.width, min=min_width, max=max_dt)
    model_params[prefix + 'amplitude'].set(guess_gaussian.amplitude, min=0, max=1.5)

    # todo: apply constraints - ** might actually need to be out in the main body IF doing any relative to other peaks. If just width, can be in here

    # return the model
    return model, model_params


def get_popt_from_lmoutput(modelresult, amp_cutoff):
    """
    Generate a list of parameters in the same format as curve_fit (popt) for easy conversion
    to old plotting and result saving methods.
    :param modelresult: ModelResult object from LMFit (returned from model.fit())
    :param amp_cutoff: minimum amplitude for a peak to be included
    :return: list of Gaussian parameters [amp1, cent1, sigma1, amp2, cent2, sigma2, ... ] for protein and non-protein components
    """
    # convert dictionary of key/value parameters into a single list of values
    keys = sorted(modelresult.best_values.keys())
    protein_output_popt = [modelresult.best_values[key] for key in keys if key.startswith(protein_prefix)]
    nonprotein_output_popt = [modelresult.best_values[key] for key in keys if key.startswith(nonprotein_prefix)]

    # remove low amplitude peaks
    protein_output_popt = remove_low_amp(protein_output_popt, amp_cutoff)
    nonprotein_output_popt = remove_low_amp(nonprotein_output_popt, amp_cutoff)

    return protein_output_popt, nonprotein_output_popt


def remove_low_amp(popt_list, amp_cutoff):
    """
    Helper method to remove low amplitude peaks for both protein and non-protein parameter lists
    :param popt_list: list of Gaussian parameters [amp1, centroid1, sigma1, amp2, centroid2, sigma2, ... ]
    :param amp_cutoff: minimum amplitude to allow
    :return: updated popt_list with low amplitude peaks removed
    """
    values_to_remove = []
    for index, value in enumerate(popt_list):
        if index % 3 == 0:
            current_amplitude = value
            if current_amplitude < amp_cutoff:
                values_to_remove.extend([popt_list[index], popt_list[index + 1], popt_list[index + 2]])
    for value in values_to_remove:
        popt_list.remove(value)

    return popt_list


# todo: deprecated
# def gaussian_fit_ciu(analysis_obj, params_obj):
#     """
#     Gaussian fitting module for single-gaussian analysis of CIU-type data. Determines estimated
#     initial parameters and fits a single Gaussian distribution to each column of the input ciu_data
#     matrix. Saves all output into a subfolder of filepath titled 'gausfitoutput'
#     :param analysis_obj: ciu analysis object in which to save all data. Uses obj.ciu_data, axes, etc
#      (ciu_data is 2D numpy, axis 0 = DT, axis 1 = CV) for all data handling
#      Smoothing, interpolation, cropping, etc should be done prior to running this method.
#     :type analysis_obj: CIUAnalysisObj
#     :param params_obj: Parameters object with gaussian parameter information
#     :type params_obj: Parameters
#     :return: saves all gaussian outputs into the ciu_obj and returns it
#     """
#     # initial setup
#     ciu_data = analysis_obj.ciu_data
#     dt_axis = analysis_obj.axes[0]    # drift time (DT) - x axis for fitting, y axis for final CIU plot
#     cv_axis = analysis_obj.axes[1]
#     filename = analysis_obj.short_filename
#     outputpath = os.path.join(os.path.dirname(analysis_obj.filename), filename)
#     if not os.path.isdir(outputpath):
#         os.makedirs(outputpath)
#
#     widthfrac = params_obj.gaussian_5_width_fraction
#     # todo: remove these from the params object (and replace with scaling values/etc for penalties)
#     # filter_width_max = params_obj.gaussian_3_width_max
#     # intensity_thr = params_obj.gaussian_2_int_threshold
#
#     intarray = np.swapaxes(ciu_data, 0, 1)
#
#     round1_diagnostics_container = FitDiagnostics(cv_axis)
#     round2_diagnostics_container = FitDiagnostics(cv_axis)
#
#     # for each CV column in the file, perform a multi-gaussian fitting and save information generated
#     print('\nFile ' + str(filename))
#
#     for cv_index, cv_col_intensities in enumerate(intarray):
#         print(cv_index + 1)
#         # Estimate initial guess data from peak fitting
#         all_peak_guesses = estimate_multi_params_all(cv_col_intensities, dt_axis, widthfrac)
#
#         param_guesses_multiple = []
#         all_fit_rounds = []
#
#         # set bounds for fitting: keep baseline and centroid on DT axis, amplitude 0 to 1.5, width 0 to len(dt_axis)
#         max_dt = dt_axis[len(dt_axis) - 1]
#         min_dt = dt_axis[0]
#         fit_bounds_lower, fit_bounds_upper = [], []
#         fit_bounds_lower_append = [0, min_dt, 0]
#         fit_bounds_upper_append = [1, max_dt, len(dt_axis)]
#
#         i = 0
#         iterate_gaussian_flag = True
#         # Iterate through peak detection until convergence criterion is met, adding one additional peak each iteration
#         while iterate_gaussian_flag:
#             # Set up initial guesses
#             try:
#                 param_guesses_multiple.extend(all_peak_guesses[i])
#                 # ensure bounds arrays maintain same shape as parameter guesses
#                 fit_bounds_lower.extend(fit_bounds_lower_append)
#                 fit_bounds_upper.extend(fit_bounds_upper_append)
#             except IndexError:
#                 # No converge with all estimated peaks. Continue with final estimate
#                 print('Included all {} peaks found, but r^2 still less than convergence criterion. '
#                       'Poor fitting possible'.format(i+1))
#                 break
#
#             # Run fitting (round 1)
#             try:
#                 popt, pcov = curve_fit(multi_gauss_func, dt_axis, cv_col_intensities, method='trf',
#                                        p0=param_guesses_multiple,
#                                        bounds=(fit_bounds_lower, fit_bounds_upper))
#                 # perr = np.sqrt(np.diag(pcov))
#             except (RuntimeError, ValueError):
#                 popt, pcov = [], []
#
#             current_fit = SingleFitStats(dt_axis, cv_col_intensities, popt=popt, cv=cv_axis[cv_index], amp_cutoff=params_obj.gaussian_2_int_threshold)
#             all_fit_rounds.append(current_fit)
#
#             # stop iterating once convergence criteria have been reached
#             if not current_fit.adjrsq < params_obj.gaussian_1_convergence:
#                 iterate_gaussian_flag = False
#
#             i += 1
#
#         print('performed {} iterations'.format(i))
#
#         # for round 1, best fit is the final one - use that for final fit data
#         round1_diagnostics_container.append_best_fit_info(all_fit_rounds[-1])
#         round1_diagnostics_container.all_fits_lists.append(all_fit_rounds)
#
#     # **************** Second round of fitting ***************
#     # perform second round of fitting on quad-selected/clean data
#     if params_obj.gaussian_3_mode == 'Mass Selected':
#         for cv_index, cv_col_intensities in enumerate(intarray):
#             # generate initial guesses using previous fits, extended by peak detection method
#             prev_gaussians = round1_diagnostics_container.gauss_lists[cv_index]
#             peak_guesses = []
#             for gaussian in prev_gaussians:
#                 peak_guesses.append(gaussian.return_popt())
#             peak_guesses.extend(estimate_multi_params_all(cv_col_intensities, dt_axis, widthfrac))
#
#             # set bounds for fitting: keep baseline and centroid on DT axis, amplitude 0 to 1.5, width 0 to len(dt_axis)
#             max_dt = dt_axis[len(dt_axis) - 1]
#             min_dt = dt_axis[0]
#             fit_bounds_lower, fit_bounds_upper = [], []
#             fit_bounds_lower_append = [0, min_dt, 0]
#             fit_bounds_upper_append = [1, max_dt, len(dt_axis)]
#
#             iteration_peaks = len(peak_guesses)
#             max_peaks = 7
#             if iteration_peaks > max_peaks:
#                 iteration_peaks = max_peaks  # cap the max num peaks
#
#             num_peaks = 0
#             r2_all_fits = []
#             r2_final_fits = []
#             current_param_guesses = []
#             scores = []
#             # popt_iterations, pcov_iterations = [], []
#             while num_peaks < iteration_peaks:
#                 print('second round, iteration {}'.format(num_peaks + 1))
#                 # update initial guesses and bounds for curve_fit
#                 try:
#                     current_param_guesses.extend(peak_guesses[num_peaks])
#                     fit_bounds_lower.extend(fit_bounds_lower_append)
#                     fit_bounds_upper.extend(fit_bounds_upper_append)
#                 except IndexError:
#                     # No converge with all estimated peaks. Continue with final estimate
#                     print('Included all {} peaks found, but r^2 still less than convergence criterion. '
#                           'Poor fitting possible'.format(num_peaks+1))
#                     break
#
#                 # perform curve fitting
#                 try:
#                     popt, pcov = curve_fit(multi_gauss_func, dt_axis, cv_col_intensities, method='trf',
#                                            p0=current_param_guesses,
#                                            bounds=(fit_bounds_lower, fit_bounds_upper))
#                 except (RuntimeError, ValueError):
#                     popt, pcov = [], []
#
#                 # compute fits and score
#                 current_fit = SingleFitStats(dt_axis, cv_col_intensities, popt=popt, cv=cv_axis[cv_index], amp_cutoff=params_obj.gaussian_2_int_threshold)
#                 penalty_scaling = 1
#
#                 current_fit.score, current_fit.peak_penalties = compute_fit_score(current_fit.gaussians, current_fit.adjrsq, dt_axis, penalty_scaling)
#                 # current_fit.plot_fit()
#
#                 # remove any poorly fitted peaks and re-fit/score (if any were removed)
#                 penalty_cutoff = 0.5
#                 updated_gaussians, removed_bool = remove_penalized_peaks(current_fit.gaussians, current_fit.peak_penalties, penalty_cutoff)
#                 # if any peaks were removed, update fits and scoring
#                 if removed_bool:
#                     # create a new fit without the removed peaks
#                     # try:
#                     #     new_guess = []
#                     #     for gaussian in updated_gaussians:
#                     #         new_guess.extend(gaussian.return_popt())
#                     #     new_popt, new_pcov = curve_fit(multi_gauss_func, dt_axis, cv_col_intensities, method='trf',
#                     #                                    p0=new_guess,
#                     #                                    bounds=(fit_bounds_lower, fit_bounds_upper))
#                     # except (RuntimeError, ValueError):
#                     #     new_popt, new_pcov = [], []
#
#                     new_popt = []
#                     for gaussian in updated_gaussians:
#                         new_popt.extend(gaussian.return_popt())
#
#                     updated_fit = SingleFitStats(dt_axis, cv_col_intensities, popt=new_popt, cv=cv_axis[cv_index], amp_cutoff=params_obj.gaussian_2_int_threshold)
#                     updated_fit.score, updated_fit.peak_penalties = compute_fit_score(updated_fit.gaussians, updated_fit.adjrsq, dt_axis, penalty_scaling)
#                     print('old score: {:.3f} {} peaks, new score: {:.3f} {} peaks'.format(current_fit.score, len(current_fit.gaussians), updated_fit.score, len(updated_fit.gaussians)))
#
#                     # updated_fit.plot_fit()
#
#                     r2_all_fits.append([current_fit, updated_fit])
#                     r2_final_fits.append(updated_fit)
#                     final_fit = updated_fit
#                 else:
#                     r2_all_fits.append(current_fit)
#                     r2_final_fits.append(current_fit)
#                     final_fit = current_fit
#
#                 # cut off scores that are dropping consistently
#                 scores.append(final_fit.score)
#                 cutoff_after_drops = 3
#                 if len(scores) >= cutoff_after_drops + 1:
#                     score_drops = []
#                     for test_score in scores[-1 - cutoff_after_drops - 1:]:
#                         # determine if the current score is improved over the score from n rounds ago
#                         score_drops.append(final_fit.score > test_score)
#                     # if the current score is not higher than the score in any of the previous n rounds, stop iterating
#                     if True not in score_drops:
#                         break
#
#                 num_peaks += 1
#
#             # get best score and use that data
#             best_score_index = int(np.argmax(scores))
#             print('best score {:.3f} with {} peaks'.format(scores[best_score_index], best_score_index + 1))
#
#             best_fit = r2_final_fits[best_score_index]
#             round2_diagnostics_container.all_fits_lists.append(r2_all_fits)
#             round2_diagnostics_container.append_best_fit_info(best_fit)
#
#     analysis_obj.filtered_gaussians = round1_diagnostics_container.gauss_lists  # gaussians_with
#     analysis_obj.gaussians = round2_diagnostics_container.gauss_lists   # gaussians_with
#
#     analysis_obj.gauss_adj_r2s = round2_diagnostics_container.adjrsq_list
#     analysis_obj.gauss_fits = [fit.y_fit for fit in round2_diagnostics_container.fit_stats_list]
#     analysis_obj.gauss_r2s = [fit.rvalue ** 2 for fit in round2_diagnostics_container.fit_stats_list]
#     analysis_obj.gauss_covariances = [fit.pcov for fit in round2_diagnostics_container.fit_stats_list]
#
#     if params_obj.gaussian_4_save_diagnostics:
#         save_gaussfits_pdf(analysis_obj, round1_diagnostics_container.gauss_lists, outputpath, filename_append='_r1')
#         if len(round2_diagnostics_container.gauss_lists) > 0:
#             save_gaussfits_pdf(analysis_obj, round2_diagnostics_container.gauss_lists, outputpath, filename_append='_r2')
#
#         filt_centroids = analysis_obj.get_attribute_by_cv('centroid', filtered=True)
#         plot_centroids(filt_centroids, analysis_obj, outputpath)
#         plot_fwhms(analysis_obj, outputpath)
#         save_gauss_params(analysis_obj, outputpath)
#
#     return analysis_obj


def remove_penalized_peaks(gaussian_list, peak_penalties, penalty_cutoff):
    """
    Remove any peaks penalized above the acceptable (cutoff) value. Returns the (possibly shortened)
    popt list a boolean (True if any peaks were removed)
    :param gaussian_list: list gaussians from curve_fit
    :param peak_penalties: list of peak penalties from compute_fit_score
    :param penalty_cutoff: float - threshold above which to remove a peak
    :return: filtered_popt list, peaks_removed boolean
    """
    final_gaussians = []
    any_peaks_removed = False
    for index, gaussian in enumerate(gaussian_list):
        # if the penalty for this Gaussian is below cutoff, include it.
        if peak_penalties[index] < penalty_cutoff:
            final_gaussians.append(gaussian)
        else:
            any_peaks_removed = True

    return final_gaussians, any_peaks_removed


def compute_width_penalty(input_width, expected_width, tolerance, steepness):
    """

    :param input_width:
    :param expected_width:
    :param tolerance:
    :param steepness:
    :return:
    """
    diff = abs(input_width - expected_width)
    if diff < tolerance:
        return 0
    else:
        penalized_width = abs(diff - tolerance)
        return steepness * penalized_width


def compute_low_amp_penalty(gaussian):
    """
    Additional penalty for Gaussians with amplitude below cutoff value. Iterative fitting requires
    allowing these peaks to be fit in cases where there are fewer peaks than proposed, but they
    should be filtered after to prevent bad fits.

    Penalty = e ^ (- (scaling) * x) gives a max penalty of 1 (amp = 0), increasing exponentially with
    values becoming noticeable below amplitude = 1/scaling

    :param gaussian: Gaussian container with fit information
    :type gaussian: Gaussian
    :return: penalty value (float)
    """
    amp_scale = 100
    return np.exp(-1 * amp_scale * gaussian.amplitude)


def compute_area_penalty(gaussian, list_of_gaussians, dt_axis):
    """
    Shared area penalty intended to penalize peaks that are almost completely overlapped
    by others.
    :param gaussian: Gaussian object to compare agaisnt the rest of the list
    :param list_of_gaussians: all gaussians currently fit at this CV
    :param dt_axis: x axis array over which to compute overlap
    :return: penalty (float)
    """
    # for this gaussian, compute how much area it shares with the rest of the list
    total_penalty = 0
    # for gaussian in list_of_gaussians:
    my_area = scipy.integrate.trapz(gaussfunc(dt_axis, *gaussian.return_popt()), dt_axis)

    other_gaussians = [x for x in list_of_gaussians if x is not gaussian]
    shared_areas = []
    for other in other_gaussians:
        shared_areas.append(shared_area_gauss(dt_axis, gaussian.return_popt(), other.return_popt()))

    # compute shared area (ratio from 0 to 1) and any penalties if > 0.25 (not much until 0.5)
    max_shared_area = np.max(shared_areas)
    shared_area_ratio = max_shared_area / my_area
    if shared_area_ratio > 0.25:
        # antibody settings: 1.5 - 1.8 * shared area ratio
        my_penalty = (1.25 * shared_area_ratio - 0.25) ** 4

        # scale penalty by area, so smaller peaks don't get overly penalized (areas often < 1)
        # my_penalty *= (my_area * 5)
        total_penalty += my_penalty

    return total_penalty


def shared_area_gauss(x_axis, gauss1_params, gauss2_params):
    """
    Compute a "shared area score" (shared area normalized against the area of the smaller peak being compared)
    and return it.
    :param x_axis: x-axis on which to plot the gaussian functions (doesn't matter as long as it's sufficiently sampled)
    :param gauss1_params: the parameters describing gaussian 1 [amplitude, centroid, width]
    :param gauss2_params: the parameters describing gaussian 2 [amplitude, centroid, width]
    :return: shared area
    """
    # shared area is the area under the lower curve
    gauss1 = gaussfunc(x_axis, *gauss1_params)
    gauss2 = gaussfunc(x_axis, *gauss2_params)
    shared_area_arr = []

    # for each point along the x (DT) axis, determine the amount of shared area
    for index in np.arange(0, len(x_axis)):
        if gauss1[index] > gauss2[index]:
            shared_area_arr.append(gauss2[index])
        else:
            shared_area_arr.append(gauss1[index])

    # return the integrated area over the provided axis
    return scipy.integrate.trapz(shared_area_arr, x_axis)


def generate_gaussians_from_popt(opt_params_list, protein_bool, cv=None, pcov=None):
    """
    Convert a list of parameters to a list of Gaussian objects. Initializes Gaussians with a collision voltage
    and covariance matrix if provided.
    :param opt_params_list: list of parameters [amp, centroid, width, amp2, cent2, width2, ... ]
    :param protein_bool: protein Gaussians (True) or non-protein (False)
    :param cv: (optional) collision voltage to associate with all Gaussians
    :param pcov: (optional) covariance matrix from fitting to associate with all Gaussians
    :return: list of Gaussian objects from params list
    :rtype: list[Gaussian]
    """
    index = 0
    gaussian_list = []
    while index < len(opt_params_list):
        gaussian_list.append(Gaussian(amplitude=opt_params_list[index],
                                      centroid=opt_params_list[index + 1],
                                      width=opt_params_list[index + 2],
                                      collision_voltage=cv,
                                      pcov=pcov,
                                      protein_bool=protein_bool))
        index += 3
    return gaussian_list


def save_fits_pdf_new(analysis_obj, best_fit_list, outputpath):
    """

    :param analysis_obj:
    :param best_fit_list:
    :param outputpath:
    :return:
    """
    gauss_name = analysis_obj.short_filename + '_fits.pdf'
    pdf_fig = matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputpath, gauss_name))

    intarray = np.swapaxes(analysis_obj.ciu_data, 0, 1)
    for cv_index in range(len(analysis_obj.axes[1])):
        plt.figure()
        best_fit = best_fit_list[cv_index]

        # plot the original raw data as a scatter plot
        plt.scatter(analysis_obj.axes[0], intarray[cv_index])

        # plot the combined 'best fit' data
        plt.plot(best_fit.x_data, best_fit.y_fit, color='black')

        # plot each component individually
        prot_index = 1
        for prot_gauss in best_fit.gaussians_protein:
            gauss_fit = gaussfunc(best_fit.x_data, prot_gauss.amplitude, prot_gauss.centroid, prot_gauss.width)
            plt.plot(best_fit.x_data, gauss_fit, ls='--', label='Protein {}'.format(prot_index))
            prot_index += 1
        nonprot_index = 1
        for nonprot_gauss in best_fit.gaussians_nonprotein:
            gauss_fit = gaussfunc(best_fit.x_data, nonprot_gauss.amplitude, nonprot_gauss.centroid, nonprot_gauss.width)
            plt.plot(best_fit.x_data, gauss_fit, ls='--', label='Non-Protein{}'.format(nonprot_index))
            nonprot_index += 1

        plt.legend(loc='best')

        penalty_string = ['{:.2f}'.format(x) for x in best_fit.peak_penalties]
        plt.title('{}V, r2: {:.3f}, score: {:.4f}, peak pens: {}'.format(analysis_obj.axes[1][cv_index], best_fit.adjrsq, best_fit.score,
                                                                         ','.join(penalty_string)))

        pdf_fig.savefig()
        plt.close()
    pdf_fig.close()


def save_gaussfits_pdf(analysis_obj, gaussian_list, outputpath, filename_append='', rsq_list=None):
    """
    Save a pdf containing an image of the data and gaussian fit for each column to pdf in outputpath.
    :param analysis_obj: container with gaussian fits to save
    :type analysis_obj: CIUAnalysisObj
    :param gaussian_list: list of gaussians to plot
    :type gaussian_list: list[list[Gaussian]]
    :param outputpath: directory in which to save output
    :param rsq_list: list of adjusted r2 values by CV for captioning (optional)
    :param filename_append: (optional) string to add to filename
    :return: void
    """
    # TODO: make all plot parameters accessible

    gauss_name = analysis_obj.short_filename + filename_append + '_gaussFit.pdf'
    pdf_fig = matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputpath, gauss_name))

    intarray = np.swapaxes(analysis_obj.ciu_data, 0, 1)
    for cv_index in range(len(analysis_obj.axes[1])):
        plt.figure()
        # plot the original raw data as a scatter plot
        plt.scatter(analysis_obj.axes[0], intarray[cv_index])

        # plot each fitted gaussian and centroid
        sum_fit = np.zeros(len(analysis_obj.axes[0]))
        for gaussian in gaussian_list[cv_index]:
            fit = gaussfunc(analysis_obj.axes[0], gaussian.amplitude, gaussian.centroid, gaussian.width)
            sum_fit += fit
            plt.plot(analysis_obj.axes[0], fit)
            plt.plot(gaussian.centroid, abs(gaussian.amplitude), '+', color='red')

        # also plot the sum of all Gaussian fits for reference
        plt.plot(analysis_obj.axes[0], sum_fit, ls='--', color='black')

        if rsq_list is not None:
            plt.title('CV: {}, R2: {:.3f}'.format(analysis_obj.axes[1][cv_index], rsq_list[cv_index]))
        else:
            plt.title('CV: {}'.format(analysis_obj.axes[1][cv_index]))

        pdf_fig.savefig()
        plt.close()
    pdf_fig.close()


def plot_centroids(centroid_lists_by_cv, analysis_obj, outputpath, y_bounds=None, nonprotein_centroids=None):
    """
    Save a png image of the centroid DTs fit by gaussians
    :param centroid_lists_by_cv: list of [list of centroids]s at each collision voltage
    :param nonprotein_centroids: non-protein components list of [list of centroids]s at each collision voltage
    :param analysis_obj: container with gaussian fits to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :param y_bounds: [lower bound, upper bound] to crop the plot to (in y-axis units, typically ms)
    :return: void
    """
    # TODO: make all plot parameters accessible
    plt.clf()

    # plot centroids at each collision voltage
    for x, y in zip(analysis_obj.axes[1], centroid_lists_by_cv):
        plt.scatter([x] * len(y), y, color='b')

    # plot non-protein components in red if they are present
    if nonprotein_centroids is not None:
        for x, y in zip(analysis_obj.axes[1], nonprotein_centroids):
            try:
                plt.scatter([x] * len(y), y, color='r')
            except TypeError:
                # empty list - continue to next cv
                continue

    # plt.scatter(self.axes[1], self.gauss_centroids)
    plt.xlabel('Collision Voltage (V)')
    plt.ylabel('ATD Centroid (ms)')
    if y_bounds is not None:
        plt.ylim(y_bounds)
    # plt.title('Centroids filtered by peak width')
    plt.grid('on')
    output_name = os.path.basename(analysis_obj.filename).rstrip('.ciu') + '_centroids.png'
    plt.savefig(os.path.join(outputpath, output_name), dpi=500)
    plt.close()
    # print('Saved TrapCVvsArrivtimecentroid ' + str(analysis_obj.raw_obj.filename) + '_.png')


def plot_fwhms(analysis_obj, outputpath):
    """
    Save a png image of the FWHM (widths) fit by gaussians.
    :param analysis_obj: container with gaussian fits to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :return: void
    """
    print('Saving TrapcCVvsFWHM_' + str(analysis_obj.raw_obj.filename) + '_.png .....')
    gauss_fwhms = analysis_obj.get_attribute_by_cv('fwhm', False)

    for x, y in zip(analysis_obj.axes[1], gauss_fwhms):
        plt.scatter([x] * len(y), y)
    # plt.scatter(self.axes[1], self.gauss_fwhms)
    plt.xlabel('Trap CV')
    plt.ylabel('ATD_FWHM')
    plt.grid('on')
    output_name = os.path.basename(analysis_obj.filename).rstrip('.ciu') + '_FWHM.png'
    plt.savefig(os.path.join(outputpath, output_name), dpi=300)
    plt.close()
    # print('Saving TrapCVvsFWHM_' + str(analysis_obj.raw_obj.filename) + '_.png')


def save_gauss_params(analysis_obj, outputpath):
    """
    Save all gaussian information to file
    :param analysis_obj: container with gaussian fits to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :return: void
    """
    output_name = os.path.basename(analysis_obj.filename).rstrip('.ciu') + '_gaussians.csv'
    with open(os.path.join(outputpath, output_name), 'w') as output:
        # save DT information too to allow for reconstruction
        dt_line = ','.join([str(x) for x in analysis_obj.axes[0]])
        output.write('Drift axis:,' + dt_line + '\n')
        output.write('# Filtered Gaussians (round 1)\n')
        output.write('# Trap CV,Amplitude,Centroid,Peak Width\n')
        index = 0
        while index < len(analysis_obj.axes[1]):
            # outputline = '{},'.format(self.axes[1][index])
            outputline = ','.join([gaussian.print_info() for gaussian in analysis_obj.protein_gaussians[index]])
            # outputline += ','.join(['{:.2f}'.format(x) for x in self.gauss_filt_params[index]])
            output.write(outputline + '\n')
            index += 1

        index = 0
        output.write('# All Gaussians (round 2)\n')
        output.write('# Trap CV,Amplitude,Centroid,Peak Width\n')
        while index < len(analysis_obj.axes[1]):
            # gauss_line = '{:.3f},{:.3f},'.format(analysis_obj.gauss_r2s[index], analysis_obj.gauss_adj_r2s[index])
            # gauss_line += ','.join(['{:.2f}'.format(x) for x in self.gauss_params[index]])
            gauss_line = ','.join([gaussian.print_info() for gaussian in analysis_obj.gaussians[index]])

            # gauss_line += ','.join([str(x) for x in self.gauss_params[index]])
            output.write(gauss_line + '\n')
            index += 1


if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
    for file in files:
        with open(file, 'rb') as analysis_file:
            current_analysis_obj = pickle.load(analysis_file)
        # gaussian_fit_ciu(current_analysis_obj, current_analysis_obj.params)  # NOTE: analysis_obj.params is DEPRECATED
