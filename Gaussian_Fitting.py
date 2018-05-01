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
import itertools
import math
import peakutils
import pickle
import tkinter
from tkinter import filedialog
import scipy.signal
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

from CIU_raw import CIURaw
from CIU_analysis_obj import CIUAnalysisObj

# imports for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CIU_Params import Parameters


class Gaussian(object):
    """
    Container for fitted gaussian information. Holds fit parameters and any associated metadata
    """
    def __init__(self, amplitude, centroid, width, collision_voltage, pcov):
        self.amplitude = amplitude
        self.centroid = centroid
        self.width = width
        self.cv = collision_voltage
        self.fwhm = 2*(math.sqrt(2*math.log(2)))*self.width
        self.resolution = self.centroid/self.fwhm
        self.fit_covariances = pcov
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
        return '{},{:.2f},{:.2f},{:.1f},{:.1f},{:.1f}'.format(self.cv,
                                                              self.centroid,
                                                              self.amplitude,
                                                              self.width,
                                                              self.fwhm,
                                                              self.resolution)

    def return_popt(self):
        """
        Re-generate Gaussian function parameter list (e.g. popt style from curve_fit) from
        gaussian object
        :return: [baseline, amplitude, centroid, width]
        """
        return [self.amplitude, self.centroid, self.width]


class FitDiagnostics(object):
    """
    Container for diagnostics and information from Gaussian fitting. Intended to enable rapid diagnostics
    and hold information from multi-round fitting and ease plotting/saving various outputs.
    Organization: A single FitDiagnostics object for a CIUAnalysis object containing all diagnostic
    information in lists organized by collision voltage.
    """

    def __init__(self, cv_axis):
        """
        Initialize an empty diagnostics object over the collision voltages of a CIU fingerprint
        :param cv_axis: collision voltage axis of the associated CIU analysis
        """
        self.cvs = cv_axis

        # First round fitting diagnostics
        self.num_peaks_list = []
        self.popt_lists = []
        self.gauss_lists = []
        self.filt_gauss_lists = []
        self.fit_stats_list = []
        self.adjrsq_list = []

        self.all_fits_lists = []    # list of (lists-by-CV) containing ALL fits performed at that CV (SingleFitStats)

        # todo: MAKE TWO diagnostics objects (or however many), one for each round
        # self.r2_num_peaks_list = []
        # self.r2_popt_lists = []
        # self.r2_gauss_lists = []
        # self.r2_fit_stats_list = []
        # self.r2_adjrsq_list = []

    def append_best_fit_info(self, fit_stats_obj):
        """
        Add the best result from a particular collision voltage to the primary lists stored
        in this object
        :param fit_stats_obj: container with *best* fit information (out of all fits at this voltage)
        :type fit_stats_obj: SingleFitStats
        :return: void
        """
        self.fit_stats_list.append(fit_stats_obj)
        self.popt_lists.append(fit_stats_obj.get_popt())
        self.gauss_lists.append(fit_stats_obj.gaussians)
        self.adjrsq_list.append(fit_stats_obj.adjrsq)


class SingleFitStats(object):
    """
    Container for holding fit information for a single multi-Gaussian fitting (one collision voltage).
    Includes r2, fit data, error estimates, etc (all output of linregress)
    Intended to use called when initializing a fit.
    """
    def __init__(self, x_data, y_data, popt, cv):
        """
        Initialize a new fit between the provided x/y data and optimized Gaussian parameters
        :param x_data: x (DT) raw data being fit by popt
        :param y_data: y (intensity) raw data being fit by popt
        :param popt: optimized parameters returned from curve_fit
        :param cv: collision voltage at which this fit occurred
        """
        self.x_data = x_data
        self.y_data = y_data
        self.y_fit = multi_gauss_func(x_data, *popt)
        self.slope, self.intercept, self.rvalue, self.pvalue, self.stderr = linregress(y_data, self.y_fit)
        self.adjrsq = adjrsquared(self.rvalue ** 2, len(y_data))
        self.gaussians = generate_gaussians_from_popt(popt, cv)

        # additional information that may be present
        self.p0 = None      # initial guess array used to generate this popt
        self.pcov = None    # output covariance matrix

        self.score = None   # score from second round fitting (r2 - penalties)
        self.peak_penalties = None      # list of penalties for each peak in the Gaussian list

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


def gaussfunc(x, a, xc, w):
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


def reconstruct_from_fits(analysis_obj):
    """
    Construct a new analysis object using the filtered Gaussian fits of the provided analysis object
    as the raw data. Must have previously performed Gaussian feature detection on the provided analysis_obj
    :param analysis_obj: CIU container with original data and **gaussian feature detection previously performed**
    :type analysis_obj: CIUAnalysisObj
    :return: new CIUAnalysisObj with reconstructed raw data
    :rtype: CIUAnalysisObj
    """
    ciu_data_by_cols = []
    dt_axis = analysis_obj.axes[0]
    # construct the raw data at each collision voltage to stitch together into a CIU matrix
    for cv_gauss_list in analysis_obj.filtered_gaussians:
        # assemble all the parameters for Gaussians at this CV
        all_params = []
        for gaussian in cv_gauss_list:
            all_params.extend(gaussian.return_popt())

        # Use the Gaussian function to construct intensity data at each DT
        intensities = multi_gauss_func(dt_axis, *all_params)

        ciu_data_by_cols.append(intensities)

    # finally, transpose the CIU data to match the typical format and return the object
    final_data = np.asarray(ciu_data_by_cols).T
    raw_obj = CIURaw(final_data, dt_axis, analysis_obj.axes[1], analysis_obj.filename)
    new_analysis_obj = CIUAnalysisObj(raw_obj, final_data, analysis_obj.axes, analysis_obj.params)
    new_analysis_obj.short_filename = analysis_obj.short_filename + '_denoised'
    return new_analysis_obj


def check_peak_dist(popt_list, current_guess_list, min_distance_dt, max_peak_width):
    """

    DEPRECATED

    Determine whether the centroid of the current guess is too close to an existing (already fit) peak.
    Note: excludes peaks above the width cutoff, as these are not used for feature detection/etc anyway
    and may overlap substantially with signal peaks (removing them may negatively impact fitting)
    :param popt_list: Current optimized parameter list (flat, as returned from curve_fit)
    :param current_guess_list: list of parameters for current guess [y0, amplitude, centroid, width]
    :param min_distance_dt: minimum distance between peaks in drift axis units
    :param max_peak_width: max width for filtering from Parameters object
    :return: boolean; True if distance is greater than minimum
    """
    # automatically allow any peaks that are too wide, as these will not impact feature detection/etc
    if current_guess_list[2] > max_peak_width:
        return True

    guess_centroid = current_guess_list[1]
    existing_centroids = popt_list[1::3]
    # return false if any existing centroid is too close to the current guess, excluding noise peaks
    for existing_centroid in existing_centroids:
        if abs(existing_centroid - guess_centroid) < min_distance_dt:
            return False
    return True


def check_peak_dists(popt_list, params_obj):
    """
    Look through all fitted peak parameters and determine if any peaks are too close to each other.
    If so, return True and the parameters list with the lower intensity of the overlapping peaks removed.
    WILL ONLY REMOVE 1 PEAK - since it is intended to function in the loop as peaks are added one
    at a time, if a peak is added too close to another, it is removed and the iteration is stopped.
    :param popt_list: list of output/fitted gaussian parameters
    :param params_obj: Parameters object
    :type params_obj: Parameters
    :return: boolean (True if peaks are too close), updated popt_list with 'bad' peak removed
    """
    # convert popt_list to Gaussian objects for easier handling
    index = 0
    gaussians = []
    while index < len(popt_list):
        gaussian = Gaussian(popt_list[index], popt_list[index + 1], popt_list[index + 2], None, None)
        # ignore peaks that are above width max - they are allowed to be close to others (noise can be anywhere)
        if not gaussian.width > params_obj.gaussian_3_width_max:
            gaussians.append(gaussian)
        index += 3

    # examine all Gaussians and determine if any are too close together
    for gaussian_combo in itertools.combinations(gaussians, 2):
        gaussian1 = gaussian_combo[0]
        gaussian2 = gaussian_combo[1]
        if abs(gaussian1.centroid - gaussian2.centroid) < params_obj.gaussian_6_min_peak_dist:
            print('dist too close, was: {:.2f}'.format(abs(gaussian1.centroid - gaussian2.centroid)))
            # these peaks are too close, return
            if gaussian1.amplitude > gaussian2.amplitude:
                gaussians.remove(gaussian2)
            else:
                gaussians.remove(gaussian1)

            # reassemble popt_list
            final_popt = []
            for gaussian in gaussians:
                final_popt.extend(gaussian.return_popt())
            return True, final_popt

    # if no peaks are too close, return False and the original popt list
    return False, popt_list


def gaussian_fit_ciu(analysis_obj, params_obj):
    """
    Gaussian fitting module for single-gaussian analysis of CIU-type data. Determines estimated
    initial parameters and fits a single Gaussian distribution to each column of the input ciu_data
    matrix. Saves all output into a subfolder of filepath titled 'gausfitoutput'
    :param analysis_obj: ciu analysis object in which to save all data. Uses obj.ciu_data, axes, etc
     (ciu_data is 2D numpy, axis 0 = DT, axis 1 = CV) for all data handling
     Smoothing, interpolation, cropping, etc should be done prior to running this method.
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with gaussian parameter information
    :type params_obj: Parameters
    :return: saves all gaussian outputs into the ciu_obj and returns it
    """
    # initial setup
    ciu_data = analysis_obj.ciu_data
    dt_axis = analysis_obj.axes[0]    # drift time (DT) - x axis for fitting, y axis for final CIU plot
    cv_axis = analysis_obj.axes[1]
    filename = analysis_obj.short_filename
    outputpath = os.path.join(os.path.dirname(analysis_obj.filename), filename)
    if not os.path.isdir(outputpath):
        os.makedirs(outputpath)

    widthfrac = params_obj.gaussian_5_width_fraction
    # todo: remove these from the params object (and replace with scaling values/etc for penalties)
    filter_width_max = params_obj.gaussian_3_width_max
    intensity_thr = params_obj.gaussian_2_int_threshold

    intarray = np.swapaxes(ciu_data, 0, 1)

    round1_diagnostics_container = FitDiagnostics(cv_axis)
    round2_diagnostics_container = FitDiagnostics(cv_axis)

    # for each CV column in the file, perform a multi-gaussian fitting and save information generated
    print('\nFile ' + str(filename))

    for cv_index, cv_col_intensities in enumerate(intarray):
        print(cv_index + 1)
        # Estimate initial guess data from peak fitting
        all_peak_guesses = estimate_multi_params_all(cv_col_intensities, dt_axis, widthfrac)

        param_guesses_multiple = []
        all_fit_rounds = []

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
                      'Poor fitting possible'.format(i+1))
                break

            # Run fitting (round 1)
            try:
                popt, pcov = curve_fit(multi_gauss_func, dt_axis, cv_col_intensities, method='trf',
                                       p0=param_guesses_multiple,
                                       bounds=(fit_bounds_lower, fit_bounds_upper))
                # perr = np.sqrt(np.diag(pcov))
            except (RuntimeError, ValueError):
                popt, pcov = [], []

            current_fit = SingleFitStats(dt_axis, cv_col_intensities, popt, cv_axis[cv_index])
            all_fit_rounds.append(current_fit)

            # stop iterating once convergence criteria have been reached
            if not current_fit.adjrsq < params_obj.gaussian_1_convergence:
                iterate_gaussian_flag = False

            i += 1

        print('performed {} iterations'.format(i))

        # for round 1, best fit is the final one - use that for final fit data
        round1_diagnostics_container.append_best_fit_info(all_fit_rounds[-1])
        round1_diagnostics_container.all_fits_lists.append(all_fit_rounds)

    for cv_index, cv_col_intensities in enumerate(intarray):
        # generate initial guesses using previous fits, extended by peak detection method
        prev_gaussians = round1_diagnostics_container.gauss_lists[cv_index]
        peak_guesses = []
        for gaussian in prev_gaussians:
            peak_guesses.append(gaussian.return_popt())
        peak_guesses.extend(estimate_multi_params_all(cv_col_intensities, dt_axis, widthfrac))

        # set bounds for fitting: keep baseline and centroid on DT axis, amplitude 0 to 1.5, width 0 to len(dt_axis)
        max_dt = dt_axis[len(dt_axis) - 1]
        min_dt = dt_axis[0]
        fit_bounds_lower, fit_bounds_upper = [], []
        fit_bounds_lower_append = [0, min_dt, 0]
        fit_bounds_upper_append = [1, max_dt, len(dt_axis)]

        iteration_peaks = len(peak_guesses)
        max_peaks = 7
        if iteration_peaks > max_peaks:
            iteration_peaks = max_peaks  # cap the max num peaks

        num_peaks = 0
        r2_all_fits = []
        r2_final_fits = []
        current_param_guesses = []
        scores = []
        # popt_iterations, pcov_iterations = [], []
        while num_peaks < iteration_peaks:
            print('second round, iteration {}'.format(num_peaks + 1))
            # update initial guesses and bounds for curve_fit
            try:
                current_param_guesses.extend(peak_guesses[num_peaks])
                fit_bounds_lower.extend(fit_bounds_lower_append)
                fit_bounds_upper.extend(fit_bounds_upper_append)
            except IndexError:
                # No converge with all estimated peaks. Continue with final estimate
                print('Included all {} peaks found, but r^2 still less than convergence criterion. '
                      'Poor fitting possible'.format(num_peaks+1))
                break

            # perform curve fitting
            try:
                popt, pcov = curve_fit(multi_gauss_func, dt_axis, cv_col_intensities, method='trf',
                                       p0=current_param_guesses,
                                       bounds=(fit_bounds_lower, fit_bounds_upper))
            except (RuntimeError, ValueError):
                popt, pcov = [], []

            # compute fits and score
            current_fit = SingleFitStats(dt_axis, cv_col_intensities, popt, cv_axis[cv_index])
            penalty_scaling = 0.5
            current_fit.score, current_fit.peak_penalties = compute_fit_score(current_fit.gaussians, current_fit.adjrsq, dt_axis, penalty_scaling)

            current_fit.plot_fit()

            # remove any poorly fitted peaks and re-fit/score (if any were removed)
            penalty_cutoff = 0.5
            updated_gaussians, removed_bool = remove_penalized_peaks(current_fit.gaussians, current_fit.peak_penalties, penalty_cutoff)
            # if any peaks were removed, update fits and scoring
            if removed_bool:
                # create a new fit without the removed peaks
                try:
                    new_guess = current_fit.get_popt()
                    new_popt, new_pcov = curve_fit(multi_gauss_func, dt_axis, cv_col_intensities, method='trf',
                                                   p0=current_fit.get_popt(),
                                                   bounds=(fit_bounds_lower, fit_bounds_upper))
                except (RuntimeError, ValueError):
                    new_popt, new_pcov = [], []

                # for gaussian in updated_gaussians:
                #     new_popt.extend(gaussian.return_popt())
                updated_fit = SingleFitStats(dt_axis, cv_col_intensities, new_popt, cv_axis[cv_index])
                updated_fit.score, updated_fit.peak_penalties = compute_fit_score(updated_fit.gaussians, updated_fit.adjrsq, dt_axis, penalty_scaling)
                print('old score: {:.3f} {} peaks, new score: {:.3f} {} peaks'.format(current_fit.score, len(current_fit.gaussians), updated_fit.score, len(updated_fit.gaussians)))

                updated_fit.plot_fit()

                r2_all_fits.append([current_fit, updated_fit])
                r2_final_fits.append(updated_fit)
                final_fit = updated_fit
            else:
                r2_all_fits.append(current_fit)
                r2_final_fits.append(current_fit)
                final_fit = current_fit

            # cut off scores that are dropping consistently
            scores.append(final_fit.score)
            cutoff_after_drops = 3
            if len(scores) >= cutoff_after_drops + 1:
                score_drops = []
                for test_score in scores[-1 - cutoff_after_drops - 1:]:
                    # determine if the current score is improved over the score from n rounds ago
                    score_drops.append(final_fit.score > test_score)
                # if the current score is not higher than the score in any of the previous n rounds, stop iterating
                if True not in score_drops:
                    break

            num_peaks += 1

        # get best score and use that data
        best_score_index = int(np.argmax(scores))
        print('best score {:.3f} with {} peaks'.format(scores[best_score_index], best_score_index + 1))

        best_fit = r2_final_fits[best_score_index]
        round2_diagnostics_container.all_fits_lists.append(r2_all_fits)
        round2_diagnostics_container.append_best_fit_info(best_fit)

    # todo: edit CIUAnalysisObj for better saving (only save the rounds diagnostic containers and "gaussians")
    analysis_obj.filtered_gaussians = round2_diagnostics_container.gauss_lists  # gaussians_with
    analysis_obj.gaussians = round2_diagnostics_container.gauss_lists   # gaussians_with

    analysis_obj.gauss_adj_r2s = round2_diagnostics_container.adjrsq_list
    analysis_obj.gauss_fits = [fit.y_fit for fit in round2_diagnostics_container.fit_stats_list]
    analysis_obj.gauss_r2s = [fit.rvalue ** 2 for fit in round2_diagnostics_container.fit_stats_list]
    analysis_obj.gauss_covariances = [fit.pcov for fit in round2_diagnostics_container.fit_stats_list]

    if params_obj.gaussian_4_save_diagnostics:
        save_gaussfits_pdf(analysis_obj, outputpath)
        plot_centroids(analysis_obj, outputpath)
        plot_fwhms(analysis_obj, outputpath)
        save_gauss_params(analysis_obj, outputpath)

    return analysis_obj


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


def compute_fit_score(gaussian_list, rsq, dt_axis, penalty_scaling):
    """
    Uses a penalty function to attempt to regularize the fitting and score peak fits optimally.
    Penalty function is designed to penalize:
        - peaks whose widths deviate from expected protein peak width
        - large numbers of peaks
        - peaks that are too close together
        - large movement compared to previous CV
    :param gaussian_list: optimized parameters from curve_fit, translated into list of Gaussian objects
    :param rsq: r-squared value comparing fitted data against observed
    :param dt_axis: x-axis over which to evaluate penalties (e.g. for shared area)
    :param penalty_scaling: how much to scale penalty (to reduce contribution relative to rsq)
    :return: score (float between 0, 1), penalties by individual peaks
    """
    # amplitudes = [gaussian.amplitude for gaussian in gaussian_list]
    # centroids = [gaussian.centroid for gaussian in gaussian_list]
    # widths = [gaussian.width for gaussian in gaussian_list]

    # compute penalties by peak to allow removal of poorly fit peaks
    peak_penalties = []
    for gaussian in gaussian_list:
        current_penalty = compute_width_penalty(gaussian.width, expected_width=0.45, tolerance=0.2, steepness=0.25)
        if len(gaussian_list) > 1:
            current_penalty += compute_area_penalty(gaussian, gaussian_list, dt_axis)
        peak_penalties.append(current_penalty)

    total_penalty = np.sum(peak_penalties)

    # width penalty function (some tolerance for allowed width)
    # total_penalty = 0
    # for width in widths:
    #     total_penalty += compute_width_penalty(width, expected_width=0.45, tolerance=0.2, steepness=1)
    #
    # # shared area penalty function
    # total_penalty += compute_area_penalty(gaussian_list, dt_axis)

    scaled_penalty = total_penalty * penalty_scaling
    score = rsq - scaled_penalty
    return score, peak_penalties


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
        my_penalty = (1.6 * shared_area_ratio - 0.25) ** 4
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
        elif gauss1[index] < gauss2[index]:
            shared_area_arr.append(gauss1[index])
        elif gauss1[index] == gauss2[index]:
            shared_area_arr.append(0)

    # return the integrated area over the provided axis
    return scipy.integrate.trapz(shared_area_arr, x_axis)


def generate_gaussians_from_popt(opt_params_list, cv=None, pcov=None):
    """
    Convert a list of parameters to a list of Gaussian objects. Initializes Gaussians with a collision voltage
    and covariance matrix if provided.
    :param opt_params_list: list of parameters [amp, centroid, width, amp2, cent2, width2, ... ]
    :param cv: (optional) collision voltage to associate with all Gaussians
    :param pcov: (optional) covariance matrix from fitting to associate with all Gaussians
    :return: list of Gaussian objects from params list
    :rtype: list[Gaussian]
    """
    index = 0
    gaussian_list = []
    while index < len(opt_params_list):
        gaussian_list.append(Gaussian(opt_params_list[index],
                                      opt_params_list[index + 1],
                                      opt_params_list[index + 2],
                                      cv,
                                      pcov))
        index += 3
    return gaussian_list


def save_gaussfits_pdf(analysis_obj, outputpath):
    """
    Save a pdf containing an image of the data and gaussian fit for each column to pdf in outputpath.
    :param analysis_obj: container with gaussian fits to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :return: void
    """
    # TODO: make all plot parameters accessible

    # ensure gaussian data has been initialized
    if analysis_obj.gauss_fits is None:
        print('No gaussian fit data in this object yet, returning')
        return

    # print('Saving Gausfitdata_' + str(analysis_obj.raw_obj.filename) + '_.pdf .....')
    gauss_name = os.path.basename(analysis_obj.filename).rstrip('.ciu') + '_gaussFit.pdf'
    pdf_fig = matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputpath, gauss_name))

    intarray = np.swapaxes(analysis_obj.ciu_data, 0, 1)
    for cv_index in range(len(analysis_obj.axes[1])):
        plt.figure()
        # plot the original raw data as a scatter plot
        plt.scatter(analysis_obj.axes[0], intarray[cv_index])
        # plot the fit data as a black dashed line
        plt.plot(analysis_obj.axes[0], analysis_obj.gauss_fits[cv_index], ls='--', color='black')

        # plot each fitted gaussian and centroid
        for gaussian in analysis_obj.filtered_gaussians[cv_index]:
            fit = gaussfunc(analysis_obj.axes[0], gaussian.amplitude, gaussian.centroid, gaussian.width)
            plt.plot(analysis_obj.axes[0], fit)
            plt.plot(gaussian.centroid, abs(gaussian.amplitude), '+', color='red')
        plt.title('CV: {}, R2: {:.3f}, stderr: {:.4f}'.format(analysis_obj.axes[1][cv_index], analysis_obj.gauss_r2s[cv_index],
                                                              analysis_obj.gauss_fit_stats[cv_index][5]))
        pdf_fig.savefig()
        plt.close()
    pdf_fig.close()
    # print('Saving Gausfitdata_' + str(analysis_obj.raw_obj.filename) + '.pdf')


def plot_centroids(analysis_obj, outputpath, y_bounds=None):
    """
    Save a png image of the centroid DTs fit by gaussians. USES FILTERED peak data
    :param analysis_obj: container with gaussian fits to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :param y_bounds: [lower bound, upper bound] to crop the plot to (in y-axis units, typically ms)
    :return: void
    """
    # TODO: make all plot parameters accessible
    # Get a list of centroids, sorted by collision voltage
    filt_centroids = analysis_obj.get_attribute_by_cv('centroid', True)

    for x, y in zip(analysis_obj.axes[1], filt_centroids):
        plt.scatter([x] * len(y), y)
    # plt.scatter(self.axes[1], self.gauss_centroids)
    plt.xlabel('Trap CV')
    plt.ylabel('ATD_centroid')
    if y_bounds is not None:
        plt.ylim(y_bounds)
    plt.title('Centroids filtered by peak width')
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
    plt.savefig(os.path.join(outputpath, output_name), dpi=500)
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
        output.write('Filtered Gaussians\n')
        output.write('Trap CV,Centroid,Amplitude,Peak Width,Baseline(y0),FWHM,Resolution\n')
        index = 0
        while index < len(analysis_obj.axes[1]):
            # outputline = '{},'.format(self.axes[1][index])
            outputline = ','.join([gaussian.print_info() for gaussian in analysis_obj.filtered_gaussians[index]])
            # outputline += ','.join(['{:.2f}'.format(x) for x in self.gauss_filt_params[index]])
            output.write(outputline + '\n')
            index += 1

        index = 0
        output.write('All gaussians fit to data\n')
        output.write('R^2,Adj R^2,Trap CV,Centroid,Amplitude,Peak Width,Baseline(y0),FWHM,Resolution\n')
        while index < len(analysis_obj.axes[1]):
            gauss_line = '{:.3f},{:.3f},'.format(analysis_obj.gauss_r2s[index], analysis_obj.gauss_adj_r2s[index])
            # gauss_line += ','.join(['{:.2f}'.format(x) for x in self.gauss_params[index]])
            gauss_line += ','.join([gaussian.print_info() for gaussian in analysis_obj.gaussians[index]])

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
        gaussian_fit_ciu(current_analysis_obj, current_analysis_obj.params)  # NOTE: analysis_obj.params is DEPRECATED
