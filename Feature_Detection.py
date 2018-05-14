"""
Module for feature detection. Relies on CIUAnalysisObj from Gaussian fitting module
Author: DP
Date: 10/10/2017
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
import scipy.interpolate
import os
import CIU_Params
import Raw_Processing

# imports for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CIU_analysis_obj import CIUAnalysisObj

TRANS_COLOR_DICT = {6: 'white',
                    0: 'red',
                    4: 'blue',
                    3: 'green',
                    2: 'yellow',
                    1: 'orange',
                    5: 'purple'}


def feature_detect_col_max(analysis_obj, params_obj):
    """
    Uses max values of each CV column to assign flat features to data. Should be roughly
    analogous to the changepoint detection + flat features from column maxes in CIU-50 analysis,
    but without reliance on the (somewhat fickle) changepoint detection
    :param analysis_obj: CIUAnalysisObj with Gaussians previously fitted
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :rtype: CIUAnalysisObj
    :return: analysis object with features saved
    """
    features = []

    # interpolate CV axis if the spacing is not equal
    cv_axis = analysis_obj.axes[1]
    bin_spacings = [cv_axis[x + 1] - cv_axis[x] for x in range(len(cv_axis) - 1)]
    unique_spacings = set(bin_spacings)
    if len(unique_spacings) > 1:
        # uneven CV spacing - interpolate axis to even spacing (smallest previous bin spacing) before analysis
        new_num_bins = len(np.arange(cv_axis[0], cv_axis[-1], min(unique_spacings))) + 1
        cv_axis = np.linspace(cv_axis[0], cv_axis[-1], new_num_bins)
        analysis_obj = Raw_Processing.interpolate_axes(analysis_obj, [analysis_obj.axes[0], cv_axis])
        print('NOTE: CV axis in file {} was not evenly spaced; Feature Detection requires even spacing. Axis has been interpolated to fit. Use "Restore Original Data" button to undo interpolation'.format(analysis_obj.short_filename))

    # compute width tolerance in DT units, CV gap in bins (NOT cv axis units)
    width_tol_dt = params_obj.feature_3_width_tol  # * analysis_obj.bin_spacing
    gap_tol_cv = params_obj.feature_4_gap_tol * analysis_obj.cv_spacing
    cv_spacing = analysis_obj.axes[1][1] - analysis_obj.axes[1][0]

    # Search each gaussian for features it matches (based on centroid)
    for cv_index, col_max_dt in enumerate(analysis_obj.col_max_dts):
        # check if any current features will accept the Gaussian
        found_feature = False
        for feature in features:
            if feature.accept_centroid(col_max_dt, width_tol_dt, cv_axis[cv_index], gap_tol_cv, cv_spacing):
                feature.cvs.append(cv_axis[cv_index])
                feature.dt_max_vals.append(col_max_dt)
                feature.cv_indices.append(cv_index)

                found_feature = True
                break

        if not found_feature:
            # no feature was found for this Gaussian, so create a new feature
            new_feature = Feature(gaussian_bool=False)
            new_feature.cvs.append(cv_axis[cv_index])
            new_feature.dt_max_vals.append(col_max_dt)
            new_feature.cv_indices.append(cv_index)
            features.append(new_feature)

    # filter features to remove 'loners' without a sufficient number of points
    filtered_features = filter_features(features, params_obj.feature_2_min_length, mode='changept')

    # finalize features by initializing data for ciu-50 analysis
    # for feature in filtered_features:
    #     feature.init_feature_data_changept(cv_index_list, feature.cvs, feature.dt_max_vals)
    analysis_obj.features_changept = filtered_features
    return analysis_obj


def feature_detect_gaussians(analysis_obj, params_obj):
    """
    Uses fitted (and filtered) multi-gaussians to assign flat features to data. Should be roughly
    analogous to the changepoint detection + flat features from column maxes in CIU-50 analysis,
    but using gaussian data enables seeing all features instead only the most intense one(s).
    :param analysis_obj: CIUAnalysisObj with Gaussians previously fitted
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :rtype: CIUAnalysisObj
    :return: analysis object with features saved
    """
    features = []
    # compute width tolerance in DT units
    width_tol_dt = params_obj.feature_3_width_tol  # * analysis_obj.bin_spacing
    gap_tol_cv = params_obj.feature_4_gap_tol  # * analysis_obj.cv_spacing
    cv_spacing = analysis_obj.axes[1][1] - analysis_obj.axes[1][0]

    # Search each gaussian for features it matches (based on centroid)
    # get the flat list of filtered gaussians
    flat_gauss_list = [x for cv_list in analysis_obj.filtered_gaussians for x in cv_list]
    for gaussian in flat_gauss_list:
        # check if any current features will accept the Gaussian
        found_feature = False
        for feature in features:
            if feature.accept_centroid(gaussian.centroid, width_tol_dt, gaussian.cv, gap_tol_cv, cv_spacing):
                feature.gaussians.append(gaussian)
                found_feature = True
                break

        if not found_feature:
            # no feature was found for this Gaussian, so create a new feature
            new_feature = Feature(gaussian_bool=True)
            new_feature.gaussians.append(gaussian)
            features.append(new_feature)
    # filter features to remove 'loners' without a sufficient number of points
    filtered_features = filter_features(features, params_obj.feature_2_min_length, mode='gaussian')
    analysis_obj.features_gaussian = filtered_features
    return analysis_obj


def ciu50_main(analysis_obj, params_obj, outputdir, gaussian_bool):
    """
    Primary feature detection runner method. Calls appropriate sub-methods using data and
    parameters from the passed analysis object
    :param analysis_obj: CIUAnalysisObj with initial data processed and parameters
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :param gaussian_bool: (bool) whether to use Gaussian or raw data Features for CIU-50 fitting
    :rtype: CIUAnalysisObj
    :return: updated analysis object with feature detect information saved
    """
    # assemble the list of Features to use
    if gaussian_bool:
        if analysis_obj.features_gaussian is None:
            feature_detect_gaussians(analysis_obj, params_obj)

        # Adjust Features to avoid inclusion of any points without col maxes
        features_list = adjust_gauss_features(analysis_obj, params_obj)

        # Catch bad inputs (not enough features to compute a transition)
        if len(features_list) <= 1:
            print('Not enough features (<=1) in file {}. No transition analysis performed'.format(analysis_obj.short_filename))
            return analysis_obj
    else:
        # detect features if none are present
        if analysis_obj.features_changept is None:
            analysis_obj = feature_detect_col_max(analysis_obj, params_obj)
        features_list = analysis_obj.features_changept

    # compute transitions
    transitions_list = compute_transitions(analysis_obj, params_obj, features_list)
    if len(transitions_list) == 0:
        print('No transitions found for file {}'.format(os.path.basename(analysis_obj.filename).rstrip('.ciu')))

    # generate output plot
    plot_transitions(transitions_list, analysis_obj, params_obj, outputdir)

    return analysis_obj


def compute_transitions(analysis_obj, params_obj, adjusted_features):
    """
    Fit logistic/sigmoidal transition functions to the transition between each sequential pair
    of features in the provided gaussian feature list. Saves Transition objects containing combined
    feature pair info, fit, and resulting CIU-50 value.
    :param adjusted_features: List of Features adjusted to only include CV data where col max is close to centroid
    :param analysis_obj: CIU analysis object with gaussian features already prepared
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :rtype: list[Transition]
    :return: list of Transition objects (also saves to analysis_obj)
    """
    # initialize transition fitting information for gaussian feature lists
    for feature in adjusted_features:
        # Get indices of each CV relative to the complete fingerprint and corresponding DT max values
        cv_indices = []
        cv_axis = list(analysis_obj.axes[1])
        for cv in feature.cvs:
            overall_index = cv_axis.index(cv)
            cv_indices.append(overall_index)
        # cv_indices = [list(analysis_obj.axes[1]).index(feature.cvs[i]) for i in feature.cvs]
        # dt_max_bins = analysis_obj.col_maxes[cv_indices[0]: cv_indices[len(cv_indices) - 1]]
        dt_max_vals = analysis_obj.col_max_dts[cv_indices[0]: cv_indices[len(cv_indices) - 1]]
        feature.init_feature_data(cv_indices, dt_max_vals)

    # Fit sigmoids for transition calculations
    index = 0
    transition_list = []
    while index < len(adjusted_features) - 1:
        current_transition = Transition(adjusted_features[index],
                                        adjusted_features[index + 1],
                                        analysis_obj)
        # check to make sure this is a transition that should be fitted (upper feature has a col max)
        if current_transition.check_features(analysis_obj, params_obj):
            current_transition.fit_transition(params_obj)
            transition_list.append(current_transition)
        else:
            print('feature {} never reaches 50% intensity, '
                  'skipping transition between {} and {}'.format(index + 2, index + 1, index + 2))
        index += 1
    analysis_obj.transitions = transition_list
    return transition_list


def filter_features(features, min_feature_length, mode):
    """
    Remove any features below the specified minimum feature length from the feature list
    :param features: list of Features
    :type features: list[Feature]
    :param min_feature_length: minimum length (number of gaussians) to be included in a feature
    :param mode: gaussian or changepoint: whether features are from gaussian fitting or changept detection
    :return: filtered feature list with too-small features removed
    """
    filtered_list = []
    for feature in features:
        if mode == 'gaussian':
            if len(feature.gaussians) >= min_feature_length:
                filtered_list.append(feature)
        elif mode == 'changept':
            if len(feature.cvs) >= min_feature_length:
                filtered_list.append(feature)
        else:
            print('invalid mode')
    return filtered_list


def adjust_gauss_features(analysis_obj, params_obj):
    """
    Run setup method to prepare Gaussian features for transition fitting. Removes any CV values
    from features for which the max DT value at that CV is outside the width tolerance from the
    feature's median DT. This is necessary because Gaussian features start/persist well before/after
    they are the most abundant peak - which can cause bad fitting if incorrect CV's are included.
    :param analysis_obj: CIUAnalysisObj with gaussian fitting and gaussian feature detect performed
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :rtype: list[Feature]
    :return: list of adjusted Features
    """
    adjusted_features = []
    for feature in analysis_obj.features_gaussian:
        final_cvs = []
        for cv in feature.cvs:
            # check if the ciu_data column max value is appropriate for this feature at this CV
            cv_index = list(analysis_obj.axes[1]).index(cv)
            dt_diff = abs(analysis_obj.col_max_dts[cv_index] - feature.gauss_median_centroid)
            if dt_diff < bin_to_ms(params_obj.feature_3_width_tol, analysis_obj.bin_spacing):
                # also check if a gap has formed and exclude features after the gap if so
                if len(final_cvs) > 0:
                    if cv - final_cvs[len(final_cvs) - 1] <= params_obj.feature_4_gap_tol:
                        # difference is within tolerances; include this CV in the adjusted feature
                        final_cvs.append(cv)
                else:
                    final_cvs.append(cv)

        # initialize the new feature using the CV list (it will only have CV and centroid data)
        if len(final_cvs) > 0:
            adj_feature = Feature(gaussian_bool=True)
            adj_feature.gauss_median_centroid = feature.gauss_median_centroid
            adj_feature.cvs = final_cvs
            adjusted_features.append(adj_feature)
            adj_feature.gaussians = feature.gaussians
    return adjusted_features


def plot_features(analysis_obj, params_obj, outputdir):
    """
    Generate a plot of features using previously saved (into the analysis_obj) feature fitting data
    :param analysis_obj: CIUAnalysisObj with fitting data previously saved to obj.features_gaussian
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :return: void
    """
    # initialize plot
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # plot the initial CIU contour plot for reference
    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, 100, cmap=params_obj.plot_01_cmap)

    # prepare and plot the actual Features using saved data
    feature_index = 1
    if params_obj.feature_1_ciu50_mode == 'gaussian':
        # plot the raw data to show what was fit
        filt_centroids = analysis_obj.get_attribute_by_cv('centroid', True)
        for x, y in zip(analysis_obj.axes[1], filt_centroids):
            plt.scatter([x] * len(y), y, c='w')

        for feature in analysis_obj.features_gaussian:
            feature_x = [gaussian.cv for gaussian in feature.gaussians]
            feature_y = [feature.gauss_median_centroid for _ in feature.gaussians]
            lines = plt.plot(feature_x, feature_y, label='Feature {} median: {:.2f}'.format(feature_index,
                                                                                            feature.get_median()))
            feature_index += 1
            plt.setp(lines, linewidth=3)
    elif params_obj.feature_1_ciu50_mode == 'standard':
        # plot the raw data to show what was fit
        plt.plot(analysis_obj.axes[1], analysis_obj.col_max_dts, 'wo')

        for feature in analysis_obj.features_changept:
            feature_x = feature.cvs
            feature_y = feature.dt_max_vals
            lines = plt.plot(feature_x, feature_y, label='Feature {} median: {:.2f}'.format(feature_index,
                                                                                            feature.get_median()))
            feature_index += 1
            plt.setp(lines, linewidth=3)
    else:
        print('invalid mode')

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = analysis_obj.short_filename
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_06_show_colorbar:
        cbar = plt.colorbar(ticks=[0, .25, .5, .75, 1])
        cbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)

    # save plot
    output_path = os.path.join(outputdir, analysis_obj.filename.rstrip('.ciu') + '_features' + params_obj.plot_02_extension)
    plt.savefig(output_path)
    plt.clf()


def print_features_list(feature_list, outputpath, mode):
    """
    Write feature information to file
    :param feature_list: list of Feature objects
    :type feature_list: list[Feature]
    :param outputpath: directory in which to save output
    :param mode: gaussian or changepoint
    :return: void
    """
    with open(outputpath, 'w') as outfile:
        index = 1
        for feature in feature_list:
            if mode == 'gaussian':
                outfile.write('Feature {},Median centroid:,{:.2f},CV range:,{} - {}\n'.format(index,
                                                                                              feature.get_median(),
                                                                                              feature.cvs[0],
                                                                                              feature.cvs[len(feature.cvs) - 1]))
                outfile.write('CV (V), Centroid, Amplitude, Width, Baseline, FWHM, Resolution\n')
                for gaussian in feature.gaussians:
                    outfile.write(gaussian.print_info() + '\n')
            else:
                outfile.write('Feature {},Median centroid:,{:.2f},CV range:,{} - {}\n'.format(index,
                                                                                              feature.get_median(),
                                                                                              feature.cvs[0],
                                                                                              feature.cvs[len(feature.cvs) - 1]))
                outfile.write('CV (V),Peak Drift Time')
                cv_index = 0
                for cv in feature.cvs:
                    outfile.write('{},{:.2f}\n'.format(cv, feature.dt_max_vals[cv_index]))
                    cv_index += 1

            index += 1


def save_ciu50_outputs(analysis_obj, outputpath, combine=False):
    """
    Print feature detection outputs to file. Must have feature detection already performed.
    **NOTE: currently, feature plot is still in the feature detect module, but could (should?)
    be moved here eventually.
    :param analysis_obj: CIU container with transition information to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :param combine: whether to output directly for this file or return a string for combining
    :return: void
    """
    output_name = os.path.join(outputpath, analysis_obj.filename + '_features.csv')
    output_string = 'Transitions:,y0 (ms),ymax (ms),CIU-50 (V),k (steepness),r_squared\n'
    trans_index = 1
    for transition in analysis_obj.transitions:
        output_string += 'transition {} -> {},'.format(trans_index, trans_index + 1)
        output_string += '{:.2f},{:.2f},{:.2f},{:.2f}'.format(*transition.fit_params)
        output_string += ',{:.3f}\n'.format(transition.rsq)
        trans_index += 1

    if combine:
        # return the output string to be written together with many files
        return output_string
    else:
        with open(output_name, 'w') as outfile:
            outfile.write(output_string)


def save_ciu50_short(analysis_obj, outputpath, combine=False):
    """
    Helper method to also save a shortened version of feature information
    :param analysis_obj: CIU container with transition information to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :param combine: If True, return a string to be combined with other files instead of saving to file
    :return:
    """
    output_name = os.path.join(outputpath, analysis_obj.filename + '_transitions-short.csv')
    output_string = ''

    # assemble the output
    for transition in analysis_obj.transitions:
        output_string += ',{:.2f}'.format(transition.fit_params[2])
    output_string += '\n'

    if combine:
        # return the output string to be written together with many files
        return output_string
    else:
        with open(output_name, 'w') as outfile:
            outfile.write(output_string)


def plot_transitions(transition_list, analysis_obj, params_obj, outputdir):
    """
    Provide a plot of provided transitions overlaid on top of the CIU contour plot for the
    provided analysis object
    :param transition_list: list of Transitions to plot
    :type transition_list: list[Transition]
    :param analysis_obj: object with CIU data to plot
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :return: void
    """
    x_axis = analysis_obj.axes[1]
    y_data = analysis_obj.col_max_dts
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # plot the initial CIU contour plot for reference
    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, 100, cmap=params_obj.plot_01_cmap)

    # plot markers for the features/segments assigned
    plt.plot(x_axis, y_data, 'wo')

    # plot all transitions
    transition_num = 0
    for transition in transition_list:
        # prepare and plot the actual transition using fitted parameters
        interp_x = np.linspace(x_axis[0], x_axis[len(x_axis) - 1], 200)
        y_fit = logistic_func(interp_x, *transition.fit_params)

        # use different colors for plotting the transition (up to 6 provided)
        if transition_num <= 6:
            trans_line_color = TRANS_COLOR_DICT[transition_num]
        else:
            trans_line_color = TRANS_COLOR_DICT[6]
        trans_plot = plt.plot(interp_x, y_fit, color=trans_line_color, label='CIU50: {:.1f}, r2=: {:.2f}'.format(transition.ciu50, transition.rsq))
        plt.setp(trans_plot, linewidth=2)
        transition_num += 1

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = analysis_obj.short_filename
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_06_show_colorbar:
        cbar = plt.colorbar(ticks=[0, .25, .5, .75, 1])
        cbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)

    # save plot to file
    filename = os.path.basename(analysis_obj.filename).rstrip('.ciu') + '_transition' + params_obj.plot_02_extension
    output_path = os.path.join(outputdir, filename)
    plt.savefig(output_path)


def bin_to_dt(bin_val, min_dt, bin_spacing):
    """
    Convert a bin value to a fingerprint-relative drift time. Adjusts for the minimum DT of
    the fingerprint to give accurate drift axis results. Should NOT be used for conversion
    of absolute bin -> DT
    :param bin_val: (int) bin number to convert to DT space
    :param min_dt: minimum DT of fingerprint
    :param bin_spacing: spacing between DT bins (conversion factor)
    :return: DT in drift axis units
    """
    dt = min_dt + (bin_val - 1) * bin_spacing
    return dt


def bin_to_ms(bin_val, bin_spacing):
    """
    Conversion from a number of bins to the corresponding time in ms (or other drift units)
    for the provided spacing. Differs from bin to dt in that the output time is NOT adjusted
    for the minimum of the fingerprint (an absolute conversion, rather than a fingerprint-relative
    conversion).
    :param bin_val: (int) number of bins to convert
    :param bin_spacing: distance between bins in time units (conversion factor)
    :return: time in drift axis units corresponding to bin val
    """
    dt = bin_val * bin_spacing
    return dt


def logistic_func(x, c, y0, x0, k):
    """
    Generalized logistic function for fitting to feature transitions
    :param x: x value (independent variable)
    :param c: height of the maximum/upper asymptote of the curve
    :param y0: height of the minimum/lower asymptote of the curve
    :param x0: centroid/midpoint of the curve (also CIU-50 value of the transition)
    :param k: steepness of the curve/transition
    :return: y = f(x)
    """
    # y = c / (1 + np.exp(-k * (x - x0))) + y0
    y = y0 + ((c - y0) / (1 + np.exp(-k * (x - x0))))
    return y


def fit_logistic(x_axis, y_data, guess_center, guess_min, guess_max, steepness_guess):
    """
    Fit a general logistic function (defined above) to data using the SciPy.optimize
    curve_fit module.
    :param x_axis: x data to fit (list or ndarray)
    :param y_data: y data to fit (list of ndarray)
    :param guess_center: initial guess for x0 - should be around the midpoint of the transition
    :param guess_max: initial guess for c - should be around the max y-value for the transition
    :param guess_min: initial guess for y0 - should be around the min y-value for the transition
    :param steepness_guess: initial guess for k. 0.1 works well for shallower transitions and still gets steep ones
    :return: popt, pcov: popt = optimized parameters [c, y0, x0, k] from fitting, pcov = covariance matrix
    """
    # guess initial params: [c, y0, x0, k], default guess k=1
    p0 = [guess_max, guess_min, guess_center, steepness_guess]
    # constrain all parameters to positive values
    fit_bounds_lower = [0, 0, 0, 0]
    fit_bounds_upper = [np.inf, np.inf, np.inf, np.inf]
    try:
        popt, pcov = scipy.optimize.curve_fit(logistic_func, x_axis, y_data, p0=p0,
                                              bounds=(fit_bounds_lower, fit_bounds_upper))
    except ValueError:
        print('Error: fitting failed due to bad input values. Please try smoothing the input data more')
        popt, pcov = [], []
    # popt, pcov = scipy.optimize.curve_fit(logistic_func, x_axis, y_data, p0=p0, maxfev=5000)
    return popt, pcov


def find_nearest(array, value):
    """
    Get the index in the array nearest to the input value. Handles values outside the
    array by returning the end value in the correct direction.
    :param array: array-like object to search within
    :param value: value to find nearest index in the array
    :return: index (int) of closest match in the array
    """
    idx = (np.abs(array - value)).argmin()
    return idx


class Feature(object):
    """
    Holder for feature information while doing feature detection
    """
    def __init__(self, gaussian_bool):
        """
        Create a new feature object to hold feature information. Intended to add to cv/centroid info over time
        :param gaussian_bool: Whether this feature was constructed from Gaussian fit data or not
        """
        self.cvs = []
        self.cv_indices = []
        self.centroids = []
        self.gauss_median_centroid = None
        self.gaussians = []
        self.gaussian_bool = gaussian_bool

        # attributes to handle conversion for use with Transitions. Will be set after fitting by a method
        self.start_cv_index = None
        self.end_cv_index = None
        self.start_cv_val = None
        self.end_cv_val = None
        self.dt_max_bins = None
        self.dt_max_vals = []

    def __str__(self):
        # display either the gaussian or changepoint version data, including median and length of list
        # if self.gaussian_bool:
        #     return '<Feature> Med: {:.1f} Len: {}'.format(self.get_median(), len(self.cvs))
        # else:
        return '<Feature> Med: {:.1f} Len: {}'.format(self.get_median(), len(self.cvs))
    __repr__ = __str__

    def refresh(self):
        """
        Refresh the centroid median and cvs using the gaussians that have been added to the feature
        :return: void
        """
        self.gauss_median_centroid = np.median([x.centroid for x in self.gaussians])
        for gaussian in self.gaussians:
            if gaussian.cv not in self.cvs:
                self.cvs.append(gaussian.cv)
        # self.cvs = [x.cv for x in self.gaussians if x.cv not in self.cvs]   # get all CV's included (without repeats)
        self.cvs = sorted(self.cvs)

    def accept_centroid(self, centroid, width_tol, collision_voltage, cv_tol, cv_spacing):
        """
        Determine whether the provided centroid is within tolerance of the feature or not. Uses
        feature detection parameters (flat width tolerance) to decide.
        :param centroid: the centroid (float) to compare against Feature
        :param width_tol: tolerance in DT units (float) to compare to centroid
        :param collision_voltage: CV position of the gaussian to compare against feature for gaps
        :param cv_tol: distance in collision voltage space that can be skipped and still accept a gaussian
        :param cv_spacing: distance between discrete points along collision voltage axis of CIU data
        :return: boolean
        """
        # Refresh current median and cvs in case more gaussians have been added since last calculation
        self.refresh()

        # ensure cv_tol is at least the cv bin spacing
        if cv_tol < cv_spacing:
            cv_tol = cv_spacing

        if abs(self.get_median() - centroid) <= width_tol:
            # centroid is within the Feature's bounds, check for gaps
            nearest_cv_index = (np.abs(np.asarray(self.cvs) - collision_voltage)).argmin()
            nearest_cv = self.cvs[nearest_cv_index]
            # if collision voltage is within tolerance of the nearest CV in the feature already, return True
            return abs(collision_voltage - nearest_cv) <= cv_tol

    def get_median(self):
        """
        Return the median centroid (DT units) of this Feature uniformly for Gaussian and non-Gaussian
        Features.
        :return: (float) feature median
        """
        if self.gaussian_bool:
            return self.gauss_median_centroid
        else:
            return np.median(self.dt_max_vals)

    def init_feature_data(self, cv_index_list, dt_val_list):
        """
        Import and set data to use with Transition class. Adapted to removed ChangeptFeature subclass
        Note: *requires gaussian feature detection to have been performed previously*
        :param cv_index_list: list of indices of collision voltages that make up this feature (args for sublist of CV axis)
        :param dt_val_list: list of max_dt values in ms for each collision voltage in the feature
        """
        self.start_cv_val = self.cvs[0]
        self.end_cv_val = self.cvs[len(self.cvs) - 1]

        self.start_cv_index = cv_index_list[0]
        self.end_cv_index = cv_index_list[len(cv_index_list) - 1]

        self.dt_max_vals = dt_val_list


class Transition(object):
    """
    Store information about a CIU transition, including the starting and ending feature,
    their combined CV/index range and DT data, and fitted logistic function parameters
    and CIU50.
    """
    def __init__(self, feature1, feature2, analysis_obj):
        """
        Create a combined Transition object from two identified features. Features MUST be
        adjacent in CV space for this to make sense.
        :param feature1: Lower CV ("earlier/starting") Feature object
        :param feature2: Higher CV ("later/ending") Feature object
        :param analysis_obj: CIUAnalysisObj with data to be fitted
        :type analysis_obj: CIUAnalysisObj
        """
        # initialize data from analysis_obj
        self.whole_cv_axis = analysis_obj.axes[1]
        dt_axis = analysis_obj.axes[0]
        self.whole_dt_maxes = analysis_obj.col_max_dts
        ciu_data = analysis_obj.ciu_data
        self.filename = analysis_obj.short_filename

        self.center_guess_gaussian = None

        self.feature1 = feature1    # type: Feature
        self.feature2 = feature2    # type: Feature

        self.start_cv = feature1.start_cv_val
        self.end_cv = feature2.end_cv_val
        self.start_index = feature1.start_cv_index
        self.end_index = feature2.end_cv_index
        self.feat_distance = self.feature2.start_cv_val - self.feature1.end_cv_val

        self.combined_x_axis = self.whole_cv_axis[self.start_index: self.end_index + 1]     # +1 b/c slicing
        self.combined_y_vals = self.whole_dt_maxes[self.start_index: self.end_index + 1]    # +1 b/c slicing

        # Raw y data for final transition fitting
        y_col_data = np.swapaxes(ciu_data, 0, 1)
        y_wtd_avg_cols = []
        y_median_cols = []
        for cv_col in y_col_data:
            # the weighted sum of a column is the product of intensity * drift time for each bin
            wtd_sum_dt = 0
            int_sum = 0
            wtd_sum_dts = []
            for i in range(len(cv_col)):
                # cv_col[i] is the intensity at drift bin i; axes[0][i] is the drift time at bin i
                wtd_value = cv_col[i] * dt_axis[i]
                wtd_sum_dt += wtd_value
                int_sum += cv_col[i]
                wtd_sum_dts.append(wtd_sum_dt)
            # spectral centroid (average) = (sum of (DT_value * intensity)) / (sum of intensity)
            y_wtd_avg_cols.append(wtd_sum_dt / int_sum)
            med_value = wtd_sum_dt / 2.0
            wtd_sum_dts = np.asarray(wtd_sum_dts)
            med_index = (np.abs(wtd_sum_dts - med_value)).argmin()
            med_dt = dt_axis[med_index]
            y_median_cols.append(med_dt)
        # y_avg_cols = np.average(y_col_data, axis=1)
        self.combined_y_avg_raw = y_wtd_avg_cols[self.start_index: self.end_index + 1]
        self.combined_y_median_raw = y_median_cols[self.start_index: self.end_index + 1]
        self.min_guess = None
        self.max_guess = None

        self.ciu50 = None
        self.fit_params = None
        self.fit_covariances = None
        self.rsq = None
        self.fit_param_errors = None

    def __str__(self):
        # display the transition range and CIU50 and R2 if they have been calculated
        if self.ciu50 is not None and self.rsq is not None:
            return '<Transition> range: {}-{}, ciu50: {:.1f}, rsq: {:.2f}'.format(self.start_cv,
                                                                                  self.end_cv,
                                                                                  self.ciu50,
                                                                                  self.rsq)
        else:
            return '<Transition> range: {}-{}'.format(self.start_cv, self.end_cv)
    __repr__ = __str__

    def fit_transition(self, params_obj):
        """
        Fit a logistic function to the transition using the feature information. Requires
        bin_spacing and dt_min to do conversion to DT space from bin space for plotting.
        :param params_obj: Parameters object with fit settings information
        :type params_obj: Parameters
        :return: void (saves fit parameters to object)
        """
        # initial fitting guesses: center is in between the features, min/max are median DTs of features 1 and 2
        center_guess = self.feature2.start_cv_val - (self.feat_distance / 2.0)   # halfway between features

        # todo: deprecate?
        # if gaussian_bool:
        #     self.min_guess = self.feature1.gauss_median_centroid
        #     self.max_guess = self.feature2.gauss_median_centroid
        #     steepness_guess = abs(2 * 1 / (self.feat_distance + 1))
        # else:
        # self.min_guess = dt_min + (np.median(self.feature1.dt_max_bins) - 1) * bin_spacing
        # self.max_guess = dt_min + (np.median(self.feature2.dt_max_bins) - 1) * bin_spacing
        self.min_guess = self.feature1.get_median()
        self.max_guess = self.feature2.get_median()

        # guess steepness as a function of between feature1 end and feature2 start
        steepness_guess = 2 * 1 / (self.feat_distance + 1)
        if steepness_guess < 0:
            steepness_guess = -1 * steepness_guess
            print('Caution: negative slope transition observed in file {}. Data may require additional smoothing if this is unexpected'.format(self.filename))

        # for interpolation of transition modes - determine transition region to interpolate
        pad_cv = params_obj.ciu50_2_pad_transitions_cv
        trans_start_cv = self.feature1.end_cv_val - pad_cv
        trans_end_cv = self.feature2.start_cv_val + pad_cv
        trans_distance = self.feat_distance

        # todo: deprecate? or leave at 2 always (or some kind of advanced menu)
        if params_obj.ciu50_x_1_interp_factor > 0:
            # interpolate whole dataset
            interp_x_vals = np.linspace(self.combined_x_axis[0], self.combined_x_axis[len(self.combined_x_axis) - 1],
                                        len(self.combined_x_axis) * params_obj.ciu50_x_1_interp_factor)
            interp_function = scipy.interpolate.interp1d(self.combined_x_axis, self.combined_y_vals)
            interp_y_vals = interp_function(interp_x_vals)
        else:
            interp_x_vals = self.combined_x_axis
            interp_y_vals = self.combined_y_vals

        if params_obj.ciu50_1_centroiding_mode == 'max':
            # no spectral centroiding
            final_x_vals, final_y_vals = self.assemble_transition_data(interp_x_vals, interp_y_vals, trans_start_cv,
                                                                       trans_end_cv, trans_distance)

        elif params_obj.ciu50_1_centroiding_mode == 'average':
            # spectral averaging
            interp_function_raw = scipy.interpolate.interp1d(self.combined_x_axis, self.combined_y_avg_raw)
            interp_y_vals = interp_function_raw(interp_x_vals)
            final_x_vals, final_y_vals = self.assemble_transition_data(interp_x_vals, interp_y_vals, trans_start_cv,
                                                                       trans_end_cv, trans_distance,
                                                                       interp_trans_factor=params_obj.ciu50_x_2_trans_interp_factor,
                                                                       trans_interp_fn=interp_function_raw)
        elif params_obj.ciu50_1_centroiding_mode == 'median':
            # spectral median
            interp_function_raw = scipy.interpolate.interp1d(self.combined_x_axis, self.combined_y_median_raw)
            interp_y_vals = interp_function_raw(interp_x_vals)
            final_x_vals, final_y_vals = self.assemble_transition_data(interp_x_vals, interp_y_vals, trans_start_cv,
                                                                       trans_end_cv, trans_distance,
                                                                       interp_trans_factor=params_obj.ciu50_x_2_trans_interp_factor,
                                                                       trans_interp_fn=interp_function_raw)

        else:
            print('Invalid fitting mode, skipping CIU-50')
            return

        # run the logistic fitting
        try:
            popt, pcov = fit_logistic(final_x_vals, final_y_vals, center_guess, self.min_guess, self.max_guess,
                                      steepness_guess)
            perr = np.sqrt(np.diag(pcov))
            self.fit_param_errors = perr
        except RuntimeError:
            print('fitting failed for transition {} in file {}'.format(self, self.filename))
            popt = [0, 0, 0, 0]
            pcov = []

        # check goodness of fit
        yfit = logistic_func(final_x_vals, *popt)

        slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(final_y_vals, yfit)
        # adjrsq = adjrsquared(rvalue ** 2, len(cv_col_intensities))
        rsq = rvalue ** 2

        if popt[2] < 0:
            print('WARNING: poor performance from logistic fitting for {} in file {}'.format(self.__str__(), self.filename))
        self.ciu50 = popt[2]
        self.fit_params = popt
        self.fit_covariances = pcov
        self.rsq = rsq

    def assemble_transition_data(self, interpd_x_vals, fit_y_vals, trans_start_cv, trans_end_cv,
                                 trans_distance, interp_trans_factor=1, trans_interp_fn=None):
        """

        :param interpd_x_vals: Interpolated combined x-axis data from both features. Runs from feature1 start to feature2 end
        :param fit_y_vals: y-axis data corresponding to interpd_x_vals array (same length)
        :param trans_start_cv: Starting point of transition region (including padding)
        :param trans_end_cv: End point of transition region (including padding)
        :param trans_distance: length of transition region (in CV), including padding
        :param trans_interp_fn:
        :param interp_trans_factor: Factor by which to interpolate transition region (e.g. 2 would double the number of points considered)
        :return: final x_data, final y_data (equal length 1D ndarrays)
        """
        interpd_cv_step = interpd_x_vals[1] - interpd_x_vals[0]
        trans_start_index = find_nearest(interpd_x_vals, trans_start_cv)
        trans_end_index = find_nearest(interpd_x_vals, trans_end_cv)

        if trans_interp_fn is not None:
            transition_x_vals = np.linspace(trans_start_cv, trans_end_cv, (trans_distance / interpd_cv_step) * interp_trans_factor)
            transition_y_vals = trans_interp_fn(transition_x_vals)
        else:
            # cannot interpolate because no function provided, simply use the corresponding x and y data
            transition_x_vals = interpd_x_vals[trans_start_index: trans_end_index + 1]
            transition_y_vals = fit_y_vals[trans_start_index: trans_end_index + 1]

        final_x_vals = interpd_x_vals[0: trans_start_index]
        final_x_vals = np.append(final_x_vals, transition_x_vals)
        second_half_xvals = interpd_x_vals[trans_end_index + 1: len(interpd_x_vals) - 1]
        final_x_vals = np.append(final_x_vals, second_half_xvals)

        final_y_vals = [self.min_guess for _ in range(0, trans_start_index)]
        final_y_vals = np.append(final_y_vals, transition_y_vals)
        second_half_yvals = [self.max_guess for _ in range(trans_end_index + 1, len(fit_y_vals) - 1)]
        final_y_vals = np.append(final_y_vals, second_half_yvals)

        return final_x_vals, final_y_vals

    def check_features(self, analysis_obj, params_obj):
        """
        Method containing checks for Gaussian feature based transitions. Confirms that the
        second feature contains at least one column max value (i.e. the transition to the feature does
        take place).
        :param analysis_obj: CIUAnalysisObj with axis and column max information
        :type analysis_obj: CIUAnalysisObj
        :param params_obj: Parameters object with parameter information
        :type params_obj: Parameters
        :return: True if checks are satisfied, False if not
        """
        # non-Gaussian features don't need checking - return True
        if not self.feature1.gaussian_bool and not self.feature2.gaussian_bool:
            return True

        if self.start_cv > self.end_cv:
            return False

        feature2_cv_indices = np.arange(self.feature2.start_cv_index, self.feature2.end_cv_index)
        width_tol_dt = params_obj.feature_3_width_tol * analysis_obj.bin_spacing
        for cv_index in feature2_cv_indices:
            # check if a column max is within tolerance of the feature median
            current_max_dt = analysis_obj.col_max_dts[cv_index]
            if abs(current_max_dt - self.feature2.gauss_median_centroid) <= width_tol_dt:
                self.center_guess_gaussian = analysis_obj.axes[1][cv_index]
                return True

        # no CV found with column max within tolerance - return false
        return False


# todo: deprecated
# def ciu50_gaussians(analysis_obj, params_obj, outputdir):
#     """
#     CIU-50 method using Gaussian features instead of changepoint features. Requires that gaussian
#     fitting and feature detection have previously been performed on the analysis_obj
#     :param analysis_obj: CIUAnalysisObj with gaussians and feature data
#     :type analysis_obj: CIUAnalysisObj
#     :param params_obj: Parameters object with parameter information
#     :type params_obj: Parameters
#     :param outputdir: directory in which to save output
#     :rtype: CIUAnalysisObj
#     :return: analysis object
#     """
#     if analysis_obj.features_gaussian is None:
#         feature_detect_gaussians(analysis_obj, params_obj)
#
#     # Adjust Features to avoid inclusion of any points without col maxes
#     adj_features = adjust_gauss_features(analysis_obj, params_obj)
#     # analysis_obj.features_gaussian = adj_features
#
#     # Catch bad inputs (not enough features to compute a transition)
#     if len(adj_features) <= 1:
#         filename = os.path.basename(analysis_obj.filename).rstrip('.ciu')
#         print('Not enough features (<=1) in file {}. No transition analysis performed'.format(filename))
#         return analysis_obj
#
#     # compute transitions and save output
#     transitions_list = compute_transitions(analysis_obj, params_obj=params_obj, adjusted_features=adj_features)
#     if len(transitions_list) == 0:
#         print('No transitions found for file {}'.format(os.path.basename(analysis_obj.filename).rstrip('.ciu')))
#     trans_num = 0
#     for transition in transitions_list:
#         transition.plot_transition(analysis_obj, params_obj, outputdir, trans_num)
#         trans_num += 1
#     plt.clf()
#     return analysis_obj


# testing
if __name__ == '__main__':
    import tkinter
    from tkinter import filedialog

    # open a filechoose to choose .ciu files (change to .csv and add generate_raw_obj/process_raw_obj from CIU2 main if analyzing _raw.csv data)
    root = tkinter.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
    main_dir = os.path.dirname(files[0])

    # initialize parameters to defaults
    params = CIU_Params.Parameters()
    params.set_params(CIU_Params.parse_params_file(CIU_Params.hard_descripts_file))

    # load files and run feature detection and/or CIU-50
    for file in files:
        with open(file, 'rb') as analysis_file:
            obj = pickle.load(analysis_file)

        obj = feature_detect_col_max(obj, params)
        obj = ciu50_main(obj, params, main_dir, gaussian_bool=False)
