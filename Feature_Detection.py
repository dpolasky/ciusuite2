"""
Module for feature detection. Relies on CIUAnalysisObj from Gaussian fitting module
Author: DP
Date: 10/10/2017
"""

import numpy as np
import changepy
import changepy.costs
import pickle
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
import scipy.interpolate
import os

# imports for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CIU_analysis_obj import CIUAnalysisObj
    from CIU_Params import Parameters


class Feature(object):
    """
    Holder for feature information while doing feature detection
    """
    def __init__(self):
        """
        Create a new feature object to hold feature information. Intended to add to cv/centroid info over time
        """
        self.cvs = []
        self.centroids = []
        self.gauss_median_centroid = None
        self.gaussians = []

        # attributes to handle conversion for use with Transitions. Will be set after fitting by a method
        self.start_cv_index = None
        self.end_cv_index = None
        self.start_cv_val = None
        self.end_cv_val = None
        self.dt_max_bins = None
        self.dt_max_vals = None

    def __str__(self):
        # display either the gaussian or changepoint version data, including median and length of list
        if self.gauss_median_centroid is not None:
            return '<Feature> Med: {:.1f} Len: {}'.format(self.gauss_median_centroid, len(self.cvs))
        else:
            return '<Feature> Med: {:.1f} Len: {}'.format(np.median(self.dt_max_vals), len(self.cvs))
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

    def accept_centroid(self, centroid, width_tol, collision_voltage, cv_tol):
        """
        Determine whether the provided centroid is within tolerance of the feature or not. Uses
        feature detection parameters (flat width tolerance) to decide.
        :param centroid: the centroid (float) to compare against Feature
        :param width_tol: tolerance in DT units (float) to compare to centroid
        :param collision_voltage: CV position of the gaussian to compare against feature for gaps
        :param cv_tol: distance in collision voltage space that can be skipped and still accept a gaussian
        :return: boolean
        """
        # Refresh current median and cvs in case more gaussians have been added since last calculation
        self.refresh()
        if abs(self.gauss_median_centroid - centroid) <= width_tol:
            # centroid is within the Feature's bounds, check for gaps
            nearest_cv_index = (np.abs(np.asarray(self.cvs) - collision_voltage)).argmin()
            nearest_cv = self.cvs[nearest_cv_index]
            # if collision voltage is within tolerance of the nearest CV in the feature already, return True
            return abs(collision_voltage - nearest_cv) <= cv_tol

    def init_feature_data_changept(self, cv_index_list, cv_val_list, dt_bin_list, dt_val_list):
        """
        Init method from ChangepointFeature object - merged into single object with Gaussian
        Features. Initializes key parameters for transition fitting from partitioned data.
        :param cv_index_list: List of indices of CV values in the complete dataset
        :param cv_val_list: List of CV values making up this feature
        :param dt_bin_list: List of indices of peak max DT values at each CV
        :param dt_val_list: List of max DT value at each CV
        :return: void
        """
        self.start_cv_val = cv_val_list[0]
        self.end_cv_val = cv_val_list[len(cv_val_list) - 1]
        self.cvs = cv_val_list

        self.start_cv_index = cv_index_list[0]
        self.end_cv_index = cv_index_list[len(cv_index_list) - 1]

        self.dt_max_bins = dt_bin_list
        self.dt_max_vals = dt_val_list

    def init_feature_data_gauss(self, cv_index_list, dt_bin_list, dt_val_list):
        """
        Import and set data to use with Transition class. Adapted to removed ChangeptFeature subclass
        Note: *requires gaussian feature detection to have been performed previously*
        :param cv_index_list: list of indices of collision voltages that make up this feature (args for sublist of CV axis)
        :param dt_bin_list: list of max_dt_bin entries for each collision voltage in the feature
        :param dt_val_list: list of max_dt values in ms for each collision voltage in the feature
        """
        self.start_cv_val = self.cvs[0]
        self.end_cv_val = self.cvs[len(self.cvs) - 1]

        self.start_cv_index = cv_index_list[0]
        self.end_cv_index = cv_index_list[len(cv_index_list) - 1]

        self.dt_max_bins = dt_bin_list
        self.dt_max_vals = dt_val_list


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
    width_tol_dt = params_obj.feature_gauss_width_tol * analysis_obj.bin_spacing
    gap_tol_cv = params_obj.feature_gauss_gap_tol  # * analysis_obj.cv_spacing

    # Search each gaussian for features it matches (based on centroid)
    # get the flat list of filtered gaussians
    flat_gauss_list = [x for cv_list in analysis_obj.filtered_gaussians for x in cv_list]
    for gaussian in flat_gauss_list:
        # check if any current features will accept the Gaussian
        found_feature = False
        for feature in features:
            if feature.accept_centroid(gaussian.centroid, width_tol_dt, gaussian.cv, gap_tol_cv):
                feature.gaussians.append(gaussian)
                found_feature = True
                break

        if not found_feature:
            # no feature was found for this Gaussian, so create a new feature
            new_feature = Feature()
            new_feature.gaussians.append(gaussian)
            features.append(new_feature)
    # filter features to remove 'loners' without a sufficient number of points
    filtered_features = filter_features(features, params_obj.feature_gauss_min_length, mode='gaussian')
    # for feature in filtered_features:
    #     cv_index_list = []
    #     feature.init_feature_data_gauss()
    analysis_obj.features_gaussian = filtered_features
    return analysis_obj


def compute_transitions_gaussian(analysis_obj, params_obj, adjusted_features):
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
        dt_max_bins = analysis_obj.col_maxes[cv_indices[0]: cv_indices[len(cv_indices) - 1]]
        dt_max_vals = analysis_obj.col_max_dts[cv_indices[0]: cv_indices[len(cv_indices) - 1]]
        feature.init_feature_data_gauss(cv_indices, dt_max_bins, dt_max_vals)

    # Fit sigmoids for transition calculations
    index = 0
    transition_list = []
    while index < len(adjusted_features) - 1:
        current_transition = Transition(adjusted_features[index],
                                        adjusted_features[index + 1],
                                        analysis_obj)
        # check to make sure this is a transition that should be fitted (upper feature has a col max)
        if current_transition.check_features(analysis_obj, params_obj):
            current_transition.fit_transition(analysis_obj.bin_spacing, dt_min=analysis_obj.axes[0][0],
                                              fit_mode=params_obj.ciu50_gauss_mode, gaussian_bool=False)
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
            if dt_diff < bin_to_dt(params_obj.feature_gauss_width_tol,
                                   analysis_obj.axes[0][0],
                                   analysis_obj.bin_spacing):
                # difference is within tolerance; include this CV in the adjusted feature
                final_cvs.append(cv)

        # initialize the new feature using the CV list (it will only have CV and centroid data)
        if len(final_cvs) > 0:
            adj_feature = Feature()
            adj_feature.gauss_median_centroid = feature.gauss_median_centroid
            adj_feature.cvs = final_cvs
            adjusted_features.append(adj_feature)
    return adjusted_features


def plot_features(analysis_obj, params_obj, outputdir, mode):
    """
    Generate a plot of features using gaussian-based feature fitting data previously saved
    to a CIUAnalysisObj.
    :param analysis_obj: CIUAnalysisObj with fitting data previously saved to obj.features_gaussian
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :param mode: gaussian or changept: whether features are from Gaussian or Changepoint based fitting
    :return: void
    """
    # plot the initial CIU contour plot for reference
    plt.clf()
    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, 100, cmap='Blues')

    # plot blue circles of the gaussian centroids found
    # filt_centroids = analysis_obj.get_attribute_by_cv('centroid', True)
    # for x, y in zip(analysis_obj.axes[1], filt_centroids):
    #     plt.scatter([x] * len(y), y)

    # prepare and plot the actual transition using fitted parameters
    feature_index = 1

    if mode == 'gaussian':
        for feature in analysis_obj.features_gaussian:
            feature_x = [gaussian.cv for gaussian in feature.gaussians]
            feature_y = [feature.gauss_median_centroid for _ in feature.gaussians]
            lines = plt.plot(feature_x, feature_y, label='Feature {} median: {:.2f}'.format(feature_index,
                                                                                            feature.gauss_median_centroid))
            feature_index += 1
            plt.setp(lines, linewidth=3, linestyle='--')
    elif mode == 'changept':
        for feature in analysis_obj.features_changept:
            feature_x = feature.cvs
            feature_y = feature.dt_max_vals
            lines = plt.plot(feature_x, feature_y, label='Feature {} median: {:.2f}'.format(feature_index,
                                                                                            np.median(feature.dt_max_vals)))
            feature_index += 1
            plt.setp(lines, linewidth=3, linestyle='--')
    else:
        print('invalid mode')
    plt.legend(loc='best')
    output_path = os.path.join(outputdir, analysis_obj.filename.rstrip('.ciu') + '_features' + params_obj.ciuplot_4_extension)
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
                                                                                              feature.gauss_median_centroid,
                                                                                              feature.cvs[0],
                                                                                              feature.cvs[len(feature.cvs) - 1]))
                outfile.write('CV (V), Centroid, Amplitude, Width, Baseline, FWHM, Resolution\n')
                for gaussian in feature.gaussians:
                    outfile.write(gaussian.print_info() + '\n')
            else:
                outfile.write('Feature {},Median centroid:,{:.2f},CV range:,{} - {}\n'.format(index,
                                                                                              np.median(feature.dt_max_vals),
                                                                                              feature.cvs[0],
                                                                                              feature.cvs[len(feature.cvs) - 1]))
                outfile.write('CV (V),Peak Drift Time')
                cv_index = 0
                for cv in feature.cvs:
                    outfile.write('{},{:.2f}\n'.format(cv, feature.dt_max_vals[cv_index]))
                    cv_index += 1

            index += 1


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

        self.ciu50 = None
        self.fit_params = None
        self.fit_covariances = None
        self.rsq = None

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

    def fit_transition(self, bin_spacing, dt_min, fit_mode, gaussian_bool):
        """
        Fit a logistic function to the transition using the feature information. Requires
        bin_spacing and dt_min to do conversion to DT space from bin space for plotting.
        :param bin_spacing: spacing between drift bins (float)
        :param dt_min: minimum/starting DT for this transition
        :param fit_mode: type of interpolation/fitting to do. Options: 'maxes_only', 'maxes_interpolated',
        'transition_average', 'transition_median'. See parameters file for detailed information.
        :param gaussian_bool: whether this fitting is being done on gaussian data (True) or changepoint (False)
        :return: void (saves fit parameters to object)
        """
        # initial fitting guesses: center is in between the features, min/max are median DTs of features 1 and 2
        center_guess = self.feature2.start_cv_val - (self.feat_distance / 2.0)   # halfway between features
        min_guess = dt_min + (np.median(self.feature1.dt_max_bins) - 1) * bin_spacing
        max_guess = dt_min + (np.median(self.feature2.dt_max_bins) - 1) * bin_spacing
        # guess steepness as a function of between feature1 end and feature2 start
        steepness_guess = self.feat_distance / 10.0

        # for interpolation of transition modes - determine transition region to interpolate
        trans_start_cv = self.feature1.end_cv_val
        trans_end_cv = self.feature2.start_cv_val
        trans_distance = trans_end_cv - trans_start_cv

        # Perform interpolation as specified by the user parameters
        if fit_mode == 1:   # DEPRECATED - basic interpolation always (?) gives better results
            final_x_vals = self.combined_x_axis
            final_y_vals = self.combined_y_vals

        if fit_mode == 2:
            final_x_vals = np.linspace(self.combined_x_axis[0], self.combined_x_axis[len(self.combined_x_axis) - 1],
                                       len(self.combined_x_axis) * 10)
            interp_function = scipy.interpolate.interp1d(self.combined_x_axis, self.combined_y_vals)
            final_y_vals = interp_function(final_x_vals)

        elif fit_mode == 3:
            interp_function_raw = scipy.interpolate.interp1d(self.combined_x_axis, self.combined_y_avg_raw)
            final_x_vals, final_y_vals = self.interpolate_transition(interp_function_raw, trans_start_cv, trans_end_cv,
                                                                     trans_distance)

        elif fit_mode == 4:
            interp_function_raw = scipy.interpolate.interp1d(self.combined_x_axis, self.combined_y_median_raw)
            final_x_vals, final_y_vals = self.interpolate_transition(interp_function_raw, trans_start_cv, trans_end_cv,
                                                                     trans_distance)

        else:
            print('Invalid fitting mode, skipping CIU-50')
            return

        # run the logistic fitting
        try:
            popt, pcov = fit_logistic(final_x_vals, final_y_vals, center_guess, min_guess, max_guess,
                                      steepness_guess)
        except RuntimeError:
            print('fitting failed for {}'.format(self))
            popt = [0, 0, 0, 0]
            pcov = []

        # check goodness of fit
        yfit = logistic_func(final_x_vals, *popt)

        slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(final_y_vals, yfit)
        # adjrsq = adjrsquared(rvalue ** 2, len(cv_col_intensities))
        rsq = rvalue ** 2

        if popt[2] < 0:
            print('WARNING: poor performance from logistic fitting for {}'.format(self))
        self.ciu50 = popt[2]
        self.fit_params = popt
        self.fit_covariances = pcov
        self.rsq = rsq

    def interpolate_transition(self, interp_function_raw, interp_start_cv, interp_end_cv, trans_distance):
        """
        Helper function for interpolating transition region, using the provided interpolation function.
        :param interp_function_raw: Interpolation function from SciPy to use for interpolation
        :param interp_start_cv: CV at which to begin interpolation (start of transition region)
        :param interp_end_cv: CV at which to end interpolation (end of transition region)
        :param trans_distance: length of transition region (indices) to determine number of points to use for interp
        :return: final_x_value np array, final_y_value np array for logistic fitting
        """
        # Use collision voltage step size to interpolate 5 extra bins per CV in the transition region
        try:
            cv_step = self.feature1.cvs[1] - self.feature1.cvs[0]
        except IndexError:
            # feature of size 1 - CV step must be feature distance
            cv_step = trans_distance
        transition_x_vals = np.linspace(interp_start_cv, interp_end_cv, (trans_distance / cv_step) * 5)  # interpolate to 0.5V step
        transition_y_vals = interp_function_raw(transition_x_vals)

        # get index values (position in combined x-axis array) for interpolation start/end CV
        interp_start_index = np.where(self.combined_x_axis == interp_start_cv)[0][0]
        interp_end_index = np.where(self.combined_x_axis == interp_end_cv)[0][0]

        # assemble the x and y arrays (standard res start/end and interpolated high-res transition)
        final_x_vals = self.combined_x_axis[0: interp_start_index]
        final_x_vals = np.append(final_x_vals, transition_x_vals)
        second_half_xvals = self.combined_x_axis[interp_end_index + 1: len(self.combined_x_axis) - 1]
        final_x_vals = np.append(final_x_vals, second_half_xvals)

        final_y_vals = self.combined_y_vals[0: interp_start_index]
        final_y_vals = np.append(final_y_vals, transition_y_vals)
        second_half_yvals = self.combined_y_vals[interp_end_index + 1: len(self.combined_y_vals) - 1]
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
        if self.start_cv > self.end_cv:
            return False

        feature2_cv_indices = np.arange(self.feature2.start_cv_index, self.feature2.end_cv_index)
        width_tol_dt = params_obj.feature_gauss_width_tol * analysis_obj.bin_spacing
        for cv_index in feature2_cv_indices:
            # check if a column max is within tolerance of the feature median
            current_max_dt = analysis_obj.col_max_dts[cv_index]
            if abs(current_max_dt - self.feature2.gauss_median_centroid) <= width_tol_dt:
                self.center_guess_gaussian = analysis_obj.axes[1][cv_index]
                return True

        # no CV found with column max within tolerance - return false
        return False

    def plot_transition(self, analysis_obj, params_obj, outputdir):
        """
        Provide a plot of this transition overlaid on top of the CIU contour plot
        :param analysis_obj: object with CIU data to plot
        :type analysis_obj: CIUAnalysisObj
        :param params_obj: Parameters object with parameter information
        :type params_obj: Parameters
        :param outputdir: directory in which to save output
        :return: void
        """
        x_axis = analysis_obj.axes[1]
        y_data = analysis_obj.col_max_dts

        # plot the initial CIU contour plot for reference
        plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, 100, cmap='jet')

        # plot blue circles of the features/segments assigned
        plt.plot(x_axis, y_data, 'bo')

        # plot the detected changepoints as vertical lines (for testing)
        for cv_val in analysis_obj.changepoint_cvs:
            plt.axvline(x=cv_val)

        # prepare and plot the actual transition using fitted parameters
        interp_x = np.linspace(x_axis[0], x_axis[len(x_axis) - 1], 200)
        y_fit = logistic_func(interp_x, *self.fit_params)
        plt.plot(interp_x, y_fit, 'white', label='CIU50: {:.1f}, r2=: {:.2f}'.format(self.ciu50, self.rsq))
        plt.legend(loc='best')
        filename = os.path.basename(analysis_obj.filename).rstrip('.ciu') + '_transition' + params_obj.ciuplot_4_extension
        output_path = os.path.join(outputdir, filename)
        plt.savefig(output_path)
        # print('c (max): {:.2f}, y0 (min): {:.2f}, x0: {:.2f}, k: {:.2f}'.format(*self.fit_params))


def changepoint_detect(analysis_obj):
    """
    Perform changepoint detection on a CIUAnalysisObj using option 1 below. Return found changepoints
    and also save to analysis object's changepoint_cvs attribute.

    DEV NOTE:
    several options are available to try to perform this:
        i) Convert data to time-series like (only one y value per x) by taking max of each CV col
        ii) offset secondary/etc fitted gaussian centroids in CV (e.g. 5.01V instead of 5) to allow multiple
        iii) partition dataset so that we use the fitted centroids, but only ever have 1 in a partition

    NOTE2:
    There are several package options for Python changepoint analysis:
        i) Very simple, Py3 compatible: https://github.com/ruipgil/changepy
        ii) looks best/most complete, easy to use, but py2 only:  https://github.com/choderalab/cpdetect
        iii) Hard to understand how to use, but looks ok: https://github.com/hildensia/bayesian_changepoint_detection
        iv) Super simple, doesn't look like it does everything well, but easy: https://github.com/JackKelly/bayesianchangepoint

    :param analysis_obj: analysis object with data to be analyzed
    :type analysis_obj: CIUAnalysisObj
    :return: output = list of indices of changepoints, cv_shifts = list of CV values corresponding to changepoints
    """
    # Compute changepoints using Ruipgil's changepy module (best found)
    cost_func = changepy.costs.normal_mean(analysis_obj.col_maxes, variance=0.1)
    output = changepy.pelt(cost_func, len(analysis_obj.col_maxes))
    cv_shifts = [analysis_obj.axes[1][x] for x in output]

    analysis_obj.changepoint_cvs = cv_shifts
    return output, cv_shifts


def changepoint_detect_gaussian(analysis_obj):
    """
    Method to use Gaussian features detected in place of ChangePy module for changepoint detection.
    Guesses midpoints of feature overlap regions to be the change location.
    :param analysis_obj: CIUAnalysis object with gaussian fitting done
    :type analysis_obj: CIUAnalysisObj
    :return: output = list of indices of changepoints, cv_shifts = list of CV values corresponding to changepoints
    """
    # for each valid feature transition, estimate a changepoint
    feat_index = 0
    cv_axis = analysis_obj.axes[1]
    changepoint_values = [cv_axis[0]]   # ensure the starting point of the fingerprint is included
    while feat_index < len(analysis_obj.features_gaussian) - 1:  # in analysis_obj.features_gaussian:
        feature = analysis_obj.features_gaussian[feat_index]
        next_feature = analysis_obj.features_gaussian[feat_index + 1]
        # compute overlap region
        overlap_region_cvs = [x for x in feature.cvs if x in next_feature.cvs]
        if len(overlap_region_cvs) > 0:
            # if an overlap is found, use the middle of it as the changepoint guess
            # changept_guess = np.median(overlap_region_cvs)
            changept_guess = overlap_region_cvs[len(overlap_region_cvs) // 2]
        else:
            # if no overlap is found, use halfway between the end of the first feature and start of the next
            # changept_guess = np.median([feature.end_cv_val, next_feature.start_cv_val])
            changept_guess = abs(next_feature.start_cv_val - feature.end_cv_val) // 2 + feature.end_cv_val

        changepoint_values.append(changept_guess)
        feat_index += 1
    changepoint_indices = [list(cv_axis).index(x) for x in changepoint_values]
    return changepoint_indices, changepoint_values


def partition_to_features(analysis_obj, cv_bin_shift_list, min_feature_len, flat_width_tol=1):
    """
    Use detected changepoint indices to parition CIU data into segments and fit flat features
    to each segment. Applies a width tolerance (in fit_flat_feature) to remove off-feature
    points (may update). Applies a minimum # of points filter to ignore bogus/in-transition
    'features'.
    :param analysis_obj: CIU_object with data to analyze
    :type analysis_obj: CIUAnalysisObj
    :param cv_bin_shift_list: 'output' from changepoint_detect: list of cv_indices corresponding to changepts
    :param min_feature_len: Minimum number of observations to be considered a feature
    :param flat_width_tol: Allowed deviation (in bins) around a feature's most common value
    :rtype: list[Feature]
    :return: list of fitted Feature objects (also saves list to analysis_obj)
    """

    # Partition data into feature segments based on changegpoint analysis
    partitioned_segments = []
    features = []
    prev_index = 0
    cv_bin_shift_list.append(len(analysis_obj.axes[1]))    # ensure that the end of the fingerprint is also a border
    for change_index in cv_bin_shift_list:
        if change_index == prev_index:
            continue
        segment = analysis_obj.col_maxes[prev_index: change_index]
        partitioned_segments.append(segment)

        x_axis = analysis_obj.axes[1][prev_index: change_index]
        feature_dt_bins, feature_cvs, cv_index_list = fit_flat_feature(segment, x_axis, prev_index, flat_width_tol)

        if len(feature_dt_bins) > min_feature_len:
            feature_dt_vals = [bin_to_dt(x, min_dt=analysis_obj.axes[0][0], bin_spacing=analysis_obj.bin_spacing)
                               for x in feature_dt_bins]
            feature = Feature()
            feature.init_feature_data_changept(cv_index_list, feature_cvs, feature_dt_bins, feature_dt_vals)
            features.append(feature)

        prev_index = change_index
    analysis_obj.features = features
    return features


def compute_transitions(analysis_obj, params_obj, features):
    """
    Fit logistic/sigmoidal transition functions to the transition between each sequential pair
    of features in the provided feature list. Saves Transition objects containing combined
    feature pair info, fit, and resulting CIU-50 value.
    :param analysis_obj: CIU analysis object with original data/metadata for this analysis
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param features: list of Feature objects from the analysis_obj
    :type features: list[Feature]
    :rtype: list[Transition]
    :return: list of Transition objects (also saves to analysis_obj)
    """
    # Fit sigmoids for transition calculations
    index = 0
    transition_list = []
    while index < len(features) - 1:
        current_transition = Transition(features[index],
                                        features[index + 1],
                                        analysis_obj)
        current_transition.fit_transition(analysis_obj.bin_spacing, dt_min=analysis_obj.axes[0][0],
                                          fit_mode=params_obj.ciu50_cpt_mode, gaussian_bool=False)
        transition_list.append(current_transition)
        index += 1
    analysis_obj.transitions = transition_list
    return transition_list


def fit_flat_feature(data_segment, x_axis_data, starting_index, bin_tolerance):
    """
    Feature 'detection' within a segment partitioned by changepoint detection. Fits a
    flat line to the most common bin value in the segment and removes any values that
    fall outside the tolerance from that value.
    :param data_segment: List of column max values, in bins
    :param x_axis_data: List of x-axis (CV) values corresponding to segment data for returning
    :param starting_index: starting index of x-axis (CV) in overall dataset for maintaining index list correctly
    :param bin_tolerance: distance (in bins) a point is allowed to deviate from mode and remain in the feature
    :return: list of features
    """
    # mode_bin = scipy.stats.mode(data_segment)[0][0]    # mode returns [[modes], [counts]] so [0][0] is the acutal mode
    med_bin = np.median(data_segment)
    feature_values = []
    feature_cvs = []
    feature_cv_indices = []
    index = 0
    for entry in data_segment:
        if (entry - bin_tolerance) <= med_bin <= (entry + bin_tolerance):
            # include this bin
            feature_values.append(entry)
            feature_cvs.append(x_axis_data[index])
            feature_cv_indices.append(index + starting_index)
        index += 1
    return feature_values, feature_cvs, feature_cv_indices


def bin_to_dt(bin_val, min_dt, bin_spacing):
    dt = min_dt + (bin_val - 1) * bin_spacing
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
    y = c / (1 + np.exp(-k * (x - x0))) + y0
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
    popt, pcov = scipy.optimize.curve_fit(logistic_func, x_axis, y_data, p0=p0, maxfev=5000,
                                          bounds=(fit_bounds_lower, fit_bounds_upper))
    # popt, pcov = scipy.optimize.curve_fit(logistic_func, x_axis, y_data, p0=p0, maxfev=5000)
    return popt, pcov


def feature_detect_changept(analysis_obj, params_obj):
    """
    Run changepoint detection based feature finding. Represents the first half of the original
    changepoint-based CIU50 method, but saves features to the analysis object for plotting/viewing.
    :param analysis_obj: CIUAnalysisObj
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :rtype: CIUAnalysisObj
    :return: updated Analysis object with features_changepoint set
    """
    change_indices, change_cvs = changepoint_detect(analysis_obj)
    features_list = partition_to_features(analysis_obj,
                                          change_indices,
                                          params_obj.feature_cpt_min_length,
                                          params_obj.feature_cpt_width_tol)
    filtered_features = filter_features(features_list, params_obj.feature_cpt_min_length, mode='changept')
    analysis_obj.features_changept = filtered_features
    return analysis_obj


def ciu50_main(analysis_obj, params_obj, outputdir):
    """
    Primary feature detection runner method. Calls appropriate sub-methods using data and
    parameters from the passed analysis object
    :param analysis_obj: CIUAnalysisObj with initial data processed and parameters
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :rtype: CIUAnalysisObj
    :return: updated analysis object with feature detect information saved
    """
    # change_indices, change_cvs = changepoint_detect(analysis_obj)
    # features_list = partition_to_features(analysis_obj,
    #                                       change_indices,
    #                                       params_obj.min_feature_length,
    #                                       params_obj.flat_width_tolerance)
    if analysis_obj.features_changept is None:
        analysis_obj = feature_detect_changept(analysis_obj, params_obj)
    transitions_list = compute_transitions(analysis_obj, params_obj, analysis_obj.features_changept)
    if len(transitions_list) == 0:
        print('No transitions found for file {}'.format(os.path.basename(analysis_obj.filename).rstrip('.ciu')))
    for transition in transitions_list:
        transition.plot_transition(analysis_obj, params_obj, outputdir)
    plt.clf()
    return analysis_obj


def ciu50_gaussians(analysis_obj, params_obj, outputdir):
    """
    CIU-50 method using Gaussian features instead of changepoint features. Requires that gaussian
    fitting and feature detection have previously been performed on the analysis_obj
    :param analysis_obj: CIUAnalysisObj with gaussians and feature data
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :rtype: CIUAnalysisObj
    :return: analysis object
    """
    if analysis_obj.features_gaussian is None:
        feature_detect_gaussians(analysis_obj, params_obj)

    # Adjust Features to avoid inclusion of any points without col maxes
    adj_features = adjust_gauss_features(analysis_obj, params_obj)
    # analysis_obj.features_gaussian = adj_features

    # Catch bad inputs (not enough features to compute a transition)
    if len(adj_features) <= 1:
        filename = os.path.basename(analysis_obj.filename).rstrip('.ciu')
        print('Not enough features (<=1) in file {}. No transition analysis performed'.format(filename))
        return analysis_obj

    # compute transitions and save output
    transitions_list = compute_transitions_gaussian(analysis_obj, params_obj=params_obj, adjusted_features=adj_features)
    if len(transitions_list) == 0:
        print('No transitions found for file {}'.format(os.path.basename(analysis_obj.filename).rstrip('.ciu')))
    for transition in transitions_list:
        transition.plot_transition(analysis_obj, params_obj, outputdir)
    plt.clf()
    return analysis_obj


# testing
if __name__ == '__main__':
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
    main_dir = os.path.dirname(files[0])

    for file in files:
        with open(file, 'rb') as analysis_file:
            obj = pickle.load(analysis_file)
        # ciu50_main(obj, main_dir)
        # obj.features_gaussian = feature_detect_gaussians(obj)
        # plot_feature_gaussians(obj, main_dir)
        # ciu50_gaussians(obj, main_dir)
