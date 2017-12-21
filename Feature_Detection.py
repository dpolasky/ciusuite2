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
import os


class Feature(object):
    """
    Holder for feature information while doing feature detection
    """
    def __init__(self, mean_centroid=None, length=None, cvs=None, centroids=None):
        """
        Create a new feature object to hold feature information. Intended to add to cv/centroid info over time
        :param mean_centroid: Average centroid of this feature
        :param length: Number of x-axis values over which this feature exists
        """
        self.mean_centroid = mean_centroid
        self.length = length

        self.cvs = []
        if cvs is not None:
            self.cvs = cvs

        self.centroids = []
        if centroids is not None:
            self.centroids = []

    def finish(self):
        """
        Method to compute mean centroid and length once all cv and centroid information has been added to
        the respective arrays.
        :return: void
        """
        self.length = len(self.cvs)
        self.mean_centroid = np.average(np.asarray(self.centroids))

    def combine(self, feature_to_absorb):
        """
        Combine this feature with another, by adding that feature's cvs and centroids to this ones
        :param feature_to_absorb: Feature object
        :return: void
        """
        self.cvs.extend(feature_to_absorb.cvs)
        self.centroids.extend(feature_to_absorb.centroids)
        self.finish()


def feature_detect_gauss(ciu_obj, ratio_change):
    """
    Used fitted gaussians to determine the locations of CIU features in the provided data.
    :param ciu_obj: CIUAnalysisObj with gaussian data previously calculated
    :param ratio_change: ratio change (0 < x < 1) to centroid to indicate movement to a new feature
    :return: list of Feature objects
    """
    centroids = np.asarray(ciu_obj.gauss_centroids)
    cv_axis = ciu_obj.axes[1]

    init_centroid = np.average(centroids[0:2])

    features = []
    current_feature_cent = init_centroid
    current_feature = Feature()

    index = 0
    for centroid in centroids:
        change_from_last = (centroid - current_feature_cent) / float(current_feature_cent)
        if change_from_last > ratio_change:
            # new feature - finish current feature and begin a new one
            current_feature.finish()
            features.append(current_feature)
            current_feature = Feature()
        current_feature.centroids.append(centroid)
        current_feature.cvs.append(cv_axis[index])

        current_feature_cent = centroid
        index += 1
    # append final feature
    current_feature.finish()
    features.append(current_feature)
    return features


def remove_loner_features(feature_list):
    """
    Method to combine series' of individual "features" that are generated during a sustained movement
    of peak centroids over many collision voltages. May or may not keep in final version
    :param feature_list: List of Feature objects
    :return: updated list of Feature objects with loners combined
    """
    index = 0
    new_feature_list = []
    while index < len(feature_list):
        feature = feature_list[index]
        if feature.length == 1:
            while check_next(feature_list, index + 1):
                # if the next feature's length is 1 too, add it to this one as well
                feature.combine(feature_list[index + 1])
                index += 1
            new_feature_list.append(feature)
            index += 1
        else:
            new_feature_list.append(feature)
            index += 1
    return new_feature_list


def check_next(feature_list, index):
    """
    Helper method for remove loners - returns True if the feature at the specified index has a length of 1
    :param feature_list: List of features
    :param index: index at which to look
    :return:
    """
    try:
        if feature_list[index].length == 1:
            return True
        else:
            return False
    except IndexError:
        # this is the last feature, ignore
        return False


def print_features_list(feature_list, outputpath):
    """
    Write feature information to file
    :param feature_list: list of Feature objects
    :param outputpath: directory in which to save output
    :return: void
    """
    with open(outputpath, 'w') as outfile:
        index = 1
        for feature in feature_list:
            cvline = 'CV:,' + ','.join(['{:.2f}'.format(x) for x in feature.cvs])
            centroidline = 'Centroid:,' + ','.join(['{:.2f}'.format(x) for x in feature.centroids])
            outfile.write('Feature,{}\nmean centroid,{},length,{}\n{}\n{}\n\n'
                          .format(index, feature.mean_centroid, feature.length, cvline, centroidline))
            index += 1


class ChangeptFeature(object):
    """
    Object to store information about a partitioned segment of a CIU fingerprint containing
    a flat feature. Stores the range of indices and CV values covered by the feature and
    its centroid DT index and value.
    """
    def __init__(self, cv_index_list, cv_val_list, dt_bin_list, dt_val_list):
        """
        Object to store information for a single feature in a CIU fingerprint. Holds primarily
        information about the range of collision voltages/indices over which the feature is
        present and the drift bin containing the max value of each contained CV column
        :param cv_index_list: list of indices of collision voltages that make up this feature (args for sublist of CV axis)
        :param cv_val_list: list of actual collision voltage values that make up this feature (sublist of CV axis)
        :param dt_bin_list: list of max_dt_bin entries for each collision voltage in the feature
        :param dt_val_list: list of max_dt values in ms for each collision voltage in the feature
        """
        self.start_cv_val = cv_val_list[0]
        self.end_cv_val = cv_val_list[len(cv_val_list) - 1]
        self.cv_vals = cv_val_list

        self.start_cv_index = cv_index_list[0]
        self.end_cv_index = cv_index_list[len(cv_index_list) - 1]
        self.cv_indices = cv_index_list

        self.dt_max_bins = dt_bin_list
        self.dt_max_vals = dt_val_list


class Transition(object):
    """
    Store information about a CIU transition, including the starting and ending feature,
    their combined CV/index range and DT data, and fitted logistic function parameters
    and CIU50.
    """
    def __init__(self, feature1, feature2, whole_cv_axis, whole_dt_maxes):
        """
        Create a combined Transition object from two identified features. Features MUST be
        adjacent in CV space for this to make sense.
        :param feature1: Lower CV ("earlier/starting") Feature object
        :param feature2: Higher CV ("later/ending") Feature object
        :param whole_cv_axis:
        :param whole_dt_maxes:
        """
        self.feature1 = feature1
        self.feature2 = feature2

        self.start_cv = feature1.start_cv_val
        self.end_cv = feature2.end_cv_val
        self.start_index = feature1.start_cv_index
        self.end_index = feature2.end_cv_index

        self.combined_x_axis = whole_cv_axis[self.start_index: self.end_index]
        self.combined_y_vals = whole_dt_maxes[self.start_index: self.end_index]

        self.ciu50 = None
        self.fit_params = None
        self.fit_covariances = None

    def fit_transition(self, bin_spacing, dt_min):
        # initial fitting guesses: center is in between the features, min/max are median DTs of features 1 and 2
        center_guess = self.feature2.start_cv_val  # first value of second feature
        min_guess = dt_min + (np.median(self.feature1.dt_max_bins) - 1) * bin_spacing
        max_guess = dt_min + (np.median(self.feature2.dt_max_bins) - 1) * bin_spacing
        # guess steepness by getting distance between feature1 end and feature2 start
        feat_distance = self.feature2.start_cv_val - self.feature1.end_cv_val
        steepness_guess = feat_distance / 100.0
        # steepness_guess = 0.15
        try:
            popt, pcov = fit_logistic(self.combined_x_axis, self.combined_y_vals, center_guess, min_guess, max_guess,
                                      steepness_guess)
        except RuntimeError:
            print('fitting FAILED!')
            popt = [0, 0, 0, 0]
            pcov = []
        if popt[2] < 0:
            print("""WARNING: poor performance from logistic fitting. This can happen if
                the transitions are """)
        self.ciu50 = popt[2]
        self.fit_params = popt
        self.fit_covariances = pcov

    def plot_transition(self, analysis_obj, outputdir):
        """
        Provide a plot of this transition overlaid on top of the CIU contour plot
        :param analysis_obj: object with CIU data to plot
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
        plt.plot(interp_x, y_fit, 'white', label='CIU50 = {:.2f}, k= {:.2f}'.format(self.ciu50, self.fit_params[3]))
        plt.legend(loc='best')
        output_path = os.path.join(outputdir, analysis_obj.filename + '_transition' + analysis_obj.params.plot_extension)
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
    :return: output = list of indices of changepoints, cv_shifts = list of CV values corresponding to changepoints
    """
    # Compute changepoints using Ruipgil's changepy module (best found)
    cost_func = changepy.costs.normal_mean(analysis_obj.col_maxes, variance=0.1)
    output = changepy.pelt(cost_func, len(analysis_obj.col_maxes))
    cv_shifts = [analysis_obj.axes[1][x] for x in output]

    analysis_obj.changepoint_cvs = cv_shifts
    return output, cv_shifts


def partition_to_features(analysis_obj, cv_bin_shift_list, min_feature_len, flat_width_tol=1):
    """
    Use detected changepoint indices to parition CIU data into segments and fit flat features
    to each segment. Applies a width tolerance (in fit_flat_feature) to remove off-feature
    points (may update). Applies a minimum # of points filter to ignore bogus/in-transition
    'features'.
    :param analysis_obj: CIU_object with data to analyze
    :param cv_bin_shift_list: 'output' from changepoint_detect: list of cv_indices corresponding to changepts
    :param min_feature_len: Minimum number of observations to be considered a feature
    :param flat_width_tol: Allowed deviation (in bins) around a feature's most common value
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
            feature = ChangeptFeature(cv_index_list, feature_cvs, feature_dt_bins, feature_dt_vals)
            features.append(feature)

        prev_index = change_index
    analysis_obj.features = features
    return features


def compute_transitions(analysis_obj, features):
    """
    Fit logistic/sigmoidal transition functions to the transition between each sequential pair
    of features in the provided feature list. Saves Transition objects containing combined
    feature pair info, fit, and resulting CIU-50 value.
    :param analysis_obj: CIU analysis object with original data/metadata for this analysis
    :param features: list of Feature objects from the analysis_obj
    :return: list of Transition objects (also saves to analysis_obj)
    """
    # Fit sigmoids for transition calculations
    index = 0
    transition_list = []
    while index < len(features) - 1:
        current_transition = Transition(features[index],
                                        features[index + 1],
                                        analysis_obj.axes[1],
                                        analysis_obj.col_max_dts)
        current_transition.fit_transition(analysis_obj.bin_spacing, dt_min=analysis_obj.axes[0][0])
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
    mode_bin = scipy.stats.mode(data_segment)[0][0]     # mode returns [[modes], [counts]] so [0][0] is the acutal mode
    feature_values = []
    feature_cvs = []
    feature_cv_indices = []
    index = 0
    for entry in data_segment:
        if (entry - bin_tolerance) <= mode_bin <= (entry + bin_tolerance):
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
    popt, pcov = scipy.optimize.curve_fit(logistic_func, x_axis, y_data, p0=p0, maxfev=5000)
    # bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    return popt, pcov


def feature_detect_main(analysis_obj, outputdir):
    """
    Primary feature detection runner method. Calls appropriate sub-methods using data and
    parameters from the passed analysis object
    :param analysis_obj: CIUAnalysisObj with initial data processed and parameters
    :param outputdir: directory in which to save output
    :return: updated analysis object with feature detect information saved
    """
    change_indices, change_cvs = changepoint_detect(analysis_obj)
    features_list = partition_to_features(analysis_obj,
                                          change_indices,
                                          analysis_obj.params.min_feature_length,
                                          analysis_obj.params.flat_width_tolerance)
    transitions_list = compute_transitions(analysis_obj, features_list)
    if len(transitions_list) == 0:
        print('No transitions found for file {}' + os.path.basename(analysis_obj.filename).rstrip('.ciu'))
    for transition in transitions_list:
        transition.plot_transition(analysis_obj, outputdir)
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
        feature_detect_main(obj, main_dir)

