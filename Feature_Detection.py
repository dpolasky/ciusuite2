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


def changepoint_detect(analysis_obj, min_feature_length):
    """
    Perform changepoint detection based feature finding on a CIUAnalysisObj

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
    :return:
    """
    # option 1: using max value of column as only value, with ruipgil changepy - BEST
    col_maxes = np.argmax(analysis_obj.ciu_data, axis=0)
    bin_spacing = analysis_obj.axes[0][1] - analysis_obj.axes[0][0]
    col_max_dts = [analysis_obj.axes[0][0] + (x - 1) * bin_spacing for x in col_maxes]

    output = changepy.pelt(changepy.costs.normal_mean(col_maxes, variance=0.1), len(col_maxes))
    cv_shifts = [analysis_obj.axes[1][x] for x in output]

    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, 100, cmap='jet')
    for cv_val in cv_shifts:
        plt.axvline(x=cv_val)
    # plt.show()
    print(cv_shifts)

    # data partition
    partitioned_segments = []
    feature_cv_lists = []
    feature_value_lists = []
    prev_index = 0
    output.append(len(analysis_obj.axes[1]))
    for change_index in output:
        if change_index == prev_index:
            continue
        segment = col_maxes[prev_index: change_index]
        partitioned_segments.append(segment)

        x_axis = analysis_obj.axes[1][prev_index: change_index]

        # fit data
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_axis, segment)
        fit_y = [slope * x + intercept for x in x_axis]
        # plt.plot(x_axis, segment, 'bo')
        # plt.plot(x_axis, fit_y)
        # plt.show()

        feature_values, feature_cvs = fit_flat_feature(segment, x_axis, 1)
        if len(feature_values) > min_feature_length:
            feature_cv_lists.append(feature_cvs)
            feature_value_lists.append(feature_values)

        prev_index = change_index
    for x in feature_cv_lists:
        print(x)

    # Fit sigmoids for transition calculations
    index = 0
    while index < len(feature_cv_lists) - 1:
        # Make a combined 'feature' including all data from start of 1st to end of 2nd feature
        combined_feature_start = feature_cv_lists[index][0]
        combined_feature_end = feature_cv_lists[index + 1][len(feature_cv_lists[index + 1]) - 1]
        start_index = find_nearest(analysis_obj.axes[1], combined_feature_start)
        end_index = find_nearest(analysis_obj.axes[1], combined_feature_end)

        combined_x_axis = analysis_obj.axes[1][start_index: end_index]
        combined_y_vals = col_max_dts[start_index: end_index]

        # initial fitting guesses: center is in between the features, min/max are median DTs of features 1 and 2
        center_guess = feature_cv_lists[index + 1][0]   # first value of second feature
        min_guess = analysis_obj.axes[0][0] + (np.median(feature_value_lists[index]) - 1) * bin_spacing
        max_guess = analysis_obj.axes[0][0] + (np.median(feature_value_lists[index + 1]) - 1) * bin_spacing
        fit_logistic(combined_x_axis, combined_y_vals, center_guess, min_guess, max_guess)

        index += 1


def fit_flat_feature(data_segment, x_axis_data, bin_tolerance):
    """
    Feature 'detection' within a segment partitioned by changepoint detection. Fits a
    flat line to the most common bin value in the segment and removes any values that
    fall outside the tolerance from that value.
    :param data_segment: List of column max values, in bins
    :param x_axis_data: List of x-axis (CV) values corresponding to segment data for returning
    :param bin_tolerance: distance (in bins) a point is allowed to deviate from mode and remain in the feature
    :return: list of features
    """
    mode_bin = scipy.stats.mode(data_segment)[0][0]     # mode returns [[modes], [counts]] so [0][0] is the acutal mode
    feature_values = []
    feature_cvs = []
    index = 0
    for entry in data_segment:
        if (entry - bin_tolerance) <= mode_bin <= (entry + bin_tolerance):
            # include this bin
            feature_values.append(entry)
            feature_cvs.append(x_axis_data[index])
            index += 1
    return feature_values, feature_cvs


def logistic_func(x, c, y0, x0, k):
    """

    :param x:
    :param x0:
    :param k:
    :return:
    """
    y = c / (1 + np.exp(-k * (x - x0))) + y0
    return y


def fit_logistic(x_axis, y_data, guess_center, guess_min, guess_max):
    """

    :param x_axis:
    :param y_data:
    :return:
    """
    # guess initial params: [l, x0, k], default guess k=1
    p0 = [guess_max, guess_min, guess_center, 1]

    popt, pcov = scipy.optimize.curve_fit(logistic_func, x_axis, y_data, p0=p0, maxfev=5000)

    plt.plot(x_axis, y_data, 'bo')

    interp_x = np.linspace(x_axis[0], x_axis[len(x_axis) - 1], 200)
    y_fit = logistic_func(interp_x, *popt)
    plt.plot(interp_x, y_fit, 'r')
    plt.show()
    print('c (max): {:.2f}, y0 (min): {:.2f}, x0: {:.2f}, k: {:.2f}'.format(*popt))


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()  # get the index of the value nearest to the input value
    return idx


# testing
if __name__ == '__main__':
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])

    for file in files:
        with open(file, 'rb') as analysis_file:
            obj = pickle.load(analysis_file)
        changepoint_detect(obj, 3)
