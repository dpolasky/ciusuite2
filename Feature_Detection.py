"""
Module for feature detection. Relies on CIUAnalysisObj from Gaussian fitting module
Author: DP
Date: 10/10/2017
"""

import numpy as np


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