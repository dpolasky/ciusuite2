"""
Dan Polasky
10/6/17
"""
import os
import numpy as np
import tkinter
from tkinter import filedialog
import pickle
from CIU_Params import Parameters
from Feature_Detection import Feature, Transition
from CIU_raw import CIURaw
from typing import List


class CIUAnalysisObj(object):
    """
    Container for analysis/processed information from a CIU fingerprint. Requires a CIURaw object
    to start, additional fields added as data is processed.
    """
    def __init__(self, ciu_raw_obj, ciu_data, axes, params_obj):
        """
        Initialize with raw data and axes. Allows addition of Gaussian fitting data later
        :param ciu_raw_obj: Object containing initial raw data, axes, and filepath of the analysis
        :param ciu_data: pre-processed data (smoothed/interpolated/cropped/etc) - can be modified repeatedly
        :param axes: modified axes corresponding to ciu_data (axes[0] = DT, axes[1] = CV)
        :param params_obj: Parameters object with information about how this object was processed
        :type params_obj: Parameters
        """
        # basic information and objects
        self.raw_obj = ciu_raw_obj  # type: CIURaw
        self.raw_obj_list = None    # used for replicates (averaged fingerprints) only
        self.ciu_data = ciu_data
        self.axes = axes            # convention: axis 0 = DT, axis 1 = CV
        self.crop_vals = None
        self.params = params_obj  # type: Parameters
        self.filename = None        # filename of .ciu file saved
        self.short_filename = None

        # CIU data manipulations for common use
        self.bin_spacing = self.axes[0][1] - self.axes[0][0]    # distance between two adjacent DT bins
        self.cv_spacing = self.axes[1][1] - self.axes[1][0]     # distance between two adjacent CV columns
        self.col_maxes = np.argmax(self.ciu_data, axis=0)       # Index of maximum value in each CV column (in DT bins)
        self.col_max_dts = [self.axes[0][0] + (x - 1) * self.bin_spacing for x in self.col_maxes]  # DT of maximum value

        # Feature detection results
        self.transitions = []   # type: List[Transition]
        self.features_gaussian = None   # type: List[Feature]
        self.features_changept = None   # type: List[Feature]

        # Gaussian fitting parameters - not always initialized with the object
        self.gaussians = None
        self.filtered_gaussians = None
        self.gauss_adj_r2s = None
        self.gauss_fits = None
        self.gauss_covariances = None
        self.gauss_r2s = None
        self.gauss_fit_stats = None

        self.classif_predicted_outputs = None
        self.classif_data = None
        self.classif_gaussfeats = None

    def __str__(self):
        """
        Display the filename of the object as reference
        :return: void
        """
        return '<CIUAnalysisObj> file: {}'.format(os.path.basename(self.filename.rstrip('.ciu')))
    __repr__ = __str__

    def refresh_data(self):
        """
        Recalculate column max values and other basic data attributes. Should be performed after any
        adjustments to the ciu_data (e.g. crop, interpolate, smooth, etc)
        :return:
        """
        self.bin_spacing = self.axes[0][1] - self.axes[0][0]  # distance between two adjacent DT bins
        self.cv_spacing = self.axes[1][1] - self.axes[1][0]  # distance between two adjacent CV columns
        self.col_maxes = np.argmax(self.ciu_data, axis=0)  # Index of maximum value in each CV column (in DT bins)
        self.col_max_dts = [self.axes[0][0] + (x - 1) * self.bin_spacing for x in self.col_maxes]  # DT of maximum value

    def get_attribute_by_cv(self, attribute, filtered):
        """
        Return a list of lists of the specified attribute at each collision voltage (i.e. [[centroid 1], [centroid 1,
        centroid 2], ..]. Attributes must be exact string matches for the attribute name in the Gaussian object
        :param attribute: Name (string) of the gaussian attribute to get. Options = 'centroid', 'amplitude', 'width',
        etc. See Gaussian class for details.
        :param filtered: if True, returns from filtered_gaussians instead of gaussians
        :return: CV-sorted list of centroid lists
        """
        attribute_list = []
        if filtered:
            for cv_sublist in self.filtered_gaussians:
                attribute_list.append([getattr(gaussian, attribute) for gaussian in cv_sublist])
        else:
            for cv_sublist in self.gaussians:
                attribute_list.append([getattr(gaussian, attribute) for gaussian in cv_sublist])
        return attribute_list

    def get_attribute_flat(self, attribute, filtered_bool):
        """
        Return a flattened list (not sorted by CV) of all attributes from a list of Gaussian objects.
        Attribute string must exactly match attribute name in Gaussian object.
        :param attribute: Name (string) of the gaussian attribute to get. Options = 'centroid', 'amplitude', 'width',
        etc. See Gaussian class for details.
        :param filtered_bool: if True, returns from filtered_gaussians instead of gaussians
        :return: list of attribute values
        """
        if filtered_bool:
            return [getattr(gaussian, attribute) for cv_sublist in self.filtered_gaussians for gaussian in cv_sublist]
        else:
            return [getattr(gaussian, attribute) for cv_sublist in self.gaussians for gaussian in cv_sublist]

    def save_ciu50_outputs(self, outputpath, mode, combine=False):
        """
        Print feature detection outputs to file. Must have feature detection already performed.
        **NOTE: currently, feature plot is still in the feature detect module, but could (should?)
        be moved here eventually.
        :param mode: 'gaussian' or 'changept' - which type of feature to save
        :param outputpath: directory in which to save output
        :param combine: whether to output directly for this file or return a string for combining
        :return: void
        """
        output_name = os.path.join(outputpath, self.filename + '_features.csv')
        output_string = ''

        # DEPRECATED, now that Feature detection has its own output methods
        # assemble the output
        # output_string += 'Features:, CV_lower (V),CV_upper (V),DT mode,DT_lower,DT_upper, rsq\n'
        # feat_index = 1
        # if mode == 'gaussian':
        #     features_list = self.features_gaussian
        # else:
        #     features_list = self.features_changept

        # for feature in features_list:
        #     output_string += 'Feature {},'.format(feat_index)
        #     output_string += '{},{},'.format(feature.start_cv_val, feature.end_cv_val)
        #     output_string += '{:.2f},'.format(scipy.stats.mode(feature.dt_max_vals)[0][0])
        #     output_string += '{:.2f},{:.2f}\n'.format(np.min(feature.dt_max_vals),
        #                                               np.max(feature.dt_max_vals))
        #     feat_index += 1
        output_string += 'Transitions:,y0 (ms),ymax (ms),CIU-50 (V),k (steepness),r_squared\n'
        trans_index = 1
        for transition in self.transitions:
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

    def save_ciu50_short(self, outputpath, combine=False):
        """
        Helper method to also save a shortened version of feature information
        :param outputpath: directory in which to save output
        :param combine: If True, return a string to be combined with other files instead of saving to file
        :return:
        """
        output_name = os.path.join(outputpath, self.filename + '_transitions-short.csv')
        output_string = ''

        # assemble the output
        for transition in self.transitions:
            output_string += ',{:.2f}'.format(transition.fit_params[2])
        output_string += '\n'

        if combine:
            # return the output string to be written together with many files
            return output_string
        else:
            with open(output_name, 'w') as outfile:
                outfile.write(output_string)


# testing
if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(filetypes=[('pickled gaussian files', '.pkl')])
    files = list(files)
    file_dir = os.path.dirname(files[0])

    for file in files:
        with open(file, 'rb') as first_file:
            ciu1 = pickle.load(first_file)
        ciu1.plot_centroids(file_dir, [10, 20])
        ciu1.save_gauss_params(file_dir)
