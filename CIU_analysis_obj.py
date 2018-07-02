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
        self.short_filename = os.path.basename(self.raw_obj.filename).rstrip('_raw.csv')

        # CIU data manipulations for common use
        self.bin_spacing = self.axes[0][1] - self.axes[0][0]    # distance between two adjacent DT bins
        self.cv_spacing = self.axes[1][1] - self.axes[1][0]     # distance between two adjacent CV columns
        self.col_maxes = np.argmax(self.ciu_data, axis=0)       # Index of maximum value in each CV column (in DT bins)
        self.col_max_dts = [self.axes[0][x] for x in self.col_maxes]       # DT of maximum value

        # Feature detection results
        self.transitions = []   # type: List[Transition]
        self.features_gaussian = None   # type: List[Feature]
        self.features_changept = None   # type: List[Feature]

        # Gaussian fitting results - raw and following feature detection included
        # self.gaussians = None
        self.raw_protein_gaussians = None
        self.raw_nonprotein_gaussians = None
        self.feat_protein_gaussians = None
        self.gauss_fits_by_cv = None

        # classification (unknown) outputs
        self.classif_predicted_label = None
        self.classif_transformed_data = None
        self.classif_probs_by_cv = None
        self.classif_probs_avg = None

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
        self.col_max_dts = [self.axes[0][x] for x in self.col_maxes]
        # self.col_max_dts = [self.axes[0][0] + (x - 1) * self.bin_spacing for x in self.col_maxes]  # DT of maximum value

    # todo: deprecate
    # def get_attribute_by_cv(self, attribute, filtered):
    #     """
    #     Return a list of lists of the specified attribute at each collision voltage (i.e. [[centroid 1], [centroid 1,
    #     centroid 2], ..]. Attributes must be exact string matches for the attribute name in the Gaussian object
    #     :param attribute: Name (string) of the gaussian attribute to get. Options = 'centroid', 'amplitude', 'width',
    #     etc. See Gaussian class for details.
    #     :param filtered: if True, returns from filtered_gaussians instead of gaussians
    #     :return: CV-sorted list of centroid lists
    #     """
    #     attribute_list = []
    #     if filtered:
    #         for cv_sublist in self.protein_gaussians:
    #             attribute_list.append([getattr(gaussian, attribute) for gaussian in cv_sublist])
    #     else:
    #         for cv_sublist in self.gaussians:
    #             attribute_list.append([getattr(gaussian, attribute) for gaussian in cv_sublist])
    #     return attribute_list
    #
    # def get_attribute_flat(self, attribute, filtered_bool):
    #     """
    #     Return a flattened list (not sorted by CV) of all attributes from a list of Gaussian objects.
    #     Attribute string must exactly match attribute name in Gaussian object.
    #     :param attribute: Name (string) of the gaussian attribute to get. Options = 'centroid', 'amplitude', 'width',
    #     etc. See Gaussian class for details.
    #     :param filtered_bool: if True, returns from filtered_gaussians instead of gaussians
    #     :return: list of attribute values
    #     """
    #     if filtered_bool:
    #         return [getattr(gaussian, attribute) for cv_sublist in self.protein_gaussians for gaussian in cv_sublist]
    #     else:
    #         return [getattr(gaussian, attribute) for cv_sublist in self.gaussians for gaussian in cv_sublist]


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
