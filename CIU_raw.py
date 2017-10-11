"""
Raw data object for CIU-type analyses.
Author: Dan Polasky
Date: 10/6/2017
"""
import os


class CIURaw:
    """
    Raw data container for CIU-type data. Analogous to the _raw.csv file in original CIUSuite.
    Holds raw data, axis information, and metadata/parameters as needed
    """
    def __init__(self, raw_data_no_axes, drift_axis, activation_axis, filepath):
        """
        Constructor - requires input data and axis information
        :param raw_data_no_axes: 2D numpy array containing raw data. Rows (axis 0) refer to drift time,
        columns (axis 1) refer to activation steps.
        :param activation_axis: List of activation values corresponding to columns of raw data matrix
        :param drift_axis: List of drift time values corresponding to rows of raw data matrix
        :param filepath: full system path to raw file used
        """
        self.rawdata = raw_data_no_axes
        self.dt_axis = drift_axis
        self.cv_axis = activation_axis
        self.filepath = filepath
        self.filename = os.path.basename(filepath)


