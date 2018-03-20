"""
Methods for raw data pre-processing (smoothing, interpolation, etc) for CIU-type data
Author: Dan Polasky
Date: 10/6/2017
"""

import numpy as np
import scipy.signal
import scipy.interpolate
from CIU_raw import CIURaw
from CIU_analysis_obj import CIUAnalysisObj
from CIU_Params import Parameters


def get_data(fname):
    """
    Read _raw.csv file and generate a CIURaw object containing its raw data and filename
    :param fname: string - path to _raw.csv file to read
    :rtype: CIURaw
    :return: CIURaw object with rawdata, axes, and filename initialized
    """
    # TODO: ADD ROBUST ERROR CHECKING HERE
    rawdata = np.genfromtxt(fname, missing_values=[""], filling_values=[0], delimiter=",")
    row_axis = rawdata[1:, 0]
    col_axis = rawdata[0, 1:]
    raw_obj = CIURaw(rawdata[1:, 1:], row_axis, col_axis, fname)
    return raw_obj


# Generate lists of trap collision energies and drift times used for the plots ###
def get_axes(rawdata):
    row_axis = rawdata[1:, 0]
    col_axis = rawdata[0, 1:]
    return row_axis, col_axis


def normalize_by_col(raw_data_matrix):
    """
    Generate a normalized dataset where each column has been normalized to 1 individually. Returns
    normalized 2D matrix only.
    :param raw_data_matrix: CIURaw.rawdata - 2D numpy array with no axes
    :return: Normalized 2D numpy array of raw data
    """
    maxint = np.amax(raw_data_matrix, axis=0)  # find the maximum value for each energy
    if 0 in maxint:
        # set any columns with 0 total intensity to have normalized intensity of 0 as well
        norm = raw_data_matrix / maxint

        index_tup = np.where(maxint == 0)
        index_list = index_tup[0]
        for index in index_list:
            # zero the column at this index (where the max value is 0 - replace all NANs with 0s)
            norm[:, index] = np.zeros(np.shape(norm[:, index]))
    else:
        # no zero values, so simply divide
        norm = raw_data_matrix / maxint
    return norm


def sav_gol_smooth(ciu_data_matrix, smooth_window, smooth_order):
    """
    Apply savitsky-golay smoothing to each column (CV) of the 2D matrix supplied
    :param ciu_data_matrix: input matrix (2D, columns (axis 1) gets smoothed) - supply without axes
    :param smooth_window: Savitsky golay smoothing window to apply
    :param smooth_order: polynomial order to use for smoothing
    :return: smoothed data (same size/format as input)
    """
    # ensure window length is odd
    if smooth_window % 2 == 0:
        smooth_window += 1

    # swap axes to access the columns
    cv_data = np.swapaxes(ciu_data_matrix, 0, 1)
    output_data = np.ndarray(np.shape(cv_data))

    # smooth each column and return the data (axes swapped back to normal)
    index = 0
    while index < len(cv_data):
        smoothed_col = scipy.signal.savgol_filter(cv_data[index], smooth_window, polyorder=smooth_order)
        output_data[index] = smoothed_col
        index += 1
    output_data = np.swapaxes(output_data, 0, 1)

    return output_data


def smooth_main(analysis_obj, params_obj):
    """
    Integrated smoothing method for analysis object/parameter object combinations for general use
    :param analysis_obj: CIU container with data to smooth
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: parameter container with smoothing parameters
    :type params_obj: Parameters
    :return: updated analysis_obj (.ciu_data updated, no other changes)
    """
    norm_data = analysis_obj.ciu_data
    # norm_data = analysis_obj.raw_obj.rawdata
    # norm_data = normalize_by_col(norm_data)

    if params_obj.smoothing_1_method is not None:
        # ensure window size is odd
        if params_obj.smoothing_2_window % 2 == 0:
            params_obj.smoothing_2_window += 1

        i = 0
        while i < params_obj.smoothing_3_iterations:
            if params_obj.smoothing_1_method.lower() == '2d savitzky-golay':
                norm_data = sgolay2d(norm_data, params_obj.smoothing_2_window, order=2)

            elif params_obj.smoothing_1_method.lower() == '1d savitzky-golay':
                norm_data = sav_gol_smooth(norm_data, params_obj.smoothing_2_window, smooth_order=2)

            else:
                print('Invalid smoothing method, no smoothing applied')
                break
            i += 1

    # renormalize data
    norm_data = normalize_by_col(norm_data)

    analysis_obj.ciu_data = norm_data
    return analysis_obj


def sgolay2d(z, window_size, order, derivative=None):
    """
    ADAPTED FROM THE SCIPY COOKBOOK, at http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    accessed 2/14/2018. Performs a 2D Savitzky-Golay smooth on the the provided 2D array z using parameters
    window_size and polynomial order.
    :param z: 2D numpy array of data to smooth
    :param window_size: filter size (int), must be odd to use for smoothing
    :param order: polynomial order (int) to use
    :param derivative: optional (string), values = row, col, both, or None
    :return: smoothed 2D numpy array
    """

    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k-n, n) for k in range(order + 1) for n in range(k+1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    a_mat = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        a_mat[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    z_mat = np.zeros(new_shape)
    # top band
    band = z[0, :]
    z_mat[:half_size, half_size:-half_size] = band - np.abs(np.flipud(z[1:half_size+1, :]) - band)
    # bottom band
    band = z[-1, :]
    z_mat[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size-1:-1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    z_mat[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1:half_size+1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    z_mat[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size-1:-1]) - band)
    # central band
    z_mat[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    z_mat[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1:half_size+1, 1:half_size+1])) - band)
    # bottom right corner
    band = z[-1, -1]
    z_mat[-half_size:, -half_size:] = band + np.abs(np.flipud(np.fliplr(z[-half_size-1:-1, -half_size-1:-1])) - band)
    # top right corner
    band = z_mat[half_size, -half_size:]
    z_mat[:half_size, -half_size:] = band - np.abs(np.flipud(z_mat[half_size+1:2*half_size+1, -half_size:]) - band)
    # bottom left corner
    band = z_mat[-half_size:, half_size].reshape(-1, 1)
    z_mat[-half_size:, :half_size] = band - np.abs(np.fliplr(z_mat[-half_size:, half_size+1:2*half_size+1]) - band)

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(a_mat)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(z_mat, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(a_mat)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(z_mat, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(a_mat)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(z_mat, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(a_mat)[1].reshape((window_size, -1))
        r = np.linalg.pinv(a_mat)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(z_mat, -r, mode='valid'), scipy.signal.fftconvolve(z_mat, -c, mode='valid')


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


def crop(analysis_obj, crop_vals):
    """
    Crops the data and axes arrays to the nearest values specified in the crop vals list.
    If provided crop values are outside the axes, will crop to the nearest value.
    :param analysis_obj: CIUAnalysisObj with data to crop
    :type analysis_obj: CIUAnalysisObj
    :param crop_vals: list of values to crop to in form [cv_low, cv_high, dt_low, dt_high]
    :rtype: CIUAnalysisObj
    :return: New CIUAnalysisObj with cropped data and new axes
    """
    # Determine the indices corresponding to the values nearest to those entered by the user
    dt_axis = analysis_obj.axes[0]
    cv_axis = analysis_obj.axes[1]
    ciu_data_matrix = analysis_obj.ciu_data

    # Crop
    dt_low = find_nearest(dt_axis, crop_vals[0])
    dt_high = find_nearest(dt_axis, crop_vals[1])
    cv_low = find_nearest(cv_axis, crop_vals[2])
    cv_high = find_nearest(cv_axis, crop_vals[3])

    # allow single axis cropping - ignore one axis if its provided values are equal
    crop_dt = True
    crop_cv = True
    if dt_high == dt_low:
        crop_dt = False
    if cv_low == cv_high:
        crop_cv = False

    # crop the data and axes
    if crop_dt:
        ciu_data_matrix = ciu_data_matrix[dt_low:dt_high + 1]  # crop the rows
        dt_axis = dt_axis[dt_low:dt_high + 1]
    ciu_data_matrix = np.swapaxes(ciu_data_matrix, 0, 1)
    if crop_cv:
        ciu_data_matrix = ciu_data_matrix[cv_low:cv_high + 1]  # crop the columns
        cv_axis = cv_axis[cv_low:cv_high + 1]
    ciu_data_matrix = np.swapaxes(ciu_data_matrix, 0, 1)
    new_axes = [dt_axis, cv_axis]

    # save output to the analysis object
    # analysis_obj.ciu_data = ciu_data_matrix
    # analysis_obj.axes = new_axes

    # save output to a new analysis object (clears fitting results/etc that can fail if axes are different)
    new_obj = CIUAnalysisObj(analysis_obj.raw_obj, ciu_data_matrix, new_axes, analysis_obj.params)

    return new_obj


def interpolate_axes(analysis_obj, new_axes):
    """
    interpolate along the collision voltage (x) axis to allow for unevenly spaced data collection
    :param analysis_obj: input data object
    :type analysis_obj: CIUAnalysisObj
    :param new_axes: new axes onto which to interpolate in form [dt_axis, cv_axis]
    :return: updated object with interpolate CIUData and axes
    :rtype: CIUAnalysisObj
    """
    # interpolate the existing data, then reframe on new axes
    interp_func = scipy.interpolate.interp2d(analysis_obj.axes[0],
                                             analysis_obj.axes[1],
                                             analysis_obj.ciu_data.T)
    interp_data = interp_func(new_axes[0], new_axes[1])
    ciu_interp_data = interp_data.T

    # save to analysis object and return
    analysis_obj.ciu_data = ciu_interp_data
    analysis_obj.axes = new_axes
    return analysis_obj


def average_ciu(list_of_data_matrices):
    """
    Average CIU fingerprints and return the averaged and standard deviation matrices (SD only
    if n >= 3).
    :param list_of_data_matrices: List of CIU data matrices (2D numpy array with no axes label info)
    :return: averaged_matrix, std_dev_matrix as 2D numpy arrays of same shape as input
    """
    avg_matrix = np.mean(list_of_data_matrices, axis=0)
    std_matrix = np.std(list_of_data_matrices, axis=0)

    return avg_matrix, std_matrix


def equalize_axes(flat_analysisobj_list, crop_vals_plus_flag=None):
    """
    Ensure that all objects have identical axes. If not, crop to the smallest shared
    region and interpolate so that all axes are identical. Return updated obj list
    :param flat_analysisobj_list: flat list of CIUAnalysisObj's
    :type flat_analysisobj_list: list[CIUAnalysisObj]
    :param crop_vals_plus_flag: list of previously saved axis values to enable comparison across lists
    :return: updated object list with equalized axes, list of final cropping values used
    :rtype: list[CIUAnalysisObj], list
    """
    if crop_vals_plus_flag is None:
        base_axes = flat_analysisobj_list[0].axes
        dt_start_min = base_axes[0][0]
        dt_start_max = base_axes[0][len(base_axes[0]) - 1]
        cv_start_min = base_axes[1][0]
        cv_start_max = base_axes[1][len(base_axes[1]) - 1]
        max_len_dt = len(base_axes[0])
        max_len_cv = len(base_axes[1])
        adjust_flag = False     # if at least one file has different axes, ALL will be adjusted
        crop_vals_plus_flag = [dt_start_min, dt_start_max, cv_start_min, cv_start_max, max_len_dt, max_len_cv, adjust_flag]
    else:
        dt_start_min = crop_vals_plus_flag[0]
        dt_start_max = crop_vals_plus_flag[1]
        cv_start_min = crop_vals_plus_flag[2]
        cv_start_max = crop_vals_plus_flag[3]
        max_len_dt = crop_vals_plus_flag[4]
        max_len_cv = crop_vals_plus_flag[5]
        adjust_flag = crop_vals_plus_flag[6]

        base_dt_axis = np.linspace(dt_start_min, dt_start_max, max_len_dt)
        base_cv_axis = np.linspace(cv_start_min, cv_start_max, max_len_cv)
        base_axes = [base_dt_axis, base_cv_axis]

    for analysis_obj in flat_analysisobj_list:
        if np.array_equal(analysis_obj.axes[0], base_axes[0]) and np.array_equal(analysis_obj.axes[1], base_axes[1]):
            adjust_flag = False
            continue
        else:
            # Determine minimum shared region in both dimensions
            if analysis_obj.axes[0][0] > dt_start_min:
                dt_start_min = analysis_obj.axes[0][0]
            if analysis_obj.axes[0][len(analysis_obj.axes[0]) - 1] < dt_start_max:
                dt_start_max = analysis_obj.axes[0][len(analysis_obj.axes[0]) - 1]
            if analysis_obj.axes[1][0] > cv_start_min:
                cv_start_min = analysis_obj.axes[1][0]
            if analysis_obj.axes[1][len(analysis_obj.axes[1]) - 1] < cv_start_max:
                cv_start_max = analysis_obj.axes[1][len(analysis_obj.axes[1]) - 1]

            # Determine max axis length for interpolation (to ensure no files are undersampled)
            if len(analysis_obj.axes[0]) > max_len_dt:
                max_len_dt = len(analysis_obj.axes[0])
            if len(analysis_obj.axes[1]) > max_len_cv:
                max_len_cv = len(analysis_obj.axes[1])

            adjust_flag = True

    if adjust_flag:
        print('Axes differed in some files; interpolating to equalize...')
        output_obj_list = []
        # adjust ALL files to the same final axes
        crop_vals = [dt_start_min, dt_start_max, cv_start_min, cv_start_max, max_len_dt, max_len_cv]
        for analysis_obj in flat_analysisobj_list:
            new_dt_axis = np.linspace(dt_start_min, dt_start_max, max_len_dt)
            new_cv_axis = np.linspace(cv_start_min, cv_start_max, max_len_cv)
            analysis_obj = interpolate_axes(analysis_obj, new_axes=[new_dt_axis, new_cv_axis])
            output_obj_list.append(analysis_obj)
        crop_vals_plus_flag = crop_vals
        crop_vals_plus_flag.append(True)
        return output_obj_list, crop_vals_plus_flag
    else:
        return flat_analysisobj_list, crop_vals_plus_flag


def equalize_axes_2d_list(analysis_obj_list_by_label):
    """
    Axis checking method for 2D lists (e.g. in classification) where list order/shape must be
    preserved. Axes are equalized across ALL sublists (every object anywhere in the 2D list)
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObjs
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :return: updated list of lists with axes equalized
    :rtype: list[list[CIUAnalysisObj]], output_axes_list
    """
    output_list_by_label = []
    crop_vals_to_equalize = None
    for analysis_obj_list in analysis_obj_list_by_label:
        output_list, crop_vals_to_equalize = equalize_axes(analysis_obj_list, crop_vals_to_equalize)

    # loop through a second time in case any changes occurred in later lists (e.g. if list 2 is cropped, this will also get list 1)
    for analysis_obj_list in analysis_obj_list_by_label:
        output_list, crop_vals_to_equalize = equalize_axes(analysis_obj_list, crop_vals_to_equalize)
        output_list_by_label.append(output_list)
    return output_list_by_label, crop_vals_to_equalize
