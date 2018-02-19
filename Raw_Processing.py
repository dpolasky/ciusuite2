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
    norm_data = normalize_by_col(analysis_obj.raw_obj.rawdata)

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
    A = np.empty((window_size**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros(new_shape)
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(np.flipud(z[1:half_size+1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size-1:-1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1:half_size+1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size-1:-1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1:half_size+1, 1:half_size+1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(np.flipud(np.fliplr(z[-half_size-1:-1, -half_size-1:-1])) - band)
    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(np.flipud(Z[half_size+1:2*half_size+1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band)

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')


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
    new_axes = analysis_obj.axes
    ciu_data_matrix = analysis_obj.ciu_data

    # check for interpolation
    if not len(dt_axis) == crop_vals[4]:
        ciu_data_matrix, new_axes = interpolate_axis(ciu_data_matrix, new_axes, 0, crop_vals[4])
    if not len(cv_axis) == crop_vals[5]:
        ciu_data_matrix, new_axes = interpolate_axis(ciu_data_matrix, new_axes, 1, crop_vals[5])

    dt_axis = new_axes[0]
    cv_axis = new_axes[1]

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
    # ciu_data_matrix = analysis_obj.ciu_data
    if crop_dt:
        ciu_data_matrix = ciu_data_matrix[dt_low:dt_high + 1]  # crop the rows
        dt_axis = dt_axis[dt_low:dt_high + 1]
    ciu_data_matrix = np.swapaxes(ciu_data_matrix, 0, 1)
    if crop_cv:
        ciu_data_matrix = ciu_data_matrix[cv_low:cv_high + 1]  # crop the columns
        cv_axis = cv_axis[cv_low:cv_high + 1]
    ciu_data_matrix = np.swapaxes(ciu_data_matrix, 0, 1)
    new_axes = [dt_axis, cv_axis]

    # crop_obj = CIUAnalysisObj(analysis_obj.raw_obj, ciu_data_matrix, new_axes,
    #                           analysis_obj.gauss_params)
    # crop_obj.params = analysis_obj.params
    # crop_obj.raw_obj_list = analysis_obj.raw_obj_list
    # save output to the analysis object
    analysis_obj.ciu_data = ciu_data_matrix
    analysis_obj.axes = new_axes
    return analysis_obj


def interpolate_axis(norm_data, axes, axis_to_interp, num_bins):
    """
    interpolate along the collision voltage (x) axis to allow for unevenly spaced data collection
    :param norm_data: input data (with axes still present) to interpolate
    :param axes: axes list to use for interpolation (x-axis, y-axis)
    :param axis_to_interp: which axis (DT = 0, CV = 1) to interpolate
    :param num_bins: number of bins to have after interpolate
    :return: interpolated data matrix, updated axes
    """
    # Update the desired axis
    interp_axis = axes[axis_to_interp]
    start_val = interp_axis[0]
    end_val = interp_axis[len(interp_axis) - 1]
    new_axis_vals = np.linspace(start_val, end_val, num_bins)

    if axis_to_interp == 0:
        # interpolate DT (rows)
        new_axes = [new_axis_vals, axes[1]]
    else:
        new_axes = [axes[0], new_axis_vals]

    for x in [axes[0], axes[1], norm_data]:
        print(np.shape(x))
    interp_func = scipy.interpolate.interp2d(axes[0], axes[1], norm_data.T)
    interp_data = interp_func(new_axes[0], new_axes[1])
    ciu_interp_data = interp_data.T

    # interpolated_data = []
    # for ycolumn in norm_data[:, :]:
    #     interp_func = scipy.interpolate.interp1d(xaxis, ycolumn)
    #     new_intensities = interp_func(new_cv_axis)
    #     interpolated_data.append(new_intensities)

    return ciu_interp_data, new_axes


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


