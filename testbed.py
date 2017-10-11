"""
Testing module for new algorithms, etc.
DP
10/6/17
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog
from CIU_raw import CIURaw
import Raw_Processing
import CIU_analysis
from CIU_analysis_obj import CIUAnalysisObj
import Feature_Detection

# copied plot module from CIUPlot_DP as starting point
titlex = "Trap Collision Voltage (V)"
# titley = "Collision cross section (A2)"
titley = 'Drift time (ms)'
# default_smooth = 5
default_smooth = 5
# default_crop = [500, 570, 50, 60]     # [cv low, cv high, dt low, dt high]
default_crop = None
# plot_extension = '.pdf'
plot_extension = '.png'
save_output_csv = False
output_title = None
interp_bins = 0     # 0 for no interpolation, 200 is default to interpolate


def get_data(fname):
    """
    Read _raw.csv file and generate a CIURaw object containing its raw data and filename
    :param fname: string - path to _raw.csv file to read
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


def write_ciu_csv(save_path, ciu_data, axes=None):
    """
    Method to write an _raw.csv file for CIU data. If 'axes' is provided, assumes that the ciu_data
    array does NOT contain axes and if 'axes' is None, assumes ciu_data contains axes.
    :param save_path: Full path to save location (SHOULD end in _raw.csv)
    :param ciu_data: 2D numpy array containing CIU data in standard format (rows = DT bins, cols = CV)
    :param axes: (optional) axes labels, provided as (row axis, col axis). if provided, assumes the data array does not contain axes labels.
    :return: void
    """
    with open(save_path, 'w') as outfile:
        if axes is not None:
            # write axes first if they're provided
            args = ['{}'.format(x) for x in axes[1]]    # get the cv-axis now to write to the header
            line = ','.join(args)
            line = ',' + line
            outfile.write(line + '\n')

            index = 0
            for row in ciu_data:
                # insert the axis label at the start of each row
                args = ['{}'.format(x) for x in row]
                args.insert(0, str(axes[0][index]))
                index += 1
                line = ','.join(args)
                outfile.write(line + '\n')
        else:
            # axes are included, so just write everything to file with comma separation
            args = ['{}'.format(x) for x in ciu_data]
            line = ','.join(args)
            outfile.write(line + '\n')


def ciu_plot(data, axes, output_dir, plot_title, x_title, y_title, extension):
    """
    Generate a CIU plot in the provided directory
    :param data: 2D numpy array with rows = DT, columns = CV
    :param axes: axis labels (list of [DT-labels, CV-labels]
    :param output_dir: directory in which to save the plot
    :param plot_title: filename and plot title, INCLUDING file extension (e.g. .png, .pdf, etc)
    :param x_title: x-axis title
    :param y_title: y-axis title
    :param extension: file extension for plotting, default png. Must be image format (.png, .pdf, .jpeg, etc)
    :return: void
    """
    plt.clf()
    output_path = os.path.join(output_dir, plot_title + extension)
    plt.title(plot_title)
    plt.contourf(axes[1], axes[0], data, 100, cmap='jet')  # plot the data
    plt.xlabel(titlex)
    plt.ylabel(titley)
    plt.colorbar(ticks=[0, .25, .5, .75, 1])  # plot a colorbar
    plt.savefig(output_path)
    plt.close()


# Plot the CIU Fingerprint
def ciu_plot_main(input_dir, output_dir, raw_file, smooth_window=None, crop_vals=None, interpolate_bins=0,
                  save_csv=False, save_title=None, extension='.png'):
    """
    Updated CIUSuite_plot to enable modular use with other programs (DP edits). Generates a CIU plot
    for the provided file in the output directory supplied.
    :param input_dir: directory containing input file
    :param output_dir: directory in which to save output
    :param raw_file: (file) the _raw.csv file for which to generate a fingerprint
    :param smooth_window: (odd int) Savitsky-Golay smoothing window. If None, no smoothing is applied.
    :param crop_vals: (list) [CV_min, CV_max, DT_min, DT_max] values to use for cropping. CV = collision voltage,
    DT = drift time
    :param interpolate_bins: (int) number of bins to use for interpolation, 0 for no interpolation. Default 0.
    :param save_csv: if True, write the processed file (with smoothing/cropping/etc) to _raw.csv
    :param save_title: if saving csv, can provide a title to save. If not provided, defaults to filename + '_Processed'
    :param extension: file extension for plotting, default png. Must be image format (.png, .pdf, .jpeg, etc)
    :return: none
    """
    os.chdir(input_dir)
    raw_obj = get_data(raw_file)
    filename = raw_obj.filename

    # normalize, smooth, and crop data (if requested)
    norm_data = Raw_Processing.normalize_by_col(raw_obj.rawdata)

    # interpolate data
    axes = (raw_obj.dt_axis, raw_obj.cv_axis)
    if interpolate_bins > 0:
        norm_data, axes = Raw_Processing.interpolate_cv(norm_data, axes, interpolate_bins)

    if smooth_window is not None:
        norm_data = Raw_Processing.sav_gol_smooth(norm_data, smooth_window)

    if crop_vals is not None:  # If no cropping, use the whole matrix
        norm_data, axes = Raw_Processing.crop(norm_data, axes, crop_vals)

    analysis_obj = CIUAnalysisObj(raw_obj, norm_data, axes)
    CIU_analysis.gaussian_fit_ciu(analysis_obj)

    features = Feature_Detection.feature_detect_gauss(analysis_obj, 0.02)
    no_loners = Feature_Detection.remove_loner_features(features)
    Feature_Detection.print_features_list(features, 'features.csv')
    Feature_Detection.print_features_list(no_loners, 'no-loners.csv')

    if save_csv:
        if save_title is not None:
            save_name = os.path.join(os.path.dirname(raw_file), save_title + '_raw.csv')
        else:
            save_name = raw_file.rstrip('_raw.csv') + '_Processed_raw.csv'
        write_ciu_csv(save_name, norm_data, axes)

    # make the ciu plot
    if extension is None:
        extension = '.png'
    title = filename.rstrip('_raw.csv')
    ciu_plot(norm_data, axes, output_dir, title, titlex, titley, extension)


if __name__ == '__main__':
    """Default/legacy CIU_plot behavior: Asks user to select _raw.csv files and makes fingerprints of them 
    (using 'default smooth' found in this file)"""
    root = tkinter.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(multiple=True, filetypes=[('_raw.csv', '_raw.csv')])
    file_dir = os.path.dirname(os.path.abspath(files[0]))
    for file in files:
        ciu_plot_main(file_dir, file_dir, file, smooth_window=default_smooth, crop_vals=default_crop,
                      interpolate_bins=interp_bins,
                      extension=plot_extension,
                      save_csv=save_output_csv,
                      save_title=output_title)

