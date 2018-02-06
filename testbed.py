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
import Raw_Processing
import Gaussian_Fitting
from CIU_analysis_obj import CIUAnalysisObj
import pickle
import Feature_Detection


# ********* PARAMETERS TO EDIT IN THE DEVELOPMENT/TESTING VERSION ***************

titlex = "Trap Collision Voltage (V)"
titley = 'Drift time (ms)'

default_smooth = 7
num_smooths = 1

# default_crop = [500, 570, 50, 60]     # [cv low, cv high, dt low, dt high]
default_crop = None
interp_bins = 0     # 0 for no interpolation, 200 is default to interpolate

plot_extension = '.png'
save_output_csv = False
output_title = None

# gaussian peak fitting parameters
gaussian_int_thr = 0.1     # intensity threshold for peak fitting. Default 0.1
gaussian_min_spacing = 5   # Min spacing IN DRIFT BINS between peaks.
gaussian_width_max = 4      # maximum width to be considered a peak (used for filtering noise peaks) default 1.7
centroid_bound_filter = None    # centroid bounds for filtering IN MS in the form [lower bound, upper bound]
centroid_plot_bounds = None     # plot y-axis bounds in ms for the centroid vs CV plot as [lower bound, upper bound]

# ********* END: PARAMETERS TO EDIT IN THE DEVELOPMENT/TESTING VERSION ***************


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
    plt.xlabel(x_title)
    plt.ylabel(y_title)
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
    raw_obj = Raw_Processing.get_data(raw_file)
    filename = raw_obj.filename

    # normalize, smooth, and crop data (if requested)
    norm_data = Raw_Processing.normalize_by_col(raw_obj.rawdata)

    # interpolate data
    axes = (raw_obj.dt_axis, raw_obj.cv_axis)
    if interpolate_bins > 0:
        norm_data, axes = Raw_Processing.interpolate_cv(norm_data, axes, interpolate_bins)

    if smooth_window is not None:
        i = 0
        while i < num_smooths:
            norm_data = Raw_Processing.sav_gol_smooth(norm_data, smooth_window)
            i += 1

    if crop_vals is not None:  # If no cropping, use the whole matrix
        norm_data, axes = Raw_Processing.crop(norm_data, axes, crop_vals)

    analysis_obj = CIUAnalysisObj(raw_obj, norm_data, axes)
    Gaussian_Fitting.gaussian_fit_ciu_old(analysis_obj,
                                          intensity_thr=gaussian_int_thr,
                                          min_spacing=gaussian_min_spacing,
                                          filter_width_max=gaussian_width_max,
                                          centroid_bounds=centroid_bound_filter)
    # save output
    title = filename.rstrip('_raw.csv')
    outputpath = os.path.join(os.path.dirname(analysis_obj.raw_obj.filepath), title)

    analysis_obj.params.set_params({'cropping': crop_vals,
                                    'gaussian_centroid_bound_filter': centroid_bound_filter,
                                    'gaussian_centroid_plot_bounds': centroid_plot_bounds,
                                    'gaussian_int_threshold': gaussian_int_thr,
                                    'gaussian_min_spacing': gaussian_min_spacing,
                                    'gaussian_width_max': gaussian_width_max,
                                    'interpolation': interpolate_bins,
                                    'output_save_csv': save_csv,
                                    'ciuplot_3_plot_title': title,
                                    'ciuplot_4_extension': extension,
                                    'ciuplot_1_x_title': titlex,
                                    'ciuplot_2_y_title': titley,
                                    'smoothing_3_iterations': num_smooths,
                                    'smoothing_1_method': 'Savitsky-Golay',
                                    'smoothing_2_window': smooth_window})

    analysis_obj.save_gaussfits_pdf(outputpath)
    analysis_obj.plot_centroids(outputpath, centroid_plot_bounds)
    analysis_obj.plot_fwhms(outputpath)
    analysis_obj.save_gauss_params(outputpath)
    print('Job completed')

    # save (pickle) the analysis object for later retrieval
    picklefile = os.path.join(os.path.dirname(analysis_obj.raw_obj.filepath), filename.rstrip('_raw.csv') + '.pkl')

    with open(picklefile, 'wb') as pkfile:
        pickle.dump(analysis_obj, pkfile)

    # features = Feature_Detection.feature_detect_gauss(analysis_obj, 0.02)
    # no_loners = Feature_Detection.remove_loner_features(features)
    # Feature_Detection.print_features_list(features, filename + '_features.csv')
    # Feature_Detection.print_features_list(no_loners, filename + '_no-loners.csv')

    if save_csv:
        if save_title is not None:
            save_name = os.path.join(os.path.dirname(raw_file), save_title + '_raw.csv')
        else:
            save_name = raw_file.rstrip('_raw.csv') + '_Processed_raw.csv'
        write_ciu_csv(save_name, norm_data, axes)

    # make the ciu plot
    if extension is None:
        extension = '.png'
    ciu_plot(norm_data, axes, output_dir, title, titlex, titley, extension)


if __name__ == '__main__':
    # browse for files to analyze
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


# testing 2
# if __name__ == '__main__':
#     # open pickled files
#     root = tkinter.Tk()
#     root.withdraw()
#     files = filedialog.askopenfilenames(filetypes=[('pickled gaussian files', '.pkl')])
#     files = list(files)
#     file_dir = os.path.dirname(files[0])
#
#     all_ciu_data = []
#     axes = []
#     for file in files:
#         with open(file, 'rb') as first_file:
#             ciu1 = pickle.load(first_file)
#
#         data = ciu1.ciu_data
#         all_ciu_data.append(data)
#         axes = ciu1.axes
#
#     # average
#     avg_data, std_devs = Raw_Processing.average_ciu(all_ciu_data)
#
#     # save output
#     avg_name = os.path.join(file_dir, files[0].rstrip('_raw.csv') + '_avg.csv')
#     std_name = os.path.join(file_dir, files[0].rstrip('_raw.csv') + '_std.csv')
#     write_ciu_csv(avg_name, avg_data, axes)
#     write_ciu_csv(std_name, std_devs, axes)
#     ciu_plot(avg_data, axes, file_dir, 'average', 'CV', 'DT', '.png')
#     ciu_plot(std_devs, axes, file_dir, 'stdev', 'CV', 'DT', '.png')
