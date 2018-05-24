"""
Module with updated versions of the original CIUSuite modules to preserve (and improve)
functionality of the original CIUSuite. Designed to operate in the framework of CIUSuite2,
with CIUAnalysisObj objects providing the primary basis for handling data.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.interpolate
from CIU_analysis_obj import CIUAnalysisObj
from CIU_Params import Parameters

rmsd_plot_scaling = 1


def ciu_plot(analysis_obj, params_obj, output_dir):
    """
    Generate a CIU plot in the provided directory
    :param analysis_obj: preprocessed CIUAnalysisObj with data to be plotted
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param output_dir: directory in which to save the plot
    :return: void
    """
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # save filename as plot title, unless a specific title is provided
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = analysis_obj.short_filename
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    else:
        plot_title = ''

    # use filename and provided extension as output path with specific figure size and DPI
    output_title = analysis_obj.short_filename
    output_path = os.path.join(output_dir, output_title + params_obj.plot_02_extension)

    # generate the contour plot with options specified by use
    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, 100, cmap=params_obj.ciuplot_cmap_override)

    if params_obj.plot_06_show_colorbar:
        cbar = plt.colorbar(ticks=[0, .25, .5, .75, 1])
        cbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)

    plt.savefig(output_path)
    plt.close()

    # todo: fix this (in or out) and deal with method return value
    # save csv if desired
    if params_obj.output_1_save_csv:
        save_path = os.path.join(output_dir, plot_title)
        save_path += '_raw.csv'
        write_ciu_csv(save_path, analysis_obj.ciu_data, analysis_obj.axes)

    return 'returning a value so that the mainloop doesnt stop'


def rmsd_difference(data_1, data_2):
    """
    Compute overall RMSD for the comparison of two matrices
    :param data_1: matrix 1 (numpy.ndarray)
    :param data_2: matrix 2 (numpy.ndarray) - MUST be same shape as matrix 1
    :return: difference matrix (ndarray), rmsd (float) in percent
    """
    # data_1[data_1 < 0.1] = 0.1
    # num_entries_1 = np.count_nonzero(data_1)
    # data_2[data_2 < 0.1] = 0.1
    # num_entries_2 = np.count_nonzero(data_2)

    data_1[data_1 < 0.1] = 0
    num_entries_1 = np.count_nonzero(data_1)
    data_2[data_2 < 0.1] = 0
    num_entries_2 = np.count_nonzero(data_2)
    dif = data_1 - data_2
    rmsd = ((np.sum(dif ** 2) / (num_entries_1 + num_entries_2)) ** 0.5) * 100
    return dif, rmsd


def rmsd_plot(difference_matrix, axes, contour_scale, tick_scale, rtext, outputdir, params_obj,
              file1, file2, blue_label=None, red_label=None):
    """
    Make a CIUSuite comparison RMSD plot with provided parameters
    :param difference_matrix: 2D ndarray with differences to plot
    :param axes: [DT axis, CV axis] - axes labels to use for plot
    :param contour_scale: Scaling axis for the contour plot, (default = -1 to 1 in 100 increments)
    :param tick_scale: Scaling axis for ticks on contour plot scalebar (default = -1 to 1 in 6 increments)
    :param rtext: RMSD label to apply to plot
    :param outputdir: directory in which to save plot
    :param file1: filename of first file
    :param file2: filename of second file
    :param blue_label: (optional) custom colorbar label for the second file
    :param red_label: (optional) custom colorbar label for the first file
    :param params_obj: Parameters container with plotting information
    :type params_obj: Parameters
    :return: void
    """
    # initial plot setup
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # save filename as plot title, unless a specific title is provided
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = 'Red: {} \nBlue: {}'.format(file1, file2)
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')

    # make the RMSD contour plot
    plt.contourf(axes[1], axes[0], difference_matrix, contour_scale, cmap="bwr", ticks="none")
    plt.tick_params(axis='x', which='both', bottom='off', top='off', left='off', right='off')
    plt.tick_params(axis='y', which='both', bottom='off', top='off', left='off', right='off')

    # plot labels and legends
    if params_obj.plot_07_show_legend:
        plt.annotate(rtext, xy=(200, 10), xycoords='axes points', fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_06_show_colorbar:
        colorbar = plt.colorbar(ticks=tick_scale)
        if blue_label is not None and red_label is not None:
            colorbar.ax.set_yticklabels([red_label, 'Equal', blue_label])
        colorbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)

    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)

    # save and close
    plt.savefig(os.path.join(outputdir, '{}-{}{}'.format(file1, file2, params_obj.plot_02_extension)))
    plt.close()


def std_dev_plot(analysis_obj, std_dev_matrix, params_obj, output_dir):
    """
    Plot the original CIUSuite-style standard deviation plot from averaged fingerprints
    :param analysis_obj: Averaged analysis object (for axes and filename)
    :type analysis_obj: CIUAnalysisObj
    :param std_dev_matrix: standard deviation data in same shape as analysis_obj.ciu_data
    :param params_obj: Parameters information
    :type params_obj: Parameters
    :param output_dir: directory in which to save output
    :return: void
    """
    # initial plot setup
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # save filename as plot title, unless a specific title is provided
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = analysis_obj.short_filename
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')

    # plot standard deviation contour plot, normalized to the maximum std dev observed
    max_std_dev = np.max(std_dev_matrix)
    contour_scale = np.linspace(0, max_std_dev, 100, endpoint=True)
    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], std_dev_matrix, contour_scale, cmap=params_obj.plot_01_cmap)

    # plot desired labels and legends
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_06_show_colorbar:
        colorbar_scale = np.linspace(0, max_std_dev, 6, endpoint=True)   # plot colorbar
        cbar = plt.colorbar(ticks=colorbar_scale, format='%.2f')
        cbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        std_text = 'Max std deviation: {:.2f}'.format(max_std_dev)
        plt.annotate(std_text, xy=(150, 10), xycoords='axes points', fontsize=params_obj.plot_13_font_size)

    # save and close
    plt.savefig(os.path.join(output_dir, analysis_obj.short_filename + '_stdev.png'))
    plt.close()


# todo: deprecate (?)
def compare_by_cv(norm_data_1, norm_data_2, axes, smooth_window=None, crop_vals=None):
    """
    Generate an RMSD comparison at EACH collision energy and plot RMSD vs CV.
    NOTE: files must have same number of collision energies (columns) (or be cropped to be so)
    :param norm_data_1: preprocessed data to be subtracted from (2D ndarray, DT=axis 0, CV=axis 1)
    :param norm_data_2: preprocessed data to be subtracted
    :param axes: axis labels for [DT axis, CV axis]. Can be from either file (must be same for both to compare)
    # :param outputdir: directory in which to save output
    :param smooth_window: (optional) smoothing to apply PRIOR to analysis
    :param crop_vals: (optional) cropping to apply PRIOR to analysis
    :return: Dictionary of {CV : RMSD}
    """
    # swap axes for column by column comparison
    swapped_1 = norm_data_1.swapaxes(0, 1)
    swapped_2 = norm_data_2.swapaxes(0, 1)

    # compare by column
    index = 0
    rmsd_dict = {}
    while index < len(swapped_1):
        cv = axes[0][index]
        dif, rmsd = rmsd_difference(swapped_1[index], swapped_2[index])
        rmsd_dict[cv] = rmsd
        index += 1
    return rmsd_dict


def interpolate_axes(axis1, axis2, num_bins):
    """
    Method to interpolate different length axes to the same length/scale to enable subtractive
    (or other) comparison of data. The interpolated axis will have twice the number of bins as the
    larger of the initial axes and will go from the minimum to maximum value of both axes to ensure
    no data is missed.
    :param axis1: numpy array (1D) of values
    :param axis2: numpy array (1D) of values
    :param num_bins: number of bins onto which to interpolate
    :return: interpolated axis (numpy array)
    """
    # Determine axis sizes and ranges
    min_val = np.min([np.min(axis1), np.min(axis2)])  # minimum value between both axes
    max_val = np.max([np.max(axis1), np.max(axis2)])  # max value between both axes
    return np.linspace(min_val, max_val, num_bins)


def compare_basic_raw(analysis_obj1, analysis_obj2, params_obj, outputdir):
    """
    Basic CIU comparison between two raw files. Prints compare plot and csv to output dir.
    Adapted to use CIUAnalysis objects as inputs, so now assumes data will be preprocessed prior
    to running this method.
    :param analysis_obj1: preprocessed (and loaded) CIUAnalysisObj with data to be used as file 1
    :type analysis_obj1: CIUAnalysisObj
    :param analysis_obj2: preprocessed CIUAnalysisObj with data to be used as file 2
    :type analysis_obj2: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :return: RMSD value (writes plot to output directory)
    """
    norm_data_1 = analysis_obj1.ciu_data
    norm_data_2 = analysis_obj2.ciu_data
    axes = analysis_obj1.axes

    interp_flag = False
    dt_axis = analysis_obj1.axes[0]
    cv_axis = analysis_obj1.axes[1]
    # ensure that the data are the same in both dimensions, and interpolate if not matched in either
    if not len(analysis_obj1.axes[0]) == len(analysis_obj2.axes[0]) or not len(analysis_obj1.axes[1]) == len(analysis_obj2.axes[1]):
        interp_flag = True
        print('axes in files {}, {} did not match; interpolating to compare'.format(
            os.path.basename(analysis_obj1.filename),
            os.path.basename(analysis_obj2.filename)))
        num_bins_dt = np.max([len(analysis_obj1.axes[0]), len(analysis_obj2.axes[0])])  # length of the DT axis
        dt_axis = interpolate_axes(analysis_obj1.axes[0], analysis_obj2.axes[0], num_bins_dt)
        num_bins_cv = np.max([len(analysis_obj1.axes[1]), len(analysis_obj2.axes[1])])
        cv_axis = interpolate_axes(analysis_obj1.axes[1], analysis_obj2.axes[1], num_bins_cv)

    if interp_flag:
        # interpolate the original CIU data from each object onto the new (matched) axes
        interp_fn1 = scipy.interpolate.interp2d(analysis_obj1.axes[1],
                                                analysis_obj1.axes[0],
                                                analysis_obj1.ciu_data)
        interp_fn2 = scipy.interpolate.interp2d(analysis_obj2.axes[1],
                                                analysis_obj2.axes[0],
                                                analysis_obj2.ciu_data)
        norm_data_1 = interp_fn1(cv_axis, dt_axis)
        norm_data_2 = interp_fn2(cv_axis, dt_axis)
        axes = [dt_axis, cv_axis]

    dif, rmsd = rmsd_difference(norm_data_1, norm_data_2)

    rtext = "RMSD = " + '%2.2f' % rmsd
    title = '{} - {}'.format(analysis_obj1.short_filename,
                             analysis_obj2.short_filename)

    contour_scaling = np.linspace(-rmsd_plot_scaling, rmsd_plot_scaling, 50, endpoint=True)
    colorbar_scaling = np.linspace(-rmsd_plot_scaling, rmsd_plot_scaling, 3, endpoint=True)
    rmsd_plot(difference_matrix=dif,
              axes=axes,
              file1=analysis_obj1.short_filename,
              file2=analysis_obj2.short_filename,
              contour_scale=contour_scaling,
              tick_scale=colorbar_scaling,
              rtext=rtext,
              outputdir=outputdir,
              params_obj=params_obj,
              blue_label=params_obj.compare_2_custom_blue,
              red_label=params_obj.compare_1_custom_red)

    if params_obj.output_1_save_csv:
        save_path = os.path.join(outputdir, title)
        save_path += '_raw.csv'
        write_ciu_csv(save_path, dif, axes)

    # if return_dif:
    #     return RMSD, dif
    return rmsd


def delta_dt(analysis_obj):
    """
    Converts a CIU dataset to delta-drift time by moving the x-axis (DT) so that the max value
    (or centroid if gaussian fitting has been performed) of the first CV column is at 0.
    :param analysis_obj: CIUAnalysisObj to be shifted
    :type analysis_obj: CIUAnalysisObj
    :rtype: CIUAnalysisObj
    :return: new CIUAnalysisObj with shifted data
    """
    # Determine location of max in 1st CV column
    # if analysis_obj.gauss_params is not None:
    #     # gaussian fitting has been done, use first centroid in first CV column as center
    #     filt_centroids = [x[2::4] for x in analysis_obj.gauss_filt_params]
    #     centroid_xval = filt_centroids[0]
    # else:
    # gaussian fitting not done, simply use max of first column
    first_col = analysis_obj.ciu_data[:, 0]
    index_of_max = np.argmax(first_col)
    centroid_xval = analysis_obj.axes[0][index_of_max]

    # Shift the DT axis (ONLY) so that the max value of the column is at DT = 0
    old_dt_axis = analysis_obj.axes[0]
    new_dt_axis = old_dt_axis - centroid_xval
    new_axes = [new_dt_axis, analysis_obj.axes[1]]

    # create new CIUAnalysisObj with the new axis and return it
    # shift_analysis_obj = CIUAnalysisObj(analysis_obj.raw_obj, analysis_obj.ciu_data, new_axes,
    #                                     analysis_obj.gauss_params)
    # shift_analysis_obj.params = analysis_obj.params
    # shift_analysis_obj.raw_obj_list = analysis_obj.raw_obj_list
    analysis_obj.axes = new_axes
    return analysis_obj


def write_ciu_csv(save_path, ciu_data, axes=None):
    """
    Method to write an _raw.csv file for CIU data. If 'axes' is provided, assumes that the ciu_data
    array does NOT contain axes and if 'axes' is None, assumes ciu_data contains axes.
    :param save_path: Full path to save location (SHOULD end in _raw.csv)
    :param ciu_data: 2D numpy array containing CIU data in standard format (rows = DT bins, cols = CV)
    :param axes: (optional) axes labels, provided as (row axis, col axis). if provided,
    assumes the data array does not contain axes labels.
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
