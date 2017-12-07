"""
Main entry point for CIUSuite 2. Designed to allow the user to choose files and perform
processing to generate analysis objects, and process analysis objects. Probably will need
a (very) basic GUI of some kind.
"""

# GUI test
import tkinter as tk
import pygubu
from tkinter import messagebox
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
import Raw_Processing
import Gaussian_Fitting
from CIU_analysis_obj import CIUAnalysisObj
import pickle
import CIU_Params


hard_file_path_ui = r"C:\Users\dpolasky\Desktop\CIUSuite2.ui"
hard_params_file = r"C:\Users\dpolasky\Desktop\CIU_params.txt"
hard_output_default = r"C:\Users\dpolasky\Desktop\test"


class CIUSuite2(object):
    """

    """
    def __init__(self, master_window):
        """

        :param master_window:
        """
        # create a Pygubu builder
        self.builder = builder = pygubu.Builder()

        # load the UI file
        builder.add_from_file(hard_file_path_ui)
        # create widget using provided root (Tk) window
        self.mainwindow = builder.get_object('CIU_app_top', master_window)

        callbacks = {
            'on_button_rawfile_clicked': self.on_button_rawfile_clicked,
            'on_button_analysisfile_clicked': self.on_button_analysisfile_clicked,
            'on_button_printparams_clicked': self.on_button_printparams_clicked,
            'on_button_changedir_clicked': self.on_button_changedir_clicked,
            'on_button_oldplot_clicked': self.on_button_oldplot_clicked,
            # 'on_button_oldcompare_clicked': self.on_button_oldcompare_clicked,
            # 'on_button_oldavg_clicked': self.on_button_oldavg_clicked,
            # 'on_button_olddeltadt_clicked': self.on_button_olddeltadt_clicked
        }
        builder.connect_callbacks(callbacks)

        # load parameter file
        self.params_obj = CIU_Params.Parameters()
        self.params_obj.set_params(CIU_Params.parse_params_file(hard_params_file))

        params_text = self.builder.get_object('Text_params')
        params_text.delete(1.0, tk.END)
        params_text.insert(tk.INSERT, 'Parameters loaded from hard file')

        self.analysis_file_list = []
        self.output_dir = hard_output_default

    def on_button_rawfile_clicked(self):
        """
        Open a filechooser for the user to select raw files, then process them
        :return:
        """
        raw_files = open_files([('_raw.csv', '_raw.csv')])

        # run raw processing
        for raw_file in raw_files:
            raw_obj = generate_raw_obj(raw_file)
            analysis_obj = process_raw_obj(raw_obj, self.params_obj)
            analysis_filename = save_analysis_obj(analysis_obj)
            self.analysis_file_list.append(analysis_filename)

        # update the list of analysis files to display
        self.display_analysis_files()

    def on_button_analysisfile_clicked(self):
        """
        Open a filechooser for the user to select previously process analysis (.ciu) files
        :return:
        """
        analysis_files = open_files([('CIU files', '.ciu')])
        self.analysis_file_list = analysis_files
        self.display_analysis_files()

    def display_analysis_files(self):
        """
        Write analysis filenames to the main text window and update associated controls. References
        self.analysis_list for filenames
        :return: void
        """
        displaystring = ''
        index = 1
        for file in self.analysis_file_list:
            displaystring += '{}: {}\n'.format(index, os.path.basename(file).rstrip('.ciu'))
            index += 1

        # clear any existing text, then write the list of files to the display
        self.builder.get_object('Text_analysis_list').delete(1.0, tk.END)
        self.builder.get_object('Text_analysis_list').insert(tk.INSERT, displaystring)

        # update total file number counter
        # self.builder.get_object('Entry_num_files').textvariable.set(str(len(self.analysis_file_list)))
        self.builder.get_object('Entry_num_files').config(state=tk.NORMAL)
        self.builder.get_object('Entry_num_files').delete(0, tk.END)
        self.builder.get_object('Entry_num_files').insert(0, str(len(self.analysis_file_list)))
        self.builder.get_object('Entry_num_files').config(state=tk.DISABLED)

    def on_button_printparams_clicked(self):
        """
        print parameters from selected files (or all files) to console
        :return: void
        """
        # Determine if a file range has been specified
        files_to_read = self.check_file_range_entries()

        for file in files_to_read:
            # load analysis obj and print params
            analysis_obj = load_analysis_obj(file)
            print('\nParameters used in file {}:'.format(os.path.basename(file)))
            analysis_obj.params.print_params_to_console()

    def check_file_range_entries(self):
        """
        Check to see if user has entered a valid range in either of the file range entries
        and return the range if so, or return the default range (all files) if not.
        :return: sublist of self.analysis_file_list bounded by ranges
        """
        try:
            start_file = int(self.builder.get_object('Entry_start_files').get())
        except ValueError:
            # no number was entered or an invalid number - use default start location (0)
            start_file = 0
        try:
            end_file = int(self.builder.get_object('Entry_end_files').get())
        except ValueError:
            end_file = len(self.analysis_file_list)

        files_to_read = self.analysis_file_list[start_file: end_file]
        return files_to_read

    def on_button_changedir_clicked(self):
        """
        Open a file chooser to change the output directory and update the display
        :return: void
        """
        newdir = filedialog.askdirectory()
        self.output_dir = newdir
        self.update_dir_entry()

    def update_dir_entry(self):
        """
        Update the graphical display of the output directory
        :return: void
        """
        self.builder.get_object('Text_outputdir').config(state=tk.NORMAL)
        self.builder.get_object('Text_outputdir').delete(1.0, tk.INSERT)
        self.builder.get_object('Text_outputdir').insert(tk.INSERT, self.output_dir)
        self.builder.get_object('Text_outputdir').config(state=tk.DISABLED)

    def on_button_oldplot_clicked(self):
        """
        Run old CIU plot method to generate a plot in the output directory
        :return:
        """


# ****** CIU Main I/O methods ******
def open_files(filetype):
    """
    Open a tkinter filedialog to choose files of the specified type
    :param filetype: filetype filter in form [(name, extension)]
    :return: list of selected files
    """
    files = filedialog.askopenfilenames(filetype=filetype)
    return files


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


def generate_raw_obj(raw_file):
    """
    Open an _raw.csv file and read its data into a CIURaw object to return
    :param raw_file: (string) filename of the _raw.csv file to read
    :return: CIURaw object with raw data, filename, and axes
    """
    raw_obj = Raw_Processing.get_data(raw_file)
    return raw_obj


def process_raw_obj(raw_obj, params_obj):
    """
    Run all initial raw processing stages (data import, smoothing, interpolation, cropping)
    on a raw file using the parameters provided in a Parameters object. Returns a new
    analysis object with the processed data
    :param raw_obj: the CIURaw object containing the raw data to process
    :param params_obj: Parameters object containing processing parameters
    :return: CIUAnalysisObj with processed data
    """
    # normalize, smooth, and crop data (if requested)
    norm_data = Raw_Processing.normalize_by_col(raw_obj.rawdata)

    # interpolate data
    axes = (raw_obj.dt_axis, raw_obj.cv_axis)
    if params_obj.interpolation_bins is not None:
        norm_data, axes = Raw_Processing.interpolate_cv(norm_data, axes, params_obj.interpolation_bins)

    if params_obj.smoothing_window is not None:
        i = 0
        while i < params_obj.smoothing_iterations:
            norm_data = Raw_Processing.sav_gol_smooth(norm_data, params_obj.smoothing_window)
            i += 1

    if params_obj.cropping_window_values is not None:  # If no cropping, use the whole matrix
        norm_data, axes = Raw_Processing.crop(norm_data, axes, params_obj.cropping_window_values)

    analysis_obj = CIUAnalysisObj(raw_obj, norm_data, axes)
    analysis_obj.params = params_obj

    return analysis_obj


def run_gaussian_fitting(analysis_obj):
    """
    Perform gaussian fitting on an analysis object. NOTE: object must have initial raw processing
    already performed and a parameters object instantiated. Updates the analysis object.
    :param analysis_obj: CIUAnalysisObj with normalized data and parameters obj already present
    :return: void (updates the analysis_obj)
    """
    params = analysis_obj.params
    Gaussian_Fitting.gaussian_fit_ciu(analysis_obj,
                                      intensity_thr=params.gaussian_int_threshold,
                                      min_spacing=params.gaussian_min_spacing,
                                      filter_width_max=params.gaussian_width_max,
                                      centroid_bounds=params.gaussian_centroid_bound_filter)


def save_gaussian_outputs(analysis_obj, outputpath):
    """
    Write Gaussian output data and diagnostics to file location specified by outputpath
    :param analysis_obj: CIUAnalysisObj with gaussian fitting previously performed
    :param outputpath: directory in which to save output
    :return: void
    """
    analysis_obj.save_gaussfits_pdf(outputpath)
    analysis_obj.plot_centroids(outputpath, analysis_obj.params.gaussian_centroid_plot_bounds)
    analysis_obj.plot_fwhms(outputpath)
    analysis_obj.save_gauss_params(outputpath)


def save_analysis_obj(analysis_obj, outputdir=None):
    """
    Pickle the CIUAnalysisObj for later retrieval
    :param analysis_obj: CIUAnalysisObj to save
    :param outputdir: (optional) directory in which to save. Default = raw file directory
    :return: full path to save location
    """
    file_extension = '.ciu'

    if outputdir is not None:
        picklefile = os.path.join(outputdir, analysis_obj.raw_obj.filename.rstrip('_raw.csv') + file_extension)
    else:
        picklefile = os.path.join(os.path.dirname(analysis_obj.raw_obj.filepath),
                                  analysis_obj.raw_obj.filename.rstrip('_raw.csv') + file_extension)

    with open(picklefile, 'wb') as pkfile:
        pickle.dump(analysis_obj, pkfile)

    return picklefile


def load_analysis_obj(analysis_filename):
    """
    Load a pickled analysis object back into program memory
    :param analysis_filename: full path to file location to load
    :return: CIUAnalysisObj
    """
    with open(analysis_filename, 'rb') as analysis_file:
        analysis_obj = pickle.load(analysis_file)
    return analysis_obj


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    ciu_app = CIUSuite2(root)
    root.mainloop()
