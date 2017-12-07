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
import Original_CIU
import numpy as np


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
            'on_button_oldcompare_clicked': self.on_button_oldcompare_clicked,
            'on_button_oldavg_clicked': self.on_button_oldavg_clicked,
            'on_button_olddeltadt_clicked': self.on_button_olddeltadt_clicked
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

        # update directory to match the loaded files
        self.output_dir = os.path.dirname(self.analysis_file_list[0])
        self.update_dir_entry()

    def on_button_analysisfile_clicked(self):
        """
        Open a filechooser for the user to select previously process analysis (.ciu) files
        :return:
        """
        analysis_files = open_files([('CIU files', '.ciu')])
        self.analysis_file_list = analysis_files
        self.display_analysis_files()

        # update directory to match the loaded files
        self.output_dir = os.path.dirname(self.analysis_file_list[0])
        self.update_dir_entry()

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
        :return: void (saves to output dir)
        """
        for analysis_file in self.analysis_file_list:
            analysis_obj = load_analysis_obj(analysis_file)
            Original_CIU.ciu_plot(analysis_obj, self.params_obj, self.output_dir)

    def on_button_oldcompare_clicked(self):
        """
        Run old (batched) RMSD-based CIU plot comparison on selected files
        :return: void (saves to output dir)
        """
        if len(self.analysis_file_list) == 1:
            # re-open filechooser to get second file
            self.analysis_file_list.append(filedialog.askopenfilename(filetypes=[('_raw.csv', '_raw.csv')]))

        if len(self.analysis_file_list) == 2:
            # compare_basic_raw(test_file1, test_file2, test_dir, test_smooth, test_crop)
            Original_CIU.compare_basic_raw(self.analysis_file_list[0], self.analysis_file_list[1],
                                           self.params_obj, self.output_dir)

        elif len(self.analysis_file_list) > 2:
            rmsd_print_list = ['File 1, File 2, RMSD (%)']
            # batch compare - compare all against all.
            for file in self.analysis_file_list:
                # don't compare the file against itself
                skip_index = self.analysis_file_list.index(file)
                index = 0
                while index < len(self.analysis_file_list):
                    if not index == skip_index:
                        ciu1 = load_analysis_obj(file)
                        ciu2 = load_analysis_obj(self.analysis_file_list[index])
                        rmsd = Original_CIU.compare_basic_raw(ciu1, ciu2, self.params_obj, self.output_dir)
                        rmsd_print_list.append('{},{},{:.2f}'.format(os.path.basename(file).rstrip('.ciu'),
                                                                     os.path.basename(self.analysis_file_list[index]).rstrip('.ciu'),
                                                                     rmsd))
                    index += 1

            # print output to csv
            with open(os.path.join(self.output_dir, 'batch_RMSDs.csv'), 'w') as rmsd_file:
                for rmsd_string in rmsd_print_list:
                    rmsd_file.write(rmsd_string + '\n')

    def on_button_oldavg_clicked(self):
        """
        Average several processed files into a replicate object and save it for further
        replicate processing methods
        :return:
        """
        analysis_obj_list = [load_analysis_obj(x) for x in self.analysis_file_list]
        averaged_obj = average_ciu(analysis_obj_list, self.params_obj, self.output_dir)
        self.analysis_file_list = [averaged_obj.filename]
        self.display_analysis_files()

    def on_button_olddeltadt_clicked(self):
        """
        Edit files to align initial CV's with each other for difference analyses. Designed
        (originally) to output updated _raw.csv for processing, but might change.
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


def average_ciu(analysis_obj_list, params_obj, output_dir):
    """
    Generate and save replicate object (a CIUAnalysisObj with averaged ciu_data and a list
    of raw_objs) that can be used for further analysis
    :param analysis_obj_list: list of CIUAnalysisObj's to average
    :param params_obj: Parameters object with options
    :param output_dir: directory in which to save output
    :return: averaged analysis object
    """
    raw_obj_list = []
    ciu_data_list = []
    for analysis_obj in analysis_obj_list:
        raw_obj_list.append(analysis_obj.raw_obj)
        ciu_data_list.append(analysis_obj.ciu_data)

    # generate the average object
    avg_data = np.mean(ciu_data_list, axis=0)
    averaged_obj = CIUAnalysisObj(raw_obj_list[0], avg_data, analysis_obj_list[0].axes)
    averaged_obj.params = analysis_obj_list[0].params
    averaged_obj.raw_obj_list = raw_obj_list
    averaged_obj.filename = save_analysis_obj(averaged_obj, filename_append='_Avg')

    # save averaged object to file and return it
    return averaged_obj


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


def save_analysis_obj(analysis_obj, filename_append='', outputdir=None):
    """
    Pickle the CIUAnalysisObj for later retrieval
    :param analysis_obj: CIUAnalysisObj to save
    :param filename_append: Addtional filename to append to the raw_obj name (e.g. 'AVG')
    :param outputdir: (optional) directory in which to save. Default = raw file directory
    :return: full path to save location
    """
    file_extension = '.ciu'

    if outputdir is not None:
        picklefile = os.path.join(outputdir, analysis_obj.raw_obj.filename.rstrip('_raw.csv')
                                  + filename_append + file_extension)
    else:
        picklefile = os.path.join(os.path.dirname(analysis_obj.raw_obj.filepath),
                                  analysis_obj.raw_obj.filename.rstrip('_raw.csv') + filename_append + file_extension)

    analysis_obj.filename = picklefile
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
