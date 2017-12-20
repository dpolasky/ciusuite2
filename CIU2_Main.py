"""
Main entry point for CIUSuite 2. Designed to allow the user to choose files and perform
processing to generate analysis objects, and process analysis objects. Probably will need
a (very) basic GUI of some kind.
"""

# GUI test
import tkinter as tk
import pygubu
from tkinter import filedialog
import os
import Raw_Processing
import Gaussian_Fitting
from CIU_analysis_obj import CIUAnalysisObj
import pickle
import CIU_Params
import Original_CIU
import numpy as np
import Feature_Detection

hard_file_path_ui = r"C:\CIUSuite2\CIUSuite2.ui"
hard_params_file = r"C:\CIUSuite2\CIU_params.txt"
hard_output_default = r"C:\Users\dpolasky\Desktop\test"


class CIUSuite2(object):
    """

    """
    def __init__(self):
        """

        """
        # create a Pygubu builder
        self.builder = builder = pygubu.Builder()

        # load the UI file
        builder.add_from_file(hard_file_path_ui)
        # create widget using provided root (Tk) window
        # self.mainwindow = builder.get_object('CIU_app_top', master_window)
        self.mainwindow = builder.get_object('CIU_app_top')

        self.mainwindow.protocol('WM_DELETE_WINDOW', self.on_close_window)

        callbacks = {
            'on_button_rawfile_clicked': self.on_button_rawfile_clicked,
            'on_button_analysisfile_clicked': self.on_button_analysisfile_clicked,
            'on_button_paramload_clicked': self.on_button_paramload_clicked,
            'on_button_printparams_clicked': self.on_button_printparams_clicked,
            'on_button_changedir_clicked': self.on_button_changedir_clicked,
            'on_button_oldplot_clicked': self.on_button_oldplot_clicked,
            'on_button_oldcompare_clicked': self.on_button_oldcompare_clicked,
            'on_button_oldavg_clicked': self.on_button_oldavg_clicked,
            'on_button_olddeltadt_clicked': self.on_button_olddeltadt_clicked,
            'on_button_gaussfit_clicked': self.on_button_gaussfit_clicked,
            'on_button_feature_detect_clicked': self.on_button_feature_detect_clicked
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

    def run(self):
        self.mainwindow.mainloop()

    def on_close_window(self):
        self.mainwindow.destroy()

    def on_button_rawfile_clicked(self):
        """
        Open a filechooser for the user to select raw files, then process them
        :return:
        """
        # clear analysis list
        self.analysis_file_list = []

        raw_files = open_files([('_raw.csv', '_raw.csv')])
        # run raw processing
        for raw_file in raw_files:
            self.update_progress(raw_files.index(raw_file), len(raw_files))

            raw_obj = generate_raw_obj(raw_file)
            analysis_obj = process_raw_obj(raw_obj, self.params_obj)
            analysis_filename = save_analysis_obj(analysis_obj)
            self.analysis_file_list.append(analysis_filename)

        # update the list of analysis files to display
        self.display_analysis_files()

        if len(raw_files) > 0:
            # update directory to match the loaded files
            self.output_dir = os.path.dirname(self.analysis_file_list[0])
            self.update_dir_entry()
            self.progress_done()

    def on_button_analysisfile_clicked(self):
        """
        Open a filechooser for the user to select previously process analysis (.ciu) files
        :return:
        """
        analysis_files = open_files([('CIU files', '.ciu')])
        self.analysis_file_list = analysis_files
        self.display_analysis_files()

        # update directory to match the loaded files
        try:
            self.output_dir = os.path.dirname(self.analysis_file_list[0])
            self.update_dir_entry()
        except IndexError:
            # no files selected (user probably hit 'cancel' - ignore
            pass

    def on_button_paramload_clicked(self):
        """
        Open a user chosen parameter file into self.params
        :return: void
        """
        new_param_file = open_files([('params file', '.txt')])[0]
        new_param_obj = CIU_Params.Parameters()
        new_param_obj.set_params(CIU_Params.parse_params_file(new_param_file))
        self.params_obj = new_param_obj

        # update parameter location display
        new_text = 'Parameters loaded from {}'.format(os.path.basename(new_param_file))
        params_text = self.builder.get_object('Text_params')
        params_text.delete(1.0, tk.END)
        params_text.insert(tk.INSERT, new_text)

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
        Note: subtracts 1 from file numbers because users count from 1, not 0
        :return: sublist of self.analysis_file_list bounded by ranges
        """
        try:
            start_file = int(self.builder.get_object('Entry_start_files').get()) - 1
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
        # Determine if a file range has been specified
        files_to_read = self.check_file_range_entries()

        for analysis_file in files_to_read:
            self.update_progress(files_to_read.index(analysis_file), len(files_to_read))

            analysis_obj = load_analysis_obj(analysis_file)
            Original_CIU.ciu_plot(analysis_obj, self.params_obj, self.output_dir)
        self.progress_done()

    def on_button_oldcompare_clicked(self):
        """
        Run old (batched) RMSD-based CIU plot comparison on selected files
        :return: void (saves to output dir)
        """
        # Determine if a file range has been specified
        files_to_read = self.check_file_range_entries()

        if len(files_to_read) == 1:
            # re-open filechooser to get second file
            newfile = filedialog.askopenfilename(filetypes=[('_raw.csv', '_raw.csv')])
            self.analysis_file_list.append(newfile)
            files_to_read.append(newfile)

        if len(files_to_read) == 2:
            # compare_basic_raw(test_file1, test_file2, test_dir, test_smooth, test_crop)
            ciu1 = load_analysis_obj(files_to_read[0])
            ciu2 = load_analysis_obj(files_to_read[1])
            Original_CIU.compare_basic_raw(ciu1, ciu2, self.params_obj, self.output_dir)

        elif len(files_to_read) > 2:
            rmsd_print_list = ['File 1, File 2, RMSD (%)']
            # batch compare - compare all against all.
            for file in files_to_read:
                self.update_progress(files_to_read.index(file), len(files_to_read))

                # don't compare the file against itself
                skip_index = files_to_read.index(file)
                index = 0
                while index < len(files_to_read):
                    if not index == skip_index:
                        ciu1 = load_analysis_obj(file)
                        ciu2 = load_analysis_obj(files_to_read[index])
                        rmsd = Original_CIU.compare_basic_raw(ciu1, ciu2, self.params_obj, self.output_dir)
                        printstring = '{},{},{:.2f}'.format(os.path.basename(file).rstrip('.ciu'),
                                                            os.path.basename(files_to_read[index]).rstrip('.ciu'),
                                                            rmsd)
                        rmsd_print_list.append(printstring)
                    index += 1

            # print output to csv
            with open(os.path.join(self.output_dir, 'batch_RMSDs.csv'), 'w') as rmsd_file:
                for rmsd_string in rmsd_print_list:
                    rmsd_file.write(rmsd_string + '\n')
        self.progress_done()

    def on_button_oldavg_clicked(self):
        """
        Average several processed files into a replicate object and save it for further
        replicate processing methods
        :return:
        """
        # Determine if a file range has been specified
        files_to_read = self.check_file_range_entries()

        analysis_obj_list = [load_analysis_obj(x) for x in files_to_read]
        averaged_obj = average_ciu(analysis_obj_list, self.params_obj, self.output_dir)
        self.analysis_file_list = [averaged_obj.filename]
        self.display_analysis_files()
        self.progress_done()

    def on_button_olddeltadt_clicked(self):
        """
        Edit files to a delta DT axis. x-axis will be adjusted so that the maximum y-value
        in the first column of the 2D CIU matrix (or first centroid if gaussian fitting
        has been performed) will be set to an x-value of 0, and all other x-values will
        be adjusted accordingly.
        :return: saves new .ciu file
        """
        # Determine if a file range has been specified
        files_to_read = self.check_file_range_entries()

        new_file_list = []
        for file in files_to_read:
            self.update_progress(files_to_read.index(file), len(files_to_read))

            analysis_obj = load_analysis_obj(file)
            shifted_obj = Original_CIU.delta_dt(analysis_obj)
            newfile = save_analysis_obj(shifted_obj, filename_append='_delta', outputdir=self.output_dir)
            new_file_list.append(newfile)
            # also save _raw.csv output if desired
            if self.params_obj.save_output_csv:
                save_path = file.rstrip('.ciu') + '_delta_raw.csv'
                Original_CIU.write_ciu_csv(save_path, shifted_obj.ciu_data, shifted_obj.axes)
        self.analysis_file_list = new_file_list
        self.display_analysis_files()
        self.progress_done()

    def on_button_gaussfit_clicked(self):
        """
        Run Gaussian fitting on the analysis object list (updating the objects and leaving
        the current list in place). Saves Gaussian diagnostics/info to file in self.output_dir
        :return: void
        """
        # Determine if a file range has been specified
        files_to_read = self.check_file_range_entries()

        new_file_list = []
        for file in files_to_read:
            self.update_progress(files_to_read.index(file), len(files_to_read))

            analysis_obj = load_analysis_obj(file)
            analysis_obj = Gaussian_Fitting.gaussian_fit_ciu(analysis_obj, self.params_obj)

            filename = save_analysis_obj(analysis_obj, outputdir=self.output_dir)
            new_file_list.append(filename)
        self.display_analysis_files()
        self.progress_done()

    def on_button_feature_detect_clicked(self):
        """
        Run feature detection workflow to generate CIU-50 (transition) outputs for selected
        files
        :return: void
        """
        files_to_read = self.check_file_range_entries()
        new_file_list = []

        all_outputs = ''
        filename = ''
        combine_flag = False
        for file in files_to_read:
            # update progress and load file
            self.update_progress(files_to_read.index(file), len(files_to_read))
            analysis_obj = load_analysis_obj(file)

            # run feature detection
            analysis_obj = Feature_Detection.feature_detect_main(analysis_obj, outputdir=self.output_dir)
            filename = save_analysis_obj(analysis_obj, outputdir=self.output_dir)
            new_file_list.append(filename)

            if not analysis_obj.params.combine_output_file:
                analysis_obj.save_feature_outputs(self.output_dir)
                combine_flag = False
            else:
                file_string = os.path.basename(filename).rstrip('.ciu') + '\n'
                all_outputs += file_string
                all_outputs += analysis_obj.save_feature_outputs(self.output_dir, True)
                combine_flag = True

        if combine_flag:
            outputpath = os.path.join(self.output_dir, os.path.basename(filename.rstrip('.ciu')) + '_features.csv')
            save_existing_output_string(outputpath, all_outputs)

        self.display_analysis_files()
        self.progress_done()

    def update_progress(self, current_analysis, num_analyses):
        """
        Update the progress bar to display the current progress through the analysis list
        :param current_analysis: the file NUMBER currently being worked on by the program
        :param num_analyses: the total number of files in the current analysis
        :return: void
        """
        current_prog = (current_analysis + 1) / float(num_analyses) * 100
        prog_string = 'Processing {} of {}'.format(current_analysis + 1, num_analyses)

        progress_bar = self.builder.get_object('Progressbar_main')
        progress_bar['value'] = current_prog

        self.builder.get_object('Entry_progress').config(state=tk.NORMAL)
        self.builder.get_object('Entry_progress').delete(0, tk.END)
        self.builder.get_object('Entry_progress').insert(0, prog_string)
        self.builder.get_object('Entry_progress').config(state=tk.DISABLED)
        self.mainwindow.update()

    def progress_done(self):
        self.builder.get_object('Entry_progress').config(state=tk.NORMAL)
        self.builder.get_object('Entry_progress').delete(0, tk.END)
        self.builder.get_object('Entry_progress').insert(0, 'Done!')
        self.builder.get_object('Entry_progress').config(state=tk.DISABLED)

        # added to keep program from exiting when run from command line - not sure if there's a better fix
        self.run()


# ****** CIU Main I/O methods ******
def open_files(filetype):
    """
    Open a tkinter filedialog to choose files of the specified type
    :param filetype: filetype filter in form [(name, extension)]
    :return: list of selected files
    """
    files = filedialog.askopenfilenames(filetype=filetype)
    return files


def save_existing_output_string(full_output_path, string_to_save):
    """
    Write an existing (e.g. combined from several objects) string to file
    :param full_output_path: full path filename in which to save output
    :param string_to_save: string to save
    :return: void
    """
    with open(full_output_path, 'w') as outfile:
        outfile.write(string_to_save)


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
    # ciu_app = CIUSuite2(root)
    ciu_app = CIUSuite2()
    ciu_app.run()
    # root.mainloop()
