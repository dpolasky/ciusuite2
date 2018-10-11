"""
Main entry point for CIUSuite 2 and location of all GUI handling code, along with some
basic file operations.


CIUSuite 2 Copyright (C) 2018 Daniel Polasky and Sugyan Dixit

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import pygubu
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
import os
import subprocess
import pickle
import sys
import multiprocessing

from CIU_analysis_obj import CIUAnalysisObj
import CIU_Params
import Raw_Processing
import Original_CIU
import Gaussian_Fitting
import Feature_Detection
import Classification
import Raw_Data_Import
import SimpleToolTip

# Set global matplotlib params for good figure layouts and a non-interactive backend
import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.use('Agg')


# Load resource file paths, supporting both live code and code bundled by PyInstaller
if getattr(sys, 'frozen', False):
    root_dir = sys._MEIPASS
    program_data_dir = os.path.join(os.environ['ALLUSERSPROFILE'], 'CIUSuite2')
else:
    root_dir = os.path.dirname(__file__)
    program_data_dir = root_dir

hard_params_file = os.path.join(program_data_dir, 'CIU2_param_info.csv')
hard_twimextract_path = os.path.join(program_data_dir, 'TWIMExtract', 'jars', 'TWIMExtract.jar')
hard_file_path_ui = os.path.join(root_dir, 'UI', 'CIUSuite2.ui')
hard_crop_ui = os.path.join(root_dir, 'UI', 'Crop_vals.ui')
hard_agilent_ext_path = os.path.join(root_dir, os.path.join('Agilent_Extractor', 'MIDAC_CIU_Extractor.exe'))
hard_tooltips_file = os.path.join(root_dir, 'tooltips.txt')
help_file = os.path.join(root_dir, 'CIUSuite2_Manual.pdf')
about_file = os.path.join(root_dir, 'README.txt')


class CIUSuite2(object):
    """
    Primary graphical class for running CIU2 user interface. Uses PyGuBu builder to create
    interface and handles associated events.
    """
    def __init__(self, tk_root_window):
        """
        Create a new GUI window, connect feedback to buttons, and load parameters
        """
        self.tk_root = tk_root_window

        # create a Pygubu builder
        self.builder = builder = pygubu.Builder()

        # load the UI file
        builder.add_from_file(hard_file_path_ui)

        # create widget using provided root (Tk) window
        self.mainwindow = builder.get_object('CIU_app_top')
        self.mainwindow.protocol('WM_DELETE_WINDOW', self.on_close_window)

        callbacks = {
            'on_button_help_clicked': self.on_button_help_clicked,
            'on_button_about_clicked': self.on_button_about_clicked,
            'on_button_rawfile_clicked': self.on_button_rawfile_clicked,
            'on_button_analysisfile_clicked': self.on_button_analysisfile_clicked,
            'on_button_vendor_raw_clicked': self.on_button_vendor_raw_clicked,
            'on_button_printparams_clicked': self.on_button_printparams_clicked,
            'on_button_update_param_defaults_clicked': self.on_button_update_param_defaults_clicked,
            'on_button_changedir_clicked': self.on_button_changedir_clicked,
            'on_button_plot_options_clicked': self.on_button_plot_options_clicked,
            'on_button_oldplot_clicked': self.on_button_oldplot_clicked,
            'on_button_oldcompare_clicked': self.on_button_oldcompare_clicked,
            'on_button_oldavg_clicked': self.on_button_oldavg_clicked,
            'on_button_crop_clicked': self.on_button_crop_clicked,
            'on_button_interpolate_clicked': self.on_button_interpolate_clicked,
            'on_button_restore_clicked': self.on_button_restore_clicked,
            'on_button_gaussfit_clicked': self.on_button_gaussfit_clicked,
            'on_button_feature_detect_clicked': self.on_button_feature_detect_clicked,
            'on_button_ciu50_clicked': self.on_button_ciu50_clicked,
            'on_button_gaussian_reconstruction_clicked': self.on_button_gaussian_reconstruct_clicked,
            'on_button_classification_supervised_clicked': self.on_button_classification_supervised_clicked,
            'on_button_classify_unknown_clicked': self.on_button_classify_unknown_clicked,
            'on_button_smoothing_clicked': self.on_button_smoothing_clicked
        }
        builder.connect_callbacks(callbacks)
        self.initialize_tooltips()

        # load parameter file
        self.params_obj = CIU_Params.Parameters()
        self.params_obj.set_params(CIU_Params.parse_params_file(hard_params_file))
        self.param_file = hard_params_file

        # holder for feature information in between user assessment - plan to replace with better solution eventually
        self.temp_feature_holder = None

        self.analysis_file_list = []
        self.output_dir = root_dir
        self.output_dir_override = False

    def run(self):
        """
        Run the main GUI loop
        :return: void
        """
        self.mainwindow.mainloop()

    def on_close_window(self):
        """
        Close (destroy) the app window and the Tkinter root window to stop the process.
        :return: void
        """
        self.mainwindow.destroy()
        self.tk_root.destroy()

    def initialize_tooltips(self):
        """
        Register tooltips for all buttons/widgets that need tooltips
        :return: void
        """
        tooltip_dict = parse_tooltips_file(hard_tooltips_file)
        for tip_key, tip_value in tooltip_dict.items():
            SimpleToolTip.create(self.builder.get_object(tip_key), tip_value)

    def on_button_help_clicked(self):
        """
        Open the manual
        :return: void
        """
        subprocess.Popen(help_file, shell=True)

    def on_button_about_clicked(self):
        """
        Open the license/about information file
        :return: void
        """
        subprocess.Popen(about_file, shell=True)

    def on_button_rawfile_clicked(self):
        """
        Open a filechooser for the user to select raw files, then process them
        :return:
        """
        # clear analysis list
        self.analysis_file_list = []
        raw_files = self.open_files(filetype=[('_raw.csv', '_raw.csv')])
        self.load_raw_files(raw_files)

    def load_raw_files(self, raw_filepaths):
        """
        Helper method for loading _raw.csv files. Created since the code is accessed by multiple
        other methods - used for modularity
        :param raw_filepaths: list of full system path strings to _raw.csv files to load
        :return: void
        """
        if len(raw_filepaths) > 0:
            # Ask user for smoothing input
            plot_keys = [x for x in self.params_obj.params_dict.keys() if 'smoothing' in x]
            param_success, param_dict = self.run_param_ui('Initial Smoothing Parameters', plot_keys)
            if param_success:
                self.progress_started()

                # run raw processing
                analysis_filenames = []
                for raw_file in raw_filepaths:
                    try:
                        raw_obj = Raw_Processing.get_data(raw_file)
                    except ValueError as err:
                        messagebox.showerror('Data Import Error', message='{}{}. Problem: {}. Press OK to continue'.format(*err.args))
                        continue

                    try:
                        analysis_obj = Raw_Processing.process_raw_obj(raw_obj, self.params_obj)
                    except ValueError:
                        # all 0's in raw data - skip this file an notify the user
                        messagebox.showerror('Empty raw data file!', 'The raw file {} is empty (no intensity values > 0) and will NOT be loaded. Press OK to continue.'.format(os.path.basename(raw_file)))
                        continue
                    analysis_filename = save_analysis_obj(analysis_obj, param_dict, os.path.dirname(raw_obj.filepath))
                    analysis_filenames.append(analysis_filename)
                    self.update_progress(raw_filepaths.index(raw_file), len(raw_filepaths))

                # update the list of analysis files to display
                self.display_analysis_files(analysis_filenames)

                # update directory to match the loaded files
                if not self.output_dir_override:
                    if len(self.analysis_file_list) > 0:
                        self.output_dir = os.path.dirname(self.analysis_file_list[0])
                        self.update_dir_entry()
        self.progress_done()

    def on_button_analysisfile_clicked(self):
        """
        Open a filechooser for the user to select previously process analysis (.ciu) files
        :return:
        """
        analysis_files = self.open_files(filetype=[('CIU files', '.ciu')])
        self.display_analysis_files(analysis_files)

        # update directory to match the loaded files
        try:
            if not self.output_dir_override:
                self.output_dir = os.path.dirname(self.analysis_file_list[0])
                self.update_dir_entry()
        except IndexError:
            # no files selected (user probably hit 'cancel') - ignore
            return

        # check if parameters in loaded files match the current Parameter object
        # self.check_params()

    def display_analysis_files(self, files_to_display):
        """
        Write analysis filenames to the main text window and update associated controls. References
        self.analysis_list for filenames
        :param files_to_display: list of strings (full system paths to file location)
        :return: void
        """
        # update the internal analysis file list
        self.analysis_file_list = files_to_display

        displaystring = ''
        index = 1
        for file in files_to_display:
            displaystring += '{}: {}\n'.format(index, os.path.basename(file).rstrip('.ciu'))
            index += 1

        # clear any existing text, then write the list of files to the display
        self.builder.get_object('Text_analysis_list').delete(1.0, tk.END)
        self.builder.get_object('Text_analysis_list').insert(tk.INSERT, displaystring)

        # update total file number counter
        self.builder.get_object('Entry_num_files').config(state=tk.NORMAL)
        self.builder.get_object('Entry_num_files').delete(0, tk.END)
        self.builder.get_object('Entry_num_files').insert(0, str(len(files_to_display)))
        self.builder.get_object('Entry_num_files').config(state=tk.DISABLED)

    def on_button_restore_clicked(self):
        """
        Restore the original dataset using the Raw_obj for each analysis object requested.
        Can be used to undo cropping, delta-dt, parameter changes, etc. Differs from reprocess
        in that a NEW object is created, so any gaussian fitting/etc is reset in this method.
        :return: void
        """
        plot_keys = [x for x in self.params_obj.params_dict.keys() if 'smoothing' in x]
        param_success, param_dict = self.run_param_ui('Initial Smoothing on Restored Data', plot_keys)
        if param_success:
            files_to_read = self.check_file_range_entries()
            output_files = []
            self.progress_started()
            for analysis_file in files_to_read:
                # load analysis obj and print params
                analysis_obj = load_analysis_obj(analysis_file)

                # update parameters and re-process raw data
                try:
                    new_obj = Raw_Processing.process_raw_obj(analysis_obj.raw_obj, self.params_obj, short_filename=analysis_obj.short_filename)
                except ValueError:
                    # all 0's in raw data - skip this file an notify the user
                    messagebox.showerror('Empty raw data file!', 'The raw data in file {} is empty (no intensity values > 0) and will NOT be loaded. Press OK to continue.'.format(analysis_obj.short_filename))
                    continue

                # new_obj = process_raw_obj(analysis_obj.raw_obj, self.params_obj, short_filename=analysis_obj.short_filename)
                filename = save_analysis_obj(new_obj, param_dict, outputdir=self.output_dir)
                output_files.append(filename)
                self.update_progress(files_to_read.index(analysis_file), len(files_to_read))

            self.display_analysis_files(output_files)
        self.progress_done()

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

    def on_button_update_param_defaults_clicked(self):
        """
        Update the default parameters (in the param info csv file) to be the current settings
        in self.params_obj
        :return: void
        """
        CIU_Params.update_param_csv(self.params_obj, hard_params_file)
        self.progress_done()

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
        if newdir == '':
            # user hit cancel - don't set directory
            return
        self.output_dir = newdir
        self.output_dir_override = True     # stop changing directory on file load if the user has specified a directory
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

    def run_param_ui(self, section_title, list_of_param_keys):
        """
        Run the parameter UI for a given set of parameters.
        :param section_title: Title to display on the popup parameter editing window (string)
        :param list_of_param_keys: List of parameter keys. All values MUST be parameter names (as in
        the __init__ method of the Parameters object.
        :return: void
        """
        list_of_param_keys = sorted(list_of_param_keys)     # keep parameter keys in alphabetical order
        param_ui = CIU_Params.ParamUI(section_name=section_title,
                                      params_obj=self.params_obj,
                                      key_list=list_of_param_keys,
                                      param_descripts_file=hard_params_file)
        param_ui.lift()
        param_ui.grab_set()     # prevent users from hitting multiple windows simultaneously
        param_ui.wait_window()
        param_ui.grab_release()

        # Only update parameters if the user clicked 'okay' (didn't click cancel or close the window)
        if param_ui.return_code == 0:
            return_vals = param_ui.refresh_values()
            self.params_obj.set_params(return_vals)
            return True, return_vals
        return False, None

    def on_button_plot_options_clicked(self):
        """
        Update parameters object with new plot options for all methods
        :return: void
        """
        plot_keys = [x for x in self.params_obj.params_dict.keys() if 'plot' in x and 'override' not in x]
        self.run_param_ui('Plot parameters', plot_keys)
        self.progress_done()

    def on_button_oldplot_clicked(self):
        """
        Run old CIU plot method to generate a plot in the output directory
        :return: void (saves to output dir)
        """
        plot_keys = [x for x in self.params_obj.params_dict.keys() if 'ciuplot_cmap_override' in x]
        param_success, param_dict = self.run_param_ui('Plot parameters', plot_keys)
        if param_success:
            # Determine if a file range has been specified
            files_to_read = self.check_file_range_entries()
            self.progress_started()
            updated_filelist = []
            for analysis_file in files_to_read:
                analysis_obj = load_analysis_obj(analysis_file)
                Original_CIU.ciu_plot(analysis_obj, self.params_obj, self.output_dir)
                self.update_progress(files_to_read.index(analysis_file), len(files_to_read))

                # save analysis obj to ensure that parameter changes are noted correctly
                filename = save_analysis_obj(analysis_obj, param_dict, outputdir=self.output_dir)
                updated_filelist.append(filename)

            self.display_analysis_files(updated_filelist)
        self.progress_done()

    def on_button_oldcompare_clicked(self):
        """
        Run old (batched) RMSD-based CIU plot comparison on selected files
        :return: void (saves to output dir)
        """
        # Determine if a file range has been specified
        files_to_read = self.check_file_range_entries()
        self.progress_started()

        if len(files_to_read) == 1:
            # re-open filechooser to choose a list of files to compare to this standard
            newfiles = self.open_files(filetype=[('CIU files', '.ciu')])
            if len(newfiles) == 0:
                return

            # run parameter dialog for comparison
            param_keys = [x for x in self.params_obj.params_dict.keys() if 'compare_' in x and 'batch' not in x]
            param_success, param_dict = self.run_param_ui('Plot parameters', param_keys)
            if param_success:
                rmsd_print_list = ['File 1, File 2, RMSD (%)']

                # load all files and check axes for comparison
                std_file = files_to_read[0]
                std_obj = load_analysis_obj(std_file)
                compare_objs = [load_analysis_obj(file) for file in newfiles]
                all_objs = [std_obj]
                all_objs.extend(compare_objs)
                check_axes_and_warn(all_objs)

                # compare each CIU analysis object against the standard
                index = 0
                updated_filelist = []
                for compare_obj in compare_objs:
                    rmsd = Original_CIU.compare_basic_raw(std_obj, compare_obj, self.params_obj, self.output_dir)
                    printstring = '{},{},{:.2f}'.format(std_obj.short_filename,
                                                        compare_obj.short_filename,
                                                        rmsd)
                    rmsd_print_list.append(printstring)
                    index += 1
                    self.update_progress(compare_objs.index(compare_obj), len(compare_objs))

                # save analysis objs to ensure that parameter changes are noted correctly
                for analysis_obj in all_objs:
                    filename = save_analysis_obj(analysis_obj, param_dict, outputdir=self.output_dir)
                    updated_filelist.append(filename)
                self.display_analysis_files(updated_filelist)

        if len(files_to_read) == 2:
            # Direct compare between two files
            param_keys = [x for x in self.params_obj.params_dict.keys() if 'compare_' in x and 'batch' not in x]
            param_success, param_dict = self.run_param_ui('Plot parameters', param_keys)
            if param_success:
                ciu1 = load_analysis_obj(files_to_read[0])
                ciu2 = load_analysis_obj(files_to_read[1])
                updated_obj_list = check_axes_and_warn([ciu1, ciu2])
                Original_CIU.compare_basic_raw(updated_obj_list[0], updated_obj_list[1], self.params_obj, self.output_dir)

                # save analysis objs to ensure that parameter changes are noted correctly
                updated_filelist = []
                for analysis_obj in updated_obj_list:
                    filename = save_analysis_obj(analysis_obj, param_dict, outputdir=self.output_dir)
                    updated_filelist.append(filename)
                self.display_analysis_files(updated_filelist)

        elif len(files_to_read) > 2:
            param_keys = [x for x in self.params_obj.params_dict.keys() if 'compare_' in x]
            param_success, param_dict = self.run_param_ui('Plot parameters', param_keys)
            if param_success:
                rmsd_print_list = ['File 1, File 2, RMSD (%)']
                # batch compare - compare all against all.
                f1_index = 0
                loaded_files = [load_analysis_obj(x) for x in files_to_read]
                loaded_files = check_axes_and_warn(loaded_files)

                updated_filelist = []
                for analysis_obj in loaded_files:
                    # don't compare the file against itself
                    skip_index = loaded_files.index(analysis_obj)
                    f2_index = 0
                    while f2_index < len(loaded_files):
                        # skip either comparisons to self or reverse and self comparisons depending on parameter
                        if self.params_obj.compare_batch_1_both_dirs:
                            if f2_index == skip_index:
                                f2_index += 1
                                continue
                        else:
                            # skip reverse and self comparisons
                            if f2_index >= f1_index:
                                f2_index += 1
                                continue

                        ciu1 = analysis_obj
                        ciu2 = loaded_files[f2_index]
                        rmsd = Original_CIU.compare_basic_raw(ciu1, ciu2, self.params_obj, self.output_dir)

                        printstring = '{},{},{:.2f}'.format(ciu1.short_filename, ciu2.short_filename, rmsd)
                        rmsd_print_list.append(printstring)
                        f2_index += 1

                    filename = save_analysis_obj(analysis_obj, param_dict, outputdir=self.output_dir)
                    updated_filelist.append(filename)

                    f1_index += 1
                    self.update_progress(loaded_files.index(analysis_obj), len(loaded_files))
                self.display_analysis_files(updated_filelist)

                # print output to csv
                with open(os.path.join(self.output_dir, 'batch_RMSDs.csv'), 'w') as rmsd_file:
                    for rmsd_string in rmsd_print_list:
                        rmsd_file.write(rmsd_string + '\n')
        self.progress_done()

    def on_button_smoothing_clicked(self):
        """
        Reprocess raw data with new smoothing (and interpolation?) parameters
        :return: void
        """
        param_keys = [x for x in self.params_obj.params_dict.keys() if 'smoothing' in x]
        param_success, param_dict = self.run_param_ui('Smoothing Parameters', param_keys)
        if param_success:
            files_to_read = self.check_file_range_entries()
            self.progress_started()
            new_file_list = []

            for file in files_to_read:
                analysis_obj = load_analysis_obj(file)
                analysis_obj = Raw_Processing.smooth_main(analysis_obj, self.params_obj)
                analysis_obj.refresh_data()
                filename = save_analysis_obj(analysis_obj, param_dict, outputdir=self.output_dir)
                new_file_list.append(filename)

                # also save _raw.csv output if desired
                if self.params_obj.output_1_save_csv:
                    save_path = file.rstrip('.ciu') + '_delta_raw.csv'
                    Original_CIU.write_ciu_csv(save_path, analysis_obj.ciu_data, analysis_obj.axes)
                self.update_progress(files_to_read.index(file), len(files_to_read))

            self.display_analysis_files(new_file_list)
        self.progress_done()

    def on_button_oldavg_clicked(self):
        """
        Average several processed files into a replicate object and save it for further
        replicate processing methods
        :return:
        """
        # Determine if a file range has been specified
        files_to_read = self.check_file_range_entries()
        if len(files_to_read) < 2:
            messagebox.showerror('Too Few Files', 'At least 2 files must be selected to Average')
            return

        self.progress_started()

        # Check that axes are the same across objects
        analysis_obj_list = [load_analysis_obj(x) for x in files_to_read]
        analysis_obj_list = check_axes_and_warn(analysis_obj_list)

        # Compute averaged CIU data and generate output plots
        averaged_obj, std_data = Original_CIU.average_ciu(analysis_obj_list)

        # Save averaged object as a .ciu file
        averaged_obj.filename = save_analysis_obj(averaged_obj, {}, self.output_dir,
                                                  filename_append='_Avg')
        averaged_obj.short_filename = os.path.basename(averaged_obj.filename).rstrip('.ciu')

        # plot averaged object and standard deviation and save output average CSV file
        pairwise_rmsds, rmsd_strings = Original_CIU.get_pairwise_rmsds(analysis_obj_list, self.params_obj)
        Original_CIU.ciu_plot(averaged_obj, self.params_obj, self.output_dir)
        Original_CIU.std_dev_plot(averaged_obj, std_data, pairwise_rmsds, self.params_obj, self.output_dir)
        Original_CIU.save_avg_rmsd_data(analysis_obj_list, self.params_obj, averaged_obj.short_filename, self.output_dir)

        self.display_analysis_files([averaged_obj.filename])
        self.progress_done()

    def on_button_crop_clicked(self):
        """
        Open a dialog to ask user for crop inputs, then crop selected data accordingly.
        NOTE: preserves parameters from original object, as no changes have been made here
        :return: saves new .ciu files
        """
        # run the cropping UI
        if len(self.analysis_file_list) > 0:
            # check axes for equality and interpolate if different
            files_to_read = self.check_file_range_entries()
            loaded_files = [load_analysis_obj(x) for x in files_to_read]

            # Warn user if any files have features or Gaussians, as these will be erased by cropping
            warned_yet_flag = False
            for quick_obj in loaded_files:
                if quick_obj.features_changept is not None or quick_obj.raw_protein_gaussians is not None or quick_obj.features_gaussian is not None:
                    if not warned_yet_flag:
                        messagebox.showwarning('Processing Results will be Erased', 'All processing results (feature detection, Gaussian fitting, etc.) are erased when cropping files to prevent axes errors. If you do not want this, press cancel on the next screen.')
                        warned_yet_flag = True

            crop_vals = Raw_Processing.run_crop_ui(loaded_files[0].axes, hard_crop_ui)
            if crop_vals is None:
                # user hit cancel, or no values were provided
                self.progress_done()
                return

            self.progress_started()

            new_file_list = []
            # for file in files_to_read:
            for analysis_obj in loaded_files:
                new_obj = Raw_Processing.crop(analysis_obj, crop_vals)
                new_obj.refresh_data()
                new_obj.crop_vals = crop_vals
                newfile = save_analysis_obj(new_obj, {}, outputdir=self.output_dir)
                new_file_list.append(newfile)
                self.update_progress(loaded_files.index(analysis_obj), len(loaded_files))

            self.display_analysis_files(new_file_list)
        else:
            messagebox.showwarning('No Files Selected', 'Please select files before performing cropping/interpolation')
        self.progress_done()

    def on_button_interpolate_clicked(self):
        """
        Perform interpolation of data onto new axes using user provided parameters. Overwrites
        existing objects with new objects containing new data and axes to prevent unstable
        behavior if old values (e.g. features) were used after interpolation.
        :return: void
        """
        if len(self.analysis_file_list) > 0:

            # Warn user if any files have features or Gaussians, as these will be erased by cropping
            files_to_read = self.check_file_range_entries()
            loaded_files = [load_analysis_obj(x) for x in files_to_read]
            warned_yet_flag = False
            for quick_obj in loaded_files:
                if quick_obj.features_changept is not None or quick_obj.raw_protein_gaussians is not None or quick_obj.features_gaussian is not None:
                    if not warned_yet_flag:
                        messagebox.showwarning('Processing Results will be Erased', 'All processing results (feature detection, Gaussian fitting, etc.) are erased when cropping files to prevent axes errors. If you do not want this, press cancel on the next screen.')
                        warned_yet_flag = True

            # interpolation parameters
            param_keys = [x for x in self.params_obj.params_dict.keys() if 'interpolate' in x]
            param_success, param_dict = self.run_param_ui('Interpolation Parameters', param_keys)

            if param_success:
                self.progress_started()

                # determine which axis to interpolate
                if self.params_obj.interpolate_1_axis == 'collision voltage':
                    interp_cv = True
                    interp_dt = False
                elif self.params_obj.interpolate_1_axis == 'drift time':
                    interp_cv = False
                    interp_dt = True
                else:
                    interp_cv = True
                    interp_dt = True

                new_file_list = []
                for file in files_to_read:
                    analysis_obj = load_analysis_obj(file)

                    # Restore the original data and interpolate from that to ensure constant level of interpolation
                    try:
                        new_obj = Raw_Processing.process_raw_obj(analysis_obj.raw_obj, self.params_obj,
                                                                 short_filename=analysis_obj.short_filename)
                    except ValueError:
                        # all 0's in raw data - skip this file an notify the user
                        messagebox.showerror('Empty raw data file!',
                                             'The raw data in file {} is empty (no intensity values > 0) and will NOT be loaded. Press OK to continue.'.format(
                                                 analysis_obj.short_filename))
                        continue

                    # compute new axes
                    new_axes = Raw_Processing.compute_new_axes(old_axes=new_obj.axes, interpolation_scaling=int(self.params_obj.interpolate_2_scaling), interp_cv=interp_cv, interp_dt=interp_dt)
                    if not self.params_obj.interpolate_3_onedim:
                        new_obj = Raw_Processing.interpolate_axes(new_obj, new_axes)
                    else:
                        # 1D interpolation mode, pass the correct axis to 1D interp method
                        if interp_dt:
                            new_obj = Raw_Processing.interpolate_axis_1d(new_obj, interp_dt, new_axes[0])
                        elif interp_cv:
                            new_obj = Raw_Processing.interpolate_axis_1d(new_obj, interp_dt, new_axes[1])
                        else:
                            messagebox.showerror('Invalid Mode', '1D interpolation can only be performed on one axis at a time. Please select 2D interpolation for both axes or select a single axis for 1D interpolation.')
                            break

                    # create a new analysis object to prevent unstable behavior with new axes
                    interp_obj = CIUAnalysisObj(new_obj.raw_obj, new_obj.ciu_data, new_obj.axes, self.params_obj, short_filename=new_obj.short_filename)

                    filename = save_analysis_obj(interp_obj, param_dict, outputdir=self.output_dir)
                    new_file_list.append(filename)
                    self.update_progress(files_to_read.index(file), len(files_to_read))

                self.display_analysis_files(new_file_list)
        else:
            messagebox.showwarning('No Files Selected',
                                   'Please select files before performing cropping/interpolation')
        self.progress_done()

    def on_button_gaussfit_clicked(self):
        """
        Run Gaussian fitting on the analysis object list (updating the objects and leaving
        the current list in place). Saves Gaussian diagnostics/info to file in self.output_dir
        :return: void
        """
        # Ask user to specify protein or nonprotein mode
        t1_param_keys = [x for x in self.params_obj.params_dict.keys() if 'gauss_t1' in x]
        t1_param_success, t1_param_dict = self.run_param_ui('Gaussian Fitting Mode', t1_param_keys)
        if t1_param_success:
            # Determine parameters for appropriate mode
            if self.params_obj.gauss_t1_1_protein_mode:
                t2_param_keys = [x for x in self.params_obj.params_dict.keys() if ('gaussian' in x and '_nonprot_' not in x)]
            else:
                t2_param_keys = [x for x in self.params_obj.params_dict.keys() if 'gaussian' in x]

            # Finally, run feature detection in the appropriate mode
            t2_param_success, t2_param_dict = self.run_param_ui('Gaussian Fitting Parameters', t2_param_keys)
            if t2_param_success:
                # Determine if a file range has been specified
                files_to_read = self.check_file_range_entries()
                self.progress_started()
                print('\n**** Starting Gaussian Fitting - THIS MAY TAKE SOME TIME - CIUSuite 2 will not respond until fitting is completed ****')

                new_file_list = []
                all_output = ''
                for file in files_to_read:
                    analysis_obj = load_analysis_obj(file)
                    analysis_obj, csv_output = Gaussian_Fitting.main_gaussian_lmfit(analysis_obj, self.params_obj, self.output_dir)
                    all_output += csv_output

                    t2_param_dict.update(t1_param_dict)
                    filename = save_analysis_obj(analysis_obj, t2_param_dict, outputdir=self.output_dir)
                    new_file_list.append(filename)
                    self.update_progress(files_to_read.index(file), len(files_to_read))

                if self.params_obj.gaussian_5_combine_outputs:
                    outputpath = os.path.join(self.output_dir, 'All_gaussians.csv')
                    try:
                        with open(outputpath, 'w') as output:
                            output.write(all_output)
                    except PermissionError:
                        messagebox.showerror('Please Close the File Before Saving',
                                             'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(
                                                 outputpath))
                        with open(os.path.join(outputpath, outputpath), 'w') as output:
                            output.write(all_output)

                self.display_analysis_files(new_file_list)

                # prompt the user to run feature detection in Gaussian mode
                if len(files_to_read) > 0:
                    messagebox.showinfo('Success!', 'Gaussing fitting finished successfully. Please run Feature Detection in "gaussian" mode to finalize Gaussian assignments to features (this is required for CIU50 analysis, classification, and reconstruction).')
        self.progress_done()

    def on_button_gaussian_reconstruct_clicked(self):
        """
        Create a new CIUAnalysisObj from the fitted Gaussians of selected objects. Must
        have Gaussian feature detection already performed.
        :return: void
        """

        param_keys = [x for x in self.params_obj.params_dict.keys() if 'reconstruct' in x]
        param_success, param_dict = self.run_param_ui('Gaussian Reconstruction Parameters', param_keys)
        if param_success:
            files_to_read = self.check_file_range_entries()
            self.progress_started()
            new_file_list = []

            # If reading in Gaussians from template, load template files and construct Gaussians from provided information
            if self.params_obj.reconstruct_1_mode == 'Load File':
                template_files = filedialog.askopenfilenames(filetypes=[('Gaussian Fit CSV', '.csv')])
                if len(template_files) > 0:
                    self.output_dir = os.path.dirname(template_files[0])

                    for template_file in template_files:
                        # load Gaussian information and generate a CIUAnalysisObj
                        gaussians_by_cv, axes = Gaussian_Fitting.parse_gaussian_list_from_file(template_file)
                        new_analysis_obj = Gaussian_Fitting.reconstruct_from_fits(gaussians_by_cv, axes, os.path.basename(template_file).rstrip('.csv'), self.params_obj)
                        filename = save_analysis_obj(new_analysis_obj, param_dict, outputdir=self.output_dir)

                        # also save an _raw.csv file with the generated data
                        Original_CIU.write_ciu_csv(os.path.join(self.output_dir, new_analysis_obj.short_filename + '_raw.csv'),
                                                   new_analysis_obj.ciu_data,
                                                   new_analysis_obj.axes)

                        new_file_list.append(filename)
                        self.update_progress(template_files.index(template_file), len(template_files))

            else:
                # use loaded .ciu files and read from the Gaussian fitting/feature detection results
                for file in files_to_read:
                    # load file
                    analysis_obj = load_analysis_obj(file)

                    # check to make sure the analysis_obj has Gaussian data fitted
                    if analysis_obj.feat_protein_gaussians is None:
                        messagebox.showerror('Gaussian feature detection required', 'Data in file {} does not have Gaussian feature detection performed. Please run Gaussian feature detection, then try again.')
                        break

                    # If gaussian data exists, perform the analysis
                    final_gausslists, final_axes = Gaussian_Fitting.check_recon_for_crop(analysis_obj.feat_protein_gaussians, analysis_obj.axes)

                    # Save a new analysis object constructed from the fits. NOTE: saves previous objects parameter information for reference
                    new_obj = Gaussian_Fitting.reconstruct_from_fits(final_gausslists, final_axes, analysis_obj.short_filename, analysis_obj.params)
                    filename = save_analysis_obj(new_obj, param_dict, outputdir=self.output_dir)

                    # also save an _raw.csv file with the generated data
                    Original_CIU.write_ciu_csv(os.path.join(self.output_dir, new_obj.short_filename + '_raw.csv'),
                                               new_obj.ciu_data,
                                               new_obj.axes)

                    new_file_list.append(filename)
                    self.update_progress(files_to_read.index(file), len(files_to_read))

            self.display_analysis_files(new_file_list)
        self.progress_done()

    def on_button_feature_detect_clicked(self):
        """
        Run feature detection routine.
        :return: void
        """
        # Ask user to specify standard or Gaussian mode
        t1_param_keys = [x for x in self.params_obj.params_dict.keys() if 'feature_t1' in x]
        t1_param_success, t1_param_dict = self.run_param_ui('Feature Detection Mode', t1_param_keys)

        if t1_param_success:
            # Determine feature detection parameters for appropriate mode
            if self.params_obj.feature_t1_1_ciu50_mode == 'standard':
                t2_param_keys = [x for x in self.params_obj.params_dict.keys() if ('feature_t2' in x and '_gauss_' not in x)]
            else:
                t2_param_keys = [x for x in self.params_obj.params_dict.keys() if 'feature_t2' in x]

            # Finally, run feature detection in the appropriate mode
            t2_param_success, t2_param_dict = self.run_param_ui('Feature Detection Parameters', t2_param_keys)
            if t2_param_success:
                t2_param_dict.update(t1_param_dict)
                files_to_read = self.check_file_range_entries()
                self.progress_started()
                new_file_list = []

                all_outputs = 'Filename,Features Detected\n'
                for file in files_to_read:
                    # load file
                    analysis_obj = load_analysis_obj(file)

                    if self.params_obj.feature_t1_1_ciu50_mode == 'gaussian':
                        # GAUSSIAN MODE
                        # check to make sure the analysis_obj has Gaussian data fitted if needed
                        if analysis_obj.raw_protein_gaussians is None:
                            messagebox.showwarning('Gaussian fitting required', 'Data in file {} does not have Gaussian fitting'
                                                                                'performed. Please run Gaussian fitting, then try '
                                                                                'again.'.format(analysis_obj.short_filename))
                            break

                        # Detect features in the appropriate mode
                        try:
                            analysis_obj = Feature_Detection.feature_detect_gaussians(analysis_obj, self.params_obj)
                        except ValueError:
                            messagebox.showerror('Uneven CV Axis', 'Error: Activation axis in file {} was not evenly spaced. Please use the "Interpolate Data" button to generate an evenly spaced axis. If using Gaussian fitting mode, Gaussian fitting MUST be redone after interpolation.'.format(analysis_obj.short_filename))
                            continue
                        features_list = analysis_obj.features_gaussian
                        filename = save_analysis_obj(analysis_obj, t2_param_dict, outputdir=self.output_dir)
                        new_file_list.append(filename)
                    else:
                        # STANDARD MODE
                        analysis_obj = Feature_Detection.feature_detect_col_max(analysis_obj, self.params_obj)
                        features_list = analysis_obj.features_changept
                        filename = save_analysis_obj(analysis_obj, t2_param_dict, outputdir=self.output_dir)
                        new_file_list.append(filename)

                    # save output
                    Feature_Detection.plot_features(features_list, analysis_obj, self.params_obj, self.output_dir)
                    outputpath = os.path.join(self.output_dir, analysis_obj.short_filename + '_features.csv')
                    if self.params_obj.feature_t2_6_ciu50_combine_outputs:
                        all_outputs += analysis_obj.short_filename
                        all_outputs += Feature_Detection.print_features_list(features_list, outputpath, mode=self.params_obj.feature_t1_1_ciu50_mode, combine=True)
                    else:
                        Feature_Detection.print_features_list(features_list, outputpath,
                                                              mode=self.params_obj.feature_t1_1_ciu50_mode, combine=False)
                    self.update_progress(files_to_read.index(file), len(files_to_read))

                if self.params_obj.feature_t2_6_ciu50_combine_outputs:
                    outputpath = os.path.join(self.output_dir, '_all-features.csv')
                    save_existing_output_string(outputpath, all_outputs)

                self.display_analysis_files(new_file_list)
        self.progress_done()

    def on_button_ciu50_clicked(self):
        """
        Run feature detection workflow to generate CIU-50 (transition) outputs for selected
        files
        :return: void
        """
        # Ask user to specify standard or Gaussian mode
        t1_param_keys = [x for x in self.params_obj.params_dict.keys() if ('_t1_' in x and '_ciu50_' in x)]
        t1_param_success, t1_param_dict = self.run_param_ui('CIU50 Mode', t1_param_keys)

        if t1_param_success:
            # Determine feature detection parameters for appropriate mode
            if self.params_obj.feature_t1_1_ciu50_mode == 'standard':
                t2_param_keys = [x for x in self.params_obj.params_dict.keys() if
                                 ('ciu50' in x and '_t2_' in x and '_gauss_' not in x)]
            else:
                t2_param_keys = [x for x in self.params_obj.params_dict.keys() if ('ciu50' in x and '_t2_' in x)]

            # Finally, run analysis in the appropriate mode
            t2_param_success, t2_param_dict = self.run_param_ui('CIU50 Parameters', t2_param_keys)
            if t2_param_success:
                t2_param_dict.update(t1_param_dict)
                files_to_read = self.check_file_range_entries()
                self.progress_started()

                new_file_list = []
                all_outputs = ''
                short_outputs = 'Filename,CIU50 1,CIU50 2,(etc)\n'
                filename = ''
                combine_flag = False
                gaussian_bool = False
                if self.params_obj.feature_t1_1_ciu50_mode == 'gaussian':
                    gaussian_bool = True

                for file in files_to_read:
                    # load file
                    analysis_obj = load_analysis_obj(file)

                    # Ensure features are detected for this file
                    feature_list = analysis_obj.get_features(gaussian_bool)
                    if feature_list is None:
                        # returning False means features aren't detected for the selected mode. Warn user
                        messagebox.showerror('No Features in File {}'.format(analysis_obj.short_filename),
                                             message='Feature Detection has not been performed in {} mode for file {}. Please '
                                                     'perform feature detection before CIU50 analysis.'.format(self.params_obj.feature_t1_1_ciu50_mode, analysis_obj.short_filename))
                        continue

                    # run CIU50 analysis
                    analysis_obj = Feature_Detection.ciu50_main(feature_list, analysis_obj, self.params_obj, outputdir=self.output_dir, gaussian_bool=gaussian_bool)
                    filename = save_analysis_obj(analysis_obj, t2_param_dict, outputdir=self.output_dir)
                    new_file_list.append(filename)

                    # save outputs to file in combined or stand-alone modes
                    if not self.params_obj.feature_t2_6_ciu50_combine_outputs:
                        Feature_Detection.save_ciu50_outputs(analysis_obj, self.output_dir)
                        Feature_Detection.save_ciu50_short(analysis_obj, self.output_dir)
                        combine_flag = False
                    else:
                        file_string = os.path.basename(filename).rstrip('.ciu') + '\n'
                        all_outputs += file_string
                        all_outputs += Feature_Detection.save_ciu50_outputs(analysis_obj, self.output_dir, combine=True)
                        short_outputs += os.path.basename(filename).rstrip('.ciu')
                        short_outputs += Feature_Detection.save_ciu50_short(analysis_obj, self.output_dir, combine=True)
                        combine_flag = True
                    self.update_progress(files_to_read.index(file), len(files_to_read))

                if combine_flag:
                    # save final combined output
                    outputpath = os.path.join(self.output_dir, os.path.basename(filename.rstrip('.ciu')) + '_ciu50s.csv')
                    outputpath_short = os.path.join(self.output_dir, os.path.basename(filename.rstrip('.ciu')) + '_ciu50-short.csv')
                    save_existing_output_string(outputpath, all_outputs)
                    save_existing_output_string(outputpath_short, short_outputs)

                self.display_analysis_files(new_file_list)
        self.progress_done()

    def on_button_classification_supervised_clicked(self):
        """
        Run supervised classification from Classification module. Currently set up to use file dialogs
        to ask for user specified class labels and data files, but may change in future.
        :return: void
        """
        num_classes = simpledialog.askinteger('Class Number', 'Into how many classes do you want to group?')
        if num_classes is None:
            # user hit cancel - return
            print('classification canceled')
            return
        data_labels = []
        obj_list_by_label = []
        endfile = ''
        for index in range(0, num_classes):
            # Read in the .CIU files and labels for each class, canceling the process if the user hits 'cancel'
            label = simpledialog.askstring('Class Name', 'Enter the name (class label) for class {}'.format(index + 1))
            if label is None:
                print('classification canceled')
                return
            files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
            if len(files) == 0:
                print('classification canceled')
                return
            if len(files) < 3:
                messagebox.showerror('Not Enough Files', 'At least 3 replicates are required for classification. Please select at least 3 files per class and try again.')
                return

            obj_list = []
            for file in files:
                with open(file, 'rb') as analysis_file:
                    obj = pickle.load(analysis_file)
                obj_list.append(obj)
                endfile = file
            data_labels.append(label)
            obj_list_by_label.append(obj_list)

        # If no data has been loaded, use file location as default output dir
        current_dir_text = self.builder.get_object('Text_outputdir').get(1.0, tk.END).rstrip('\n')
        if current_dir_text == '(No files loaded yet)':
            self.output_dir = os.path.dirname(endfile)

        # check axes
        obj_list_by_label, equalized_axes_list = Raw_Processing.equalize_axes_2d_list(obj_list_by_label)

        # get classification parameters
        param_keys = [x for x in self.params_obj.params_dict.keys() if 'classif' in x]
        param_success, param_dict = self.run_param_ui('Classification Parameters', param_keys)
        if param_success:
            if self.params_obj.classif_3_unk_mode == 'Gaussian':
                # Ensure Gaussian features are present and prepare them for classification
                max_num_gaussians = 0
                for obj_list in obj_list_by_label:
                    for analysis_obj in obj_list:
                        if analysis_obj.features_gaussian is None:
                            messagebox.showerror('No Gaussian Features Fitted', 'Error: Gaussian feature classification selected, '
                                                                                'but Gaussian Feature Detection has not been performed yet. '
                                                                                'Please run Gaussian Feature Detection on all files being used '
                                                                                'for classification and try again.')
                            # cancel the classification
                            self.progress_done()
                        else:
                            # make ready the gaussians (saved into analysis object and length checked)
                            gaussians_by_cv = Classification.prep_gaussfeats_for_classif(analysis_obj.features_gaussian, analysis_obj)
                            for gaussian_list in gaussians_by_cv:
                                if len(gaussian_list) > max_num_gaussians:
                                    max_num_gaussians = len(gaussian_list)
                # save num Gaussians to ensure all matrices same size
                self.params_obj.silent_clf_4_num_gauss = max_num_gaussians
            else:
                max_num_gaussians = 0   # no Gaussians in non-Gaussian mode

            # Run the classification
            if self.params_obj.classif_5_auto_featselect == 'automatic':
                self.progress_print_text('LDA in progress (may take a few minutes)...', 50)
                scheme = Classification.main_build_classification(data_labels, obj_list_by_label, self.params_obj, self.output_dir)
                scheme.final_axis_cropvals = equalized_axes_list
                scheme.num_gaussians = max_num_gaussians
                Classification.save_scheme(scheme, self.output_dir)
            else:
                # manual feature selection mode: run feature selection, pause, and THEN run LDA with user input
                self.progress_print_text('Feature Evaluation in progress...', 50)
                shaped_label_list = []
                for index, label in enumerate(data_labels):
                    shaped_label_list.append([label for _ in range(len(obj_list_by_label[index]))])

                # run feature selection
                Classification.univariate_feature_selection(shaped_label_list, obj_list_by_label, self.params_obj, self.output_dir)

                # prompt for user input to select desired features
                input_success_flag = False
                selected_features = []
                while not input_success_flag:
                    input_success_flag, selected_features = parse_user_cvfeats_input()

                # Run LDA using selected features
                self.progress_print_text('LDA in progress (may take a few minutes)...', 50)
                scheme = Classification.main_build_classification(data_labels, obj_list_by_label, self.params_obj,
                                                                  self.output_dir, known_feats=selected_features)
                scheme.final_axis_cropvals = equalized_axes_list
                scheme.num_gaussians = max_num_gaussians
                Classification.save_scheme(scheme, self.output_dir)

        self.progress_done()

    def on_button_classify_unknown_clicked(self):
        """
        Open filechooser to load classification scheme object (previously saved), then run
        classification on all loaded .ciu files against that scheme.
        :return: void
        """
        # load files from table
        files_to_read = self.check_file_range_entries()
        self.progress_started()
        analysis_obj_list = []
        for file in files_to_read:
            # load file
            analysis_obj = load_analysis_obj(file)
            analysis_obj_list.append(analysis_obj)

        # check parameters
        param_keys = [x for x in self.params_obj.params_dict.keys() if 'unk' in x]
        param_success, param_dict = self.run_param_ui('Unknown Classification Parameters', param_keys)
        if param_success:
            # load classification scheme file
            scheme_file = filedialog.askopenfilename(filetypes=[('Classification File', '.clf')])
            if not scheme_file == '':
                scheme = Classification.load_scheme(scheme_file)

                # ensure Gaussian features are present if requested
                if self.params_obj.classif_3_unk_mode == 'Gaussian':
                    # make sure the scheme is in Gaussian mode, cancel classification if not
                    if scheme.num_gaussians == 0:
                        messagebox.showerror('Non-Gaussian Scheme Selected', 'Gaussian mode specified, but the selected classification scheme (.clf file) is NOT a Gaussian classification scheme. No analysis will be performed.')
                        self.progress_done()

                    for analysis_obj in analysis_obj_list:
                        if analysis_obj.features_gaussian is None:
                            messagebox.showerror('No Gaussian Features Fitted',
                                                 'Error: No Gaussian Features in file: {} . Gaussian Feature classification selected, '
                                                 'but Gaussian Feature Detection has not been performed yet. '
                                                 'Please run Gaussian Feature Detection on all files being used'
                                                 'and try again.')
                            # skip file
                            continue
                        else:
                            # prepare gaussian data for classification
                            final_gaussians = Classification.prep_gaussfeats_for_classif(analysis_obj.features_gaussian, analysis_obj)
                            analysis_obj.classif_gaussfeats = final_gaussians

                            # make ready the gaussians (saved into analysis object and length checked)
                            skip_flag = False
                            for gaussian_list in final_gaussians:
                                if len(gaussian_list) > scheme.num_gaussians:
                                    messagebox.showerror('Gaussian Extrapolation Error',
                                                         'Warning: there are more overlapping features in file {} than in any of the training data used the classification scheme. '
                                                         'To fit, ensure that the Gaussian Feature Detection outputs for this file are similar to the data used to build the chosen '
                                                         'classification scheme (especially in the maximum number of features that overlap at one CV)'.format(
                                                             analysis_obj.short_filename))
                                    # skip file
                                    skip_flag = True
                                    break
                            if skip_flag:
                                continue

                # check axes against Scheme's saved axes and equalize if needed
                scheme_cvs = [x.cv for x in scheme.selected_features]
                try:
                    analysis_obj_list = Raw_Processing.equalize_unk_axes_classif(analysis_obj_list, scheme.final_axis_cropvals, scheme_cvs)
                except (ValueError, TypeError) as err:
                    # raised when the exact CV from the scheme cannot be found in the unknown object. Warn user with dialog
                    messagebox.showerror('Axis Mismatch', message='{}. File: {}, CV: {}'.format(*err.args))
                    self.progress_done()

                # analyze unknowns using the scheme and save outputs
                successful_objs_for_plot = []
                new_filename_list = []
                for analysis_obj in analysis_obj_list:
                    # Finally, perform the classification and save outputs
                    analysis_obj = scheme.classify_unknown(analysis_obj, self.params_obj, self.output_dir)
                    filename = save_analysis_obj(analysis_obj, param_dict, self.output_dir)
                    new_filename_list.append(filename)

                    successful_objs_for_plot.append(analysis_obj)
                    self.update_progress(analysis_obj_list.index(analysis_obj), len(analysis_obj_list))

                if len(successful_objs_for_plot) > 0:
                    # Classification.save_predictions(successful_objs_for_plot, scheme.selected_features, scheme.unique_labels, self.output_dir, scheme.name)
                    scheme.plot_all_unknowns(successful_objs_for_plot, self.params_obj, self.output_dir)

                    # compile all successfully fitted LDA and prediction data for output csv
                    all_transform_data = [x.classif_transformed_data for x in successful_objs_for_plot]
                    all_filenames = [x.short_filename for x in successful_objs_for_plot]
                    all_predictions = [x.classif_predicted_label for x in successful_objs_for_plot]
                    all_probs = [x.classif_probs_by_cv for x in successful_objs_for_plot]
                    # Classification.save_lda_output_unk(all_transform_data, all_filenames, scheme.selected_features, self.output_dir)
                    Classification.save_lda_and_predictions(scheme, all_transform_data, all_predictions, all_probs, all_filenames, self.output_dir, True)
                    Classification.plot_probabilities(self.params_obj, scheme, all_probs, self.output_dir, unknown_bool=True)

                self.display_analysis_files(new_filename_list)
        self.progress_done()

    def check_params(self):
        """
        Check each file the in the analysis file list to see if its parameters match the
        current Parameters object. If not, display the file numbers of mismatched files.
        :return: void
        """
        mismatched_files = []
        file_index = 1
        for file in self.analysis_file_list:
            analysis_obj = load_analysis_obj(file)
            if not analysis_obj.params.compare(self.params_obj):
                # parameters don't match, add to list
                mismatched_files.append(file_index)
            file_index += 1

        # write output to text display
        if len(mismatched_files) == 0:
            match_string = 'Parameters in each file match the current parameters'
            self.builder.get_object('Text_ParamsMatch').config(state=tk.NORMAL)
            self.builder.get_object('Text_ParamsMatch').delete(1.0, tk.INSERT)
            self.builder.get_object('Text_ParamsMatch').insert(tk.INSERT, match_string)
            self.builder.get_object('Text_ParamsMatch').config(state=tk.DISABLED)
        else:
            file_string = ','.join([str(x) for x in mismatched_files])
            mismatch_string = 'Unsynchronized parameters in file(s): {}'.format(file_string)
            self.builder.get_object('Text_ParamsMatch').config(state=tk.NORMAL)
            self.builder.get_object('Text_ParamsMatch').delete(1.0, tk.INSERT)
            self.builder.get_object('Text_ParamsMatch').insert(tk.INSERT, mismatch_string)
            self.builder.get_object('Text_ParamsMatch').config(state=tk.DISABLED)

    def progress_print_text(self, text_to_print, prog_percent):
        """
        Set the progress bar to print text (e.g. 'analysis in progress...')
        :param text_to_print: text to display in the progress bar
        :param prog_percent: percent to fill the progress bar
        :return: void
        """
        # update progress
        progress_bar = self.builder.get_object('Progressbar_main')
        progress_bar['value'] = prog_percent
        # display text
        self.builder.get_object('Entry_progress').config(state=tk.NORMAL)
        self.builder.get_object('Entry_progress').delete(0, tk.END)
        self.builder.get_object('Entry_progress').insert(0, text_to_print)
        self.builder.get_object('Entry_progress').config(state=tk.DISABLED)
        # refresh display
        self.mainwindow.update()

    def progress_started(self):
        """
        Display a message showing that the operation has begun
        :return: void
        """
        self.builder.get_object('Entry_progress').config(state=tk.NORMAL)
        self.builder.get_object('Entry_progress').delete(0, tk.END)
        self.builder.get_object('Entry_progress').insert(0, 'Processing... (This window will not respond until processing completes!)')
        self.builder.get_object('Entry_progress').config(state=tk.DISABLED)

        self.builder.get_object('Progressbar_main')['value'] = 10
        self.mainwindow.update()

    def update_progress(self, current_analysis, num_analyses):
        """
        Update the progress bar to display the current progress through the analysis list
        :param current_analysis: the file NUMBER currently being worked on by the program
        :param num_analyses: the total number of files in the current analysis
        :return: void
        """
        current_prog = (current_analysis + 1) / float(num_analyses) * 100
        prog_string = 'Processed {} of {}'.format(current_analysis + 1, num_analyses)

        progress_bar = self.builder.get_object('Progressbar_main')
        progress_bar['value'] = current_prog

        self.builder.get_object('Entry_progress').config(state=tk.NORMAL)
        self.builder.get_object('Entry_progress').delete(0, tk.END)
        self.builder.get_object('Entry_progress').insert(0, prog_string)
        self.builder.get_object('Entry_progress').config(state=tk.DISABLED)
        self.mainwindow.update()

    def progress_done(self):
        """
        Called after methods finished to ensure GUI mainloop continues
        :return: void
        """
        self.builder.get_object('Entry_progress').config(state=tk.NORMAL)
        self.builder.get_object('Entry_progress').delete(0, tk.END)
        self.builder.get_object('Entry_progress').insert(0, 'Done!')
        self.builder.get_object('Entry_progress').config(state=tk.DISABLED)

        self.builder.get_object('Progressbar_main')['value'] = 100
        # added to keep program from exiting when run from command line - not sure if there's a better way to do this, but seems to work
        self.run()
        # return 0

    def open_files(self, filetype):
        """
        Open a tkinter filedialog to choose files of the specified type. NOTE: withdraws the main window of the
        UI (self) to prevent users from clicking on any other GUI elements while the filedialog is active.
        :param filetype: filetype filter in form [(name, extension)]
        :return: list of selected files
        """
        self.mainwindow.withdraw()
        files = filedialog.askopenfilenames(filetypes=filetype)
        self.mainwindow.deiconify()
        return files

    def on_button_vendor_raw_clicked(self):
        """
        Open the Agilent CIU extractor app with a specified output directory. The extractor app will generate _raw.csv
        files into the specified directory. Then runs Agilent data converter script to handle duplicate DTs and
        edit the CV header to be the correct values. Finally, loads data into CIUSuite 2 using the standard load
        raw files method.
        :return: void
        """
        param_keys = [x for x in self.params_obj.params_dict.keys() if 'vendor' in x]
        param_success, param_dict = self.run_param_ui('Which Vendor Raw Data Type to Extract?', param_keys)
        if param_success:
            vendor_type = self.params_obj.vendor_1_type

            if vendor_type == 'Agilent':
                # Call Agilent extractor using the directory chosen by the user for output
                raw_dir = filedialog.askdirectory(title='Choose directory in which to save _raw.csv files from Agilent data')
                agilent_args = '{} "{}"'.format(hard_agilent_ext_path, raw_dir)
                completed_proc = subprocess.run(agilent_args)

                if not completed_proc.returncode == 0:
                    messagebox.showerror('Data Extraction Error', 'Error: Agilent Data Extraction Failed. Returning.')
                    self.progress_done()
                    return

                # Ask user for the extracted files
                raw_files = filedialog.askopenfilenames(title='Choose the extracted files to analyze', initialdir=raw_dir, filetypes=[('_raw.csv', '_raw.csv')])
                cv_headers, parsing_success = [], False

                # first, edit file CV headers since we can't get that information from the Agilent raw library
                try:
                    original_header = Raw_Data_Import.get_header(raw_files[0])
                except IndexError:
                    # no files selected by the user (cancel). Return to mainloop
                    self.progress_done()
                    return

                while not parsing_success:
                    cv_headers, parsing_success = Raw_Data_Import.ask_input_cv_data(original_header)

                for raw_file in raw_files:
                    Raw_Data_Import.read_agilent_and_correct(raw_file, cv_headers)

                # Finally, load the provided raw files
                self.load_raw_files(raw_files)

            elif vendor_type == 'Waters':
                # get raw data folders (because Waters saves data in folders)
                raw_vendor_files = Raw_Data_Import.get_data(self.params_obj.silent_filechooser_dir)

                if len(raw_vendor_files) > 0:
                    # save the location of the files into the params obj for convenient future reference
                    update_dict = {'silent_filechooser_dir': os.path.dirname(raw_vendor_files[0])}
                    CIU_Params.update_specific_param_vals(update_dict, hard_params_file)

                    # Run basic extraction method, which contains the option to open TWIMExtract if more complex extractions required
                    range_vals = Raw_Data_Import.run_twimex_ui()
                    if range_vals is not None:
                        # choose directory in which to save output
                        raw_dir = filedialog.askdirectory(title='Choose directory in which to save extracted _raw.csv files')
                        if raw_dir is not '':
                            self.progress_started()
                            raw_files = Raw_Data_Import.twimex_single_range(range_vals, raw_vendor_files, raw_dir, hard_twimextract_path)

                            if len(raw_files) > 0:
                                # Finally, load the provided raw files
                                self.load_raw_files(raw_files)
                            else:
                                print('Error: no raw files found! Check the chosen save directory')

            else:
                print('Invalid vendor, no files loaded')

        self.progress_done()


# ****** CIU Main I/O methods ******
def save_existing_output_string(full_output_path, string_to_save):
    """
    Write an existing (e.g. combined from several objects) string to file
    :param full_output_path: full path filename in which to save output
    :param string_to_save: string to save
    :return: void
    """
    try:
        with open(full_output_path, 'w') as outfile:
            outfile.write(string_to_save)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving',
                             'The file {} is open or being used by another program! Please close it, THEN press the OK button to retry saving'.format(full_output_path))
        with open(full_output_path, 'w') as outfile:
            outfile.write(string_to_save)


def check_axes_and_warn(loaded_obj_list):
    """
    Check that all CIUAnalysisObj's in list have same axis, and interpolate and warn the user
    if not.
    :param loaded_obj_list: list of loaded objects
    :type loaded_obj_list: list[CIUAnalysisObj]
    :return: updated list of objects with axes equalized
    :rtype list[CIUAnalysisObj]
    """
    loaded_obj_list, final_axes, adjust_flag = Raw_Processing.equalize_axes_main(loaded_obj_list)

    if adjust_flag:
        messagebox.showinfo('Different axes in file(s)', 'FYI: At least some of the loaded files had different axes and/or unevely spaced axes. '
                                                         'Data was interpolated and/or re-framed onto identical, evenly spaced axes. '
                                                         'Please click OK to continue.')
    return loaded_obj_list


def parse_user_cvfeats_input():
    """
    Open a dialog to ask the user for manual feature selection input. Parse to values and return them
    and a success flag if parsing succeeded. Intended to be run inside a flag-checking loop that exits
    once successful parsing has been achieved.
    :return: boolean (success flag), list of parsed values
    """
    parsed_values = []

    user_input = simpledialog.askstring('Enter Collision Voltages to use for Classification', 'Enter the desired Collision Voltage(s), separated by commas. Use no characters other than numbers (decimals okay) and commas.')
    splits = user_input.split(',')
    for cv_split in splits:
        try:
            cv = float(cv_split.strip())
            parsed_values.append(cv)
        except ValueError:
            messagebox.showerror('Invalid Number', 'The entry {} is not a valid number. Please try again')
            return False, parsed_values

    # initialize a list of CFeatures using the parsed values
    selected_features = []
    for value in parsed_values:
        selected_features.append(Classification.CFeature(value, None, None, None))

    return True, selected_features


def parse_tooltips_file(tooltips_file):
    """
    Parse the tooltips.txt file for all tooltips to display in the GUI. Returns a dictionary
    with all object names and descriptions to display
    :param tooltips_file: File to parse (tab-delimited text, headers = #)
    :return: Dictionary, key=component name to pass to Pygubu, value=description
    """
    tooltip_dict = {}
    try:
        with open(tooltips_file, 'r') as tfile:
            lines = list(tfile)
            for line in lines:
                # skip headers and blank lines
                if line.startswith('#') or line.startswith('\n'):
                    continue
                line = line.replace('"', '')
                splits = line.rstrip('\n').split('\t')
                try:
                    key = splits[0].strip()
                    value = splits[1].strip()
                    tooltip_dict[key] = value
                except ValueError:
                    print('Tooltip not parsed for line: {}'.format(line))
                    continue
        return tooltip_dict

    except FileNotFoundError:
        print('params file not found!')


def save_analysis_obj(analysis_obj, params_dict, outputdir, filename_append=''):
    """
    Pickle the CIUAnalysisObj for later retrieval
    :param analysis_obj: CIUAnalysisObj to save
    :type analysis_obj: CIUAnalysisObj
    :param params_dict: Dict with specific params to update in the analysis object's parameters object.
    For new analysis objects, save the full dict from CIU2Main.params_obj
    :type params_dict: dict
    :param filename_append: Addtional filename to append to the raw_obj name (e.g. 'AVG')
    :param outputdir:  directory in which to save.
    :return: full path to save location
    """
    file_extension = '.ciu'

    # update parameters with only those changed in the current analysis (or all params for new analysis objects)
    analysis_obj.params.set_params(params_dict)
    # analysis_obj.params = update_params_in_obj(analysis_obj, params_dict)

    # Generate filename and short filename and save to object
    if analysis_obj.short_filename is None:
        filename = os.path.basename(analysis_obj.raw_obj.filename.rstrip('_raw.csv'))
    else:
        filename = analysis_obj.short_filename
    picklefile = os.path.join(outputdir, filename + filename_append + file_extension)
    analysis_obj.filename = picklefile
    analysis_obj.short_filename = os.path.basename(picklefile.rstrip('.ciu'))

    # Save the .ciu file and return the final filename
    try:
        with open(picklefile, 'wb') as pkfile:
            pickle.dump(analysis_obj, pkfile)
    except IOError:
        messagebox.showerror('File Save Error', 'Error: file {} could not be saved!'.format(picklefile))

    return picklefile


def load_analysis_obj(analysis_filename):
    """
    Load a pickled analysis object back into program memory
    :param analysis_filename: full path to file location to load
    :rtype: CIUAnalysisObj
    :return: CIUAnalysisObj
    """
    with open(analysis_filename, 'rb') as analysis_file:
        analysis_obj = pickle.load(analysis_file)
        analysis_obj.filename = analysis_filename
        analysis_obj.short_filename = os.path.basename(analysis_filename).rstrip('.ciu')
    return analysis_obj


if __name__ == '__main__':
    # Build the GUI and start its mainloop (run) method
    multiprocessing.freeze_support()
    root = tk.Tk()
    root.withdraw()
    print('building GUI...')
    ciu_app = CIUSuite2(root)
    print('Starting CIUSuite 2')
    ciu_app.run()
