"""
Main entry point for CIUSuite 2. Designed to allow the user to choose files and perform
processing to generate analysis objects, and process analysis objects. Probably will need
a (very) basic GUI of some kind.
"""

# GUI test
import pygubu
import pygubu.widgets.simpletooltip as tooltip
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
import os
import subprocess
import pickle
import numpy as np

import Raw_Processing
from CIU_raw import CIURaw
from CIU_analysis_obj import CIUAnalysisObj
import CIU_Params
from CIU_Params import Parameters
import Original_CIU
import Gaussian_Fitting
import Feature_Detection
import Classification

# hard_file_path_ui = r"C:\CIUSuite2\CIUSuite2_2.ui"
hard_file_path_ui = r"C:\CIUSuite2\CIUSuite2.ui"

# hard_params_file = r"C:\CIUSuite2\CIU_params.txt"
hard_params_file = CIU_Params.hard_descripts_file
hard_output_default = r"C:\Users\dpolasky\Desktop\test"
hard_params_ui = r"C:\CIUSuite2\Param_editor.ui"
hard_crop_ui = r"C:\CIUSuite2\Crop_vals.ui"


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
        # self.mainwindow = builder.get_object('CIU_app_top', master_window)
        self.mainwindow = builder.get_object('CIU_app_top')

        self.mainwindow.protocol('WM_DELETE_WINDOW', self.on_close_window)

        callbacks = {
            'on_button_rawfile_clicked': self.on_button_rawfile_clicked,
            'on_button_analysisfile_clicked': self.on_button_analysisfile_clicked,
            'on_button_paramload_clicked': self.on_button_paramload_clicked,
            'on_button_printparams_clicked': self.on_button_printparams_clicked,
            'on_button_reproc_files_clicked': self.on_button_reproc_files_clicked,
            'on_button_changedir_clicked': self.on_button_changedir_clicked,
            'on_button_oldplot_clicked': self.on_button_oldplot_clicked,
            'on_button_oldcompare_clicked': self.on_button_oldcompare_clicked,
            'on_button_oldavg_clicked': self.on_button_oldavg_clicked,
            'on_button_olddeltadt_clicked': self.on_button_olddeltadt_clicked,
            'on_button_crop_clicked': self.on_button_crop_clicked,
            'on_button_restore_clicked': self.on_button_restore_clicked,
            'on_button_gaussfit_clicked': self.on_button_gaussfit_clicked,
            'on_button_ciu50_clicked': self.on_button_ciu50_clicked,
            'on_button_ciu50_gaussian_clicked': self.on_button_ciu50_gaussian_clicked,
            'on_button_feature_gaussian_clicked': self.on_button_feature_gaussian_clicked,
            'on_button_feature_changept_clicked': self.on_button_feature_changept_clicked,
            'on_button_classification_supervised_clicked': self.on_button_classification_supervised_clicked,
            'on_button_classify_unknown_clicked': self.on_button_classify_unknown_clicked,
            # 'on_button_classify_build_scheme_clicked': self.on_button_classify_build_scheme_clicked,
            'on_button_smoothing_clicked': self.on_button_smoothing_clicked
        }
        builder.connect_callbacks(callbacks)
        self.initialize_tooltips()

        # load parameter file
        self.params_obj = CIU_Params.Parameters()
        self.params_obj.set_params(CIU_Params.parse_params_file(hard_params_file))
        self.param_file = hard_params_file

        # params_text = self.builder.get_object('Text_params')
        # params_text.delete(1.0, tk.END)
        # params_text.insert(tk.INSERT, 'Parameters loaded from hard file')

        # holder for feature information in between user assessment - plan to replace with better solution eventually
        self.temp_feature_holder = None

        self.analysis_file_list = []
        self.output_dir = hard_output_default

    def run(self):
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
        tooltip.create(self.builder.get_object('Text_analysis_list'), 'Selected .ciu files appear here. Any processing will be '
                                                                      '\nperformed on all files in the table, unless the '
                                                                      '\nboxes below indicate only a subset of the files.')
        tooltip.create(self.builder.get_object('Text_ParamsMatch'), 'When parameters are edited, this box displays if edits have been saved to the .ciu files')
        tooltip.create(self.builder.get_object('Button_RawFile'), 'Select text files to analyze. Data will be loaded and converted to generate a .ciu file, '
                                                                  '\nwhich will appear in the table below')
        tooltip.create(self.builder.get_object('Button_AnalysisFile'), 'Select previously processed .ciu files to load. Files will be displayed in the table below')
        tooltip.create(self.builder.get_object('Button_change_outputdir'), 'Select a new directory in which to save output files, including .ciu files.'
                                                                           '\nBy default, this is set to the directory containing the loaded .csv or .ciu '
                                                                           '\n files in the table.')
        # tooltip.create(self.builder.get_object('Button_reproc_files'), 'THIS BUTTON IS WAITING FOR A FINAL DECISION ON HOW WE ARE HANDLING PARAMETER EDITS AND DOES THE'
        #                                                                '\nSAME THING AS RESTORE ORIGINAL DATA FOR NOW')
        tooltip.create(self.builder.get_object('Button_restore_raw'), 'Clears ALL previous processing, cropping, and analysis from a .ciu file, restoring the original raw data.'
                                                                      '\nCannot be undone')
        tooltip.create(self.builder.get_object('Entry_start_files'), 'To process only some files in the table above, enter the number corresponding to the beginning'
                                                                     '\nof the range of files to analyze. Must be an integer value only')
        tooltip.create(self.builder.get_object('Entry_end_files'), 'To process only some files in the table above, enter the number corresponding to the end'
                                                                   '\nof the range of files to analyze. Must be an integer value only')
        tooltip.create(self.builder.get_object('Button_old_plot'), 'Generate a CIU fingerprint plot for all files in the table above')
        tooltip.create(self.builder.get_object('Button_old_compare'), 'Perform RMSD comparisons between the files in the table above.')
        tooltip.create(self.builder.get_object('Button_old_average'), 'Average selected fingerprints in the table above to form a new .ciu file with the'
                                                                      '\naveraged data. Intended only for replicate analyses of the same data. '
                                                                      '\nUse the range entries beneath the table to select replicate data if multiple'
                                                                      '\ndatasets are present in the table.')
        tooltip.create(self.builder.get_object('Button_old_deltadt'), 'Convert selected files to a delta-drift time axis. Each file has its drift time axis'
                                                                      '\nshifted such that the apex/peak drift time in the first (lowest energy) column is '
                                                                      '\nset to 0. This enables RMSD comparisons of shifting drift times separately from '
                                                                      '\ndifferences in starting/initial drift time.')
        tooltip.create(self.builder.get_object('Button_crop'), 'Crop CIU data in one or both axes. A popup window will open to input crop dimensions'
                                                               '\nUpdates the .ciu file with the cropped data and axes. To undo, use the Restore Original Data button.')

    def on_button_rawfile_clicked(self):
        """
        Open a filechooser for the user to select raw files, then process them
        :return:
        """
        # clear analysis list
        self.analysis_file_list = []

        raw_files = open_files([('_raw.csv', '_raw.csv')])
        if len(raw_files) > 0:
            self.progress_started()

            # run raw processing
            for raw_file in raw_files:
                raw_obj = generate_raw_obj(raw_file)
                analysis_obj = process_raw_obj(raw_obj, self.params_obj)
                analysis_filename = save_analysis_obj(analysis_obj, self.params_obj)
                self.analysis_file_list.append(analysis_filename)
                self.update_progress(raw_files.index(raw_file), len(raw_files))

            # update the list of analysis files to display
            self.display_analysis_files()

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
            # no files selected (user probably hit 'cancel') - ignore
            return

        # check if parameters in loaded files match the current Parameter object
        self.check_params()

    def on_button_paramload_clicked(self):
        """
        Open a user chosen parameter file into self.params
        :return: void
        """
        # self.builder.get_object('Button_AnalysisFile').config(state=tk.DISABLED)
        #
        # self.on_button_edit_params_clicked()
        #
        # self.builder.get_object('Button_AnalysisFile').config(state=tk.NORMAL)
        try:
            new_param_file = open_files([('params file', '.txt')])[0]
        except IndexError:
            # no file loaded - user probably clicked cancel. Ignore the button call
            return

        new_param_obj = CIU_Params.Parameters()
        new_param_obj.set_params(CIU_Params.parse_params_file(new_param_file))
        self.params_obj = new_param_obj
        self.param_file = new_param_file

        # update parameter location display
        new_text = 'Parameters loaded from {}'.format(os.path.basename(new_param_file))
        params_text = self.builder.get_object('Text_params')
        params_text.delete(1.0, tk.END)
        params_text.insert(tk.INSERT, new_text)

        # check if files are loaded
        if len(self.analysis_file_list) > 0:
            # TODO: prompt user to overwrite params or not
            print('TO-DO: ask user for overwrite or not')

    def on_button_edit_params_clicked(self):
        """
        Open the current parameter file in Notepad to allow the user to edit. Waits for
        the notepad application to close before returning to the CIU2 GUI.
        Updates the current Params object once the parameter file has been closed
        :return: void
        """
        param_args = ['notepad.exe', self.param_file]
        return_proc = subprocess.run(param_args)
        print(return_proc)

        # once done, update stuff and continue
        new_param_obj = CIU_Params.Parameters()
        new_param_obj.set_params(CIU_Params.parse_params_file(self.param_file))
        self.params_obj = new_param_obj

        new_text = 'Parameters updated in {}'.format(os.path.basename(self.param_file))
        params_text = self.builder.get_object('Text_params')
        params_text.delete(1.0, tk.END)
        params_text.insert(tk.INSERT, new_text)

        # check if files are loaded
        if len(self.analysis_file_list) > 0:
            # TODO: prompt user to overwrite params or not
            print('TO-DO: ask user for overwrite or not')

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

    def on_button_reproc_files_clicked(self):
        """
        Re-run processing from raw and update parameter file in all loaded Analysis objects/.ciu files
        Updates ciu_data and parameters in the object, but does NOT delete saved fitting/features/etc
        information from the object.
        :return: void
        """
        files_to_read = self.check_file_range_entries()
        output_files = []
        self.progress_started()
        for analysis_file in files_to_read:
            # load analysis obj and print params
            analysis_obj = load_analysis_obj(analysis_file)

            # update parameters, ciu_data, and axes but retain all other object information
            analysis_obj = reprocess_raw(analysis_obj, self.params_obj)
            filename = save_analysis_obj(analysis_obj, self.params_obj, outputdir=self.output_dir)
            output_files.append(filename)
            self.update_progress(files_to_read.index(analysis_file), len(files_to_read))

        self.display_analysis_files()
        self.progress_done()

    def on_button_restore_clicked(self):
        """
        Restore the original dataset using the Raw_obj for each analysis object requested.
        Can be used to undo cropping, delta-dt, parameter changes, etc. Differs from reprocess
        in that a NEW object is created, so any gaussian fitting/etc is reset in this method.
        :return: void
        """
        files_to_read = self.check_file_range_entries()
        output_files = []
        self.progress_started()
        for analysis_file in files_to_read:
            # load analysis obj and print params
            analysis_obj = load_analysis_obj(analysis_file)

            # update parameters and re-process raw data
            new_obj = process_raw_obj(analysis_obj.raw_obj, self.params_obj)
            filename = save_analysis_obj(new_obj, self.params_obj, outputdir=self.output_dir)
            output_files.append(filename)
            self.update_progress(files_to_read.index(analysis_file), len(files_to_read))

        self.display_analysis_files()
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
                                      key_list=list_of_param_keys)
        param_ui.wait_window()

        # Only update parameters if the user clicked 'okay' (didn't click cancel or close the window)
        if param_ui.return_code == 0:
            return_vals = param_ui.refresh_values()
            self.params_obj.set_params(return_vals)
            self.check_params()
            return True
        return False

    def on_button_oldplot_clicked(self):
        """
        Run old CIU plot method to generate a plot in the output directory
        :return: void (saves to output dir)
        """
        plot_keys = [x for x in self.params_obj.params_dict.keys() if 'ciuplot' in x]
        if self.run_param_ui('Plot parameters', plot_keys):
            # Determine if a file range has been specified
            files_to_read = self.check_file_range_entries()
            self.progress_started()
            for analysis_file in files_to_read:
                analysis_obj = load_analysis_obj(analysis_file)
                Original_CIU.ciu_plot(analysis_obj, self.params_obj, self.output_dir)
                self.update_progress(files_to_read.index(analysis_file), len(files_to_read))

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
            newfiles = filedialog.askopenfilenames(filetypes=[('_raw.csv', '_raw.csv')])
            if len(newfiles) == 0:
                return

            rmsd_print_list = ['File 1, File 2, RMSD (%)']
            std_file = files_to_read[0]
            index = 0
            for file in newfiles:
                compare_obj = load_analysis_obj(file)
                rmsd = Original_CIU.compare_basic_raw(std_file, compare_obj, self.params_obj, self.output_dir)
                printstring = '{},{},{:.2f}'.format(os.path.basename(std_file).rstrip('.ciu'),
                                                    os.path.basename(newfiles[index]).rstrip('.ciu'),
                                                    rmsd)
                rmsd_print_list.append(printstring)
                index += 1
                self.update_progress(newfiles.index(file), len(newfiles))

        if len(files_to_read) == 2:
            # Direct compare between two files
            ciu1 = load_analysis_obj(files_to_read[0])
            ciu2 = load_analysis_obj(files_to_read[1])
            Original_CIU.compare_basic_raw(ciu1, ciu2, self.params_obj, self.output_dir)

        elif len(files_to_read) > 2:
            batch_keys = [x for x in self.params_obj.params_dict.keys() if 'compare_batch' in x]
            self.run_param_ui('Plot parameters', batch_keys)

            rmsd_print_list = ['File 1, File 2, RMSD (%)']
            # batch compare - compare all against all.
            f1_index = 0
            for file in files_to_read:
                # don't compare the file against itself
                skip_index = files_to_read.index(file)
                f2_index = 0
                while f2_index < len(files_to_read):
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

                    ciu1 = load_analysis_obj(file)
                    ciu2 = load_analysis_obj(files_to_read[f2_index])
                    rmsd = Original_CIU.compare_basic_raw(ciu1, ciu2, self.params_obj, self.output_dir)
                    printstring = '{},{},{:.2f}'.format(os.path.basename(file).rstrip('.ciu'),
                                                        os.path.basename(files_to_read[f2_index]).rstrip('.ciu'),
                                                        rmsd)
                    rmsd_print_list.append(printstring)
                    f2_index += 1
                f1_index += 1
                self.update_progress(files_to_read.index(file), len(files_to_read))

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
        title = 'Smoothing Parameters'
        key_list = [x for x in self.params_obj.params_dict.keys() if 'smoothing' in x]
        # key_list = ['smoothing_1_method']
        if self.run_param_ui(title, key_list):
            files_to_read = self.check_file_range_entries()
            self.progress_started()
            new_file_list = []

            for file in files_to_read:
                analysis_obj = load_analysis_obj(file)
                analysis_obj = Raw_Processing.smooth_main(analysis_obj, self.params_obj)
                analysis_obj.refresh_data()
                filename = save_analysis_obj(analysis_obj, self.params_obj, outputdir=self.output_dir)
                new_file_list.append(filename)

                # also save _raw.csv output if desired
                if self.params_obj.output_1_save_csv:
                    save_path = file.rstrip('.ciu') + '_delta_raw.csv'
                    Original_CIU.write_ciu_csv(save_path, analysis_obj.ciu_data, analysis_obj.axes)
                self.update_progress(files_to_read.index(file), len(files_to_read))

            self.analysis_file_list = new_file_list
            self.display_analysis_files()
        self.progress_done()

        # param_ui = CIU_Params.ParamUI('Smoothing Parameters', self.params_obj, key_list)
        #
        # # wait for user to close the window
        # param_ui.wait_window()
        # # print(param_ui.return_code)
        #
        # # Only update parameters if the user clicked 'okay' (didn't click cancel or close the window)
        # if param_ui.return_code == 0:
        #     return_vals = param_ui.refresh_values()
        #     # print(return_vals)
        #     self.params_obj.set_params(return_vals)
        # # self.params_obj.print_params_to_console()
        # self.run()

    def on_button_oldavg_clicked(self):
        """
        Average several processed files into a replicate object and save it for further
        replicate processing methods
        :return:
        """
        # Determine if a file range has been specified
        files_to_read = self.check_file_range_entries()
        self.progress_started()

        analysis_obj_list = [load_analysis_obj(x) for x in files_to_read]
        averaged_obj = average_ciu(analysis_obj_list)
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
        self.progress_started()

        new_file_list = []
        for file in files_to_read:
            analysis_obj = load_analysis_obj(file)
            shifted_obj = Original_CIU.delta_dt(analysis_obj)
            newfile = save_analysis_obj(shifted_obj, analysis_obj.params, filename_append='_delta', outputdir=self.output_dir)
            new_file_list.append(newfile)
            # also save _raw.csv output if desired
            if self.params_obj.output_1_save_csv:
                save_path = file.rstrip('.ciu') + '_delta_raw.csv'
                Original_CIU.write_ciu_csv(save_path, shifted_obj.ciu_data, shifted_obj.axes)
            self.update_progress(files_to_read.index(file), len(files_to_read))

        self.analysis_file_list = new_file_list
        self.display_analysis_files()
        self.progress_done()

    def on_button_crop_clicked(self):
        """
        Open a dialog to ask user for crop inputs, then crop selected data accordingly.
        NOTE: preserves parameters from original object, as no changes have been made here
        :return: saves new .ciu files
        """
        # run the cropping UI
        if len(self.analysis_file_list) > 0:
            obj1 = load_analysis_obj(self.analysis_file_list[0])
            crop_vals = run_crop_ui(obj1.axes)

            if crop_vals is None:
                # user hit cancel, or no values were provided
                self.progress_done()
                return

            files_to_read = self.check_file_range_entries()
            self.progress_started()

            new_file_list = []
            for file in files_to_read:
                analysis_obj = load_analysis_obj(file)
                crop_obj = Raw_Processing.crop(analysis_obj, crop_vals)
                analysis_obj.refresh_data()
                crop_obj.crop_vals = crop_vals
                # newfile = save_analysis_obj(crop_obj, filename_append='_crop', outputdir=self.output_dir)
                newfile = save_analysis_obj(crop_obj, analysis_obj.params, outputdir=self.output_dir)
                new_file_list.append(newfile)
                # also save _raw.csv output if desired
                if self.params_obj.output_1_save_csv:
                    save_path = file.rstrip('.ciu') + '_crop_raw.csv'
                    Original_CIU.write_ciu_csv(save_path, crop_obj.ciu_data, crop_obj.axes)
                self.update_progress(files_to_read.index(file), len(files_to_read))

            self.analysis_file_list = new_file_list
            self.display_analysis_files()
        else:
            messagebox.showwarning('No Files Selected', 'Please select files before performing cropping/interpolation')
        self.progress_done()

    def on_button_gaussfit_clicked(self):
        """
        Run Gaussian fitting on the analysis object list (updating the objects and leaving
        the current list in place). Saves Gaussian diagnostics/info to file in self.output_dir
        :return: void
        """
        param_keys = [x for x in self.params_obj.params_dict.keys() if 'gaussian' in x]
        if self.run_param_ui('Gaussian Fitting Parameters', param_keys):
            # Determine if a file range has been specified
            files_to_read = self.check_file_range_entries()
            self.progress_started()

            new_file_list = []
            for file in files_to_read:
                analysis_obj = load_analysis_obj(file)
                analysis_obj = Gaussian_Fitting.gaussian_fit_ciu(analysis_obj, self.params_obj)

                filename = save_analysis_obj(analysis_obj, self.params_obj, outputdir=self.output_dir)
                new_file_list.append(filename)
                self.update_progress(files_to_read.index(file), len(files_to_read))

            self.display_analysis_files()
        self.progress_done()

    def on_button_ciu50_clicked(self):
        """
        Run feature detection workflow to generate CIU-50 (transition) outputs for selected
        files
        :return: void
        """
        param_keys = [x for x in self.params_obj.params_dict.keys() if 'ciu50' in x]
        if self.run_param_ui('CIU-50 Parameters', param_keys):
            files_to_read = self.check_file_range_entries()
            self.progress_started()

            new_file_list = []
            all_outputs = ''
            short_outputs = ''
            filename = ''
            combine_flag = False
            for file in files_to_read:
                # load file
                analysis_obj = load_analysis_obj(file)

                # run feature detection
                analysis_obj = Feature_Detection.ciu50_main(analysis_obj, self.params_obj, outputdir=self.output_dir)
                filename = save_analysis_obj(analysis_obj, self.params_obj, outputdir=self.output_dir)
                new_file_list.append(filename)

                if not self.params_obj.ciu50_cpt_2_combine_outputs:
                    analysis_obj.save_ciu50_outputs(self.output_dir, mode='changept')
                    analysis_obj.save_ciu50_short(self.output_dir)
                    combine_flag = False
                else:
                    file_string = os.path.basename(filename).rstrip('.ciu') + '\n'
                    all_outputs += file_string
                    all_outputs += analysis_obj.save_ciu50_outputs(self.output_dir, mode='changept', combine=True)
                    short_outputs += os.path.basename(filename).rstrip('.ciu')
                    short_outputs += analysis_obj.save_ciu50_short(self.output_dir, combine=True)
                    combine_flag = True
                self.update_progress(files_to_read.index(file), len(files_to_read))

            if combine_flag:
                outputpath = os.path.join(self.output_dir, os.path.basename(filename.rstrip('.ciu')) + '_ciu50s.csv')
                outputpath_short = os.path.join(self.output_dir, os.path.basename(filename.rstrip('.ciu')) + '_ciu50-short.csv')
                save_existing_output_string(outputpath, all_outputs)
                save_existing_output_string(outputpath_short, short_outputs)

            self.display_analysis_files()
        self.progress_done()

    def on_button_ciu50_gaussian_clicked(self):
        """
        Repeat of CIU50 button, but with Gaussian feature det instead of changepoint. Will clean up
        so there's only one method once final analysis method is decided on.
        :return: void
        """
        param_keys = [x for x in self.params_obj.params_dict.keys() if 'ciu50_gauss' in x]
        if self.run_param_ui('CIU-50 Parameters', param_keys):
            files_to_read = self.check_file_range_entries()
            self.progress_started()

            new_file_list = []
            all_outputs = ''
            short_outputs = ''
            filename = ''
            combine_flag = False
            for file in files_to_read:
                # load file
                analysis_obj = load_analysis_obj(file)

                # run feature detection
                analysis_obj = Feature_Detection.ciu50_gaussians(analysis_obj, self.params_obj, outputdir=self.output_dir)
                filename = save_analysis_obj(analysis_obj, self.params_obj, outputdir=self.output_dir)
                new_file_list.append(filename)

                if not self.params_obj.ciu50_gauss_2_combine_outputs:
                    analysis_obj.save_ciu50_outputs(self.output_dir, mode='gaussian')
                    analysis_obj.save_ciu50_short(self.output_dir)
                    combine_flag = False
                else:
                    file_string = os.path.basename(filename).rstrip('.ciu') + '\n'
                    all_outputs += file_string
                    all_outputs += analysis_obj.save_ciu50_outputs(self.output_dir, mode='gaussian', combine=True)
                    short_outputs += os.path.basename(filename).rstrip('.ciu')
                    short_outputs += analysis_obj.save_ciu50_short(self.output_dir, True)
                    combine_flag = True
                self.update_progress(files_to_read.index(file), len(files_to_read))

            if combine_flag:
                outputpath = os.path.join(self.output_dir, os.path.basename(filename.rstrip('.ciu')) + '_ciu50s.csv')
                outputpath_short = os.path.join(self.output_dir,
                                                os.path.basename(filename.rstrip('.ciu')) + '_ciu50-short.csv')
                save_existing_output_string(outputpath, all_outputs)
                save_existing_output_string(outputpath_short, short_outputs)

            self.display_analysis_files()
        self.progress_done()

    def on_button_feature_gaussian_clicked(self):
        """
        Run Gaussian-based feature detection routine. Ensure Gaussians have been fit previously.
        :return: void
        """
        param_keys = [x for x in self.params_obj.params_dict.keys() if 'feature_gauss' in x]
        if self.run_param_ui('Feature Detection Parameters from Gaussians', param_keys):
            files_to_read = self.check_file_range_entries()
            self.progress_started()
            new_file_list = []

            for file in files_to_read:
                # load file
                analysis_obj = load_analysis_obj(file)

                # check to make sure the analysis_obj has Gaussian data fitted
                if analysis_obj.gaussians is None:
                    messagebox.showwarning('Gaussian fitting required', 'Data in file {} does not have Gaussian fitting'
                                                                        'performed. Please run Gaussian fitting, then try '
                                                                        'again.')
                    break

                # If gaussian data exists, perform the analysis
                analysis_obj = Feature_Detection.feature_detect_gaussians(analysis_obj, self.params_obj)
                filename = save_analysis_obj(analysis_obj, self.params_obj, outputdir=self.output_dir)
                new_file_list.append(filename)

                Feature_Detection.plot_features(analysis_obj, self.params_obj, self.output_dir, mode='gaussian')
                outputpath = os.path.join(self.output_dir, os.path.basename(filename.rstrip('.ciu')) + '_features.csv')
                Feature_Detection.print_features_list(analysis_obj.features_gaussian, outputpath, mode='gaussian')
                self.update_progress(files_to_read.index(file), len(files_to_read))

            self.display_analysis_files()
        self.progress_done()

    def on_button_feature_changept_clicked(self):
        """
        Run simple flat feature detection routine. Does NOT require Gaussian fitting
        :return: void
        """
        param_keys = [x for x in self.params_obj.params_dict.keys() if 'feature_cpt' in x]
        if self.run_param_ui('Feature Detection Parameters', param_keys):
            files_to_read = self.check_file_range_entries()
            self.progress_started()
            new_file_list = []

            for file in files_to_read:
                # load file
                analysis_obj = load_analysis_obj(file)

                analysis_obj = Feature_Detection.feature_detect_changept(analysis_obj, self.params_obj)
                filename = save_analysis_obj(analysis_obj, self.params_obj, outputdir=self.output_dir)
                new_file_list.append(filename)

                Feature_Detection.plot_features(analysis_obj, self.params_obj, self.output_dir, mode='changept')
                outputpath = os.path.join(self.output_dir, os.path.basename(filename.rstrip('.ciu')) + '_features.csv')
                Feature_Detection.print_features_list(analysis_obj.features_changept, outputpath, mode='changept')
                self.update_progress(files_to_read.index(file), len(files_to_read))

            self.display_analysis_files()
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

        # self.progress_print_text('Feature evaluation in progress (may take some time)...', 50)
        # all_features = Classification.classify_1_eval_features(data_labels, obj_list_by_label, self.output_dir)
        # self.temp_feature_holder = [all_features, data_labels, obj_list_by_label]

        # Run the classification
        self.progress_print_text('LDA in progress (may take a few minutes)...', 50)
        scheme = Classification.main_build_classification(data_labels, obj_list_by_label, self.output_dir)
        Classification.save_scheme(scheme, self.output_dir)

        self.progress_done()

    # def on_button_classify_build_scheme_clicked(self):
    #     """
    #     Have the user select the number of features and/or minimum score to use in generation
    #     of a final scheme object to save for eventual unknown classification.
    #     :return: void
    #     """
    #     param_keys = [x for x in self.params_obj.params_dict.keys() if 'classif_feats' in x]
    #     if not self.run_param_ui('Feature Detection Parameters', param_keys):
    #         # user hit 'cancel' - don't perform the analysis
    #         return
    #
    #     if self.temp_feature_holder is None:
    #         messagebox.showwarning('Error', 'Features have not yet been evaluated; Please use the Eval Features'
    #                                         ' button before this one.')
    #         return
    #
    #     # Determine the selected features based on user input
    #     all_features = self.temp_feature_holder[0]
    #     selected_features = Classification.select_features(all_features, self.params_obj)
    #
    #     # build final scheme with selected features and save to file
    #     labels = self.temp_feature_holder[1]
    #     obj_list_by_list = self.temp_feature_holder[2]
    #     scheme = Classification.classify_2_build_scheme(selected_features, labels, obj_list_by_list, self.output_dir)
    #     Classification.save_scheme(scheme, self.output_dir)

    def on_button_classify_unknown_clicked(self):
        """
        Open filechooser to load classification scheme object (previously saved), then run
        classification on all loaded .ciu files against that scheme.
        :return: void
        """
        # TODO: add axis exact match checking against scheme

        scheme_file = filedialog.askopenfilename(filetypes=[('Classification File', '.clf')])
        if not scheme_file == '':
            scheme = Classification.load_scheme(scheme_file)

            files_to_read = self.check_file_range_entries()
            self.progress_started()
            new_file_list = []

            for file in files_to_read:
                # load file
                analysis_obj = load_analysis_obj(file)

                prediction_outputs = scheme.classify_unknown(analysis_obj, self.output_dir)
                analysis_obj.classif_predicted_outputs = prediction_outputs

                filename = save_analysis_obj(analysis_obj, params_obj=self.params_obj, outputdir=self.output_dir)
                new_file_list.append(filename)
                self.update_progress(files_to_read.index(file), len(files_to_read))

            self.display_analysis_files()
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
        self.builder.get_object('Entry_progress').insert(0, 'Processing...')
        self.builder.get_object('Entry_progress').config(state=tk.DISABLED)

        self.builder.get_object('Progressbar_main')['value'] = 1
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
        self.builder.get_object('Entry_progress').config(state=tk.NORMAL)
        self.builder.get_object('Entry_progress').delete(0, tk.END)
        self.builder.get_object('Entry_progress').insert(0, 'Done!')
        self.builder.get_object('Entry_progress').config(state=tk.DISABLED)

        self.builder.get_object('Progressbar_main')['value'] = 100
        # added to keep program from exiting when run from command line - not sure if there's a better fix
        self.run()


# ****** CIU Main I/O methods ******
def open_files(filetype):
    """
    Open a tkinter filedialog to choose files of the specified type
    :param filetype: filetype filter in form [(name, extension)]
    :return: list of selected files
    """
    files = filedialog.askopenfilenames(filetypes=filetype)
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
    :rtype: CIURaw
    :return: CIURaw object with raw data, filename, and axes
    """
    raw_obj = Raw_Processing.get_data(raw_file)
    return raw_obj


def process_raw_obj(raw_obj, params_obj):
    """
    Run all initial raw processing stages (data import, smoothing, interpolation, cropping)
    on a raw file using the parameters provided in a Parameters object. Returns a NEW
    analysis object with the processed data
    :param raw_obj: the CIURaw object containing the raw data to process
    :type raw_obj: CIURaw
    :param params_obj: Parameters object containing processing parameters
    :type params_obj: Parameters
    :rtype: CIUAnalysisObj
    :return: CIUAnalysisObj with processed data
    """
    # normalize data and save axes information
    norm_data = Raw_Processing.normalize_by_col(raw_obj.rawdata)
    axes = (raw_obj.dt_axis, raw_obj.cv_axis)

    # interpolate data if desired TODO: move to own button/remove from here
    if params_obj.interp_1_method:
        norm_data, axes = Raw_Processing.interpolate_cv(norm_data, axes, params_obj.interp_2_bins)

    # save a CIUAnalysisObj with the information above
    analysis_obj = CIUAnalysisObj(raw_obj, norm_data, axes)
    analysis_obj = Raw_Processing.smooth_main(analysis_obj, params_obj)

    # save parameters and return
    analysis_obj.params = params_obj
    return analysis_obj


def update_params_in_obj(analysis_obj, params_obj):
    """
    Save the provided parameters object over the analysis_obj's current parameters object
    :param analysis_obj: CIUAnalysisObj object
    :param params_obj: Parameters object
    :type params_obj: Parameters
    :rtype: CIUAnalysisObj
    :return: updated analysis_obj
    """
    analysis_obj.params = params_obj
    return analysis_obj


def reprocess_raw(analysis_obj, params_obj):
    """
    Wrapper method to differentiate between running raw processing methods (smoothing/etc)
    and generation of new CIUAnalysis objects. Updates ciu_data, axes, and parameters, but
    retains all other information in the analysis_obj
    :param analysis_obj: CIUAnalysisObj to reprocess
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object containing processing parameters
    :type params_obj: Parameters
    :rtype: CIUAnalysisObj
    :return: the existing analysis object with reprocessed ciu_data and axes
    """
    # ALTERNATIVE METHOD - rename to 'update params' and only change params obj...
    # TODO: move/remove this method?

    raw_obj = analysis_obj.raw_obj
    norm_data = Raw_Processing.normalize_by_col(raw_obj.rawdata)
    axes = (raw_obj.dt_axis, raw_obj.cv_axis)
    analysis_obj.ciu_data = norm_data
    analysis_obj.axes = axes

    # smooth
    analysis_obj = Raw_Processing.smooth_main(analysis_obj, params_obj)

    # interpolate data if desired
    if params_obj.interp_1_method:
        norm_data, axes = Raw_Processing.interpolate_cv(norm_data, axes, params_obj.interp_2_bins)

    # Smooth data if desired (column-by-column)
    # if params_obj.smoothing_1_method is not None:
    #     i = 0
    #     while i < params_obj.smoothing_3_iterations:
    #         norm_data = Raw_Processing.sav_gol_smooth(norm_data, params_obj.smoothing_2_window)
    #         i += 1

    # check for previously saved cropping and use those values if present
    if analysis_obj.crop_vals is not None:
        analysis_obj = Raw_Processing.crop(analysis_obj, analysis_obj.crop_vals)

    analysis_obj.params = params_obj
    return analysis_obj


def average_ciu(analysis_obj_list):
    """
    Generate and save replicate object (a CIUAnalysisObj with averaged ciu_data and a list
    of raw_objs) that can be used for further analysis
    :param analysis_obj_list: list of CIUAnalysisObj's to average
    :type analysis_obj_list: list[CIUAnalysisObj]
    :rtype: CIUAnalysisObj
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
    averaged_obj.filename = save_analysis_obj(averaged_obj, analysis_obj_list[0].params, filename_append='_Avg')

    # save averaged object to file and return it
    return averaged_obj


def save_analysis_obj(analysis_obj, params_obj, filename_append='', outputdir=None):
    """
    Pickle the CIUAnalysisObj for later retrieval
    :param analysis_obj: CIUAnalysisObj to save
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object to update/save with the analysis object
    :type params_obj: Parameters
    :param filename_append: Addtional filename to append to the raw_obj name (e.g. 'AVG')
    :param outputdir: (optional) directory in which to save. Default = raw file directory
    :return: full path to save location
    """
    file_extension = '.ciu'

    # update parameters
    analysis_obj.params = params_obj

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
    :rtype: CIUAnalysisObj
    :return: CIUAnalysisObj
    """
    with open(analysis_filename, 'rb') as analysis_file:
        analysis_obj = pickle.load(analysis_file)
    return analysis_obj


class CropUI(object):
    """
    Simple dialog with several fields build with Pygubu for inputting crop values
    """
    def __init__(self, init_axes):
        # Get crop input from the Crop_vals UI form
        self.builder = pygubu.Builder()

        # load the UI file
        self.builder.add_from_file(hard_crop_ui)
        # create widget using provided root (Tk) window
        self.mainwindow = self.builder.get_object('Crop_toplevel')
        self.mainwindow.protocol('WM_DELETE_WINDOW', self.on_close_window)

        callbacks = {
            'on_button_cancel_clicked': self.on_button_cancel_clicked,
            'on_button_crop_clicked': self.on_button_crop_clicked
        }
        self.builder.connect_callbacks(callbacks)
        self.crop_vals = []

        self.init_values(init_axes)

    def run(self):
        """
        Run the UI and return the output values
        :return: List of crop values [dt low, dt high, cv low, cv high]
        """
        self.mainwindow.mainloop()
        return self.return_values()

    def on_close_window(self):
        """
        Quit the mainwindow to stop the mainloop and get it to return
        the crop values, then destroy it to remove it from screen.
        :return: the provided crop values, or None if none were provided
        """
        self.mainwindow.quit()
        self.mainwindow.destroy()
        return self.return_values()

    def init_values(self, axes):
        """
        Display starting axes values in the object for the user to edit
        :param axes: list of [dt_axis, cv_axis] (each their own list)
        :return: void
        """
        dt_axis = axes[0]
        cv_axis = axes[1]

        self.builder.get_object('Entry_dt_start').delete(0, tk.END)
        self.builder.get_object('Entry_dt_start').insert(0, dt_axis[0])
        self.builder.get_object('Entry_dt_end').delete(0, tk.END)
        self.builder.get_object('Entry_dt_end').insert(0, dt_axis[len(dt_axis) - 1])

        self.builder.get_object('Entry_cv_start').delete(0, tk.END)
        self.builder.get_object('Entry_cv_start').insert(0, cv_axis[0])
        self.builder.get_object('Entry_cv_end').delete(0, tk.END)
        self.builder.get_object('Entry_cv_end').insert(0, cv_axis[len(cv_axis) - 1])

        self.builder.get_object('Entry_interp_dt').delete(0, tk.END)
        self.builder.get_object('Entry_interp_dt').insert(0, len(dt_axis))
        self.builder.get_object('Entry_interp_cv').delete(0, tk.END)
        self.builder.get_object('Entry_interp_cv').insert(0, len(cv_axis))

    def on_button_cancel_clicked(self):
        """
        Close the UI window and return None
        :return: will return None from on_close_window
        """
        return self.on_close_window()

    def on_button_crop_clicked(self):
        """
        Return the cropping values entered in the fields, after checking for invalid values.
        If any values are invalid, returns without setting the crop_vals list
        :return: sets crop_vals list [dt_low, dt_high, cv_low, cv_high] to the object's field for retrieval
        """
        dt_low = self.builder.get_object('Entry_dt_start').get()
        dt_high = self.builder.get_object('Entry_dt_end').get()
        cv_low = self.builder.get_object('Entry_cv_start').get()
        cv_high = self.builder.get_object('Entry_cv_end').get()
        cv_length = self.builder.get_object('Entry_interp_cv').get()
        dt_length = self.builder.get_object('Entry_interp_dt').get()
        try:
            dt_low = float(dt_low)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid starting DT: must be a number (decimals OK)')
            return -1
        try:
            dt_high = float(dt_high)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid ending DT: must be a number (decimals OK)')
            return -1
        try:
            cv_low = float(cv_low)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid starting CV: must be a number (decimals OK)')
            return -1
        try:
            cv_high = float(cv_high)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid ending CV: must be a number (decimals OK)')
            return -1
        try:
            cv_length = int(cv_length)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid length of CV axis: must be an integer (no decimals)')
            return -1
        try:
            dt_length = int(dt_length)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid length of DT axis: must be an integer (no decimals)')
            return -1

        # if all values are valid, save the list and close the window
        self.crop_vals = [dt_low, dt_high, cv_low, cv_high, dt_length, cv_length]
        return self.on_close_window()

    def return_values(self):
        """
        Returns the crop_values if they are specified
        :return: crop_vals list [dt_low, dt_high, cv_low, cv_high] or None if they're not set
        """
        if len(self.crop_vals) > 0:
            return self.crop_vals
        else:
            return None


def run_crop_ui(axes):
    """
    Run the crop UI widget from PyGuBu to get cropping values and return them
    :param axes: starting axes to display in the widget
    :return: List of crop values [dt_low, dt_high, cv_low, cv_high], or None if none were specified or
    the user hit the 'cancel' button
    """
    crop_app = CropUI(axes)
    crop_vals = crop_app.run()
    return crop_vals


if __name__ == '__main__':
    # Build the GUI and start its mainloop (run) method
    root = tk.Tk()
    root.withdraw()
    ciu_app = CIUSuite2(root)
    ciu_app.run()
    # test = run_crop_ui()
    # print(test)
