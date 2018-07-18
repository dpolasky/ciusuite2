"""
Module for raw (instrument vendor) data import into CIUSuite2. Currently includes Waters
and Agilent handling based on TWIMExtract and MIDAC, respectively.
2/1/2018
DP
"""
from PyQt5 import QtWidgets
import sys
import os
import numpy as np
import tkinter
from tkinter import messagebox
from tkinter import ttk


# File chooser for raw data, created after extensive searching on stack overflow
class FileDialog(QtWidgets.QFileDialog):
    """
    Generate a directory-only filechooser, as both Waters and Agilent raw data comes in folders
    rather than files
    """
    def __init__(self, *args):
        QtWidgets.QFileDialog.__init__(self, *args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.DirectoryOnly)

        self.tree = self.findChild(QtWidgets.QTreeView)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)


def get_data():
    """
    Run the QtWidget directory mode filedialog and return filelist generated
    :return: List of directories chosen
    """
    app = QtWidgets.QApplication(sys.argv)
    dialog = FileDialog()
    dialog.show()
    app.exec_()
    files = dialog.selectedFiles()
    return files


def check_data(file_list, accepted_endings):
    """
    Run checks to confirm that all files selected are either .raw or .d folders prior to continuing
    :param file_list: list of file path strings
    :param accepted_endings: list of acceptable file ending strings (e.g. '.raw')
    :return: True if data is valid, False if not
    """
    for file in file_list:
        splits = os.path.basename(file).split('.')
        file_ending = splits[len(splits) - 1]
        if file_ending not in accepted_endings:
            return False

    # no problems found, return True
    return True

# def run_extractor(input_path, output_path, mode_int, func_num=None, range_path=None, rule_bool=None, combine_bool=None):
#     """
#     Run the tool with specified arguments (after formatting them, combining to 1 string, and adding the tool dir)
#     :param input_path: Full path to the input .raw folder to extract from
#     :param output_path: Full path to directory in which to place output file(s)
#     :param mode_int: 0 = RT, 1 = DT, 2 = MZ. Dimension to save upon extraction
#     :param func_num: Function number of file to analyze. If not supplied, defaults to extracting all functions
#     :param range_path: OPTIONAL. Full path to the range file to use to extract. If not specified, full range used
#     :param rule_bool: OPTIONAL. 'true' or 'false'. If true, program will expect range_arg to point to a .rul file
#     :param combine_bool: OPTIONAL. 'true' or 'false'. If true, outputs from the same raw file will be combined
#     :return: none
#     """
#     # format arguments that are present. If range/rule/combine are not supplied, do not pass anything
#     input_path = '-i "' + input_path + '"'    # use quotes to allow spaces/etc in filenames
#     output_path = '-o "' + output_path + '"'
#     mode_int = '-m ' + str(mode_int)
#     if func_num is not None:
#         func_num = '-f ' + str(func_num)
#     else:
#         func_num = ''
#     if range_path is not None:
#         range_path = '-r "' + range_path + '"'
#     else:
#         range_path = ''
#     if rule_bool is not None:
#         rule_bool = '-rulemode ' + rule_bool
#     else:
#         rule_bool = ''
#     if combine_bool is not None:
#         combine_bool = '-combinemode ' + combine_bool
#     else:
#         combine_bool = ''
#
#     arg_list = [tool_arg, input_path, output_path, mode_int, func_num, range_path, rule_bool, combine_bool]
#     arg_fmt = ['{}'.format(x) for x in arg_list]
#     args = ' '.join(arg_fmt)
#     # print(args)
#     subprocess.run(args)


# test for Agilent runner stuff
# def run_agilent_extractor(extractor_path):
#     """
#     Run the Agilent extractor program (GUI mode) and return the folder path(s) with generated CIU files to be
#     assembled and loaded.
#     ** this part is actually super easy - the key is figuring out how to get the output from the extractor into
#     CIUSuite 2 and/or get a list of files written from the extractor to open with CIUSuite 2. **
#     *** USE AGILENT_EXT_RUNNER.PY AS A BASE FOR THIS - it has things set up perfectly for running command line,
#     and would be a good basis for running the GUI. Even if the GUI is run, the output will still need to be
#     edited to put the CV header in (for now), so doing any fancier doesn't make sense until that's fixed.
#     :param extractor_path: Full system path to the extractor tool
#     :return:
#     """
#
#     completed_proc = subprocess.run(extractor_path)
#
#     if completed_proc.returncode == 0:
#         # process finished successfully
#         print('yay!')


def read_agilent_and_correct(filename, cv_axis_to_use, overwrite=True):
    """
    Parse through the file and edit as needed. Saves a copy in the same directory as the input file,
    unless overwrite is True, in which case it replaces the input file with an updated one.
    :param filename: full path to file to edit
    :param cv_axis_to_use: list of collision voltage (or other activation values) to use. Must be same
    length as number of columns in the input file.
    :param overwrite: whether to overwrite the original file or save a copy with '_edit' appended
    :return: void
    """
    edited_lines = []

    # check how many 0 lines to remove
    rawdata = np.genfromtxt(filename, missing_values=[""], filling_values=[0], delimiter=",")
    dt_axis = rawdata[1:, 0]
    num_zeros = np.count_nonzero(dt_axis == 0)
    zeros_skipped_so_far = 0

    with open(filename) as original_file:
        # read through the file, saving all lines (except those we're skipping)
        for index, line in enumerate(original_file):
            splits = line.rstrip('\n').split(',')
            if index == 0:
                # save header information
                if not len(splits) - 1 == len(cv_axis_to_use):
                    print('CV axis provided is NOT the same length as the CV axis in file {}, skipping file'.format(os.path.basename(filename)))
                    return
                new_header_info = ','.join([str(x) for x in cv_axis_to_use])
                new_header = splits[0] + ',' + new_header_info + '\n'
                edited_lines.append(new_header)

            else:
                # all other lines: skip zeros if needed, otherwise, save
                if zeros_skipped_so_far < num_zeros - 1:
                    zeros_skipped_so_far += 1
                else:
                    # save this line
                    edited_lines.append(line)

    # save a copy of the file with the edited information
    if overwrite:
        new_filename = filename
    else:
        new_filename = filename.rstrip('_raw.csv') + '_edit_raw.csv'
    with open(new_filename, 'w') as new_file:
        for line in edited_lines:
            new_file.write(line)


def ask_input_cv_data(original_header):
    """
    Method to prompt the user to enter the CV axis data for an Agilent file into a dialog.
    Displays the original header values in a table for the user to fill out with appropriate CV values
    for each. Returns parsed values as a list.
    :param original_header: list of strings parsed from the original file header line.
    :return: list of CV values, success boolean
    """
    # root = tkinter.Tk()
    # root.withdraw()

    output_cvs = run_header_ui(original_header)
    if output_cvs is not None:
        return output_cvs, True
    else:
        return [], False

    # splits = input_string.split(',')
    # for split in splits:
    #     try:
    #         cv = float(split.strip())
    #         output_cvs.append(cv)
    #     except ValueError:
    #         messagebox.showerror('Error Parsing Value', 'Value {} could not be parsed to a number. Please remove any non-number characters and try again')
    #         return [], False
    #
    # return output_cvs, True


def get_header(filename):
    """
    Return the header of a _raw.csv file. SKIPS FIRST ENTRY, as that isn't part of the header for _raw.csv files
    :param filename: full path to file
    :return: list of strings parsed from the first line of the file
    """
    with open(filename, 'r') as myfile:
        line = myfile.readline()
        splits = line.rstrip('\n').split(',')
        return splits[1:]


def run_header_ui(original_header_list):
    """
    Run the HeaderUI graphical menu
    :param original_header_list: list of strings containing original header information
    :return: list of floats - return values from the HeaderUI, or None if the user canceled or something failed
    """
    header_ui = HeaderUI(original_header_list)

    # prevent users from hitting multiple windows simultaneously
    header_ui.grab_set()
    header_ui.wait_window()
    header_ui.grab_release()

    # Only update parameters if the user clicked 'okay' (didn't click cancel or close the window)
    if header_ui.return_code == 0:
        return header_ui.return_vals
    else:
        return None


class HeaderUI(tkinter.Toplevel):
    """
    Graphical menu for editing Agilent headers
    """

    def __init__(self, original_header_list):
        """
        Initialize the graphical menu with one line for each value in the original header
        :param original_header_list: list of strings containing original header information
        """
        tkinter.Toplevel.__init__(self)
        self.title('Enter the Activation Values')
        self.return_code = -2

        # return values = list of header values
        self.return_vals = []
        self.entry_vars = []
        self.labels = []

        # display a label (name) and entry (value) for each part of the original header
        row = 1
        labels_frame = ttk.Frame(self, relief='raised', padding='2 2 2 2')
        labels_frame.grid(column=0, row=0)
        ttk.Label(labels_frame, text='Enter the correct activation values for each file/segment in the CIU dataset').grid(row=0, column=0, sticky='e', columnspan=2)
        for header_string in original_header_list:
            entry_var = tkinter.StringVar()
            self.labels.append(header_string)
            # display the original header and an entry box for the new value
            ttk.Label(labels_frame, text=header_string).grid(row=row, column=0, sticky='e')
            ttk.Entry(labels_frame, textvariable=entry_var, width=25).grid(row=row, column=1)

            self.entry_vars.append(entry_var)
            row += 1

        # Finally, add 'okay' and 'cancel' buttons to the bottom of the dialog
        button_frame = ttk.Frame(self, padding='5 5 5 5')
        button_frame.grid(column=0, row=1)
        ttk.Button(button_frame, text='Cancel', command=self.cancel_button_click).grid(row=0, column=0, sticky='w')
        ttk.Label(button_frame, text='                           ').grid(row=0, column=1)
        ttk.Button(button_frame, text='OK', command=self.ok_button_click).grid(row=0, column=2, sticky='e')

    def ok_button_click(self):
        """
        When the user clicks 'OK', confirm that all values are valid and if so, close the window.
        If not, prompt for corrective action.
        All values are saved into self.return_vals to be retrieved in the run_header_ui method
        :return: void
        """
        fail_vals = []
        for index, entry_var in enumerate(self.entry_vars):
            entry_string = entry_var.get()
            try:
                self.return_vals.append(float(entry_string))
            except ValueError:
                fail_vals.append((self.labels[index], entry_string))

        if len(fail_vals) == 0:
            # no params failed, so close window and update parameter values
            self.return_code = 0
            self.on_close_window()
        else:
            # some parameters failed. Tell the user which ones and keep the window open
            error_string = 'The following entry(s) have inappropriate (non-number) values:\n'
            for fail_tup in fail_vals:
                error_string += 'entry {}: {}\n'.format(fail_tup[0], fail_tup[1])
            messagebox.showwarning(title='Entry Error', message=error_string)
            return

    def cancel_button_click(self):
        """
        Close the window and return without updating any parameter values
        :return: void
        """
        self.return_code = -1
        self.on_close_window()

    def on_close_window(self):
        """
        Close the window
        :return: void
        """
        self.quit()
        self.destroy()


if __name__ == '__main__':
    # testing
    # raw_dialog = FileDialog()
    raw_file_dirs = get_data()
    print(raw_file_dirs)

    #
    mypath = r"C:\Users\Dan-7000\Desktop\AgilentCIU_memfixed\release\MIDAC_CIU_Extractor.exe"
