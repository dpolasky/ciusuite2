"""
Module for raw (instrument vendor) data import into CIUSuite2. Currently includes Waters
and Agilent handling based on TWIMExtract and MIDAC, respectively.
2/1/2018
DP
"""
from PyQt5 import QtWidgets
import sys
import subprocess
import os


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


if __name__ == '__main__':
    # testing
    # raw_dialog = FileDialog()
    raw_file_dirs = get_data()
    print(raw_file_dirs)
