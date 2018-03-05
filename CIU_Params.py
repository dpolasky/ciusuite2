"""
Module for Parameter object to hold parameter information in analysis objects and associated
parameter UI, etc
"""
import tkinter
import numpy as np
from tkinter import messagebox
from tkinter import ttk

# Dictionary containing descriptions of all parameters for display in menus/etc
hard_descripts_file = r"C:\CIUSuite2\CIU2_param_info.csv"

# PARAM_DESCRIPTIONS = {'smoothing_1_method': 'Method with which to smooth data. Savitsky-Golay or None',
#                       'smoothing_2_window': 'Size of the filter for the applied smoothing method. Default is 5',
#                       'smoothing_3_iterations': 'the number of times to apply the smoothing. Default is 1',
#
#                       'ciuplot_1_x_title': 'title to display on x - axis of CIU plot',
#                       'interpolation_bins': 'If provided, the data will be interpolated along the collision voltage axis to have the specified number of bins',
#                       'ciuplot_2_y_title': 'title to display on y - axis of CIU plot',
#                       'ciuplot_4_extension': 'file format in which to save CIU plot(acceptable values are .png, .pdf, .jpg)',
#                       'save_output_csv': 'Whether to write an _raw.csv output file with the processed data (True or False)',
#                       'ciuplot_3_plot_title': 'Optional title for the plot. If provided, will label the plot with the title.',
#
#                       'min_feature_length': 'The minimum number of points (collision voltages) across which a feature must be present to be counted as a real feature. Default = 3. Decrease to catch small features (if real) present at only a few collision voltages (e.g. quick transitions and/or large voltage steps).',
#                       'flat_width_tolerance': 'tolerance (in drift bins) around the most common apex drift bin for a CV column to allow for inclusion into a feature. Default = 4. Higher values allow more slanted features to be detected/allowed, but may result in poor fitting. Lower values result in strict filtering to very flat features only.',
#                       'combine_output_file': '',
#                       'cv_gap_tolerance': '',
#                       'ciu50_mode': '',
#
#                       'gaussian_int_threshold': 'Minimum intensity to allow a peak to be fit. Default: 0.1 (10%)',
#                       'gaussian_width_max': 'An optional filter to remove noise peaks from future analysis steps and plotting. Removes any peak from analysis with a width greater than the parameter. Default: 4',
#                       'gaussian_centroid_bound_filter': 'Optional filter to remove peaks from plotting and analysis outside provided DT bounds. format: [DT_lower_bound, DT_upper_bound]',
#                       'gaussian_centroid_bounds': 'Optional DT-axis bounds for displaying centroids in plot (only in plot, no change to analysis) format: [DT_lower_bound, DT_upper_bound]',
#                       'gaussian_width_fraction': 'Parameter describing approximate width ratio of peaks prior to fit. Default 0.01 (typically doesnt need to be adjusted by user)',
#                       'gaussian_convergence_r2': 'The minimum r squared value for the multi-peak fitting to accept. The program will attempt to fit a single peak to the data, and will add peaks until the convergence fit is reached. If overfitting is occurring (too many peaks), reduce this value. In underfitting is occurring, increase it.'
#                       }
#
# # Dictionary containing value requirements for all parameters as tuples of (type, value range/list])
# PARAM_REQS = {'smoothing_1_method': ('string', ['Savitsky-Golay', 'None']),
#               'smoothing_2_window': ('int', [0, np.inf]),
#               'smoothing_3_iterations': ('int', [0, np.inf]),
#               'ciuplot_1_x_title': ('anystring', []),
#               'ciuplot_2_y_title': ('anystring', []),
#               'ciuplot_4_extension': ('string', ['.png', '.pdf', '.jpg']),
#               'save_output_csv': ('bool', ['true', 'false']),
#               'ciuplot_3_plot_title': ('anystring', [])
#               }


def parse_param_descriptions(param_file):
    """
    Read in parameter descriptions and requirements from text (csv) file
    :param param_file: file to read (full system path)
    :return: dictionaries of 1) parameter display names, 2) parameter descriptions, 3) parameter
    requirements. All dicts will have keys corresponding to attributes of the Parameters object
    """
    names = {}
    descriptions = {}
    reqs = {}

    with open(param_file) as p_file:
        lines = list(p_file)
        for line in lines:
            # skip header
            if line.startswith('#'):
                continue

            line = line.rstrip('\n')
            # parse a parameter name and description from the line
            splits = line.split(',')
            key = splits[0].strip()
            names[key] = splits[2].strip()
            descriptions[key] = splits[7].strip()

            # parse parameter requirements
            param_type = splits[3].strip()
            if param_type == 'int':
                # parse lower and upper bounds
                if splits[4].strip() == 'ninf':
                    lower_bound = -np.inf
                else:
                    lower_bound = int(splits[4].strip())
                if splits[5].strip() == 'inf':
                    upper_bound = np.inf
                else:
                    upper_bound = int(splits[5].strip())
                reqs[key] = (param_type, [lower_bound, upper_bound])
            elif param_type == 'float':
                # parse lower and upper bounds
                if splits[4].strip() == 'ninf':
                    lower_bound = -np.inf
                else:
                    lower_bound = float(splits[4].strip())
                if splits[5].strip() == 'inf':
                    upper_bound = np.inf
                else:
                    upper_bound = float(splits[5].strip())
                reqs[key] = (param_type, [lower_bound, upper_bound])
            elif param_type == 'string' or param_type == 'bool':
                req_vals = [x.strip() for x in splits[6].strip().split(';')]
                # convert 'none' strings to actual Nonetype
                # for index, value in enumerate(req_vals):
                #     if value == 'none':
                #         req_vals[index] = None
                reqs[key] = (param_type, req_vals)
            elif param_type == 'anystring':
                reqs[key] = (param_type, [])
            else:
                print('invalid type, parsing failed for line: {}'.format(line))

    return names, descriptions, reqs


PAR_NAMES, PAR_DESCRIPTS, PAR_REQS = parse_param_descriptions(hard_descripts_file)


class Parameters(object):
    """
    Object to hold all parameters used in generation of a CIU_analysis object. Starts with
    nothing initialized and adds parameters over time.
    """

    def __init__(self):
        """
        Initialize an empty parameters object with all params set to None
        """
        self.params_dict = {}

        # Smoothing and processing parameters
        self.smoothing_1_method = None
        self.smoothing_2_window = None
        self.smoothing_3_iterations = None
        self.interp_1_method = None
        self.interp_2_bins = None

        # Plotting and saving output parameters
        self.ciuplot_4_extension = None
        self.ciuplot_3_plot_title = None
        self.ciuplot_1_x_title = None
        self.ciuplot_2_y_title = None
        # self.compare_plot_1_x_title = None
        # self.compare_plot_2_y_title = None
        # self.compare_plot_3_plot_title = None
        # self.compare_plot_4_extension = None
        self.output_1_save_csv = None
        self.compare_batch_1_both_dirs = None

        # Feature detection/CIU-50 parameters
        self.gaussian_1_convergence = None
        self.gaussian_2_int_threshold = None
        self.gaussian_3_width_max = None
        self.gaussian_4_save_diagnostics = None
        self.gaussian_5_width_fraction = None
        self.gaussian_6_min_peak_dist = None

        self.feature_cpt_min_length = None
        self.feature_cpt_width_tol = None
        self.feature_cpt_gap_tol = None

        self.feature_gauss_min_length = None
        self.feature_gauss_width_tol = None
        self.feature_gauss_gap_tol = None

        self.ciu50_cpt_mode = None
        self.ciu50_gauss_mode = None
        self.ciu50_cpt_2_combine_outputs = None
        self.ciu50_gauss_2_combine_outputs = None
        self.ciu50_3_interp_factor = None
        self.ciu50_4_trans_interp_factor = None
        self.ciu50_5_pad_transitions_cv = None

        self.classif_feats_1_num_feats = None
        self.classif_feats_2_min_score = None
        self.classif_2_score_dif_tol = None
        self.classif_1_training_size = None
        self.classif_3_mode = None

    def set_params(self, params_dict):
        """
        Set a series of parameters given a dictionary of (parameter name, value) pairs
        :param params_dict: Dictionary, key=param name, value=param value
        :return: void
        """
        for name, value in params_dict.items():
            try:
                # only set the attribute if it is present in the object - otherwise, raise attribute error
                self.__getattribute__(name)
                self.__setattr__(name, value)
            except AttributeError:
                # no such parameter
                print('No parameter name for param: ' + name)
                continue
        self.update_dict()

    def print_params_to_console(self):
        """
        Method to read all parameters out to the console (alphabetical order)
        :return: void
        """
        for paramkey, value in sorted(self.__dict__.items()):
            print('{}: {}'.format(paramkey, value))

    def compare(self, other_params):
        """
        Compare all parameters in this object to another parameters object. Returns True if
        all parameters are identical, False if not
        :param other_params: Parameters object
        :return: boolean
        """
        for param_key in self.params_dict.keys():
            if param_key == 'params_dict':
                continue
            if not self.params_dict[param_key] == other_params.params_dict[param_key]:
                return False
        return True

    def update_dict(self):
        """
        Build (or rebuild) a dictionary of all attributes contained in this object
        :return: void
        """
        for field in vars(self):
            value = self.__getattribute__(field)
            self.params_dict[field] = value


def parse_params_file_newcsv(params_file):
    """
    Parse a CIU2_param_info.csv file for all parameters. Returns a params_dict that can be used to
    set_params on a Parameters object
    :param params_file: File to parse (.csv), headers = '#'
    :return: params_dict: Dictionary, key=param name, value=param value
    """
    param_dict = {}
    try:
        with open(params_file, 'r') as pfile:
            lines = list(pfile)
            for line in lines:
                # skip headers and blank lines
                if line.startswith('#') or line.startswith('\n'):
                    continue
                splits = line.rstrip('\n').split(',')
                value = splits[1].strip()

                # catch 'None' values and convert to None
                if value == 'None':
                    param_dict[splits[0].strip()] = None
                else:
                    # try parsing numbers
                    try:
                        try:
                            param_dict[splits[0].strip()] = int(value)
                        except ValueError:
                            param_dict[splits[0].strip()] = float(value)
                    except ValueError:
                        # string value - try parsing booleans or leave as a string
                        if value.lower() in ['true', 't', 'yes', 'y']:
                            param_dict[splits[0].strip()] = True
                        elif value.lower() in ['false', 'f', 'no', 'n']:
                            param_dict[splits[0].strip()] = False
                        else:
                            param_dict[splits[0].strip()] = splits[1].strip()
        return param_dict
    except FileNotFoundError:
        print('params file not found!')


def parse_params_file(params_file):
    """
    Parse a CIU_params.txt file for all parameters. Returns a params_dict that can be used to
    set_params on a Parameters object
    :param params_file: File to parse (.txt), headers = '#'
    :return: params_dict: Dictionary, key=param name, value=param value
    """
    param_dict = {}
    try:
        with open(params_file, 'r') as pfile:
            lines = list(pfile)
            for line in lines:
                # skip headers and blank lines
                if line.startswith('#') or line.startswith('\n'):
                    continue
                splits = line.rstrip('\n').split(',')
                value = splits[1].strip()

                # catch 'None' values and convert to None
                if value == 'None':
                    param_dict[splits[0].strip()] = None
                else:
                    # try parsing numbers
                    try:
                        try:
                            param_dict[splits[0].strip()] = int(value)
                        except ValueError:
                            param_dict[splits[0].strip()] = float(value)
                    except ValueError:
                        # string value - try parsing booleans or leave as a string
                        if value.lower() in ['true', 't', 'yes', 'y']:
                            param_dict[splits[0].strip()] = True
                        elif value.lower() in ['false', 'f', 'no', 'n']:
                            param_dict[splits[0].strip()] = False
                        else:
                            param_dict[splits[0].strip()] = splits[1].strip()
        # parse crop_values into list
        # if param_dict['cropping_window_values'] is not None:
        #     # parse the list
        #     string_val = param_dict['cropping_window_values'].replace('[', '')
        #     string_val = string_val.replace(']', '')
        #     try:
        #         crop_list = [float(x) for x in string_val.split(',')]
        #         param_dict['cropping_window_values'] = crop_list
        #     except ValueError:
        #         print('Invalid cropping values: must be in form [float,float,float,float]')
        return param_dict
    except FileNotFoundError:
        print('params file not found!')


def parse_param_value(param_string):
    """
    Parsing hierarchy for strings being passed (or parsed) into parameters.
    :param param_string: stripped line being parsed
    :return:
    """
    if param_string == 'None':
        return None
    else:
        # try parsing numbers
        try:
            try:
                return int(param_string)
            except ValueError:
                return float(param_string)
        except ValueError:
            # string value - try parsing booleans or leave as a string
            if param_string.lower() in ['t', 'true', 'yes', 'y']:
                return True
            elif param_string.lower() in ['f', 'false', 'no', 'n']:
                return False
            else:
                return param_string.strip()


class ParamUI(tkinter.Toplevel):
    """
    Modular parameter editing UI popup class. Designed to take a list of parameters of arbitrary
    length and provide their names, current values, and definitions into a dialog for editing.
    """
    def __init__(self, section_name, params_obj, key_list):
        """
        Initialize a graphical menu with the parameters listed by key in the 'key_list' input.
        :param params_obj: Parameters object with parameter value information
        :type params_obj: Parameters
        :param key_list: list of keys (corresponding to keys in params.obj.params_dict and also the
        PARAM_DESCRIPTIONS dict) to display in the menu.
        """
        tkinter.Toplevel.__init__(self)
        self.title(section_name)
        self.return_code = -2

        # return values = dictionary by parameter of the entry
        self.return_vals = {}
        self.keys = key_list
        self.entry_vars = {}

        # display a label (name), entry (value), and label (description) for each parameter in the key list
        row = 0
        labels_frame = ttk.Frame(self, relief='raised', padding='2 2 2 2')
        labels_frame.grid(column=0, row=0)
        for param_key in key_list:
            entry_var = tkinter.StringVar()
            # Get the current parameter value to display, or 'None' if the value is not set
            param_val = params_obj.params_dict[param_key]
            if param_val is not None:
                # change booleans to display as strings rather than 1 or 0
                if isinstance(param_val, bool):
                    if param_val:
                        entry_var.set('true')
                    elif not param_val:
                        entry_var.set('false')
                else:
                    entry_var.set(params_obj.params_dict[param_key])
            else:
                entry_var.set('None')

            # display the parameter name, value, and description
            ttk.Label(labels_frame, text=PAR_NAMES[param_key]).grid(row=row, column=0, sticky='e')

            # display values the user can enter (int/float, strings) as an entry field
            param_type = PAR_REQS[param_key][0]
            if param_type == 'int' or param_type == 'float' or param_type == 'anystring':
                ttk.Entry(labels_frame, textvariable=entry_var, width=25).grid(row=row, column=1)
            else:
                # for fields where the user must choose, display a dropdown menu instead
                option_vals = [x for x in PAR_REQS[param_key][1]]
                # prevent the first option from not appearing in menu. Not sure why this duplication is necessary...
                option_vals.insert(0, option_vals[0])
                menu = ttk.OptionMenu(labels_frame, entry_var, *option_vals)
                menu.grid(row=row, column=1)
                menu['menu'].config(background='white')

            ttk.Label(labels_frame, text=PAR_DESCRIPTS[param_key], wraplength=500).grid(row=row, column=2, sticky='w')

            self.entry_vars[param_key] = entry_var
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
        :return: void
        """
        fail_params = []
        for param_key in self.keys:
            if not self.check_param_value(param_key):
                fail_params.append(param_key)

        if len(fail_params) == 0:
            # no params failed, so close window and update parameter values
            self.return_code = 0
            self.on_close_window()
        else:
            # some parameters failed. Tell the user which ones and keep the window open
            param_string = 'The following parameter(s) have inappropriate values:\n'
            for param in fail_params:
                if PAR_REQS[param][0] == 'string' or PAR_REQS[param][0] == 'bool':
                    # print acceptable values list for string/bool
                    vals_string = ', '.join(PAR_REQS[param][1])
                    param_string += '{}: value must be one of ({})\n'.format(PAR_NAMES[param], vals_string)
                else:
                    # print type only for float/int
                    param_string += '{}: value type must be {}, and within bounds\n'.format(PAR_NAMES[param], PAR_REQS[param][0])
            messagebox.showwarning(title='Parameter Error', message=param_string)
            return

    def check_param_value(self, param_key):
        """
        Check an individual parameter against its requirements
        :param param_key: key to parameter dictionary to be checked
        :return: True if the current value of the corresponding entry is valid, False if not
        """
        param_type = PAR_REQS[param_key][0]
        param_val_list = PAR_REQS[param_key][1]
        if param_type == 'int':
            # If the param is an int, the value must be within the values specified in the requirement tuple
            try:
                entered_val = int(self.entry_vars[param_key].get())
            except ValueError:
                return False
            return param_val_list[0] <= entered_val <= param_val_list[1]

        elif param_type == 'float':
            try:
                entered_val = float(self.entry_vars[param_key].get())
            except ValueError:
                return False
            return param_val_list[0] <= entered_val <= param_val_list[1]

        elif param_type == 'string' or param_type == 'bool':
            # check whether the string or "boolean" (really a string) value is in the allowed list
            entered_val = self.entry_vars[param_key].get().strip().lower()
            # if entered_val == 'none':
            #     return 'none' in param_val_list
            # else:
            check_val_list = [x.strip().lower() for x in param_val_list]    # check against lower case/stripped
            return entered_val in check_val_list

        elif param_type == 'anystring':
            # Things like titles can be any string - no checking required
            return True

    def cancel_button_click(self):
        """
        Close the window and return without updating any parameter values
        :return: void
        """
        self.return_code = -1
        self.on_close_window()

    def refresh_values(self):
        """
        Update the values for all parameters
        :return:
        """
        for key in self.entry_vars.keys():
            parsed_val = parse_param_value(self.entry_vars[key].get())
            try:
                if parsed_val.strip().lower() == 'none':
                    self.return_vals[key] = None
                else:
                    # not a 'none' string, set it as is
                    self.return_vals[key] = parsed_val
            except AttributeError:
                # not a string, return value as is
                self.return_vals[key] = parsed_val
        return self.return_vals

    def on_close_window(self):
        """
        Close the window
        :return: void
        """
        self.quit()
        self.destroy()


def test_param_ui():
    # testing
    myparams = Parameters()
    mydict = parse_params_file(r"C:\CIUSuite2\CIU_params.txt")
    myparams.set_params(mydict)
    key_list = ['smoothing_1_method', 'smoothing_2_window', 'smoothing_3_iterations']

    param_ui = ParamUI('test section', myparams, key_list)
    test_top = tkinter.Toplevel()
    test_top.title('test top')
    param_ui.wait_window()
    print('finished wait')


# testing
if __name__ == '__main__':
    # myparams = Parameters()
    # mydict = parse_params_file(r"C:\Users\dpolasky\Desktop\CIU_params.txt")
    # myparams.set_params(mydict)
    # myparams.print_params_to_console()
    root = tkinter.Tk()
    root.withdraw()

    test_param_ui()
