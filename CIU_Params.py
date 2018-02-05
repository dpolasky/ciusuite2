"""
Module for Parameter object to hold parameter information in analysis objects and associated
parameter UI, etc
"""
import tkinter
import numpy as np
from tkinter import messagebox

# Dictionary containing descriptions of all parameters for display in menus/etc
PARAM_DESCRIPTIONS = {'smoothing_method': 'Method with which to smooth data. Savitsky-Golay or None',
                      'smoothing_window': 'Size of the filter for the applied smoothing method. Default is 5',
                      'smoothing_iterations': 'the number of times to apply the smoothing. Default is 1',

                      'plot_x_title': 'title to display on x - axis of CIU plot',
                      'interpolation_bins': 'If provided, the data will be interpolated along the collision voltage axis to have the specified number of bins',
                      'plot_y_title': 'title to display on y - axis of CIU plot',
                      'plot_extension': 'file format in which to save CIU plot(acceptable values are .png, .pdf, .jpg)',
                      'save_output_csv': 'Whether to write an _raw.csv output file with the processed data (True or False)',
                      'output_title': 'Optional title for the plot. If provided, will label the plot with the title; otherwise, the name of the raw file is used for the title.',

                      'min_feature_length': 'The minimum number of points (collision voltages) across which a feature must be present to be counted as a real feature. Default = 3. Decrease to catch small features (if real) present at only a few collision voltages (e.g. quick transitions and/or large voltage steps).',
                      'flat_width_tolerance': 'tolerance (in drift bins) around the most common apex drift bin for a CV column to allow for inclusion into a feature. Default = 4. Higher values allow more slanted features to be detected/allowed, but may result in poor fitting. Lower values result in strict filtering to very flat features only.',
                      'combine_output_file': '',
                      'cv_gap_tolerance': '',
                      'ciu50_mode': '',

                      'gaussian_int_threshold': 'Minimum intensity to allow a peak to be fit. Default: 0.1 (10%)',
                      'gaussian_width_max': 'An optional filter to remove noise peaks from future analysis steps and plotting. Removes any peak from analysis with a width greater than the parameter. Default: 4',
                      'gaussian_centroid_bound_filter': 'Optional filter to remove peaks from plotting and analysis outside provided DT bounds. format: [DT_lower_bound, DT_upper_bound]',
                      'gaussian_centroid_plot_bounds': 'Optional DT-axis bounds for displaying centroids in plot (only in plot, no change to analysis) format: [DT_lower_bound, DT_upper_bound]',
                      'gaussian_width_fraction': 'Parameter describing approximate width ratio of peaks prior to fit. Default 0.01 (typically doesnt need to be adjusted by user)',
                      'gaussian_convergence_r2': 'The minimum r squared value for the multi-peak fitting to accept. The program will attempt to fit a single peak to the data, and will add peaks until the convergence fit is reached. If overfitting is occurring (too many peaks), reduce this value. In underfitting is occurring, increase it.'
                      }

# Dictionary containing value requirements for all parameters as tuples of (type, value range/list])
PARAM_REQS = {'smoothing_method': ('string', ['Savitsky-Golay', 'None']),
              'smoothing_window': ('int', [0, np.inf]),
              'smoothing_iterations': ('int', [0, np.inf])}


class Parameters(object):
    """
    Object to hold all parameters used in generation of a CIU_analysis object. Starts with
    nothing initialized and adds parameters over time.
    """

    def __init__(self):
        """
        Initialize an empty parameters object with all params set to None
        """
        # Smoothing and processing parameters
        self.smoothing_method = None
        self.smoothing_window = None
        self.smoothing_iterations = None
        self.cropping_window_values = None
        self.interpolation_bins = None

        # Gaussian fitting and filtering parameters
        self.gaussian_int_threshold = None
        # self.gaussian_min_spacing = None
        self.gaussian_width_max = None
        self.gaussian_centroid_bound_filter = None
        self.gaussian_centroid_plot_bounds = None
        self.gaussian_width_fraction = None
        self.gaussian_save_diagnostics = None
        self.gaussian_convergence_r2 = None

        # Plotting and saving output parameters
        self.plot_extension = None
        self.save_output_csv = None
        self.output_title = None
        self.plot_x_title = None
        self.plot_y_title = None

        # Feature detection/CIU-50 parameters
        self.min_feature_length = None
        self.flat_width_tolerance = None
        self.combine_output_file = None
        self.ciu50_mode = None
        self.cv_gap_tolerance = None
        self.params_dict = {}

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
                splits = line.rstrip('\n').split('=')
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
                        if value in ['True', 'true', 'yes', 'Yes', 'Y', 'y']:
                            param_dict[splits[0].strip()] = True
                        elif value in ['False', 'false', 'no', 'No', 'N', 'n']:
                            param_dict[splits[0].strip()] = False
                        else:
                            param_dict[splits[0].strip()] = splits[1].strip()
        # parse crop_values into list
        if param_dict['cropping_window_values'] is not None:
            # parse the list
            string_val = param_dict['cropping_window_values'].replace('[', '')
            string_val = string_val.replace(']', '')
            try:
                crop_list = [float(x) for x in string_val.split(',')]
                param_dict['cropping_window_values'] = crop_list
            except ValueError:
                print('Invalid cropping values: must be in form [float,float,float,float]')
        return param_dict
    except FileNotFoundError:
        print('params file not found!')


class ParamUI(tkinter.Toplevel):
    """
    Modular parameter editing UI popup class. Designed to take a list of parameters of arbitrary
    length and provide their names, current values, and definitions into a dialog for editing.
    """
    def __init__(self, section_name, params_obj, key_list):
        """
        Initialize a graphical menu with the parameters listed by key in the 'key_list' input.
        :param params_obj: Parameters object with parameter value information
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
        for param_key in key_list:
            entry_var = tkinter.StringVar()
            # Get the current parameter value to display, or 'None' if the value is not set
            if params_obj.params_dict[param_key] is not None:
                entry_var.set(params_obj.params_dict[param_key])
            else:
                entry_var.set('None')

            # display the parameter name, value, and description
            tkinter.Label(self, text=param_key).grid(row=row, column=0)
            tkinter.Entry(self, textvariable=entry_var).grid(row=row, column=1)
            tkinter.Label(self, text=PARAM_DESCRIPTIONS[param_key]).grid(row=row, column=2)

            self.entry_vars[param_key] = entry_var
            row += 1

        # Finally, add 'okay' and 'cancel' buttons to the bottom of the dialog
        tkinter.Button(self, text='Cancel', command=self.cancel_button_click).grid(row=row, column=0)
        tkinter.Button(self, text='OK', command=self.ok_button_click).grid(row=row, column=1)

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
                if PARAM_REQS[param][0] == 'string' or PARAM_REQS[param][0] == 'bool':
                    # print acceptable values list for string/bool
                    vals_string = ', '.join(PARAM_REQS[param][1])
                    param_string += '{}: value must be one of ({})\n'.format(param, vals_string)
                else:
                    # print type only for float/int
                    param_string += '{}: value type must be {}, and within bounds\n'.format(param, PARAM_REQS[param][0])
            messagebox.showwarning(title='Parameter Error', message=param_string)
            return

    def check_param_value(self, param_key):
        """
        Check an individual parameter against its requirements
        :param param_key: key to parameter dictionary to be checked
        :return: True if the current value of the corresponding entry is valid, False if not
        """
        param_type = PARAM_REQS[param_key][0]
        param_val_list = PARAM_REQS[param_key][1]
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
            return self.entry_vars[param_key].get() in param_val_list

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
            self.return_vals[key] = self.entry_vars[key].get()
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
    key_list = ['smoothing_method', 'smoothing_window', 'smoothing_iterations']

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
