"""
Module for Parameter object to hold parameter information in analysis objects and associated
parameter UI, etc
"""
import tkinter

# Dictionary containing descriptions of all parameters for display in menus/etc
PARAM_DESCRIPTIONS = {'smoothing_method': 'Method with which to smooth data. Savitsky-Golay or None',
                      'smoothing_window': 'Size of the filter for the applied smoothing method. Default is 5',
                      'smoothing_iterations': 'the number of times to apply the smoothing. Default is 1'}


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
        self.params_dict = None

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
        self.params_dict = params_dict

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

        # return values = list by parameter of the entry
        self.return_vals = []
        self.entry_vars = []

        # display a label (name), entry (value), and label (description) for each parameter in the key list
        row = 0
        for param_key in key_list:
            entry_var = tkinter.StringVar()

            name = param_key
            value = params_obj.params_dict[param_key]
            description = PARAM_DESCRIPTIONS[param_key]
            label = tkinter.Label(self, text=name).grid(row=row, column=0)

            entry = tkinter.Entry(self, textvariable=entry_var).grid(row=row, column=1)
            self.entry_vars.append(entry_var)

            label2 = tkinter.Label(self, text=description).grid(row=row, column=2)
            row += 1

    def refresh_values(self):
        """
        Update the values for all parameters
        :return:
        """
        for entry in self.entry_vars:
            self.return_vals.append(entry.get())
        return self.return_vals


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
