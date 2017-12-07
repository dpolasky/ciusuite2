"""
Dan Polasky
10/6/17
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
from Gaussian_Fitting import gaussfunc
import tkinter
from tkinter import filedialog
import pickle
from CIU_Params import Parameters


class CIUAnalysisObj(object):
    """
    Container for analysis/processed information from a CIU fingerprint. Requires a CIURaw object
    to start, additional fields added as data is processed.
    """
    def __init__(self, ciu_raw_obj, ciu_data, axes, gauss_params=None):
        """
        Initialize with raw data and axes. Allows addition of Gaussian fitting data later
        :param ciu_raw_obj: Object containing initial raw data, axes, and filepath of the analysis
        :param ciu_data: pre-processed data (smoothed/interpolated/cropped/etc) - can be modified repeatedly
        :param axes: modified axes corresponding to ciu_data (axes[0] = DT, axes[1] = CV)
        :param gauss_params: List of lists of parameters for gaussians fitted to each CV (column of ciu_data)
        params are [baseline height, amplitude, centroid, width]
        """
        self.raw_obj = ciu_raw_obj
        self.raw_obj_list = None    # used for replicates (averaged fingerprints) only
        self.ciu_data = ciu_data
        self.axes = axes            # convention: axis 0 = DT, axis 1 = CV
        self.params = Parameters()
        self.filename = None        # filename of .ciu file saved

        # Gaussian fitting parameters - not always initialized with the object
        self.gauss_params = None
        self.gauss_baselines = None
        self.gauss_amplitudes = None
        self.gauss_centroids = None
        self.gauss_widths = None
        self.gauss_fwhms = None
        self.gauss_adj_r2s = None
        self.gauss_fits = None
        self.gauss_covariances = None
        self.gauss_r2s = None
        self.gauss_resolutions = None
        self.gauss_fit_stats = None
        self.gauss_filt_params = None
        if gauss_params is not None:
            # initialize param lists if parameters are provided
            self.init_gauss_lists(gauss_params)

    def init_gauss_lists(self, gauss_params):
        """
        Initialize human readable lists of gaussian parameters (4 parameters per gaussian)
        :param gauss_params: list of gaussian parameters for each column of ciu array (each CV)
        :return: void
        """
        self.gauss_params = gauss_params
        self.gauss_baselines = [x[0::4] for x in gauss_params]
        self.gauss_amplitudes = [x[1::4] for x in gauss_params]
        self.gauss_centroids = [x[2::4] for x in gauss_params]
        self.gauss_widths = [x[3::4] for x in gauss_params]

    def save_gaussfits_pdf(self, outputpath):
        """
        Save a pdf containing an image of the data and gaussian fit for each column to pdf in outputpath.
        :param outputpath: directory in which to save output
        :return: void
        """
        # ensure gaussian data has been initialized
        if self.gauss_fits is None:
            print('No gaussian fit data in this object yet, returning')
            return

        print('Saving Gausfitdata_' + str(self.raw_obj.filename) + '_.pdf .....')
        pdf_fig = PdfPages(os.path.join(outputpath, 'Gausfitdata_' + str(self.raw_obj.filename) + '_.pdf'))
        intarray = np.swapaxes(self.ciu_data, 0, 1)
        for k in range(len(self.axes[1])):
            plt.figure()
            # plot the original raw data as a scatter plot
            plt.scatter(self.axes[0], intarray[k])
            # plot the fit data as a black dashed line
            plt.plot(self.axes[0], self.gauss_fits[k], ls='--', color='black')
            # plot centroids of gaussians to overlay
            index = 0
            for centroid in self.gauss_centroids[k]:
                # plot each fitted gaussian and centroid
                fit = gaussfunc(self.axes[0], 0, self.gauss_amplitudes[k][index],
                                self.gauss_centroids[k][index], self.gauss_widths[k][index])
                plt.plot(self.axes[0], fit)
                plt.plot(centroid, abs(self.gauss_amplitudes[k][index]), '+', color='red')
                index += 1
            plt.title('CV: {}, R2: {:.3f}, stderr: {:.4f}'.format(self.axes[1][k], self.gauss_r2s[k],
                                                                  self.gauss_fit_stats[k][5]))
            pdf_fig.savefig()
            plt.close()
        pdf_fig.close()
        print('Saving Gausfitdata_' + str(self.raw_obj.filename) + '.pdf DONE')

    def plot_centroids(self, outputpath, y_bounds=None):
        """
        Save a png image of the centroid DTs fit by gaussians. USES FILTERED peak data
        :param outputpath: directory in which to save output
        :param y_bounds: [lower bound, upper bound] to crop the plot to (in y-axis units, typically ms)
        :return: void
        """
        print('Saving TrapCVvsArrivtimecentroid ' + str(self.raw_obj.filename) + '_.png .....')

        # plot the centroid(s), including plotting multiple centroids at each voltage if present
        filt_centroids = [x[2::4] for x in self.gauss_filt_params]
        for x, y in zip(self.axes[1], filt_centroids):
            plt.scatter([x] * len(y), y)
        # plt.scatter(self.axes[1], self.gauss_centroids)
        plt.xlabel('Trap CV')
        plt.ylabel('ATD_centroid')
        if y_bounds is not None:
            plt.ylim(y_bounds)
        plt.title('Centroids filtered by peak width')
        plt.grid('on')
        plt.savefig(os.path.join(outputpath, str(self.raw_obj.filename.rstrip('_raw.csv')) + '_centroids.png'),
                    dpi=500)
        plt.close()
        print('Saving TrapCVvsArrivtimecentroid ' + str(self.raw_obj.filename) + '_.png DONE')

    def plot_fwhms(self, outputpath):
        """
        Save a png image of the FWHM (widths) fit by gaussians.
        :param outputpath: directory in which to save output
        :return: void
        """
        print('Saving TrapcCVvsFWHM_' + str(self.raw_obj.filename) + '_.png .....')
        for x, y in zip(self.axes[1], self.gauss_fwhms):
            plt.scatter([x] * len(y), y)
        # plt.scatter(self.axes[1], self.gauss_fwhms)
        plt.xlabel('Trap CV')
        plt.ylabel('ATD_FWHM')
        plt.grid('on')
        plt.savefig(os.path.join(outputpath, str(self.raw_obj.filename) + '_FWHM.png'), dpi=500)
        plt.close()
        print('Saving TrapCVvsFWHM_' + str(self.raw_obj.filename) + '_.png DONE')

    def save_gauss_params(self, outputpath):
        """
        Save all gaussian information to file
        :param outputpath: directory in which to save output
        :return: void
        """
        with open(os.path.join(outputpath, str(self.raw_obj.filename.rstrip('_raw.csv')) + '_GaussFits.csv'), 'w') as output:
            output.write('Filtered Gaussians\n')
            output.write('Trap CV,Baseline(y0),Amplitude,Centroid,Peak Width\n')
            index = 0
            while index < len(self.axes[1]):
                outputline = '{},'.format(self.axes[1][index])
                outputline += ','.join(['{:.2f}'.format(x) for x in self.gauss_filt_params[index]])
                output.write(outputline + '\n')
                index += 1

            index = 0
            output.write('All gaussians fit to data\n')
            output.write('Trap CV,R^2,Adj R^2,Baseline(y0),Amplitude,Centroid,Peak Width\n')
            while index < len(self.gauss_centroids):
                gauss_line = '{},{:.3f},{:.3f},'.format(self.axes[1][index], self.gauss_r2s[index], self.gauss_adj_r2s[index])
                gauss_line += ','.join(['{:.2f}'.format(x) for x in self.gauss_params[index]])
                # gauss_line += ','.join([str(x) for x in self.gauss_params[index]])
                output.write(gauss_line + '\n')
                index += 1


# testing
if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(filetypes=[('pickled gaussian files', '.pkl')])
    files = list(files)
    file_dir = os.path.dirname(files[0])

    for file in files:
        with open(file, 'rb') as first_file:
            ciu1 = pickle.load(first_file)
        ciu1.plot_centroids(file_dir, [10, 20])
        ciu1.save_gauss_params(file_dir)
