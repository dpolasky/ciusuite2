"""
Module for classification schemes for CIU data groups
Authors: Dan Polasky, Sugyan Dixit
Date: 1/11/2018
"""
import time

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from matplotlib.backends.backend_pdf import PdfPages

# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.metrics import precision_score
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import f_classif, GenericUnivariateSelect
from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_validate
# from sklearn.metrics import roc_curve, auc
# from scipy import interp

from Gaussian_Fitting import Gaussian

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import itertools
# import sys
import multiprocessing
from typing import List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CIU_analysis_obj import CIUAnalysisObj
    from CIU_Params import Parameters
    from Feature_Detection import Feature


class ClassificationScheme(object):
    """
    Container for final classification information, to be saved and used in future for unknown
    analyses, etc.
    """
    def __init__(self):
        self.lda = None     # type: LinearDiscriminantAnalysis
        self.classifier = None
        self.classifier_type = None
        self.params = None
        self.final_axis_cropvals = None
        self.classif_mode = None
        self.classif_prec_score = None
        self.explained_variance_ratio = None

        self.selected_features = []     # type: List[CFeature]
        self.all_features = None    # type: List[CFeature]

        self.numeric_labels = None
        self.class_labels = None
        self.unique_labels = None
        self.transformed_test_data = None
        self.test_filenames = None
        self.input_feats = None

        # cross validation information
        self.crossval_test_score = None
        self.all_crossval_data = None

        self.num_gaussians = None

    def __str__(self):
        label_string = ','.join(self.unique_labels)
        return '<Classif_Scheme> type: {}, data: {}'.format(self.classifier_type, label_string)
    __repr__ = __str__

    def classify_unknown(self, unk_ciu_obj, params_obj, output_path, unk_label='Unknown'):
        """
        Classify a test dataset according to this classification scheme. Selects features from
        the test dataset (ciudata), classifies, and returns output and score metrics.
        :param unk_ciu_obj: CIUAnalysisObj containing the ciu_data from unknown to be fitted
        :type unk_ciu_obj: CIUAnalysisObj
        :param unk_label: label for the unknown data
        :param params_obj: parameters information
        :type params_obj: Parameters
        :param output_path: directory in which to save output
        :return: updated analysis object with prediction data saved
        :rtype: CIUAnalysisObj
        """
        # unk_ciudata = unk_ciu_obj.ciu_data
        # todo: FIX MAX NUM GAUSSIAN COMMUNICATION (seems okay?)
        unk_ciudata = get_classif_data(unk_ciu_obj, params_obj, ufs_mode=False, num_gauss_override=self.num_gaussians)

        # Assemble feature data for fitting
        unk_input_x, fake_labels, filenames, input_feats = arrange_data_for_lda([unk_ciudata], unk_label, self.selected_features, [unk_ciu_obj.short_filename], method=params_obj.classif_4_data_structure)

        # if params_obj.classif_3_mode == 'Gaussian':
        #     unk_input_x = np.asarray(unk_input_x).T

        # Fit/classify data according to scheme LDA and classifier
        unknown_transformed_lda = self.lda.transform(unk_input_x)
        pred_class_label = self.classifier.predict(unknown_transformed_lda)
        pred_probs_by_cv = self.classifier.predict_proba(unknown_transformed_lda)
        pred_probs_avg = np.average(pred_probs_by_cv, axis=0)

        # create plots and save information to object
        unk_ciu_obj.classif_predicted_label = pred_class_label
        unk_ciu_obj.classif_probs_by_cv = pred_probs_by_cv
        unk_ciu_obj.classif_probs_avg = pred_probs_avg
        unk_ciu_obj.classif_transformed_data = unknown_transformed_lda



        unknown_plot_info = [(unknown_transformed_lda, unk_ciu_obj.short_filename)]
        plot_classification_decision_regions(self, output_path, unknown_tups=unknown_plot_info, filename=unk_ciu_obj.short_filename)

        return unk_ciu_obj

    def plot_all_unknowns(self, unk_ciuobj_list, params_obj, output_path):
        """
        Same as classify unknown, except that all unknowns are plotted on a single output plot
        and labeled by filename
        :param unk_ciuobj_list: list of CIUAnalysis containers for each unknown - MUST have transformed data already set
        :type unk_ciuobj_list: list[CIUAnalysisObj]
        :param params_obj: parameters information
        :type params_obj: Parameters
        :param output_path: directory in which to save output
        :return: void
        """
        all_plot_tups = [(x.classif_transformed_data, x.short_filename) for x in unk_ciuobj_list]
        plot_classification_decision_regions(self, output_path, unknown_tups=all_plot_tups, filename='all')


def save_lda_output_unk(list_transformed_data, list_filenames, list_feats, output_path):
    outputname = 'output_lda_unk.csv'
    output_final = os.path.join(output_path, outputname)
    features = [x.cv for x in list_feats]
    with open(output_final, 'w') as outfile:
        num_lds = np.arange(1, len(list_transformed_data[0][0])+1)
        lineheader = 'filename, feats,' + ','.join(str(x) for x in num_lds)
        outfile.write(lineheader + '\n')
        for index, (transformed_data, fname) in enumerate(zip(list_transformed_data, list_filenames)):
            # num_lds = np.arange(1, len(transformed_data[0])+1)
            # lineheader = 'filename, feats,' + ','.join(str(x) for x in num_lds)
            # outfile.write(lineheader+'\n')
            for ind in range(len(transformed_data[:, 0])):
                feats = str(features[ind])
                joined_lds = ','.join([str(x) for x in transformed_data[ind]])
                line1 = '{}, {}, {}, \n'.format(fname, feats, joined_lds)
                outfile.write(line1)
        outfile.close()


def save_lda_output(transformed_data, filenames, input_feats, output_path, explained_variance_ratio=None):
    """
    Save csv output from LDA, including transformed test data prediction accuracy scores
    :param output_path: directory in which to save output
    :return: void
    """
    outputname = 'output_lda.csv'
    feats = input_feats
    output_final = os.path.join(output_path, outputname)
    with open(output_final, 'w') as outfile:
        num_lds = np.arange(1, len(transformed_data[0]) + 1)
        try:
            lineheader = 'filename, feats,' + ','.join(str(x) for x in num_lds)
            outfile.write(lineheader + '\n')
            # OLD WAY - multiple features/probabilities per class
            for index in range(len(transformed_data[:, 0])):
                fnames = str(filenames[index])
                features = str(feats[index])
                joined_lds = ','.join([str(x) for x in transformed_data[index]])
                line1 = '{}, {}, {}, \n'.format(fnames, features, joined_lds)
                outfile.write(line1)
            # line2 = 'Explained_variance_ratio\n'
            if explained_variance_ratio is not None:
                joined_exp_var = ','.join([str(x) for x in explained_variance_ratio])
                line2 = 'Explained_variance_ratio, {}, {},\n'.format(' ', joined_exp_var)
                outfile.write(line2)
        except IndexError:
            lineheader = 'filename,'+','.join(str(x) for x in num_lds)
            outfile.write(lineheader + '\n')
            for index in range(len(transformed_data[:, 0])):
                # NEW WAY - only one probability per class (no features)
                fnames = str(filenames[index])
                joined_lds = ','.join([str(x) for x in transformed_data[index]])
                outfile.write('{}, {}, \n'.format(fnames, joined_lds))
            if explained_variance_ratio is not None:
                joined_exp_var = ','.join([str(x) for x in explained_variance_ratio])
                line2 = 'Explained_variance_ratio, {},\n'.format(joined_exp_var)
                outfile.write(line2)
        outfile.close()



def save_predictions(list_of_analysis_objs, params_obj, features_list, class_labels, output_path):
    """

    :param list_of_analysis_objs: list of CIUAnalysis containers with unknown data - MUST have transformed data already set
    :type list_of_analysis_objs: list[CIUAnalysisObj]
    :param params_obj: Parameters object with param information
    :type params_obj: Parameters
    :param features_list: list of selected Features
    :type features_list: list[CFeature]
    :param class_labels: list of labels for each class
    :param output_path: directory in which to save output
    :return: void
    """
    outputname = 'All_Unknowns_classif.csv'
    output_final = os.path.join(output_path, outputname)
    with open(output_final, 'w') as outfile:
        header_labels = ','.join(['Prob for {}'.format(x) for x in class_labels])
        if params_obj.classif_4_data_structure == 'flat':
            header = 'File,Predicted Class,{}\n'.format(header_labels)
        else:
            header = 'File,Features,Predicted Class,{}\n'.format(header_labels)
        outfile.write(header)

        for analysis_obj in list_of_analysis_objs:
            cvs = [x.cv for x in features_list]
            predict_class = analysis_obj.classif_predicted_label
            predict_prob_feat = analysis_obj.classif_probs_by_cv
            # OLD WAY
            if params_obj.classif_4_data_structure == 'stacked':
                # For feature-by-feature method, count the most common classification for this unknown (statistical mode)
                counts = np.bincount(predict_class)
                class_mode = np.argmax(counts)
                main_lines = []
                for index, cv in enumerate(cvs):
                    joined_probs = ','.join([str(x) for x in predict_prob_feat[index]])
                    line = '{},{},{},{},\n'.format(analysis_obj.short_filename, cv, predict_class[index], joined_probs)
                    main_lines.append(line)
                probs = ','.join(str(x) for x in analysis_obj.classif_probs_avg)
                line2 = '{},{},{},{}, \n'.format(analysis_obj.short_filename, 'Combined', class_mode, probs)

                # write at the end to allow type checking to finish
                for line in main_lines:
                    outfile.write(line)
                outfile.write(line2)
            elif params_obj.classif_4_data_structure == 'flat':
                # NEW WAY
                probs = ','.join(str(x) for x in analysis_obj.classif_probs_avg)
                line2 = '{},{},{}, \n'.format(analysis_obj.short_filename, analysis_obj.classif_predicted_label[0], probs)
                outfile.write(line2)


def get_unique_labels(label_list):
    """
    Return a list of unique labels (i.e. without duplicates) from the provided label list
    :param label_list: list of labels (strings)
    :return: list of unique labels (strings)
    """
    unique_labels = []
    for label in label_list:
        if label not in unique_labels:
            unique_labels.append(label)
    return unique_labels


class CFeature(object):
    """
    Container for classification feature information in feature selection.
    """
    def __init__(self, cv, cv_index, mean_score, std_dev_score):
        self.cv = cv
        self.cv_index = cv_index
        # self.score_list = score_list
        self.mean_score = mean_score
        self.std_dev_score = std_dev_score

    def __str__(self):
        return '<CFeature> cv: {}, score: {:.2f}'.format(self.cv, self.mean_score)
    __repr__ = __str__


class CrossValProduct(object):
    """
    Method to generate combinations of training and test datasets and run LDA + classification to generate training
    score and test score on selected features
    """
    def __init__(self, data_list_by_label, label_list, training_size, features):
        """
        Initialize a CrossValProduct method with data, label, training size, and features
        :param data_list_by_label: ciu_data list for all classes, sorted by class label
        :param label_list: label list for data
        :param training_size: num of ciu data in each class to be used as training set
        :param features: cv features
        """
        self.data_list_by_label = data_list_by_label
        self.label_list = label_list
        self.training_size = training_size
        self.features = features

        # assemble training and test data and label
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

        # results info
        self.train_scores = []
        self.train_scores_mean = None
        self.train_scores_std = None
        self.test_scores = []
        self.test_scores_mean = None
        self.test_scores_std = None
        self.probabs = []

        # assembled data
        self.all_class_combination_lists = None
        self.all_products = None

    def assemble_class_combinations(self, params_obj):
        """
        Generate training and test datasets with labels with the given training size
        :param params_obj: parameters information
        :type params_obj: Parameters
        :return: List of class combination lists for all classes. Each class combo list is a list of DataCombination
        objects. Intended to go directly into itertools.product
        :rtype: list[list[DataCombination]]
        """
        shaped_label_list = []
        for index, label in enumerate(self.label_list):
            shaped_label_list.append([label for _ in range(len(self.data_list_by_label[index]))])

        # generate all combinations of each class input datalist (products)
        all_class_combo_lists = []
        for class_index, class_ciu_list in enumerate(self.data_list_by_label):
            # create all combinations of specified training/test sizes from this class's data
            class_combo_list = []
            # class_ciu_data_list = [x.ciu_data for x in class_ciu_list]

            # training_size = len(class_ciu_list) - 1   # should be self.training_size

            for training_data_tuple, training_label_tuple in zip(itertools.combinations(class_ciu_list, self.training_size),
                                                                 itertools.combinations(shaped_label_list[class_index], self.training_size)):
                # training_data_list = [x.ciu_data for x in training_data_tuple]
                training_data_list = [get_classif_data(x, params_obj) for x in training_data_tuple]

                test_data_list = [x for x in class_ciu_list if x not in training_data_tuple]
                test_data_list = [get_classif_data(x, params_obj) for x in test_data_list]
                # [self.test_labels_tup[test_index] for _ in range(len((test_data[0])))]
                test_label_list = [shaped_label_list[class_index][0] for _ in range(len(test_data_list))]

                # create Train/Test DataProduct for this combination
                current_combo = DataCombination(training_data_list, training_label_tuple, test_data_list, test_label_list)
                current_combo.prepare_data(self.features, params_obj)
                class_combo_list.append(current_combo)

            all_class_combo_lists.append(class_combo_list)
        return all_class_combo_lists


def assemble_products(all_class_combination_lists):
    """
    :param all_class_combination_lists:
    Assemble the products of all class combinations
    :return:
    """
    probs = []
    train_scores, test_scores = [], []
    for combo_tuple in itertools.product(*all_class_combination_lists):
        print('...', end="")
        # stack training and test data and labels
        stacked_train_data, stacked_train_labels = [], []
        stacked_test_data, stacked_test_labels = [], []
        for combo in combo_tuple:
            stacked_train_data.append(combo.training_data_final)
            stacked_train_labels.append(combo.training_labels_string)
            stacked_test_data.append(combo.test_data_final)
            stacked_test_labels.append(combo.test_labels_string)

        final_train_data, final_train_labels = [], []
        for rep_index, replicate_data_list in enumerate(stacked_train_data):
            for feature_data in replicate_data_list:
                final_train_data.append(feature_data)
            final_train_labels = np.concatenate((final_train_labels, stacked_train_labels[rep_index]))
        final_train_data = np.asarray(final_train_data)
        final_train_labels = np.asarray(final_train_labels)

        final_test_data, final_test_labels = [], []
        for rep_index, replicate_data_list in enumerate(stacked_test_data):
            for feature_data in replicate_data_list:
                final_test_data.append(feature_data)
            final_test_labels = np.concatenate((final_test_labels, stacked_test_labels[rep_index]))
        final_test_data = np.asarray(final_test_data)
        final_test_labels = np.asarray(final_test_labels)

        # stacked_train_data = np.vstack(np.vstack(x for x in stacked_train_data))
        # stacked_train_labels = np.concatenate(np.vstack(stacked_train_labels))
        # stacked_test_data = np.vstack(np.vstack(np.vstack(x for x in stacked_test_data)))
        # stacked_test_labels = np.concatenate(np.vstack(stacked_test_labels))

        # replacing 'stacked' with 'final' in all lines below
        enc = LabelEncoder()
        enc.fit(final_train_labels)

        numeric_label_train = enc.transform(final_train_labels) + 1
        numeric_label_test = enc.transform(final_test_labels) + 1

        train_score, test_score = lda_clf_pipeline(final_train_data, numeric_label_train,
                                                   final_test_data, numeric_label_test)
        train_scores.append(train_score)
        test_scores.append(test_score)

    train_scores_mean = np.mean(train_scores)
    train_scores_std = np.std(train_scores)
    test_scores_mean = np.mean(test_scores)
    test_scores_std = np.std(test_scores)
    probs_mean = np.mean(probs)
    probs_std = np.std(probs)
    return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, probs_mean, probs_std


class DataCombination(object):
    """
    Container object for holding training and test data for a single way of combining the datasets in a class.
    Training and test datasets can be of arbitrary size. Holds corresponding label arrays of same shape as
    train/test data.
    """
    def __init__(self, train_data, train_labels, test_data, test_labels):
        """

        :param train_data: n-length tuple containing ciu datasets for training
        :param train_labels: same shape tuple with labels
        :param test_data: n-length tuple containing ciu datasets for testing
        :param test_labels: same shape tuple with labels
        """
        self.training_data_tup = train_data
        self.training_labels_tup = train_labels
        self.test_data_tup = test_data
        self.test_labels_tup = test_labels

        self.training_data_final = []
        self.training_labels_string = []
        self.test_data_final = []
        self.test_labels_string = []

    def prepare_data(self, features_list, params_obj):
        """
        Assemble concatenated data and label arrays for the specified slices of the input data (CV columns/features)
        :param features_list: list of selected Features with cv data
        :param params_obj: Parameters object with settings information
        :type params_obj: Parameters
        :return: void
        """
        train_data_final, train_labels_final, empty_filenames, final_cvs = arrange_data_for_lda(self.training_data_tup, self.training_labels_tup, features_list, method=params_obj.classif_4_data_structure)
        test_data_final, test_labels_final, empty_filenames2, final_cvs = arrange_data_for_lda(self.test_data_tup, self.test_labels_tup, features_list, method=params_obj.classif_4_data_structure)

        self.training_data_final = train_data_final
        self.training_labels_string = train_labels_final
        self.test_data_final = test_data_final
        self.test_labels_string = test_labels_final


class DataProduct(object):
    """
    Container for label and product information for data combinations used in feature selection.
    """
    def __init__(self, data_list, label_list):
        """
        Initialize a new DataProduct with associated label and data
        :param data_list: list of ciu_data matrices (normalized)
        :param label_list: list of class labels (same length as data_list)
        """
        self.data_input = data_list
        self.labels_input = label_list

        self.combined_data = None
        self.combined_label_arr = None
        self.numeric_label_arr = None

        # results information
        self.fit_scores = None
        self.fit_pvalues = None
        self.fit_sc = None

        self.train_scores = None
        self.test_scores = None

        # run data preparation
        self.prepare_data()

    def prepare_data(self):
        """
        Manipulate data into concatenated arrays suitable for input into feature selection algorithm.
        :return: void
        """
        self.combined_data = np.concatenate(self.data_input)

        label_array_list = []
        index = 0
        for label in self.labels_input:
            label_array = [label for _ in range(len(self.data_input[index]))]
            label_array_list.append(label_array)
            index += 1
        self.combined_label_arr = np.concatenate(label_array_list)

        # use the label encoder to generate a numeric label list (could just do this manually??)
        # output is just the class numbers in an array the shape of the input
        encoder = LabelEncoder()
        label_list = encoder.fit(self.combined_label_arr)
        self.numeric_label_arr = label_list.transform(self.combined_label_arr) + 1

    def __str__(self):
        label_string = ','.join(self.labels_input)
        return '<DataProduct> labels: {}'.format(label_string)
    __repr__ = __str__


def prep_gaussfeats_for_classif(features_list, analysis_obj):
    """
    Assemble a Gaussian-list-by-CV list from input features data. Fills any gaps between and within
    features with Gaussian data from the filtered_gaussians list and assembles a complete list of
    Gaussians by CV.
    :param features_list: list of Features
    :type features_list: list[Feature]
    :param analysis_obj: CIUAnalysisObj with gaussian fitting and gaussian feature detect performed
    :type analysis_obj: CIUAnalysisObj
    :return: List of (Gaussian lists) sorted by CV
    :rtype: list[list[Gaussian]] asdf
    """
    # make an empty list for Gaussians at each CV
    final_gaussian_lists = [[] for _ in analysis_obj.axes[1]]

    features_list = close_feature_gaps(features_list, analysis_obj.axes[1])

    # iterate over features, filling any gaps within the feature and entering Gaussians into the final list
    for feature in features_list:
        # determine if the feature contains gaps
        gaussian_cvs = [gaussian.cv for gaussian in feature.gaussians]

        for cv in feature.cvs:
            # append Gaussian(s) at this CV to the final list
            cv_index = np.where(analysis_obj.axes[1] == cv)[0][0]
            this_cv_gaussian = [x for x in feature.gaussians if x.cv == cv]
            final_gaussian_lists[cv_index].extend(this_cv_gaussian)

            if cv not in gaussian_cvs:
                # a gap is present within this feature- create a Gaussian at median centroid/values to fill it
                new_gaussian = Gaussian(baseline=0,
                                        amplitude=np.median([x.amplitude for x in feature.gaussians]),
                                        centroid=feature.gauss_median_centroid,
                                        width=np.median([x.width for x in feature.gaussians]),
                                        collision_voltage=cv,
                                        pcov=None)
                final_gaussian_lists[cv_index].append(new_gaussian)

    # Finally, check if all CVs have been covered by features. If not, add highest amplitude Gaussian from non-feature list
    for cv_index, cv in enumerate(analysis_obj.axes[1]):
        if len(final_gaussian_lists[cv_index]) == 0:
            # no Gaussians have been added here yet. Include highest amplitude one from filtered_gaussians
            cv_gaussians_from_obj = analysis_obj.filtered_gaussians[cv_index]
            sorted_by_amp = sorted(cv_gaussians_from_obj, key=lambda x: x.amplitude)
            try:
                final_gaussian_lists[cv_index].append(sorted_by_amp[0])
            except IndexError:
                # no Gaussians found at this CV in the original fitting - leave empty
                continue

    analysis_obj.classif_gaussfeats = final_gaussian_lists
    return final_gaussian_lists


def close_feature_gaps(features_list, cv_axis):
    """
    Check all features for gaps in their CV lists, and fill in the gaps if any exist by inserting
    appropriate CV values
    :param features_list: list of Features
    :type features_list list[Feature]
    :param cv_axis: analysis_obj.axes[1]
    :return: updated features list with gaps closed (in feature.cvs ONLY)
    :rtype: list[Feature]
    """
    cv_step = cv_axis[1] - cv_axis[0]

    for feature in features_list:
        for index, current_cv in enumerate(feature.cvs):
            try:
                next_cv = feature.cvs[index + 1]
            except IndexError:
                # reached the end, ignore
                continue
            while not (next_cv - current_cv) == cv_step:
                # a gap is present - insert the next value to fill it
                correct_next = current_cv + cv_step
                feature.cvs.insert(index + 1, correct_next)
                next_cv = feature.cvs[index + 1]
    return features_list


def get_classif_data(analysis_obj, params_obj, ufs_mode=False, num_gauss_override=None):
    """
    Initialize a classification data matrix in each analysis object in the lists according to the
    classification mode specified in the parameters object. In All_Data mode, this is simply the
    ciu_data matrix. In Gaussian mode, it will be Gaussian information from the object's fitted Gaussian
    lists.
    :param analysis_obj: analysis objects
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :param ufs_mode: boolean, True if using for UFS (feature selection), which requires only centroids from gaussian fitting
    :param num_gauss_override: MUST be provided for Unknown data fitting (in Gaussian mode ONLY) - the number of gaussians in the scheme being used
    :return: classification data matrix
    """
    classif_data = None

    if params_obj.classif_3_mode == 'All_Data':
        classif_data = analysis_obj.ciu_data

    elif params_obj.classif_3_mode == 'Gaussian':
        classif_data = []

        # for unknown data, num gaussians is provided (use it); for building scheme, num gaussians comes from params object (as a convenient save location)
        if num_gauss_override is not None:
            max_num_gaussians = num_gauss_override
        else:
            max_num_gaussians = params_obj.silent_clf_4_num_gauss

        # use Gaussian features if available, otherwise just all filtered Gaussians
        if analysis_obj.features_gaussian is not None:
            if analysis_obj.classif_gaussfeats is None:
                gaussian_list_by_cv = prep_gaussfeats_for_classif(analysis_obj.features_gaussian, analysis_obj)
            else:
                gaussian_list_by_cv = analysis_obj.classif_gaussfeats
        else:
            gaussian_list_by_cv = analysis_obj.filtered_gaussians

        if not ufs_mode:
            # assemble matrix of gaussian data
            for gaussian_list in gaussian_list_by_cv:
                attributes = ['cent', 'width', 'amp']
                num_attributes = len(attributes)
                attribute_list = np.zeros(max_num_gaussians * num_attributes)

                attribute_index = 0
                if len(gaussian_list) == 0:
                    continue
                for gaussian in gaussian_list:
                    attribute_list[attribute_index] = gaussian.centroid
                    attribute_index += 1
                    attribute_list[attribute_index] = gaussian.width
                    attribute_index += 1
                    attribute_list[attribute_index] = gaussian.amplitude
                    attribute_index += 1

                classif_data.append(attribute_list)
            classif_data = np.asarray(classif_data).T
        else:
            # for UFS, only use centroids
            for gaussian_list in gaussian_list_by_cv:
                cent_list = np.zeros(max_num_gaussians)
                for gauss_index, gaussian in enumerate(gaussian_list):
                    cent_list[gauss_index] = gaussian.centroid

                classif_data.append(cent_list)
            classif_data = np.asarray(classif_data).T

    else:
        print('WARNING: INVALID CLASSIFICATION MODE: {}'.format(params_obj.classif_3_mode))

    return classif_data


def main_build_classification(labels, analysis_obj_list_by_label, params_obj, output_dir):
    """
    Main method for classification. Performs feature selection followed by LDA and classification
    and generates output and plots. Returns a ClassificationScheme object to be saved for future
    classification of unknowns.
    :param labels: list of class labels (strings)
    :param analysis_obj_list_by_label: list of lists of analysis objects, sorted by class
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param output_dir: directory in which to save plots/output
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :return: ClassificationScheme object with the generated scheme
    :rtype: ClassificationScheme
    """
    # generate a list of lists of labels in the same shape as the analysis object list
    shaped_label_list = []
    for index, label in enumerate(labels):
        shaped_label_list.append([label for _ in range(len(analysis_obj_list_by_label[index]))])

    # run feature selection
    all_features = univariate_feature_selection(shaped_label_list, analysis_obj_list_by_label, params_obj, output_dir)

    # assess all features to determine which to use in the final scheme
    # best_features = all_feature_crossval_lda(all_features, analysis_obj_list_by_label, shaped_label_list, output_dir)
    best_features, crossval_score, all_crossval_data = crossval_main(analysis_obj_list_by_label, labels, output_dir, params_obj, all_features)

    # perform LDA and classification on the selected/best features
    constructed_scheme = lda_ufs_best_features(best_features, analysis_obj_list_by_label, shaped_label_list, params_obj, output_dir)
    constructed_scheme.crossval_test_score = crossval_score
    constructed_scheme.all_crossval_data = all_crossval_data

    # plot output here for now, will probably move eventually
    plot_classification_decision_regions(constructed_scheme, output_dir)
    return constructed_scheme


def univariate_feature_selection(shaped_label_list, analysis_obj_list_by_label, params_obj, output_path):
    """

    :param shaped_label_list: list of lists of class labels with the same shape as the analysis object list
    :param analysis_obj_list_by_label: list of lists of analysis objects, sorted by class
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param params_obj: parameters information
    :type params_obj: Parameters
    :param output_path: directory in which to save plot
    :return: list of selected features, list of all features (both CFeature object lists)
    """
    cv_axis = analysis_obj_list_by_label[0][0].axes[1]

    # generate all combinations of replicate datasets within the labels
    products = generate_products_for_ufs(analysis_obj_list_by_label, shaped_label_list, params_obj)

    # Create a CFeature object to hold the information for this CV (feature)
    scores = [product.fit_sc for product in products]
    mean_score = np.mean(scores, axis=0)
    std_score = np.std(scores, axis=0)

    features = []
    for cv_index, cv in enumerate(cv_axis):
        feature = CFeature(cv, cv_index, mean_score[cv_index], std_score[cv_index])
        features.append(feature)

    sorted_features = sorted(features, key=lambda x: x.mean_score, reverse=True)
    # num_features = 1
    # selected_features = sorted_features[0: num_features]
    plot_feature_scores(sorted_features, output_path)
    return sorted_features


def generate_products_for_ufs(analysis_obj_list_by_label, shaped_label_list, params_obj):
    """
    Generate all combinations of replicate data across classes for feature selection. Will
    create a DataProduct object with the key information for each combination.
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's, sorted by class label
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param shaped_label_list: list of lists of class labels with matching shape of analysis_obj_by_label
    :param params_obj: parameter info
    :type params_obj: Parameters
    :return: list of DataProduct objects for each combination
    :rtype: list[DataProduct]
    """
    products = []
    for object_tuple, label_tuple in zip(itertools.product(*analysis_obj_list_by_label), itertools.product(*shaped_label_list)):
        # create a DataProduct object for this combination
        # data_list = [x.ciu_data for x in object_tuple]
        data_list = [get_classif_data(x, params_obj, ufs_mode=True) for x in object_tuple]

        label_list = [x for x in label_tuple]
        product = DataProduct(data_list, label_list)

        # Run feature selection for this combination
        # todo: make these parameters accessible
        select = GenericUnivariateSelect(score_func=f_classif, mode='percentile', param=100)
        select.fit(product.combined_data, product.numeric_label_arr)

        product.fit_pvalues = select.pvalues_
        product.fit_scores = select.scores_
        product.fit_sc = -np.log10(select.pvalues_)  # not sure what this is - ask Suggie

        products.append(product)
    return products


def crossval_main(analysis_obj_list_by_label, labels, outputdir, params_obj, features_list):
    """

    :param analysis_obj_list_by_label:
    :param labels:
    :param outputdir:
    :param params_obj:
    :type params_obj: Parameters
    :param features_list: List of CFeatures, sorted in decreasing order of score
    :type features_list: list[CFeature]
    :return: list of selected features, test score for that # features, and all cross validation data
    """
    # training_size = params_obj.classif_1_training_size
    # determine training size as size of the smallest class - 1 (1 test file at a time)
    min_class_size = np.min([len(x) for x in analysis_obj_list_by_label])
    training_size = min_class_size - 1

    current_features_list = []
    train_score_means = []
    train_score_stds = []
    test_score_means = []
    test_score_stds = []


    # num_cores = multiprocessing.cpu_count() - 1
    num_cores = 3
    # print(num_cores)
    # pool = multiprocessing.Pool(num_cores)
    results = []

    time_start = time.time()
    for ind, feature in enumerate(features_list):
        print('\nNum features: {}'.format(ind+1))
        current_features_list.append(feature)

        # Generate all combinations
        crossval_obj = CrossValProduct(analysis_obj_list_by_label, labels, training_size, current_features_list)
        crossval_combos = crossval_obj.assemble_class_combinations(params_obj)
        result = assemble_products(crossval_combos)
        # result = pool.apply_async(func=assemble_products, args=[crossval_combos])
        results.append(result)

    for result in results:
        # output = result.get()
        output = result

        # get scores and plot and stuff
        train_score_means.append(output[0])
        train_score_stds.append(output[1])
        test_score_means.append(output[2])
        test_score_stds.append(output[3])


    print('classification done in {:.2f}'.format(time.time() - time_start))

    train_score_means = np.array(train_score_means)
    train_score_stds = np.array(train_score_stds)
    test_score_means = np.array(test_score_means)
    test_score_stds = np.array(test_score_stds)
    crossval_data = (train_score_means, train_score_stds, test_score_means, test_score_stds)
    save_crossval_score(crossval_data, outputdir)
    plot_crossval_scores(crossval_data, outputdir)

    # determine best features list from crossval scores
    best_num_feats, best_score = peak_crossval_score_detect(test_score_means, params_obj.classif_2_score_dif_tol)
    output_features = features_list[0: best_num_feats]
    return output_features, best_score, crossval_data


def save_crossval_score(crossval_data, outputpath):
    train_score_means = crossval_data[0]
    train_score_stds = crossval_data[1]
    test_score_means = crossval_data[2]
    test_score_stds = crossval_data[3]
    outfilename = os.path.join(outputpath, 'output_crossval.csv')
    with open(outfilename, 'w') as outfile:
        lineheader = 'num_feats, train_score_mean, train_score_std, test_score_mean, test_score_std, \n'
        outfile.write(lineheader)
        for ind in range(len(train_score_means)):
            line = '{}, {}, {}, {}, {}, \n'.format(ind+1, train_score_means[ind], train_score_stds[ind],
                                                   test_score_means[ind], test_score_stds[ind])
            outfile.write(line)
        outfile.close()


def peak_crossval_score_detect(test_score_means, diff_from_max):
    """
    Determine the best set of features based on crossvalidation testing scores generated in
    crossval_main. Chooses the first 'peak' (point after which score decreases) that is within
    tolerance of the overall maximum score in the data. This is to choose the minimum number of
    features while still achieving a high score.
    :param test_score_means: list of scores for cross validation test data, in increasing order of number of features
    :param diff_from_max: maximum distance below the max value of test_score_means that a peak is allowed for selection. Default 0.05
    :return: best number of features (index of test_score_means) and score (value)
    """
    max_score = np.max(test_score_means)

    for index, value in enumerate(test_score_means):
        try:
            if test_score_means[index + 1] < value:
                # stop here (reached a peak) if within tolerance of max
                if max_score - value <= diff_from_max:
                    return index + 1, value     # index + 1 because we're determining the NUM of features, indexed from 1 (not 0)
        except IndexError:
            # reached the end of the list - return final index/value
            return index, value


def plot_crossval_scores(crossval_data, outputdir):
    """
    Make plots of mean and std dev scores for training and test data from cross validation.
    :param crossval_data: tuple of (training means, training stds, test means, test stds) lists
    :param outputdir: directory in which to save output
    :return: void
    """
    train_score_means = crossval_data[0]
    train_score_stds = crossval_data[1]
    test_score_means = crossval_data[2]
    test_score_stds = crossval_data[3]

    xax = np.arange(0, len(train_score_means))  # todo: fix so that axis is # features (not # features - 1)
    plt.plot(xax, train_score_means, color='blue', marker='s', label='train_score')
    plt.fill_between(xax, train_score_means-train_score_stds, train_score_means+train_score_stds, color='blue', alpha=0.2)
    plt.plot(xax, test_score_means, color='green', marker='o', label='test_score')
    plt.fill_between(xax, test_score_means-test_score_stds, test_score_means+test_score_stds, color='green', alpha=0.2)

    outputname = os.path.join(outputdir, 'crossval_scores.pdf')
    plt.savefig(outputname)
    plt.close()


def arrange_data_for_lda(flat_data_matrix_list, flat_label_list, features_list, flat_filenames=None, method='flat'):
    """
    Prepare data for LDA by arranging selected CV columns (from the original matrix) into
    the desired shape. Multiple options supported at this time, will likely choose best eventually.
    :param flat_data_matrix_list: list of 2D arrays of data (drift bin OR gaussian info, collision voltage)
    :param flat_label_list: list of class labels corresponding to input data, in same shape as flat_data_matrix
    :param features_list: list of Features (collision voltage indicies) to use
    :type features_list: list[CFeature]
    :param flat_filenames: (optional) list of filenames in same shape as flat label list. If provided, filenames are returned
    :param method: 'flat' or 'stacked': whether to assemble features in a single list per datafile (flat) or in separate lists (stacked)
    :return: assembled raw data list, assembled labels list - ready for LDA as x, y
    """
    cvfeat_index_list = [x.cv_index for x in features_list]
    cvfeats_list = [x.cv for x in features_list]

    lda_ciu_data, lda_label_data, lda_filenames, lda_feat_cvs = [], [], [], []

    if method == 'stacked':
        # OLD WAY
        # loop over each replicate of data provided
        for data_index in range(len(flat_data_matrix_list)):
            # loop over each feature (collision voltage) desired, saving the requested data in appropriate form
            for index, data_cv_index in enumerate(cvfeat_index_list):
                lda_ciu_data.append(flat_data_matrix_list[data_index].T[data_cv_index])
                lda_label_data.append(flat_label_list[data_index])
                if flat_filenames is not None:
                    lda_filenames.append(flat_filenames[data_index])
                lda_feat_cvs.append(cvfeats_list[index])

    elif method == 'flat':
        # NEW WAY
        for data_index in range(len(flat_data_matrix_list)):
            # for this method, there is only one label/etc per replicate - so these can be assembled outside the feature loop
            lda_label_data.append(flat_label_list[data_index])
            if flat_filenames is not None:
                lda_filenames.append(flat_filenames[data_index])

            # loop over each feature (collision voltage) desired, saving the requested data in appropriate form
            rep_data = []
            for index, data_cv_index in enumerate(cvfeat_index_list):
                rep_data.extend(flat_data_matrix_list[data_index].T[data_cv_index])
            lda_ciu_data.append(rep_data)

    else:
        print('Invalid method! No LDA performed')

    return lda_ciu_data, lda_label_data, lda_filenames, lda_feat_cvs


def lda_clf_pipeline(stacked_train_data, stacked_train_labels, stacked_test_data, stacked_test_labels):
    """

    :param stacked_train_data:
    :param stacked_train_labels:
    :param stacked_test_data:
    :param stacked_test_labels:
    :return:
    """
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=5)
    lda.fit(stacked_train_data, stacked_train_labels)
    train_lda = lda.transform(stacked_train_data)
    test_lda = lda.transform(stacked_test_data)
    svm = SVC(kernel='linear', C=1, probability=True)
    svm.fit(train_lda, stacked_train_labels)
    train_score = svm.score(train_lda, stacked_train_labels)
    test_score = svm.score(test_lda, stacked_test_labels)
    # y_pred = svm.predict(test_lda)
    # confmat = confusion_matrix(y_true=stacked_test_labels, y_pred=y_pred)
    # probas = svm.predict_proba(test_lda)

    # output_queue.put((train_score, test_score))
    return train_score, test_score  # , probas


def lda_ufs_best_features(features_list, analysis_obj_list_by_label, shaped_label_list, param_obj, output_dir):
    """

    :param features_list: list of selected features from feature selection
    :type features_list: list[CFeature]
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's, sorted by class label
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param shaped_label_list: list of lists of class labels with matching shape of analysis_obj_by_label
    :param param_obj: parameters information
    :type param_obj: Parameters
    :param output_dir: directory in which to save output plot
    :return: generated classification scheme object with LDA and SVC performed
    :rtype: ClassificationScheme
    """
    # flatten input lists (sorted by class label) into a single list
    flat_ciuraw_list = [get_classif_data(x, param_obj) for label_obj_list in analysis_obj_list_by_label for x in label_obj_list]
    flat_label_list = [x for label_list in shaped_label_list for x in label_list]
    flat_filename_list = [x.short_filename for class_list in analysis_obj_list_by_label for x in class_list]

    # create a concatenated array with the selected CV columns from each raw dataset
    input_x_ciu_data, input_label_data, input_filenames, input_feats = arrange_data_for_lda(flat_ciuraw_list, flat_label_list, features_list, flat_filename_list, method=param_obj.classif_4_data_structure)

    # finalize input data for LDA
    input_x_ciu_data = np.asarray(input_x_ciu_data)
    input_y_labels, target_label = createtargetarray_featureselect(input_label_data)

    # run LDA
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=5)
    lda.fit(input_x_ciu_data, input_y_labels)
    x_lda = lda.transform(input_x_ciu_data)
    expl_var_r = lda.explained_variance_ratio_

    # build classification scheme
    clf = SVC(kernel='linear', C=1, probability=True)
    clf.fit(x_lda, input_y_labels)
    y_pred = clf.predict(x_lda)
    prec_score = precision_score(input_y_labels, y_pred, pos_label=1, average='weighted')

    # initialize classification scheme object and return it
    scheme = ClassificationScheme()
    scheme.selected_features = features_list
    scheme.classifier = clf
    scheme.classifier_type = 'SVC'
    scheme.classif_prec_score = prec_score
    scheme.lda = lda
    scheme.explained_variance_ratio = expl_var_r
    scheme.numeric_labels = input_y_labels
    scheme.class_labels = target_label
    scheme.unique_labels = get_unique_labels(target_label)
    scheme.transformed_test_data = x_lda
    scheme.test_filenames = input_filenames
    scheme.params = clf.get_params()
    scheme.input_feats = input_feats

    save_lda_output(x_lda, input_filenames, input_feats, output_dir, explained_variance_ratio=expl_var_r)

    return scheme


def plot_feature_scores(feature_list, output_path):
    """
    Plot feature score by collision voltage
    :param feature_list: list of CFeatures
    :type feature_list: list[CFeature]
    :param output_path: directory in which to save output
    :return: void
    """
    mean_scores = [x.mean_score for x in feature_list]
    std_scores = [x.std_dev_score for x in feature_list]
    cv_axis = [x.cv for x in feature_list]

    # threshold = -np.log10(0.05)
    plt.errorbar(x=cv_axis, y=mean_scores, yerr=std_scores, ls='none', marker='o', color='black')
    plt.axhline(y=0.0, color='black', ls='--')
    plt.xlabel('Feature')
    plt.ylabel('-Log10(pvalues)')
    plt.savefig(os.path.join(output_path, 'feature_scores.pdf'))
    plt.close()


def plot_classification_decision_regions(class_scheme, output_path, unknown_tups=None, filename=''):
    """

    :param class_scheme:
    :param output_path:
    :param unknown_tups: tuples of unknown (data, label). data = transformed lda data for unknown
    :param filename:
    :return:
    """
    markers = ('s', 'x', 'o', '^', 'v', 'D', '<', '>', '4', '8', 'h', 'H', '1', '2', '3', '+', '*', 'p', 'P')
    colors = ['deepskyblue', 'mediumspringgreen', 'fuchsia', 'lightgreen', 'gray', 'cyan', 'yellow', 'magenta']
    cmap = ListedColormap(colors[:len(class_scheme.unique_labels)])
    # decide whether the data has 1d or nds
    shape_lda = np.shape(class_scheme.transformed_test_data)

    fig = plt.figure()
    ax = plt.subplot(111)

    # plot 1D or 2D decision regions
    if shape_lda[1] == 1:
        x1_min, x1_max = np.floor(class_scheme.transformed_test_data.min()), np.ceil(class_scheme.transformed_test_data.max())
        if unknown_tups is not None:
            for unknown_tup in unknown_tups:
                min_unk = unknown_tup[0].min()
                max_unk = unknown_tup[0].max()
                x1_min = min([min_unk, x1_min])
                x1_max = max([max_unk, x1_max])

        # create a grid to evaluate model
        x2_min, x2_max = -3, 3
        x_grid, y_grid = np.mgrid[x1_min:x1_max:100j, x2_min:x2_max:100j]
        z = class_scheme.classifier.predict(x_grid.ravel().reshape(-1, 1))
        z = z.reshape(x_grid.shape)

        plt.contourf(x_grid, y_grid, z, alpha=0.2, cmap=cmap)
        plot_sklearn_lda_1ld(class_scheme, markers, colors)
        if unknown_tups is not None:
            for ind, unknown_tup in enumerate(unknown_tups):
                marklist = list(itertools.islice(itertools.cycle(markers), ind+1, ind+1+len(unknown_tups)))
                plt.scatter(unknown_tup[0], np.zeros(np.shape(unknown_tup[0])), marker=marklist[ind], color='black', facecolors='none',
                            alpha=1, label=unknown_tup[1])

    if shape_lda[1] == 2:
        x1_min, x1_max = np.floor(class_scheme.transformed_test_data[:, 0].min()), np.ceil(class_scheme.transformed_test_data[:, 0].max())
        x2_min, x2_max = np.floor(class_scheme.transformed_test_data[:, 1].min()), np.ceil(class_scheme.transformed_test_data[:, 1].max())
        if unknown_tups is not None:
            for unknown_tup in unknown_tups:
                x1_min_unk, x1_max_unk = np.floor(unknown_tup[0][:, 0].min()), np.ceil(unknown_tup[0][:, 0].max())
                x2_min_unk, x2_max_unk = np.floor(unknown_tup[0][:, 1].min()), np.ceil(unknown_tup[0][:, 1].max())
                x1_min = min([x1_min_unk, x1_min])
                x1_max = max([x1_max_unk, x1_max])
                x2_min = min([x2_min_unk, x2_min])
                x2_max = max([x2_max_unk, x2_max])

        num_grid_bins = 1000
        x_grid_1, x_grid_2 = np.meshgrid(np.arange(x1_min, x1_max, abs(x1_max - x1_min) / num_grid_bins), np.arange(x2_min, x2_max, abs(x2_max - x2_min) / num_grid_bins))
        z = class_scheme.classifier.predict(np.array([x_grid_1.ravel(), x_grid_2.ravel()]).T)
        z = z.reshape(x_grid_1.shape)

        plt.contourf(x_grid_1, x_grid_2, z, alpha=0.2, cmap=cmap)
        plot_sklearn_lda_2ld(class_scheme, markers, colors)

        if unknown_tups is not None:
            for ind, unknown_tup in enumerate(unknown_tups):
                plt.scatter(x=unknown_tup[0][:, 0], y=unknown_tup[0][:, 1], marker=markers[ind], color='black', alpha=0.5, label=unknown_tup[1])

    cv_string = ', '.join([str(np.round(x.cv, 1)) for x in class_scheme.selected_features])
    plt.title('From CVs: {}'.format(cv_string))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)

    plt.savefig(os.path.join(output_path, filename + '_classif_output.png'))
    plt.close()


def createtargetarray_featureselect(inputlabel):
    """

    :param inputlabel:
    :return:
    """
    arr = []
    for i in inputlabel:
        arr.append(i)
        # arr.append(i.split('_')[0])
    enc = LabelEncoder()
    label = enc.fit(arr)
    arr = label.transform(arr) + 1
    arr_decode = label.inverse_transform((arr - 1))
    return arr, arr_decode


def write_features_scores(score_dict, fname='avg_score_features_sfs.txt'):
    """

    :param score_dict:
    :param fname:
    :return:
    """
    with open(fname, 'w') as f:
        for key, val in sorted(score_dict.items()):
            feature_idx = val['feature_idx']
            avg_score = val['avg_score']
            f.write(str(avg_score) + '    ' + str(feature_idx) + '\n')
    f.close()


def plot_sklearn_lda_2ld(class_scheme, marker, color):
    """

    :param class_scheme:
    :param marker:
    :param color:
    :return:
    """
    x_data = class_scheme.transformed_test_data
    y_values = class_scheme.numeric_labels
    unique_labels = class_scheme.unique_labels

    for label, marker, color in zip(range(0, len(unique_labels)), marker, color):
        plt.scatter(x=x_data[:, 0][y_values == label + 1], y=x_data[:, 1][y_values == label + 1], marker=marker, color=color, alpha=0.5, label=np.unique(unique_labels)[label])
    plt.xlabel('LD1')
    plt.ylabel('LD2')


def plot_sklearn_lda_1ld(class_scheme, marker, color):
    """

    :param class_scheme:
    :param marker:
    :param color:
    :return:
    """
    x_data = class_scheme.transformed_test_data
    y_values = class_scheme.numeric_labels
    unique_labels = class_scheme.unique_labels

    for label, marker, color in zip(range(0, len(unique_labels)), marker, color):
        plt.scatter(x_data[:, 0][y_values == label + 1], np.zeros(np.shape(x_data[:, 0][y_values == label + 1])), marker=marker, color=color,
                    alpha=0.5, label=unique_labels[label])
    plt.xlabel('LD1')


def save_scheme(scheme, outputdir):
    """
    Save a ClassificationScheme object into the provided output directory using pickle
    :param scheme: classification object to save
    :type scheme: ClassificationScheme
    :param outputdir: directory in which to save output
    :return: void
    """
    unique_labels = get_unique_labels(scheme.class_labels)
    save_name = 'Classifier_' + '_'.join(unique_labels)
    output_path = os.path.join(outputdir, save_name + '.clf')

    with open(output_path, 'wb') as save_file:
        pickle.dump(scheme, save_file)


def load_scheme(filepath):
    """
    Load a classification scheme object previously saved using pickle
    :param filepath: full system path to file location to load
    :return: loaded scheme object
    :rtype: ClassificationScheme
    """
    with open(filepath, 'rb') as loadfile:
        scheme = pickle.load(loadfile)
    return scheme


# if __name__ == '__main__':
    # Read the data
    # import tkinter
    # from tkinter import filedialog
    # from tkinter import simpledialog
    # import CIU_Params
    # import Raw_Processing

    #
    # root = tkinter.Tk()
    # root.withdraw()

    # num_classes = simpledialog.askinteger('Class Number', 'Into how many classes do you want to group?')
    # num_classes = 4
    # data_labels = []
    # obj_list_by_label = []
    # # main_dir = 'C:\\'
    # main_dir = r"C:\Users\dpolasky\Desktop\CIUSuite2\CIU2 test data\Classify_Sarah\apo_cdl_pi_3way"

    # class_labels = ['Igg1', 'Igg2', 'Igg4', 'Igg4']
    # f1 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_1.ciu'
    # f2 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_2.ciu'
    # f3 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_3.ciu'
    # f4 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_1.ciu'
    # f5 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_2.ciu'
    # f6 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_3.ciu'
    # f7 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_1.ciu'
    # f8 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_2.ciu'
    # f9 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_3.ciu'
    # f10 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_1.ciu'
    # f11 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_2.ciu'
    # f12 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_3.ciu'

    # class_labels = ['cdl', 'pi', 'apo']
    # class1_files = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.ciu') and class_labels[0] in x.lower()]
    # class2_files = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.ciu') and class_labels[1] in x.lower()]
    # class3_files = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.ciu') and os.path.join(main_dir, x) not in class1_files and os.path.join(main_dir, x) not in class2_files]
    # fs = [class1_files, class2_files, class3_files]
    # f_class1 = [f1, f2, f3]
    # f_class2 = [f4, f5, f6]
    # f_class3 = [f7, f8, f9]
    # f_class4 = [f10, f11, f12]
    # fs = [f_class1, f_class2, f_class4, f_class4]
    # fs= [f_class1, f_class2]

    # sarahfile = r"C:\Users\sugyan\Documents\CIUSuite\Classification\IgGdata\Iggs_datalist.csv"
    # files = np.genfromtxt(sarahfile, skip_header=1, delimiter=',', dtype='str')
    # class_sarahfiles = np.unique(files[:, 0])
    # # CDL, PA, PC, PE, PG, PI, PPIX, PS [in order]
    # filelist = [[] for i in range(len(class_sarahfiles))]
    # for index, classtype in enumerate(files[:, 0]):
    #     for class_ind, class_unique in enumerate(class_sarahfiles):
    #         if class_unique == classtype:
    #             filelist[class_ind].append((files[:, 1][index]))
    #
    # class_labels = [class_sarahfiles[0], class_sarahfiles[1], class_sarahfiles[3], class_sarahfiles[3]]
    # fs = [filelist[0], filelist[1], filelist[3], filelist[3]]
    #
    # for class_index in range(0, num_classes):
    #     # Read in the .CIU files and labels for each class
    #     # label = simpledialog.askstring('Class Name', 'What is the name of this class?')
    #     class_label = class_labels[class_index]
    #     # files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
    #     files = fs[class_index]
    #     main_dir = os.path.dirname(files[0])
    #
    #     obj_list = []
    #     for file in files:
    #         with open(file, 'rb') as analysis_file:
    #             obj = pickle.load(analysis_file)
    #         obj_list.append(obj)
    #     data_labels.append(class_label)
    #     obj_list_by_label.append(obj_list)
    #
    # unkdata_name = open(r"C:\Users\sugyan\Documents\CIUSuite\Classification\IgGdata\IgG1_3.ciu", 'rb')
    # unkdata_ = pickle.load(unkdata_name)
    #
    # params = CIU_Params.Parameters()
    # params.set_params(CIU_Params.parse_params_file(CIU_Params.hard_descripts_file))
    # obj_list_by_label, equalized_axes_list = Raw_Processing.equalize_axes_2d_list(obj_list_by_label)
    #
    # scheme = main_build_classification(data_labels, obj_list_by_label, params, main_dir)
    # scheme.classify_unknown(unkdata_, params, main_dir, unk_label='Unknown')

    # featurescaling_lda(data_labels, obj_list_by_label, main_dir)
    # class_comparison_lda(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_withprediction(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_lda_withprediction(data_labels, obj_list_by_label, main_dir)

    # univariate_feature_selection(data_labels, obj_list_by_label)
    # output_scheme = main_build_classification(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_runldaonfeats_suggie(data_labels, obj_list_by_label, main_dir)
