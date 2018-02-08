"""
Module for classification schemes for CIU data groups
Authors: Dan Polasky, Sugyan Dixit
Date: 1/11/2018
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.backends.backend_pdf import PdfPages

# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# from sklearn.metrics import accuracy_score
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import f_classif, GenericUnivariateSelect
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import itertools
from typing import List


class ClassificationScheme(object):
    """
    Container for final classification information, to be saved and used in future for unknown
    analyses, etc.
    """
    def __init__(self):
        self.lda = None     # type: LinearDiscriminantAnalysis
        self.classifier = None      # type: SVC
        self.classifier_type = None

        self.selected_features = []     # type: List[CFeature]
        self.all_features = None    # type: List[CFeature]

        self.numeric_labels = None
        self.class_labels = None
        self.transformed_test_data = None

    def __str__(self):
        label_string = ','.join(self.class_labels)
        return '<Classif_Scheme> type: {}, data: {}'.format(self.classifier_type, label_string)
    __repr__ = __str__

    def classify_unknown(self, ciudata):
        """
        Classify a test dataset according to this classification scheme. Selects features from
        the test dataset (ciudata), classifies, and returns output and score metrics.
        :param ciudata: normalized CIU dataset (2D numpy array)
        :return:
        """
        # prepare data for classification by concatenating selected feature (CV) columns
        concat_data = []
        transpose_data = ciudata.T
        feature_cv_indices = [x.cv_index for x in self.selected_features]
        for cv_index in feature_cv_indices:
            concat_data = np.concatenate((concat_data, transpose_data[cv_index]))

        # classify
        prediction = self.classifier.predict(concat_data)
        return prediction


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


class DataProduct(object):
    """
    Container for label and product information for data combinations used in feature selection.
    """
    def __init__(self, data1, data2, label1, label2):
        """
        Initialize a new DataProduct with associated label and data
        :param data1: ciu_data (normalized)
        :param data2: ciu_data (normalized)
        :param label1: label for data1
        :param label2: label for data2
        """
        self.data1 = data1
        self.data2 = data2
        self.label1 = label1
        self.label2 = label2

        self.combined_data = None
        self.combined_label_arr = None
        self.numeric_label_arr = None

        # results information
        self.fit_scores = None
        self.fit_pvalues = None
        self.fit_sc = None

        # run data preparation
        self.prepare_data()

    def prepare_data(self):
        """
        Manipulate data into concatenated arrays suitable for input into feature selection algorithm.
        :return: void
        """
        self.combined_data = np.concatenate((self.data1, self.data2))

        label_arr_1 = np.asarray([self.label1 for _ in range(len(self.data1))])
        label_arr_2 = np.asarray([self.label2 for _ in range(len(self.data2))])
        self.combined_label_arr = np.concatenate((label_arr_1, label_arr_2))

        # use the label encoder to generate a numeric label list (could just do this manually??)
        # output is just the class numbers in an array the shape of the input
        encoder = LabelEncoder()
        label_list = encoder.fit(self.combined_label_arr)
        self.numeric_label_arr = label_list.transform(self.combined_label_arr) + 1

    def __str__(self):
        return '<DataProduct> labels: {}, {}'.format(self.label1, self.label2)
    __repr__ = __str__


def generate_products_for_ufs(analysis_obj_list_by_label, shaped_label_list):
    """
    Generate all combinations of replicate data across classes for feature selection. Will
    create a DataProduct object with the key information for each combination.
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's, sorted by class label
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param shaped_label_list: list of lists of class labels with matching shape of analysis_obj_by_label
    :return: list of DataProduct objects for each combination
    :rtype: list[DataProduct]
    """
    products = []
    for object_tuple, label_tuple in zip(itertools.product(*analysis_obj_list_by_label), itertools.product(*shaped_label_list)):
        # create a DataProduct object for this combination
        data1 = object_tuple[0].ciu_data
        data2 = object_tuple[1].ciu_data
        product = DataProduct(data1, data2, label_tuple[0], label_tuple[1])

        # Run feature selection for this combination
        # todo: make these parameters accessible
        select = GenericUnivariateSelect(score_func=f_classif, mode='percentile', param=100)
        select.fit(product.combined_data, product.numeric_label_arr)

        product.fit_pvalues = select.pvalues_
        product.fit_scores = select.scores_
        product.fit_sc = -np.log10(select.pvalues_)  # not sure what this is - ask Suggie

        products.append(product)
    return products


def univariate_feature_selection(shaped_label_list, analysis_obj_list_by_label):
    """

    :param shaped_label_list: list of lists of class labels with the same shape as the analysis object list
    :param analysis_obj_list_by_label: list of lists of analysis objects, sorted by class
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :return: list of selected features, list of all features (both CFeature object lists)
    """
    cv_axis = analysis_obj_list_by_label[0][0].axes[1]

    # generate all combinations of replicate datasets within the labels
    products = generate_products_for_ufs(analysis_obj_list_by_label, shaped_label_list)

    # Create a CFeature object to hold the information for this CV (feature)
    scores = [product.fit_sc for product in products]
    mean_score = np.mean(scores, axis=0)
    std_score = np.std(scores, axis=0)

    features = []
    for cv_index, cv in enumerate(cv_axis):
        feature = CFeature(cv, cv_index, mean_score[cv_index], std_score[cv_index])
        features.append(feature)

    # todo: make this selection its own method with multiple ways (best N features, all above cutoff)
    sorted_features = sorted(features, key=lambda x: x.mean_score, reverse=True)
    num_features = 5
    selected_features = sorted_features[0: num_features]

    return selected_features, features


def lda_ufs_best_features(features_list, analysis_obj_list_by_label, shaped_label_list):
    """

    :param features_list: list of selected features from feature selection
    :type features_list: list[CFeature]
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's, sorted by class label
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param shaped_label_list: list of lists of class labels with matching shape of analysis_obj_by_label
    :return: generated classification scheme object with LDA and SVC performed
    :rtype: ClassificationScheme
    """
    # flatten input lists (sorted by class label) into a single list
    flat_ciuraw_list = [x.ciu_data for label_obj_list in analysis_obj_list_by_label for x in label_obj_list]
    flat_label_list = [x for label_list in shaped_label_list for x in label_list]

    selected_cv_indices = [x.cv_index for x in features_list]

    # create a concatenated array with the selected CV columns from each raw dataset
    input_x_ciu_data = []
    for ciuraw_data in flat_ciuraw_list:
        selected_cols = []
        # concatenate all selected columns
        for cv_index in selected_cv_indices:
            selected_cols = np.concatenate((selected_cols, ciuraw_data[cv_index]), axis=0)
        input_x_ciu_data.append(selected_cols)

    # finalize input data for LDA
    input_x_ciu_data = np.asarray(input_x_ciu_data)
    input_y_labels, target_label = createtargetarray_featureselect(flat_label_list)

    # run LDA
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=5)
    lda.fit(input_x_ciu_data, input_y_labels)
    x_lda = lda.transform(input_x_ciu_data)

    # build classification scheme
    # clf = LogisticRegression(C=10)
    clf = SVC(kernel='linear', C=10)
    # clf = RandomForestClassifier(n_estimators=10, criterion='entropy', n_jobs=-1)
    # clf = KNC(n_neighbors=2,p=2,metric='minkowski', n_jobs=-1)
    clf.fit(x_lda, input_y_labels)

    # initialize classification scheme object and return it
    scheme = ClassificationScheme()
    scheme.selected_features = features_list
    scheme.classifier = clf
    scheme.classifier_type = 'SVC'
    scheme.lda = lda
    scheme.numeric_labels = input_y_labels
    scheme.class_labels = target_label
    scheme.transformed_test_data = x_lda
    # scheme.input_ciu_data = input_x_ciu_data

    return scheme


def main_build_classification(labels, analysis_obj_list_by_label, output_dir):
    """
    Main method for classification. Performs feature selection followed by LDA and classification
    and generates output and plots. Returns a ClassificationScheme object to be saved for future
    classification of unknowns.
    :param labels: list of class labels (strings)
    :param analysis_obj_list_by_label: list of lists of analysis objects, sorted by class
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param output_dir: directory in which to save plots/output
    :return: ClassificationScheme object with the generated scheme
    :rtype: ClassificationScheme
    """
    # generate a list of lists of labels in the same shape as the analysis object list
    shaped_label_list = []
    for index, label in enumerate(labels):
        shaped_label_list.append([label for _ in range(len(analysis_obj_list_by_label[index]))])

    # run feature selection
    best_features, all_features = univariate_feature_selection(shaped_label_list, analysis_obj_list_by_label)

    # perform LDA and classification on the selected/best features
    constructed_scheme = lda_ufs_best_features(best_features, analysis_obj_list_by_label, shaped_label_list)

    # plot output here for now, will probably move eventually
    labels_name = 'Univariatefeatureselection' + '_'.join(labels) + '_'
    output_path = os.path.join(output_dir, labels_name)
    plot_stuff_suggie(constructed_scheme, output_path)

    return constructed_scheme


def plot_stuff_suggie(class_scheme, output_path):
    """

    :param class_scheme:
    :param output_path:
    :return:
    """
    pdf_out = PdfPages(output_path + 'ldatransform_features_univariate_fclassif_SVM_C10_linear_classif.pdf')

    x_min, x_max = np.floor(class_scheme.transformed_test_data.min()), np.ceil(class_scheme.transformed_test_data.max())
    y_min, y_max = -3, 3
    # yy = np.linspace(y_min, y_max)
    colors = ['blue', 'red', 'lightgreen', 'gray', 'cyan']
    # cmap = ListedColormap(colors[:len(np.unique(class_scheme.numeric_labels))])
    XX, YY = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    XXX = np.c_[XX.ravel(), YY.ravel()]
    Z = class_scheme.classifier.predict(XX.ravel().reshape(-1, 1))
    Z = Z.reshape(XX.shape)
    plt.contourf(XX, YY, Z, alpha=0.4, cmap=plt.cm.Paired)
    cv_string = ','.join([str(x.cv) for x in class_scheme.selected_features])
    plt.title('From CVs: {}'.format(cv_string))
    plot_sklearn_lda_1ld(class_scheme.transformed_test_data, class_scheme.numeric_labels, class_scheme.class_labels)
    pdf_out.savefig()
    plt.close()

    plt.close()
    pdf_out.close()


# class RawTransformer(object):
#     """
#     modified from Suggie
#     """
#     def __init__(self, initial_dict):
#         """
#
#         :param initial_dict:
#         """
#         self.raw_dict = initial_dict
#         self.raw_x_array = self.transformdata_rawdata()
#         self.decoded_array = []
#
#     def transformdata_rawdata(self):
#         """
#
#         :return:
#         """
#         arr = np.hstack((self.raw_dict.values())).T
#         return arr
#
#     def createtargetarray_rawdata(self):
#         arr = []
#         for i, (keys, values) in enumerate(zip(self.raw_dict.keys(), self.raw_dict.values())):
#             arr.append(np.repeat(str(keys).split('_')[0], len(values[0])))
#         arr = np.concatenate(arr)
#         enc = LabelEncoder()
#         label = enc.fit(arr)
#         arr = label.transform(arr) + 1
#         arr_decode = label.inverse_transform((arr - 1))
#         return arr, arr_decode


def createtargetarray_featureselect(inputlabel):
    arr = []
    for i in inputlabel:
        arr.append(i.split('_')[0])
    enc = LabelEncoder()
    label = enc.fit(arr)
    arr = label.transform(arr) + 1
    arr_decode = label.inverse_transform((arr - 1))
    return arr, arr_decode


# def createdict(list, labels):
#     dict = {e: list[i] for i, e in enumerate(labels)}
#     return dict


# def curatedata_bestfeature_combined(array, feature_idx):
#     newarr = []
#     for i in feature_idx:
#         newarr.append(array[i][:])
#     return newarr


def write_features_scores(dict, fname='avg_score_features_sfs.txt'):
    with open(fname, 'w') as f:
        for key, val in sorted(dict.items()):
            feature_idx = val['feature_idx']
            avg_score = val['avg_score']
            f.write(str(avg_score) + '    ' + str(feature_idx) + '\n')
    f.close()


def plot_sklearn_lda_3ld(X, y, target_label, title='LDA: Projection of first 3 LDs'):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=9, azim=142)
    for label, marker, color in zip(range(0, len(np.unique(target_label))), ('^', 's', 'o', 'd'),
                                    ('blue', 'red', 'green', 'black')):
        ax.scatter(xs=X[:, 0][y == label + 1], ys=X[:, 1][y == label + 1], zs=X[:, 2][y == label + 1], color=color,
                   edgecolor='k', marker=marker, s=40, label=np.unique(target_label)[label])
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)


def plot_sklearn_lda_2ld(X, y, target_label, title='LDA: Projection of first 2 LDs'):
    ax = plt.subplot(111)
    for label, marker, color in zip(range(0, len(np.unique(target_label))), ('^', 's', 'o', 'd'),
                                    ('blue', 'red', 'green', 'black')):
        plt.scatter(x=X[:, 0][y == label + 1], y=X[:, 1][y == label + 1], marker=marker, color=color, alpha=0.5,
                    label=np.unique(target_label)[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.grid()


def plot_sklearn_lda_1ld(X, y, target_label):
    for label, marker, color in zip(range(0, len(np.unique(target_label))), ('^', 's', 'o', 'd', 'x'),
                                    ('red', 'blue', 'green', 'gray', 'cyan')):
        plt.scatter(X[:, 0][y == label + 1], np.zeros(np.shape(X[:, 0][y == label + 1])), marker=marker, color=color,
                    alpha=0.5, label=np.unique(target_label)[label])
    plt.legend(loc='best')
    plt.xlabel('LD1')
    plt.ylim((1, -1))


def plot_decision_regions(X, y, target_label, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ['blue', 'red', 'lightgreen', 'gray', 'cyan']
    # cmap = ListedColormap(['blue', 'red', 'lightgreen'])
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx],
                    label=np.unique(target_label)[idx])

    plt.legend(loc='best', fontsize='small')


def load_analysis_obj(analysis_filename):
    """
    Load a pickled analysis object back into program memory
    :param analysis_filename: full path to file location to load
    :return: CIUAnalysisObj
    """
    with open(analysis_filename, 'rb') as analysis_file:
        analysis_obj = pickle.load(analysis_file)
    return analysis_obj


# deprecated
# def class_comparison_lda(labels, analysis_obj_list_by_label, output_dir):
#     """
#     Modularized version of class comparision algorithm from Suggie. Takes a list of labels and a
#     list of AnalysisObjs for each label and performs a 2-stage LDA based classification. Output
#     is saved with the voltages of greatest difference and their scores for assay development.
#     :param labels = list of strings corresponding to class labels
#     :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's. Same length as labels list, with
#     arbitrary number of Analysis objects for each label.
#     :param output_dir: directory in which to save output
#     :return: void
#     """
#     # CVs for plot titles later
#     cv1 = analysis_obj_list_by_label[0][0].axes[1]
#
#     inarray_list = []
#     inlabel_list = []
#     label_index = 0
#     for label in labels:
#         # Get the list of datasets corresponding to this label
#         label_obj_list = analysis_obj_list_by_label[label_index]
#         data_index = 1
#
#         for analysis_obj in label_obj_list:
#             # Add the formatted label and dataset to their input lists for the feature selection
#             inlabel_list.append('{}_{}'.format(label, data_index))
#             ciu_data = analysis_obj.ciu_data
#             inarray_list.append(ciu_data.T)     # append transposed CIU data matrix
#             data_index += 1
#         label_index += 1
#     run_lda_raw(inarray_list, inlabel_list, labels, output_dir, cv1)
#
#
# # deprecated
# def classification_lda_maxvals(labels, analysis_obj_list_by_label, output_dir):
#     """
#     Analogous to classification_lda_raw, except uses only the max value in each CV column (set to 1,
#     with all other DTs set to 0) as the input data.
#     :param labels = list of strings corresponding to class labels
#     :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's. Same length as labels list, with
#     arbitrary number of Analysis objects for each label.
#     :param output_dir: directory in which to save output
#     :return: void
#     """
#     cv1 = analysis_obj_list_by_label[0][0].axes[1]
#
#     inarray_list = []
#     inlabel_list = []
#     label_index = 0
#     for label in labels:
#         # Get the list of datasets corresponding to this label
#         label_obj_list = analysis_obj_list_by_label[label_index]
#         data_index = 1
#
#         for analysis_obj in label_obj_list:
#             # Add the formatted label and dataset to their input lists for the feature selection
#             inlabel_list.append('{}_{}'.format(label, data_index))
#             ciu_data_col = analysis_obj.ciu_data.T
#
#             # reduce ciu_data to max values only (normalized to 1.0)
#             new_ciu = np.asarray([[np.argmax(col)] for col in ciu_data_col])
#             # new_ciu = np.zeros(np.shape(ciu_data_col))
#             # for i in range(len(ciu_data_col)):
#             #     for j in range(len(ciu_data_col[0])):
#             #         if ciu_data_col[i][j] == np.max(ciu_data_col[i]):
#             #             new_ciu[i][j] = 1.0
#
#             # inarray_list.append(ciu_data_col)  # append transposed CIU data matrix
#             inarray_list.append(new_ciu)
#             data_index += 1
#         label_index += 1
#     run_lda_raw(inarray_list, inlabel_list, labels, output_dir, cv1)
#
#
# # deprecated
# def run_lda_raw(inarray_list, inlabel_list, label_list, output_dir, cv_labels):
#     # create the dictionary needed to create the data and target array
#     dict_init = createdict(inarray_list, inlabel_list)  # dictionary
#     transform_obj = RawTransformer(dict_init)  # data
#     X_ = transform_obj.raw_x_array
#     y_, target_label = transform_obj.createtargetarray_rawdata()  # target
#
#     print(np.shape(X_))
#
#     # initiate LDA from sklearn
#     # lda = LDA(solver='eigen', n_components=5)
#     lda = LDA(solver='svd', n_components=2)
#
#
#     # initiate feature selection from mlxtend package
#     # 4 different algorithms are available. See
#     # rabst.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector
#     # to toggle between different selection algorithms, change the bool values for forward and floating attributes
#     # below is set up for all 4 feature selection module
#
#     # SFFS and SBFS is computationally more expensive because it either excludes after inclusion or includes
#     # after exclusion to maximize the score
#
#     # Sequential Forward Selection (sfs)
#     # sfs = SFS(lda, k_features=(1, np.shape(X_)[1]), forward=True, floating=False, scoring='accuracy', cv=5, n_jobs=-1)
#     sfs = SFS(lda, k_features=(1, np.shape(X_)[1]), forward=True, floating=False, scoring='accuracy', cv=0, n_jobs=-1)
#
#     # Sequential Backward Selection (sbs)
#     # sfs = SFS(lda, k_features = (1, np.shape(X_)[1]), forward = False, floating = False, scoring = 'accuracy', cv = 5, n_jobs = -1)
#
#     # Sequential Forward Floating Selection (SFFS)
#     # sfs = SFS(lda, k_features = (1, np.shape(X_)[1]), forward = True, floating = True, scoring = 'accuracy', cv = 5, n_jobs = -1)
#
#     # Sequential Backward Floating Selection (SBFS)
#     # sfs = SFS(lda, k_features = (1, np.shape(X_)[1]), forward = False, floating = True, scoring = 'accuracy', cv = 5, n_jobs = -1)
#
#     ############
#
#     # fit the feature selection for the data
#     sfs.fit(X_, y_)
#
#     print('best combination (ACCURACY: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_))
#     # print ('\nAll Subsets:\n', sfs.subsets_)
#
#     labels_name = 'LDA_' + '_'.join(label_list) + '_'
#     output_path = os.path.join(output_dir, labels_name)
#
#     # write the avg score and features in a text file for reference
#     write_features_scores(dict=sfs.subsets_, fname=output_path + 'avg_score_features_sfs_unevenclassnum.txt')
#
#     # plot the accuracy vs features using plot_sfs module from mlxtend
#     plot_sfs(sfs.get_metric_dict(), kind='std_err')
#     plt.savefig(output_path + 'performance_vs_features_sfs.pdf')
#     plt.close()
#
#     # with the best k features from the feature selection algorithm, we now compare the raw data for each k feature
#
#     # create pdf to store figures of lda transformed data for each feature
#     pdf_fig = pdfpage(output_path + 'ldatransform_features_sfs.pdf')
#
#     # create X and y input array
#     for i in sfs.k_feature_idx_:
#         arr = []
#         for k in inarray_list:
#             arr.append(k[i])
#         arr = np.array(arr)  # X array
#         y_, target_label = createtargetarray_featureselect(inlabel_list)  # target array
#
#         # initiate LDA. The solver is now changed to SVD from eigen. For some reason, scikitlearn cannot compute eigen values. Not sure exactly what causes it. There are some posts online refering to internal bug. Anyway, svd creates very similar results. I personally like eigen solver because I understand that better.
#         # n_components is n_classes - 1. Since we have only 2 classes, we'll only obtain 1 component. However, with classes > 2 its necessary to evaluate the explained variance ratio for each component in order to properly assess the transformation.
#         lda2 = LDA(solver='svd', n_components=5)
#
#         # fit LDA
#         lda2.fit(arr, y_)
#         print('Explained_variance_ratio: ',
#               lda2.explained_variance_ratio_)  # important to evaluate, especially when more than one LDs.
#         # transform the dataset with lda
#         x_lda = lda2.transform(arr)
#
#         # evaluate score. This is a harsh metric especially for multi class classification. The performance of such
#         # transformation and classification needs to be evaluated in better ways, for example, creating training and
#         # test datasets, cross-validation, assessing other performance metrics etc. Will be working on that soon.
#         print('Score: ', lda2.score(arr, y_))
#
#         if len(label_list) == 2:
#             plot_sklearn_lda_1ld(x_lda, y_, target_label)
#         elif len(label_list) == 3:
#             lr = LogisticRegression()
#             lr = lr.fit(x_lda, y_)
#
#             plot_decision_regions(x_lda, y_, target_label, classifier=lr)
#         else:
#             lr = LogisticRegression()
#             lr = lr.fit(x_lda, y_)
#
#             plot_sklearn_lda_3ld(x_lda, y_, target_label)
#         plt.title(str(cv_labels[i]))
#         pdf_fig.savefig()
#         plt.close()
#
#     pdf_fig.close()

# if __name__ == '__main__':
#     # Read the data
#     import tkinter
#     from tkinter import filedialog
#     from tkinter import simpledialog
#     root = tkinter.Tk()
#     root.withdraw()
#
#     num_classes = simpledialog.askinteger('Class Number', 'Into how many classes do you want to group?')
#
#     data_labels = []
#     obj_list_by_label = []
#     main_dir = 'C:\\'
#     for index in range(0, num_classes):
#         # Read in the .CIU files and labels for each class
#         label = simpledialog.askstring('Class Name', 'What is the name of this class?')
#         files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
#         main_dir = os.path.dirname(files[0])
#
#         obj_list = []
#         for file in files:
#             with open(file, 'rb') as analysis_file:
#                 obj = pickle.load(analysis_file)
#             obj_list.append(obj)
#         data_labels.append(label)
#         obj_list_by_label.append(obj_list)
#
#     class_comparison_lda(data_labels, obj_list_by_label, main_dir)
#     # classification_lda_maxvals(data_labels, obj_list_by_label, main_dir)

def save_scheme(scheme, outputdir):
    """
    Save a ClassificationScheme object into the provided output directory using pickle
    :param scheme: classification object to save
    :type scheme: ClassificationScheme
    :param outputdir: directory in which to save output
    :return: void
    """
    save_name = 'Classifier_' + '_'.join(scheme.class_labels)
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


if __name__ == '__main__':
    # Read the data
    import tkinter
    from tkinter import filedialog
    from tkinter import simpledialog

    root = tkinter.Tk()
    root.withdraw()

    # num_classes = simpledialog.askinteger('Class Number', 'Into how many classes do you want to group?')
    num_classes = 2
    data_labels = []
    obj_list_by_label = []
    main_dir = 'C:\\'

    class_labels = ['Igg1', 'Igg2', 'Igg4', 'Igg4']
    f1 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_1.ciu'
    f2 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_2.ciu'
    f3 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_3.ciu'
    f4 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_1.ciu'
    f5 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_2.ciu'
    f6 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_3.ciu'
    f7 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_1.ciu'
    f8 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_2.ciu'
    f9 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_3.ciu'
    f10 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_1.ciu'
    f11 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_2.ciu'
    f12 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_3.ciu'

    f_class1 = [f1, f2, f3]
    f_class2 = [f4, f5, f6]
    f_class3 = [f7, f8, f9]
    f_class4 = [f10, f11, f12]

    fs = [f_class1, f_class2, f_class4, f_class4]
    # fs= [f_class1, f_class2]

    for class_index in range(0, num_classes):
        # Read in the .CIU files and labels for each class
        # label = simpledialog.askstring('Class Name', 'What is the name of this class?')
        class_label = class_labels[class_index]
        # files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
        files = fs[class_index]
        main_dir = os.path.dirname(files[0])

        obj_list = []
        for file in files:
            with open(file, 'rb') as analysis_file:
                obj = pickle.load(analysis_file)
            obj_list.append(obj)
        data_labels.append(class_label)
        obj_list_by_label.append(obj_list)

    # featurescaling_lda(data_labels, obj_list_by_label, main_dir)
    # class_comparison_lda(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_withprediction(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_lda_withprediction(data_labels, obj_list_by_label, main_dir)

    # univariate_feature_selection(data_labels, obj_list_by_label)
    output_scheme = main_build_classification(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_runldaonfeats_suggie(data_labels, obj_list_by_label, main_dir)
