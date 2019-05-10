#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import pandas as pd 
import numpy as np
import re
import nltk
from tldextract import extract  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, cross_validate, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.externals.joblib import dump, load
import pickle


def jaccard_guess(word, first_label, second_label):
    """
    This function computes Jaccard distance between the 'word' and the labels.
    It returns the label with shortest distance from the word.
    
    Parameters
    ----------
    word : misspelled word that needs to be matched to a label
    first_label : first possible match
    second_label : second possible match
    
    Returns
    -------
    Returns one of the two labels that is closest to the word.
    """
    word1 = set([x for x in word] )
    label1 = set([x for x in first_label] )
    label2 = set([x for x in second_label])

    union_w1_l1 = word1 | label1
    int_w1_l1 = word1 & label1
    union_w1_l2 = word1 | label2
    int_w1_l2 = word1 & label2
    jaccard_dist_label_1 = 1.0 - 1.0*len(int_w1_l1)/len(union_w1_l1)
    jaccard_dist_label_2 = 1.0 - 1.0*len(int_w1_l2)/len(union_w1_l2)
    
    if min(jaccard_dist_label_1,jaccard_dist_label_2) > 0.6:
        return np.nan
    
    if jaccard_dist_label_1<jaccard_dist_label_2:
        return first_label
    else:
        return second_label

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    NOTE: THIS FUNCTION IS TAKEN FROM THE FOLLOWING SOURCE:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def save_model(model, path, filename):
    """
    Save model to file
    
    Parameters
    ----------
    model : model object
    path : path for saving model
    filename : name of the saved model
    """
    file = os.path.join(path, "{}.pkl".format(filename))

    # save the model to disk
    # dump(model, file)
    pickle.dump(model, open(file, 'wb'))


def phase1_data_preparation(dataset_dir, dataset_name):
    """
    Phase 1: Perform data preparation steps and split into training and evaluation sets.
    
    Parameters
    ----------
    dataset_dir : directory of dataset
    dataset_name : name of dataset
    
    Returns
    -------
    X_train : train dataset
    X_test : test dataset 
    y_train : train labels
    y_test : test labels
    """
    
    # Read 'ctrl+A' separated dataset
    df = pd.read_table(os.path.join(dataset_dir, dataset_name), sep= '\x01', header=None,
                       names = ['url', 'origin', 'label'], nrows = None )

    # Drop 'origin' as it is not available in production on real unknown domains
    df.drop('origin', axis=1, inplace=True)

    # Drop rows for with either 'label' or 'url' are missing
    df.dropna(subset=['label', 'url'], inplace=True)
    
    # Convert label to lowercase
    df['label'] = df['label'].str.lower()
    
    # Clean misspelled labels
    df['label'] = df['label'].apply(lambda x: jaccard_guess(x, "legit", "dga" ))
    df.dropna(subset=['label'], inplace=True)
    
    # Map 'dga' to 1 and 'legit' to 0
    df['label'] = df['label'].apply(lambda x: 1 if x == 'dga' else 0)

    # Split URL into primary domain and tld
    df['url_extract'] = df['url'].apply(extract)
    df['prim_domain'] = df['url_extract'].apply(lambda x: x.domain)
    df['tld'] = df['url_extract'].apply(lambda x: x.suffix)
    df.drop('url_extract', axis=1, inplace=True)
    df['prim_domain'].replace('', np.nan, inplace=True)
    df.dropna(subset=['label', 'prim_domain'], inplace=True)

    # Get rid of duplicate primary domains by grouping and choosing most frequent label and tld
    df_grouped = df.groupby('prim_domain')
    df1 = df_grouped.apply(lambda x: x[['label']].mode()).reset_index().drop('level_1', axis =1)
    df2 = df_grouped.apply(lambda x: x[['tld']].mode()).reset_index().drop('level_1', axis=1)
    df = pd.merge(df1, df2, on = 'prim_domain', how= 'left')

    # Evaluate class distribution for binary classification
    class_dist = df['label'].value_counts()
    class_dist =  100.0* class_dist/class_dist.sum()
    if class_dist.min() < 30.0:
        print("Warning: The class distribution is unbalanced. Use appropriate methods to address this problem.")
        
    # Split dataset into train and test set in a stratified fashion
    X = df.drop('label', axis =1)
    y = df['label'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.3,shuffle = True, random_state = 170)

    print("Data preparation: done.", "\n")
    return X_train, X_test, y_train, y_test


class feature_extractor(BaseEstimator, TransformerMixin):
    """
    Extracts features from the dataset
    
    Attributes
    ----------
    vec_char_prim_domain : estimator for vectorizing character counts
    vec_2gram_prim_domain : estimator for vectorizing bigram counts
    feature_names : names of extracted features
    
    """
    def __init__(self):
        self.X = None
        self.vec_char_prim_domain = None
        self.vec_2gram_prim_domain = None
        self.feature_names = None
        
    def get_vectorized_features(self, df, vectorizer, column_name):
        df_vectorized = vectorizer.transform(df[column_name]).toarray()
        df_vectorized = pd.DataFrame(df_vectorized, columns=vectorizer.get_feature_names(), index= df.index)
        return df_vectorized
    
    def get_vectorizer(self, df, column_name):
        vectorizer = CountVectorizer()
        vectorizer.fit(df[column_name])
        return vectorizer
    
    def get_raw_featues(self, df):
        df['num_tld'] = df['tld'].apply(lambda x: len(x.split('.')))
        df['len_tld'] = df['tld'].apply(len)
        df['len_prim_domain'] = df['prim_domain'].apply(len)

        df['char_prim_domain'] = df["prim_domain"].apply(lambda t: "_ ".join(list(t)))
        df['2gram_prim_domain'] = df["prim_domain"].apply(lambda t: " ".join(["".join(x) for x in nltk.ngrams(t, 2)]))
        return df
    
    def fit(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
       
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        
        df = X.copy()

        # Get unvectorized features
        df = self.get_raw_featues(df)

        # Fit vectorizers
        self.vec_char_prim_domain = self.get_vectorizer(df, 'char_prim_domain')
        self.vec_2gram_prim_domain = self.get_vectorizer(df, '2gram_prim_domain')
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
       
        Returns
        -------
        df : pandas dataframe consisting of extracted features
        """
        df = X.copy()
        df = self.get_raw_featues(df)
        
        df_char_prim_domain = self.get_vectorized_features(df, self.vec_char_prim_domain, 'char_prim_domain')
        df_2gram_prim_domain = self.get_vectorized_features(df, self.vec_2gram_prim_domain, '2gram_prim_domain')
        df.drop(['char_prim_domain', '2gram_prim_domain', 'prim_domain', 'tld'], axis = 1, inplace=True)

        # Join feature datasets
        df = pd.concat([df, df_char_prim_domain, df_2gram_prim_domain], axis = 1)
        self.feature_names = df.columns.tolist()
        return df.astype(float)


def select_important_features(X_train, y_train, feature_names):
    """
    Tree based feature selection.
    
    Features are selected based on the median feature_importances_ values of 10 estimators
    output by 10-fold cross validation of the training data.
    
    Parameters
    ----------
    X_train : pandas dataframe, shape (n_samples, n_features)
              Training data.
    
    y_train : pandas dataframe, shape (n_samples)
            Target data.

    feature_names: list of all features the dataset
    
    Returns
    -------
    df : pandas dataframe consisting of extracted features
    """
    
    rfc = RandomForestClassifier(n_estimators=10, n_jobs=6)
    scores_train_cv = cross_validate(rfc, X_train, y_train, cv =10, n_jobs = 6, return_estimator = True,
                                 scoring = ['accuracy', 'precision', 'recall', 'roc_auc'])
    
    
    cv_metrics = {'cv'+k[4:]: scores_train_cv[k].mean() for k in ['test_accuracy', 'test_precision', 
                                                                  'test_recall', 'test_roc_auc']}

    # Compile important features
    important_features = []
    for est in scores_train_cv['estimator']:
        feat_importance = pd.DataFrame(list(zip(feature_names, est.feature_importances_ )))
        #feat_importance.sort_values(1, ascending=False, inplace = True)
        feat_importance = feat_importance[feat_importance[1]>=feat_importance[1].median()]
        important_features = important_features + list(feat_importance[0].values)

    important_features = list(set(important_features))
    return important_features, cv_metrics


class dimensionality_reduction(BaseEstimator, TransformerMixin):
    """
    Dimensionality Reduction
    
    Carries out dimensionality reduction using PCA and selects number of components
    based on explained variance. It returns original dataset rather than the transformed
    dataset if the reduction in dimensions is not significant (tradeoff with model 
    interpretability)
    
    Attributes
    ----------
    pca : object
          Fitted estimator 
    n_components : number of principal components
    
    Parameters
    ----------
    results_dir: directory for saving results (explained variance plot) 
    
    """
    
    def __init__(self, results_dir = os.path.join(".", "results")):
        self.pca = None
        self.do_pca = None
        self.n_components =  None
        self.results_dir = results_dir
        
    def fit(self, X_train, y_train = None):
        """
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
                  Training data.
            
        X_test : array-like, shape (n_samples, n_features)
                 Testing data.
       
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.pca = PCA()
        self.pca.fit(X_train)
        cum_sum = np.cumsum(self.pca.explained_variance_ratio_)
        num_original_features = X_train.shape[1]
        
        # Number of dimensions need to preserve 95% of variance of training set
        self.n_components = np.argmax(cum_sum >= 0.95)+1
        percentage_n_components = 100.0*self.n_components/num_original_features

        # Plot Explained Variance Vs Dimensions
        f11 = plt.figure()
        plt.plot(range(len(cum_sum)), cum_sum)
        plt.xlabel('Dimensions')
        plt.ylabel('Explained variance')
        plt.savefig(os.path.join(self.results_dir, 'PCA_explained_variance.png'))
        plt.show()
        print('PCA:')
        print('====')
        print('Percentage of n_components for 95% explained variance: ', percentage_n_components)
        print()
        
        # Skip PCA if the %reduction in dimensions in not less than 75% of the number of original features.
        if percentage_n_components <= 75.0:
            self.do_pca = True
        else:
            self.do_pca = False
            
        return self.pca

    def transform(self, X, y =None ):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
       
        Returns
        -------
        X : array-like consisting of transformed features or original dataset
        """
        if self.do_pca:
            #print('do_pca: ', self.do_pca)
            X = self.pca.transform(X)[:,:self.n_components]
            return pd.DataFrame(X)
        else: 
            return X


def hyperparameter_tuning(X_train, y_train):
    """
    Randomized Search
    
    Hyperparameter tuning and selection of the best model using Randomized Search
    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
              Training data.
    
    y_train : array-like, shape (n_samples)
    
    Returns
    -------
    rand_search : object
    """
    param_rand = {'n_estimators': [10, 25, 50], 
                  'min_samples_split' : [2, 4, 6],
                  'max_features': [20, 25, 30]}

    # Randomized search using 10-fold cross validation
    clf = RandomForestClassifier(n_jobs=6)
    rand_search = RandomizedSearchCV(clf, param_rand, scoring = 'roc_auc', cv = 10, n_jobs=6)
    rand_search.fit(X_train, y_train)
    return rand_search


def evaluate_model_on_test_set(rfc, X_test, y_test, results_dir = os.path.join(".", "results")):
    '''
    Testing
    
    Evaluate model on test set (out of sample data)
    
    Parameters
    ----------
    X_test : array-like, shape (n_samples, n_features)
              Testing data.
    
    y_test : array-like, shape (n_samples)
             Target
    
    Returns
    -------
    results_dict : dict of performance metrics
    """
    
    '''
    y_test_pred = rfc.predict_proba(X_test)
    file_name = os.path.join(results_dir, 'performance_metrics.json')

    # Compute metrics
    results_dict = {'ROC AUC': metrics.roc_auc_score(y_test, y_test_pred[:,1]),
                    'Accuracy': metrics.accuracy_score(y_test, y_test_pred[:, 1] > 0.5),
                    'Precision': metrics.precision_score(y_test, y_test_pred[:,1]>0.5),
                    'Recall': metrics.recall_score(y_test, y_test_pred[:,1]>0.5),
                    'F1 score':  metrics.f1_score(y_test, y_test_pred[:,1]>0.5)}

    # Save results as json file
    with open(file_name, 'w') as fp:
        json.dump(results_dict, fp)
    
    print('Performance Metrics on Test Data')
    print('================================')
    print('ROC AUC: ', results_dict['ROC AUC'])
    print('Accuracy: ', results_dict['Accuracy'])
    print('Precision: ', results_dict['Precision'])
    print('Recall: ', results_dict['Recall'])
    print('F1 score', results_dict['F1 score'])

    # Plot ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred[:,1])
    f1 = plt.figure()
    plt.plot(fpr, tpr,  'k')
    plt.plot([0,1], [0,1], 'r--')
    # plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC = '+ str(results_dict['ROC AUC'])[0:6])
    plt.savefig( os.path.join(results_dir, 'roc_auc.png'))
    plt.show()
    
    # Precision-recall curve
    precisions, recall, thresholds = metrics.precision_recall_curve(y_test, y_test_pred[:,1])
    f1_score = 2.0/(1.0/(precisions + 0.0001) + 1.0/(recall + 0.0001))
    f2_score = (2.0**2 + 1.0)/(1.0/(precisions + 0.0001) + 2.0**2/(recall + 0.0001))
    f2 = plt.figure()
    plt.plot(thresholds, precisions[:-1], 'g-', label = "Precision")
    plt.plot(thresholds, recall[:-1],  'b-', label = "Recall")
    plt.plot(thresholds, f1_score[:-1],  'r--', label = "F1 Score")
    plt.plot(thresholds, f2_score[:-1],  'k--', label = "F2 Score")
    plt.xlabel("Threshold")
    plt.legend(loc = "lower left")
    plt.savefig( os.path.join(results_dir, 'precision_recall.png'))
    plt.show()
    
    return results_dict


def get_calibrated_model(rfc, X_train, X_test, y_train, y_test):
    """
    Calibrate the Classifier
    
    Carries out probablity calibration of using the entire dataset, and outputs the final model.
    
    Parameters
    ----------
    rfc : object
          best model
          
    X_train : array-like, shape (n_samples, n_features)
             Training data.
    
    y_train : array-like, shape (n_samples)
             Target
    
    X_test : array-like, shape (n_samples, n_features)
              Training data.
    
    y_test : array-like, shape (n_samples)
             Target
    
    
    Returns
    -------
    model : object
            calibrated model
    
    """
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    # Calibrate model using isotonic regression
    model = CalibratedClassifierCV(rfc, method='isotonic', cv =5)
    model.fit(X_all, y_all)

    return model
