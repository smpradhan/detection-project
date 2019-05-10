import os
import pandas as pd
from detector import *

# TRAIN/TEST
def train_test_model(dataset_directory, dataset_name, model_directory, results_directory):
    """
    Train and test

    This function trains and tests the classification model, and saves data pipeline and model for future prediction

    Parameters
    ----------
    dataset_directory : directory where dataset is located

    dataset_name : filename of dataset

    model_directory: directory for saving fitted models

    results_directory: directory for saving results
    """

    # Create directories for saving models and results if they don't already exist
    for dir in [model_directory, results_directory]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Perform data preparation steps and split into training and evaluation sets
    X_train, X_test, y_train, y_test = phase1_data_preparation(dataset_directory, dataset_name)

    # Create pipeline for feature extraction and scaling
    data_pipeline = Pipeline([
        ('feature_extractor', feature_extractor()),
        ('scaler', (MinMaxScaler()))
    ])

    # Transform train and test datasets using pipeline to create new features and scale them
    X_train = data_pipeline.fit_transform(X_train)
    X_test = data_pipeline.transform(X_test)
    feature_names  = data_pipeline.get_params()['feature_extractor'].feature_names
    X_train = pd.DataFrame(X_train, columns= feature_names)
    X_test = pd.DataFrame(X_test, columns= feature_names)

    # Save data pipeline
    save_model(data_pipeline, model_directory, 'data_pipeline')
    print("Fit and save data pipeline: done.", "\n")

    # Feature selection: carry out tree based feature selection
    important_features, cv_metrics = select_important_features(X_train, y_train, feature_names)

    # Save important features to file
    pd.DataFrame(important_features).to_csv(os.path.join(model_directory, 'important_features.csv'), index=False)
    print("Feature Selection")
    print("==================")
    print('Total number of features before feature selection: ', X_train.shape[1])
    print('Total number of features after feature selection: ', len(important_features))
    print()
    print("Feature selection: done.", '\n')
    print("Save selected features: done.", '\n')

    X_train = X_train[important_features]
    X_test = X_test[important_features]

    # Dimensionality reduction
    pcat = dimensionality_reduction()
    pcat.fit(X_train)
    X_train = pcat.transform(X_train)
    X_test = pcat.transform(X_test)

    # Save model
    save_model(pcat, model_directory, 'dimensionality_reduction')

    print("Dimensionality reduction: done.", '\n')
    print("Save dimensionality reduction model: done.", '\n')

    # Hyperparameter tuning and selection of the best model using Randomized Search
    rand_search = hyperparameter_tuning(X_train, y_train)
    print("Hyperparameter tuning using Randomized Search: done.", '\n')

    # Print ROC_AUC of best model
    rfc = rand_search.best_estimator_
    mean_auc_roc = rand_search.cv_results_['mean_test_score'].max()
    print('ROC AUC of the best model selected using randomized search: ', mean_auc_roc, '\n')
    print('Selection of the best model: done.' )

    # Compute and save evaluation metrics
    evaluate_model_on_test_set(rfc, X_test, y_test, results_directory)
    print('Compute and save evaluation metrics for the best model: done.', '\n')

    # Plot Learning Curve
    title = "Learning Curve"
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    cv_splits = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    plot = plot_learning_curve(rfc, title, X_all, y_all, ylim=(0.50, 1.01), cv=cv_splits, n_jobs=4)
    plot.savefig(os.path.join(results_directory, 'learning_curve.png'))
    print('Generate learning curve: done')

    # Train Model and calibrate the classifier using the entire dataset
    final_model = get_calibrated_model(rfc, X_train, X_test, y_train, y_test)
    save_model(final_model, model_directory, 'final_model')
    print('Train classifier and calibrate it using the entire dataset: done', '\n')
    print("Training and testing completed!")

if __name__ == "__main__":
    DATASET_DIRECTORY = './dataset'
    DATASET_NAME = "dga-dataset.txt"
    MODEL_DIRECTORY = './model'
    RESULTS_DIRECTORY = './results'

    train_test_model(DATASET_DIRECTORY, DATASET_NAME, MODEL_DIRECTORY, RESULTS_DIRECTORY)