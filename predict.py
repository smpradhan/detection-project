#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from sklearn.externals.joblib import load
import matplotlib
matplotlib.use('PS')
from detector import dimensionality_reduction, feature_extractor


def predict(dataset_name, dataset_dir, model_dir, output_dir, output_name):
    """
    Make predictions on query dataset

    Parameters
    ----------
        dataset_name: name of CSV input file to query (without header-row and consisting of two columns: 'url' and 'origin')
        dataset_dir: location of input file
        model_dir: location of models saved during training
        output_dir: name of output directory
        output_name: name of output file

    Returns
    -------
        List of predictions which are saved to output_dir
        """

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read query dataset
    df = pd.read_csv(os.path.join(dataset_dir, dataset_name), header=None, names=['url', 'origin'])

    # Load data pipeline estimator for feature extraction
    data_pipeline = load(os.path.join(model_dir, 'data_pipeline.pkl'))

    # Load estimator for dimensionality reduction 
    dimensionality_reduction = load(os.path.join(model_dir, "dimensionality_reduction.pkl"))

    # Load model
    final_model = load(os.path.join(model_dir, "final_model.pkl"))

    # Load important features selected by feature selection
    selected_features = pd.read_csv(os.path.join(model_dir, "important_features.csv")).T.values[0].tolist()

    # Generate features
    df = data_pipeline.transform(df)

    # Carry out feature selection
    feature_names = data_pipeline.get_params()['feature_extractor'].feature_names
    df = pd.DataFrame(df, columns=feature_names)
    df = df[selected_features]

    # Dimensionality reduction
    df = dimensionality_reduction.transform(df)

    # Predict
    prediction = ['dga' if x == 1 else 'legit' for x in final_model.predict(df)]
    print(prediction)

    # Write to file
    pd.DataFrame(prediction).to_csv(os.path.join(output_dir, output_name), header=None)


def setup_output_dir():
    return

if __name__ == "__main__":
    # Set path of saved models and outputs
    QUERY_DATASET = 'query_dataset.csv'  # Name of CSV input file to query
    QUERY_DATASET_DIRECTORY = './query'  # Location of input file
    MODEL_DIRECTORY = './model'  # Location of models saved during training
    OUTPUT_DIRECTORY = './results'  # Name of output directory
    OUTPUT_NAME = 'prediction.csv'  # Name of output file

    # Make prediction
    predict(QUERY_DATASET, QUERY_DATASET_DIRECTORY, MODEL_DIRECTORY, OUTPUT_DIRECTORY, OUTPUT_NAME)
    print("Predictions saved to: ", OUTPUT_DIRECTORY)
