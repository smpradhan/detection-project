## Domain Generation Algorithm Detector
This machine learning classification model detects domain names which are likely generated by malware as randomized rendezvous points. The source code consists of three python files: detector.py, train_test.py, and predict.py

#### Training and testing the model
The model can be trained using the following command:

`python train_test.py`

The paths of the dataset directory, the directory for saving models and results must
be specified in this file.

Upon completion of training and testing the following can be found in the "results" directory:    
a. performance metrics    
b. explained variance plot  
c. roc auc curve   
d. precision-recall curve    
e. learning curve    

#### Making predictions on the new data
Predictions on the new dataset without label can be carried out using the following command:

`python predict.py`

The filename and directory of the dataset, the location of previously trained models,
and the filename and directory for saving output predictions must be specified in this file.

Sample query dataset is included in "query" folder. At the end of analysis the predictions are saved in the directory specified above.

#### Dependencies
The following libraries are needed for this project: Python 3.x, scikit-learn, nltk, tldextract, matplotlib, numpy, pandas.
