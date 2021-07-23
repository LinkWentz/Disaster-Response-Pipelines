# Imports.
import numpy as np
import pandas as pd
import pickle
import re
import sqlite3 as sql
import sys
import warnings
# Custom imports.
sys.path.append('../universal')
import universal_functions as uf
# scikit-learn imports.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def load_data(database_filepath):
    """Load categorized_messages table from provided database and return X and Y
    matrices as well as the labels for the Y matrix.
    
    args:
        database_filepath - location of the sql database from which to pull 
        the messages data.
    """
    # Get data from database.
    conn = sql.connect(database_filepath)
    df = pd.read_sql_query('SELECT * FROM categorized_messages', conn)
    conn.close()
    df.drop('index', axis = 1, inplace = True)
    # Unpack data.
    X = np.array(df['message'])
    Y = np.array(df[df.columns[5:]])
    Y_labels = df.columns[5:]
    
    return X, Y, Y_labels

def build_model():
    """Construct a GridSearchCV model which can be fit to a multi output
    classification problem.
    """
    # Define base pipeline.
    pipeline = Pipeline([
        ('feature_extraction', TfidfVectorizer(tokenizer = uf.tokenize)),
        ('classifier', MultiOutputClassifier(LogisticRegression()))
    ])
    
    param_grid = [
        {'classifier__estimator__fit_intercept': [False, True],
         'classifier__estimator__penalty': ['l2'],
         'classifier__estimator__solver': ['sag', 'saga'],
         'classifier__estimator__max_iter': [1000, 2000, 3000],
         'classifier': [MultiOutputClassifier(LogisticRegression())]},
        {'classifier__estimator__penalty': ['l2'],
         'classifier__estimator__loss': ['hinge', 'squared_hinge'],
         'classifier__estimator__C': [1, 2],
         'classifier__estimator__multi_class': ['ovr', 'crammer_singer'],
         'classifier__estimator__fit_intercept': [True, False],
         'classifier': [MultiOutputClassifier(LinearSVC())]},
        {'classifier__estimator__fit_intercept': [True, False],
         'classifier__estimator__early_stopping': [True],
         'classifier__estimator__max_iter': [1000, 2000, 3000],
         'classifier__estimator__penalty': ['l2'],
         'classifier': [MultiOutputClassifier(Perceptron())]}
    ]
    
    cv_model = GridSearchCV(pipeline, param_grid)
    
    return cv_model

def evaluate_model(model, X_test, Y_test, Y_labels = None):
    """Evaluate the precision, recall, f1, and support of the provided model.
    
    args:
        model - trained sklearn classifier to be evaluated.
        X_test - test data the classifier will predict on.
        Y-test - correct labels to which the predictions will be compared.
    """
    # Generate predictions.
    preds = model.predict(X_test)
    # Score predictions.
    report = classification_report(Y_test, preds, target_names = Y_labels)
    # Output results.
    print(report)

def save_model(model, model_filepath):
    """Export provided model to provided filepath as a pickle.
    
    args:
        model - scikit-learn model.
        model_filepath - path to which the model should be exported.
    """
    pickle.dump(model, open(model_filepath, "wb"))

def main():
    if len(sys.argv) == 3:
        warnings.filterwarnings(action = 'ignore', 
                                category = ConvergenceWarning)
        warnings.filterwarnings(action = 'ignore', 
                                category = UndefinedMetricWarning)
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, Y_labels = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, Y_labels)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()