# Imports.
import numpy as np
import os
import pandas as pd
import pickle
import re
import sqlite3 as sql
import sys
import warnings
# Custom imports.
cwd = os.getcwd()

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append('../universal')
import universal_functions as uf

os.chdir(cwd)
# scikit-learn imports.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def load_data(database_filepath):
    """Load categorized_messages table from provided database and return X and Y
    dataframes.
    
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
    X = df['message']
    Y = df[df.columns[5:]]
    
    return X, Y

def build_model():
    """Construct a GridSearchCV model which can be fit to a multi output
    classification problem.
    """
    # Define base pipeline.
    pipeline = Pipeline([
        ('feature_extraction', TfidfVectorizer(tokenizer = uf.tokenize)),
        ('classifier', MultinomialNB())
    ])
    
    param_grid = [
        {'classifier__estimator__criterion': ['gini', 'entropy'],
         'classifier__estimator__max_depth': [None, 1000, 2000],
         'classifier__estimator__max_features': ['auto', 'sqrt', 'log2'],
         'classifier': [MultiOutputClassifier(DecisionTreeClassifier())]},
        {'classifier': [MultiOutputClassifier(MultinomialNB())]}
    ]
    
    cv_model = GridSearchCV(pipeline, param_grid)
    
    return cv_model

def evaluate_model(model, X_test, Y_test):
    """Evaluate the precision, recall, f1, and support of the provided model.
    
    args:
        model - trained sklearn classifier to be evaluated.
        X_test - test data the classifier will predict on.
    """
    # Generate predictions.
    preds = model.predict(X_test)
    # Score predictions.
    report = classification_report(Y_test, preds, target_names = Y_test.columns)
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
        warning_types = [ConvergenceWarning, UndefinedMetricWarning]
        for warning in warning_types:
            warnings.filterwarnings(action = 'ignore', category = warning)
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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