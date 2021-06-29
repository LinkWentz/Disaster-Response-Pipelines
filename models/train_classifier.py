import sys
import numpy as np
import pandas as pd
import pickle
import re
import sqlite3 as sql

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stopwords = set(stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    """This function loads categorized_messages from SQL database and splits the
    data into X and Y dataframes.
    
    args:
        database_filepath - location of the sql database from which to pull data
    """
    # Get data from database.
    conn = sql.connect(database_filepath)
    df = pd.read_sql_query('SELECT * FROM categorized_messages LIMIT 2000', conn)
    conn.close()
    # Unpack data.
    X = df['message']
    Y = df[df.columns[2:]]
    
    return X, Y

def tokenize(string):
    """Returns an iterable of clean tokens made from the provided string.
    """
    lemmer = WordNetLemmatizer()
    # Normalize string.
    string = string.lower()
    string = re.sub('[\']', '', string)# Remove apostrophes
    string = re.sub('[^a-zA-Z0-9]', ' ', string)# Convert non-alphanum to space
    string = re.sub(' {2,}', ' ', string)# Convert multiple spaces to single
    string = re.sub('^ ', '', string)# Remove leading space
    string = re.sub(' $', '', string)# Remove trailing space
    # Tokenize string.
    tokens = string.split(' ')
    tokens = [word for word in tokens if word not in stopwords]
    tokens = list(map(lemmer.lemmatize, tokens))
    
    return tokens

def build_model(classifier = DecisionTreeClassifier()):
    """Constructs model using a classifier. This model can be used to predict
    multi-label output from text data.
    
    optional args:
        classifier - sklearn classifier with multilabel output
            defaults to DecisionTreeClassifier()
    """
    # Define base pipeline.
    pipeline = Pipeline([
        ('feature_extraction', TfidfVectorizer(tokenizer = tokenize)),
        ('classifier', classifier)
    ])
    # Set up grid search cross validation.
    param_grid = {
        'classifier__random_state': [20]
    }
    cv_model = GridSearchCV(pipeline, param_grid)
    
    return cv_model

def score(y_true, y_pred, avg_setting = 'macro'):
    """Returns the F1, precision, and recall scores of the predicted labels 
    compared to the correct labels.
    
    args:
        y-true - correct labels
        t-pred - predicted labels
    optional args:
        avg_setting - value for the average parameter of every sklearn scorer.
            defaults to 'macro'
    """
    f1 = f1_score(y_true, y_pred, average = avg_setting)
    precision = precision_score(y_true, y_pred, average = avg_setting)
    recall = recall_score(y_true, y_pred, average = avg_setting)
    
    return f1, precision, recall

def evaluate_model(model, X_test, Y_test):
    """Evaluates the f1, precision, and recall of the provided model.
    
    args:
        model - trained sklearn classifier to be evaluated
        X_test - test data the classifier will predict on
        Y-test - correct labels to which the predictions will be compared
    """
    # Generate predictions.
    preds = model.predict(X_test)
    # Score predictions.
    f1, precision, recall = score(Y_test, preds)
    # Output results.
    result_message = ('F1 score: {:.2f}\n'\
                      'Precision: {:.2f}\n'\
                      'Recall: {:.2f}')
    print(result_message.format(f1, precision, recall))

def save_model(model, model_filepath):
    """Exports provided model as pickle to provided filepath./
    
    args:
        model - sklearn classifier
        model_filepath - path to which the model should be exported
    """
    pickle.dump(model, open(model_filepath, "wb"))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(selected_classifier)
        
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