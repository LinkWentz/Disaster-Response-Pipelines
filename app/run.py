import json
import re
import numpy as np
import pandas as pd
import plotly
import sqlite3 as sql

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def title(text):
    text = re.sub('_', ' ', text)
    text = text.title()
    return text

conn = sql.connect('../data/DisasterResponse.db')
# Get main messages data
df = pd.read_sql_query('SELECT * FROM categorized_messages', conn)
df.drop('index', axis = 1, inplace = True)
# Get messages bag of words
bow = pd.read_sql_query('SELECT * FROM messages_bag_of_words', conn)
bow.drop('index', axis = 1, inplace = True)
bow = bow.T
bow['sum'] = np.matrix(bow).sum(axis = 1)
bow = bow.sort_values('sum', ascending = False)

conn.close()
# Get category labels and convert them to title case
category_labels = [title(category) for category in df.columns[5:]]
# load model
model = joblib.load("../models/classifier.pkl")


# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Save user input in query
    query = request.args.get('query', '')
    # Get the predictions for each category
    category_values = model.predict([query])[0]
    # Associate the predictions with the category labels
    category_list = list(zip(category_labels, 
                             category_values))
    # Reshape list into table
    category_table = []
    cats_per_row = 3
    for i in np.arange(0, 36, cats_per_row):
        category_table.append(category_list[i:i+cats_per_row])
    # Extract data needed for visuals
    cat_count_counts = df.groupby('cat_count').count()['message']
    cat_count_labels = list(np.arange(0, 10))
    
    word_counts = bow[0:20]
    word_counts = word_counts['sum']
    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cat_count_labels,
                    y=cat_count_counts
                )
            ],

            'layout': {
                'title': 'Messages By Amount of Categories',
                'yaxis': {
                    'title': ''
                },
                'width':500,
                'height':500
            }
        },
        {
            'data': [
                Bar(
                    x=word_counts,
                    y=word_counts.index,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Most Common Words',
                'yaxis': {
                    'title': '',
                    'categoryorder':'total ascending'
                },
                'width':500,
                'height':500
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, query=query, 
                           graphJSON=graphJSON, category_table=category_table)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()