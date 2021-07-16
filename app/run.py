import json
import re
import plotly
import numpy as np
import pandas as pd

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorized_messages', engine)

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
    category_labels = model.predict([query])[0]
    # Associate the predictions with the category labels
    category_list = list(zip(category_labels, 
                             category_labels))
    # Reshape list into table
    category_table = []
    cats_per_row = 3
    for i in np.arange(0, 36, cats_per_row):
        category_table.append(category_list[i:i+cats_per_row])
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_labels = list(genre_counts.index)
    
    cat_count_counts = df.groupby('cat_count').count()['message']
    cat_count_labels = list(np.arange(0, 10))
    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_labels,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': '',
                'yaxis': {
                    'title': "Count"
                },
                'width':330,
                'height':330
            }
        },
        {
            'data': [
                Bar(
                    x=cat_count_labels,
                    y=cat_count_counts
                )
            ],

            'layout': {
                'title': '',
                'yaxis': {
                    'title': "Distribution of Amount of Categories"
                },
                'width':330,
                'height':330
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