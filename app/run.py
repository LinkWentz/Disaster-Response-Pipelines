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
category_names = [title(category) for category in df.columns[3:]]
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    category_table = []
    # save user input in query
    query = request.args.get('query', '')
    if query != '':
        category_labels = model.predict([query])[0]
        
        category_list = list(zip(category_names, 
                                 category_labels))
        category_table = []
        cats_per_row = 3
        for i in np.arange(0, 36, cats_per_row):
            category_table.append(category_list[i:i+cats_per_row])
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals 
    genre_counts = df.groupby('request').count()['message']
    genre_names = list(genre_counts.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
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
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, query=query, graphJSON=graphJSON, category_table=category_table)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()