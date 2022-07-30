import json
import joblib
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
# from sklearn.externals import joblib
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


# load data
engine = create_engine('sqlite:///../data/data.db')
df = pd.read_sql_table('processed', engine)

# load model
model = joblib.load("../models/random_forest.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Compute genre bar plot values
    genre_counts = df["genre"].value_counts()
    genre_names = genre_counts.index
    genre_counts = genre_counts.values

    # Compute message length histogram values.
    message_length_data = df["message"].str.len()
    message_length_data = message_length_data[message_length_data < 1000]

    # Target distribution.
    target_distribution_1 = df.drop(columns=['id', 'message', 'original', 'genre']).mean()
    target_distribution_0 = 1 - target_distribution_1  # Inverse the values, therefore to start from 0 and end with 1.
    class_names = target_distribution_1.index

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Histogram(
                    y=message_length_data
                )
            ],

            'layout': {
                'title': 'Histogram of Message Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Length"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=class_names,
                    y=target_distribution_1,
                    name='Class = 1'
                ),
                Bar(
                    x=class_names,
                    y=target_distribution_0,
                    name='Class = 0',
                    marker=dict(
                        color='rgb(212, 228, 247)'
                    )
                )
            ],
            'layout': {
                'title': 'Distribution of Each Target Class',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "Target",
                },
                'barmode': 'stack'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    # TODO: Fix column misalignment.
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
