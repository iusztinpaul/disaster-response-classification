import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from sqlalchemy import create_engine

from app import plots
# Import the tokenize function that is used by pickle.
from models.train_classifier import load_model, tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/data.db')
df = pd.read_sql_table('processed', engine)

# load model
model = load_model("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # create visuals
    graphs = [
        plots.create_genre_distribution_plot(df),
        plots.create_message_length_plot(df),
        plots.create_target_distribution_plot(df),
        plots.create_genre_target_distribution_plot(df)
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
    # TODO: Add radio buttons to pick genres.
    query = pd.DataFrame(data={
        "message": [query],
        "genre": ["news"]  # Added the most common genre as a default.
    })

    # use model to predict classification for query
    classification_labels = model.predict(query)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query["message"].values[0],
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
