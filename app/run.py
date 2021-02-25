import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster_messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # Data preparation visual 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Data preparation visual 2
    df_categories = df.iloc[:,3:39]
    df_cat_frequency = df_categories.sum().reset_index(name='frequency')

    # Data preparation visual 3
    df_cat_frequency_desc = df_cat_frequency.sort_values(by='frequency', ascending=False)[1:]
    df_cat_frequency_desc['rel_frequency']=(df_cat_frequency_desc['frequency']/df_cat_frequency.sort_values(by='frequency', ascending=False)[1:].sum()['frequency'])*100
    df_top10_cat_perc=df_cat_frequency_desc[:10]
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
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genres"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_cat_frequency['index'],
                    y=df_cat_frequency['frequency']
                )
            ],

            'layout': {
                'title': 'Distribution of disaster categries',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Disaster Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_top10_cat_perc['index'],
                    y=df_top10_cat_perc['rel_frequency']
                )
            ],

            'layout': {
                'title': 'Top 10 Categories in Percent',
                'yaxis': {
                    'title': "Percent"
                },
                'xaxis': {
                    'title': "Category"
                }
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
