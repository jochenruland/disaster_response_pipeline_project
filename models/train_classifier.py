# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier




def load_data(database_filepath):
    ''' Function that loads dataset from a database into a Pandas Dataframe
        and splits the data into an Input Dataset and an Output Dataset

        Arguments:
            database_filepath - filepath

        Returns:
            X - pandas Series
            Y - pandas Dataframe
            category_names  - list of Y column names
    '''
    engine = create_engine('sqlite:///Desaster_messages_categories.db')
    df = pd.read_sql('SELECT * FROM Desaster_messages_categories', engine)

    X = df['message'] # define input data
    Y = df.iloc[:,3:39] # define output data
    category_names = list(Y.columns)
    print(Y.head(1), Y.shape, category_names)
    print(X.head(1), X.shape)

    return X, Y, category_names



def tokenize(text):
    ''' Function that tokenizes a text input, removes stopwords and then
        lemmatizes and stems the tokens and adds them to a list

        Arguments:
            text - string

        Returns:
            clean_tokens - list
    '''
    # Normalize text and tokenize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)

    # Remove Stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize and stem
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tokens = []
    for tok in tokens:
        clean_tok1 = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok2 = stemmer.stem(clean_tok1)
        clean_tokens.append(clean_tok2)

    return clean_tokens




def build_model():
    ''' Function that creates a supervised ML model, putting the estimators into
        a pipeline object and adding GridSearchCV for optimization

        Arguments:
            none

        Returns:
            cv - ML model
    '''
    # specify estimators for ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
        'clf__estimator__min_samples_split': [2, 3],
        #'clf__estimator__n_estimators': [50, 100],
        }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    ''' Function that makes a prediction on the test data using the previously trained
        model then evaluates the results
        Arguments:
            model -
            X_test - Test input, pandas Series
            Y_test - Test output, pandas Dataframe
            category_names - list

            Returns:
            none
    '''
    # Prediction using trained model
    Y_pred= model.predict(X_test)

    # Evalation of model
    for c in category_names:
        print(c)
        print(classification_report(Y_test[c], Y_pred[c]))


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
