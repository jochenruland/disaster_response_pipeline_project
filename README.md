# Disaster Response Pipeline Project
A Flask Webapp including a supervised machine learning pipeline to classify disaster messages into 36 categories like water, food, aid, etc.

The project consists of 3 main pillars:
1. The file `process_data.py` contains an ETL Pipeline to extract the data from 2 csv files, transform the data to one dataframe and load the result into a SQLite database file.

2. The file `train_classifier.py` contains a ML Pipeline which loads the data from the SQLite database, splits up the category column and transforms the categories into binary values. The file further contains a custom tokenizer to normalize, tokanize, lemmantize and stem the messages text in preparation for use in the machine learning model. The ML model uses `CountVectorizer(), TfidfTransformer()` and `MultiOutputClassifier(RandomForestClassifier()` as well as `GridSearchCV` to process the text messages and classify them into 36 categories. The trained model is finally saved in a pickle file named `classifier.pkl`.

3. The file `run.py` starts the Flask Webapp and prepares 3 visualizations which are then displayed on the frontend.

## Installation
Clone this repo to the preferred directory on your computer using `git clone https://github.com/jochenruland/disaster_response_pipeline_project`. The file `/app/run.py` starts the Webapp.

### Libraries
You must have installed the following libraries to run the code:
`pandas`
`numpy`
`re`
`chardet`
`sqlalchemy`
`nltk`
`sklearn`
`pickle`
`json`
`plotly`
`flask`
`joblib`

### Program and dataset files:

### FILES
- `data/process_data.py`: The ETL pipeline used to extract, load and transform the data needed for model building.
- `data/DisasterResponse.db`: SQLite database file where the result from the ETL pipeline is saved.
- `models/train_classifier.py`: The Machine Learning pipeline used to train and test the model, and evaluate its results. The model is saved as `classifier.pkl`.
- `app/templates/*.html`: HTML templates for the Webapp.
- `app/run.py`: Starts the Python server for the Webapp.


### Instructions to run the application:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/


## License
The MIT License (MIT)

Copyright (c) 2021 Jochen Ruland

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
