import sys
import pandas as pd
import numpy as np
import re
import chardet
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' Function that takes two CSV files from the given paths, creates two
        dataframes, merges the two dataframes on their common ID and returns
        the joint dataframe

        Arguments:
            messages_filepath - pathname
            categories_filepath - pathname

        Returns:
            df - dataframe object
    '''

    # load messages dataset
    messages = pd.read_csv(messages_filepath, sep=',', encoding='utf-8', dtype='str')

    # load categories dataset
    categories = pd.read_csv(categories_filepath, sep=',', dtype=str)

    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id').sort_values(by='id')

    return df


def clean_data(df):
    ''' Function that transforms the catagorical data into a new dataframe of
        binary values (0,1) for the n categries, then drops the 'categories'
        column from the original dataframe and concats the 2 dataframes to a new
        one showing the categorical value for each message in binary values

        Arguments:
            df - dataframe object

        Returns:
            df - cleaned dataframe object
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    row = row.apply(lambda x: x[:len(x)-2])
    category_colnames = list(row)

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[len(x)-1]) # only once
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(['categories'], inplace=True, axis=1)

    # drop the original column from `df`
    df.drop(['original'], inplace=True, axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # remove dublicates
    df.drop_duplicates(subset=['message'], keep='first', inplace=True)

    return df



def save_data(df, database_filename):
    ''' Save the clean dataset into an sqlite database '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Desaster_messages_categories', engine, if_exists='replace',index=False)



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
