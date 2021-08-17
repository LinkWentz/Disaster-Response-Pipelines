# Imports.
from itertools import chain
import numpy as np
import os
import pandas as pd
import re
import sqlite3 as sql
import sys
# Custom imports.
cwd = os.getcwd()

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append('../universal')
import universal_functions as uf

os.chdir(cwd)

def load_data(messages_filepath, categories_filepath):
    """Load messages and their associated category labels from their csv files
    and merge them into a pandas dataframe.

    args:
        messages_filepath - path to the messages csv file.
        categories_filepath - path to the categories csv file.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    messages_and_categories = messages.merge(categories, on = 'id')

    return messages_and_categories

def condense_category_string(category_string, cat_sep = ';', val_sep = '-',\
                             ones = []):
    """Take a string like this:
        'alpha-0;beta-1;charlie-2;delta-1'
    and convert it to this:
        'beta;delta'

    Any nonbinary category values are converted to 0 by default. To control this
    behaivior see the optional arguments.

    args:
        category_string - string formatted as above.
    optional args:
        cat_sep - delimiter between category-value pairs.
            defaults to ';'
        val_sep - delimiter between category name and value.
            defaults to '-'
        ones - iterable of possible nonbinary category values which should be
        replaced with a 1 instead of a 0.
            defaults to []
    """
    # Make sure all values are strings in replace_with_one argument.
    ones = set(map(str, ones))
    # Seperate categories and their values.
    category_list = category_string.split(cat_sep)
    cat_val_pairs = [category.split(val_sep) for category in category_list]
    # Convert any nonbinary values to a pre-specified value or 0.
    cat_val_pairs = np.array(cat_val_pairs)
    values = cat_val_pairs[:, 1]
    values = list(map(lambda x : '1' if x in ones or x == '1' else '0',
                      values))
    cat_val_pairs[:, 1] = values
    # Rejoin all categories with the value '1'.
    dense_cats = [category for category, value in cat_val_pairs if value == '1']
    dense_cats_string = cat_sep.join(dense_cats)

    return dense_cats_string

def clean_data(df):
    """Dummy the values in the "categories" column of the provided pandas
    dataframe, add a column representing the number of categories assigned to
    each message, and drop duplicate messages.
    """
    # Initialize all categories using temporary entry to preserve categories
    # with no values.
    initializer = df.loc[0].copy()
    initializer['categories'] = re.sub('0', '1', initializer['categories'])
    df = df.append(initializer)
    # Dummy-ify categories.
    df['categories'] = list(map(condense_category_string, df['categories']))
    dummy_categories = df.categories.str.get_dummies(sep = ';')
    df = pd.concat([df[df.columns[:5]], dummy_categories], axis = 1)
    df = df.drop('categories', axis = 1)
    # Remove temporary entry
    df = df[:df.shape[0] - 1]
    # Make list of all categories.
    dummy_columns = list(df.columns)[4:]
    # Since the duplicates are removed by message this sorting ensures that the
    # duplicates with the most columns are preserved.
    df['cat_count'] = df[dummy_columns].sum(axis = 1)
    # Drop duplicate messages.
    df = df.sort_values('cat_count', ascending = False)
    df.drop_duplicates(subset = ['message'], inplace = True)
    # Reorder columns.
    column_order = ['id', 'message', 'original', 'genre', 'cat_count']
    column_order.extend(dummy_columns)
    df = df[column_order]

    return df

def get_most_common_words(df, count = 20, column_name = 'message'):
    """Count the occurrences of each word in the specified column of the
    provided dataframe and return a series, the index of which contains the top
    most occuring words, and the values of which are the amounts of occurences
    of those words.

    args:
        df - dataframe with column to analyze.
    optional args:
        count - the amount of words to include in the resultant dataframe.
            defaults to 20
        column_name - name of column to analyze in provided dataframe.
            defaults to 'message'
    """
    strings = df[column_name].copy()
    # Convert the strings to tokens.
    token_lists = strings.apply(np.vectorize(uf.tokenize, lemmatize = False))
    # Combine every list of tokens into a single series.
    tokens = pd.Series((chain.from_iterable(token_lists)))
    most_common_words = tokens.value_counts(sort = True)
    # Reduce to the top (count) tokens.
    most_common_words = most_common_words.iloc[0:count]

    return most_common_words

def save_data(data, table_name, database_filename):
    """Save provided pandas dataframe or series to the specified sql database
    file.

    args:
        data - pandas dataframe or series to save.
        database_filename - path of the SQL database file to which the dataframe
        should be saved.
    """
    conn = sql.connect(database_filename)
    data.to_sql(table_name, conn, if_exists = 'replace')
    conn.close()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'\
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        clean_df = clean_data(df)
        most_common_words = get_most_common_words(df, count = 20)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(clean_df, 'categorized_messages', database_filepath)
        save_data(most_common_words, 'most_common_words', database_filepath)

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
