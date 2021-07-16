import chardet
import sys
import numpy as np
import pandas as pd
import sqlite3 as sql

def condense_category_string(category_string, cat_sep = ';', val_sep = '-'):
    """This function takes a string like this:
        'alpha-0;beta-1;charlie-0;delta-1'
    and converts it to this:
        'beta;delta'
    
    args:
        category_string - string formatted as above
    optional args:
        cat_sep - delimiter between category-value pairs
            defaults to ';'
        val_sep - delimiter between category name and value
            defaults to '-'
    """
    # Seperate categories and their values.
    category_list = category_string.split(cat_sep)
    cat_val_pairs = [category.split(val_sep) for category in category_list]
    # Rejoin all categories with the value '1'.
    dense_cats = [category for category, value in cat_val_pairs if value == '1']
    dense_cats_string = cat_sep.join(dense_cats)
    
    return dense_cats_string

def load_data(messages_filepath, categories_filepath):
    """Load messages and their associated category labels from two csv files
    and merge into a pandas dataframe.
    
    args:
        messages_filepath - path to the messages csv file
        categories_filepath - path to the categories csv file
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    messages_and_categories = messages.merge(categories, on = 'id')
    
    return messages_and_categories

def clean_data(df):
    """Dummy the values in the "categories" column of the provided pandas 
    dataframe, find the original language of each message, and remove all 
    duplicates.
    """
    # Dummy-ify categories.
    df['categories'] = list(map(condense_category_string, df['categories']))
    dummy_categories = df.categories.str.get_dummies(sep = ';')
    df = pd.concat([df[df.columns[:5]], dummy_categories], axis = 1)
    df = df.drop('categories', axis = 1)
    # Make list of all categories
    dummy_columns = list(df.columns)[4:]
    # Since the duplicates are removed by message this sorting ensures that the
    # duplicates with the most columns are preserved.
    df['cat_count'] = df[dummy_columns].sum(axis = 1)
    # Drop duplicate messages.
    df = df.sort_values('cat_count', ascending = False)
    df.drop_duplicates(subset = ['message'], inplace = True)
    # Reorder columns
    column_order = ['id', 'message', 'original', 'genre', 'cat_count']
    column_order.extend(dummy_columns)
    df = df[column_order]
    
    return df

def save_data(df, database_filename):
    """Save provided pandas dataframe to the specified sql database file.
    
    args:
        df - pandas dataframe to save
        database_filename - path of the SQL database file to which the dataframe
        should be saved.
    """
    conn = sql.connect(database_filename)
    df.to_sql('categorized_messages', conn)
    conn.close()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'\
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