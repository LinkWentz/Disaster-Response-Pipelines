import numpy as np
import pandas as pd
import sqlite3 as sql

messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')

# Dummy-ify categories.
def condense_category_string(category_string,
                             cat_sep = ';',
                             val_sep = '-'):
    """This function takes a string like this:
        'alpha-0;beta-1;charlie-0;delta-1'
    And converts it to this:
        'beta;delta'
    
    args:
        category_string - string formatted as above
    optional args:
        cat_sep - delimiter between category-value pairs
            defaults to ';'
        val_sep - delimiter between category name and value
            defaults to '-'
    """
    # Convert categories into category-value pairs.
    category_list = category_string.split(cat_sep)
    labelled_cats = [category.split(val_sep) for category in category_list]
    # Keep every category which has a value of '1'.
    dense_cats = [category for category, value in labelled_cats if value == '1']
    
    dense_cats_string = cat_sep.join(dense_cats)
    
    return dense_cats_string

categories['categories'] = list(map(condense_category_string, 
                                    categories['categories']))
dummy_categories = categories.categories.str.get_dummies(sep = ';')
categories = pd.concat([categories['id'], dummy_categories], axis = 1)

# Merge tables
messages = messages.merge(categories, on = 'id')

# Remove duplicates.
# Sort duplicates by amount of categories to ensure that the duplicate with the
# most categorizations is the one kept.
category_matrix = np.matrix(messages[list(messages.columns)[4:]])
cat_count = category_matrix.sum(axis = 1)
messages['cat_count'] = cat_count
messages = messages.sort_values('cat_count', ascending = False)
# Some duplicate messages had different IDs, making this a better approach.
messages.drop_duplicates(subset = ['message'], inplace = True)

messages = messages.drop('cat_count', axis = 1).sort_values('id')

# Write to SQL database.
conn = sql.connect('disaster_messages.db')
messages.to_sql('categorized_messages', conn)
conn.close()