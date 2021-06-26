import numpy as np
import pandas as pd
import sqlite3 as sql

# Import data.
msgs = pd.read_csv('messages.csv')
cats = pd.read_csv('categories.csv')

cat_msgs = msgs.merge(cats, right_on = 'id', left_on = 'id')

# Merge tables.
def extract_correct_cats(categories):
    categories = categories.split(';')
    correct_cats = [cat[:-2] for cat in categories if cat[-1] == '1']
    return ';'.join(correct_cats)
cat_msgs['categories'] = list(map(extract_correct_cats, cat_msgs.categories))
cat_msgs = pd.concat([cat_msgs.drop('categories', axis = 1), cat_msgs.categories.str.get_dummies(sep = ';')], axis = 1)

# Remove duplicates.
cat_msgs['cat_count'] = np.matrix(cat_msgs[list(cat_msgs.columns)[4:]]).sum(axis = 1)
cat_msgs = cat_msgs.sort_values('cat_count', ascending = False)
cat_msgs.drop_duplicates(subset = ['message'], inplace = True)
cat_msgs = cat_msgs.drop('cat_count', axis = 1).sort_values('id')

# Write to SQL database.
conn = sql.connect('drp.db')
cat_msgs.to_sql('categorized_messages', conn)
conn.close()