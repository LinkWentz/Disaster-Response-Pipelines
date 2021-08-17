# Imports.
import re
# nltk imports.
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(stopwords.words('english'))

def tokenize(string, lemmatize = True):
    """Return an iterable of clean tokens made from the provided string.

    optional args:
        lemmatize - boolean determining whether the tokens should be lemmatized
            defaults to True
    """
    # Normalize string.
    string = string.lower()
    substitutions = [
        ('[\']', '',), # Remove apostrophes.
        ('[^a-zA-Z0-9]', ' ',), # Convert non-alphanumeric chars to spaces.
        (' {2,}', ' ',), # Convert multiple spaces to single space.
        ('^ ', '',), # Remove leading space.
        (' $', '',)  # Remove trailing space.
    ]
    for pattern, replacement in substitutions:
        string = re.sub(pattern, replacement, string)
    # Tokenize string.
    tokens = string.split(' ')
    tokens = [word for word in tokens if word not in stopwords]
    # Lemmatize tokens.
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(word) for word in tokens]

    return tokens
