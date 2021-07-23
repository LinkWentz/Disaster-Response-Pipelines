# Imports.
import re
# nltk imports.
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(stopwords.words('english'))
def tokenize(string):
    """Returns an iterable of clean tokens made from the provided string.
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

    return tokens