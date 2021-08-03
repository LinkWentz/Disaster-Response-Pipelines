# Disaster Response Pipelines
This project was made in an effort to classify messages sent during disasters in order to direct those messages to the appropriate response agencies. This project consists of 2 pipelines and a web app. The first pipeline processes data from Figure Eight containing messages and associated categories, and saves that data to an SQL database. The second pipeline builds a model which assigns any combination of 35 categories to a given message. The web app allows you to interface with the model, giving you the ability to input a message which the model will then categorize. The web app also features some information about the data set which should help you to write your messages.

### File Structure
```
.
├── app
│   ├── static
│   │   ├── stylesheet.css
│   ├── templates
│   │   ├── master.html
│   └── run.py
├── data
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── models
│   ├── classifier.pkl
│   └── train_classifier.py
├── models
│   └── universal_functions.py
└── README.md
```

### Usage
The general order of operations for using this project should be the first process the data second to train the classifier and third run the web. However you can also run the web up without going through the first 2 steps. As I have included the clean data and train classifier already.

To process the data you must first change your working directory to data folder. Once you've done this you can use the,nd line to run process data.pi. Process data takes 3 arguments which are all file paths. The first is the messages themselves. The second is the categories that correspond to those messages. And the third is the name of the database you would like to output the cleaned data to. If the arguments are not provided in the program is run anyway the output will be a default,nd which you can copy and paste.

To train the classifier you do much the same thing. First changer directory to the models directory. Then run train classifier.pi which takes 2 arguments the first should be the file path for the database that was generated from the data processing step and the second should be the name of the file to which the classifier should be saved. Again running the program with out these inputs. Will result in a. Default,nd which you can copy paste.

To run the web app you again need to change your working directory this time to be out fuller. He done so then run run.pi. This cryptic so only one argument the name of the classifier that you trained previously. You can then connect to the web up through the port 3001.

Once in the web app you will be met by a grid. Containing the names of each of the 35 categories. On the left there will be. A. Text box in which you can enter message you would like to classify. Once classified any category into which the message fits will be highlighted green. Once classified any category into which the message fits will be highlighted green. Lastly you'll see to me visualisations below the main interface which if I've done my job right should be self explanatory.

# Dependencies
- [Python](https://www.python.org/)
- [NumPy](https://pypi.org/project/numpy/): Package for array computing with Python.
- [Pandas](https://pypi.org/project/pandas/): Python package that provides fast, flexible, and expressive data structures.
- [Plotly](https://pypi.org/project/plotly/): An open-source, interactive data visualization library for Python.
- [Natural Language Toolkit](https://pypi.org/project/nltk/): Python package for natural language processing.
- [scikit-learn](https://pypi.org/project/scikit-learn/): A set of python modules for machine learning and data mining.
- [Flask](https://pypi.org/project/Flask/): A simple framework for building complex web applications.
- [Bootstrap](https://getbootstrap.com/): The most popular framework for building responsive, mobile-first sites.
