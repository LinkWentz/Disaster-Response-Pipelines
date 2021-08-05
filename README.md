# Disaster Response Pipelines
This project was made in an effort to classify messages sent during disasters in order to direct those messages to the appropriate response agencies.

This project consists of 2 pipelines and a web app. The first pipeline processes data from Figure Eight containing messages and associated categories, and saves that data to an SQL database. The second pipeline builds a model which assigns any combination of 35 categories to a given message. The web app allows you to interface with the model, giving you the ability to input a message which the model will then categorize. The web app also features some information about the data set which should help you to write your messages.

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
The general order of operations for using this project should be the first: process the data, second: train the classifier and third: run the web app. However you can also run the web app without going through the first 2 steps, as I have included a pre-trained classifier in the models folder. The default commands to run these scripts are below, though do note that your working directory needs to be the directory in which the script is located.

```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db # Process the data.
python train_classifier.py ../data/DisasterResponse.db classifier.pkl                    # Train the classifier.
python run.py classifier.pkl                                                             # Run the web app.
```

You can then connect to the web app through the port 3001. Once in the web app you will be met by a grid containing the names of each of the 35 categories. On the left there will be a text box in which you can enter a message you would like to classify. Once classified any category into which the message fits will be highlighted in green. Lastly you'll see two visualisations below the main interface which, if I've done my job right, should be self explanatory.

### Sources
[Appen](https://appen.com/): I got this data from Figure Eight through Udacity. Figure Eight has since been aquired by Appen and their URL now redirects to Appen so that is probably where they'd like me to send you now.

### Dependencies
- [NumPy](https://pypi.org/project/numpy/): Package for array computing with Python.
- [Pandas](https://pypi.org/project/pandas/): Python package that provides fast, flexible, and expressive data structures.
- [Plotly](https://pypi.org/project/plotly/): An open-source, interactive data visualization library for Python.
- [Natural Language Toolkit](https://pypi.org/project/nltk/): Python package for natural language processing.
- [scikit-learn](https://pypi.org/project/scikit-learn/): A set of python modules for machine learning and data mining.
- [Flask](https://pypi.org/project/Flask/): A simple framework for building complex web applications.
- [Bootstrap](https://getbootstrap.com/): The most popular framework for building responsive, mobile-first sites.
