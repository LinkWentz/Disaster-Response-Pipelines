# Disaster Response Pipelines
This project was made in an effort to classify messages sent during disasters in order to direct those messages to the appropriate response agencies.

This project consists of 2 pipelines and a web app. The first pipeline processes data from Figure Eight containing messages and associated categories, and saves that data to an SQL database. The second pipeline builds a model which assigns any combination of 36 categories to a given message. The web app allows you to interface with the model, giving you the ability to input a message which the model will then categorize. The web app also features some information about the training data set which should help you to draft your messages.

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
│   └── train_classifier.py
├── screenshots
│   ├── Classified Message.jpg
│   └── Home Page.jpg
├── universal
|   └── universal_functions.py
└── README.md
```

### Usage
The general order of operations for using this project should be the first: process the data, second: train the classifier and third: run the web app. However you can also skip the first step if you so wish, as I have included a premade database in the data folder. Each step should be performed from a terminal instance, first by changing to the appropriate directory for the step, then running the appropriate command. I've provided a guide to what the appropriate directory and command are for each step below.

```
step                       |    directory    |    command
# Process the data.             ./data            python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
# Train the classifier.         ./models          python train_classifier.py ../data/DisasterResponse.db classifier.pkl
# Run the web app.              ./app             python run.py ../data/DisasterResponse.db ../models/classifier.pkl
```

You can then connect to the web app through the address "localhost:3001". Once in the web app you will be met by a grid containing the names of each of the 36 categories. On the left there will be a text box in which you can enter a message you would like to classify. This should be what you see when you first open the web app:

![Home Page](https://github.com/LinkWentz/Disaster-Response-Pipelines/blob/master/screenshots/Home%20Page.jpg)

Once classified any category into which the message fits will be highlighted in green. This is what it will look like once you've classified a message (results may vary):

![Classified Message](https://github.com/LinkWentz/Disaster-Response-Pipelines/blob/master/screenshots/Classified%20Message.jpg)

Lastly you'll see two visualizations below the main interface which, if I've done my job right, should be self explanatory. If you have trouble getting this project to work, first verify that your environment has all the necessary dependencies.

### How it works
Now that we have a brief overview of the pipeline, let's look at each step in detail. First we'll discuss the data processing step.

The first action taken in the data processing step is to load the data. The data is split into 2 tables: messages and categories. These tables are joined as soon as they're loaded. Some simple text processing is used to convert the categories-which are encoded as single strings-into features. Any duplicate messages are then removed; with the duplicate that has the most categories being preserved. Additionally a separate table is generated containing the 20 most common words across all of the messages. Of note is that the text processing method that is used means that if a category is never applied to a message, that category does not become a feature in the final table. To compensate for this, a blank message that is classified as every category is added and subsequently removed from the table in order to "initialize" each feature.

Once the data is processed it can then be used to train the classifier. The data is first split 80/20 into train and test sets respectively. Then, using cross validation, the best combination of hyperparameters for the best classifier is selected and that model is then exported into a pickle file. Note here that the range of useable classifiers was significantly reduced by the inclusion of features with only one value, specifically the "child_alone" column.

Once the classifier is trained, the backend of the web app is complete, and the web app can be run. The main feature of the web app (the ability to classify messages) uses the pickled classifier to predict the categorization of the provided messages. As for the visualizations, the first (representing the amount of messages that had 0, 1, 2... etc. categories attributed to them) was made by simply summing the category values for each message. The second visualization uses the most common words table that was made in the data processing step.

### Sources
- [Appen](https://appen.com/): I got this data from Figure Eight through Udacity. Figure Eight has since been acquired by Appen and their URL now redirects to Appen, so that is probably where they'd like me to send you now.

### Dependencies
- [NumPy 1.20.2](https://pypi.org/project/numpy/): Package for array computing with Python.
- [Pandas 1.2.4](https://pypi.org/project/pandas/): Python package that provides fast, flexible, and expressive data structures.
- [Plotly 4.14.3](https://pypi.org/project/plotly/): An open-source, interactive data visualization library for Python.
- [Natural Language Toolkit 3.6.2](https://pypi.org/project/nltk/): Python package for natural language processing.
- [scikit-learn 0.24.2](https://pypi.org/project/scikit-learn/): A set of python modules for machine learning and data mining.
- [Flask 1.1.2](https://pypi.org/project/Flask/): A simple framework for building complex web applications.
- [Bootstrap 4.3.1](https://getbootstrap.com/): The most popular framework for building responsive, mobile-first sites.
