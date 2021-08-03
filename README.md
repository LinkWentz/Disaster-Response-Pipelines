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
