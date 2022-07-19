# Disaster Response Pipeline Project

## Contents
1. [Project overview](#1)
2. [Instructions for usage](#2)
3. [Description of files](#3)



## 1. Project overview

This project's main goal is to use NLP in order to separate disaster time messages into different categories. This is aimed to help authorities and institutes in prioritizing the help they provide to residents in disaster time.

The data set provided by Figure Eight contains 30000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, Superstorm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and hundreds of different disasters. The messages have been classified in 36 different categories related to disaster response, and they have been stripped of sensitive information in their entirety.


## 2. Instructions for usage:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the root directory to run your web app.
    `python app/run.py`

3. Go to web address in terminal

## 3. File description:

```text
DisasterResponsePipeline/
├──── README.md
├──── ETL_Pipeline Preparation.ipynb # Notebook to prepare cleaning function
├──── ML_Pipeline Preparation.ipynb  # Notebook to test out machine learning algorithms
├──── app/
     ├──── run.py              # Flask file that runs app
     ├──── templates/
            ├──── master.html  # main page of web app
            └──── go.html      # classification result page of web app  
├──── data/
     ├──── disaster_categories.csv  # data to process
     ├──── disaster_messages.csv    # data to process
     ├──── process_data.py
     └──── DisasterResponse.db      # database - saves clean data
└──── models/
     ├──── train_classifier.py
     └──── classifier.pkl           # saved model (RandomForestClassifier)
```
