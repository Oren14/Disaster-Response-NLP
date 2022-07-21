# import libraries
import sys
import pandas as pd
import numpy as np
import sqlalchemy
import nltk
from nltk.tokenize import word_tokenize,wordpunct_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.svm import SVC
import unittest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    
    '''
    Takes data from a database and loads it to a pandas series.

            Parameters:
                    database_filepath (string): database file path.
            Returns:
                    X (pandas.Series) : masseges from database
                    y (pandas.Series) : masseges classification
                    category_names (List) : the massgess categories
    '''
    
    
    # load data from database
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Recovery_table', engine)

    X = df['message']
    y = df.drop(['id','message','original','genre'], axis=1)
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):
    
    '''
    Takes text and tokenize/lemmatize/commen case it.

            Parameters:
                    text (string): raw text
            Returns:
                    text (list): toknized words
    '''
    
    # init Lemmatizer and Tokenizer for acceptable NLP words (no punctuation)
    wnl = WordNetLemmatizer()
    tokenizer = nltk.RegexpTokenizer(r"\w+")

    text = tokenizer.tokenize(text.lower())
    text = [wnl.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return text



def build_model():

    '''
    Creates ML pipeline:
    1. tokenize the data
    2. transirm it to Tdf form
    3. creates the ML meodel instance - random forest


            Returns:
                    pipeline (pipeline): a fited pipeline for NLP
    '''
    


    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('rf', MultiOutputClassifier(rf()))
                    ])
    
    parameters = {'rf__estimator__n_estimators': [50,100,120]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    
    '''
    Iterate through the classes and compute f1, precision and accuracy

            Parameters:
                    model (sklearn.ensemble.model):  model for NLP
                    X_test (pandas.Series): massges data
                    Y_test (pandas.Series): massges lables
                    category_names (list): category names of lables
                    
            Returns:
                    None
    '''
    


    for i,cat in enumerate(category_names):
        print('For category "{}" this is the results:'.format(cat))
        print(classification_report(y_pred=y_pred.reshape(36,-1)[i].astype('int'),
                                    y_true=np.array(Y_test).reshape(36,-1)[i].astype('int'),

                                   )
             )
        print('--------------------------')



def save_model(model, model_filepath):
    
    '''
    Takes a trained model and saved it

            Parameters:
                    model (sklearn.ensemble.model): model for NLP
                    model_filepath (string): path to save the model
                    
            Returns:
                    None
    '''
    
    # create an iterator object with write permission - model.pkl
    with open(model_filepath, 'wb') as files:
        pickle.dump(model, files)


        

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
