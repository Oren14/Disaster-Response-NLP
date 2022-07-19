# import libraries
import sys
import pandas as pd
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(right=categories, on='id', how='inner')

    return df



def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    categories.head()

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[:-2]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')

    categories.drop(index=categories[categories['related']>1].index.tolist(),inplace=True)
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    dupy = df[df.duplicated()].index.tolist()
    df.drop(dupy, inplace=True)

    df.drop(index=df[df['other_aid'].isna()].index.tolist(),inplace=True)

    return df



def save_data(df, database_filename):
    engine = sqlalchemy.create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('Recovery_table', engine, index=False)



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
