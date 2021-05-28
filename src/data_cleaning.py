import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


class CleanText(object):
    '''
    The data needs to be entered with the column of interest (text).
    The cleaning includes removing punctuation, lemmatizing, and 
    stopwords for dimensionality reduction.
    '''
    def __init__(self, data, column):
        self.data = data
        self.punctuations = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
        self.column = column


    def remove_punctuation(self, text):
        '''
        In this function we will remove all punctuation from the 
        text based columns.

        Parameter: The text from the column of interest

        Return: The text with punctuations removed
        '''
        for punctuation in self.punctuations:
            text = str(text)
            text = text.replace(punctuation, ' ')

        return text

    
    def lemmatized_text(self, text):
        '''
        This function will group together different forms of the same
        words, allowing utilization of base words for more relevant 
        results.

        Parameters: The text from the column of interest

        Return: Augmented text
        '''
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)
        out = ' '.join([lemmatizer.lemmatize(w) for w in words])
        
        return out


    def remove_stopwords(self, text):
        '''
        This function will remove all stop words from the nltk pacakge
        that traditionally add little signifigance to analysis.

        Parameters: The text from the column of interest

        Return: The text less stopwords
        '''
        all_stopwords = stopwords.words('english')
        words = word_tokenize(text)
        out = ' '.join([w for w in words if not w in all_stopwords])

        return out


    def perform_all_tasks(self):
        '''
        Remove punctuation, stopwords, and lemmatize
        Execute all functions in the class.
        '''
        self.data[self.column] = self.data[self.column].apply(self.remove_punctuation).str.lower()
        self.data[self.column] = self.data[self.column].apply(self.lemmatized_text)
        self.data[self.column] = self.data[self.column].apply(self.remove_stopwords)

        return self.data


if __name__ == '__main__':

    # Test my class
    df = pd.read_csv('../data/Training_DataSet.csv')
    print(df['VehSellerNotes'].head())

    df = CleanText(df, 'VehSellerNotes').perform_all_tasks()
    

    print(df['VehSellerNotes'].head())