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

    
    def escape_html(self, text):
        '''
        In this function we will attempt to remove all html punctuation
        and similar from the text within the necessary columns. Replacing
        with a suitable replacement, then removing double blank spaces if 
        created.

        Parameter: The text from the column of interest

        Return: The text with the html remove/replaced
        '''
        text = str(text)
        text = text.replace('#x27;', '')
        text = text.replace('#x27;17;', '')
        text = text.replace('&lt;br&gt;', ' ')
        text = text.replace('&#x3D;', ' ')
        text = text.replace('&lt;', ' ')
        text = text.replace('&gt;', ' ')
        text = text.replace('&amp;', 'and')

        return text


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


    def clear_spaces(self, text):
        '''
        Remove the extra spaces that were created from previous
        steps in this class.

        Parameters: The text from the column of interest

        Return: The text less additional spacing
        '''
        text = ' '.join(text.split())

        return text


    def perform_all_tasks(self):
        '''
        Remove html, punctuation, stopwords, extra spaces and lemmatize
        Execute all functions in the class.
        '''
        self.data[self.column] = self.data[self.column].apply(self.escape_html)
        self.data[self.column] = self.data[self.column].apply(self.remove_punctuation).str.lower()
        self.data[self.column] = self.data[self.column].apply(self.lemmatized_text)
        self.data[self.column] = self.data[self.column].apply(self.remove_stopwords)
        self.data[self.column] = self.data[self.column].apply(self.clear_spaces)

        return self.data


def ext_color_reduction(data, column):
    '''
    This function will attempt to reduce the number of features when one hot
    encoding the colors for exterior.

    Parameters: The data, and the column of interest

    Return: Minimized color scale
    '''
    df = data.copy()
    df[column] = df[column].fillna('unspecified')

    df.loc[df[column].str.contains('billet'), column] = 'billet silver metallic'
    df.loc[df[column].str.contains('bright white'), column] = 'bright white metallic'
    df.loc[df[column].str.contains('brilliant black'), column] = 'brillant black pearl'
    df.loc[df[column].str.contains('black forest green'), column] = 'black forest green pearl'
    df.loc[df[column].str.contains('brilliant silver'), column] = 'silver'
    df.loc[df[column].str.contains('cashmere'), column] = 'cashmere pearl'
    df.loc[df[column].str.contains('certified'), column] = ''
    df.loc[df[column].str.contains('deep'), column] = 'red crystal pearl'
    df.loc[df[column].str.contains('diamond'), column] = 'diamond black pearl'
    df.loc[(df[column].str.contains('granite')) & (df['VehMake'] == 'jeep'), column] = 'granite crystal metallic'
    df.loc[df[column].str.contains('ivory'), column] = 'ivory'
    df.loc[df[column].str.contains('brown'), column] = 'brown'
    df.loc[df[column].str.contains('steel'), column] = 'maximum steel metallic'
    df.loc[df[column].str.contains('recon green'), column] = 'green'
    df.loc[df[column].str.contains('red line'), column] = 'redline pearl'
    df.loc[df[column].str.contains('rhino'), column] = 'rhino'
    df.loc[df[column].str.contains('ruby red'), column] = 'red'
    df.loc[df[column].str.contains('sangria'), column] = 'sangria metallic'
    df.loc[df[column].str.contains('summit'), column] = 'white'
    df.loc[df[column].str.contains('true blue'), column] = 'true blue pearl'
    df.loc[df[column].str.contains('velvet'), column] = 'velvet red pearl'
    df.loc[df[column].str.contains('walnut'), column] = 'walnut brown metallic'
    df.loc[df[column].str.contains('beig'), column] = 'beige'
    # Cadillac specific
    df.loc[(df[column].str.contains('black')) & (df['VehMake'] == 'cadillac'), column] = 'black metallic'
    df.loc[(df[column].str.contains('blue')) & (df['VehMake'] == 'cadillac'), column] = 'blue metallic'
    df.loc[(df[column].str.contains('bronze')) & (df['VehMake'] == 'cadillac'), column] = 'bronze dune metallic'
    df.loc[(df[column].str.contains('charcoal')) & (df['VehMake'] == 'cadillac'), column] = 'gray'
    df.loc[(df[column].str.contains('crystal white')) & (df['VehMake'] == 'cadillac'), column] = 'crystal white tricoat'
    df.loc[(df[column].str.contains('adriatic')) & (df['VehMake'] == 'cadillac'), column] = 'dark blue'
    df.loc[(df[column].str.contains('granite')) & (df['VehMake'] == 'cadillac'), column] = 'dark granite metallic'
    df.loc[(df[column].str.contains('amethyst')) & (df['VehMake'] == 'cadillac'), column] = 'deep amethyst metallic'
    df.loc[(df[column].str.contains('harbor blue')) & (df['VehMake'] == 'cadillac'), column] = 'harbor blue metallic'
    df.loc[(df[column].str.contains('gy')) & (df['VehMake'] == 'cadillac'), column] = 'gray'
    df.loc[(df[column].str.contains('midnight')) & (df['VehMake'] == 'cadillac'), column] = 'midnight sky metallic'
    df.loc[(df[column].str.contains('pearl')) & (df['VehMake'] == 'cadillac'), column] = 'pearl white'
    df.loc[(df[column].str.contains('radiant')) & (df['VehMake'] == 'cadillac'), column] = 'radiant silver metallic'
    df.loc[(df[column].str.contains('red passion')) & (df['VehMake'] == 'cadillac'), column] = 'red'
    df.loc[(df[column].str.contains('red horizon')) & (df['VehMake'] == 'cadillac'), column] = 'red horizon'
    df.loc[(df[column].str.contains('silver')) & (df['VehMake'] == 'cadillac'), column] = 'silver'
    df.loc[(df[column].str.contains('stellar black')) & (df['VehMake'] == 'cadillac'), column] = 'stellar black metallic'
    df.loc[(df[column].str.contains('white')) & (df['VehMake'] == 'cadillac'), column] = 'white'



    # Reduce the excessive number of exterior colors
    color_conditions = [
        (df[column] == 'black black'),
        (df[column] == 'black crystal'), 
        (df[column] == 'black limited'),
        (df[column] == 'bright sil'),
        (df[column] == 'dark blue'),
        (df[column] == 'dark brown'),
        (df[column] == 'dark gray'),
        (df[column] == 'dark red'),
        (df[column] == 'db black'),
        (df[column] == 'dk. gray'),
        (df[column] == 'green clearcoat'),
        (df[column] == 'maroon'),
        (df[column] == 'mineral gray'),
        (df[column] == 'red line'),
        (df[column] == 'other'),
        (df[column] == 'not specified')
    ]
     
    color_values = [
        'black',
        'black',
        'black',
        'silver',
        'blue',
        'brown',
        'gray',
        'red',
        'black',
        'gray',
        'green', 
        'burgundy',
        'gray',
        'redline pearl',
        'unspecified',
        'unspecified'
    ]

    df[column] = np.select(color_conditions, color_values, df[column])

    return df


def int_color_reduction(data, column):
    '''
    This function will attempt to reduce the number of features when one hot
    encoding the colors for interior.

    Parameters: The data, and the column of interest

    Return: Minimized color scale
    '''
    df = data.copy()
    df[column] = df[column].fillna('unspecified')
    # Cadillac specific
    df.loc[(df[column].str.contains('black leather')) & (df['VehMake'] == 'cadillac'), column] = 'black'
    df.loc[(df[column].str.contains('beige')) & (df['VehMake'] == 'cadillac'), column] = 'sahara beige'
    df.loc[(df[column].str.contains('carbon')) & (df['VehMake'] == 'cadillac'), column] = 'carbon plum'
    df.loc[(df[column].str.contains('cirus')) & (df['VehMake'] == 'cadillac'), column] = 'cirus'
    df.loc[(df[column].str.contains('jet black')) & (df['VehMake'] == 'cadillac'), column] = 'jet black'
    df.loc[(df[column].str.contains('maple')) & (df['VehMake'] == 'cadillac'), column] = 'sugar maple'
    df.loc[(df[column].str.contains('brown')) & (df['VehMake'] == 'cadillac'), column] = 'tan'
    df.loc[(df[column].str.contains('grey')) & (df['VehMake'] == 'cadillac'), column] = 'gray'
    df.loc[(df[column].str.contains('granite')) & (df['VehMake'] == 'cadillac'), column] = 'gray'
    df.loc[(df[column].str.contains('cream')) & (df['VehMake'] == 'cadillac'), column] = 'tan'
    df.loc[(df[column].str.contains('bronze')) & (df['VehMake'] == 'cadillac'), column] = 'tan'
    df.loc[(df[column].str.contains('platinum')) & (df['VehMake'] == 'cadillac'), column] = 'gray'
    df.loc[(df[column].str.contains('other')) & (df['VehMake'] == 'cadillac'), column] = 'unspecified'
    # Jeep specific
    df.loc[(df[column].str.contains('beige')) & (df['VehMake'] == 'jeep'), column] = 'beige'
    df.loc[(df[column].str.contains('ruby red')) & (df['VehMake'] == 'jeep'), column] = 'red'
    df.loc[(df[column].str.contains('red')) & (df['VehMake'] == 'jeep'), column] = 'red'
    df.loc[(df[column].str.contains('brown')) & (df['VehMake'] == 'jeep'), column] = 'brown'
    df.loc[(df[column].str.contains('light frost')) & (df['VehMake'] == 'jeep'), column] = 'beige'
    df.loc[(df[column].str.contains('gray')) & (df['VehMake'] == 'jeep'), column] = 'gray'
    df.loc[(df[column].str.contains('charcoal')) & (df['VehMake'] == 'jeep'), column] = 'gray'
    df.loc[(df[column].str.contains('grey')) & (df['VehMake'] == 'jeep'), column] = 'gray'
    df.loc[(df[column].str.contains('cream')) & (df['VehMake'] == 'jeep'), column] = 'tan'
    df.loc[(df[column].str.contains('ebony')) & (df['VehMake'] == 'jeep'), column] = 'black'
    df.loc[(df[column].str.contains('graphite')) & (df['VehMake'] == 'jeep'), column] = 'gray'
    df.loc[(df[column].str.contains('indigo blue')) & (df['VehMake'] == 'jeep'), column] = 'brown'
    df.loc[(df[column].str.contains('pewter')) & (df['VehMake'] == 'jeep'), column] = 'tan'
    df.loc[(df[column].str.contains('sterling')) & (df['VehMake'] == 'jeep'), column] = 'gray'
    df.loc[(df[column].str.contains('tan')) & (df['VehMake'] == 'jeep'), column] = 'tan'
    df.loc[(df[column].str.contains('black')) & (df['VehMake'] == 'jeep'), column] = 'black'
    df.loc[(df[column].str.contains('not specified')) & (df['VehMake'] == 'jeep'), column] = 'unspecified'
    df.loc[(df[column].str.contains('other')) & (df['VehMake'] == 'jeep'), column] = 'unspecified'

    return df


def drive_reduction(data, column):
    '''
    This function will attempt to reduce the number of features when one hot
    encoding the drive type.

    Parameters: The data, and the column of interest

    Return: Minimized drive scale
    '''
    df = data.copy()
    df[column] = df[column].fillna('unspecified')
     # Cadillac specific
    df.loc[(df[column].str.contains('fwd')) & (df['VehMake'] == 'cadillac'), column] = '2wd'
    df.loc[(df[column].str.contains('front wheel')) & (df['VehMake'] == 'cadillac'), column] = '2wd'
    df.loc[(df[column].str.contains('front-wheel')) & (df['VehMake'] == 'cadillac'), column] = '2wd'
    df.loc[(df[column].str.contains('all wheel')) & (df['VehMake'] == 'cadillac'), column] = 'awd'
    df.loc[(df[column].str.contains('all-wheel')) & (df['VehMake'] == 'cadillac'), column] = 'awd'
    df.loc[(df[column].str.contains('allwheeldrive')) & (df['VehMake'] == 'cadillac'), column] = 'awd'
    # Jeep specific
    df.loc[(df[column].str.contains('awd')) & (df['VehMake'] == 'jeep'), column] = 'awd'
    df.loc[(df[column].str.contains('all wheel')) & (df['VehMake'] == 'jeep'), column] = 'awd'
    df.loc[(df[column].str.contains('all-wheel')) & (df['VehMake'] == 'jeep'), column] = 'awd'
    df.loc[(df[column].str.contains('allwheel')) & (df['VehMake'] == 'jeep'), column] = 'awd'
    df.loc[(df[column].str.contains('4x4')) & (df['VehMake'] == 'jeep'), column] = '4wd'
    df.loc[(df[column].str.contains('4-wheel')) & (df['VehMake'] == 'jeep'), column] = '4wd'
    df.loc[(df[column].str.contains('four wheel')) & (df['VehMake'] == 'jeep'), column] = '4wd'
    df.loc[(df[column].str.contains('4 wheel')) & (df['VehMake'] == 'jeep'), column] = '4wd'

    return df


def engine_reduction(data, column):
    '''
    This function will attempt to reduce the number of features when one hot
    encoding the engine type.

    Parameters: The data, and the column of interest

    Return: Minimized engine scale
    '''
    df = data.copy()
    df[column] = df[column].fillna('unspecified')
     # Cadillac specific
    df.loc[(df[column].str.contains('unspecified')) & (df['VehMake'] == 'cadillac'), column] = '3.6 v6'
    df.loc[(df['VehMake'] == 'cadillac'), column] = '3.6 v6'
    
    # Jeep specific
    df.loc[(df[column].str.contains('3.0')) & (df['VehMake'] == 'jeep'), column] = '3.0 diesel'
    df.loc[(df[column].str.contains('diesel')) & (df['VehMake'] == 'jeep'), column] = '3.0 diesel'
    df.loc[(df[column].str.contains('3.6')) & (df['VehMake'] == 'jeep'), column] = '3.6 v6'
    df.loc[(df[column].str.contains('hemi')) & (df['VehMake'] == 'jeep'), column] = 'hemi'
    df.loc[(df[column].str.contains('supercharged')) & (df['VehMake'] == 'jeep'), column] = 'hemi'
    df.loc[(df[column].str.contains('supercharger')) & (df['VehMake'] == 'jeep'), column] = 'hemi'
    df.loc[(df[column].str.contains('5.7')) & (df['VehMake'] == 'jeep'), column] = '5.7 v8'
    df.loc[(df[column].str.contains('6.2')) & (df['VehMake'] == 'jeep'), column] = 'hemi'
    df.loc[(df[column].str.contains('6.4')) & (df['VehMake'] == 'jeep'), column] = 'hemi'
    df.loc[(df[column].str.contains('8 cylinder')) & (df['VehMake'] == 'jeep'), column] = '5.7 v8'
    df.loc[(df[column].str.contains('8-cylinder')) & (df['VehMake'] == 'jeep'), column] = '5.7 v8'
    df.loc[(df[column].str.contains('v8')) & (df['VehMake'] == 'jeep'), column] = '5.7 v8'
    df.loc[(df[column].str.contains('v-8')) & (df['VehMake'] == 'jeep'), column] = '5.7 v8'
    df.loc[(df[column].str.contains('6 cylinder')) & (df['VehMake'] == 'jeep'), column] = '3.6 v6'
    df.loc[(df[column].str.contains('6-cylinder')) & (df['VehMake'] == 'jeep'), column] = '3.6 v6'
    df.loc[(df[column].str.contains('v6')) & (df['VehMake'] == 'jeep'), column] = '3.6 v6'
    df.loc[(df[column].str.contains('v-6')) & (df['VehMake'] == 'jeep'), column] = '3.6 v6'

    return df


if __name__ == '__main__':

    # Test my class
    df = pd.read_csv('../data/Training_DataSet.csv')
    print(df['VehSellerNotes'].head())

    df = CleanText(df, 'VehSellerNotes').perform_all_tasks()
    

    print(df['VehSellerNotes'].head())