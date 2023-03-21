#%%
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import warnings
from matplotlib import pyplot as plt
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import seaborn as sns


nltk.download('stopwords')


warnings.filterwarnings('ignore')
np.random.seed = 42

df = pd.read_csv('PreProcessedData.csv')
#print(df.head())

#print(data.loc[0:3, "title"])

# removing id number from df
df = df.drop(df.columns[0], axis=1)

# dropping duplicated rows (9600/69045)
df = df.drop_duplicates()



# print(df.head())
# print(len(df['text'].unique()))
#print((df[df['text'].duplicated(keep=False)])["text"])

# there are 9 instances where same news is labelled both as true and false
# print(df[df.duplicated(subset=['title', 'text'], keep=False)])

# there aren't any cases where both are NaN
# print(df.loc[df['text'].isna() & df['title'].isna()])

df["Ground Label"]=np.where(df["Ground Label"]=="true",1,0)
df['text']=df['text'].astype('string')
df['title']=df['title'].astype('string')

na_ratio_cols = df.isna().mean(axis=0)
print(na_ratio_cols)

df.info()
df.hist(column='Ground Label')

# filling empty and na spaces
def changeNA(text):
    if not isinstance(text, str) or pd.isnull(text) or len(text) < 3:
        return "brak danych"
    return text

df['title'] = df['title'].apply(lambda x: changeNA(x))
df['text'] = df['text'].apply(lambda x: changeNA(x))
#%%

# contractions
# zamienia skróty na pełne słowa (np: u - you, don't - do not)
# jakies tureckie znaczki byly ktorych funkcja fix nie ogarniala, stad try except
def remove_contractions(text):
    try:
        s = ' '.join([contractions.fix(word) for word in text.split()])
    except:
        s = text
    return s

df['title'] = df['title'].apply(lambda x: remove_contractions(x))
df['text'] = df['text'].apply(lambda x: remove_contractions(x))
#%%


# stop words
# usuwa słowa nie noszące informacji same w sobie, np zdanie "Donald Trump is being under control of police" zamieni na "Donald Trump under control police"
stop_words = set(stopwords.words('english'))
stop_words.add('http')
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

df['title'] = df['title'].apply(lambda x: remove_stopwords(x))
df['text'] = df['text'].apply(lambda x: remove_stopwords(x))


#stemming
# zamienia słowa w ich podstawę bez końcówek
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


df['title'] = df['title'].apply(lambda x: remove_stopwords(x))
# takes some time:
df["text"] = df["text"].apply(lambda x: stem_words(x))

#print(df['title'].head())

#tokenizacja
#Tworzymy nową kolumnę z tablicą słów użytych w tekscie, bo komp i tak nie rozumie zdan tylko se ogarnia gdzie byly jakie slowa uzywane z jakimi innymi slwoami
df['tokenised_text'] = df['text'].apply(lambda x: word_tokenize(x))
df['tokenised_title'] = df['title'].apply(lambda x: word_tokenize(x))

def count_proper_nouns(token_text):
    proper_nouns_counter = 0
    tagged = nltk.pos_tag(token_text)
    for i in range(len(tagged)):
        word, pos = tagged[i]
        if pos == 'NNP':
            if i!=0 and not tagged[i-1][1] in ['.', '!', '?']:
                proper_nouns_counter+=1
    return proper_nouns_counter
#Zliczanie nazw wlasnych - zajmuje duzo czasu
df['words_counter'] = df['text'].apply(lambda x: len(x.split()))
                
df['proper_nouns_counter'] = df['tokenised_text'].apply(lambda x: count_proper_nouns(x))/df['words_counter']
df['coma_counter'] = df['tokenised_text'].apply(lambda arr: len(list(filter(lambda x: x == ',', arr))))/df['words_counter']
df['exclamation_mark_counter'] = df['tokenised_text'].apply(lambda arr: len(list(filter(lambda x: x == '!', arr))))/df['words_counter']
df['question_mark_counter'] = df['tokenised_text'].apply(lambda arr: len(list(filter(lambda x: x == '?', arr))))/df['words_counter']
#%%


# splitting data

X_train, X_test = train_test_split(
    df, test_size=0.3, random_state=42
)

X_train, X_val = train_test_split(
    X_train, test_size=0.3, random_state=42
)

# X_train - our training
# X_val - our validation
# X_test - external validation

X_train.info()
na_ratio_cols = X_train.isna().mean(axis=0)
print(na_ratio_cols)

na_ratio_cols = X_test.isna().mean(axis=0)
print(na_ratio_cols)

na_ratio_cols = X_val.isna().mean(axis=0)
print(na_ratio_cols)

corrMatrix=df.corr()
