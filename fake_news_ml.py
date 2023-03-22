# %%
import pandas as pd
import numpy as np
import warnings
from langdetect import detect
from matplotlib import pyplot as plt
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from numpy import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sns

# nltk.download('stopwords')

warnings.filterwarnings('ignore')
np.random.seed = 42

df = pd.read_csv('PreProcessedData.csv')


# usuwamy kolumnę z id
df = df.drop(df.columns[0], axis=1)


df["Ground Label"] = np.where(df["Ground Label"] == "true", 1, 0)
df['text'] = df['text'].astype('string')
df['title'] = df['title'].astype('string')

na_ratio_cols = df.isna().mean(axis=0)
print(na_ratio_cols)

df.info()
df.hist(column='Ground Label')

# %%
# podział danych

X = df.drop('Ground Label', axis=1)
y = df['Ground Label']

df, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

df, X_val, y_train, y_val = train_test_split(
    df, y_train, test_size=0.3, random_state=42
)

# X_train - zbiór treningowy
# X_val - zbiór walidacyjny wewnętrzny
# X_test - zbiór walidacyjny zewnętrzny

# %%
# wykrywanie języka - bo wszystkie nie-angielskie to fake (767/59445) - wszystkie do usunięcia
def is_english(text):
    try:
        lang = detect(text)
    except:
        return False
    return lang == 'en'


def changeNA(text):
    if not isinstance(text, str) or pd.isnull(text) or len(text) < 3:
        return "no data"
    return text


# wypełniamy puste
df['title'] = df['title'].apply(lambda x: changeNA(x))
df['text'] = df['text'].apply(lambda x: changeNA(x))

# usuwanie nie-angielskich
df['is_english'] = df['text'].apply(lambda x: is_english(x))
df = df[df['is_english']]


# %%

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

print('contractions removed')
# %%


# stop words
# usuwa słowa nie noszące informacji same w sobie, np zdanie "Donald Trump is being under control of police" zamieni na "Donald Trump under control police"
stop_words = set(stopwords.words('english'))
stop_words.add('http')


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])


df['title'] = df['title'].apply(lambda x: remove_stopwords(x))
df['text'] = df['text'].apply(lambda x: remove_stopwords(x))

print('stopwords removed')

# stemming
# zamienia słowa w ich podstawę bez końcówek
stemmer = PorterStemmer()

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


df['title'] = df['title'].apply(lambda x: stem_words(x))
# takes some time:
df["text"] = df["text"].apply(lambda x: stem_words(x))

print("stemming done")
# print(df['title'].head())

# tokenizacja
# Tworzymy nową kolumnę z tablicą słów użytych w tekscie, bo komp i tak nie rozumie zdan tylko se ogarnia gdzie byly jakie slowa uzywane z jakimi innymi slwoami
df['tokenized_text'] = df['text'].apply(lambda x: word_tokenize(x))
df['tokenized_title'] = df['title'].apply(lambda x: word_tokenize(x))

print("tokenization done")

def count_proper_nouns(token_text):
    proper_nouns_counter = 0
    tagged = nltk.pos_tag(token_text)
    for i in range(len(tagged)):
        word, pos = tagged[i]
        if pos == 'NNP':
            if i != 0 and not tagged[i - 1][1] in ['.', '!', '?']:
                proper_nouns_counter += 1
    return proper_nouns_counter


# Zliczanie nazw wlasnych - zajmuje duzo czasu
df['words_counter'] = df['text'].apply(lambda x: len(x.split()))

df['proper_nouns_counter'] = df['tokenized_text'].apply(lambda x: count_proper_nouns(x)) / df['words_counter']
df['coma_counter'] = df['tokenized_text'].apply(lambda arr: len(list(filter(lambda x: x == ',', arr)))) / df[
    'words_counter']
df['exclamation_mark_counter'] = df['tokenized_text'].apply(lambda arr: len(list(filter(lambda x: x == '!', arr)))) / \
                                 df['words_counter']
df['question_mark_counter'] = df['tokenized_text'].apply(lambda arr: len(list(filter(lambda x: x == '?', arr)))) / df[
    'words_counter']

print("word counting done")

# %%




X_train.info()
na_ratio_cols = X_train.isna().mean(axis=0)
print(na_ratio_cols)

na_ratio_cols = X_test.isna().mean(axis=0)
print(na_ratio_cols)

na_ratio_cols = X_val.isna().mean(axis=0)
print(na_ratio_cols)

corrMatrix = df.corr()


# wektoryzacja - rozbicie tekstu na kolumny zliczające ilość wystąpień słów

# X_stats_train = X_train.drop(['title', 'text', 'tokenized_text', 'tokenized_title'], axis=1)
# X_stats_val = X_val.drop(['title', 'text', 'tokenized_text', 'tokenized_title'], axis=1)
#
# vectorizer_title = TfidfVectorizer()
# Xv_title_train = vectorizer_title.fit_transform(X_train['title'])
# Xv_title_val = vectorizer_title.transform(X_val['title'])
#
# vectorizer_text = TfidfVectorizer()
# Xv_text_train = vectorizer_text.fit_transform(X_train['text'])
# Xv_text_val = vectorizer_text.transform(X_val['text'])
#
# coś sie tu wykrzacza
# X_train= hstack([X_stats_train, Xv_title_train, Xv_text_train])
# X_test= hstack([X_stats_val, Xv_title_val, Xv_text_val])
#
# print(list(X_train.columns))
