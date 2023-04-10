# %%
import time

import pandas as pd
import numpy as np
import warnings
from langdetect import detect
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#from numpy import hstack
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
#np.random.seed = 42

df = pd.read_csv('PreProcessedData.csv')

df = df.sample(frac = 0.01, random_state=1)

# usuwamy kolumnę z id
df = df.drop(df.columns[0], axis=1)

# zmiana typów danych
df["Ground Label"] = np.where(df["Ground Label"] == "true", 1, 0)
df['text'] = df['text'].astype('string')
df['title'] = df['title'].astype('string')

# informacje o ramce przed przetworzeniem
na_ratio_cols = df.isna().mean(axis=0)
print(na_ratio_cols)

df.info()
df.hist(column='Ground Label')


# zliczanie ilości wystąpień, a dopiero potem usunięcie duplikatów - w ten sposob nie tracimy informacji o ilośi wystąpień

df['occurrences'] = df.groupby(['title', 'text'])['title'].transform('count')
df = df.drop_duplicates()

# wykrywanie języka - bo wszystkie nie-angielskie to fake (767/59445) - wszystkie do usunięcia
def is_english(text):
    try:
        lang = detect(text)
    except:
        return False
    return lang == 'en'


def changeNA(text):
    if not isinstance(text, str) or pd.isnull(text) or len(text) < 3:
        return ""
    return text


# wypełniamy puste
df['title'] = df['title'].apply(lambda x: changeNA(x))
df['text'] = df['text'].apply(lambda x: changeNA(x))

# usuwanie nie-angielskich
df['is_english'] = df['text'].apply(lambda x: is_english(x))
df = df[df['is_english']]
df.drop('is_english', axis=1)

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

# X_val['title'] = X_val['title'].apply(lambda x: remove_stopwords(x))
# X_val['text'] = X_val['text'].apply(lambda x: remove_stopwords(x))

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
# Tworzymy nową kolumnę z tablicą słów użytych w tekscie, do zrobienia kolumn zliczających liczbę nazw wlasnych itd.
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

def word_counting(df):
    df['words_counter'] = df['text'].apply(lambda x: len(x.split()))

    df['proper_nouns_counter'] = df['tokenized_text'].apply(lambda x: count_proper_nouns(x)) / df['words_counter']
    df['coma_counter'] = df['tokenized_text'].apply(lambda arr: len(list(filter(lambda x: x == ',', arr)))) / df[
        'words_counter']
    df['exclamation_mark_counter'] = df['tokenized_text'].apply(
        lambda arr: len(list(filter(lambda x: x == '!', arr)))) / \
                                     df['words_counter']
    df['question_mark_counter'] = df['tokenized_text'].apply(lambda arr: len(list(filter(lambda x: x == '?', arr)))) / \
                                  df['words_counter']


word_counting(df)
# word_counting(X_val)

print("word counting done")

# drop kolumn - list, ich nie użyjemy w tworzeniu modelu
X = df.drop(['Ground Label', 'tokenized_text', 'tokenized_title'], axis=1)
y = df["Ground Label"]

#Dzielimy na zbiór treningowy i zbiór testowy (pomijam na razie walidacyjny, ogarnie sie)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# ponowna zmiana typów, nie naprawia żadnego z błędów
# X_train['text'] = X_train['text'].astype('string')
# X_train['title'] = X_train['title'].astype('string')
#
# X_train['is_english'] = X_train['is_english'].astype('float64')
# X_train['words_counter'] = X_train['words_counter'].astype('float64')
#
# X_test['is_english'] = X_test['is_english'].astype('float64')
# X_test['words_counter'] = X_test['words_counter'].astype('float64')
#
# X_test['text'] = X_test['text'].astype('string')
# X_test['title'] = X_test['title'].astype('string')


print(type(X_train))
print(type(y_train))

# text_cols = ["title", "text"]
# num_cols = ['occurrences', 'is_english', 'words_counter', 'proper_nouns_counter',
#        'coma_counter', 'exclamation_mark_counter', 'question_mark_counter']

print(X_train.columns)
print(X_test.columns)
print(X_train.dtypes)

# na necie robili takie proste transformery do samego pobrania danych:
def get_numeric(x):
    # zakładamy, że wszystko po kolumnach title i text (indeksy 0 i 1) to dane numeryczne
    return [record[2:] for record in x]

def get_text(x):
    return [record[:2] for record in x]

transformer_get_numeric = FunctionTransformer(get_numeric)
transformer_get_text = FunctionTransformer(get_text)


# pipeline, który robi całe przetwarzanie i potem jeszcze tworzenie modelu
# TODO BŁĄD: złe wymiary czegoś. nie mam siły do tej kurwy.
# TODO (oprócz naprawy): pewnie sie da wrzucić te apply'e rozsiane po całym pliku tutaj


pipe = Pipeline([
    ('features', FeatureUnion([
            ('numeric_features', Pipeline([
                ('selector', transformer_get_numeric)
            ])),
             ('text_features', Pipeline([
                ('selector', transformer_get_text),
                #('cat_trans', OneHotEncoder()),
                ('vec', TfidfVectorizer(analyzer='word'))
            ]))
         ])),
    ('classifier', None)
])


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# test różnych klasyfikatorów
classifiers = [
    DummyClassifier(strategy='stratified'),
    ExtraTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    LogisticRegression(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier()
]

models_df = pd.DataFrame()
for model in classifiers:
    # odpowiednio zmieniamy jego paramety - dobieramy metodę klasyfikacji
    pipe_params = {
        'classifier': model
    }
    pipe.set_params(**pipe_params)

    # trenujemy tak przygotowany model (cały pipeline) mierząc ile to trwa
    start_time = time.time()

    pipe.fit(X_train, y_train)
    end_time = time.time()

    # sprawdzamy jak wyszło
    score = pipe.score(X_test, y_test)

    # zbieramy w dict parametry dla Pipeline i wyniki
    param_dict = {
        'model': model.__class__.__name__,
        'score': score,
        'time_elapsed': end_time - start_time
    }

    models_df = models_df.append(pd.DataFrame(param_dict, index=[0]))

models_df.reset_index(drop=True, inplace=True)
models_df.sort_values('score', ascending=False)


# jeden klasyfikator - ten sam błąd
# pipeline = Pipeline([
#     ('features', FeatureUnion([
#             ('numeric_features', Pipeline([
#                 ('selector', transformer_get_numeric)
#             ])),
#              ('text_features', Pipeline([
#                 ('selector', transformer_get_text),
#                 ('cat_trans', OneHotEncoder()),
#                 ('vec', TfidfVectorizer(analyzer='word'))
#             ]))
#          ])),
#     ('clf', RandomForestClassifier())
# ])
#
# pipeline.fit(X_train, y_train)
