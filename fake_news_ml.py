# %% 1
import pandas as pd
import numpy as np
import warnings
from langdetect import detect
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

df = df.sample(frac=0.01)


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

# %% 2

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


# usuwanie nie-angielskich
df['is_english'] = df['text'].apply(lambda x: is_english(x))
df = df[df['is_english']]
df.drop('is_english', axis=1)

# %% 3

# podział danych

X = df.drop('Ground Label', axis=1)
y = df['Ground Label']

df, df_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

df, X_val, y_train, y_val = train_test_split(
    df, y_train, test_size=0.3, random_state=42
)

# df - zbiór treningowy
# X_val - zbiór walidacyjny wewnętrzny
# X_test - zbiór walidacyjny zewnętrzny

# %% 4

def nanToOne(x):
    if pd.isna(x):
        return 1
    return x

# wypełniamy puste
df['title'] = df['title'].apply(lambda x: changeNA(x))
df['text'] = df['text'].apply(lambda x: changeNA(x))

df_test['title'] = df_test['title'].apply(lambda x: changeNA(x))
df_test['text'] = df_test['text'].apply(lambda x: changeNA(x))

# contractions
# zamienia skróty na pełne słowa (np: u - you, don't - do not)
def remove_contractions(text):
    try:
        s = ' '.join([contractions.fix(word) for word in text.split()])
    except:
        s = text
    return s


df['title'] = df['title'].apply(lambda x: remove_contractions(x))
df['text'] = df['text'].apply(lambda x: remove_contractions(x))

df_test['title'] = df_test['title'].apply(lambda x: remove_contractions(x))
df_test['text'] = df_test['text'].apply(lambda x: remove_contractions(x))

print('contractions removed')
# %% 5


# stop words
# usuwa słowa nie noszące informacji same w sobie, np zdanie "Donald Trump is being under control of police" zamieni na "Donald Trump under control police"
stop_words = set(stopwords.words('english'))
stop_words.add('http')


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])


df['title'] = df['title'].apply(lambda x: remove_stopwords(x))
df['text'] = df['text'].apply(lambda x: remove_stopwords(x))

df_test['title'] = df_test['title'].apply(lambda x: remove_stopwords(x))
df_test['text'] = df_test['text'].apply(lambda x: remove_stopwords(x))

print('stopwords removed')

# stemming
# zamienia słowa w ich podstawę bez końcówek
stemmer = PorterStemmer()

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


df['title'] = df['title'].apply(lambda x: stem_words(x))
df["text"] = df["text"].apply(lambda x: stem_words(x))

df_test['title'] = df_test['title'].apply(lambda x: stem_words(x))
df_test["text"] = df_test["text"].apply(lambda x: stem_words(x))

print("stemming done")
# print(df['title'].head())

# tokenizacja
# Tworzymy nową kolumnę z tablicą słów użytych w tekscie, do zrobienia kolumn zliczających liczbę nazw wlasnych itd.
df['tokenized_text'] = df['text'].apply(lambda x: word_tokenize(x))
df['tokenized_title'] = df['title'].apply(lambda x: word_tokenize(x))

df_test['tokenized_text'] = df_test['text'].apply(lambda x: word_tokenize(x))
df_test['tokenized_title'] = df_test['title'].apply(lambda x: word_tokenize(x))

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


df_test['words_counter'] = df_test['text'].apply(lambda x: len(x.split()))

df_test['proper_nouns_counter'] = df_test['tokenized_text'].apply(lambda x: count_proper_nouns(x)) / df_test['words_counter']
df_test['coma_counter'] = df_test['tokenized_text'].apply(lambda arr: len(list(filter(lambda x: x == ',', arr)))) / df_test[
    'words_counter']
df_test['exclamation_mark_counter'] = df_test['tokenized_text'].apply(lambda arr: len(list(filter(lambda x: x == '!', arr)))) / \
                                 df_test['words_counter']
df_test['question_mark_counter'] = df_test['tokenized_text'].apply(lambda arr: len(list(filter(lambda x: x == '?', arr)))) / df_test[
    'words_counter']

print("word counting done")

# %% 6

corrMatrix = df.corr()

# wektoryzacja - tekst jako kolumny wystąpien, bo komp i tak nie rozumie zdan tylko se ogarnia gdzie byly jakie slowa uzywane z jakimi innymi slwoami

#vectorizer = TfidfVectorizer(max_df=0.7, min_df=50)
vectorizerTitle = TfidfVectorizer()
vectorizerText = TfidfVectorizer()
vectorizerTitle.fit(df['title'])
vectorizerText.fit(df['text'])
vec_title = pd.DataFrame.sparse.from_spmatrix(vectorizerTitle.transform(df['title']))
vec_text = pd.DataFrame.sparse.from_spmatrix(vectorizerText.transform(df['text']))

vec_title_test = pd.DataFrame.sparse.from_spmatrix(vectorizerTitle.transform(df_test['title']))
vec_text_test = pd.DataFrame.sparse.from_spmatrix(vectorizerText.transform(df_test['text']))

'''
prawie nic nie usuwa
def correlations(df, max_acceptable_corr):
        corr = df.corr()
        correlations_map = {}
        for col in corr.columns: 
            correlations_map[col] = corr.loc[(abs(corr[col]) > max_acceptable_corr).tolist(), col].index.tolist()
        keys_to_remove = []
        for key in correlations_map:
            if len(correlations_map[key]) <= 1:
                keys_to_remove.append(key)
            else:
                correlations_map[key].remove(key)
        for key in keys_to_remove:
            del correlations_map[key]
        return correlations_map

        
def remove_correlated_cols(df, max_acceptable_corr):
    start_cols = df.shape[1]
    c = correlations(df, max_acceptable_corr)
    for key in c:
        if not key in df.columns:
            continue
        cols = c[key]
        for col in cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
    print("Pozostało " + str(df.shape[1]) + " kolumn.")
    print("Usunięto " + str(start_cols - df.shape[1]) + " kolumn.")
    
remove_correlated_cols(vec_title, 0.8) 
'''

# łączenie
df = df.reset_index(drop=True)
df_numeric = df.iloc[:, [2,6,7,8,9,10]]
vectorized = pd.concat([vec_text, vec_title], axis=1)
X_train = pd.concat([df_numeric, vectorized], axis=1)

df_test = df_test.reset_index(drop=True)
df_test_num = df_test.iloc[:, [2,6,7,8,9,10]]
X_test = pd.concat([df_test_num, vec_text_test, vec_title_test], axis=1, join='inner')

# %% 7

# tworzenie modelu
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

X_test.isnull().any()
X_train['occurrences'] = X_train['occurrences'].apply(lambda x: nanToOne(x))
X_test['occurrences'] = X_test['occurrences'].apply(lambda x: nanToOne(x))

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

model1 = MultinomialNB().fit(X_train, y_train)
y_hat = model1.predict(X_test)
print('accuracy: ', accuracy_score(y_test, y_hat))

model2 = LogisticRegression().fit(X_train, y_train)
y_hat2 = model2.predict(X_test)
print('accuracy: ', accuracy_score(y_test, y_hat2))

