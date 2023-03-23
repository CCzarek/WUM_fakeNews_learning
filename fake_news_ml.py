# %%
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

# df - zbiór treningowy
# X_val - zbiór walidacyjny wewnętrzny
# X_test - zbiór walidacyjny zewnętrzny


# zliczanie ilości wystąpień, a dopiero potem usunięcie duplikatów - w ten sposob nie tracimy informacji o ilośi wystąpień

df['occurrences'] = df.groupby(['title', 'text'])['title'].transform('count')
df = df.drop_duplicates()

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
        return ""
    return text


# wypełniamy puste
df['title'] = df['title'].apply(lambda x: changeNA(x))
df['text'] = df['text'].apply(lambda x: changeNA(x))

# usuwanie nie-angielskich
df['is_english'] = df['text'].apply(lambda x: is_english(x))
df = df[df['is_english']]
df.drop('is_english', axis=1)

# %%

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

corrMatrix = df.corr()

# wektoryzacja - tekst jako kolumny wystąpien, bo komp i tak nie rozumie zdan tylko se ogarnia gdzie byly jakie slowa uzywane z jakimi innymi slwoami

vectorizer = TfidfVectorizer()
vec_title = pd.DataFrame.sparse.from_spmatrix(vectorizer.fit_transform(df['title']))
vec_text = pd.DataFrame.sparse.from_spmatrix(vectorizer.fit_transform(df['text']))

vec = pd.DataFrame(hstack([vec_title, vec_text]))

#print(vec.shape)

additional_cols = df.drop(['title', 'text', 'tokenized_text', 'tokenized_title'], axis=1)
vec = vec.join(additional_cols)

print("vectorization done")
#vec.to_csv('processed_fakenews.csv', index=False)

