# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from math import sqrt
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

# Run the following command in the python console to download stopwords
# import nltk
# nltk.download('stopwords')

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', None)

data_reviews = "data/reviews.csv"
data_listings = "data/listings.csv"
data_calendar = "data/calendar.csv"

df_reviews = pd.read_csv(data_reviews, index_col=0, sep=',')
df_listings = pd.read_csv(data_listings, index_col=0, sep=',')

df_review_scores = df_listings[['review_scores_rating']]

df_reviews_w_comments = pd.merge(df_review_scores, df_reviews, left_on='id', right_on='listing_id', how='inner')

# %%
df_reviews_w_comments.shape
# (416394, 5)

# %%
df_reviews_w_comments.isnull().sum()

# %%
df_reviews_w_comments.dropna(subset=['review_scores_rating', 'comments'], inplace=True)

# %%
df_reviews_w_comments.isnull().sum()

# %%
df_reviews_w_comments.shape
# (416070, 5)

# %% Summary Statistics
avg_rating = np.mean(df_reviews_w_comments["review_scores_rating"])
print("Average Review Scores Rating: {}".format(avg_rating))

max_rating = np.max(df_reviews_w_comments["review_scores_rating"])
print("Maximum Review Scores Rating: {}".format(max_rating))

min_rating = np.min(df_reviews_w_comments["review_scores_rating"])
print("Minimum Review Scores Rating: {}".format(min_rating))

# %% Box Plot
base_color = sns.color_palette()[0]
plt.figure(figsize=(12,20))
sns.boxplot(data=df_reviews_w_comments, y='review_scores_rating', color= base_color)
plt.show()

# %% Rating Distribution Plot
col_name = 'review_scores_rating'
hist_kws={"alpha": 0.3}
plt.figure(figsize=(20,10))
plt.xticks(np.arange(0, 101, step=5))
sns.distplot(df_reviews_w_comments[col_name], hist_kws=hist_kws)
plt.show()

# %%

def createTokenizedArray(sentences):
    # Initialize tokenizer and empty array to store modified sentences.
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizedArray = []

    for i in range(0, len(sentences)):
        # Convert sentence to lower case.
        sentence = sentences.iloc[i]['comments']
        sentence = sentence.lower()

        # Split sentence into array of words with no punctuation.
        words = tokenizer.tokenize(sentence)

        # Append word array to list.
        tokenizedArray.append(words)

    return tokenizedArray  # send modified contents back to calling function.


# -------------------------------------------------------------
# Create array of words with no punctuation or stop words.
# -------------------------------------------------------------
def removeStopWords(tokenList):
    stopWords = set(stopwords.words('english'))
    shorterSentences = []  # Declare empty array of sentences.

    for sentence in tokenList:
        shorterSentence = []  # Declare empty array of words in single sentence.
        for word in sentence:
            if word not in stopWords:

                # Remove leading and trailing spaces.
                word = word.strip()

                # Ignore single character words and digits.
                if (len(word) > 1 and word.isdigit() == False):
                    # Add remaining words to list.
                    shorterSentence.append(word)
        shorterSentences.append(shorterSentence)

    return shorterSentences


# -------------------------------------------------------------
# Removes suffixes and rebuids the sentences.
# -------------------------------------------------------------
def stemWords(sentenceArrays):
    ps = PorterStemmer()
    stemmedSentences = []

    for sentenceArray in sentenceArrays:
        stemmedArray = []  # Declare empty array of words.
        for word in sentenceArray:
            stemmedArray.append(ps.stem(word))  # Add stemmed word.

        # Convert array back to sentence of stemmed words.
        delimeter = ' '
        sentence = delimeter.join(stemmedArray)

        # Append stemmed sentence to list of sentences.
        stemmedSentences.append(sentence)

    return stemmedSentences


def generateWordList(wordDf, scoreStart, scoreEnd, n_gram_size):
    resultDf = wordDf[(wordDf['review_scores_rating'] >= scoreStart) &
                      (wordDf['review_scores_rating'] <= scoreEnd)]

    sentences = [sentence.split() for sentence in resultDf['comments_processed']]
    wordArray = []
    for i in range(0, len(sentences)):
        wordArray += sentences[i]

    counterList = Counter(ngrams(wordArray, n_gram_size)).most_common(80)

    print("\n***{} N-Grams".format(n_gram_size))
    for i in range(0, len(counterList)):
        print("Occurrences: ", str(counterList[i][1]), end=" ")
        delimiter = ' '
        print("  N-Gram: ", delimiter.join(counterList[i][0]))

    return counterList

# Vectorization transforms words to numbers so they can be used in different predictive machine learning algorithms.
# Vectorization is used to create a master number vector that represents unique words for all sentences.
# Then, for each sentence, the number of occurrences of each word are recorded with a copy of the master number vector.

# Text feature extraction

#-------------------------------------------------------------
# Creates a matrix of word vectors.
#-------------------------------------------------------------
def vectorizeList(stemmedList):
    # Create CV with the Ngram range from 1 to 4.
    cv = CountVectorizer(binary=True, ngram_range=(1, 4))

    cv.fit(stemmedList)
    X = cv.transform(stemmedList)
    print("\nNumber vector size: {}".format(X.shape))

    return X

# %%
def modelAndPredict(X, target):

    model = LogisticRegression(solver='lbfgs', multi_class='auto', penalty='l2', max_iter=100)

    # Find the prediction accuracy of logistic regression model using 5-fold cross-validation
    mean_cross_val_score = cross_val_score(model, X, target, cv=5, scoring='accuracy').mean()
    print("The mean accuracy score using 5-fold cross-validation is: {}".format(mean_cross_val_score))

    # Create training set with 70% of data and test set with 30% of data. Target is sentiments.
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, train_size=0.70, test_size=0.30
    )

    # model.fit(X_train, y_train.values.ravel())
    model.fit(X_train, y_train)

    # Use .values.ravel() on y_train to solve DataConversionWarning.

    # Get the mean accuracy on the given test data and labels.
    score = model.score(X_test, y_test)
    print("\n\n*** The mean accuracy score is: {}".format(score))

    # Predict target values.
    y_prediction = model.predict(X_test)

    # Calculate deviation between actual and predicted values.
    rmse = sqrt(metrics.mean_squared_error(y_test, y_prediction))
    print("The root mean square error calculation is: {}".format(rmse))

    return y_test, y_prediction

# %%
# Confusion Matrix
# Draw the confusion matrix.
def showFormattedConfusionMatrix(y_test, y_predicted):
    # Show simple confusion matrix with no formatting.
    cm = metrics.confusion_matrix(y_test.values, y_predicted)
    print("Simple Confusion Matrix")
    print(cm)

    # Show confusion matrix with colored background.

    inds = ['Very Bad', 'Bad', 'Neutral', 'Good', 'Very Good']
    cols = ['Very Bad', 'Bad', 'Neutral', 'Good', 'Very Good']

    try:
        df = pd.DataFrame(cm, index=inds, columns=cols)

        plt.figure(figsize=(7, 6))

        ax = sns.heatmap(df, cmap='Blues', annot=True, fmt='g')

        bottom, top = ax.get_ylim()
        ax.set(title="AirBnb Tokyo Review Sentiment Actual vs Predicted")
        ax.set_ylim(bottom + 0.5, top - 0.5)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.show()

    except:
        print("\nException occurred during plotting. Please try again.")


# %%

# Prepare the data.
tokenizedList = createTokenizedArray(df_reviews_w_comments[['comments']])

#
df_reviews_w_comments['comments_processed'] = tokenizedList
#

#
df_reviews_w_comments['comments_processed'] = removeStopWords(df_reviews_w_comments['comments_processed'])

#
df_reviews_w_comments['comments_processed'] = stemWords(df_reviews_w_comments['comments_processed'])

# %% Find meaningful ngrams for good ratings (e.g. 90 to 100)
#
# # Create two column matrix.
dfSub = df_reviews_w_comments[['review_scores_rating', 'comments_processed']]

# Range of good ratings to be analyzed
SCORE_RANGE_START = 90.0
SCORE_RANGE_END   = 100.0

# Create ngram lists with # of occurrences.
NGRAM_SIZE = 1
unigrams = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, NGRAM_SIZE)

NGRAM_SIZE = 2
bigrams = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, NGRAM_SIZE)

NGRAM_SIZE = 3
trigrams = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, NGRAM_SIZE)

NGRAM_SIZE = 4
quadgrams = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, NGRAM_SIZE)

# %% Find meaningful ngrams for poor ratings (e.g. 0 to 55)

# Range of bad ratings to be analyzed
SCORE_RANGE_START = 0.0
SCORE_RANGE_END   = 55.0

# Create ngram lists with # of occurrences.
NGRAM_SIZE = 1
unigrams = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, NGRAM_SIZE)

NGRAM_SIZE = 2
bigrams = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, NGRAM_SIZE)

NGRAM_SIZE = 3
trigrams = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, NGRAM_SIZE)

NGRAM_SIZE = 4
quadgrams = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, NGRAM_SIZE)


# %% Logistic Regression

# Transforms words to numbers so they can be used in machine learning algorithms.
vectorizedList = vectorizeList(df_reviews_w_comments['comments_processed'])

# Number vector size: (416070, 14699630)
# A total of 14,699,630 unique words exist across 416070 review comments.

# %%

# 0 (bad), 1 (neutral), 2 (good)
# def classify_ratings(row):
#     if row['review_scores_rating'] >= 80.0:
#         val = 2
#     elif row['review_scores_rating'] >= 60.0:
#         val = 1
#     else:
#         val = 0
#     return val

# 0 (very bad), 1 (bad),  2 (neutral), 3 (good), 4 (very good)
def classify_ratings(row):
    if row['review_scores_rating'] >= 90.0:
        val = 4
    elif row['review_scores_rating'] >= 80.0:
        val = 3
    elif row['review_scores_rating'] >= 60.0:
        val = 2
    elif row['review_scores_rating'] >= 40.0:
        val = 1
    else:
        val = 0
    return val


df_reviews_w_comments['sentiment'] = df_reviews_w_comments.apply(classify_ratings, axis=1)

# %%

# Get predictions and scoring data.
# Target is the rating that we want to predict.

# X_test, y_test, y_predicted = modelAndPredict(vectorizedList, df_reviews_w_comments[['sentiment']])
y_test, y_predicted = modelAndPredict(vectorizedList, df_reviews_w_comments[['sentiment']])

# %%
showFormattedConfusionMatrix(y_test, y_predicted)

# %% Precision, Recall, and F1 Score
from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(y_test, y_predicted, average=None))
print("Precision: {}".format(precision_score(y_test, y_predicted, average='micro')))
print()

print(recall_score(y_test, y_predicted, average=None))
print("Recall: {}".format(recall_score(y_test, y_predicted, average='micro')))
print()

print(f1_score(y_test, y_predicted, average=None))
print("F1 Score: {}".format(f1_score(y_test, y_predicted, average='micro')))
