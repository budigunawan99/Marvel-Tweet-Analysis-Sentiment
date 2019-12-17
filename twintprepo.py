#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 23:33:02 2019

@author: bnawan
"""

from optimus import Optimus
import nest_asyncio
import pandas as pd
import json
import csv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import TextBlob
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import time
# import matplotlib.pyplot as plt
# import seaborn as sns, numpy as np
from polyglot.downloader import downloader
print(downloader.supported_languages_table("sentiment2", 86))
from polyglot.text import Text
print(downloader.list(show_packages=False))


def stopword_remover(value, args):
    return ' '.join([word for word in value.split() if word not in (args)])

def cleaning_tweet(dataframe, column):
    cleaned = dataframe\
                .cols.replace_regex(column, r'http\S+|www.\S+|pic.\S+', '')\
                .cols.remove_accents(column)\
                .cols.remove_special_chars(column)\
                .cols.lower([column])\
                .cols.trim("*")\
                .cols.replace_regex(column, '“', '')\
                .cols.replace_regex(column, '”', '')\
                .cols.replace_regex(column, '&nbsp', ' ')\
                .cols.replace_regex(column, '\n', ' ')
    return cleaned

def kamus(file, temp, delimiter):
    with open(file, "r") as myCSVfile:
        dataFromFile = csv.reader(myCSVfile, delimiter=delimiter)
        for row in dataFromFile:
            temp[row[0]]=row[1]
    return temp

def normalization(tweet, kamus):
    tweet=tweet.split(" ")
    j = 0
    for string in tweet:
        if string.upper() in kamus:
            tweet[j] = kamus[string.upper()]
                    
        j+=1
    return ' '.join(tweet)

def normalization_id(tweet, kamus):
    tweet=tweet.split(" ")
    j = 0
    for string in tweet:
        if string.lower() in kamus:
            tweet[j] = kamus[string.lower()]
            # print(tweet)
        j+=1
    return ' '.join(tweet)

def sentiment_measure(sentence):
    # temp = TextBlob(sentence).sentiment[0]
    # if temp >= 0.0:
    #     return 0.0 # Neutral
    # # elif temp >= 0.0:
    # #     return 1.0 # Positive
    # else:
    #     return 1.0 # Negative
    temp = Text(sentence)
    try:
        temps = temp.polarity
    except:
        temps = 0
        print("Language is not supported")
    
    if temps == 0:
        return "neutral" # Neutral
    elif temps > 0:
        return "positive" # Positive
    else:
        return "negative" # Negative


def main():
    stopword_id = stopwords.words('indonesian')
    stopword_en = stopwords.words('english')

    kosakata_alay_id = "colloquial-indonesian-lexicon.csv"
    kosakata_alay_en = "slang_en.csv"
    
    hashkamus_id = {}
    hashkamus_en = {}
    
    kamus_alay_id = kamus(kosakata_alay_id, hashkamus_id, ",")
    kamus_alay_en = kamus(kosakata_alay_en, hashkamus_en, "=")

    op = Optimus()
    
    # Solve compatibility issues with notebooks and RunTime errors.
    nest_asyncio.apply()
    filename = "tweets_marvel.json"
    kolom = ["date", "username", "tweet", "hashtags", "likes_count"]

    df = op.load.json(filename)

    df = df.select(kolom)
    print(df)

    clean_tweets = cleaning_tweet(df, "tweet")
    clean_tweets = clean_tweets.cols.apply("tweet", normalization_id, "string", kamus_alay_id, "udf")
    clean_tweets = clean_tweets.cols.apply("tweet", normalization, "string", kamus_alay_en, "udf")
    superclean_tweets = cleaning_tweet(clean_tweets, "tweet")
    superclean_tweets = superclean_tweets.cols.apply("tweet",stopword_remover, "string", stopword_id, "udf")
    superclean_tweets = superclean_tweets.cols.apply("tweet",stopword_remover, "string", stopword_en, "udf")

    print(superclean_tweets.count())
    tweets = superclean_tweets.select("tweet").rdd.flatMap(lambda x: x).collect()
    

    with open("tweet.txt", "w") as tweetfile:
        tweetfile.write(str(tweets))

    sentiment=udf(sentiment_measure, StringType())
    featured_tweet=superclean_tweets.withColumn("sentiment", sentiment(superclean_tweets["tweet"]))
    
    featured_tweet.show()
    featured_tweet_pandas = featured_tweet.toPandas()
    featured_tweet_pandas.to_csv("output/prepo_output.csv", index=False)
    
if __name__== "__main__":
    main()