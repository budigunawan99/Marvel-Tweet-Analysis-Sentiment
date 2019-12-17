#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:33:02 2019

@author: bnawan
"""

# Machine Learning imports
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import re
from wordcloud import WordCloud

# get a word count per sentence column
def word_count(sentence):
    return len(sentence.split())

def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

def visualize_sentiment(df):
    df.value_counts().plot.bar(color = 'orange', figsize = (6, 4))
    return plt.show()

def visualize_char_length(df):
    df.str.len().plot.hist(color = 'pink', figsize = (6, 4))
    return plt.show()

def visualize_wordcount_dist(x,y,z,label):
    plt.figure(figsize=(12,6))
    plt.xlim(0,45)
    plt.xlabel('word count')
    plt.ylabel('frequency')
    plt.hist([x, y, z], color=['r','b','g'], alpha=0.5, label=label)
    plt.legend(loc='upper right')
    return plt.show()

def visualize_most_frequent_date(df,title):
    # get most common words in training dataset
    all_value = []
    for line in list(df):
        all_value.append(line)

    value_freq = collections.Counter(all_value).most_common(15)
    frequency = pd.DataFrame(value_freq, columns=['date', 'freq'])
    frequency.plot(x='date', y='freq', kind='bar', figsize=(15, 7), color = 'cornflowerblue')
    plt.title(title)
    return plt.show()
       
def visualize_most_frequent_tweet(df,title):
    cv = CountVectorizer(ngram_range=(3, 3))
    words = cv.fit_transform(df)

    sum_words = words.sum(axis=0)

    words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

    frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'salmon')
    plt.title(title)
    return plt.show()

def visualize_most_frequent_hashtag(df,title):
    HT = hashtag_extract(df)
    HT = sum(HT,[])

    a = nltk.FreqDist(HT)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                    'Count': list(a.values())})

    # selecting top 20 most frequent hashtags     
    d = d.nlargest(columns="Count", n = 15) 
    plt.figure(figsize=(16,5))
    ax = sns.barplot(data=d, y= "Hashtag", x = "Count")
    ax.set(xlabel = 'Count')
    plt.title(title)
    return plt.show()

def visualize_wordcloud(df,title,color):
    word = ' '.join([text for text in df])
    wordcloud = WordCloud(background_color = color,width=800, height=500, random_state = 0, max_font_size = 110).generate(word)

    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title, fontsize = 22)
    plt.show()

def main():
    data = pd.read_csv('output/prepo_output.csv',error_bad_lines=False)
    data.columns = ['date','username','tweet','hashtags','likes_count','sentiment']
    data['tweet'] = data['tweet'].fillna("")
    data['word count'] = data['tweet'].apply(word_count)

    print(data["tweet"][data.date == "2018-11-13"])
    print(data.head(10))

    #visualize positive wordcloud
    visualize_wordcloud(data['tweet'][data['sentiment'] == 'positive'], "Positive Wordcloud", "white")
    #visualize negative wordcloud
    visualize_wordcloud(data['tweet'][data['sentiment'] == 'negative'], "Negative Wordcloud", "black")
    #visualize sentiment distribution
    visualize_sentiment(data['sentiment'])
    #visualize length character in tweet
    visualize_char_length(data['tweet'])
    #visualize word count per sentiment
    visualize_wordcount_dist(   data['word count'][data.sentiment == "neutral"],
                                data['word count'][data.sentiment == "positive"],
                                data['word count'][data.sentiment == "negative"],
                                ['neutral','positive','negative'])
    #visualize date of the most tweet occured
    visualize_most_frequent_date(data['date'],"Date of The Most Tweet Occured - Top 15")
    #visualize date of the most positive tweet occured
    visualize_most_frequent_date(data['date'][data.sentiment == "positive"],"Date of The Most Positive Tweet Occured - Top 15")
    #visualize date of the most negative tweet occured
    visualize_most_frequent_date(data['date'][data.sentiment == "negative"],"Date of The Most Negative Tweet Occured - Top 15")
    #visualize the most frequent word in tweet
    visualize_most_frequent_tweet(data["tweet"],"Most Frequently Occuring 3-grams Words - Top 30")
    #visualize the most frequent positive word in tweet
    visualize_most_frequent_tweet(data["tweet"][data.sentiment == "positive"],"Most Frequently Occuring Positive 3-grams Words - Top 30")
    #visualize the most frequent negative word in tweet
    visualize_most_frequent_tweet(data["tweet"][data.sentiment == "negative"],"Most Frequently Occuring Negative 3-grams Words - Top 30")
    #visualize the most frequent word in date of the most tweet occured 
    visualize_most_frequent_tweet(data["tweet"][data.date == "2018-11-13"],"Most Frequent 3-grams Word in Date of The Most Tweet Occured - Top 30")
    #visualize the most frequent positive word in date of the most positive tweet occured 
    visualize_most_frequent_tweet(data["tweet"][data.date == "2018-11-13"][data.sentiment == "positive"],"Most Frequent Positive 3-grams Word in Date of The Most Tweet Occured - Top 30")
    #visualize the most frequent negative word in date of the most negative tweet occured 
    visualize_most_frequent_tweet(data["tweet"][data.date == "2018-11-13"][data.sentiment == "negative"],"Most Frequent Negative 3-grams Word in Date of The Most Tweet Occured - Top 30")
    #visualize the most frequent hashtag
    visualize_most_frequent_hashtag(data["hashtags"],"Most Frequently Occuring Hashtags - Top 15")
    #visualize the most frequent positive hashtag
    visualize_most_frequent_hashtag(data["hashtags"][data.sentiment == "positive"],"Most Frequently Occuring Hashtags in Positive Tweet - Top 15")
    #visualize the most frequent negative hashtag
    visualize_most_frequent_hashtag(data["hashtags"][data.sentiment == "negative"],"Most Frequently Occuring Hashtags in Negative Tweet  - Top 15")
    #visualize the most frequent hashtag in date of the most tweet occured
    visualize_most_frequent_hashtag(data["hashtags"][data.date == "2018-11-13"], "Most Frequent Hashtag in Date of The Most Tweet Occured - Top 15")
    #visualize the most frequent positive hashtag in date of the most positive tweet occured
    visualize_most_frequent_hashtag(data["hashtags"][data.date == "2018-11-13"][data.sentiment == "positive"], "Most Frequent Hashtag in Postive Tweet in Date of The Most Tweet Occured - Top 15")
    #visualize the most frequent negative hashtag in date of the most negative tweet occured
    visualize_most_frequent_hashtag(data["hashtags"][data.date == "2018-11-13"][data.sentiment == "negative"], "Most Frequent Hashtag in Negative Tweet in Date of The Most Tweet Occured - Top 15")

if __name__== "__main__":
    main()

