#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 23:33:02 2019

@author: bnawan
"""

import twint

tweet = twint.Config()
tweet.Search  = "marvel"
tweet.Since = "2018-01-01"
tweet.Until = "2018-12-31"
tweet.Limit = 30000
tweet.Lang = "id"
tweet.Store_json = True
tweet.Output = "tweets_marvel.json"

twint.run.Search(tweet)