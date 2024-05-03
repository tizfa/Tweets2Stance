import pandas as pd
import numpy as np
import sys
import os
import re

from src.tweets2stance_llm.utils.config import *
from src.tweets2stance_llm.utils.utils_pickle import *

def add_sentences(tweets_str, election):
  topic_sentences = load_sentences(election)
  if election == 'IT19':
    sentences = topic_sentences['sentence_ita'].values
  else:
    sentences = topic_sentences['sentence'].values

  last_index = len(sentences) - 1

  tweets_str += '\n\n'
  tweets_str += 'statements: ['

  for id, sentence in enumerate(sentences):
    tweets_str += '{' + f'"id":{id}' + f',"statement":"{sentence}"' + '}'
    if id == last_index:
      tweets_str += ']'
    else:
      tweets_str += ','

  return tweets_str

def batch_tweets(tweets):
  # da usare con il prompt che già contiene gli statement da considerare!
  #'~|~'.join(tweets_party.tweet.values)
  batch = list()
  tweets_str = ''
  path_folder = f'{data_folder_LLM}/{election}/{dataset}/{party}'
  if not os.path.exists(path_folder):
    os.makedirs(path_folder)

  b_i = 0
  last_index = len(tweets.tweet.values)-1
  for i, t in enumerate(tweets.tweet.values):
    if len(tweets_str) + len(t) > 10000:
      print(f'done batch {b_i}')
      batch.append(tweets_str)
      with open(f'{path_folder}/tweets_with_separator_batch{b_i}.txt', 'w') as f:
        f.write(tweets_str)
      b_i += 1
      tweets_str = ''
    else:
      tweets_str += f'{t}~|~'

  if tweets_str:
    tweets_str += '\n\nDONE!!!'
    batch.append(tweets_str)
    print(f'last_oneee: {b_i}')
    with open(f'{path_folder}/tweets_with_separator_batch{b_i}.txt', 'w') as f:
        f.write(tweets_str)
  return batch, tweets_str