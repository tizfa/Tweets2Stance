import pandas as pd
import numpy as np
import sys
import os
import re

from src.config import *
from src.utils_pickle import *
      
def load_sentences(election):
    election_ = election
    if election in ['IT19', 'GB19']:
    #election_ = f'{election}19'
        df_topics_sentences = pd.read_csv(projectDir + f'/translated_topic_and_sentences/translated_topics_and_sentences_{election_}.csv').dropna()
    else:
        df_topics_sentences = load_pickle(projectDir + '/translated_topic_and_sentences/', f'translated_topics_and_sentences_{election_}').dropna()
    return df_topics_sentences

def load_raw_vaa(election, dataset):
    if election == 'IT19':
        tweets = load_pickle(data_folder + f'/{election}/', f'{election}_{dataset}')
    else:
        tweets = load_pickle(data_folder + f'/{election}/', f'{election}_{dataset}_raw')
    data_path_folder = projectDir + f'/data/{election}'
    #print(data_path_folder)
    #print(len(tweets))
    return tweets

def get_golden_labels(election, party=None):
    election_ = election
    golden_labels = load_pickle(f'{projectDir}/dict_golden_labels/',f'dict_golden_labels_VAA_{election_}')
    if party:
        golden_labels = golden_labels[party]
    return golden_labels

def clean(tweets, logger, remove_emoji=False):
    tweets['tweet'].replace(to_replace="^RT (@\w+ ?)+: ", value=r"", regex=True, inplace=True)
    tweets['tweet'] = tweets.apply(lambda row: row['tweet'].replace('\u2066', '').replace('\u2069', ''), axis=1)
    # e.g. détour agréable pour un brunch à sainte-luce-sur-mer avec notre chef ⁦@yfblanchet⁩, le chef du ⁦@partiquebecois⁩ ⁦@pascalberube⁩ et nos candidats ⁦@krimichaud et ⁦@blanchettemax⁩ ! ⁦

    filtered = [
              re.sub(r'https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE).
              replace('\n', ' ').
              replace('\r', ' ').
              replace('&amp;', '&').
              replace('&gt;', '>').
              replace('&gt', '>').
              replace('&lt;', '<').
              replace('&lt', '<').lower() for tweet in list(tweets['tweet'])]
    
    def handle_white_spaces(tweets):
        tweets = [re.sub('\s{2,}', ' ', tweet, flags=re.MULTILINE) for tweet in tweets]
        tweets = [re.sub('^\s{1,}', '', tweet, flags=re.MULTILINE) for tweet in tweets]
        tweets = [tweet.strip() for tweet in tweets]
        return tweets

    if remove_emoji:
        if logger:
            logger.info(f'In cleaning, removing emoji')
        else:
            print(f'In cleaning, removing emoji')
            
        emoj = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002500-\U00002BEF"  # chinese char
                u"\U00002702-\U000027B0"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  # dingbats
                u"\u3030"
                              "]+", re.UNICODE)
    
        filtered = [re.sub(emoj, '', tweet) for tweet in filtered ]
    filtered = handle_white_spaces(filtered)
    tweets['tweet'] = filtered

    #print('empty tweets:')
    #print(tweets[tweets['tweet'] == ''])
    #emtpy_tweets = tweets[tweets['tweet'] == '']
    #print(f'#emtpy_tweets: {emtpy_tweets.str_referenced_tweets.value_counts()}')
    #del empty_tweets
    # rimuovo eventuali tweet vuoti
    tweets = tweets[tweets['tweet'] != '']
    tweets.reset_index(drop=True, inplace=True)
    
    return tweets
