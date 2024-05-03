import pandas as pd
import os
from src.tweets2stance_llm.utils.config import *
from src.tweets2stance_llm.utils.utils_pickle import *


def divide_tweets_by_party_topic():
    opt_dataset = OPTIMAL_SETTINGS_T2S_ZSC['dataset']
    opt_topic_threshold = OPTIMAL_SETTINGS_T2S_ZSC['topic_threshold']

    for election in PARTIES_VAA.keys():
        print(f'election: {election}')
        for model in MODELS[election]:
            print(f'model: {model}')
            print(f'dataset: {opt_dataset}')
            data_path_folder = f'{projectDir}/data/{election}'
            filename = f'{election}_{opt_dataset}_df_zsc_tweets'
            print(f'opened classified tweets: {data_path_folder}/{model.split("/")[-1]}/{filename}')
            results_df = load_pickle(f'{data_path_folder}/{model.split("/")[-1]}/', filename)

            try:
                df_topics_sentences = pd.read_csv(
                    projectDir + f'/data/sentences_topics/translated_topics_and_sentences_{election}.csv').dropna()
            except FileNotFoundError as e:
                # try with pkl
                df_topics_sentences = load_pickle(f'{projectDir}/data/sentences_topics/',
                                                  f'translated_topics_and_sentences_{election}')

            all_topics = list(df_topics_sentences['topic'])
            parties = list(results_df.screen_name.unique())

            for topic in all_topics:
                if 'use of marijuana' in topic:
                    topic = topic.replace('/', ' or ')
                print(f'topic: {topic}')

                tmp_all_parties = results_df[
                    (results_df['topic'] == topic) & (results_df['score_topic'] >= opt_topic_threshold)]
                for party in parties:
                    tmp_party = tmp_all_parties[tmp_all_parties['screen_name'] == party]

                    path_tweets = f'{projectDir}/data_for_T2S/{opt_dataset}/{election}/{party}'
                    if not os.path.exists(path_tweets):
                        os.makedirs(path_tweets)

                    filename = f'{path_tweets}/{topic}.csv'
                    print(f'saved tweets: {filename}')
                    tmp_party.to_csv(filename, index=False)

                    print('----------')
                print('..........')
            print('-.-.-.-.-.-.-.-')
        print('....----.....----')
    print('==============')
    print(f'DONE')


if __name__ == '__main__':
    divide_tweets_by_party_topic()
