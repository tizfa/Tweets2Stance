import re
import os
from src.tweets2stance_zsc.utils import *
import src.tweets2stance_zsc.conf as INST

if not os.path.exists('./log/'):
    os.makedirs('./log/')
setup_logger('execution_log', './log/preprocessing_execution.log')
logger = logging.getLogger('execution_log')


def preprocessing(tweets, remove_mention_hashtags_emojii=True):
    # for italian elections, remove_mention_hashtags_emojii=True (about mentions: removed those at the beginning of a tweet,
    # since they just indicate a reply)
    tweets['tweet'].replace(to_replace="^RT (@\w+ ?)+: ", value=r"", regex=True, inplace=True)
    tweets['tweet'] = tweets.apply(lambda row: row['tweet'].replace('\u2066', '').replace('\u2069', ''), axis=1)
    if remove_mention_hashtags_emojii:
        # remove leading mentions (they mostly refer to a reply)
        tweets['tweet'].replace(to_replace="^(@\w+ )+", value=r"", regex=True, inplace=True)
        # remove hashtags
        tweets['tweet'].replace(to_replace='#\w+', value=r"", regex=True, inplace=True)

    filtered = [
        re.sub(r'https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE).
            replace('++', '').
            replace('\n', ' ').
            replace('&gt', '').
            replace('&lt', '').lower() for tweet in list(tweets['tweet'])]

    def handle_white_spaces(tweets):
        tweets = [re.sub('\s{2,}', ' ', tweet, flags=re.MULTILINE) for tweet in tweets]
        tweets = [re.sub('^\s{1,}', '', tweet, flags=re.MULTILINE) for tweet in tweets]
        # remove leading and trailing whitespaces
        tweets = [tweet.strip() for tweet in tweets]
        return tweets

    if remove_mention_hashtags_emojii:
        # rimuovo le emoji
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

        filtered = [re.sub(emoj, '', tweet) for tweet in filtered]

    filtered = handle_white_spaces(filtered)
    tweets['tweet'] = filtered

    # rimuovo eventuali tweet vuoti
    tweets = tweets[tweets['tweet'] != '']
    tweets.reset_index(drop=True, inplace=True)

    # rimuovo tweet con 1,2,3 parole
    tweets = tweets[tweets['tweet'].apply(lambda x: len(x.split(' ')) >= INST.MIN_NUM_WORDS)]
    tweets.reset_index(drop=True, inplace=True)

    return tweets


def preprocess_vaas():
    # to pre-process new vaa elections
    logger.info('preprocessing vaas...')
    for vaa in INST.ELECTIONS_LIST:
        logger.info(f'VAA: {vaa}')
        for d in INST.DATASETS:
            logger.info(f'Dataset: {d}')
            for suffix in INST.SUFFIXES:
                logger.info(f'suffix: {suffix}')
                if suffix == '':
                    remove_mention_hashtags_emojii = True
                else:
                    remove_mention_hashtags_emojii = False

                preprocess_single_election_dataset(vaa, d, suffix, remove_mention_hashtags_emojii)
            logger.info('-----')
        logger.info('======')


def print_num_tweets_vaa_preprocessed():
    logger.info('printing vaas\' num tweets')
    for vaa in INST.ELECTIONS_LIST:
        for d in INST.DATASETS:
            for suffix in INST.SUFFIXES:
                tweets = load_pickle(f'{INST.DATA_PATH_FOLDER}/{vaa}/', f'{vaa}_{d}_preprocessed{suffix}')
                print(f'{vaa}, {d}, {suffix} : ')
                print(tweets.user_screen_name.value_counts())
                print('******')
            print('-----')
        print('======')


def preprocess_single_election_dataset(election, dataset, suffix='', remove_mention_hashtags_emojii=True):
    tweets = load_pickle(f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{dataset}_raw')

    tweets = preprocessing(tweets, remove_mention_hashtags_emojii)

    filename = f'{election}_{dataset}_preprocessed{suffix}'
    data_path_folder = f'{INST.DATA_PATH_FOLDER}/{election}/'
    save_pickle(tweets, data_path_folder + '/', filename)
    logger.info(f'saved into: {data_path_folder}/{filename}')
    logger.info('******')


if __name__ == '__main__':
    try:
        # To process a list of elections (vaas)
        preprocess_vaas()
        print_num_tweets_vaa_preprocessed()

        # To preprocess tweets for a single election, first decide which dataset to used and set the OPT_DATASET variable in src/conf.py
        #preprocess_single_election_dataset('IT19', INST.OPT_DATASET)
        #logger.info('DONE')
    except Exception as e:
        logger.exception(f'ERROR DETECTED:')