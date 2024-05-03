import os
import time
from google_trans_new.google_trans_new import google_translator
from src.tweets2stance_zsc.utils import *
import src.tweets2stance_zsc.conf as INST

if not os.path.exists('./log/'):
    os.makedirs('./log/')
setup_logger('execution_log', './log/translation_execution.log')
logger = logging.getLogger('execution_log')

#translator = google_translator(url_suffix='it', timeout=9)

# given a dataframe and the name of one of its columns, translates the elements of that column into English
# index_begin_from is useful if at some point the translation stops due to errors.

def translate_column_from_df(df, column_name, election, dataset, suffix, index_begin_from=0):
    i = index_begin_from
    too_many_req = False
    try:
        logger.info('Translation has begun...')
        if index_begin_from == 0:
            logger.info('initialize the \'translation\' column...')
            arr = list()
            for i in range(0, len(df)):
                arr.append('')
            df['translation'] = arr
            del arr

        tot = len(list(df[column_name]))
        # traduco i tweet
        for tweet, index, lang in zip(list(df[column_name][index_begin_from:]), df.index[index_begin_from:], df['lang'][index_begin_from:]):
            go_on = False
            retry = 10
            translated = 'NOT TRANSLATED'
            if lang != 'en':
                translator = google_translator(url_suffix=lang, timeout=9)
                translated = 'NOT TRANSLATED'
                while not go_on:
                    try:
                        translated = translator.translate(tweet, lang_tgt='en')
                        go_on = True
                    except Exception as e:
                        print(f'Exception in translating, but trying again: {str(e)}')
                        print(f'{i}) tweet: {tweet}')
                        if retry == 0:
                            if '429' in str(e):
                                too_many_req = True
                            return translator, tweet, too_many_req
                        retry -= 1
                        time.sleep(2)

                df.at[index, 'translation'] = translated
            else:
                df.at[index, 'translation'] = tweet
            i += 1
            if i % 150 == 0:
                logger.info(f'done {i}/{tot} tweets.')
                save_pickle(df, f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{dataset}preprocessed_translated{suffix}')

        # aggiorno il dataframe
        logger.info('saving last pickle...')
        save_pickle(df, f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{dataset}preprocessed_translated{suffix}')
        logger.info('End.')
        return None, '', too_many_req
    except Exception as e:
        logger.info(f'Exception: {str(e)}')
        logger.info(f'last translated tweet index: {i}. In any case, check if this is true.')
        raise e


def translate_elections():
    logger.info('Translating tweets from election tweets...')
    for election in INST.ELECTIONS_LIST:
        logger.info(f'VAA: {election}')
        for d in INST.DATASETS:
            logger.info(f'Dataset: {d}')
            for suffix in INST.SUFFIXES:
                logger.info(f'suffix: {suffix}')
                # carico i tweet pre-processati
                tweets = load_pickle(f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{d}_preprocessed{suffix}')

                translate_column_from_df(tweets, 'tweet', election, d, suffix, index_begin_from=0)


def restart_election_translation(election, dataset, index_begin_from=0):
    # if there has been an error and the translation was interrupted, load the dataframe of partially translated tweets
    # and set 'index_begin_from = <index of the tweet from which to restart the translation>' appropriately

    # index_begin_from = <index>
    # tweets = load_pickle(f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{d}preprocessed_translated')
    tweets = load_pickle(f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{dataset}preprocessed_translated')
    translate_column_from_df(tweets, 'tweet', election, dataset, index_begin_from)


def rename_columns_translation():
    logger.info('renaming columns "{\'tweet\': \'original_tweet\', \'translation\': \'tweet\'}" ')
    for election in INST.ELECTIONS_LIST:
        logger.info(f'VAA: {election}')
        for d in INST.DATASETS:
            logger.info(f'Dataset: {d}')
            for suffix in INST.SUFFIXES:
                logger.info(f'suffix: {suffix}')
                tweets = load_pickle(f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{d}preprocessed_translated{suffix}')
                tweets.rename(columns={'tweet': 'original_tweet', 'translation': 'tweet'}, inplace=True)
                save_pickle(tweets, f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{d}_preprocessed_translated{suffix}')


def translate_single_election(election, dataset, suffix=''):
    tweets = load_pickle(f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{dataset}_preprocessed{suffix}')

    translate_column_from_df(tweets, 'tweet', election, dataset, suffix, index_begin_from=0)

    tweets = load_pickle(f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{dataset}preprocessed_translated{suffix}')
    tweets.rename(columns={'tweet': 'original_tweet', 'translation': 'tweet'}, inplace=True)
    save_pickle(tweets, f'{INST.DATA_PATH_FOLDER}/{election}/', f'{election}_{dataset}_preprocessed_translated{suffix}')


if __name__ == '__main__':
    # The google_trans_new package allows you to specify the input language through the 'url_suffix' field (by passing
    # the value of the 'lang' field of the tweet).
    try:

        # uncomment the following lines (except for 'restart_election_translation') to translate multiple VAAs
        translate_elections()
        logger.info('Translation DONE')
        #restart_election_translation(election, dataset, index_begin_from=0)
        rename_columns_translation()

        # To translate a single election
        #translate_single_election('IT19', INST.OPT_DATASET)

        logger.info('DONE')
    except Exception as e:
        logger.exception('ERROR DETECTED: ')