from transformers import pipeline
import pandas as pd
import argparse
import sys
import os
import numpy as np
from utils import *
import json
import src.tweets2stance_zsc.conf as INST

if not os.path.exists('./log/'):
    os.makedirs('./log/')
setup_logger('execution_log', './log/classification_execution.log')
logger = logging.getLogger('execution_log')


def get_classifier(model_name):
    logger.info(f'Getting classifier: {model_name}')
    # device = 0 -> working with GPU
    # https://huggingface.co/transformers/master/main_classes/pipelines.html#transformers.ZeroShotClassificationPipeline
    classifier = pipeline("zero-shot-classification", model=model_name, tokenizer=model_name, device=0)
    logger.info(f'Got classifier: {model_name}')
    return classifier


def classify_tweets(election, dataset, suffix, model_name, df_tweets, topics, sentences, classifier, df_partial=pd.DataFrame()):
    logger.info('Classifying tweets...')
    append_to_partial = False
        
    if type(df_partial) != 'NoneType' and not df_partial.empty:
        append_to_partial = True

    filename = f'{election}_{dataset}_zsc_tweets{suffix}'
    path = f'{INST.DATA_PATH_FOLDER}/{election}/{model_name.split("/")[-1]}'
    if not os.path.exists(path):
        os.makedirs(path)
    logger.info(f'tweets will be saved into {path}/{filename}')

    # append results in a list so that you can create a dataframe at the end
    results = []
    count = len(df_partial) if type(df_partial) != 'NoneType' and not df_partial.empty else 0
    if 'date_obj' not in list(df_tweets) and 'created_at' in list(df_tweets) and not isinstance(df_tweets.iloc[0]['created_at'], str):
        logger.info(f'Taking the created_at field: {df_tweets.iloc[0]["created_at"]}')
        dates = list(df_tweets['created_at'])
    else:
        logger.info(f'Taking the date_obj field: {df_tweets.iloc[0]["date_obj"]}')
        dates = list(df_tweets['date_obj'])
    for tweet, id, user_id, screen_name, date_obj in zip(list(df_tweets['tweet']), list(df_tweets['tweet_id']),
                                                         list(df_tweets['user_id']),
                                                         list(df_tweets['user_screen_name']),
                                                         dates):
        # multi_label: Whether or not multiple candidate labels can be true. If False, the scores are normalized such that the sum of the label likelihoods for each sequence is 1. 
        # If True, the labels are considered independent and probabilities are normalized for each candidate by doing a softmax of the entailment score vs. the contradiction score.

        # hypothesis_template="This text is about {}." -> '{}' verrà sostituito automaticamente con le candidate labels. Non devi fare niente :D
        if 'DeBERTa' in model_name:
            template = '{}'
        elif 'bart' in model_name:
            template = "This text is about {}."
        else:
            # covid-twitter-bert
            template = "This example is {}."

        out = classifier(tweet, topics, hypothesis_template=template, multi_label=True)
        out['tweet_id'] = id
        out['user_id'] = user_id
        out['screen_name'] = screen_name
        out['created_at'] = date_obj
        out['labels'] = out['labels']  # np.array(out['labels'])
        out['scores'] = out['scores']  # np.array(out['scores'])

        out_s = classifier(tweet, sentences, hypothesis_template="{}", multi_label=True)
        out['labels'].extend(out_s['labels'])
        out['scores'].extend(out_s['scores'])

        out['labels'] = np.array(out['labels'])
        out['scores'] = np.array(out['scores'])
        results.append(out)
        '''
        example of 'out':
        { 'labels': ['conspiracy', 'not conspiracy'],
          'scores': [0.992976188659668, 0.0070238420739769936],
          'sequence': 'Covid Crusade: Franklin Graham Preaches Vaccine Gospel for Deep State https://t.co/uwO0uFsa7I'}
        '''

        count += 1
        if count % 1000 == 0:
            logger.info('count: {} | saving tweets...'.format(count))
            df = pd.DataFrame(results)
            if append_to_partial:
                df = df_partial.append(df)
                df.reset_index(drop=True, inplace=True)
            save_pickle(df, path + '/', filename)
            logger.info(df)
            logger.info(f'tweets saved: {path}/{filename}')

    # create and save df
    logger.info(f'saving tweets for the last time... {path}/{filename}')
    df = pd.DataFrame(results)
    if append_to_partial:
        df = df_partial.append(df)
        df.reset_index(drop=True, inplace=True)

    save_pickle(df, path + '/', filename)
    logger.info(df)
    logger.info('tweets saved.')
    return df


def pretty_save_df(election, dataset, model_name, suffix):
    # get classified tweets
    filename = f'{election}_{dataset}_zsc_tweets{suffix}'
    data_path_folder = f'{INST.DATA_PATH_FOLDER}/{election}/{model_name.split("/")[-1]}'
    logger.info(f'Pretty save df: opening {data_path_folder}/{filename}')
    df = load_pickle(f'{data_path_folder}/', filename)
    df.reset_index(drop=True, inplace=True)
    if 'index' in list(df):
        df.drop(columns='index', inplace=True)

    try:
        df_topic = pd.read_csv(f'{INST.TOPICS_SENTENCES_FOLDER}/translated_topics_and_sentences_{election}.csv').dropna()
    except FileNotFoundError as e:
        # try with pkl
        df_topic = load_pickle(f'{INST.TOPICS_SENTENCES_FOLDER}/', f'translated_topics_and_sentences_{election}')

    parsed_result = list()

    # itero i tweet
    for sequence, labels, scores, id, user_id, screen_name, date_obj in zip(list(df['sequence']), list(df['labels']),
                                                                            list(df['scores']), list(df['tweet_id']),
                                                                            list(df['user_id']),
                                                                            list(df['screen_name']),
                                                                            list(df['created_at'])):
        sequence = sequence.replace('\n', ' ')

        indices_topics = list()
        indices_sentences = list()
        field_topic = 'topic'
        field_sentence = 'sentence'
        for topic, sentence in zip(list(df_topic[field_topic]), list(df_topic[field_sentence])):
            indices_topics.append(np.where(labels == topic)[0][0])
            indices_sentences.append(np.where(labels == sentence)[0][0])

        list_topics = labels[indices_topics]
        list_sentences = labels[indices_sentences]
        list_topics_scores = scores[indices_topics]
        list_sentences_scores = scores[indices_sentences]

        if len(list_topics) != len(df_topic.index) or len(list_sentences) != len(df_topic.index) or len(
                list_topics_scores) != len(df_topic.index) or len(list_sentences_scores) != len(df_topic.index):
            logger.info('There has been an error in processing tweets...')
            logger.info(f'len list_topics: {len(list_topics)}')
            logger.info(f'len list_sentences: {len(list_sentences)}')
            logger.info(f'len list_topics_scores: {len(list_topics_scores)}')
            logger.info(f'len list_sentences_scores: {len(list_sentences_scores)}')
            logger.info(f'labels: {labels}')
            return

        # inserisco il tweet e il threshold nella/nelle relative liste nel dizionario
        for topic, score_topic, sentence, score_sentence in zip(list_topics, list_topics_scores, list_sentences,
                                                                list_sentences_scores):
            parsed_result.append({
                'created_at': date_obj,
                'tweet_id': str(id),
                'tweet': sequence,
                'user_id': str(user_id),
                'screen_name': screen_name,
                'topic': topic,
                'score_topic': score_topic,
                'sentence': sentence,
                'score_sentence': score_sentence
            })

    topics_df = pd.DataFrame(parsed_result)
    # logger.info('dropping duplicates...')
    # topics_df.drop_duplicates(inplace=True)
    logger.info(f'saving pickle as \'{data_path_folder}/{election}_{dataset}_df_zsc_tweets{suffix}\'')
    filename = f'{election}_{dataset}_df_zsc_tweets{suffix}'
    save_pickle(topics_df, f'{data_path_folder}/', filename)

    return topics_df


def get_conf_params():
    parser = argparse.ArgumentParser(description='Python Wrapper Tweepy')
    parser.add_argument('-i', '--input', type=str, help='json configuration file with path')
    args = parser.parse_args()
    print(args)
    print(os.getcwd())

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            settings = json.load(f)
    except IOError:
        print('settings file ' + args.input + ' not found')
        sys.exit(-1)

    return settings


def load_tweets_to_classify(election, dataset, model_name, suffix, type_, index_begin_from):
    # loading tweets
    data_path_folder = f'{INST.DATA_PATH_FOLDER}/{election}'

    filename = f'{election}_{dataset}_preprocessed_translated{suffix}'
    tweets_to_classify = load_pickle(data_path_folder + '/', filename)
    tweets_to_classify['tweet'] = tweets_to_classify['tweet'].apply(lambda t: t.rstrip())
    df_partial = pd.DataFrame()
    if type_ == 'restart':
        df_partial = load_pickle(f'{data_path_folder}/{model_name.split("/")[-1]}/', f'{election}_D7_zsc_tweets{suffix}')
    # tweets_to_classify['date_obj'] = tweets_to_classify['created_at'].apply(lambda d: datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%fZ"))

    # loading topics
    try:
        df_topics_sentences = pd.read_csv(f'{INST.TOPICS_SENTENCES_FOLDER}/translated_topics_and_sentences_{election}.csv').dropna()
    except FileNotFoundError as e:
        # try with pkl
        df_topics_sentences = load_pickle(f'{INST.TOPICS_SENTENCES_FOLDER}/', f'translated_topics_and_sentences_{election}')

    topics = list(df_topics_sentences['topic'])
    sentences = list(df_topics_sentences['sentence'])

    logger.info(f'topics and sentences: {df_topics_sentences}')
    logger.info(f'df_partial: {df_partial}')
    logger.info(f'last two tweets_to_classify: {tweets_to_classify[index_begin_from-2:]}')
    tweets_to_classify = tweets_to_classify[index_begin_from:]

    return tweets_to_classify, topics, sentences, df_partial


def classify_vaas(dict_vaas, type_='begin'):
    logger.info('Beginning classification ZSC...')

    for election, election_info in dict_vaas.items():
        logger.info(f'ELECTION {election}: {election_info}')
        index_begin_from = election_info['index_begin_from']
        models = INST.MODELS[election] if election_info['models'] == 'all' else election_info['models']
        for model_name in models:
            classifier = get_classifier(model_name)
            suffixes = INST.SUFFIXES if 'covid' in model_name else ['']
            for dataset in INST.DATASETS:
                for suffix in suffixes:
                    logger.info(f'SUFFIX: {suffix}')
                    tweets_to_classify, topics, sentences, df_partial = load_tweets_to_classify(election, dataset, model_name, suffix, type_, index_begin_from)

                    classify_tweets(election, dataset, suffix, model_name, tweets_to_classify, topics, sentences, classifier, df_partial)
                    pretty_save_df(election, dataset, model_name, suffix)

                    logger.info(f'DONE {election}, {model_name}, {dataset}, suffix: {suffix}')

                    logger.info('******')
                logger.info('*-*-*-*-*')
            logger.info('-----')
        logger.info('======')

    logger.info('DONE ALL')


if __name__ == '__main__':
    try:
        # to classify tweets for a single election, use the inputs/in_ZSC_single_election.json file as input to this script
        settings = get_conf_params()
        for type_, dict_vaas in settings.items():
            logger.info(f'{type_}: {dict_vaas}')
            if dict_vaas:
                classify_vaas(dict_vaas, type_)
    except Exception as e:
        logger.exception('ERROR DETECTED: ')



