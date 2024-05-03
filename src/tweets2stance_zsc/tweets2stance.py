import pandas as pd
import sys
import os
from src.tweets2stance_zsc.utils import *
import src.tweets2stance_zsc.conf as INST

if not os.path.exists('./log/'):
    os.makedirs('./log/')
setup_logger('execution_log', './log/tweets2stance.log')
logger = logging.getLogger('execution_log')

def compute_final_label(elem, final_thresholds):
    if len(final_thresholds) == 4:
        answer = 1
        if elem >= final_thresholds[3]:
            answer = 5
        elif elem >= final_thresholds[2]:
            answer = 4
        elif elem >= final_thresholds[1]:
            answer = 3
        elif elem >= final_thresholds[0]:
            answer = 2
    elif len(final_thresholds) == 3:
        if elem >= final_thresholds[2]:
            answer = 4  # to be transformed into 5 subsequently
        elif elem >= final_thresholds[1]:
            answer = 3  # to be transformed into 4 subsequently
        elif elem >= final_thresholds[0]:
            answer = 2
        else:
            answer = 1
    else:
        sys.exit(-1)

    return answer


import re


def compute_sentences_result_per_user(election, model_name, dataset, algorithm, suffix,
                                      threshold_value=None):

    final_path = f'{INST.DATA_PATH_FOLDER}/{election}/{model_name.split("/")[-1]}/{dataset}/{algorithm}'
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    logger.info('===================')
    logger.info(f'MODEL_NAME: {model_name}')
    logger.info(f'ALGORITHM: {algorithm}')
    logger.info(f'FINAL_PATH: {final_path}')
    logger.info(f'SUFFIX: {suffix}')

    data_path_folder = f'{INST.DATA_PATH_FOLDER}/{election}'
    filename = f'{election}_{dataset}_df_zsc_tweets{suffix}'
    logger.info(f'opened classified tweets: {data_path_folder}/{model_name.split("/")[-1]}/{filename}')
    results_df = load_pickle(f'{data_path_folder}/{model_name.split("/")[-1]}/', filename)
    # logger.info(results_df)

    try:
        df_topics_sentences = pd.read_csv(f'{INST.DATA_PATH_FOLDER}/data/sentences_topics/translated_topics_and_sentences_{election}.csv').dropna()
    except FileNotFoundError as e:
        # try with pkl
        df_topics_sentences = load_pickle(f'{INST.DATA_PATH_FOLDER}/data/sentences_topics/',
                                          f'translated_topics_and_sentences_{election}')

    all_topics = list(df_topics_sentences['topic'])
    result_avg = list()

    min_num_tweets = None
    final_thresholds = [0.2, 0.4, 0.6, 0.8]
    thresholds_topics = [0.5, 0.6, 0.7, 0.8, 0.9]
    if 'algorithm_4' in algorithm:
        s = re.search('_4_min_num_tweets_(\d)', algorithm)
        if not s:
            sys.exit(-1)
        min_num_tweets = int(s.group(1))

    for user in results_df['screen_name'].unique():
        logger.info(f'user {user}')
        user_df = results_df[(results_df['screen_name'] == user) & (results_df['topic'].isin(all_topics))]
        topics = list(user_df['topic'].unique())
        logger.info(f'# topics: {len(topics)}')

        for topic in topics:
            logger.info(f'topic: {topic}')
            if threshold_value:
                # if I want to use a specific threshold value
                thresholds_topics = [threshold_value]
            for threshold_topic in thresholds_topics:
                logger.info(f'threshold: {threshold_topic}')
                # retrieve only the tweets from the user 'user' and topic 'topic' that have exceeded the 'threshold_topic'
                actual_df = user_df[(user_df['topic'] == topic) & (user_df['score_topic'] >= threshold_topic)]
                field_sentence = 'sentence'
                field_topic = 'topic'
                # extracting the corresponding sentence
                sentence = list(df_topics_sentences.loc[df_topics_sentences[field_topic] == topic][field_sentence])[0]
                if actual_df.empty:
                    logger.info(
                        f'df empty for user {user}, topic \'{topic}\', threshold_topic: \'{threshold_topic}\' | setting 3 as the answer')
                    result_avg.append([user, sentence, topic, None, 3, threshold_topic, 0])
                    continue

                if 'algorithm_4' in algorithm and min_num_tweets and len(actual_df) < min_num_tweets:
                    logger.info(
                        f'less than min num tweets {min_num_tweets} for user {user}, topic \'{topic}\', threshold_topic: \'{threshold_topic}\' | setting 3 as the answer')
                    result_avg.append([user, sentence, topic, None, 3, threshold_topic, len(actual_df)])
                    continue
                elif 'algorithm_4' in algorithm and not min_num_tweets:
                    sys.exit(-1)

                avg = None

                if 'algorithm_1' in algorithm:
                    # extracting values and weights from dataframe
                    values = list(actual_df['score_sentence'])
                    weights = list(actual_df['score_topic'])

                    num, den = 0, 0

                    for value, weight in zip(values, weights):
                        # updating numerator and denominator
                        num += (value * weight)
                        den += weight
                    # Computing the weighted average and the value of the response (from 1 to 5)
                    avg = num / den
                    answer = compute_final_label(avg, final_thresholds)
                elif 'algorithm_3' in algorithm:
                    actual_df['label_tweet'] = actual_df['score_sentence'].apply(
                        lambda score_s: compute_final_label(score_s, final_thresholds))
                    tmp_df = pd.DataFrame(actual_df['label_tweet'].value_counts())
                    majority_labels = list(tmp_df[tmp_df['label_tweet'] == tmp_df.values.max()].index)
                    # logger.info(f'tmp_df: {tmp_df}')
                    if len(majority_labels) == 1:
                        answer = majority_labels[0]
                    else:
                        #logger.info(f'computing the mean for user {user}, topic \'{topic}\', threshold_topic: \'{threshold_topic}\'')
                        answer = round(sum(actual_df['label_tweet']) / len(actual_df))
                elif 'algorithm_2' in algorithm:
                    actual_df['label_tweet'] = actual_df['score_sentence'].apply(
                        lambda score_s: compute_final_label(score_s, final_thresholds))
                    answer = round(sum(actual_df['label_tweet']) / len(actual_df))
                elif 'algorithm_4' in algorithm:
                    #logger.info(f'in algorithm 4 with min_num_tweets: {min_num_tweets}')
                    actual_df['label_tweet'] = actual_df['score_sentence'].apply(
                        lambda score_s: compute_final_label(score_s, [0.25, 0.5, 0.75]))
                    tmp_df = pd.DataFrame(actual_df['label_tweet'].value_counts())
                    #logger.info(f'tmp_df: {tmp_df}')
                    majority_labels = list(tmp_df[tmp_df['label_tweet'] == tmp_df.values.max()].index)
                    if len(majority_labels) == 1:
                        answer = majority_labels[0]
                    else:
                        #logger.info(f'computing the mean for user {user}, topic \'{topic}\', threshold_topic: \'{threshold_topic}\'')
                        #logger.info(f'tmp_df: {tmp_df}')
                        answer = round(sum(actual_df['label_tweet']) / len(actual_df))
                    #logger.info(f'answer: {answer}')
                    if answer == 3:
                        answer = 4
                    elif answer == 4:
                        answer = 5
                    #logger.info(f'final answer: {answer}')
                else:
                    logger.info(f'algorithm: {algorithm} not implemented')
                    sys.exit(-1)

                result_avg.append([user, sentence, topic, avg, answer, threshold_topic, len(actual_df)])

    df_tot = pd.DataFrame(result_avg, columns=['screen_name', 'sentence', 'topic', 'avg_agreement', 'agreement_level',
                                               'threshold_topic', 'num_tweets'])
    # 'df_final_results_' +  model_name.split("/")[-1] + '_' + algorithm
    logger.info(f'Saved file: {final_path}/df_agreements{suffix}')
    save_pickle(df_tot, final_path + '/', f'df_agreements{suffix}')

    return df_tot


def T2S_all_elections():
    for election in INST.ELECTIONS_LIST:
        logger.info(f'election: {election}')
        for model in INST.MODELS[election]:
            logger.info(f'model: {model}')
            for data in INST.DATASETS:
                logger.info(f'dataset: {data}')
                for alg in INST.ALGORITHMS:
                    logger.info(f'algorithm: {alg}')
                    for suffix in INST.SUFFIXES:
                        if 'emoji' in suffix and 'covid' not in model:
                            continue
                        logger.info(f'suffix: {suffix}')
                        compute_sentences_result_per_user(election, data, model, alg, suffix=suffix, threshold_value=None)
                logger.info('------------------')
            logger.info('-.-.-.-.-.-.-.-.-.--.')
        logger.info('----.....----.....----...')
    logger.info('DONE')


if __name__ == '__main__':
    try:
        # To compute the users' stance for all VAA elections
        T2S_all_elections()

        # For a single election and configuration (language model, dataset, algorithm, threshold_value:
        #election = 'IT19'
        #suffix = ''

        # threshold_value = None if you want to run Tweets2Stance for every threshold value (for tweet filtering).
        # Otherwise, set threshold_value = <a_threshold_value>, e.g., threshold_value = 0.6

        #threshold_value = None
        #compute_sentences_result_per_user(election, INST.OPT_MODEL_NAME, INST.OPT_DATASET, INST.OPT_ALGORITHM, suffix=suffix, threshold_value=threshold_value)
    except Exception as e:
        logger.exception('ERROR DETECTED: ')