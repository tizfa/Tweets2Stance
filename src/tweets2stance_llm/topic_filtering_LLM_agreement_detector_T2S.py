import sys
import glob
sys.path.append('..')
from src.tweets2stance_llm.utils.utils_load import *
from src.tweets2stance_llm.utils.utils_pickle import *
from src.tweets2stance_llm.utils.utils import *

from src.tweets2stance_zsc.tweets2stance import compute_final_label


def save_tweets():
    print('saving tweets in a suitable format...')
    os.makedirs(os.path.dirname(data_folder_LLM_filter_T2S_stance), exist_ok=True)
    dataset = OPTIMAL_SETTINGS_T2S_ZSC['dataset']

    for llm_type in ['mixtral', 'GPT4']:
        print(f'LLM: {llm_type}')
        for election, parties in PARTIES_VAA.items():
            print(f'Election: {election}')
            tweets_T2S = load_pickle(f'{data_folder}/{election}/', f'{election}_{dataset}_df_zsc_tweets')

            for party in parties:
                if party in PARTIES_TO_REMOVE:
                    continue
                print(f'party: {party}')
                for path_topic_csv in glob.glob(f"{data_folder_LLM}/{llm_type}/{dataset}/{election}/{party}/*.csv"):
                    topic = path_topic_csv.split('/')[-1].replace('.csv', '')
                    tweets_filtered_llm = pd.read_csv(path_topic_csv, dtype={'tweet_id': str})

                    tmp_tweets_T2S = tweets_T2S[(tweets_T2S['topic'] == topic) & (tweets_T2S['screen_name'] == party)][
                        ['tweet_id', 'screen_name', 'topic', 'sentence', 'score_topic', 'score_sentence']]

                    tmp = pd.merge(tweets_filtered_llm, tmp_tweets_T2S, how='left', left_on='tweet_id',
                                   right_on='tweet_id')

                    filepath = f'{data_folder_LLM_filter_T2S_stance}/{llm_type}/{election}/{party}'
                    os.makedirs(os.path.dirname(filepath + '/'), exist_ok=True)

                    tmp.to_csv(f'{filepath}/{topic}.csv', index=False)


def compute_sentences_result_per_party():
    print('compute_sentences_result_per_party()')
    algorithm = OPTIMAL_SETTINGS_T2S_ZSC['algorithm']
    for llm_type in ['mixtral', 'GPT4']:
        print(f'LLM: {llm_type}')
        results = list()
        for election, parties in PARTIES_VAA.items():
            statements_df = load_sentences(election)

            min_num_tweets = None
            if 'algorithm_4' in algorithm:
                s = re.search('_4_min_num_tweets_(\d)', algorithm)
                if not s:
                    sys.exit(-1)
                min_num_tweets = int(s.group(1))

            for party in parties:
                if party in PARTIES_TO_REMOVE:
                    continue

                print(f'party: {party}')

                for path_topic_csv in glob.glob(
                        f"{data_folder_LLM_filter_T2S_stance}/{llm_type}/{election}/{party}/*.csv"):
                    topic = path_topic_csv.split('/')[-1].replace('.csv', '')
                    sentence = statements_df[statements_df['topic'] == topic].sentence.values[0]
                    actual_df = pd.read_csv(path_topic_csv, dtype={'score_sentence': float})

                    print(f'topic: {topic}')

                    if actual_df.empty:
                        print(f'df empty for party {party}, topic \'{topic}\', | setting 3 as the answer')
                        # VAA, Party, Topic, Sentence, Stance
                        results.append([election, party, sentence, topic, 3])
                        continue

                    if min_num_tweets and len(actual_df) < min_num_tweets:
                        print(
                            f'less than min num tweets {min_num_tweets} for party {party}, topic \'{topic}\' | setting 3 as the answer')
                        results.append([election, party, sentence, topic, 3])
                        continue
                    elif not min_num_tweets:
                        sys.exit(-1)

                    # since the optimal algorithm is algorithm4_min_num_tweets_3, we reported the code for this
                    # algorithm only.
                    actual_df['label_tweet'] = actual_df['score_sentence'].apply(
                        lambda score_s: compute_final_label(score_s, [0.25, 0.5, 0.75]))
                    tmp_df = pd.DataFrame(actual_df['label_tweet'].value_counts())

                    majority_labels = list(tmp_df[tmp_df['label_tweet'] == tmp_df.values.max()].index)

                    if len(majority_labels) == 1:
                        answer = majority_labels[0]
                    else:
                        answer = round(sum(actual_df['label_tweet']) / len(actual_df))

                    if answer == 3:
                        answer = 4
                    elif answer == 4:
                        answer = 5

                    results.append([election, party, sentence, topic, answer])

        df_tot = pd.DataFrame(results, columns=['VAA', 'Party', 'Sentence', 'Topic', 'Stance'])
        final_path = f"{data_folder_LLM_filter_T2S_stance}/{llm_type}"
        filename = f"{final_path}/results_LLM_filtering_step_stance_algorithm_4.csv"

        df_tot.to_csv(filename, index=False)
        print(f'Saved file: {filename}')


if __name__ == '__main__':
    save_tweets()
    compute_sentences_result_per_party()