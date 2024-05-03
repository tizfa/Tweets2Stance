'''
Topic Filtering and Agreement Detector step implemented with LLM.

First, execute it for topic filtering, then for stance detection.
'''

import argparse
import json
import csv
import time
import ast
import glob
from openai import AzureOpenAI, BadRequestError
import tiktoken

from src.tweets2stance_llm.utils.utils import *
from src.tweets2stance_llm.utils.utils_load import *

jupyter_ = False

class Executor:
    def __init__(self, args, logger):
        self.recovering_file_path = f'{log_dir}/recovering'
        self.logger = logger
        self.args = args
        self.args.num_labels = int(args.num_labels)
        self.recovering_data = {}
        self.llm = None
        self.summed_seconds = 0
        self.how_many = 0
        self.csv_writers = {}
        self.csv_writer_stance_detection = {}
        self.count_tweets = 0
        self.count_done = 0
        self.gpt4_enc = tiktoken.encoding_for_model("gpt-4")
    
    def set_recover(self):
        self.recovering_file_path += f'_{self.args.llm_type}_step_{self.args.mode_type}_{self.args.num_labels}.json'
        got_from_file = False
        if os.path.exists(self.recovering_file_path):
            # Open the existing file
            with open(self.recovering_file_path, 'r') as file:
                try:
                    self.recovering_data = json.load(file)
                    got_from_file = True
                    self.logger.info('Got recovering data from file')
                except Exception as e:
                    self.logger.error(f'Error in getting recovering data from file: {str(e)}')

        if not got_from_file:
            self.recovering_data = {
                # <dataset> -> <election> -> <user> -> tweets e flag done -> dict of tweets
            }
            for d in DATASETS:
                self.recovering_data[d] = {}
                for e, parties in PARTIES_VAA.items():
                    self.recovering_data[d][e] = {}
                    for party in parties:
                        self.recovering_data[d][e][party] = {
                            'done': False
                        }
                        if self.args.mode_type == 'topic_filtering':
                            self.recovering_data[d][e][party]['tweets'] = {}
                        else:
                            self.recovering_data[d][e][party]['topics'] = {}
            with open(self.recovering_file_path, 'w') as file:
                json.dump(self.recovering_data, file)
            self.logger.info('New recovering data')
            
        self.logger.info(f'Recovering data: {self.recovering_data}')

    def set_csv_writers(self, statements, dataset, election, party):

        if self.args.mode_type == 'topic_filtering':
            self.logger.info('Setting csv writers for topic filtering')
            self.csv_writers[dataset] = {
                election: {
                    party: {}
                }
            }
    
            for topic in statements:
                # topics in this case
                if 'use of marijuana' in topic:
                    topic = topic.replace('/', ' or ')

                filepath_csv = f'{data_folder_LLM}/{self.args.llm_type}/{dataset}/{election}/{party}/{topic}.csv'
                os.makedirs(os.path.dirname(filepath_csv), exist_ok=True)
                    
                write_header = False
                if not os.path.exists(filepath_csv):
                    write_header = True

                f = open(filepath_csv, mode= 'a+')
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if write_header:
                    csv_writer.writerow(['tweet_id', 'user_id', 'user_screen_name', 'tweet'])
                    f.flush()
                    
                self.csv_writers[dataset][election][party][topic.lower()] = {
                    'csv_writer': csv_writer,
                    'file': f
                }
        else:
            self.logger.info('Setting csv writers for "compute stance". A single csv file with columns VAA, Party, Topic, Stance')

            filepath_csv = f'{data_folder_LLM}/{self.args.llm_type}/{dataset}/results_step_stance_detection_{self.args.num_labels}_labels.csv'
            os.makedirs(os.path.dirname(filepath_csv), exist_ok=True)

            write_header = False
            if not os.path.exists(filepath_csv):
                write_header = True

            f = open(filepath_csv, mode='a+')
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if write_header:
                csv_writer.writerow(['VAA', 'Party', 'Topic', 'Sentence', 'Stance'])
                f.flush()

            self.csv_writer_stance_detection[dataset] = {
                'csv_writer': csv_writer,
                'file': f
            }

    def close_csv_writers(self, statements, dataset, election, party):
        if self.args.mode_type == 'topic_filtering':
            self.logger.info('Closing files and csv writers for topic filtering')
            for topic in statements:
                self.csv_writers[dataset][election][party][topic.lower()]['file'].close()
        else:
            self.logger.info('Closing files and csv writers for "compute stance"')
            self.csv_writer_stance_detection[dataset]['file'].close()

    def add_tweet_recovering_data(self, dataset, election, party, tweet_id):
        self.recovering_data[dataset][election][party]['tweets'][tweet_id] = True
        with open(self.recovering_file_path, 'w') as file:
                json.dump(self.recovering_data, file)

    def add_topic_recovering_data(self, dataset, election, party, topic):
        self.recovering_data[dataset][election][party]['topics'][topic] = True
        with open(self.recovering_file_path, 'w') as file:
                json.dump(self.recovering_data, file)

    def done_party_recovering_data(self, dataset, election, party):
        self.recovering_data[dataset][election][party]['done'] = True
        with open(self.recovering_file_path, 'w') as file:
                json.dump(self.recovering_data, file)

    def get_prompt(self, statements, data_str, sentence=None):
        if self.args.mode_type == 'topic_filtering':
            prompt = get_prompt_topic_filtering(statements, data_str)
        else:
            prompt = get_prompt_compute_stance(data_str, sentence, self.args.num_labels)
        return prompt

    def process_topic_filtering(self, tweets, dataset, election, party, topics):
        self.logger.info(f'Total tweets for {election}, {dataset}, {party}: {tweets.shape[0]}')
        count = 0
        for index, row in tweets.iterrows():
            if self.recovering_data[dataset][election][party]['tweets'].get(row['tweet_id']):
                self.logger.info(f'{election}, {dataset}, {party} ALREADY DONE. Going on.')
                count += 1
                self.count_tweets += 1
                self.logger.info(f'Done {count} over {tweets.shape[0]} for {election}, {dataset}, {party}')
                self.logger.info(f'Done {self.count_tweets} tweets over the total number.')
                continue

            data_str = row['tweet'].replace('"', '')
            self.logger.info(f'Processing tweet: {row["tweet_id"]}')
            
            prompt = self.get_prompt(topics, data_str.replace('[', '').replace(']', ''))
            #prompt += data_str

            self.logger.info('Starting LLM!!!')
            self.logger.info(f'Prompt: {prompt}END')
            start_time = time.time()
            
            # Simple inference example
            if self.args.llm_type == 'mixtral':
                output = self.llm(
                    f"[INST] {prompt} [/INST]", # Prompt
                    max_tokens=-1,  # Generate up to 512 tokens
                    stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
                    echo=False,       # Whether to echo the prompt
                    #seed=42
                    temperature=0,
                    top_k=40,
                    repeat_penalty=1.1,
                    min_p=0.05,
                    top_p=0.95
                )
            else:
                self.logger.info('GPT4 using AzureAPI')
                client = AzureOpenAI(
                    api_key="201f106463c943daa9e3706b0b58e0f0",
                    api_version="2023-07-01-preview",
                    azure_endpoint="https://cnr1openai.openai.azure.com/"
                )

                tokens = self.gpt4_enc.encode(prompt)
                self.logger.info(f'No. of tokens for this prompt: {len(tokens)}')
                output = None

                try:
                    output = client.chat.completions.create(
                        model="VAA",  # e.g. gpt-35-instant
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        max_tokens=256,
                        temperature=0.2,
                        seed=42,
                        #top_p=0.95,
                        #top_k=40,
                        #repeat_penalty=1.1
                    )
                except BadRequestError as be:
                    self.logger.info(f'BAD REQUEST ERROR: {str(be)}')
                    #self.logger.info(f'Setting matching topics to []')
                    #reply = 'Bad Request'
                
            #self.llm.reset()
            elapsed_time = time.time() - start_time

            # Print the elapsed time in seconds
            self.summed_seconds += elapsed_time
            self.how_many += 1
            self.logger.info(f"Elapsed time: {elapsed_time} seconds. AVG: {self.summed_seconds/self.how_many}")
            
            reply = ''
            if self.args.llm_type == 'mixtral':
                reply = output['choices'][0]['text']
            else:
                reply = output.choices[0].message.content if output and output.choices[0].message.content else 'Bad Request'
                
            self.logger.info(f'Reply: {reply}')
            
            regex = '\[.*?\]'#' *\[( ?\d+ ?,)* ?\d+ ?\]|\[ *\]'
            
            indices_topics = list()

            reply = re.sub(' ? \/\/ ?implied by ".*?" ?\n?', '', reply)
            reply = re.sub(' ? \/\/ .*? implies ".*?" ?\n?', '', reply)
            m = re.search(regex, reply.replace('\n', ' ').replace('centers','centres'))
            try:
                matching_topics_ = list()
                matching_topics = list()
                if m:
                    try:
                        matching_topics_ = ast.literal_eval(m.group())
                    except Exception as e:
                        self.logger.info(f'Error in ast.literal_eval ({str(e)}). Not considering this answer. No matches. Going on.')
                        self.logger.info('For debugging purposes: ')
                        self.logger.info(f'Prompt: {prompt}')
                        self.logger.info(f'Reply: {reply}')
                        count += 1
                        self.count_tweets += 1
                        self.logger.info(f'Done {count} over {tweets.shape[0]} for {election}, {dataset}, {party}')
                        self.logger.info(f'Done {self.count_tweets} tweets over the total number.')
                        
                else:
                    self.logger.info('No match found in reply.')
                
                #self.logger.info(f'Getting matching topics from {indices_in_topic_tweets}')
                #matching_topics = np.take(topics, indices_in_topic_tweets)
                self.logger.info(f'Matching topics from reply: {matching_topics_}')
                # Checking topics:
                for t in matching_topics_:
                    try:
                        self.csv_writers[dataset][election][party][t]
                        matching_topics.append(t)
                    except Exception as e:
                        self.logger.info(f'TOPIC ERROR: {str(e)}. Not considering "{t}"')
                        pass

                self.logger.info(f'Matching topics: {matching_topics}')
                self.logger.info(f'Saving tweet in topic file. Tweet: {row["tweet"]}')
                for t in matching_topics:
                    self.csv_writers[dataset][election][party][t]['csv_writer'].writerow([row['tweet_id'], row['user_id'], row['user_screen_name'], row['tweet']])
                    self.csv_writers[dataset][election][party][t]['file'].flush()
                    self.logger.info(f'wrote tweet to {dataset}/{election}/{party}/{t}.csv')

                self.add_tweet_recovering_data(dataset, election, party, row['tweet_id'])
                self.count_tweets += 1
                count += 1
                self.logger.info(f'Done {count} over {tweets.shape[0]} for {election}, {dataset}, {party}')
                self.logger.info(f'Done {self.count_tweets} tweets over the total number.')
            except Exception as e:
                self.logger.error(f'error processing prompt: {str(e)}')
                self.logger.info(f'Prompt: {prompt}')
                self.logger.info(f'Reply: {reply}')
                raise e

    def process_stance(self, all_tweets_party, dataset, election, party, statements_df):
        golden_labels = get_golden_labels(election, party)
        for path_topic_csv in glob.glob(f"{data_folder_LLM}/{self.args.llm_type}/{dataset}/{election}/{party}/*.csv"):
            topic = path_topic_csv.split('/')[-1].replace('.csv', '')

            if self.recovering_data[dataset][election][party]['topics'].get(topic):
                self.logger.info(f'{election}, {dataset}, {party} ALREADY DONE. Going on.')
                self.count_done += 1
                self.logger.info(f'Done {self.count_done} configurations over the total number.')
                continue

            # getting tweets filtered during the topic filtering step
            tweets = pd.read_csv(path_topic_csv, dtype={'tweet_id': str})
            sentence = statements_df[statements_df['topic'] == topic].sentence.values[0]
            self.logger.info(f'Sentence: {sentence}')

            try:
                if len(tweets.tweet.values) > 0:
                    # using not cleaned tweets
                    tweets = pd.merge(tweets[['tweet_id']], all_tweets_party[
                        ['tweet_id', 'tweet', 'date_obj' if election != 'GB19' else 'created_at']], how='inner',
                                      left_on='tweet_id', right_on='tweet_id')

                    # elections, party, topic needing cut
                    if election == 'GB19' and self.args.llm_type == 'mixtral':
                        tweets.sort_values(by=['date_obj' if election != 'GB19' else 'created_at'], ascending=False,
                                           inplace=True)
                        self.logger.info(
                            f'Need to cut tweets for ({election}, {party}, {topic}). Current tweets: {tweets.shape[0]}')
                        if party == 'Conservatives' and topic == 'UK membership in EU':
                            tweets = tweets[:715]  # first prompt: 717
                        elif party == 'Conservatives' and topic == 'no deal for brexit':
                            tweets = tweets[:680]  # first prompt: 681
                        elif party == 'TheGreenParty' and topic == 'UK membership in EU':
                            tweets = tweets[:566]  # first prompt: 568
                        elif party == 'LibDems' and topic == 'UK membership in EU':
                            tweets = tweets[:562]  # first prompt: 563
                        self.logger.info(f'No. of tweets to consider: {tweets.shape[0]}')
                    # GB19, Party: Conservatives, Topic: UK membership in EU, tweets to consider: 726

                    tweets_str = '~|~'.join(tweets.tweet.values)
                    prompt = self.get_prompt(None, tweets_str, sentence)
                    reply = ''
                    self.logger.info('Starting LLM!!!')
                    self.logger.info(f'Prompt: {prompt}END')
                    start_time = time.time()
                    # Simple inference example
                    output = ''
                    if self.args.llm_type == 'mixtral':
                        output = self.llm(
                            f"[INST] {prompt} [/INST]",  # Prompt
                            max_tokens=-1,  # Generate up to 512 tokens
                            stop=["</s>"],
                            # Example stop token - not necessarily correct for this specific model! Please check before using.
                            echo=False,  # Whether to echo the prompt
                            # seed=42
                            temperature=0,  # prompt1 and prompt2: temperature 0
                            top_k=40,
                            repeat_penalty=1.1,
                            min_p=0.05,
                            top_p=0.95
                        )
                        self.llm.reset()
                    else:
                        self.logger.info('GPT4 using AzureAPI')
                        client = AzureOpenAI(
                            api_key="201f106463c943daa9e3706b0b58e0f0",
                            api_version="2023-07-01-preview",
                            azure_endpoint="https://cnr1openai.openai.azure.com/"
                        )

                        tokens = self.gpt4_enc.encode(prompt)
                        self.logger.info(f'No. of tokens for this prompt: {len(tokens)}')
                        output = None

                        try:
                            output = client.chat.completions.create(
                                model="VAA",  # e.g. gpt-35-instant
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt,
                                    },
                                ],
                                max_tokens=512,
                                temperature=0.2,
                                seed=42,
                                # top_p=0.95,
                                # top_k=40,
                                # repeat_penalty=1.1
                            )
                        except BadRequestError as be:
                            self.logger.info(f'BAD REQUEST ERROR: {str(be)}')
                        # self.logger.info('GPT4 API still not used!!!')
                        # sys.exit(-1)

                    elapsed_time = time.time() - start_time

                    # Print the elapsed time in seconds
                    self.summed_seconds += elapsed_time
                    self.how_many += 1
                    self.logger.info(
                        f"Elapsed time: {elapsed_time} seconds. AVG: {self.summed_seconds / self.how_many}")

                    reply = ''
                    if self.args.llm_type == 'mixtral':
                        reply = output['choices'][0]['text']
                    else:
                        reply = output.choices[0].message.content if output and output.choices[
                            0].message.content else 'Bad Request'

                    self.logger.info(f'Reply: {reply}')
                    regex = '"stance": "(completely disagree|disagree|neither disagree nor agree|agree|completely agree|neutral|none|unknown|neither_disagree_nor_agree|indirectly disagree|insufficient_information|neutral leaning towards agree|undetermined|indeterminable)"'
                    m = re.search(regex, reply)
                    if not m:
                        self.logger.info(
                            'Reply with erroneous format. But maybe I can go on if the reply refers to the neither disagree nor agree stance.')
                        m_ = re.search(
                            'Based on the provided tweets, the (.*?) \'?\"?(completely disagree|disagree|neither disagree nor agree|agree|completely agree|neutral|none|unknown|neither_disagree_nor_agree|indirectly disagree|insufficient_information|neutral leaning towards agree)\'?\"?',
                            reply)
                        # m_ = re.search("Based on the provided tweets, the (user\'s )?stance towards \".*?\" is \'neither disagree nor agree\'", reply)
                        if not m_ and 'Therefore, I cannot provide a JSON formatted response with a clear stance' not in reply and 'Based on the provided tweets, the stance is "neither disagree nor agree"' not in reply and 'Based on the provided tweet, I cannot determine any stance towards the statement' not in reply and 'Based on the provided tweets, the stance of the users towards the statement "the federal government should guarantee a minimum income for all Canadian adults regardless of whether or not they have a job" is \'neither disagree nor agree\'' not in reply and 'Based on the provided tweets, the user\'s stance towards "Quebec should become an independent state" is \'neither disagree nor agree\'' not in reply and 'Based on the provided tweets, I cannot determine a clear stance' not in reply and 'Stance: neither disagree nor agree' not in reply and 'Based on the provided tweets, the stance of the user is \'neither disagree nor agree\'' not in reply and 'seems to be \'neither disagree nor agree\'' not in reply and 'the stance appears to be closer to \'disagree\' than any other option' not in reply:
                            self.logger.error('Reply with erroneous format. TERMINATING')
                            sys.exit(-1)
                        elif '(but leans towards agree)' in reply:
                            stance = 'agree'
                        elif "'disagree'" in reply:
                            stance = 'disagree'
                        else:
                            stance = m_.group(2)  # 'neither disagree nor agree'
                    else:
                        stance = m.group(1)
                        if stance == 'neutral' or stance == 'neither_disagree_nor_agree' or stance == 'none' or stance == 'insufficient_information' or stance == 'unknown' or stance == 'undetermined' or stance == 'indeterminable':
                            stance = 'neither disagree nor agree'
                        elif stance == 'indirectly disagree':
                            stance = 'disagree'
                        elif stance == 'neutral leaning towards agree':
                            stance = 'agree'
                else:
                    self.logger.info('NO TWEETS AVAILABLE!!! Stance is "neither disagree nor agree"')
                    stance = 'neither disagree nor agree'

                self.logger.info(f'STANCE: {stance}')
                stance_int = MAP_STANCE_TO_INT[stance]
                self.logger.info(f'Golden label: {golden_labels[sentence]}')
                self.csv_writer_stance_detection[dataset]['csv_writer'].writerow(
                    [election, party, topic, sentence, stance_int])
                self.csv_writer_stance_detection[dataset]['file'].flush()
                self.logger.info(
                    f'Saved file {data_folder_LLM}/{self.args.llm_type}/{dataset}/results_step_stance_detection_{self.args.num_labels}_labels.csv')

                self.add_topic_recovering_data(dataset, election, party, topic)
                self.count_done += 1
                self.logger.info(f'Done {self.count_done} configurations over the total number.')
                self.logger.info('==========')
            except Exception as e:
                self.logger.error(f'error processing prompt: {str(e)}')
                self.logger.info(f'Prompt: {prompt}')
                self.logger.info(f'Reply: {reply}')
                raise e

    def run_llm(self, tweets, dataset, election, party):
        field = 'topic' if self.args.mode_type == 'topic_filtering' else 'sentence'
        statements = load_sentences(election)[field].values

        self.set_csv_writers(statements, dataset, election, party)

        if self.args.mode_type == 'topic_filtering':
            dir_filtering = f'{data_folder_LLM}/{self.args.llm_type}/{dataset}/{election}/{party}'
            if not os.path.exists(dir_filtering):
                os.makedirs(dir_filtering)

        if self.args.mode_type == 'topic_filtering':
            self.process_topic_filtering(tweets, dataset, election, party, statements)
        else:
            info_context_length = load_pickle(f'{data_folder_insights}/',
                                              f'token_length_before_after_topic_filtering_{self.args.llm_type}')
            offset = info_context_length[self.args.llm_type][election][party]['num_tweets_all_cut']
            self.logger.info(
                f'Num_tweets to consider to not get out of context (tweets were ordered from the most recent): {offset[0]}')
            tweets = tweets[:offset[0]]
            self.process_stance(tweets, dataset, election, party, statements)

        self.close_csv_writers(statements, dataset, election, party)
    
    def set_llm(self):
        if self.args.llm_type == 'mixtral':
            from llama_cpp import Llama
            # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    
            self.llm = Llama(
                model_path="/workspace/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  # Download the model file first
                n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
                n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
                n_gpu_layers=-1,         # The number of layers to offload to GPU, if you have GPU acceleration available
                #seed=42,
                #temperature=0
            )

    def run(self):
        self.set_recover()
    
        self.set_llm()
    
        if self.args.llm_type == 'mixtral' and not self.llm:
            self.logger.error('llm is None!!!')
            sys.exit(-1)
            
        for dataset in DATASETS:
            for election, parties in PARTIES_VAA.items():
                for party in parties:
                    try:
                        if party in PARTIES_TO_REMOVE:
                            continue

                        tweets = load_raw_vaa(election, dataset)
                        tweets = tweets[tweets['user_screen_name'] == party]
                        self.logger.info(f'tweets ({election}, {dataset}, {party}): {tweets.shape}')
                        
                        if self.recovering_data[dataset][election][party]['done']:
                            self.count_tweets += tweets.shape[0]
                            self.logger.info(
                                f'{election}, {dataset}, {party} ALREADY DONE. Count_tweets: {self.count_tweets}.Going on.')
                            if self.args.mode_type == 'compute_stance':
                                tmp_topics = glob.glob(
                                    f"{data_folder_LLM}/{self.args.llm_type}/{dataset}/{election}/{party}/*.csv")
                                self.count_done += len(tmp_topics)
                                self.logger.info(f'Done {self.count_done} configurations over the total number.')
                            continue
                        
                        self.logger.info('cleaning tweets...')
                        remove_emoji = True if self.args.mode_type == 'topic_filtering' else False
                        self.logger.info(f'Remove emoji: {remove_emoji}')
                        tweets = clean(tweets, self.logger, remove_emoji=remove_emoji)
                        tweets.reset_index(drop=True, inplace=True)
                        
                        self.run_llm(tweets, dataset, election, party)
                        
                        self.done_party_recovering_data(dataset, election, party)
    
                        self.logger.info('Resetting the llm')
                        self.set_llm()
                        #https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api
                    except Exception as e:
                        self.logger.error(f'error: {str(e)}')
                        self.logger.info(f'CURRENT_DATA: {election}, {dataset}, {party}')
                        raise e


def main():
    parser = argparse.ArgumentParser(description='Python Wrapper Tweepy')
    parser.add_argument('-llm', '--llm_type', type=str, help='llm_type')
    parser.add_argument('-mode', '--mode_type', type=str, help='mode_type')
    parser.add_argument('-nl', '--num_labels', type=str, help='llm_type')
    args = parser.parse_args()
    if args.llm_type not in MODELS:
        print('llm_type should be either "mixtral" or "GPT4"')
        sys.exit(-1)
    if args.mode_type not in ['topic_filtering', 'compute_stance']:
        print('mode_type should be "topic_filtering" or "compute_stance"')
        sys.exit(-1)
    if args.num_labels not in ['5', '3']:
        print('num_labels should be either "5" or "3"')
        sys.exit(-1)

    logger = setup_logger('execution_log', f'{log_dir}/log_simulate_with_{args.llm_type}_step_{args.mode_type}.log')
    logger.info(f'LLM: {args.llm_type}, mode_type: {args.mode_type}')

    try:
        ex = Executor(args, logger)
        ex.run()
        logger.info(f'DONE ALL!!!')
    except Exception:
        logger.exception('EXCEPTION: ')
        sys.exit(-1)


if __name__ == "__main__":
    main()
    