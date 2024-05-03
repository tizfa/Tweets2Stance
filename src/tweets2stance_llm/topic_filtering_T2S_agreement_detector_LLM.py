'''
Topic Filtering with T2S, Agreement Detector with LLM.

Topic Filtering's data is retrievede with script src/tweets2stance/topic_filtering_T2S.py
'''

import argparse
import json
import csv
import time
import glob
import tiktoken
from openai import AzureOpenAI, BadRequestError

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
        self.recovering_file_path += f'T2S_filtered_{self.args.llm_type}_step_{self.args.mode_type}_{self.args.num_labels}.json'
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
                            'topics': {},
                            'done': False
                        }
            with open(self.recovering_file_path, 'w') as file:
                json.dump(self.recovering_data, file)
            self.logger.info('New recovering data')
            
        self.logger.info(f'Recovering data: {self.recovering_data}')

    def set_csv_writers(self, dataset):
        self.logger.info('Setting csv writers for "compute stance". A single csv file with columns VAA, Party, Topic, Stance')

        filepath_csv = f'{data_folder_T2S}/{dataset}/results_T2S_filtered_step_stance_detection_with_{self.args.llm_type}_{self.args.num_labels}_labels.csv'
        os.makedirs(os.path.dirname(filepath_csv), exist_ok=True)

        write_header = False
        if not os.path.exists(filepath_csv):
            write_header = True

        f = open(filepath_csv, mode= 'a+')
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if write_header:
            csv_writer.writerow(['VAA', 'Party', 'Topic', 'Sentence', 'Stance'])
            f.flush()

        self.csv_writer_stance_detection[dataset] = {
            'csv_writer': csv_writer,
            'file': f
        }

    def close_csv_writers(self, dataset):
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

    def process_stance(self, dataset, election, party, statements_df):
        golden_labels = get_golden_labels(election, party)
        for path_topic_csv in glob.glob(f"{data_folder_T2S}/{dataset}/{election}/{party}/*.csv"):
            topic = path_topic_csv.split('/')[-1].replace('.csv', '') #or
            
            if self.recovering_data[dataset][election][party]['topics'].get(topic):
                self.logger.info(f'{election}, {dataset}, {party} ALREADY DONE. Going on.')
                self.count_done += 1
                self.logger.info(f'Done {self.count_done} configurations over the total number.')
                continue
                
            tweets = pd.read_csv(path_topic_csv, dtype={'tweet_id': str})
            self.logger.info(f'topic: {topic}')
            if ' or ' in topic:
                topic = topic.replace(' or ', '/')
            sentence = statements_df[statements_df['topic'] == topic].sentence.values[0]
            self.logger.info(f'Sentence: {sentence}')
            
            try:
                if len(tweets.tweet.values) > 0:
                    # not cleaned tweets
                    #tweets = pd.merge(tweets[['tweet_id']], all_tweets_party[['tweet_id', 'tweet', 'date_obj' if election != 'GB19' else 'created_at']], how='inner', left_on='tweet_id', right_on='tweet_id')
                    
                    # elections, party, topic needing cut
                    if election == 'GB19' and self.args.llm_type == 'mixtral':
                        self.logger.info(f'Need to cut tweets for ({election}, {party}, {topic}). Current tweets: {tweets.shape[0]}')
                        # check notebook/getTokenLengthT2S.ipynb
                        if party == 'Conservatives' and topic == 'UK membership in EU':
                            tweets = tweets[:813] # first prompt: 813
                        elif party == 'TheGreenParty' and topic == 'UK membership in EU':
                            tweets = tweets[:644] # first prompt: 644
                        self.logger.info(f'No. of tweets to consider: {tweets.shape[0]}')
                    # GB19, Party: Conservatives, Topic: UK membership in EU, tweets to consider: 726
                    
                    tweets_str = '~|~'.join(tweets.tweet.values)
                    prompt = get_prompt_compute_stance(tweets_str, sentence, self.args.num_labels)
                    reply = ''
                    self.logger.info('Starting LLM!!!')
                    self.logger.info(f'Prompt: {prompt}END')
                    start_time = time.time()
                    # Simple inference example
                    output = ''
                    if self.args.llm_type == 'mixtral':
                        output = self.llm(
                            f"[INST] {prompt} [/INST]", # Prompt
                            max_tokens=-1,  # Generate up to 512 tokens
                            stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
                            echo=False,       # Whether to echo the prompt
                            #seed=42
                            temperature=0, # prompt1 and prompt2: temperature 0
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
                                #top_p=0.95,
                                #top_k=40,
                                #repeat_penalty=1.1
                            )
                        except BadRequestError as be:
                            self.logger.info(f'BAD REQUEST ERROR: {str(be)}')
    
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
                    regex = '"stance": "(completely disagree|disagree|neither disagree nor agree|agree|completely agree|neutral|none|unknown|neither_disagree_nor_agree|indirectly disagree|insufficient_information|neutral leaning towards agree|undetermined|indeterminable|no stance provided based on the tweets|lean agree|cannot be determined from the tweets provided|cannot be determined from the given tweets|not mentioned|undefined|not applicable|not available|cannot be determined from the provided tweets|not determinable from the given information|.+)"'
                    m = re.search(regex, reply)
                    if not m:
                        self.logger.info('Reply with erroneous format. But maybe I can go on if the reply refers to the neither disagree nor agree stance.')
                        m_ = re.search('(Based on the provided tweets|based solely on the information provided in the tweets|based on the information provided in the tweets), (.*?) \'?\"?(completely disagree|disagree|neither disagree nor agree|agree|completely agree|neutral|none|unknown|neither_disagree_nor_agree|indirectly disagree|insufficient_information|neutral leaning towards agree)\'?\"?', reply)
                        #m_ = re.search('Based on the provided tweets, the (.*?) \'?\"?(completely disagree|disagree|neither disagree nor agree|agree|completely agree|neutral|none|unknown|neither_disagree_nor_agree|indirectly disagree|insufficient_information|neutral leaning towards agree)\'?\"?', reply)
                        #m_ = re.search("Based on the provided tweets, the (user\'s )?stance towards \".*?\" is \'neither disagree nor agree\'", reply)
                        if not m_ and 'Therefore, I cannot provide a JSON formatted response with a clear stance' not in reply and 'Based on the provided tweets, the stance is "neither disagree nor agree"' not in reply and 'Based on the provided tweet, I cannot determine any stance towards the statement' not in reply and 'Based on the provided tweets, the stance of the users towards the statement "the federal government should guarantee a minimum income for all Canadian adults regardless of whether or not they have a job" is \'neither disagree nor agree\'' not in reply and 'Based on the provided tweets, the user\'s stance towards "Quebec should become an independent state" is \'neither disagree nor agree\'' not in reply and 'Based on the provided tweets, I cannot determine a clear stance' not in reply and 'Stance: neither disagree nor agree' not in reply and 'Based on the provided tweets, the stance of the user is \'neither disagree nor agree\'' not in reply and 'seems to be \'neither disagree nor agree\'' not in reply and 'the stance appears to be closer to \'disagree\' than any other option' not in reply and 'promise to balance the budget by 2' not in reply:
                            self.logger.error('Reply with erroneous format. TERMINATING')
                            sys.exit(-1)
                        elif '(but leans towards agree)' in reply:
                            stance = 'agree'
                        elif "'disagree'" in reply:
                            stance = 'disagree'
                        elif 'promise to balance the budget by 2' in reply:
                            stance = 'agree'
                        else: 
                            stance = m_.group(3)#'neither disagree nor agree'
                    else:
                        found_stance = m.group(1)
                        if found_stance in ['completely disagree', 'disagree', 'neither disagree nor agree', 'agree', 'completely agree']:
                            stance = found_stance
                        else:
                            stance = 'neither disagree nor agree'
                            
                        if stance == 'neutral' or stance == 'neither_disagree_nor_agree' or stance == 'none' or stance == 'insufficient_information' or stance == 'unknown' or stance == 'undetermined' or stance == 'indeterminable' or stance == 'no stance provided based on the tweets' or stance == 'cannot be determined from the tweets provided' or stance == 'cannot be determined from the given tweets' or stance == 'not mentioned' or stance == 'undefined' or stance == 'not applicable' or stance == 'not available' or stance == 'cannot be determined from the provided tweets' or stance in ['not determinable from the given information']:
                            stance = 'neither disagree nor agree'
                        elif stance == 'indirectly disagree':
                            stance = 'disagree'
                        elif stance == 'neutral leaning towards agree' or stance == 'lean agree':
                            stance = 'agree'
                else:
                    self.logger.info('NO TWEETS AVAILABLE!!! Stance is "neither disagree nor agree"')
                    stance = 'neither disagree nor agree'
                    
                self.logger.info(f'STANCE: {stance}')
                stance_int = MAP_STANCE_TO_INT[stance]
                self.logger.info(f'Golden label: {golden_labels[sentence]}')
                self.csv_writer_stance_detection[dataset]['csv_writer'].writerow([election, party, topic, sentence, stance_int])
                self.csv_writer_stance_detection[dataset]['file'].flush()
                self.logger.info(f'Saved file {data_folder_T2S}/{dataset}/results_T2S_filtered_step_stance_detection_with_{self.args.llm_type}_{self.args.num_labels}_labels.csv')
                
                self.add_topic_recovering_data(dataset, election, party, topic)
                self.count_done += 1
                self.logger.info(f'Done {self.count_done} configurations over the total number.')
                self.logger.info('==========')
            except Exception as e:
                self.logger.error(f'error processing prompt: {str(e)}')
                self.logger.info(f'Prompt: {prompt}')
                self.logger.info(f'Reply: {reply}')
                raise e
    
    def run_llm(self, dataset, election, party):
        self.set_csv_writers(dataset)

        statements_df = load_sentences(election)
        self.process_stance(dataset, election, party, statements_df)

        self.close_csv_writers(dataset)
    
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
        # TODO for gpt4
    
    def run(self):
        self.set_recover()
        self.set_llm()
            
        for dataset in DATASETS:
            for election, parties in PARTIES_VAA.items():
                for party in parties:
                    try:
                        if party in PARTIES_TO_REMOVE:
                            continue
                        
                        if self.recovering_data[dataset][election][party]['done']:
                            if self.args.mode_type == 'compute_stance':
                                tmp_topics = glob.glob(f"{data_folder_T2S}/{dataset}/{election}/{party}/*.csv")
                                self.count_done += len(tmp_topics)
                                self.logger.info(f'Done {self.count_done} configurations over the total number.')
                            continue

                        self.run_llm(dataset, election, party)
                        
                        self.done_party_recovering_data(dataset, election, party)
    
                        self.logger.info('Resetting the llm')
                        self.set_llm()
                        #https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api
                    except Exception as e:
                        self.logger.error(f'error: {str(e)}')
                        self.logger.info(f'CURRENT_DATA: {election}, {dataset}, {party}')
                        raise e


def main():
    if not jupyter_:
        parser = argparse.ArgumentParser(description='Python Wrapper Tweepy')
        parser.add_argument('-llm', '--llm_type', type=str, help='llm_type')
        parser.add_argument('-nl', '--num_labels', type=str, help='llm_type')
        args = parser.parse_args()
        if args.llm_type not in MODELS:
            print('llm_type should be either "mixtral" or "gpt4"')
            sys.exit(-1)
        if args.num_labels not in ['5', '3']:
            print('num_labels should be either "5" or "3"')
            sys.exit(-1)
    else:
        class Arguments:
            llm_type: str
            mode_type: str
            def __init__(self):
                self.llm_type = 'mixtral'
                self.mode_type = 'compute_stance'
        
        args = Arguments()

    args.mode_type = 'compute_stance'
    logger = setup_logger('execution_log', f'{log_dir}/log_T2S_filtered_simulate_with_{args.llm_type}_step_{args.mode_type}_{args.num_labels}.log')
    logger.info(f'LLM: {args.llm_type}, mode_type: {args.mode_type}, num_labels: {args.num_labels}')

    try:
        ex = Executor(args, logger)
        ex.run()
        logger.info(f'DONE ALL!!!')
    except Exception:
        logger.exception('EXCEPTION: ')
        sys.exit(-1)


if __name__ == "__main__":
    main()
    