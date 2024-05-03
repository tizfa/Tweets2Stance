'''
Main for topic filtering and stance detection through LLMs
'''

import argparse

from src.config import *
from src.utils import *

class Executor:
    def __init__(self, args, logger):
        self.jupyter_ = True
        self.recovering_file_path = f'{log_dir}/recovering'
        self.logger = logger
        self.args = args
        self.recovering_data = {}
        self.llm = None
        self.summed_seconds = 0
        self.how_many = 0
        self.csv_writers = {}
    
    def set_recover(self):
        self.recovering_file_path += f'_{self.args.llm_type}_step_{self.args.mode_type}.json'
        if os.path.exists(self.recovering_file_path):
            # Open the existing file
            with open(self.recovering_file_path, 'r') as file:
                self.recovering_data = json.load(file)
                self.logger.info('Got recovering data from file')
        else:
            self.recovering_data = {
                # <dataset> -> <election> -> <user> -> tweets e flag done -> dict of tweets
            }
            for d in DATASETS:
                self.recovering_data[d] = {}
                for e, parties in PARTIES_VAA.items():
                    self.recovering_data[d][e] = {}
                    for party in parties:
                        self.recovering_data[d][e][party] = {
                            'tweets': {},
                            'done': False
                        }
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
                filepath_csv = f'{data_folder_LLM}/{dataset}/{election}/{party}/{topic}.csv'
                write_header = False
                if not os.path.exists(filepath_csv):
                    write_header = True

                f = open(filepath_csv, mode= 'a')
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if write_header:
                    csv_writer.writerow(['tweet_id', 'user_id', 'user_screen_name', 'tweet'])
                    
                self.csv_writers[dataset][election][party][topic] = {
                    'csv_writer': csv_writer,
                    'file': f
                }
        else:
            self.logger.info('Setting csv writers for stance still not implemented')

    def close_csv_writers(self, statements, dataset, election, party):
        if self.args.mode_type == 'topic_filtering':
            self.logger.info('Closing files and csv writers for topic filtering')
            for topic in statements:
                self.csv_writers[dataset][election][party][topic]['file'].close()
        else:
            self.logger.info('Closing files and csv writers for stance still not implemented')


    def add_tweet_recovering_data(self, dataset, election, party, tweet_id):
        self.recovering_data[dataset][election][party]['tweets'][tweet_id] = True
        with open(self.recovering_file_path, 'w') as file:
                json.dump(self.recovering_data, file)

    def done_party_recovering_data(self, dataset, election, party):
        self.recovering_data[dataset][election][party]['done'] = True
        with open(self.recovering_file_path, 'w') as file:
                json.dump(self.recovering_data, file)

    def get_initial_prompt(self, statements):
        prompt = ''
        if self.args.mode_type == 'topic_filtering':
            prompt += 'Given the following list of topics:\n'
            for index, s in enumerate(statements):
                prompt += f'{index}) {s}\n'
        prompt += '\nWith "<integer>)" I provided the index for each topic.\n\nProvide as output a Python list of indices of topics (e.g., [0,19,2,5]) that best matches the following tweet. In case no topic matches, provide an empty Python list.\n\nThe tweet is: '
        self.logger.info(f'Initial prompt: {prompt}')
        return prompt

    def process_topic_filtering(self, tweets, dataset, election, party, topics):
        # row: pandas Series
        for index, row in tweets.iterrows():
            #tweets_str = '~|~'.join(tweets.tweet.values)
            if self.recovering_data[dataset][election][party]['tweets'].get(row['tweet_id']):
                self.logger.info(f'{election}, {dataset}, {party} ALREADY DONE. Going on.')
                continue

            data_str = row['tweet']
            
            prompt = self.get_initial_prompt(topics)
            prompt += data_str

            self.logger.info('Starting LLM!!!')
            start_time = time.time()
            # Simple inference example
            output = self.llm(
                f"[INST] {prompt} [/INST]", # Prompt
                max_tokens=1024,  # Generate up to 512 tokens
                stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
                echo=False,       # Whether to echo the prompt
                seed=42
            )
            elapsed_time = time.time() - start_time

            # Print the elapsed time in seconds
            self.summed_seconds += elapsed_time
            self.how_many += 1
            self.logger.info(f"Elapsed time: {elapsed_time} seconds. AVG: {self.summed_seconds/self.how_many}")
            reply = output['choices'][0]['text']
            regex = '\[( ?\d+ ?,)+ ?\d+ ?\]|\[ *\]'
            
            indices_topics = list()
            m = re.search(regex, reply)
            try:
                indices_in_topic_tweets = ast.literal_eval(m.group())
            except Exception as e:
                self.logger.error(f'error in regex or literal_eval: {str(e)}')
                self.logger.info(f'Prompt: {prompt}')
                self.logger.info(f'Reply: {reply}')
                raise e
                
            # TODO: get topic from indices_topics
            self.logger.info(f'Getting matching topics from {indices_in_topic_tweets}')
            matching_topics = np.take(l, indices_in_topic_tweets)
            self.logger.info(f'Matching topics: {matching_topics}')

            self.logger.info('Saving tweet in topic file...')
            for t in matching_topics:
                for index, row in tweets.iterrows():
                    self.csv_writers[dataset][election][party][t]['csv_writer'].writerow([row['tweet_id'], row['user_id'], row['user_screen_name'], row['tweet']])
                    self.logger.info(f'wrote tweet to {dataset}/{election}/{party}/{t}.csv')


    def process_stance(self, tweets, dataset, election, party, statements):
        pass
    
    def run_llm(self, tweets, dataset, election, party):
        field = 'topic' if self.args.mode_type == 'topic_filtering' else 'sentence'
        statements = load_sentences(election)[field].values

        self.set_csv_writers(statements, dataset, election, party)

        if self.args.mode_type == 'topic_filtering':
            dir_filtering = f'{data_folder_LLM}/{dataset}/{election}/{party}'
            if not os.path.exists(dir_filtering):
                os.makedirs(dir_filtering)

        if self.args.mode_type == 'topic_filtering':
            self.process_topic_filtering(tweets, dataset, election, party, statements)
        else:
            self.process_stance(tweets, dataset, election, party, statements)

        self.close_csv_writers(statements, dataset, election, party)
    
    def set_llm(self):
        if self.args.llm_type == 'mixtral':
            from llama_cpp import Llama
            # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    
            self.llm = Llama(
                model_path="/workspace/models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",  # Download the model file first
                n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
                n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
                n_gpu_layers=35,         # The number of layers to offload to GPU, if you have GPU acceleration available
                seed=42
            )
        # TODO for gpt4
    
    def run(self):
        self.set_recover()
    
        self.set_llm()
    
        if not self.llm:
            self.logger.error('llm is None!!!')
            sys.exit(-1)
            
        for dataset in DATASETS:
            for election, parties in PARTIES_VAA.items():
                for party in parties:
                    try:
                        if self.recovering_data[dataset][election][party]['done']:
                            self.logger.info(f'{election}, {dataset}, {party} ALREADY DONE. Going on.')
                            continue
                            
                        tweets = load_raw_vaa(election, dataset)
                        tweets = tweet[tweets['user_screen_name'] == party]
                        self.logger.info(f'tweets ({election}, {dataset}, {party}): {tweets.shape}')
                        
                        self.logger.info('cleaning tweets...')
                        tweets = clean(tweets)
                        
                        self.run_llm(tweets, dataset, election, party)
                        
                        with open(self.recovering_file_path, 'w') as file:
                            json.dump(self.recovering_data, file)
    
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
        parser.add_argument('-mode', '--mode_type', type=str, help='mode_type')
        args = parser.parse_args()
        if args.llm_type not in MODELS:
            print('llm_type should be either "mixtral" or "gpt4"')
            sys.exit(-1)
        if args.mode_type not in ['topic_filtering', 'compute_stance']:
            print('mode_type should be either "topic_filtering" or "compute_stance"')
            sys.exit(-1)
    else:
        class Arguments:
            llm_type: str
            mode_type: str
            def __init__(self):
                self.llm_type = 'mixtral'
                self.mode_type = 'topic_filtering'
        
        args = Arguments()
        
    logger = setup_logger('execution_log', f'{log_dir}/log_simulate_with_{args.llm_type}_step_{args.mode_type}.log')
    logger.info(f'LLM: {args.llm_type}, mode_type: {args.mode_type}')

    try:
        ex = Executor(args, logger)
        ex.run()
    except Exception:
        logger.exception('EXCEPTION: ')
        sys.exit(-1)

if __name__ == "__main__":
    main()
    