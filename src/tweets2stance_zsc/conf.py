DATA_PATH_FOLDER = './data'
TOPICS_SENTENCES_FOLDER = f'{DATA_PATH_FOLDER}/sentences_topics'

# best config for first work (tweets2stance)
OPT_MODEL_NAME = "facebook/bart-large-mnli"
OPT_ALGORITHM = 'algorithm_3'
OPT_THRESHOLD_TOPIC = 0.6
OPT_DATASET = 'D4'

MODELS = {}
ELECTIONS_LIST = ['GB19', 'CA19', 'AB19', 'AU19', 'SK20', 'BC20', 'CA21', 'NS21', 'NFL21']
for el in ELECTIONS_LIST:
    MODELS[el] = ["facebook/bart-large-mnli", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "digitalepidemiologylab/covid-twitter-bert-v2-mnli"]

ALGORITHMS = ["algorithm_1", "algorithm_2", "algorithm_3", "algorithm_4_min_num_tweets_3", "algorithm_4_min_num_tweets_2"]
DATASETS = ["D3", "D4", "D5", "D7"] #Di: i=#di mesi
THRESHOLD_TOPICS = [0.5, 0.6, 0.7, 0.8, 0.9]
SUFFIXES = ['', '_with_@#emoji']

# for pre-processing
MIN_NUM_WORDS = 4