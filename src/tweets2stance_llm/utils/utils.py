import logging
import os
import re
from src.tweets2stance_llm.utils.config import log_dir

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def setup_logger(logger_name, log_file, level=logging.INFO, scrape_profile=True):
    l = logging.getLogger(logger_name)
    l.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)
    l.addHandler(handler)

    return logging.getLogger(logger_name)

def get_prompt_topic_filtering(statements,data_str):
    prompt = ''
    prompt += 'Write again the topics I\'m providing you:\n\n'
    prompt += '`"topics"`: [\n'
    #prompt += f'TEXT_TO_ANALYZE\n"{data_str}"\n\n'
    #prompt += 'TOPICS\n'
    last_index = len(statements) - 1
    for index, s in enumerate(statements):
        prompt += f'    "{s.lower()}"'
        if index != last_index:
            prompt += ',\n'
    prompt += '\n]\n'
    prompt += '\nThen, write the last topic in the list.\n\n'
    prompt += 'Now you are a helpful AI topic labeller, capable of processing a text `"text_to_analyze"` and the `"topics"` above. Your job is to semantically assign zero, one or more topics from `"topics"` to a text `"text_to_analyze"`. You must meet ALL these requirements ONLY:\n'
    prompt += '- You must always respond in JSON format containing 2 fields, "text" and "selected_topics". "text" is the analyzed text, "selected_topics" is the set of topics from `"topics"` which are correlated with the `"text_to_analyze"`.\n'
    prompt += '- Your output must include only the JSON data. Do not add any explanation also in case no matching topics can be found.\n'
    prompt += '- You must only assign topics listed on `"topics"`, exactly as specified in the list. \n'
    prompt += '- You mustn\'t assign topics mentioned in the text but not in the `"topics"` list! For example, if the text mentions a topic <topic> which you cannot find in the `"topics"` list, you must not write it in "selected_topics".\n'
    prompt += '- If a topic is not explicitly listed but is implied in the `"text_to_analyze"`, you don\'t put it inside the "selected_topics" list.\n'
    prompt += '- You are capable. You can reason step by step, check that every requirement is met and therefore answer correctly. \n\n'
    
    prompt += '```\n'
    prompt += f'"text_to_analyze": "{data_str}"\n```\n'
    return prompt

def get_prompt_compute_stance(tweets_str, sentence, num_labels=5):
    # prompt1
    labels_str = "'completely disagree', 'disagree', 'neither disagree nor agree', 'agree', and 'completely agree'" if num_labels == 5 else "'disagree', 'neither disagree nor agree', and 'agree'"
    num_labels_word = 'five' if num_labels == 5 else 'three'
    p = '''Tweets by an unknown Twitter account are submitted in a block delimited by '=== TWEETS BEGIN ===' and '=== TWEETS END ==='. Each tweet is separated by the '~|~' separator. Tweets may include emojis in unicode (e.g, 🙏). Also, tweets could be in a language other than English (e.g., Italian).\n\n'''
    p += f'=== TWEETS BEGIN ===\n{tweets_str}\n=== TWEETS END ===\n\n'
    p += f"Your job is to assess the stance of the Twitter account in response to the statement \"{sentence}\" leveraging what the account wrote (consider all tweets you've just read). The stance can assume only these {num_labels_word} values: {labels_str}. The 'neither disagree nor agree' case refers to cases where there are tweets about the topic but the stance is neutral.\n\n"
    p += 'The stance for the statement must be in JSON format like this:\n\n'
    p += '{\n'
    p += '\t"stance": <the stance>,\n'
    p += '\t"statement": <the_statement>\n'
    p += '}\n\n'
    p += 'In your analysis, rely exclusively on the information present in the tweets, possibly drawing conclusions only based on what has been shown to you. Do not use any external information to draw your conclusions.\n'
    return p


def get_tokens(tweets_df, llm_model, llm_name, sentence=None, num_labels=5):
    tweets_str = '~|~'.join(tweets_df.tweet.values)

    p = get_prompt_compute_stance(tweets_str, sentence, num_labels)

    tokens = 0
    if llm_name == 'mixtral':
        tokens = llm_model._model.tokenize(p.encode("utf-8"), add_bos=True, special=True)
    elif llm_name == 'GPT4':
        tokens = llm_model.encode(p)
    return tokens

# Clean for stance detection
def handle_white_spaces(tweets):
        tweets = [re.sub('\s{2,}', ' ', tweet, flags=re.MULTILINE) for tweet in tweets]
        tweets = [re.sub('^\s{1,}', '', tweet, flags=re.MULTILINE) for tweet in tweets]
        tweets = [tweet.strip() for tweet in tweets]
        return tweets

# NOT USED ANYMORE
def clean_for_stance_detection(tweets_df):
    filtered = [re.sub(r'https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE).replace('\n', ' ').replace('\r', ' ') for tweet in list(tweets_df['tweet'])]
    filtered = handle_white_spaces(filtered)
    tweets_df['tweet'] = filtered
    return tweets_df