'''
Esempio di calcolo degli score per topic e sentence per un tweet.
Il classificatore può essere anche applicato a batch di tweet, ma mi sembra che funzioni più lentamente rispetto ad applicarlo tweet per tweet...
'''

from transformers import pipeline
import pandas as pd
import src.tweets2stance_zsc.conf as INST

# device = 0 -> stai dicendo che il modello lavorerà sulla GPU
# https://huggingface.co/transformers/master/main_classes/pipelines.html#transformers.ZeroShotClassificationPipeline
# con 'pipeline' stai dicendo 'crea un dataflow che utilizza quel modello come classificatore'
def get_classifier(model_name):
    print(f'Getting classifier: {model_name}')
    classifier = pipeline("zero-shot-classification", model=model_name, tokenizer=model_name, device=0)
    print(f'Got classifier: {model_name}')
    return classifier


df_topic = pd.read_csv(f'{INST.TOPICS_SENTENCES_FOLDER}/translated_topics_and_sentences_IT19.csv').dropna()
topics = list(df_topic['topic'])
sentences = list(df_topic['sentences'])

model_name = 'facebook/bart-large-mnli'
# per il mommento usiamo solo questo modello
classifier = get_classifier(model_name)

# for topics, we use the template: template = "This text is about {}."
template = "This text is about {}."
tweet = 'Example tweet'
out = classifier(tweet, topics, hypothesis_template=template, multi_label=True)

# for sentences, we don't use the template.
out_s = classifier(tweet, sentences, hypothesis_template="{}", multi_label=True)

'''
        example of 'out' in general:
        { 'labels': ['conspiracy', 'not conspiracy'], -> the topics or sentences
          'scores': [0.992976188659668, 0.0070238420739769936],
          'sequence': 'Covid Crusade: Franklin Graham Preaches Vaccine Gospel for Deep State'}
        '''

# the input of next step (the real tweets2stance framework) is a dataframe where each row represents the scores for a triplet (tweet, topic, sentence).
# columns are: 'created_at' (creation timestamp for the tweet), 'tweet_id' (in stringa), 'tweet' (il testo del tweet), 'user_id': (in stringa), 'screen_name', 'topic', 'score_topic', 'sentence', 'score_sentence'
# as long as you stick with this representation, you can get the classification scores in the most suitable way for you ;)