# Tweets2Stance
Code and dataset for Tweets2Stance.

"From Tweets to Stance: An Unsupervised Framework for User Stance Detection on Twitter": https://dl.acm.org/doi/abs/10.1007/978-3-031-45275-8_7

Cite:

    @inproceedings{gambini2023tweets,
      title={From Tweets to Stance: An Unsupervised Framework for User Stance Detection on Twitter},
      author={Gambini, Margherita and Senette, Caterina and Fagni, Tiziano and Tesconi, Maurizio},
      booktitle={International Conference on Discovery Science},
      pages={96--110},
      year={2023},
      organization={Springer}
    }
 
## Installation

Move to the project folder
`cd path/to/Tweets2Stance`

Create the python virtual environment:

`virtualenv venv --python=python3.10`

Activate the virtual environment:

`source ./venv/bin/activate`

Install libraries:

`pip install -r utils/requirements.txt`

Set up the working directory in 

## Tweets2Stance with ZSC only:
Refer to the src folder `src/tweets2stance_zsc`.

If you wish to use a language model fine-tuned on tweets, the preprocessing occurs in two ways: one that removes hashtags, emojis, and mentions at the beginning of the tweet (suffix=''), and another that retains them (suffix: '_with_@#emoji').

The parameters of the Tweets2Stance framework are:

* Language Model for zero-shot classification
* Dataset (i months before the election)
* Threshold th: threshold to filter tweets depending on the topic
* Algorithm: one of the 4 implemented algorithms

Steps to follow:

* ensure that the dataframe containing various tweets has at least the fields: _created\_at_, _lang_, _tweet\_id_, _screen\_name_, _tweet_ 
* **preprocessing**: 
  * `$nohup python3 src/tweets2stance_zsc/preprocessing.py`
  * output: `f'./data/{election}/{election}_{dataset}_preprocessed{suffix}.pkl'`
* **translation**: 
  * `$nohup python3 src/tweets2stance_zsc/translation.py`
  * starting from the processed tweets, the output is `f'./data/{election}/{election}_{dataset}_preprocessed_translated{suffix}.pkl'`
* **classification\_ZSC**: 
  * `$nohup python3 src/tweets2stance_zsc/ZSC.py -i src/tweets2stance_zsc/inputs/in_ZSC.json >/dev/null 2>&1 &`
  * from the processed and translated tweets, the output is `f'./data/{election}/{model_name.split("/")\[-1\]}/{election}_{dataset}_zsc_tweets{suffix}.pkl'`
  * the method _pretty\_save\_df()_ produces a dataframe suitable for subsequent processing. The output is
  `f'./data/{election}/{model_name.split("/")\[-1\]}/{election}_{dataset}_df_zsc_tweets{suffix}.pkl'`. 
  It is a dataframe in which each row contains classification information for a pair `(tweet, sentence)`. Remember that
each sentence is associated with only one topic. The _created\_at_ field is important, so you can classify only
the tweets from the largest dataset and build the other datasets starting from this one.
* **tweets2stance**: 
  * `$nohup python3 src/tweets2stance_zsc/tweets2stance.py`
  * from the classified tweets, the output is `f'./data/{election}/{model_name.split("/")\[-1\]}/{dataset}/{algorithm}/df_agreements{suffix}.pkl'`


### Notes on executing the ZSC
Activate the virtual environment:

`source ./venv/bin/activate`

Execute Zero-Shot Classification (ZSC):

`$nohup python3 ZSC.py -i src/tweets2stance_zsc/inputs/in_ZSC.json >/dev/null 2>&1 &`

Input file `in_ZSC.json`:

**"begin" field**: for each election, the language models (through a list) to begin classification are indicated. If all models
are still to be used, simply write `"models": "all"`. The field `index_begin_from` is always set to `0`.

**"restart" field**: indicates for which elections and language models it is necessary to restart the classification. In this case,
the field `index_begin_from` indicates the index of the tweet from which to restart. This index can be deduced from how many tweets have
already been classified and saved in the pkl file `f'./data/{election}/{model_name.split("/")\[-1\]}/{election}_{dataset}_zsc_tweets{suffix}'`

    {
      "begin": {
        "GB19": {
          "models": "all",
          "index_begin_from": 0
        },
        "CA19": {
          "models": "all",
          "index_begin_from": 0
        },
        "AB19": {
          "models": "all",
          "index_begin_from": 0
        },
        "AU19": {
          "models": "all",
          "index_begin_from": 0
        },
        "SK20": {
          "models": "all",
          "index_begin_from": 0
        },
        "BC20": {
          "models": "all",
          "index_begin_from": 0
        },
        "CA21": {
          "models": "all",
          "index_begin_from": 0
        },
        "NS21": {
          "models": "all",
          "index_begin_from": 0
        },
        "NFL21": {
          "models": "all",
          "index_begin_from": 0
        }
      },
      "restart": {}
    }

### Notes on Tweet2Stance's output

The output of the `src/tweets2stance_zsc/tweets2stance.py` script is a dataframe (contained in `df_agreements.pkl`) like this:

| screen_name  | sentence                                                                                 | topic                               | avg_agreement | agreement_level | threshold_topic | num_tweets |
|--------------|------------------------------------------------------------------------------------------|-------------------------------------|---------------|-----------------|-----------------|------------|
| albertaNDP   | anti-abortion activists should be able to protest in the immediate vicinity of an abortion clinic | anti-abortion protests close to clinic | 0.425605      | 3               | 0.5             | 25         |
| ...          | ...                                                                                      | ...                                 | ...           | ...             | ...             | ...        |


* The `avg_agreement` is set only for `algorithm_1`
* The `num_tweets` is the number of tweets after the _topic filtering_ step

## Tweets2Stance's steps with LLM
Refer to the src folder `src/tweets2stance_llm`. 

The steps leveraging T2S with ZSC uses the optimal setup: 
* dataset `D3`, 
* algorithm `alg_4 with three tweets minimum`, 
* topic threshold `0.6`, 
* the `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` language model

### T2S Topic Filtering + Agreement Detector w/ Mixtral | GPT4 
1. Divide tweets per topic (topic filtering) using the optimal setting for Tweets2Stance (with ZSC only) using the `topic_filtering_T2S.py script`:
   * `$ nohup python3 src/tweets2stance_llm/topic_filtering_T2S.py`
   * output: tweets saved in a per topic csv file f'./data_for_T2S/{opt_dataset}/{election}/{party}/{topic}.csv'. For example:
   
            data_for_T2S:
                    |_ D3
                        |_ AB19
                            |_ ABLiberal
                                    |_ Alberta carbon tax.csv
   
    Where `Alberta carbon tax.csv` contains tweets with fields `tweet_id`,`user_id` (party's Twitter id),`user_screen_name` (party's Twitter screen name),`tweet` (text)
2. Execute the agreement detector step for both llm (Mixtral and GPT4). The script will retrieve the filtered tweets from the folders created above.
   * `$ nohup python3 src/tweets2stance_llm/topic_filtering_T2S_agreement_detector_LLM.py -llm mixtral&`
   * `$ nohup python3 src/tweets2stance_llm/topic_filtering_T2S_agreement_detector_LLM.py -llm GPT4&`
   * output: `f'./data_for_T2S/{dataset}/results_T2S_filtered_step_stance_detection_with_{llm_type}_{num_labels}_labels.csv'`, where `llm_type` is either `mixtral` or `GPT4`, while `num_labels` is either `5` or `3`

### Topic Filtering and Agreement Detector w/Mixtral | GPT4

1. Execute the topic filtering step for both llm (Mixtral and GPT4):
   * `$ nohup python3 src/tweets2stance_llm/topic_filtering_LLM_agreement_detector_LLM.py -llm mixtral -mode topic_filtering &`
   * `$ nohup python3 src/tweets2stance_llm/topic_filtering_LLM_agreement_detector_LLM.py -llm GPT4 -mode topic_filtering &`
   * output: tweets saved in a per topic csv file `f'./data_for_LLM/{llm_type}/{dataset}/{election}/{party}/{topic}.csv'`. For example:
   
            data_for_LLM:
                |_ GPT4
                    |_ D3
                        |_ AB19
                            |_ ABLiberal
                                    |_ Alberta carbon tax.csv
   
    Where `Alberta carbon tax.csv` contains tweets with fields `tweet_id`,`user_id` (party's Twitter id),`user_screen_name` (party's Twitter screen name),`tweet` (text)
2. Execute the agreement detector step for both llm (Mixtral and GPT4). The script will retrieve the filtered tweets from the folders created above.
   * `$ nohup python3 src/tweets2stance_llm/topic_filtering_LLM_agreement_detector_LLM.py -llm mixtral -mode compute_stance &`
   * `$ nohup python3 src/tweets2stance_llm/topic_filtering_LLM_agreement_detector_LLM.py -llm GPT4 -mode compute_stance &`
   * output: `f'./data_for_LLM/{llm_type}/{dataset}/results_step_stance_detection_{num_labels}_labels.csv'`, where `llm_type` is either `mixtral` or `GPT4`, while `num_labels` is either `5` or `3`

### Topic Filtering w/ Mixtral | GPT4 + T2S Agreement Detector
1. Make sure you have already executed the topic filtering step for both llm (Mixtral and GPT4). See above _Topic Filtering and Agreement Detector w/Mixtral | GPT4_.
2. Execute the agreement detector with the T2S algorithm:
   * `$ nohup python3 src/tweets2stance_llm/topic_filtering_LLM_agreement_detector_T2S.py`
   * output: `f'./data_for_LLM_filter_T2S_stance/{llm_type}/results_LLM_filtering_step_stance_algorithm_4.csv'`, where `llm_type` is either `mixtral` or `GPT4`

### Evaluation
`$ nohup python3 src/tweets2stance_llm/evaluate.py`

This script compares the performance of the original T2S with revised version of T2S by replacing each module (Topic Filtering and Agreement Detector ) with either GPT-4 or Mixtral. 

The output is the csv file `./data_insights/complete_results.csv`