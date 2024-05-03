from sklearn.metrics import mean_squared_error as mse_
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from imblearn.metrics import macro_averaged_mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import os.path
import pandas as pd
import seaborn as sns
import copy

from src.utils_pickle import *

CLASSES = [1,2,3,4,5]


def compute_eval_measurements(elections, model_name, dataset, is_baseline=False):
    dict_golden_labels = load_pickle(projectDir + '/dict_golden_labels/', f'dict_golden_labels_VAA_{elections}')
    try:
        df_topic = pd.read_csv(projectDir + f'/translated_topic_and_sentences/translated_topics_and_sentences_{elections}.csv')
    except FileNotFoundError as e:
        # try with pkl
        df_topic = load_pickle(f'{projectDir}/translated_topic_and_sentences/', f'translated_topics_and_sentences_{elections}')
    
    data_path = f'{projectDir}/data/{elections}/{model_name.split("/")[-1]}/{dataset}'
    
    df_results = load_pickle(data_path + '/', f'df_agreements')
    USERS_TO_REMOVE = ['AfcpDavid', 'advantage_party', 'PCPSask', 'GreenPartySK', 'NSGreens']
    field_user = 'party' if 'party' in list(df_results) else 'screen_name'
    df_results = df_results[~df_results[field_user].isin(USERS_TO_REMOVE)]
    df_results.reset_index(drop=True, inplace=True)
    try:
        df_results.rename(columns={'party': 'screen_name', 'answer': 'agreement_level'}, inplace=True)
    except:
        pass
    
    print(f'AGREEMENT_LEVELS: {df_results.agreement_level.value_counts()}')
    
    # add column for 3-valued agreement level
    def map_to_3_labels(agr_lvl):
        if agr_lvl in [1,2]:
            return 1
        elif agr_lvl in [4,5]:
            return 3
        else:
            return 2
    
    df_results['3_agreement_level'] = df_results['agreement_level'].apply(lambda a: map_to_3_labels(a))
    
    # put df_results together with dict_golden_labels
    df_results['golden_label'] = df_results.apply(lambda row: dict_golden_labels[row['screen_name']][row['sentence']], axis=1)
    df_results['3_golden_label'] = df_results['golden_label'].apply(lambda g: map_to_3_labels(g))
    
    print(df_results)
    print(f'5 AGREEMENT LEVELS.')
    df1 = eval_measurements(elections, df_results, data_path, is_baseline)
    
    print(f'3 AGREEMENT LEVELS.')
    prefix_agreement_level_field = '3_'
    eval_measurements(elections, df_results, data_path, is_baseline, prefix_agreement_level_field)
    
    return df1


def compute_ma_mae_mse_r2_scores(y_true, y_pred, prefix_agreement_level_field=''):
    # compute MA_MAE
    if len(y_true) > 0:
      print(f'y_pred before ma_mae: {y_pred}')
      print(f'y_true before ma_mae: {y_true}')

      #ma_mae = macro_averaged_mean_absolute_error(y_true, y_pred)
      sum_mae = 0
      classes = CLASSES
      if prefix_agreement_level_field != '':
        classes = [1,2,3]

      for c in classes:
        indices_y_true_c = [index for index, element in enumerate(np.array(y_true)) if element == c]
        if indices_y_true_c:
          y_pred_c = np.array(y_pred)[indices_y_true_c]
          mae_c = mean_absolute_error([c] * len(indices_y_true_c), y_pred_c)
          sum_mae += mae_c

      ma_mae = sum_mae/len(classes)
    else:
      print(f'FOUND. ma_mae | len y_true: {len(y_true)}')
      ma_mae = None
    print(f'MA_MAE: {ma_mae}')

    # compute MAE
    if len(y_true) > 0:
      mae = mean_absolute_error(y_true, y_pred)
    else:
      print(f'FOUND. mae | len y_true: {len(y_true)}')
      mae = None
    print(f'MAE: {mae}')

    # compute WAE
    if len(y_true) > 0:
      wae = weighted_absolute_error(y_true, y_pred)
    else:
      print(f'FOUND. wae | len y_true: {len(y_true)}')
      wae = None
    print(f'WAE: {wae}')

    # compute MSE
    if len(y_true) > 0:
      mse = mse_(y_true, y_pred)
    else:
      print(f'FOUND. mse | len y_true: {len(y_true)}')
      mse = None
    print(f'MSE: {mse}')

    # compute R2_score
    if len(y_true) > 1:
      r2 = r2_score(y_true, y_pred)
    else:
      print(f'FOUND. r2 | len y_true: {len(y_true)}')
      r2 = None
    print(f'R2_score: {r2}')

    # compute precision, recall, f1
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average='micro')
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average='macro')
    prec_weighted, rec_weighted, f1_weighted,_ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average='weighted')

    # cfx matrix
    cfx_matrix = confusion_matrix(y_true, y_pred, labels=classes)

    return ma_mae, mae, wae, mse, r2, prec_micro, prec_macro, prec_weighted, rec_micro, rec_macro, rec_weighted, f1_micro, f1_macro, f1_weighted, cfx_matrix


def compute_across_dimension(dimension, list_dimension, df_results_t_topic, partial_list_results, prefix_agreement_level_field, parties):
  print(f'across dimension {dimension.upper()}:')
  if not partial_list_results:
    partial_list_results = list()

  ma_mae, mse, r2, num_elems = (None, None, None, None)
  if dimension in ['screen_name', 'sentence']:
    print(f'inside compute_across_dimension for {dimension}')
    for elem in df_results_t_topic[dimension].unique():
      y_true = list(df_results_t_topic[df_results_t_topic[dimension] == elem][f'{prefix_agreement_level_field}golden_label'])
      y_pred = list(df_results_t_topic[df_results_t_topic[dimension] == elem][f'{prefix_agreement_level_field}agreement_level'])
      if not df_results_t_topic[df_results_t_topic[dimension] == elem].empty:
        topic = df_results_t_topic[df_results_t_topic[dimension] == elem].iloc[0]['topic']
      else:
        topic = ''

      print(len(y_true))
      print(elem)
      if 0 in y_true:
        print(f'found y_true 0 in {elem}')
        sys.exit(-1)
      # number of couples that passed the threshold_topic value
      num_elems = len(df_results_t_topic[(df_results_t_topic[dimension] == elem) & (df_results_t_topic['avg_agreement'].notnull())])
      #num_elems = len(y_true)

      try:
        #print('DF_RESULTS_T_TOPIC')
        #for index, row in df_results_t_topic.iterrows():
        #  print(f'screen_name: {row["screen_name"]}, g: {row["golden_label"]}, a: {row["agreement_level"]}, sent: {row["sentence"]}')
        ma_mae, mae, wae, mse, r2, prec_micro, prec_macro, prec_weighted, rec_micro, rec_macro, rec_weighted, f1_micro, f1_macro, f1_weighted, cfx_matrix = compute_ma_mae_mse_r2_scores(y_true, y_pred, prefix_agreement_level_field)
        print(f'obtained ma_mae: {ma_mae}')
      except Exception as e:
        print(str(e))
        print(f'{dimension}, {elem} | golden_label: {y_true} | answer: {y_pred}')
        print(f'{dimension.upper()}: {elem}, mse: {mse}, r2: {r2}')
        sys.exit(-1)
      partial_list_results.append({
          'threshold_topic_value': 'llm',
          'screen_name': elem if dimension == 'screen_name' else 'all',
          'sentence': elem if dimension == 'sentence' else 'all',
          'topic': topic if dimension == 'sentence' else 'all',
          'y_true': y_true,
          'y_pred': y_pred,
          'MA_MAE': ma_mae,
          'MAE': mae,
          'WAE': wae,
          'MSE': mse,
          'R2_score': r2,
          'PRECISION_MICRO': prec_micro,
          'PRECISION_MACRO': prec_macro,
          'PRECISION_WEIGHTED': prec_weighted,
          'RECALL_MICRO': rec_micro,
          'RECALL_MACRO': rec_macro,
          'RECALL_WEIGHTED': rec_weighted,
          'F1_MICRO': f1_micro,
          'F1_MACRO': f1_macro,
          'F1_WEIGHTED': f1_weighted,
          'CFX_MATRIX': cfx_matrix,
          'num_elements': num_elems,
          'num_tweets': sum(df_results_t_topic[df_results_t_topic[dimension] == elem].num_tweets),
          'tot_num_elements': len(df_results_t_topic['sentence'].unique()) * len(parties)
      })
  else:
    print(f'inside compute_across_dimension for ALL dimension')
    # considering ALL results (across topics and parties)
    y_true = list(df_results_t_topic[f'{prefix_agreement_level_field}golden_label'])
    y_pred = list(df_results_t_topic[f'{prefix_agreement_level_field}agreement_level'])

    print(len(y_true))
    print(f'y_true: {y_true}')
    print(f'y_pred: {y_pred}')
    #print('DF_RESULTS_T_TOPIC')
    #for index, row in df_results_t_topic.iterrows():
    #  print(f'screen_name: {row["screen_name"]}, g: {row["golden_label"]}, a: {row["agreement_level"]}, sent: {row["sentence"]}')
    # number of couples that passed the threshold_topic value
    num_elems = len(df_results_t_topic[df_results_t_topic['avg_agreement'].notnull()])
    #num_elems = len(y_true)
    try:
      ma_mae, mae, wae, mse, r2, prec_micro, prec_macro, prec_weighted, rec_micro, rec_macro, rec_weighted, f1_micro, f1_macro, f1_weighted, cfx_matrix = compute_ma_mae_mse_r2_scores(y_true, y_pred, prefix_agreement_level_field)
      print(f'obtained ma_mae: {ma_mae}')
    except Exception as e:
      print(str(e))
      print(f'all, all | {prefix_agreement_level_field}golden_label: {y_true} | {prefix_agreement_level_field}agreement_level: {y_pred}')
      sys.exit(-1)
    partial_list_results.append({
          'threshold_topic_value': 'llm',
          'screen_name': 'all',
          'sentence': 'all',
          'topic': 'all',
          'y_true': y_true,
          'y_pred': y_pred,
          'MA_MAE': ma_mae,
          'MAE': mae,
          'WAE': wae,
          'MSE': mse,
          'R2_score': r2,
          'PRECISION_MICRO': prec_micro,
          'PRECISION_MACRO': prec_macro,
          'PRECISION_WEIGHTED': prec_weighted,
          'RECALL_MICRO': rec_micro,
          'RECALL_MACRO': rec_macro,
          'RECALL_WEIGHTED': rec_weighted,
          'F1_MICRO': f1_micro,
          'F1_MACRO': f1_macro,
          'F1_WEIGHTED': f1_weighted,
          'CFX_MATRIX': cfx_matrix,
          'num_elements': num_elems,
          'num_tweets': sum(df_results_t_topic.num_tweets),
          'tot_num_elements': len(df_results_t_topic['sentence'].unique()) * len(parties)
      })

  print(f'partial_list_results len: {len(partial_list_results)}')
  print('===================')
  return partial_list_results, mse, r2, num_elems


def eval_measurements(elections, df_results, final_path, is_baseline, prefix_agreement_level_field=''):
    # append to dataframe where all results are saved
    print('Evaluating mse, r2_score, plots.')
    print(f'num_topics: {len(df_results["sentence"].unique())}')
    parties = list(df_results['screen_name'].unique())
    print(f'num parties: {len(parties)}')
    
    partial_list_results = list()
    
    #df_results_t_topic = df_results[df_results['threshold_topic'] == t_topic]
    # all
    #dimension, list_dimension, df_results_t_topic, partial_list_results, threshold_topic_value
    print('all')
    partial_list_results, mse, r2, num_elems = compute_across_dimension('all', None, df_results, partial_list_results, prefix_agreement_level_field, parties)
    
    # across parties:
    print('across parties')
    partial_list_results, _, _, _ = compute_across_dimension('screen_name', parties, df_results, partial_list_results, prefix_agreement_level_field, parties)
    
    # across topics:
    print('across topics')
    partial_list_results, _, _, _ = compute_across_dimension('sentence', list(df_results['sentence'].unique()), df_results, partial_list_results, prefix_agreement_level_field, parties)
    
    # create final dataframe:
    df_eval_results = pd.DataFrame(partial_list_results)
    #final_path = f'{data_path_folder}/{model.split("/")[-1]}/{dataset}/{algorithm}'
    print(f'saving pickle of df_eval_results: {df_eval_results}')
    filename = f'{prefix_agreement_level_field}df_eval_measurements'
    print(f'saving it to {final_path}/{filename}')
    save_pickle(df_eval_results, final_path + '/', filename)
    print(f'done.')
    return df_eval_results
