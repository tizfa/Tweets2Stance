import sys
from sklearn.metrics import f1_score, mean_absolute_error
from statistics import mean

sys.path.append('..')
from src.tweets2stance_llm.utils.config import *
from src.tweets2stance_llm.utils.utils_load import *
from src.tweets2stance_llm.utils.utils_pickle import *
from src.tweets2stance_llm.utils.utils import *

MODES = ['2step', 'T2S_filtered', 'LLM_filtered_algorithm4']
NUM_LABELS = [5, 3]
MAP_METHOD = {
    '2step_mixtral': 'T2S with Mixtral',
    '2step_GPT4': 'T2S with GPT4',
    'T2S_filtered_GPT4': 'T2S filtered + stance with GPT4',
    'T2S_filtered_mixtral': 'T2S filtered + stance with mixtral',
    'LLM_filtered_algorithm4_mixtral': 'mixtral filtered + T2S stance detector',
    'LLM_filtered_algorithm4_GPT4': 'GPT4 filtered + T2S stance detector',
}


def mapping_three_labels_derived(label):
    if label in [1,2]:
        return 1
    elif label == 3:
        return 2
    else:
        return 3


def mapping_three_labels_computed(label):
    if label == 4:
        return 3
    elif label == 3:
        return 2
    elif label == 2:
        return 1


def get_df_ground_of_truth():
    # Display the resulting DataFrame
    df_got = pd.DataFrame()
    for election, parties in PARTIES_VAA.items():
        print(election)
        for party in parties:
            if party in PARTIES_TO_REMOVE:
                continue
            golden_labels = get_golden_labels(election, party)
    
            # Create DataFrame from the dictionary
            tmp_df_got = pd.DataFrame.from_dict(golden_labels, orient='index', columns=['golden_label']).reset_index()
            tmp_df_got.columns = ['statement', 'golden_label']
            tmp_df_got['VAA'] = [election] * tmp_df_got.shape[0]
            tmp_df_got['Party'] = [party] * tmp_df_got.shape[0]
            df_got = pd.concat([df_got, tmp_df_got])
    
    df_got['golden_label_three'] = df_got['golden_label'].apply(lambda label: mapping_three_labels_derived(label))
    return df_got


def get_table_results():
    dataset = OPTIMAL_SETTINGS_T2S_ZSC['dataset']
    df_got = get_df_ground_of_truth()
    results = [
        # the results for T2S with ZSC only for best settings
        {
            'Method': 'T2S',
            'avg_F1_l5': 0.29,
            'avg_F1_l3_der': 0.53,
            'avg_F1_l3_cal': '--',
            'avg_MAE_l5': 1.56,
            'avg_MAE_l3_der': 0.85
        }
    ]
    for mode in MODES:
        for llm_name in MODELS:

            tmp_data = {
                'Method': MAP_METHOD[f'{mode}_{llm_name}'],
                'avg_F1_l5': None,
                'avg_F1_l3_der': None,
                'avg_F1_l3_cal': None,
                'avg_MAE_l5': None,
                'avg_MAE_l3_der': None
            }
            for num_labels in NUM_LABELS:
                if mode == 'LLM_filtered_algorithm4' and num_labels == 3:
                    continue

                if mode == '2step':
                    if llm_name == 'mixtral':
                        filename = f'{data_folder_LLM}/{llm_name}/{dataset}/prompt1_results/results_step_stance_detection_{num_labels}_labels{suffix}.csv'
                    else:
                        filename = f'{data_folder_LLM}/{llm_name}/{dataset}/results_step_stance_detection_{num_labels}_labels{suffix}.csv'
                elif mode == 'T2S_filtered':
                    # T2S_filtered_step_stance_detection_with_GPT4
                    filename = f'{data_folder_T2S}/{dataset}/results_{mode}_step_stance_detection_with_{llm_name}_{num_labels}_labels.csv'
                else:
                    # 'LLM_filtered_algorithm4'
                    filename = f'{data_folder_LLM_filter_T2S_stance}/{llm_name}/results_LLM_filtering_step_stance_algorithm_4.csv'

                print(f'filename: {filename}')
                df_results = pd.read_csv(filename)
                df_results.drop_duplicates(inplace=True)
                if num_labels == 3:
                    df_results['llm'] = df_results['Stance'].apply(lambda label: mapping_three_labels_computed(label))
                else:
                    df_results.rename(columns={'Stance': 'llm'}, inplace=True)
                    df_results['llm_three'] = df_results['llm'].apply(lambda label: mapping_three_labels_derived(label))

                df_merged = pd.merge(
                    df_results,
                    df_got,
                    how='inner',
                    left_on=['VAA', 'Party', 'Sentence'],
                    right_on=['VAA', 'Party', 'statement']
                )

                # You can drop the duplicate column ('sentence') if needed
                # df_merged = df_merged.drop(columns=['sentence', 'statement_y'])[['screen_name', 'statement_x', 'agreement_level', 'stance', 'golden_label']]
                # df_merged.rename(columns={'agreement_level': 'tweets2stance', 'stance': 'llm', 'statement_x': 'statement'}, inplace=True)

                # Display the merged DataFrame
                # print(df_merged)

                # Compute evaluation scores
                f1_scores = list()
                f1_scores_three_derived = list()
                mae_scores = list()
                mae_scores_three_derived = list()
                for election in df_merged.VAA.unique():
                    tmp = df_merged[df_merged['VAA'] == election]
                    tmp['llm'] = tmp['llm'].astype(int)
                    tmp['golden_label_three'] = tmp['golden_label_three'].astype(int)
                    tmp['golden_label'] = tmp['golden_label'].astype(int)
                    f1_llm_three_derived = None
                    mae_llm_three_derived = None
                    if num_labels == 3:
                        f1_llm = f1_score(tmp['golden_label_three'].values, tmp['llm'].values, average='weighted')
                        mae_llm = mean_absolute_error(tmp['golden_label_three'].values, tmp['llm'].values)
                    else:
                        f1_llm = f1_score(tmp['golden_label'].values, tmp['llm'].values, average='weighted')
                        mae_llm = mean_absolute_error(tmp['golden_label'].values, tmp['llm'].values)

                        # derived
                        f1_llm_three_derived = f1_score(tmp['golden_label_three'].values, tmp['llm_three'].values,
                                                        average='weighted')
                        mae_llm_three_derived = mean_absolute_error(tmp['golden_label_three'].values,
                                                                    tmp['llm_three'].values)

                    # Display the F1 score
                    print(
                        f"({llm_name}, {num_labels} labels) VAA {election}, F1 (WEIGHTED): {f1_llm}, MAE: {mae_llm} | F1 three derived: {f1_llm_three_derived}, MAE three derived: {mae_llm_three_derived}")
                    f1_scores.append(f1_llm)
                    mae_scores.append(mae_llm)
                    f1_scores_three_derived.append(f1_llm_three_derived)
                    mae_scores_three_derived.append(mae_llm_three_derived)

                f1_avg_all = mean(f1_scores)
                mae_avg_all = mean(mae_scores)

                if num_labels == 5:
                    f1_avg_all_three_derived = mean(f1_scores_three_derived)
                    mae_avg_all_three_derived = mean(mae_scores_three_derived)

                print(f"({llm_name}, {num_labels} labels), AVG_F1 (WEIGHTED): {f1_avg_all}, AVG_MAE: {mae_avg_all}")
                if num_labels == 5:
                    tmp_data['avg_F1_l5'] = f1_avg_all
                    tmp_data['avg_MAE_l5'] = mae_avg_all
                    tmp_data['avg_F1_l3_der'] = f1_avg_all_three_derived
                    tmp_data['avg_MAE_l3_der'] = mae_avg_all_three_derived
                else:
                    tmp_data['avg_F1_l3_cal'] = f1_avg_all
                    tmp_data['avg_MAE_l3_cal'] = mae_avg_all

            results.append(tmp_data)
            df = pd.DataFrame(results)
            if not os.path.exists(data_folder_insights):
                os.makedirs(data_folder_insights)

            df.to_csv(f'{data_folder_insights}/complete_results.csv', index=False)
            print('DONE!!!')


if __name__ == '__main__':
    get_table_results()
