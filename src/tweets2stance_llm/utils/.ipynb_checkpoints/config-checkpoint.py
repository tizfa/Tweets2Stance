projectDir = '/workspace/workspace/stance_detection'
data_folder = f'{projectDir}/data'
data_folder_LLM = f'{projectDir}/data_for_LLM'
data_folder_insights = f'{projectDir}/data_insights'
data_folder_T2S = f'{projectDir}/data_for_T2S'
data_folder_LLM_filter_T2S_stance = f'{projectDir}/data_for_LLM_filter_T2S_stance'
log_dir = f'{projectDir}/logs'

PARTIES_VAA = {
    'CA19': ['liberal_party', 'CanadianGreens', 'NDP', 'BlocQuebecois', 'CPC_HQ', 'peoplespca'],
    'AB19': ['GreenPartyAB', 'ABLiberal', 'AlbertaParty', 'AfcpDavid', 'Alberta_UCP', 'albertaNDP', 'advantage_party'],
    'AU19': ['LNPQLD', 'AustralianLabor', 'Greens'],
    'SK20': ['SaskParty', 'PCPSask', 'Sask_NDP', 'GreenPartySK'],
    'BC20': ['BCGreens', 'bcndp', 'LPCBC'],
    'CA21': ['liberal_party', 'CanadianGreens', 'NDP', 'BlocQuebecois', 'CPC_HQ', 'peoplespca'],
    'NS21': ['LiberalPartyNS', 'nspc', 'NSNDP', 'NSGreens'],
    'NFL21': ['nlliberals', 'PCPartyNL', 'NLNDP'],
    ##'IT19': ['pdnetwork', 'forza_italia', 'FratellidItalia', 'LegaSalvini', 'Piu_Europa', 'Mov5Stelle'],
    'GB19': ['LibDems', 'TheGreenParty', 'UKLabour', 'Conservatives', 'reformparty_uk']
}

PARTIES_TO_REMOVE = ['AfcpDavid', 'advantage_party', 'PCPSask', 'GreenPartySK', 'NSGreens']

# D3 best F1 for both 5 labels and 3 labels
DATASETS = ['D3']#, 'D4', 'D5', 'D7']
MODELS = ['mixtral', 'GPT4']

MAP_STANCE_TO_INT = {
    'completely disagree': 1,
    'disagree': 2,
    'neither disagree nor agree': 3,
    'agree': 4, 
    'completely agree': 5
}

CONTEXT_LENGTH = {
    'mixtral': 32300,#32768, ma noi usiamo 32300 per tenere conto anche dell'output
    'GPT4': 127488 #128000, ma noi usiamo 127532 per tenere conto anche dell'output 
}
