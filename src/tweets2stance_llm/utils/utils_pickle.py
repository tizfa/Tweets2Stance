import pickle
def save_pickle(obj, path, filename):
  with open(path + filename + '.pkl', 'wb') as output:
    pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_pickle(path, filename):
  with open(path + filename + '.pkl', 'rb') as input:
    obj= pickle.load(input)
    return obj