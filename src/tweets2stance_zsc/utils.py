import pickle
import logging


def save_pickle(obj, path, filename):
  with open(path + filename + '.pkl', 'wb') as output:
    pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_pickle(path, filename):
  with open(path + filename + '.pkl', 'rb') as input:
    obj= pickle.load(input)
    return obj


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    l.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)
    l.addHandler(handler)