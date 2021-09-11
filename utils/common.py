import pickle
import os


def mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)


def save_curve(obj, save_path):
    with open(save_path, 'wb') as fo:
        pickle.dump(obj, fo)


def load_curve(save_path):
    with open(save_path, 'rb') as fo:
        curve = pickle.load(fo, encoding='bytes')
    return curve
