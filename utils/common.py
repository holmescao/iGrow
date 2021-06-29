'''
Author: your name
Date: 2021-06-12 23:38:45
LastEditTime: 2021-06-12 23:39:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NMI/utils/common.py
'''
import pickle
import os


def mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)


def save_curve(obj, save_path):
    with open(save_path, 'w'):
        pickle.dumps(obj)
