'''
Author: your name
Date: 2021-06-28 23:19:33
LastEditTime: 2021-06-28 23:20:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NMI/test.py
'''
import pandas as pd

df = pd.read_csv("result/table3/R2_of_per_cache.csv")
print(df.score.std())
