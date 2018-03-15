'''
This script does a exploratory analysis of the following problem:
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

This Project is only for academic proposes and it was made
by Normando Zubia, college professor of Universidad La Salle in
Chihuahua.

Bibliography and References are going to be upload in the future
'''

import pandas
import numpy
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

# Count number of rows
# print('# Line count:')
# for file in ['train.csv']:
#     lines = subprocess.run(['wc', '-l', '../data/{}'.format(file)], stdout=subprocess.PIPE).stdout.decode('utf-8')
#     print(lines, end='', flush=True)

'''
Result:
train.csv -> 184903891
'''
# ##################################################################
#                     Initial Exploratory analysis
# ##################################################################
# TODO: Redefine data types

#  Check types and shape
# data = pandas.read_csv("../data/train.csv", nrows=1000)
# print(data.dtypes)

'''
Features: 7
ip                  int64
app                 int64
device              int64
os                  int64
channel             int64
click_time         object
attributed_time    object

Class:
is_attributed       int64
'''

# General information from dataset
data = pandas.read_csv("../data/train.csv", nrows=10000000)

# print(data.head())

# Convert to correct types
variables = ['ip', 'app', 'device', 'os', 'channel']
for v in variables:
    data[v] = data[v].astype('category')

# set click_time and attributed_time as timeseries
data['click_time'] = pandas.to_datetime(data['click_time'])
data['attributed_time'] = pandas.to_datetime(data['attributed_time'])

# set as_attributed in train as a categorical
data['is_attributed'] = data['is_attributed'].astype('category')

# print(data.describe())

# Graphic about unique values per feature

# plt.figure(figsize=(15, 8))
# cols = ['ip', 'app', 'device', 'os', 'channel']
# uniques = [len(data[col].unique()) for col in cols]
# sns.set(font_scale=1.2)
# ax = sns.barplot(cols, uniques, palette=sns.color_palette(), log=True)
# ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
# for p, uniq in zip(ax.patches, uniques):
#     height = p.get_height()
#     ax.text(p.get_x()+p.get_width()/2.,
#             height + 10,
#             uniq,
#             ha="center")
# plt.show()

# Check class balance
# print(data['is_attributed'].value_counts())

# Graphic about class balance

# plt.figure(figsize=(6,6))
# mean = (data.is_attributed.values == 1).mean()
# ax = sns.barplot(['App Downloaded (1)', 'Not Downloaded (0)'], [mean, 1-mean])
# ax.set(ylabel='Proportion', title='App Downloaded vs Not Downloaded')
# for p, uniq in zip(ax.patches, [mean, 1-mean]):
#     height = p.get_height()
#     ax.text(p.get_x()+p.get_width()/2.,
#             height+0.01,
#             '{}%'.format(round(uniq * 100, 2)),
#             ha="center")
# plt.show()
