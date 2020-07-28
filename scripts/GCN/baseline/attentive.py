import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cell_line', nargs=1, type=str, help='cell line to run on')
parser.add_argument('--name', nargs=1, type=str, help='name of dataset')
args = parser.parse_args()

cl = args.cell_line[0]
name = args.name[0]

print("%s\t%s\n" % (cl, name))

import pickle
import numpy as np

with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/data_class1_unflattened.pickle' % cl, 'rb') as f:
    data_unflattened = pickle.load(f)

import random
random.seed(30)
idx = list(range(data_unflattened.x.shape[0]))
random.shuffle(idx)
train_mask = idx[:9000]
valid_mask = idx[9000:10000]
test_mask = idx[10000:]

training_data_x = data_unflattened.x[train_mask].long()
training_data_y = data_unflattened.y[train_mask].long()

rows = []
for idx in range(training_data_x.shape[0]):
    for idx2 in range(training_data_x.shape[1]):
        new_row = [idx, idx2+1]
        for x in training_data_x[idx][idx2]:
            new_row.append(x.item())
        new_row.append(training_data_y[idx].item())
        rows.append(new_row)

import csv
with open('DeepChrome/AttentiveChrome-PyTorch/v2PyTorch/data/%s/classification/train.csv' % cl, 'w') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(rows)

valid_data_x = data_unflattened.x[valid_mask].long()
valid_data_y = data_unflattened.y[valid_mask].long()

rows = []
for idx in range(valid_data_x.shape[0]):
    for idx2 in range(valid_data_x.shape[1]):
        new_row = [idx, idx2+1]
        for x in valid_data_x[idx][idx2]:
            new_row.append(x.item())
        new_row.append(valid_data_y[idx].item())
        rows.append(new_row)

import csv
with open('DeepChrome/AttentiveChrome-PyTorch/v2PyTorch/data/%s/classification/valid.csv' % cl, 'w') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(rows)

test_data_x = data_unflattened.x[test_mask].long()
test_data_y = data_unflattened.y[test_mask].long()

rows = []
for idx in range(test_data_x.shape[0]):
    for idx2 in range(test_data_x.shape[1]):
        new_row = [idx, idx2+1]
        for x in test_data_x[idx][idx2]:
            new_row.append(x.item())
        new_row.append(test_data_y[idx].item())
        rows.append(new_row)

import csv
with open('DeepChrome/AttentiveChrome-PyTorch/v2PyTorch/data/%s/classification/test.csv' % cl, 'w') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(rows)
