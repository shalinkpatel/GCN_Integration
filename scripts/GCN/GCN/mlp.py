import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cell_line', nargs=1, type=str, help='cell line to run on')
parser.add_argument('--name', nargs=1, type=str, help='name of dataset')
args = parser.parse_args()

cl = args.cell_line[0]
name = args.name[0]

print("%s\t%s\n" % (cl, name))

import dgl
import networkx as nx
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hypertunity as ht
from sklearn import preprocessing

with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/data_embedding.pickle' % cl, 'rb') as f:
    data_embedding = pickle.load(f)

with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/data_class1.pickle' % cl, 'rb') as f:
    data = pickle.load(f)

X = data_embedding
y = torch.tensor(list(map(lambda x: x[0], data.y)), dtype=torch.long)

model = nn.Sequential(
    nn.Linear(94, 1000),
    nn.LeakyReLU(),
    nn.Dropout(),
    nn.Linear(1000, 500),
    nn.LeakyReLU(),
    nn.Dropout(),
    nn.Linear(500, 100),
    nn.LeakyReLU(),
    nn.Dropout(),
    nn.Linear(100, 50),
    nn.LeakyReLU(),
    nn.Dropout(),
    nn.Linear(50, 2),
    nn.Sigmoid()
)

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

device = torch.device('cuda')
def train_model(net, X, y, epochs, learning_rate, train_mask, test_mask):
    model = net.to(device)
    X = X.to(device)
    y = y.to(device)
    samples = len(X)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    losses_train = []
    losses_test = []
    best_auc = -1
    correct_pred = [y[i].item() for i in test_mask]

    weight_one = sum(y.cpu().numpy().tolist())/samples
    weight = torch.tensor([weight_one, 1-weight_one]).to(device)

    #pbar = tqdm(range(epochs))
    for epoch in range(epochs):
        model.train()
        logits = model(X.float())

        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=weight)
        loss_test = F.cross_entropy(logits[test_mask], y[test_mask], weight=weight)
        losses_train.append(loss.item())
        losses_test.append(loss_test.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        pred = list(map(lambda x: np.argmax(x, axis = 0), torch.exp(F.log_softmax(logits, 1)).cpu().detach().numpy()))
        auc = roc_auc_score(correct_pred, [pred[i] for i in test_mask], average='weighted')
        best_auc = best_auc if best_auc > auc else auc

        #pbar.set_description('Best Test AUC: %.4f | Train Loss: %.4f | Test Loss: %.4f' % (best_auc, loss.item(), loss_test.item()))

    return losses_train, losses_test, model, best_auc

import random
random.seed(30)
idx = list(range(len(y)))
random.shuffle(idx)
train_mask = idx[:10000]
test_mask = idx[10000:]

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

model.apply(weight_reset)

losses_train, losses_test, model, best_auc = train_model(model, X, y, 250, 0.0005, train_mask, test_mask)
print(best_auc)
