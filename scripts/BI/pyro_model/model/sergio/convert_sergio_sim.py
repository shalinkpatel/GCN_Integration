import torch
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats

def convert_edge_list(fname : str):
	with open(fname, 'r') as f:
		lines = f.readlines()
	
	lines = list(map(lambda l: l.replace('\n', '').split(','), lines))
	s = list(map(lambda l: int(l[0]), lines))
	d = list(map(lambda l: int(l[1]), lines))
	edge_index = torch.tensor([s, d])
	torch.save(edge_index, fname.replace('generated_data', 'final_data').replace('csv', 'pt'))


def convert_expression_matrix(fname : str):
	x = torch.tensor(pd.read_csv(fname, header=0, index_col=0).to_numpy().T)
	torch.save(x, fname.replace('generated_data', 'final_data').replace('csv', 'pt'))


def create_comp_graph(fname : str, gt_fname : str):
	X = pd.read_csv(fname, header=0, index_col=0).to_numpy().T
	gt = torch.load(gt_fname)
	
	interactions = {}
	
	for i in range(X.shape[0]):
		for j in range(X.shape[0]):
			interactions[(i, j)] = abs(sp.stats.pearsonr(X[i, :], X[j, :])[0])

	gt_edges = set()
	ne = gt.shape[1]
	for i in range(ne):
		e = (gt[0, i].item(), gt[1, i].item())
		gt_edges.add(e)

	s_l = []
	d_l  = []
	for (s, d), i in interactions.items():
		if i >= 0.25 or (s, d) in gt_edges:
			s_l.append(s)
			d_l.append(d)

	print(f"Ne: {len(s_l)} | Ne_GT: {ne}")
	
	edge_index = torch.tensor([s_l, d_l])
	torch.save(edge_index, fname.replace('generated_data', 'final_data').replace('.csv', '-compgraph.pt'))


gt_fname = 'generated_data/100gene-2groups-gt-grn.csv'
convert_edge_list(gt_fname)

gt_conv_fname = 'final_data/100gene-2groups-gt-grn.pt'
expr_fname = 'generated_data/100gene-2groups-{}sparsity.csv'
for sparsity in [1, 5, 10, 15, 20]:
	print('=' * 70)
	print(f"Converting for Sparsity {sparsity}")
	full_fname = expr_fname.format(sparsity)
	convert_expression_matrix(full_fname)
	create_comp_graph(full_fname, gt_conv_fname)
